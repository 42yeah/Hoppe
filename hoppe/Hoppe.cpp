//
//  Hoppe.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include "Hoppe.hpp"
#include <fstream>
#include <thread>
#include <mutex>
#include "UGraph.hpp"


auto Hoppe::run() -> bool {
    if (pointcloud.points.size() == 0) {
        HOPPE_LOG("ERR! Can't run without point cloud");
        return false;
    }

    estimate_planes();
    
    fix_orientations();

    return true;
}

auto Hoppe::load_pointcloud(std::string path) -> void {
    HOPPE_LOG("Loading point cloud...");
    std::ifstream reader(path);
    
    if (!reader.good()) {
        HOPPE_LOG("WARNING! Bad reader: %s", path.c_str());
        return;
    }
    pointcloud.points.clear();
    while (!reader.eof()) {
        cv::Point3f p;
        reader >> p.x >> p.y >> p.z;
        pointcloud.points.push_back(p);
    }
    HOPPE_LOG("Point cloud loading done. Size: %lu", pointcloud.points.size());
}

auto Hoppe::estimate_planes() -> bool {
    HOPPE_LOG("Esimating tangent planes...");
    tangent_planes.planes.clear();
    
    if (parameters.k <= 1) {
        return false;
    }
    
    // Build nearest neighbor tree
    PointCloudIndex index(3, pointcloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    const auto num_neighbors = parameters.k + 1; // Because it contains query point itself
    
    for (auto i = 0; i < pointcloud.points.size(); i++) {
        std::vector<std::size_t> indices(num_neighbors);
        std::vector<float> out_squared_dist(num_neighbors);

        const auto &p = pointcloud.points[i];
        const auto nbhd_count = index.knnSearch(&p.x,
                                                num_neighbors,
                                                &indices[0],
                                                &out_squared_dist[0]);
        if (nbhd_count != num_neighbors) {
            HOPPE_LOG("WARNING! Failed to find enough neighbors here: %lu != %d", nbhd_count, num_neighbors);
        }

        Plane plane;

        // Calculate centroid
        cv::Point3f centroid(0.0f, 0.0f, 0.0f);
        for (auto j = 0; j < nbhd_count; j++) {
            const auto current_index = indices[j];
            if (current_index == i) {
                // That would be myself
                continue;
            }
            const auto neighbor = pointcloud.points[current_index];
            centroid += neighbor;
        }
        centroid /= (float) (nbhd_count - 1);
        plane.origin = centroid;

        // Calculate covariance matrix & normal
        cv::Matx33f covariance_mat;
        for (auto j = 0; j < nbhd_count; j++) {
            const auto current_index = indices[j];
            if (current_index == i) {
                continue;
            }
            const auto neighbor = pointcloud.points[current_index];
            const auto oy = neighbor - centroid;
            const cv::Matx31f oy_mat = { oy.x, oy.y, oy.z };
            cv::Matx33f outer_product;
            cv::mulTransposed(oy_mat, outer_product, false);
            covariance_mat += outer_product;
        }

        cv::Matx31f eigenvalues;
        cv::Matx33f eigenvectors;
        cv::eigen(covariance_mat, eigenvalues, eigenvectors);
        auto min_idx = -1;
        auto min_val = 0.0f;
        for (auto i = 0; i < 3; i++) {
            if (min_idx == -1 || eigenvalues(i, 0) < min_val) {
                min_idx = i;
                min_val = eigenvalues(i, 0);
            }
        }
        plane.normal = cv::normalize(cv::Vec3f(eigenvectors(min_idx, 0),
                                               eigenvectors(min_idx, 1),
                                               eigenvectors(min_idx, 2)));
        tangent_planes.planes.push_back(plane);
    }

    HOPPE_LOG("Tangent plane generation complete. Size: %lu", tangent_planes.planes.size());
    return true;
}

auto Hoppe::fix_orientations() -> void { 
    HOPPE_LOG("Fixing orientations...");
    
    // Construct a graph using each tangent plane's k-neighborhood,
    // And generate its minimal spanning tree.
    // After that, use depth-first search to fix plane orientation.
    UGraph graph(tangent_planes.planes.size());
    
    const auto num_neighbors = parameters.k + 1;
    PlaneCloudIndex index(3, tangent_planes, nanoflann::KDTreeSingleIndexAdaptorParams(5));
    index.buildIndex();
    
    // Use thread to parallelize operations
    const auto num_threads = std::min((int) std::thread::hardware_concurrency(),
                                      (int) tangent_planes.planes.size());
    const auto planes_per_thread = (int) ceilf(tangent_planes.planes.size() / num_threads);
    std::vector<std::thread> threads;
    std::mutex write_mutex;
    HOPPE_LOG("Parallelize planes per thread: %d-%d", planes_per_thread, num_threads);
    
    for (auto thread_id = 0; thread_id < num_threads; thread_id++) {
        
        const auto begin_tp_index = thread_id * planes_per_thread;
        const auto end_tp_index = thread_id == num_threads - 1 ?
                                    tangent_planes.planes.size() :
                                    (thread_id + 1) * planes_per_thread;
        
        threads.push_back(std::thread([&, begin_tp_index, end_tp_index, thread_id] () {
            
            write_mutex.lock();
            HOPPE_LOG("Processing from %d to %lu", begin_tp_index, end_tp_index);
            write_mutex.unlock();
            for (auto i = begin_tp_index; i < end_tp_index; i++) {
                const auto &p1 = tangent_planes.planes[i];
                std::vector<std::size_t> indices(num_neighbors);
                std::vector<float> out_squared_dist(num_neighbors);
                const auto nbhd_count = index.knnSearch(&p1.origin.x, num_neighbors, &indices[0], &out_squared_dist[0]);

                if (nbhd_count != num_neighbors) {
                    write_mutex.lock();
                    HOPPE_LOG("WARNING! Failed to find enough neighbors for plane %f %f %f",
                              p1.origin.x,
                              p1.origin.y,
                              p1.origin.z);
                    write_mutex.unlock();
                }

                // For each of its neighbors...
                for (auto j = 0; j < nbhd_count; j++) {
                    const auto p2_plane_index = indices[j];
                    if (i == p2_plane_index) {
                        continue;
                    }
                    const auto p2 = tangent_planes.planes[p2_plane_index];
                    auto cost = 1.0f - fabs(p1.normal.dot(p2.normal));
                    write_mutex.lock();
                    graph.add_edge({ IndexToTangentPlane(i),
                                     p2_plane_index,
                                     cost });
                    write_mutex.unlock();
                }
            }
            write_mutex.lock();
            HOPPE_LOG("Thread %d completed.", thread_id);
            write_mutex.unlock();

        }));
    }

    for (auto &thread : threads) {
        thread.join();
    }

    graph.clean_duplicate_edges();
    
    HOPPE_LOG("Graph generation done. #nodes: %lu, #edges: %lu",
              graph.num_nodes,
              graph.edges.size());
    
    const auto mst = graph.generate_mst();

    HOPPE_LOG("Minimal spanning tree generation done. #nodes: %lu, #edges: %lu",
              mst.num_nodes,
              mst.edges.size());
    
    auto highest = std::max_element(tangent_planes.planes.begin(),
                     tangent_planes.planes.end(),
                     [] (const auto &p1, const auto &p2) {
        return p1.origin.y > p2.origin.y;
    });
    const cv::Vec3f up(0.0f, 1.0f, 0.0f);
    if (highest->normal.dot(up) < 0.0f) {
        highest->normal = -highest->normal;
    }
    auto corrected = 0;
    auto previous = *highest;
    mst.traverse_dfs((int) (highest - tangent_planes.planes.begin()), [&] (const auto idx) {
        if (tangent_planes.planes[idx].normal.dot(previous.normal) < 0.0f) {
            tangent_planes.planes[idx].normal = -tangent_planes.planes[idx].normal;
            corrected++;
        }
        previous = tangent_planes.planes[idx];
    });
    
    HOPPE_LOG("Normal correction done. Corrected: #%d", corrected);
}

