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
#include <iostream>
#include "UGraph.hpp"


auto Hoppe::run() -> bool {
    if (pointcloud.points.size() == 0) {
        HOPPE_LOG("ERR! Can't run without point cloud");
        return false;
    }

    estimate_planes();

    fix_orientations();

    export_to_ply("planecloud.ply");
    
    cube_march();

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

auto Hoppe::sdf(cv::Point3f point) -> std::optional<float> {
    auto closest_index = -1;
    auto closest_dist = 0.0f;

    for (auto i = 0; i < tangent_planes.planes.size(); i++) {
        const auto &plane = tangent_planes.planes[i];
        const auto dist_to_sample_point = cv::norm(point - plane.origin);
        if (closest_index == -1 || dist_to_sample_point < closest_dist) {
            closest_dist = dist_to_sample_point;
            closest_index = i;
        }
    }
    if (closest_index == -1) {
        HOPPE_LOG("Could not calculate SDF as there is no plane.");
        return {};
    }
    const auto &plane = tangent_planes.planes[closest_index];
    const auto normal_p = VEC2POINT(plane.normal);

    // Calculate projected length on normal.
    const auto projected_length = (point - plane.origin).dot(normal_p);
    const auto z = plane.origin - projected_length * normal_p;
    if (cv::norm(z - plane.origin) >= parameters.density + parameters.noise) {
        return {};
    }
    return projected_length;
}

auto Hoppe::calculate_bounds(cv::Vec3f &bounding_box_min, cv::Vec3f &bounding_box_max) -> void { 
    if (tangent_planes.planes.size() <= 0) {
        HOPPE_LOG("WARNING! There are no planes, and therefore there are no bounds.");
        return;
    }
    cv::Point3f bb_min, bb_max;
    bounding_box_min = POINT2VEC(tangent_planes.planes[0].origin);
    bounding_box_max = POINT2VEC(tangent_planes.planes[0].origin);
    for (auto i = 1; i < tangent_planes.planes.size(); i++) {
        const auto &origin = tangent_planes.planes[i].origin;
        if (origin.x < bounding_box_min(0)) {
            bounding_box_min(0) = origin.x;
        } else if (origin.x > bounding_box_max(0)) {
            bounding_box_max(0) = origin.x;
        }
        if (origin.y < bounding_box_min(1)) {
            bounding_box_min(1) = origin.y;
        } else if (origin.y > bounding_box_max(1)) {
            bounding_box_max(1) = origin.y;
        }
        if (origin.z < bounding_box_min(2)) {
            bounding_box_min(2) = origin.z;
        } else if (origin.z > bounding_box_max(2)) {
            bounding_box_max(2) = origin.z;
        }
    }
}

auto Hoppe::density_estimation(cv::Vec3f bounding_box_size) -> float {
    auto density = (8.0f * bounding_box_size(0) * bounding_box_size(1) * bounding_box_size(2)) / tangent_planes.planes.size();
    HOPPE_LOG("Guestimated density: %f", density);
    return density;
}

auto Hoppe::export_to_ply(const std::string path) -> void {
    HOPPE_LOG("Saving point cloud to %s", path.c_str());
    
    std::ofstream ofs(path);
    ofs << "ply" << std::endl
        << "format ascii 1.0" << std::endl
        << "element vertex " << tangent_planes.planes.size() << std::endl
        << "property float x" << std::endl
        << "property float y" << std::endl
        << "property float z" << std::endl
        << "property uchar red" << std::endl
        << "property uchar green" << std::endl
        << "property uchar blue" << std::endl
        << "end_header" << std::endl;
        
    for (const auto &p : tangent_planes.planes) {
        const auto position = p.origin;
        ofs << position.x << " " << position.y << " " << position.z << " "
            << (int) 255 << " "
            << (int) 125 << " "
            << (int) 0 << " " << std::endl;
    }
    
    ofs.close();
}

auto Hoppe::cube_march() -> void {
    cv::Vec3f bounding_box_min, bounding_box_max;
    calculate_bounds(bounding_box_min, bounding_box_max);
    auto size = bounding_box_max - bounding_box_min;
    HOPPE_LOG("Bounding box size: %f %f %f", size(0), size(1), size(2));

    parameters.density = density_estimation(size);

    // OVERRIDE
//    bounding_box_min = cv::Vec3f(-1.0f, -1.0f, -1.0f);
//    bounding_box_max = cv::Vec3f(1.0f, 1.0f, 1.0f);
//    size = bounding_box_max - bounding_box_min;
//    parameters.density = 0.01f;

    cv::Vec3i marching_size(ceilf(size(0) / parameters.density),
                            ceilf(size(1) / parameters.density),
                            ceilf(size(2) / parameters.density));
    auto volume = 0;
    do {
        volume = marching_size(0) * marching_size(1) * marching_size(2);
        if (volume > parameters.max_volume) {
            parameters.density *= 2.0f;
            marching_size = cv::Vec3i(ceilf(size(0) / parameters.density),
                                      ceilf(size(1) / parameters.density),
                                      ceilf(size(2) / parameters.density));
        }
    } while (volume > parameters.max_volume);
    
    HOPPE_LOG("Marching cube size: %d %d %d", marching_size(0),
              marching_size(1), marching_size(2));
    marcher.init(marching_size, parameters.density);
    HOPPE_LOG("Estimated: from %f %f %f to %f %f %f",
              bounding_box_min(0), bounding_box_min(1), bounding_box_min(2),
              bounding_box_max(0), bounding_box_max(1), bounding_box_max(2));

    marcher.march([&] (cv::Point3f p) {
        return sdf(p);
    }, VEC2POINT(bounding_box_min));
}

auto Hoppe::export_mesh(const std::string path) -> void {
    HOPPE_LOG("Exporting mesh to %s as .obj format...", path.c_str());
    std::ofstream obj_file(path);

    for (auto i = 0; i < marcher.faces.size(); i++) {
        const auto &pts = marcher.faces[i].points;
        obj_file << "v " << pts[0].x << " " << pts[0].y << " " << pts[0].z << std::endl;
        obj_file << "v " << pts[1].x << " " << pts[1].y << " " << pts[1].z << std::endl;
        obj_file << "v " << pts[2].x << " " << pts[2].y << " " << pts[2].z << std::endl;
    }
    for (auto i = 0; i < marcher.faces.size(); i++) {
        const auto base_idx = i * 3 + 1;
        obj_file << "f " << base_idx << " " << (base_idx + 1) << " " << (base_idx + 2) << std::endl;
    }
    obj_file.close();
}
