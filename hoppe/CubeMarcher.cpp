//
//  CubeMarcher.cpp
//  hoppe
//
//  Created by apple on 03/04/2021.
//

#include "CubeMarcher.hpp"
#include <thread>
#include <vector>
#include <fstream>
#include <mutex>
#include <iostream>

#define FACE(a, b, c) \
faces.push_back(Triangle { edge_offset[a], \
                           edge_offset[b], \
                           edge_offset[c] } + pos)


auto CubeMarcher::init(cv::Vec3i size, float resolution) -> void {
    cell_mat.resize(size(2));
    for (auto i = 0; i < cell_mat.size(); i++) {
        cell_mat[i].resize(size(1));
        for (auto j = 0; j < cell_mat[i].size(); j++) {
            cell_mat[i][j].resize(size(0));
        }
    }
    this->size = size;
    this->resolution = resolution;
}

auto CubeMarcher::march(std::function<std::optional<float> (cv::Point3f)> sdf,
                        cv::Point3f offset) -> void {
    const auto volume = size(0) * size(1) * size(2);
    const auto num_threads = std::min(std::thread::hardware_concurrency(),
                                      (unsigned int) volume);
    const auto marches_per_thread = volume / num_threads;
    std::vector<std::thread> threads;
    HOPPE_LOG("Marching %d times with %d marches per thread", volume, marches_per_thread);
    
    // Define the offset of 8 corners of cube
    cv::Point3f sample_offset[8] = {
        { 0.0f, 0.0f, 0.0f },
        { resolution, 0.0f, 0.0f },
        { resolution, resolution, 0.0f },
        { 0.0f, resolution, 0.0f },
        { 0.0f, 0.0f, resolution },
        { resolution, 0.0f, resolution },
        { resolution, resolution, resolution },
        { 0.0f, resolution, resolution }
    };
    cv::Point3i lut_offset[8] = {
        { 0, 0, 0 },
        { 1, 0, 0 },
        { 1, 1, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 },
        { 1, 0, 1 },
        { 1, 1, 1 },
        { 0, 1, 1 },
    };
    const auto half_resolution = resolution / 2.0f;
    cv::Point3f edge_offset[12] = {
        { half_resolution, 0.0f, 0.0f },
        { resolution, half_resolution, 0.0f },
        { half_resolution, resolution, 0.0f },
        { 0.0f, half_resolution, 0.0f },
        { half_resolution, 0.0f, resolution },
        { resolution, half_resolution, resolution },
        { half_resolution, resolution, resolution },
        { 0.0f, half_resolution, resolution },
        { 0.0f, 0.0f, half_resolution },
        { resolution, 0.0f, half_resolution },
        { resolution, resolution, half_resolution },
        { 0.0f, resolution, half_resolution }
    };
    std::vector<std::vector<std::vector<std::optional<float> > > > lut;
    lut.resize(size(2));
    for (auto z = 0; z < lut.size(); z++) {
        lut[z].resize(size(1));
        for (auto y = 0; y < lut[z].size(); y++) {
            lut[z][y].resize(size(0));
        }
    }
    
    std::mutex lut_mutex;
    std::mutex face_mutex;
    auto num_faces = 0;
    faces.clear();
    
    auto maximum = offset + cv::Point3f(size(0) * resolution,
                                        size(1) * resolution,
                                        size(2) * resolution);
    HOPPE_LOG("Actual maximum: %f %f %f", maximum.x, maximum.y, maximum.z);

    for (auto thread_id = 0; thread_id < num_threads; thread_id++) {
        const auto march = [&, thread_id] () {
            for (auto j = 0; j < marches_per_thread; j++) {
                const auto index = (thread_id * marches_per_thread) + j;
                const auto x = index % size(0);
                const auto y = (index / size(0)) % size(1);
                const auto z = (index / size(0) / size(1));
                if (x == size(0) - 1 || y == size(1) - 1 || z == size(2) - 1) {
                    continue;
                }
                const cv::Point3f pos(offset + cv::Point3f(x * resolution, y * resolution, z * resolution));
                auto &cell = cell_mat[z][y][x];
                cell.state = 0;
                for (auto i = 0; i < 8; i++) {
                    auto dist = 1.0f;
                    auto x_neighbor = x + lut_offset[i].x;
                    auto y_neighbor = y + lut_offset[i].y;
                    auto z_neighbor = z + lut_offset[i].z;
                    if (!lut[z_neighbor][y_neighbor][x_neighbor].has_value()) {
                        auto dist_sdf = sdf(pos + sample_offset[i]);
                        if (dist_sdf.has_value()) {
                            dist = dist_sdf.value();
                        }
                        lut_mutex.lock();
                        lut[z_neighbor][y_neighbor][x_neighbor] = dist;
                        lut_mutex.unlock();
                    } else {
                        dist = lut[z_neighbor][y_neighbor][x_neighbor].value();
                    }
                    
                    cell.values[i] = dist;
                    if (cell.values[i] < 0) {
                        cell.state += pow(2, i);
                    }
                }
                if (cell.state == 0 || cell.state == 255) {
                    continue;
                }
                num_faces++;
                // TODO: reconstruct face...
                face_mutex.lock();
                switch (cell.state) {
                    case 0:
                    case 255:
                        break;
                        
                    case 1:
                        FACE(3, 0, 8);
                        break;
                        
                    case 2:
                        FACE(1, 9, 0);
                        break;
                        
                    case 3:
                        FACE(3, 9, 8);
                        FACE(3, 1, 9);
                        break;
                        
                    case 4:
                        FACE(10, 1, 2);
                        break;
                        
                    case 5:
                        FACE(10, 1, 2);
                        FACE(0, 8, 3);
                        break;
                        
                    case 6:
                        FACE(10, 0, 2);
                        FACE(10, 9, 0);
                        break;
                        
                    case 7:
                        FACE(2, 8, 3);
                        FACE(2, 10, 8);
                        FACE(10, 9, 8);
                        break;
                        
                    case 8:
                        FACE(11, 2, 3);
                        break;
                        
                    case 9:
                        FACE(2, 8, 11);
                        FACE(2, 0, 8);
                        break;
                        
                }
                face_mutex.unlock();
            }
        };
        threads.push_back(std::thread(march));
    }
    for (auto &t : threads) {
        t.join();
    }
    HOPPE_LOG("Marching cubes done. Potential faces: %d", num_faces);
    dump("sterfile");
    write_tmp_obj("fake.obj");
}

auto CubeMarcher::dump(std::string to) -> void { 
    HOPPE_LOG("Dumping file to %s...", to.c_str());
    
    std::ofstream ofs(to);
    for (auto z = 0; z < cell_mat.size(); z++) {
        for (auto y = 0; y < cell_mat.size(); y++) {
            for (auto x = 0; x < cell_mat.size(); x++) {
                const auto &cell = cell_mat[z][y][x];
                if (cell.state == 0 || cell.state == 255) {
                    continue;
                }
                ofs << x << ", " << y << ", " << z << " - " << cell.state << " [";
                for (auto i = 0; i < 8; i++) {
                    ofs << cell.values[i];
                    if (i != 7) {
                        ofs << ", ";
                    }
                }
                ofs << "]" << std::endl;
            }
        }
    }
    ofs.close();
}

auto CubeMarcher::write_tmp_obj(std::string path) -> void {
    std::ofstream obj_file(path);

    for (auto i = 0; i < faces.size(); i++) {
        const auto &pts = faces[i].points;
        obj_file << "v " << pts[0].x << " " << pts[0].y << " " << pts[0].z << std::endl;
        obj_file << "v " << pts[1].x << " " << pts[1].y << " " << pts[1].z << std::endl;
        obj_file << "v " << pts[2].x << " " << pts[2].y << " " << pts[2].z << std::endl;
    }
    for (auto i = 0; i < faces.size(); i++) {
        const auto base_idx = i * 3 + 1;
        obj_file << "f " << base_idx << " " << (base_idx + 1) << " " << (base_idx + 2) << std::endl;
    }
    obj_file.close();
    
}

