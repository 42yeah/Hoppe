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

                    case 10:
                        FACE(11, 2, 3);
                        FACE(1, 9, 0);
                        break;

                    case 11:
                        FACE(1, 11, 2);
                        FACE(1, 9, 11);
                        FACE(9, 8, 11);
                        break;

                    case 12:
                        FACE(11, 1, 3);
                        FACE(11, 10, 1);
                        break;

                    case 13:
                        FACE(0, 10, 1);
                        FACE(0, 8, 10);
                        FACE(8, 11, 10);
                        break;

                    case 14:
                        FACE(3, 9, 0);
                        FACE(3, 11, 9);
                        FACE(11, 10, 9);
                        break;

                    case 15:
                        FACE(9, 8, 10);
                        FACE(10, 8, 11);
                        break;

                    case 16:
                        FACE(8, 4, 7);
                        break;

                    case 17:
                        FACE(0, 7, 3);
                        FACE(0, 4, 7);
                        break;

                    case 18:
                        FACE(1, 9, 0);
                        FACE(4, 7, 8);
                        break;

                    case 19:
                        FACE(4, 1, 9);
                        FACE(4, 7, 1);
                        FACE(7, 3, 1);
                        break;

                    case 20:
                        FACE(8, 4, 7);
                        FACE(10, 1, 2);
                        break;

                    case 21:
                        FACE(7, 0, 4);
                        FACE(7, 3, 0);
                        FACE(2, 10, 1);
                        break;

                    case 22:
                        FACE(10, 0, 2);
                        FACE(10, 9, 0);
                        FACE(4, 7, 8);
                        break;

                    case 23:
                        FACE(9, 4, 10);
                        FACE(10, 4, 7);
                        FACE(10, 7, 2);
                        FACE(2, 7, 3);
                        break;

                    case 24:
                        FACE(4, 7, 8);
                        FACE(11, 2, 3);
                        break;

                    case 25:
                        FACE(11, 4, 7);
                        FACE(11, 2, 4);
                        FACE(2, 0, 4);
                        break;

                    case 26:
                        FACE(9, 0, 1);
                        FACE(3, 11, 2);
                        FACE(6, 8, 4);
                        break;

                    case 27:
                        FACE(2, 1, 11);
                        FACE(11, 1, 9);
                        FACE(11, 9, 7);
                        FACE(7, 9, 4);
                        break;

                    case 28:
                        FACE(1, 11, 10);
                        FACE(1, 3, 11);
                        FACE(8, 4, 7);
                        break;

                    case 29:
                        FACE(10, 7, 11);
                        FACE(10, 4, 7);
                        FACE(10, 1, 4);
                        FACE(1, 0, 4);
                        break;

                    case 30:
                        FACE(3, 9, 0);
                        FACE(3, 11, 9);
                        FACE(11, 10, 9);
                        FACE(0, 7, 8);
                        break;

                    case 31:
                        FACE(11, 10, 9);
                        FACE(11, 9, 4);
                        FACE(7, 11, 4);
                        break;

                    case 32:
                        FACE(9, 5, 4);
                        break;

                    case 33:
                        FACE(3, 0, 8);
                        FACE(9, 5, 4);
                        break;

                    case 34:
                        FACE(1, 4, 0);
                        FACE(1, 5, 4);
                        break;

                    case 35:
                        FACE(8, 5, 4);
                        FACE(8, 3, 5);
                        FACE(3, 1, 5);
                        break;

                    case 36:
                        FACE(4, 9, 5);
                        FACE(1, 2, 10);
                        break;

                    case 37:
                        FACE(3, 0, 8);
                        FACE(9, 5, 4);
                        FACE(6, 1, 2);
                        break;

                    case 38:
                        FACE(5, 2, 10);
                        FACE(5, 4, 2);
                        FACE(4, 0, 2);
                        break;

                    case 39:
                        FACE(4, 8, 5);
                        FACE(5, 8, 3);
                        FACE(5, 3, 10);
                        FACE(10, 3, 2);
                        break;

                    case 40:
                        FACE(11, 2, 3);
                        FACE(9, 5, 4);
                        break;

                    case 41:
                        FACE(2, 8, 11);
                        FACE(2, 0, 8);
                        FACE(9, 5, 4);
                        break;

                    case 42:
                        FACE(4, 1, 5);
                        FACE(4, 0, 1);
                        FACE(3, 11, 2);
                        break;

                    case 43:
                        FACE(5, 2, 1);
                        FACE(5, 11, 2);
                        FACE(5, 4, 11);
                        FACE(4, 8, 11);
                        break;

                    case 44:
                        FACE(11, 1, 3);
                        FACE(11, 10, 1);
                        FACE(5, 4, 9);
                        break;

                    case 45:
                        FACE(0, 10, 1);
                        FACE(0, 8, 10);
                        FACE(8, 11, 10);
                        FACE(1, 4, 9);
                        break;

                    case 46:
                        FACE(0, 3, 4);
                        FACE(4, 3, 11);
                        FACE(4, 11, 5);
                        FACE(5, 11, 10);
                        break;

                    case 47:
                        FACE(8, 11, 10);
                        FACE(8, 10, 5);
                        FACE(4, 8, 5);
                        break;

                    case 48:
                        FACE(8, 5, 7);
                        FACE(8, 9, 5);
                        break;

                    case 49:
                        FACE(9, 3, 0);
                        FACE(9, 5, 3);
                        FACE(5, 7, 3);
                        break;

                    case 50:
                        FACE(0, 7, 8);
                        FACE(0, 1, 7);
                        FACE(1, 5, 7);
                        break;

                    case 51:
                        FACE(5, 7, 1);
                        FACE(1, 7, 3);
                        break;

                    case 52:
                        FACE(8, 5, 7);
                        FACE(8, 9, 5);
                        FACE(1, 2, 10);
                        break;

                    case 53:
                        FACE(9, 3, 0);
                        FACE(9, 5, 3);
                        FACE(5, 7, 3);
                        FACE(0, 10, 1);
                        break;

                    case 54:
                        FACE(7, 10, 5);
                        FACE(7, 2, 10);
                        FACE(7, 8, 2);
                        FACE(8, 0, 2);
                        break;

                    case 55:
                        FACE(5, 7, 3);
                        FACE(5, 3, 2);
                        FACE(10, 5, 2);
                        break;

                    case 56:
                        FACE(5, 8, 9);
                        FACE(5, 7, 8);
                        FACE(11, 2, 3);
                        break;

                    case 57:
                        FACE(0, 9, 2);
                        FACE(2, 9, 5);
                        FACE(2, 5, 11);
                        FACE(11, 5, 7);
                        break;

                    case 58:
                        FACE(0, 7, 8);
                        FACE(0, 1, 7);
                        FACE(1, 5, 7);
                        FACE(8, 2, 3);
                        break;

                    case 59:
                        FACE(1, 5, 7);
                        FACE(1, 7, 11);
                        FACE(2, 1, 11);
                        break;

                    case 60:
                        FACE(8, 5, 7);
                        FACE(8, 9, 5);
                        FACE(3, 11, 10);
                        FACE(3, 10, 1);
                        break;

                    case 61:
                        FACE(9, 8, 0);
                        FACE(0, 8, 7);
                        FACE(0, 7, 11);
                        FACE(0, 11, 10);
                        FACE(0, 10, 1);
                        break;

                    case 62:
                        FACE(3, 1, 0);
                        FACE(0, 1, 10);
                        FACE(0, 10, 5);
                        FACE(0, 5, 7);
                        FACE(0, 7, 8);
                        break;

                    case 63:
                        FACE(10, 5, 7);
                        FACE(11, 10, 7);
                        break;

                    case 64:
                        FACE(5, 10, 6);
                        break;

                    case 65:
                        FACE(3, 0, 8);
                        FACE(5, 10, 6);
                        break;

                    case 66:
                        FACE(0, 1, 9);
                        FACE(10, 6, 5);
                        break;

                    case 67:
                        FACE(3, 9, 8);
                        FACE(3, 1, 9);
                        FACE(10, 6, 5);
                        break;

                    case 68:
                        FACE(5, 2, 6);
                        FACE(5, 1, 2);
                        break;

                    case 69:
                        FACE(5, 2, 6);
                        FACE(5, 1, 2);
                        FACE(0, 8, 3);
                        break;

                    case 70:
                        FACE(9, 6, 5);
                        FACE(9, 0, 6);
                        FACE(0, 2, 6);
                        break;

                    case 71:
                        FACE(8, 5, 9);
                        FACE(8, 6, 5);
                        FACE(8, 3, 6);
                        FACE(3, 2, 6);
                        break;

                    case 72:
                        FACE(5, 10, 6);
                        FACE(2, 3, 11);
                        break;

                    case 73:
                        FACE(8, 2, 0);
                        FACE(8, 11, 2);
                        FACE(6, 5, 10);
                        break;

                    case 74:
                        FACE(11, 2, 3);
                        FACE(1, 9, 0);
                        FACE(4, 10, 6);
                        break;

                    case 75:
                        FACE(1, 11, 2);
                        FACE(1, 9, 11);
                        FACE(9, 8, 11);
                        FACE(2, 5, 10);
                        break;

                    case 76:
                        FACE(6, 3, 11);
                        FACE(6, 5, 3);
                        FACE(5, 1, 3);
                        break;

                    case 77:
                        FACE(1, 0, 5);
                        FACE(5, 0, 8);
                        FACE(5, 8, 6);
                        FACE(6, 8, 11);
                        break;

                    case 78:
                        FACE(0, 3, 9);
                        FACE(9, 3, 11);
                        FACE(9, 11, 5);
                        FACE(5, 11, 6);
                        break;

                    case 79:
                        FACE(9, 8, 11);
                        FACE(9, 11, 6);
                        FACE(5, 9, 6);
                        break;

                    case 80:
                        FACE(8, 4, 7);
                        FACE(5, 10, 6);
                        break;

                    case 81:
                        FACE(0, 7, 3);
                        FACE(0, 4, 7);
                        FACE(5, 10, 6);
                        break;

                    case 82:
                        FACE(8, 4, 7);
                        FACE(5, 10, 6);
                        FACE(2, 9, 0);
                        break;

                    case 83:
                        FACE(4, 1, 9);
                        FACE(4, 7, 1);
                        FACE(7, 3, 1);
                        FACE(9, 6, 5);
                        break;

                    case 84:
                        FACE(2, 5, 1);
                        FACE(2, 6, 5);
                        FACE(7, 8, 4);
                        break;

                    case 85:
                        FACE(0, 7, 3);
                        FACE(0, 4, 7);
                        FACE(1, 2, 6);
                        FACE(1, 6, 5);
                        break;

                    case 86:
                        FACE(9, 6, 5);
                        FACE(9, 0, 6);
                        FACE(0, 2, 6);
                        FACE(5, 8, 4);
                        break;

                    case 87:
                        FACE(4, 0, 9);
                        FACE(9, 0, 3);
                        FACE(9, 3, 2);
                        FACE(9, 2, 6);
                        FACE(9, 6, 5);
                        break;

                    case 88:
                        FACE(10, 6, 5);
                        FACE(7, 8, 4);
                        FACE(0, 11, 2);
                        break;

                    case 89:
                        FACE(11, 4, 7);
                        FACE(11, 2, 4);
                        FACE(2, 0, 4);
                        FACE(7, 10, 6);
                        break;

                    case 90:
                        FACE(8, 4, 7);
                        FACE(5, 10, 6);
                        FACE(1, 9, 0);
                        FACE(2, 3, 11);
                        break;

                    case 91:
                        FACE(10, 6, 5);
                        FACE(2, 1, 11);
                        FACE(11, 1, 9);
                        FACE(11, 9, 7);
                        FACE(7, 9, 4);
                        break;

                    case 92:
                        FACE(6, 3, 11);
                        FACE(6, 5, 3);
                        FACE(5, 1, 3);
                        FACE(11, 4, 7);
                        break;

                    case 93:
                        FACE(6, 2, 11);
                        FACE(11, 2, 1);
                        FACE(11, 1, 0);
                        FACE(11, 0, 4);
                        FACE(11, 4, 7);
                        break;

                    case 94:
                        FACE(8, 4, 7);
                        FACE(0, 3, 9);
                        FACE(9, 3, 11);
                        FACE(9, 11, 5);
                        FACE(5, 11, 6);
                        break;

                    case 95:
                        FACE(6, 2, 11);
                        FACE(2, 9, 11);
                        FACE(9, 4, 11);
                        FACE(11, 4, 7);
                        break;

                    case 96:
                        FACE(9, 6, 4);
                        FACE(9, 10, 6);
                        break;

                    case 97:
                        FACE(6, 9, 10);
                        FACE(6, 4, 9);
                        FACE(8, 3, 0);
                        break;

                    case 98:
                        FACE(10, 0, 1);
                        FACE(10, 6, 0);
                        FACE(6, 4, 0);
                        break;

                    case 99:
                        FACE(4, 8, 6);
                        FACE(6, 8, 3);
                        FACE(6, 3, 10);
                        FACE(10, 3, 1);
                        break;

                    case 100:
                        FACE(1, 4, 9);
                        FACE(1, 2, 4);
                        FACE(2, 6, 4);
                        break;

                    case 101:
                        FACE(1, 4, 9);
                        FACE(1, 2, 4);
                        FACE(2, 6, 4);
                        FACE(9, 3, 0);
                        break;

                    case 102:
                        FACE(6, 4, 2);
                        FACE(2, 4, 0);
                        break;

                    case 103:
                        FACE(8, 2, 3);
                        FACE(8, 4, 2);
                        FACE(4, 6, 2);
                        break;

                    case 104:
                        FACE(9, 6, 4);
                        FACE(9, 10, 6);
                        FACE(2, 3, 11);
                        break;

                    case 105:
                        FACE(2, 8, 11);
                        FACE(2, 0, 8);
                        FACE(10, 6, 4);
                        FACE(10, 4, 9);
                        break;

                    case 106:
                        FACE(10, 0, 1);
                        FACE(10, 6, 0);
                        FACE(6, 4, 0);
                        FACE(1, 11, 2);
                        break;

                    case 107:
                        FACE(10, 9, 1);
                        FACE(1, 9, 4);
                        FACE(1, 4, 8);
                        FACE(1, 8, 11);
                        FACE(1, 11, 2);
                        break;

                    case 108:
                        FACE(3, 9, 1);
                        FACE(3, 4, 9);
                        FACE(3, 11, 4);
                        FACE(11, 6, 4);
                        break;

                    case 109:
                        FACE(0, 2, 1);
                        FACE(1, 2, 11);
                        FACE(1, 11, 6);
                        FACE(1, 6, 4);
                        FACE(1, 4, 9);
                        break;

                    case 110:
                        FACE(6, 4, 0);
                        FACE(6, 0, 3);
                        FACE(11, 6, 3);
                        break;

                    case 111:
                        FACE(11, 6, 4);
                        FACE(8, 11, 4);
                        break;

                    case 112:
                        FACE(7, 10, 6);
                        FACE(7, 8, 10);
                        FACE(8, 9, 10);
                        break;

                    case 113:
                        FACE(10, 0, 9);
                        FACE(10, 3, 0);
                        FACE(10, 6, 3);
                        FACE(6, 7, 3);
                        break;

                    case 114:
                        FACE(6, 7, 10);
                        FACE(10, 7, 8);
                        FACE(10, 8, 1);
                        FACE(1, 8, 0);
                        break;

                    case 115:
                        FACE(7, 3, 1);
                        FACE(7, 1, 10);
                        FACE(6, 7, 10);
                        break;

                    case 116:
                        FACE(6, 7, 2);
                        FACE(2, 7, 8);
                        FACE(2, 8, 1);
                        FACE(1, 8, 9);
                        break;

                    case 117:
                        FACE(1, 5, 9);
                        FACE(9, 5, 6);
                        FACE(9, 6, 7);
                        FACE(9, 7, 3);
                        FACE(9, 3, 0);
                        break;

                    case 118:
                        FACE(0, 2, 6);
                        FACE(0, 6, 7);
                        FACE(8, 0, 7);
                        break;

                    case 119:
                        FACE(3, 2, 6);
                        FACE(7, 3, 6);
                        break;

                    case 120:
                        FACE(7, 10, 6);
                        FACE(7, 8, 10);
                        FACE(8, 9, 10);
                        FACE(6, 3, 11);
                        break;

                    case 121:
                        FACE(11, 8, 7);
                        FACE(7, 8, 0);
                        FACE(7, 0, 9);
                        FACE(7, 9, 10);
                        FACE(7, 10, 6);
                        break;

                    case 122:
                        FACE(11, 2, 3);
                        FACE(6, 7, 10);
                        FACE(10, 7, 8);
                        FACE(10, 8, 1);
                        FACE(1, 8, 0);
                        break;

                    case 123:
                        FACE(10, 9, 1);
                        FACE(9, 7, 1);
                        FACE(7, 11, 1);
                        FACE(1, 11, 2);
                        break;

                    case 124:
                        FACE(7, 5, 6);
                        FACE(6, 5, 9);
                        FACE(6, 9, 1);
                        FACE(6, 1, 3);
                        FACE(6, 3, 11);
                        break;

                    case 125:
                        FACE(0, 9, 1);
                        FACE(11, 6, 7);
                        break;

                    case 126:
                        FACE(3, 1, 0);
                        FACE(1, 6, 0);
                        FACE(6, 7, 0);
                        FACE(0, 7, 8);
                        break;

                    case 127:
                        FACE(11, 6, 7);
                        break;

                    case 128:
                        FACE(7, 6, 11);
                        break;

                    case 129:
                        FACE(6, 11, 7);
                        FACE(3, 0, 8);
                        break;

                    case 130:
                        FACE(7, 6, 11);
                        FACE(1, 9, 0);
                        break;

                    case 131:
                        FACE(9, 3, 1);
                        FACE(9, 8, 3);
                        FACE(7, 6, 11);
                        break;

                    case 132:
                        FACE(7, 6, 11);
                        FACE(10, 1, 2);
                        break;

                    case 133:
                        FACE(1, 2, 10);
                        FACE(11, 7, 6);
                        FACE(4, 3, 0);
                        break;

                    case 134:
                        FACE(0, 10, 9);
                        FACE(0, 2, 10);
                        FACE(11, 7, 6);
                        break;

                    case 135:
                        FACE(2, 8, 3);
                        FACE(2, 10, 8);
                        FACE(10, 9, 8);
                        FACE(3, 6, 11);
                        break;

                    case 136:
                        FACE(6, 3, 7);
                        FACE(6, 2, 3);
                        break;

                    case 137:
                        FACE(7, 0, 8);
                        FACE(7, 6, 0);
                        FACE(6, 2, 0);
                        break;

                    case 138:
                        FACE(6, 3, 7);
                        FACE(6, 2, 3);
                        FACE(1, 9, 0);
                        break;

                    case 139:
                        FACE(2, 1, 6);
                        FACE(6, 1, 9);
                        FACE(6, 9, 7);
                        FACE(7, 9, 8);
                        break;

                    case 140:
                        FACE(10, 7, 6);
                        FACE(10, 1, 7);
                        FACE(1, 3, 7);
                        break;

                    case 141:
                        FACE(6, 10, 7);
                        FACE(7, 10, 1);
                        FACE(7, 1, 8);
                        FACE(8, 1, 0);
                        break;

                    case 142:
                        FACE(9, 6, 10);
                        FACE(9, 7, 6);
                        FACE(9, 0, 7);
                        FACE(0, 3, 7);
                        break;

                    case 143:
                        FACE(10, 9, 8);
                        FACE(10, 8, 7);
                        FACE(6, 10, 7);
                        break;

                    case 144:
                        FACE(4, 11, 8);
                        FACE(4, 6, 11);
                        break;

                    case 145:
                        FACE(3, 6, 11);
                        FACE(3, 0, 6);
                        FACE(0, 4, 6);
                        break;

                    case 146:
                        FACE(11, 4, 6);
                        FACE(11, 8, 4);
                        FACE(0, 1, 9);
                        break;

                    case 147:
                        FACE(1, 11, 3);
                        FACE(1, 6, 11);
                        FACE(1, 9, 6);
                        FACE(9, 4, 6);
                        break;

                    case 148:
                        FACE(4, 11, 8);
                        FACE(4, 6, 11);
                        FACE(10, 1, 2);
                        break;

                    case 149:
                        FACE(3, 6, 11);
                        FACE(3, 0, 6);
                        FACE(0, 4, 6);
                        FACE(11, 1, 2);
                        break;

                    case 150:
                        FACE(4, 11, 8);
                        FACE(4, 6, 11);
                        FACE(9, 0, 2);
                        FACE(9, 2, 10);
                        break;

                    case 151:
                        FACE(2, 0, 3);
                        FACE(3, 0, 9);
                        FACE(3, 9, 4);
                        FACE(3, 4, 6);
                        FACE(3, 6, 11);
                        break;

                    case 152:
                        FACE(8, 2, 3);
                        FACE(8, 4, 2);
                        FACE(4, 6, 2);
                        break;

                    case 153:
                        FACE(4, 6, 0);
                        FACE(0, 6, 2);
                        break;

                    case 154:
                        FACE(8, 2, 3);
                        FACE(8, 4, 2);
                        FACE(4, 6, 2);
                        FACE(3, 9, 0);
                        break;

                    case 155:
                        FACE(4, 6, 2);
                        FACE(4, 2, 1);
                        FACE(9, 4, 1);
                        break;

                    case 156:
                        FACE(6, 10, 4);
                        FACE(4, 10, 1);
                        FACE(4, 1, 8);
                        FACE(8, 1, 3);
                        break;

                    case 157:
                        FACE(0, 4, 6);
                        FACE(0, 6, 10);
                        FACE(1, 0, 10);
                        break;

                    case 158:
                        FACE(8, 11, 3);
                        FACE(3, 11, 6);
                        FACE(3, 6, 10);
                        FACE(3, 10, 9);
                        FACE(3, 9, 0);
                        break;

                    case 159:
                        FACE(6, 10, 9);
                        FACE(4, 6, 9);
                        break;

                    case 160:
                        FACE(9, 5, 4);
                        FACE(6, 11, 7);
                        break;

                    case 161:
                        FACE(5, 4, 9);
                        FACE(8, 3, 0);
                        FACE(2, 7, 6);
                        break;

                    case 162:
                        FACE(1, 4, 0);
                        FACE(1, 5, 4);
                        FACE(6, 11, 7);
                        break;

                    case 163:
                        FACE(8, 5, 4);
                        FACE(8, 3, 5);
                        FACE(3, 1, 5);
                        FACE(4, 11, 7);
                        break;

                    case 164:
                        FACE(7, 6, 11);
                        FACE(10, 1, 2);
                        FACE(0, 5, 4);
                        break;

                    case 165:
                        FACE(3, 0, 8);
                        FACE(9, 5, 4);
                        FACE(10, 1, 2);
                        FACE(6, 11, 7);
                        break;

                    case 166:
                        FACE(5, 2, 10);
                        FACE(5, 4, 2);
                        FACE(4, 0, 2);
                        FACE(10, 7, 6);
                        break;

                    case 167:
                        FACE(7, 6, 11);
                        FACE(4, 8, 5);
                        FACE(5, 8, 3);
                        FACE(5, 3, 10);
                        FACE(10, 3, 2);
                        break;

                    case 168:
                        FACE(3, 6, 2);
                        FACE(3, 7, 6);
                        FACE(4, 9, 5);
                        break;

                    case 169:
                        FACE(7, 0, 8);
                        FACE(7, 6, 0);
                        FACE(6, 2, 0);
                        FACE(8, 5, 4);
                        break;

                    case 170:
                        FACE(6, 3, 7);
                        FACE(6, 2, 3);
                        FACE(5, 4, 0);
                        FACE(5, 0, 1);
                        break;

                    case 171:
                        FACE(7, 3, 8);
                        FACE(8, 3, 2);
                        FACE(8, 2, 1);
                        FACE(8, 1, 5);
                        FACE(8, 5, 4);
                        break;

                    case 172:
                        FACE(10, 7, 6);
                        FACE(10, 1, 7);
                        FACE(1, 3, 7);
                        FACE(6, 9, 5);
                        break;

                    case 173:
                        FACE(5, 4, 9);
                        FACE(6, 10, 7);
                        FACE(7, 10, 1);
                        FACE(7, 1, 8);
                        FACE(8, 1, 0);
                        break;

                    case 174:
                        FACE(5, 1, 10);
                        FACE(10, 1, 0);
                        FACE(10, 0, 3);
                        FACE(10, 3, 7);
                        FACE(10, 7, 6);
                        break;

                    case 175:
                        FACE(5, 1, 10);
                        FACE(1, 8, 10);
                        FACE(8, 7, 10);
                        FACE(10, 7, 6);
                        break;

                    case 176:
                        FACE(6, 9, 5);
                        FACE(6, 11, 9);
                        FACE(11, 8, 9);
                        break;

                    case 177:
                        FACE(0, 9, 3);
                        FACE(3, 9, 5);
                        FACE(3, 5, 11);
                        FACE(11, 5, 6);
                        break;

                    case 178:
                        FACE(5, 6, 1);
                        FACE(1, 6, 11);
                        FACE(1, 11, 0);
                        FACE(0, 11, 8);
                        break;

                    case 179:
                        FACE(3, 1, 5);
                        FACE(3, 5, 6);
                        FACE(11, 3, 6);
                        break;

                    case 180:
                        FACE(6, 9, 5);
                        FACE(6, 11, 9);
                        FACE(11, 8, 9);
                        FACE(5, 2, 10);
                        break;

                    case 181:
                        FACE(1, 2, 10);
                        FACE(0, 9, 3);
                        FACE(3, 9, 5);
                        FACE(3, 5, 11);
                        FACE(11, 5, 6);
                        break;

                    case 182:
                        FACE(6, 4, 5);
                        FACE(5, 4, 8);
                        FACE(5, 8, 0);
                        FACE(5, 0, 2);
                        FACE(5, 2, 10);
                        break;

                    case 183:
                        FACE(2, 0, 3);
                        FACE(0, 5, 3);
                        FACE(5, 6, 3);
                        FACE(3, 6, 11);
                        break;

                    case 184:
                        FACE(9, 3, 8);
                        FACE(9, 2, 3);
                        FACE(9, 5, 2);
                        FACE(5, 6, 2);
                        break;

                    case 185:
                        FACE(6, 2, 0);
                        FACE(6, 0, 9);
                        FACE(5, 6, 9);
                        break;

                    case 186:
                        FACE(0, 4, 8);
                        FACE(8, 4, 5);
                        FACE(8, 5, 6);
                        FACE(8, 6, 2);
                        FACE(8, 2, 3);
                        break;

                    case 187:
                        FACE(2, 1, 5);
                        FACE(6, 2, 5);
                        break;

                    case 188:
                        FACE(10, 11, 6);
                        FACE(6, 11, 3);
                        FACE(6, 3, 8);
                        FACE(6, 8, 9);
                        FACE(6, 9, 5);
                        break;

                    case 189:
                        FACE(10, 11, 6);
                        FACE(11, 0, 6);
                        FACE(0, 9, 6);
                        FACE(6, 9, 5);
                        break;

                    case 190:
                        FACE(6, 10, 5);
                        FACE(8, 0, 3);
                        break;

                    case 191:
                        FACE(6, 10, 5);
                        break;

                    case 192:
                        FACE(7, 10, 11);
                        FACE(7, 5, 10);
                        break;

                    case 193:
                        FACE(10, 7, 5);
                        FACE(10, 11, 7);
                        FACE(3, 0, 8);
                        break;

                    case 194:
                        FACE(7, 10, 11);
                        FACE(7, 5, 10);
                        FACE(9, 0, 1);
                        break;

                    case 195:
                        FACE(3, 9, 8);
                        FACE(3, 1, 9);
                        FACE(11, 7, 5);
                        FACE(11, 5, 10);
                        break;

                    case 196:
                        FACE(11, 1, 2);
                        FACE(11, 7, 1);
                        FACE(7, 5, 1);
                        break;

                    case 197:
                        FACE(11, 1, 2);
                        FACE(11, 7, 1);
                        FACE(7, 5, 1);
                        FACE(2, 8, 3);
                        break;

                    case 198:
                        FACE(2, 11, 0);
                        FACE(0, 11, 7);
                        FACE(0, 7, 9);
                        FACE(9, 7, 5);
                        break;

                    case 199:
                        FACE(11, 10, 2);
                        FACE(2, 10, 5);
                        FACE(2, 5, 9);
                        FACE(2, 9, 8);
                        FACE(2, 8, 3);
                        break;

                    case 200:
                        FACE(2, 5, 10);
                        FACE(2, 3, 5);
                        FACE(3, 7, 5);
                        break;

                    case 201:
                        FACE(5, 8, 7);
                        FACE(5, 0, 8);
                        FACE(5, 10, 0);
                        FACE(10, 2, 0);
                        break;

                    case 202:
                        FACE(2, 5, 10);
                        FACE(2, 3, 5);
                        FACE(3, 7, 5);
                        FACE(10, 0, 1);
                        break;

                    case 203:
                        FACE(1, 3, 2);
                        FACE(2, 3, 8);
                        FACE(2, 8, 7);
                        FACE(2, 7, 5);
                        FACE(2, 5, 10);
                        break;

                    case 204:
                        FACE(1, 3, 5);
                        FACE(5, 3, 7);
                        break;

                    case 205:
                        FACE(7, 5, 1);
                        FACE(7, 1, 0);
                        FACE(8, 7, 0);
                        break;

                    case 206:
                        FACE(3, 7, 5);
                        FACE(3, 5, 9);
                        FACE(0, 3, 9);
                        break;

                    case 207:
                        FACE(5, 9, 8);
                        FACE(7, 5, 8);
                        break;

                    case 208:
                        FACE(5, 8, 4);
                        FACE(5, 10, 8);
                        FACE(10, 11, 8);
                        break;

                    case 209:
                        FACE(4, 5, 0);
                        FACE(0, 5, 10);
                        FACE(0, 10, 3);
                        FACE(3, 10, 11);
                        break;

                    case 210:
                        FACE(5, 8, 4);
                        FACE(5, 10, 8);
                        FACE(10, 11, 8);
                        FACE(4, 1, 9);
                        break;

                    case 211:
                        FACE(5, 7, 4);
                        FACE(4, 7, 11);
                        FACE(4, 11, 3);
                        FACE(4, 3, 1);
                        FACE(4, 1, 9);
                        break;

                    case 212:
                        FACE(1, 4, 5);
                        FACE(1, 8, 4);
                        FACE(1, 2, 8);
                        FACE(2, 11, 8);
                        break;

                    case 213:
                        FACE(3, 7, 11);
                        FACE(11, 7, 4);
                        FACE(11, 4, 5);
                        FACE(11, 5, 1);
                        FACE(11, 1, 2);
                        break;

                    case 214:
                        FACE(9, 10, 5);
                        FACE(5, 10, 2);
                        FACE(5, 2, 11);
                        FACE(5, 11, 8);
                        FACE(5, 8, 4);
                        break;

                    case 215:
                        FACE(4, 5, 9);
                        FACE(3, 2, 11);
                        break;

                    case 216:
                        FACE(4, 5, 8);
                        FACE(8, 5, 10);
                        FACE(8, 10, 3);
                        FACE(3, 10, 2);
                        break;

                    case 217:
                        FACE(2, 0, 4);
                        FACE(2, 4, 5);
                        FACE(10, 2, 5);
                        break;

                    case 218:
                        FACE(9, 0, 1);
                        FACE(4, 5, 8);
                        FACE(8, 5, 10);
                        FACE(8, 10, 3);
                        FACE(3, 10, 2);
                        break;

                    case 219:
                        FACE(5, 7, 4);
                        FACE(7, 2, 4);
                        FACE(2, 1, 4);
                        FACE(4, 1, 9);
                        break;

                    case 220:
                        FACE(5, 1, 3);
                        FACE(5, 3, 8);
                        FACE(4, 5, 8);
                        break;

                    case 221:
                        FACE(4, 5, 1);
                        FACE(0, 4, 1);
                        break;

                    case 222:
                        FACE(9, 10, 5);
                        FACE(10, 3, 5);
                        FACE(3, 8, 5);
                        FACE(5, 8, 4);
                        break;

                    case 223:
                        FACE(4, 5, 9);
                        break;

                    case 224:
                        FACE(4, 11, 7);
                        FACE(4, 9, 11);
                        FACE(9, 10, 11);
                        break;

                    case 225:
                        FACE(4, 11, 7);
                        FACE(4, 9, 11);
                        FACE(9, 10, 11);
                        FACE(7, 0, 8);
                        break;

                    case 226:
                        FACE(11, 1, 10);
                        FACE(11, 0, 1);
                        FACE(11, 7, 0);
                        FACE(7, 4, 0);
                        break;

                    case 227:
                        FACE(8, 9, 4);
                        FACE(4, 9, 1);
                        FACE(4, 1, 10);
                        FACE(4, 10, 11);
                        FACE(4, 11, 7);
                        break;

                    case 228:
                        FACE(2, 11, 1);
                        FACE(1, 11, 7);
                        FACE(1, 7, 9);
                        FACE(9, 7, 4);
                        break;

                    case 229:
                        FACE(3, 0, 8);
                        FACE(2, 11, 1);
                        FACE(1, 11, 7);
                        FACE(1, 7, 9);
                        FACE(9, 7, 4);
                        break;

                    case 230:
                        FACE(4, 0, 2);
                        FACE(4, 2, 11);
                        FACE(7, 4, 11);
                        break;

                    case 231:
                        FACE(8, 9, 4);
                        FACE(9, 2, 4);
                        FACE(2, 11, 4);
                        FACE(4, 11, 7);
                        break;

                    case 232:
                        FACE(10, 2, 9);
                        FACE(9, 2, 3);
                        FACE(9, 3, 4);
                        FACE(4, 3, 7);
                        break;

                    case 233:
                        FACE(4, 6, 7);
                        FACE(7, 6, 10);
                        FACE(7, 10, 2);
                        FACE(7, 2, 0);
                        FACE(7, 0, 8);
                        break;

                    case 234:
                        FACE(2, 6, 10);
                        FACE(10, 6, 7);
                        FACE(10, 7, 4);
                        FACE(10, 4, 0);
                        FACE(10, 0, 1);
                        break;

                    case 235:
                        FACE(8, 4, 7);
                        FACE(10, 1, 2);
                        break;

                    case 236:
                        FACE(1, 3, 7);
                        FACE(1, 7, 4);
                        FACE(9, 1, 4);
                        break;

                    case 237:
                        FACE(4, 6, 7);
                        FACE(6, 1, 7);
                        FACE(1, 0, 7);
                        FACE(7, 0, 8);
                        break;

                    case 238:
                        FACE(0, 7, 3);
                        FACE(0, 4, 7);
                        break;

                    case 239:
                        FACE(8, 4, 7);
                        break;

                    case 240:
                        FACE(10, 11, 9);
                        FACE(9, 11, 8);
                        break;

                    case 241:
                        FACE(9, 10, 11);
                        FACE(9, 11, 3);
                        FACE(0, 9, 3);
                        break;

                    case 242:
                        FACE(10, 11, 8);
                        FACE(10, 8, 0);
                        FACE(1, 10, 0);
                        break;

                    case 243:
                        FACE(1, 10, 11);
                        FACE(3, 1, 11);
                        break;

                    case 244:
                        FACE(11, 8, 9);
                        FACE(11, 9, 1);
                        FACE(2, 11, 1);
                        break;

                    case 245:
                        FACE(1, 5, 9);
                        FACE(5, 11, 9);
                        FACE(11, 3, 9);
                        FACE(9, 3, 0);
                        break;

                    case 246:
                        FACE(8, 0, 2);
                        FACE(11, 8, 2);
                        break;

                    case 247:
                        FACE(3, 2, 11);
                        break;

                    case 248:
                        FACE(8, 9, 10);
                        FACE(8, 10, 2);
                        FACE(3, 8, 2);
                        break;

                    case 249:
                        FACE(0, 9, 10);
                        FACE(2, 0, 10);
                        break;

                    case 250:
                        FACE(0, 4, 8);
                        FACE(4, 10, 8);
                        FACE(10, 2, 8);
                        FACE(8, 2, 3);
                        break;

                    case 251:
                        FACE(2, 1, 10);
                        break;

                    case 252:
                        FACE(9, 1, 3);
                        FACE(8, 9, 3);
                        break;

                    case 253:
                        FACE(0, 9, 1);
                        break;

                    case 254:
                        FACE(8, 0, 3);
                        break;
                        
                    default:
                        HOPPE_LOG("ERR! Unknown state: %d", cell.state);
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

