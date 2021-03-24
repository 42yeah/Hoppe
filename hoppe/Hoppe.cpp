//
//  Hoppe.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include "Hoppe.hpp"
#include <fstream>


auto Hoppe::run() -> bool {
    if (pointcloud.size() == 0) {
        HOPPE_LOG("ERR! Can't run without point cloud");
        return false;
    }

    estimate_planes();

    return true;
}

auto Hoppe::load_pointcloud(std::string path) -> void {
    HOPPE_LOG("Loading point cloud...");
    std::ifstream reader(path);
    
    if (!reader.good()) {
        HOPPE_LOG("WARNING! Bad reader: %s", path.c_str());
        return;
    }
    pointcloud.clear();
    while (!reader.eof()) {
        cv::Point3f p;
        reader >> p.x >> p.y >> p.z;
        pointcloud.push_back(p);
    }
    HOPPE_LOG("Point cloud loading done. Size: %lu", pointcloud.size());
}

auto Hoppe::estimate_planes() -> void { 
    HOPPE_LOG("Esimating tangent planes...");
    tangent_planes.clear();
    
    
}

