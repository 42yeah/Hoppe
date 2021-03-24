//
//  Hoppe.hpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#ifndef Hoppe_hpp
#define Hoppe_hpp

#define HOPPE_LOG_LEVEL 1

#if HOPPE_LOG_LEVEL == 0
#define HOPPE_LOG(...)
#elif HOPPE_LOG_LEVEL == 1
#define HOPPE_LOG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#elif HOPPE_LOG_LEVEL == 2
#define HOPPE_LOG(fmt, ...) printf("%s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "hoppe_common.hpp"


class Hoppe {
public:
    Hoppe() : parameters({ 8, -1.0f, 0.0f, 0.0f }) {}
    
    Hoppe(Parameters param) : parameters(param) {}

    ~Hoppe() = default;
    
    /// Runs the Hoppe Surface Reconstruction method.
    /// @returns true on success
    auto run() -> bool;


    /// Loads point cloud from `path`.
    /// @param path path to load point cloud
    auto load_pointcloud(std::string path) -> void;
    
    Parameters parameters;
    
private:
    auto estimate_planes() -> void;
    
    auto propagate_normals() -> void;
    
    auto create_grid_bounds() -> void;
    
    auto density_estimation() -> float;
    
    auto cube_march() -> void;
    
    auto create_mesh() -> void;
    
    auto export_to_ply() -> void;
    
    PointCloud pointcloud;
    Planes tangent_planes;
};

#endif /* Hoppe_hpp */
