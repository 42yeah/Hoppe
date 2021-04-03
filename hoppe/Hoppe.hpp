//
//  Hoppe.hpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#ifndef Hoppe_hpp
#define Hoppe_hpp


#include <optional>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "hoppe_common.hpp"
#include "CubeMarcher.hpp"

#define IN
#define OUT


class Hoppe {
public:
    Hoppe() : parameters({ 8, -1.0f, 0.0f, 0.0f, 8000000ul }) {}
    
    Hoppe(Parameters param) : parameters(param) {}

    ~Hoppe() = default;
    
    /// Runs the Hoppe Surface Reconstruction method.
    /// @returns true on success
    auto run() -> bool;


    /// Loads point cloud from `path`.
    /// @param path path to load point cloud
    auto load_pointcloud(std::string path) -> void;
    
    auto export_to_ply(const std::string path) -> void;
    
    Parameters parameters;
    
private:
    auto estimate_planes() -> bool;
    
    auto fix_orientations() -> void;
    
    auto create_grid_bounds() -> void;
    

    /// Estimate density, to be used later in sdf function
    /// @param bounding_box_size size of the bounding box
    auto density_estimation(cv::Vec3f bounding_box_size) -> float;
    
    auto cube_march() -> void;
    
    auto create_mesh() -> void;
    

    /// Calculate bounds of the bounding box.
    /// @param bounding_box_min minimal bounding box
    /// @param bounding_box_max maximum bounding box
    auto calculate_bounds(OUT cv::Vec3f &bounding_box_min,
                          OUT cv::Vec3f &bounding_box_max) -> void;


    /// Find the signed distance from the sample point to the model M.
    /// In reality, M is not really known, however we do have tangent planes.
    /// So we sample tangent plane instead.
    /// @param point point to calculate distance from
    auto sdf(cv::Point3f point) -> std::optional<float>;
    
    PointCloud pointcloud;
    Planes tangent_planes;
    CubeMarcher marcher;
};

#endif /* Hoppe_hpp */
