//
//  CubeMarcher.hpp
//  hoppe
//
//  Created by apple on 03/04/2021.
//

#ifndef CubeMarcher_hpp
#define CubeMarcher_hpp

#include <vector>
#include <opencv2/core.hpp>
#include <functional>
#include <optional>
#include <map>
#include "hoppe_common.hpp"


struct Cell {
public:
    float values[8];

    int state;
};

struct Triangle {
    cv::Point3f points[3];
    
    auto operator+(const cv::Point3f offset) -> Triangle {
        return Triangle { points[0] + offset,
                          points[1] + offset,
                          points[2] + offset };
    }
};

typedef std::vector<std::vector<std::vector<Cell> > > CellMat;

/// Implements the Marching Cube algorithm.
class CubeMarcher {
public:
    CubeMarcher() {}
    
    CubeMarcher(cv::Vec3i size, float resolution) {
        init(size, resolution);
    }
    
    auto init(cv::Vec3i size, float resolution) -> void;
    
    auto march(std::function<std::optional<float>(cv::Point3f)> sdf,
               cv::Point3f offset) -> void;
    
    
    /// For debugging purposes only
    /// @param to dump to which path
    auto dump(std::string to) -> void;
    
    std::vector<Triangle> faces;

private:
    CellMat cell_mat;
    cv::Vec3i size;
    float resolution;
};

#endif /* CubeMarcher_hpp */
