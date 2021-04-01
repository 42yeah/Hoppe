//
//  Cube.hpp
//  hoppe
//
//  Created by apple on 28/03/2021.
//

#ifndef Cube_hpp
#define Cube_hpp

#include <opencv2/core.hpp>
#include <vector>


struct SampledPoint {
    cv::Point3f point;
    
    // Signed Distance Value
    float sdv;
};

class Cell {
    std::vector<SampledPoint> corners;
    
    int situation;
};

// Grid represents a network of cells.
class Grid {
public:
    Grid() {}

    Grid(cv::Vec3f bounding_box_min, cv::Vec3f bounding_box_max, float cell_size);
    
    Grid(const Grid &) = delete;
    
    Grid &operator=(const Grid &) = delete;
    
    Grid(Grid &&) = default;
    
private:
    float cell_size,
        half_cell_size;
    
    cv::Point3i num_cells;
    int num_cells_xy;
    
    std::vector<Cell> cells;
    
    std::vector<SampledPoint> corners;
    
    cv::Vec3f bounding_box_min, bounding_box_max;
};



#endif /* Cube_hpp */
