//
//  Cube.cpp
//  hoppe
//
//  Created by apple on 28/03/2021.
//

#include "Cube.hpp"


Grid::Grid(cv::Vec3f bounding_box_min, cv::Vec3f bounding_box_max, float cell_size) : cell_size(cell_size), half_cell_size(cell_size * 0.5f) {
    this->bounding_box_min = cv::Vec3f(bounding_box_min(0) - cell_size,
                                       bounding_box_min(1) - cell_size,
                                       bounding_box_min(2) - cell_size);
    this->bounding_box_max = bounding_box_max;
    num_cells = cv::Point3i((bounding_box_max(0) - bounding_box_min(0)) / cell_size,
                            (bounding_box_max(1) - bounding_box_min(1)) / cell_size,
                            (bounding_box_max(2) - bounding_box_min(2)) / cell_size);
    num_cells_xy = num_cells.x * num_cells.y;
}
