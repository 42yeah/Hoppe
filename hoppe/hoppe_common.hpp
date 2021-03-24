//
//  hoppe_common.hpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#ifndef hoppe_common_hpp
#define hoppe_common_hpp

#include <vector>
#include <opencv2/core.hpp>


struct Plane {
    cv::Point3f origin;
    cv::Point3f normal;
};

struct Parameters {
    int k;
    float density, noise, isolevel;
};


typedef std::vector<cv::Point3f> PointCloud;
typedef std::vector<Plane> Planes;

#endif /* hoppe_common_hpp */
