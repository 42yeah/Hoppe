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
#include <nanoflann.hpp>

#define HOPPE_LOG_LEVEL 1

#if HOPPE_LOG_LEVEL == 0
#define HOPPE_LOG(...)
#elif HOPPE_LOG_LEVEL == 1
#define HOPPE_LOG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#elif HOPPE_LOG_LEVEL == 2
#define HOPPE_LOG(fmt, ...) printf("%s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define POINT2VEC(p) cv::Vec3f(p.x, p.y, p.z)
#define VEC2POINT(v) cv::Point3f(v(0), v(1), v(2))


struct Plane {
    cv::Point3f origin;
    cv::Vec3f normal;
};

struct Parameters {
    int k;
    float density, noise, isolevel;
    unsigned long max_volume;
};

class PointCloud {
public:
    inline auto kdtree_get_point_count() const -> std::size_t {
        return points.size();
    }
    
    inline auto kdtree_get_pt(std::size_t idx, int dim) const -> float {
        if (dim == 0) {
            return points[idx].x;
        } else if (dim == 1) {
            return points[idx].y;
        } else {
            return points[idx].z;
        }
    }
    
    template<class BBox>
    auto kdtree_get_bbox(BBox &) const -> bool {
        return false;
    }
    
    std::vector<cv::Point3f> points;
};

class Planes {
public:
    inline auto kdtree_get_point_count() const -> std::size_t {
        return planes.size();
    }
    
    inline auto kdtree_get_pt(std::size_t idx, int dim) const -> float {
        if (dim == 0) {
            return planes[idx].origin.x;
        } else if (dim == 1) {
            return planes[idx].origin.y;
        } else {
            return planes[idx].origin.z;
        }
    }
    
    template<class BBox>
    auto kdtree_get_bbox(BBox &) const -> bool {
        return false;
    }
    
    std::vector<Plane> planes;
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3
> PointCloudIndex;
typedef std::size_t IndexToTangentPlane;

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Planes>,
    Planes,
    3
> PlaneCloudIndex;

#endif /* hoppe_common_hpp */
