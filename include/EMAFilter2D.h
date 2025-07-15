#pragma once
#include <opencv2/core.hpp>

class EMAFilter2D {
public:
    explicit EMAFilter2D(float alpha = 0.2f) : a(alpha), first(true) {}
    
    cv::Point2f update(const cv::Point2f& z) {
        if (first) { 
            s = z; 
            first = false; 
        } else {
            s.x = a * z.x + (1 - a) * s.x;
            s.y = a * z.y + (1 - a) * s.y;
        }
        return s;
    }

private:
    float a;  // 平滑因子
    bool first;
    cv::Point2f s;
};
