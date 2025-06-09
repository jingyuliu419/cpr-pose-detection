#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>

namespace calib {

class CameraCalibrator {
public:
    explicit CameraCalibrator(const std::string& config_path);
    cv::Mat undistort(const cv::Mat& image);

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
};

}  // namespace calib
