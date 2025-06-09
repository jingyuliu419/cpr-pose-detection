#include "camera_calibrator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/imgproc/imgproc.hpp>
#ifdef undistort
#  error "undistort 被定义成宏了"
#endif


namespace calib {

CameraCalibrator::CameraCalibrator(const std::string& config_path) {
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot open calibration file: " + config_path);
    }
    fs["camera_matrix"] >> camera_matrix_;
    fs["distortion_coefficients"] >> dist_coeffs_;
    fs.release();
}

cv::Mat CameraCalibrator::undistort(const cv::Mat& image) {
    cv::Mat undistorted;
    ::cv::undistort(image, undistorted, camera_matrix_, dist_coeffs_);
    return undistorted;
}

}  // namespace calib
