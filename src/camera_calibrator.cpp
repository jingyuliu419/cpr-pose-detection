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

    if (fs["rotation_matrix"].empty() || fs["translation_vector"].empty()) {
        std::cerr << "[WARN] rotation_matrix or translation_vector missing in YAML. Skipping extrinsics.\n";
        rotation_matrix_ = cv::Mat();  // empty matrix
        translation_vector_ = cv::Mat();
    } else {
        fs["rotation_matrix"] >> rotation_matrix_;
        fs["translation_vector"] >> translation_vector_;
        translation_vector_ /= 1000.0;  
    }

    // 检查类型
    if (!camera_matrix_.data || camera_matrix_.type() != CV_64F) {
        camera_matrix_.convertTo(camera_matrix_, CV_64F);
    }
    if (!dist_coeffs_.data || dist_coeffs_.type() != CV_64F) {
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
    }
    if (!rotation_matrix_.empty() && rotation_matrix_.type() != CV_64F) {
        rotation_matrix_.convertTo(rotation_matrix_, CV_64F);
    }
    if (!translation_vector_.empty() && translation_vector_.type() != CV_64F) {
        translation_vector_.convertTo(translation_vector_, CV_64F);
    }

    fs.release();
}


cv::Mat CameraCalibrator::undistort(const cv::Mat& image) {
    cv::Mat undistorted;
    ::cv::undistort(image, undistorted, camera_matrix_, dist_coeffs_);
    return undistorted;
}

cv::Point3f CameraCalibrator::imageToWorld(const cv::Point2f& image_pt, float depth) {
    cv::Mat pt_hom = (cv::Mat_<double>(3,1) << image_pt.x, image_pt.y, 1.0);
    cv::Mat cam_coord = camera_matrix_.inv() * pt_hom * depth;  // 3x1
    cv::Mat world_coord = rotation_matrix_ * cam_coord + translation_vector_;  // 3x1
    return cv::Point3f(world_coord.at<double>(0), world_coord.at<double>(1), world_coord.at<double>(2));
}

cv::Mat CameraCalibrator::cameraMatrix() const {
    return camera_matrix_;
}

cv::Mat CameraCalibrator::distCoeffs() const {
    return dist_coeffs_;
}

cv::Mat CameraCalibrator::rotationMatrix() const {
    return rotation_matrix_;
}

cv::Mat CameraCalibrator::translationVector() const {
    return translation_vector_;
}

}  // namespace calib
