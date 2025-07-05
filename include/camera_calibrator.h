#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace calib {

class CameraCalibrator {
public:
    explicit CameraCalibrator(const std::string& config_path);

    cv::Mat undistort(const cv::Mat& image);

    // 像素坐标 → 世界坐标（需要深度）
    cv::Point3f imageToWorld(const cv::Point2f& image_pt, float depth);

    // 获取相机内参 / 畸变 / 外参
    cv::Mat cameraMatrix() const;
    cv::Mat distCoeffs() const;
    cv::Mat rotationMatrix() const;
    cv::Mat translationVector() const;
    

private:
    cv::Mat camera_matrix_;         // 相机内参
    cv::Mat dist_coeffs_;           // 畸变系数
    cv::Mat rotation_matrix_;       // 外参：旋转矩阵
    cv::Mat translation_vector_;    // 外参：平移向量
};

}  // namespace calib
