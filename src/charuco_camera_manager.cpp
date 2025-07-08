#include "charuco_camera_manager.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <iostream>
#include <mutex>

namespace charuco {
// void ensure_size(std::vector<cv::Mat>& v, size_t n){ if (v.size()<n) v.resize(n); }
CameraManager::CameraManager(const std::vector<std::string>& sources)
    : sources_(sources) {
        cb_ = [](const cv::Mat&,int){};     // 空安全回调
    }

CameraManager::~CameraManager() { stop(); }

void CameraManager::start(FrameCallback cb_) {
    cb = cb_;
    running_ = true;
    for (size_t i = 0; i < sources_.size(); ++i) {
        threads_.emplace_back(&CameraManager::captureThread, this,
                              static_cast<int>(i), sources_[i]);
    }
}

void CameraManager::stop() {
    running_ = false;
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

void CameraManager::captureThread(int index, const std::string& source) {
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "[CameraManager] Failed to open source: " << source
                  << std::endl;
        return;
    }

    while (running_) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(cap_mutex_);
            cap >> frame;
        }
        if (!frame.empty() && cb_) { cb_(frame, index); }
    }
    cap.release();
}

void CameraManager::loadCalibration(
    const std::vector<std::string>& yaml_paths) {
    camera_matrix_list_.clear();
    dist_coeffs_list_.clear();
    rvec_list_.clear();
    tvec_list_.clear();

    for (const auto& path : yaml_paths) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[CameraManager] Failed to open calibration file: "
                      << path << std::endl;
            continue;
        }

        cv::Mat K, D, Rvec, Tvec;
        fs["camera_matrix"] >> K;
        fs["distortion_coefficients"] >> D;
        fs["rvec"] >> Rvec;
        fs["tvec"] >> Tvec;

        if (K.empty() || D.empty()) {
            std::cerr << "[Warning] Invalid calibration data in: " << path
                      << std::endl;
            continue;
        }

        camera_matrix_list_.push_back(K);
        dist_coeffs_list_.push_back(D);
        rvec_list_.push_back(Rvec);
        tvec_list_.push_back(Tvec);
    }

    if (!camera_matrix_list_.empty()) { default_cam_ = 0; }
}

cv::Mat CameraManager::undistort(const cv::Mat& frame) const {
    if (default_cam_ >= camera_matrix_list_.size()) return frame.clone();
    cv::Mat undistorted;
    cv::undistort(frame, undistorted, camera_matrix_list_[default_cam_],
                  dist_coeffs_list_[default_cam_]);
    return undistorted;
}

// cv::Point3f CameraManager::pixel2world(const cv::Point2f& px, float z) const
// {
//     if (default_cam_ < 0 || default_cam_ >= camera_matrix_list_.size())
//         return {};

//     /* -------- 像素 → 相机坐标 -------- */
//     const cv::Mat& K   = camera_matrix_list_[default_cam_];
//     const cv::Mat& rv  = rvec_list_[default_cam_];
//     const cv::Mat& tv  = tvec_list_[default_cam_];
//     if (rv.empty() || tv.empty()) {
//         std::cerr << "[CameraManager] Empty r/t for cam " << default_cam_ << '\n';
//         return {};
//     }

//     cv::Mat R_cam;
//     cv::Rodrigues(rv, R_cam);

//     double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
//     double cx = K.at<double>(0,2), cy = K.at<double>(1,2);
//     double x_cam = (px.x - cx) * z / fx;
//     double y_cam = (px.y - cy) * z / fy;
//     cv::Mat p_cam = (cv::Mat_<double>(3,1) << x_cam, y_cam, z);

//     cv::Mat p_world = R_cam * p_cam + tv;                // 相机自身 world

//     /* -------- 若有 Rig 外参，再转换 -------- */
//     if (!rig_R_.empty() && !rig_t_.empty())
//         p_world = rig_R_ * p_world + rig_t_;

//     return { (float)p_world.at<double>(0),
//              (float)p_world.at<double>(1),
//              (float)p_world.at<double>(2) };
// }


void CameraManager::calibrateFromLive(int cam_idx,
                                      const std::string& yaml_output_path)
{
    if (cam_idx < 0 || cam_idx >= static_cast<int>(sources_.size())) {
        std::cerr << "[Calibration] Invalid camera index: " << cam_idx << '\n';
        return;
    }

    cv::VideoCapture cap(sources_[cam_idx]);
    if (!cap.isOpened()) {
        std::cerr << "[Calibration] Cannot open camera " << cam_idx << '\n';
        return;
    }

    /* ---------- 6×6  Charuco board，DICT_5X5_100 ---------- */
    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    auto board = cv::aruco::CharucoBoard::create(
        12, 9,          // squaresX, squaresY (格子数)
        0.015, 0.01125, // squareLength, markerLength（单位：米）
        dictionary      // 建议与图片上保持一致，如 DICT_5X5_100
    );


    std::vector<cv::Mat> frames;
    std::vector<std::vector<std::vector<cv::Point2f>>> all_corners;
    std::vector<std::vector<int>> all_ids;
    collectCalibrationImages(cap, frames, all_corners, all_ids, board);

    if (frames.empty()) {
        std::cerr << "[Calibration] No valid frames captured\n";
        return;
    }

    /* ---------- 提取 Charuco 角点 ---------- */
    std::vector<cv::Mat> charuco_corners, charuco_ids;
    for (size_t i = 0; i < frames.size(); ++i) {
        cv::Mat cc, ids;
        cv::aruco::interpolateCornersCharuco(all_corners[i], all_ids[i],
                                             frames[i], board, cc, ids);
        if (!ids.empty()) { charuco_corners.push_back(cc); charuco_ids.push_back(ids); }
    }
    if (charuco_ids.empty()) {
        std::cerr << "[Calibration] No Charuco corners detected\n";
        return;
    }

    /* ---------- 标定 ---------- */
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs   = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    double err = cv::aruco::calibrateCameraCharuco(
        charuco_corners, charuco_ids, board, frames[0].size(),
        cameraMatrix, distCoeffs, rvecs, tvecs,
        cv::noArray(), cv::noArray(), cv::noArray());

    std::cout << "[Calibration] Reprojection error: " << err << std::endl;

    cv::Mat rvec = rvecs.empty() ? cv::Mat::zeros(3,1,CV_64F) : rvecs[0];
    cv::Mat tvec = tvecs.empty() ? cv::Mat::zeros(3,1,CV_64F) : tvecs[0];

    saveCalibration(yaml_output_path, cameraMatrix, distCoeffs, rvec, tvec);

    /* ---------- 缓存到内存 ---------- */
    camera_matrix_list_.push_back(cameraMatrix);
    dist_coeffs_list_.push_back(distCoeffs);
    rvec_list_.push_back(rvec);
    tvec_list_.push_back(tvec);
    default_cam_ = static_cast<int>(camera_matrix_list_.size() - 1);
}


void CameraManager::collectCalibrationImages(cv::VideoCapture& cap,
                                            std::vector<cv::Mat>& frames,
                                            std::vector<std::vector<std::vector<cv::Point2f>>>& all_corners,
                                            std::vector<std::vector<int>>& all_ids,
                                            cv::Ptr<cv::aruco::CharucoBoard>& board,
                                            int max_samples) {
    auto dictionary = board->dictionary; // 使用 CharucoBoard 的字典
    const int min_corners = 6;          // collectCalibrationImages() 里可适当调低
    int collected = 0;

    cv::namedWindow("Charuco Capture", cv::WINDOW_NORMAL);
    
    while (collected < max_samples) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(cap_mutex_);
            cap >> frame;
        }
        if (frame.empty()) continue;

        cv::Mat frame_display = frame.clone();
        std::vector<int> marker_ids;
        std::vector<std::vector<cv::Point2f>> marker_corners;
        
        // 检测标记
        cv::aruco::detectMarkers(frame, dictionary, marker_corners, marker_ids);
        
        if (!marker_ids.empty()) {
            // 检测Charuco角点
            cv::Mat charucoCorners, charucoIds;
            cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, frame, board, 
                                                charucoCorners, charucoIds);
            
            // 确保有足够角点
            if (!charucoIds.empty() && charucoIds.rows >= min_corners) {
                all_corners.push_back(marker_corners);
                all_ids.push_back(marker_ids);
                frames.push_back(frame.clone());
                collected++;
                std::cout << "Captured calibration frame " << collected << "/" << max_samples << std::endl;
            }
            
            // 绘制检测结果
            cv::aruco::drawDetectedMarkers(frame_display, marker_corners, marker_ids);
            if (!charucoIds.empty()) {
                cv::aruco::drawDetectedCornersCharuco(frame_display, charucoCorners, charucoIds);
            }
        }
        
        // 显示状态信息
        std::string status = "Frames: " + std::to_string(collected) + "/" + std::to_string(max_samples);
        cv::putText(frame_display, status, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Charuco Capture", frame_display);
        int key = cv::waitKey(30);
        if (key == 27) break; // ESC退出
        if (key == ' ') {      // 空格键手动捕获
            if (!marker_ids.empty()) {
                all_corners.push_back(marker_corners);
                all_ids.push_back(marker_ids);
                frames.push_back(frame.clone());
                collected++;
                std::cout << "Manually captured frame " << collected << "/" << max_samples << std::endl;
            }
        }
    }
    cv::destroyWindow("Charuco Capture");
}

void CameraManager::saveCalibration(const std::string& path,
                                    const cv::Mat& K,
                                    const cv::Mat& D,
                                    const cv::Mat& rvec,
                                    const cv::Mat& tvec) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "[Calibration] Failed to write calibration to: " << path << std::endl;
        return;
    }

    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << D;
    fs << "rvec" << rvec;
    fs << "tvec" << tvec;
    fs.release();
    std::cout << "[Calibration] Saved calibration to: " << path << std::endl;
}

bool CameraManager::hasValidExtrinsics(int idx) const
{
    int i = (idx < 0) ? default_cam_ : idx;
    if (i < 0 || i >= static_cast<int>(rvec_list_.size())) return false;
    return !rvec_list_[i].empty() && !tvec_list_[i].empty();
}

// void CameraManager::setExtrinsics(const cv::Mat& R, const cv::Mat& t) {
//     rotation_ = R.clone();
//     translation_ = t.clone();
//     has_extrinsics_ = true;
// }
static void ensure_size(std::vector<cv::Mat>& v, size_t n)
{
    if (v.size() < n) v.resize(n);
}

void CameraManager::setCameraMatrix(const cv::Mat& K, int idx)
{
    if (idx < 0) idx = default_cam_ < 0 ? 0 : default_cam_;
    ensure_size(camera_matrix_list_, idx + 1);
    camera_matrix_list_[idx] = K.clone();
    if (default_cam_ < 0) default_cam_ = idx;
}

void CameraManager::setDistCoeffs(const cv::Mat& D, int idx)
{
    if (idx < 0) idx = default_cam_ < 0 ? 0 : default_cam_;
    ensure_size(dist_coeffs_list_, idx + 1);
    dist_coeffs_list_[idx] = D.clone();
    if (default_cam_ < 0) default_cam_ = idx;
}

void CameraManager::setExtrinsics(const cv::Mat& R_or_rvec, const cv::Mat& tvec, int idx)
{
    if (idx < 0) idx = default_cam_ < 0 ? 0 : default_cam_;
    ensure_size(rvec_list_, idx + 1);
    ensure_size(tvec_list_, idx + 1);

    cv::Mat rvec;
    if (R_or_rvec.rows == 3 && R_or_rvec.cols == 3) {
        cv::Rodrigues(R_or_rvec, rvec);
    } else {
        rvec = R_or_rvec.clone();
    }

    rvec_list_[idx] = rvec;
    tvec_list_[idx] = tvec.clone();
    if (default_cam_ < 0) default_cam_ = idx;
}

cv::Point3f CameraManager::pixel2world(const cv::Point2f& px, float z) const
{
    cv::Mat Kd;
    camera_matrix_list_[default_cam_].convertTo(Kd, CV_64F);
    double fx = Kd.at<double>(0, 0), fy = Kd.at<double>(1, 1);
    double cx = Kd.at<double>(0, 2), cy = Kd.at<double>(1, 2);

    float x = (px.x - cx) * z / fx;
    float y = (px.y - cy) * z / fy;
    return cv::Point3f(x, y, z);
}


} // namespace charuco