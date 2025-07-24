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

    /* ---------- 9×12  Charuco board，DICT_5X5_100 ---------- */
    auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    auto board = cv::aruco::CharucoBoard::create(
        12, 9,           // squaresX, squaresY
        0.06, 0.045,     // squareLength, markerLength（单位：米，对应 60mm / 45mm）
        dictionary
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
        rotation_ = R_or_rvec.clone();  // ✅ 显式设置
    } else {
        rvec = R_or_rvec.clone();
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        rotation_ = R.clone();          // ✅ 显式设置
    }

    rvec_list_[idx] = rvec;
    tvec_list_[idx] = tvec.clone();

    translation_ = tvec.clone();        // ✅ 显式设置
    if (default_cam_ < 0) default_cam_ = idx;
}


// 完整函数：增强 pixel2world 支持世界坐标变换（包含 Rig 外参）
cv::Point3f CameraManager::pixel2world(const cv::Point2f& px, float z) const
{
    if (default_cam_ < 0 || default_cam_ >= static_cast<int>(camera_matrix_list_.size()))
        return {};

    // -------- 取内参矩阵 --------
    cv::Mat Kd;
    camera_matrix_list_[default_cam_].convertTo(Kd, CV_64F);
    double fx = Kd.at<double>(0, 0), fy = Kd.at<double>(1, 1);
    double cx = Kd.at<double>(0, 2), cy = Kd.at<double>(1, 2);

    // -------- 像素 → 相机坐标系 --------
    double x = (px.x - cx) * z / fx;
    double y = (px.y - cy) * z / fy;
    cv::Mat pt_cam = (cv::Mat_<double>(3, 1) << x, y, z);

    // -------- 相机坐标 → 世界坐标（如果设置了 rig 外参） --------
    if (!rig_R_.empty() && !rig_t_.empty()) {
        cv::Mat pt_world = rig_R_ * pt_cam + rig_t_;
        return cv::Point3f(
            static_cast<float>(pt_world.at<double>(0)),
            static_cast<float>(pt_world.at<double>(1)),
            static_cast<float>(pt_world.at<double>(2)));
    } else {
        return cv::Point3f(static_cast<float>(x), static_cast<float>(y), z);  // fallback
    }
}

void CameraManager::setRigExtrinsicsRaw(const cv::Mat& R, const cv::Mat& t, int idx) {
    ensure_size(rig_R_raw_, idx + 1);
    ensure_size(rig_t_raw_, idx + 1);
    rig_R_raw_[idx] = R.clone();
    rig_t_raw_[idx] = t.clone();
}



void CameraManager::saveRigExtrinsicsToYaml(const std::string& path, const std::vector<int>& cam_ids) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "[RigExtrinsics] Failed to open for write: " << path << std::endl;
        return;
    }

    for (int cam_id : cam_ids) {
        if (cam_id >= static_cast<int>(rig_R_raw_.size()) || rig_R_raw_[cam_id].empty())
            continue;

        fs << "R" + std::to_string(cam_id) << rig_R_raw_[cam_id];
        fs << "t" + std::to_string(cam_id) << rig_t_raw_[cam_id];
    }

    std::cout << "[RigExtrinsics] Saved rig extrinsics for " << cam_ids.size() << " cameras." << std::endl;
}

// ========== Rig Extrinsics Loader ==========
void CameraManager::loadRigExtrinsicsFromYaml(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[RigExtrinsics] Cannot open: " << path << std::endl;
        return;
    }

    // 遍历所有 keys
    for (cv::FileNodeIterator it = fs.root().begin(); it != fs.root().end(); ++it) {
        std::string key = (*it).name();
        if (key.empty() || key[0] != 'R') continue;

        int cam_id = std::stoi(key.substr(1));
        cv::Mat R, t;
        fs[key] >> R;
        fs["t" + std::to_string(cam_id)] >> t;

        if (R.empty() || t.empty()) {
            std::cerr << "[RigExtrinsics] Missing R or t for cam " << cam_id << std::endl;
            continue;
        }

        ensure_size(rig_R_raw_, cam_id + 1);
        ensure_size(rig_t_raw_, cam_id + 1);
        rig_R_raw_[cam_id] = R.clone();
        rig_t_raw_[cam_id] = t.clone();

        cv::Mat R_inv = R.t();
        cv::Mat t_inv = -R_inv * t;
        setExtrinsics(R_inv, t_inv, cam_id);
    }

    std::cout << "[RigExtrinsics] Loaded rig extrinsics for all valid camera ids in YAML." << std::endl;
}

// ========== Save All Rig Extrinsics to YAML ==========
void CameraManager::saveAllRigExtrinsicsToYaml(
    const std::string& path,
    const std::vector<std::shared_ptr<CameraManager>>& mgrs,
    const std::vector<int>& cam_ids) {

    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "[RigExtrinsics] Cannot open for writing: " << path << std::endl;
        return;
    }

    for (size_t i = 0; i < mgrs.size(); ++i) {
        const auto& mgr = mgrs[i];
        int id = cam_ids[i];

        if (id < 0 || id >= static_cast<int>(mgr->rig_R_raw_.size()) ||
            mgr->rig_R_raw_[id].empty() || mgr->rig_t_raw_[id].empty()) {
            std::cerr << "[RigExtrinsics] Skipping cam_id=" << id << " due to empty extrinsics." << std::endl;
            continue;
        }

        fs << ("R" + std::to_string(id)) << mgr->rig_R_raw_[id];
        fs << ("t" + std::to_string(id)) << mgr->rig_t_raw_[id];
    }
    fs.release();
    std::cout << "[RigExtrinsics] Saved rig extrinsics to: " << path << std::endl;
}

// ========== Project World Axes to Image ==========
std::vector<cv::Point2f> CameraManager::projectWorldAxes2D(
        const std::vector<cv::Point3f>& axes3d, int cam_id)
{
    std::vector<cv::Point2f> img_pts;

    /* ① Rig 外参要能取到 */
    if(cam_id >= rig_R_raw_.size() || rig_R_raw_[cam_id].empty())
        return img_pts;

    /* ② K / D 统一用 default_cam_（只有一份内参时就是 0） */
    int idxK = (default_cam_ >= 0) ? default_cam_ : 0;

    try {
        cv::projectPoints(axes3d,
                          rig_R_raw_[cam_id],         // ← Rig R/t 仍用设备号
                          rig_t_raw_[cam_id],
                          camera_matrix_list_[idxK],  // ← 改这里
                          dist_coeffs_list_[idxK],    // ← 还有这里
                          img_pts);
    } catch (const std::exception& e) {
        std::cerr << "[Error] projectWorldAxes2D: " << e.what() << '\n';
    }
    return img_pts;
}


// ========== Draw Axes in Inference ==========
void CameraManager::drawOriginAxesUnified(cv::Mat& img, int cam_id) {
    if (!hasValidExtrinsics(cam_id) || cam_id >= rig_R_raw_.size()) return;
    const std::vector<cv::Point3f> world_axes = {
        {0, 0, 0}, {0.1f, 0, 0}, {0, 0.1f, 0}, {0, 0, 0.1f}
    };
    const auto img_pts = projectWorldAxes2D(world_axes, cam_id);
    if (img_pts.size() < 4) return;

    cv::drawMarker(img, img_pts[0], {0,255,255}, cv::MARKER_CROSS, 12, 2);
    cv::arrowedLine(img, img_pts[0], img_pts[1], cv::Scalar(0,0,255), 3);
    cv::arrowedLine(img, img_pts[0], img_pts[2], cv::Scalar(0,255,0), 3);
    cv::arrowedLine(img, img_pts[0], img_pts[3], cv::Scalar(255,0,0), 3);
}
const cv::Mat& CameraManager::distCoeffs(int idx) const
{
    int i = (idx < 0) ? default_cam_ : idx;
    static cv::Mat empty;                // 兜底，避免返回引用空对象
    if (i < 0 || i >= dist_coeffs_list_.size())
        return empty;
    return dist_coeffs_list_[i];
}


} // namespace charuco