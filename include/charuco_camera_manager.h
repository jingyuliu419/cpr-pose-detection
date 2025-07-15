#ifndef CHARUCO_CAMERA_MANAGER_H
#define CHARUCO_CAMERA_MANAGER_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <functional>
#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include<iostream>
/****************  QUICK PATCH – rename symbols  ****************/
#define cb          cb_          // ↩︎ 头文件用 cb_
#define cap_mutex_  cap_mtx_     // ↩︎ 头文件用 cap_mtx_
/****************************************************************/

namespace charuco {

class CameraManager {
public:
    using FrameCallback = std::function<void(cv::Mat,int)>;

    explicit CameraManager(const std::vector<std::string>& sources);
    ~CameraManager();

    /* ---------- 采集 ---------- */
    void start(FrameCallback cb);
    void stop();

    /* ---------- 单机内参 / 去畸变 ---------- */
    void loadCalibration(const std::vector<std::string>& yamls);
    cv::Mat undistort(const cv::Mat& frame) const;

    /* ---------- Rig 外参 ---------- */
    void setRigExtrinsics(const cv::Mat& R_rig_cam, const cv::Mat& t_rig_cam){
        rig_R_ = R_rig_cam.clone();  rig_t_ = t_rig_cam.clone();
    }
    bool hasRigExtrinsics() const { return !rig_R_.empty() && !rig_t_.empty(); }
    const cv::Mat& rigRmat() const { return rig_R_; }
    const cv::Mat& rigTvec() const { return rig_t_; }
    const cv::Mat& getDist() const { return dist_coeffs_list_[default_cam_]; }
    const cv::Mat& getK()   const { return camera_matrix_list_[default_cam_]; }

    /* ---------- 单帧像素 → Rig 世界 ---------- */
    cv::Point3f pixel2world(const cv::Point2f& px, float z) const;

    /* ---------- 单机 Charuco 标定 ---------- */
    void calibrateFromLive(int cam_idx, const std::string& out_yaml);
    bool hasValidExtrinsics(int idx = -1) const;          // rvec+tvec 已存在?

    /* ---------- 内参直接访问 ---------- */
    const cv::Mat& K()    const { return camera_matrix_list_[default_cam_]; }
    const cv::Mat& rvec() const { return rvec_list_[default_cam_]; }
    const cv::Mat& tvec() const { return tvec_list_[default_cam_]; }
    // 在 public 区域添加
    const cv::Mat& rotationMatrix() const { return rotation_; }
    const cv::Mat& translationVector() const { return translation_; }


    // void setCameraMatrix(const cv::Mat& K) {
    //     camera_matrix_list_.clear();
    //     camera_matrix_list_.push_back(K.clone());
    //     default_cam_ = 0;
    // }

    // void setDistCoeffs(const cv::Mat& D) {
    //     dist_coeffs_list_.clear();
    //     dist_coeffs_list_.push_back(D.clone());
    // }


    // void setExtrinsics(const cv::Mat& R, const cv::Mat& t);
    bool hasExtrinsics() const { return has_extrinsics_; }
    const cv::Mat& getRotation() const { return rotation_; }
    const cv::Mat& getTranslation() const { return translation_; }

    void setCameraMatrix(const cv::Mat& K, int idx = -1);
    void setDistCoeffs (const cv::Mat& D, int idx = -1);
    void setExtrinsics(const cv::Mat& R_or_rvec, const cv::Mat& tvec, int idx = -1);

    void loadRigExtrinsicsFromYaml(const std::string& path);
    void setRigExtrinsicsRaw(const cv::Mat& R, const cv::Mat& t, int idx);
    std::vector<cv::Point2f> projectWorldAxes2D(const std::vector<cv::Point3f>& axes3d, int cam_id);

    void drawOriginAxesUnified(cv::Mat& img, int cam_id);
    void saveRigExtrinsicsToYaml(const std::string& path, const std::vector<int>& cam_ids);
    // CameraManager.h
    static void saveAllRigExtrinsicsToYaml(
        const std::string& path,
        const std::vector<std::shared_ptr<CameraManager>>& mgrs,
        const std::vector<int>& cam_ids);



private:
    /* 线程化采集 */
    void captureThread(int idx, const std::string& url);

    /* 标定辅助 */
    void collectCalibrationImages(cv::VideoCapture& cap,
        std::vector<cv::Mat>& frames,
        std::vector<std::vector<std::vector<cv::Point2f>>>& all_corners,
        std::vector<std::vector<int>>& all_ids,
        cv::Ptr<cv::aruco::CharucoBoard>& board,
        int max_samples = 15);
    void saveCalibration(const std::string& path,
        const cv::Mat& K,const cv::Mat& D,const cv::Mat& rvec,const cv::Mat& tvec);

    /* 数据成员 ----------------------------------------------------*/
    std::vector<std::string>               sources_;
    std::vector<std::thread>               threads_;
    FrameCallback                          cb_;
    bool                                   running_ = false;
    mutable std::mutex                     cap_mtx_;

    /* 单机标定结果 */
    std::vector<cv::Mat> camera_matrix_list_, dist_coeffs_list_;
    std::vector<cv::Mat> rvec_list_, tvec_list_;
    int                  default_cam_ = -1;

    /* Rig 外参（R_rig_cam , t_rig_cam）*/
    std::vector<cv::Mat> rig_R_raw_;  // world -> cam
    std::vector<cv::Mat> rig_t_raw_;
    cv::Mat rig_R_, rig_t_;
    // 张正友单机标定外参（R_cam2world, t_cam2world）
    cv::Mat rotation_, translation_;
    bool has_extrinsics_ = false;

};

}  // namespace charuco
#endif
