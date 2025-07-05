#pragma once
#include "charuco_camera_manager.h"
#include <opencv2/aruco/charuco.hpp>
#include <vector>
#include <string>

namespace charuco {

/** 每台相机在 Rig 世界中的外参 R_rig_cam , t_rig_cam */
struct RigExtrinsics
{
    cv::Mat R;   ///< 3×3 double
    cv::Mat t;   ///< 3×1 double
};

/**
 * 采集同步帧并求解多相机外参（兼容 OpenCV 4.5–4.6）
 *
 * master_cam = 0 作为 Rig 原点
 */
class MultiCamCalibrator
{
public:
    MultiCamCalibrator(std::vector<std::shared_ptr<CameraManager>>& cams,
                       cv::Ptr<cv::aruco::CharucoBoard> board);

    /** 把 N 台相机在同一时刻的帧按 {cam0,cam1,…} 顺序压入 */
    void pushSyncFrames(const std::vector<cv::Mat>& frames);

    /** 计算外参；rig_yaml_path 为空则只返回，不落盘 */
    std::vector<RigExtrinsics>
                solveAndSave(const std::string& rig_yaml_path = "");

private:
    struct Shot2D {
        std::vector<cv::Point2f>  imgPts;
        std::vector<int>          ids;
    };
    struct CamBuffer {
        std::vector<Shot2D> shots;
        cv::Size            imgSize = {};
    };

    bool  detectCharuco(const cv::Mat& frame,
                        Shot2D&        out);

    std::vector<std::shared_ptr<CameraManager>> cams_;
    cv::Ptr<cv::aruco::CharucoBoard>            board_;
    std::vector<CamBuffer>                      buf_;  // 同步缓冲
};

} // namespace charuco
