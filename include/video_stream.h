#pragma once

// ---------- C++ 头文件 ----------
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "camera_calibrator.h"
#include "litehrnet_pose_trt.h"
#include "yolov5_trt_detector.h"
#include "rtmpose_trt.h"
#include "charuco_camera_manager.h"
#include "Triangulator.h"
#include "ui_display_thread.h"
#include "ros2_time_sync.h"
// ---------- 纯 C 头文件（FFmpeg）----------
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}  // extern "C"


namespace video {

class VideoStream {
public:
    VideoStream(int cam_id,
                const std::string& url,
                const std::string& win,
                const std::string& calib_yaml,
                std::shared_ptr<detectPerson::YOLOv5TRTDetector> det,
                std::shared_ptr<posetiny::RTMPoseTRT> pose,
                std::shared_ptr<charuco::CameraManager> cam_mgr,
                std::shared_ptr<Triangulator> triangulator);

    /* 若想保留旧接口，也可提供 setter */
    void setCameraManager(std::shared_ptr<charuco::CameraManager> mgr) { camera_mgr_ = std::move(mgr); }
    VideoStream() = default;
    ~VideoStream();

    void start();
    void stop();
    const std::string& name() const { return window_name_; }
    void enableWorldCoord(bool enable) { use_world_coord_ = enable; }

    /* 供 UI 线程访问 */
    std::mutex imshow_mutex_;
    std::queue<cv::Mat> display_queue_;
    void run_video_inference(const std::string& engine_path);
    void setPoseModel(const std::shared_ptr<posetiny::RTMPoseTRT>& model);

    cv::Mat lastFrame() const {
        std::lock_guard<std::mutex> lk(last_mtx_);
        return last_frame_.clone();
    }
    void setNode(std::shared_ptr<rclcpp::Node> node) {
        ros_node_ = std::move(node);
    }


private:
    void captureLoop();
    void decodeWithCudaLoophard();
    void inferenceLoop();

    /* ---       --- */
    std::shared_ptr<charuco::CameraManager> camera_mgr_;

    /* --- const / shared --- */
    const std::string url_, window_name_;
    std::unique_ptr<calib::CameraCalibrator> calibrator_;
    std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector_;
    std::shared_ptr<posetiny::RTMPoseTRT>             pose_model_;
    std::shared_ptr<rclcpp::Node> ros_node_;
    const std::vector<std::pair<int,int>> skeleton_;

    /* --- per-stream context --- */
    std::shared_ptr<detectPerson::YOLOv5TRTDetector::Context> detector_ctx_;
    std::shared_ptr<posetiny::RTMPoseTRT::Context>              pose_ctx_;

    /* --- thread resources --- */
    std::thread capture_thread_, infer_thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex last_mtx_;
    cv::Mat            last_frame_;

    /* --- queue: capture → infer → UI --- */
    std::mutex cap_mutex_, disp_mutex_;
    std::condition_variable cap_cv_, disp_cv_;
    // std::queue<cv::Mat> capture_queue_;

    //world
    bool use_world_coord_ = true;
    std::shared_ptr<Triangulator> triangulator_;
    int cam_id_;
    using FrameStamp = std::pair<cv::Mat, rclcpp::Time>;
    std::queue<FrameStamp> capture_queue_;



    /* --- FFmpeg --- */
    AVFormatContext* fmt_ctx_{nullptr};
    AVCodecContext*  codec_ctx_{nullptr};
    AVFrame*         av_frame_{nullptr};
    AVPacket*        av_packet_{nullptr};
};

} // namespace video
