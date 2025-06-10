#pragma once
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
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}
namespace video {

class VideoStream {
public:
    VideoStream(const std::string& url,
                const std::string& window,
                const std::string& calib_yaml,
                std::shared_ptr<detectPerson::YOLOv5TRTDetector> det,
                std::shared_ptr<pose::LiteHRNetTRT> pose,
                const std::vector<std::pair<int,int>>& skeleton,
                std::shared_ptr<rclcpp::Node> ros_node);
    ~VideoStream();

    void start();
    void stop();
    const std::string& name() const { return window_name_; }

    /* 供 UI 线程访问 */
    std::mutex imshow_mutex_;
    std::queue<cv::Mat> display_queue_;

private:
    void captureLoop();
    void inferenceLoop();

    /* --- const / shared --- */
    const std::string url_, window_name_;
    std::unique_ptr<calib::CameraCalibrator> calibrator_;
    std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector_;
    std::shared_ptr<pose::LiteHRNetTRT>             pose_model_;
    std::shared_ptr<rclcpp::Node> ros_node_;
    const std::vector<std::pair<int,int>> skeleton_;

    /* --- per-stream context --- */
    std::shared_ptr<detectPerson::YOLOv5TRTDetector::Context> detector_ctx_;
    std::shared_ptr<pose::LiteHRNetTRT::Context>              pose_ctx_;

    /* --- thread resources --- */
    std::thread capture_thread_, infer_thread_;
    std::atomic<bool> running_{false};

    /* --- queue: capture → infer → UI --- */
    std::mutex cap_mutex_, disp_mutex_;
    std::condition_variable cap_cv_, disp_cv_;
    std::queue<cv::Mat> capture_queue_;

    /* --- FFmpeg --- */
    AVFormatContext* fmt_ctx_{nullptr};
    AVCodecContext*  codec_ctx_{nullptr};
    AVFrame*         av_frame_{nullptr};
    AVPacket*        av_packet_{nullptr};
};

} // namespace video
