// include/video_stream.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include "camera_calibrator.h"
#include "yolov5_trt_demo.h"
#include "litehrnet_pose_trt.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
}
namespace video {
 extern std::mutex pose_mutex_;
 extern std::mutex imshow_mutex_;

class VideoStream {
public:
    VideoStream(const std::string& url,
                const std::string& window_name,
                const std::string& calib_path,
                std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector,
                std::shared_ptr<pose::LiteHRNetTRT> pose,
                const std::vector<std::pair<int, int>>& skeleton,
                std::shared_ptr<rclcpp::Node> ros_node);
                std::shared_ptr<rclcpp::Node> ros_node;

    ~VideoStream();
    void start();
    void stop();

private:
    // void run();

    std::string url_;
    std::string window_name_;
    std::unique_ptr<calib::CameraCalibrator> calibrator_;
    std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector_;
    std::shared_ptr<pose::LiteHRNetTRT> pose_;
    std::vector<std::pair<int, int>> skeleton_;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    AVPacket* av_packet_ = nullptr;


    // bool running_ = false;
    // std::thread worker_;

    std::shared_ptr<rclcpp::Node> ros_node_;
    std::thread capture_thread_, inference_thread_, display_thread_;
    std::atomic<bool> running_ = false;

    std::queue<cv::Mat> capture_queue_;
    std::queue<cv::Mat> display_queue_;
    std::mutex cap_mutex_, disp_mutex_;
    std::condition_variable cap_cv_, disp_cv_;

    void captureLoop();
    void inferenceLoop();
    void displayLoop();

    
};

}  // namespace video
