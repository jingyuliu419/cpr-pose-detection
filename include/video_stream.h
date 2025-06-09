// include/video_stream.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <thread> 
#include <opencv2/core.hpp>
#include "camera_calibrator.h"
#include "yolov5_trt_demo.h"
#include "litehrnet_pose_trt.h"
#include <mutex>
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
                const std::vector<std::pair<int, int>>& skeleton);

    ~VideoStream();
    void start();
    void stop();

private:
    void run();

    std::string url_;
    std::string window_name_;
    std::unique_ptr<calib::CameraCalibrator> calibrator_;
    std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector_;
    std::shared_ptr<pose::LiteHRNetTRT> pose_;
    std::vector<std::pair<int, int>> skeleton_;

    bool running_ = false;
    std::thread worker_;
    
};

}  // namespace video
