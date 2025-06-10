#define Status XStatus
#include <X11/Xlib.h>
#undef Status
//用宏重定义避免命名冲突
#include "video_stream.h"
#include "ui_display_thread.h" 
#include "ros2_time_sync.h"
#include "config_loader.h"
#include "yolov5_trt_detector.h"
#include "litehrnet_pose_trt.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <thread>
#include <memory>

int main(int argc, char** argv) {
    XInitThreads();//初始化 X11 多线程支持，保证图形界面线程安全
    cv::startWindowThread(); 
    rclcpp::init(argc, argv);//初始化 ROS2 客户端库

    config::PoseConfig config(
        "/home/ljy/project/poseDetection/config/clean_classes.yaml",
        "/home/ljy/project/poseDetection/config/coco_keypoints.yaml"
    );

    auto detector = std::make_shared<detectPerson::YOLOv5TRTDetector>();
    detector->initConfig("/home/ljy/project/poseDetection/models/engine/yolov5s.engine", 0.5f, 0.5f);

    auto pose_estimator = std::make_shared<pose::LiteHRNetTRT>(
        "/home/ljy/project/poseDetection/models/litehrnet18/litehrnet18.engine"
    );
    auto ros_node = std::make_shared<rclcpp::Node>("video_node");

    std::vector<std::shared_ptr<video::VideoStream>> video_streams;
    video_streams.emplace_back(std::make_shared<video::VideoStream>("rtsp://admin:123456@192.168.31.160:554/Streaming/Channels/101", "Cam 1",
        "/home/ljy/project/poseDetection/config/camera_gp150-160.yaml", detector, pose_estimator, config.getSkeleton(), ros_node));
    video_streams.emplace_back(std::make_shared<video::VideoStream>("rtsp://admin:123456@192.168.31.161:554/Streaming/Channels/101", "Cam 2",
        "/home/ljy/project/poseDetection/config/camera_gp150-161.yaml", detector, pose_estimator, config.getSkeleton(), ros_node));
    video_streams.emplace_back(std::make_shared<video::VideoStream>("rtsp://admin:123456@192.168.31.162:554/Streaming/Channels/101", "Cam 3",
        "/home/ljy/project/poseDetection/config/camera_gp150-162.yaml", detector, pose_estimator, config.getSkeleton(), ros_node));

    // 启动所有流
    for (auto& stream : video_streams) stream->start();

    // 启动集中式 UI 渲染线程
    cv::Size frame_size(640, 480);  // 举个例子
    std::thread ui_thread(ui::UiThreadFunc, std::ref(video_streams), frame_size);


    // 启动时间同步
    sync::TimeSyncNode sync;
    sync.start();

    // 主线程等待
    while (rclcpp::ok()) std::this_thread::sleep_for(std::chrono::seconds(1));

    // 停止所有流
    for (auto& stream : video_streams) stream->stop();
    sync.stop();
    rclcpp::shutdown();
    return 0;
}
