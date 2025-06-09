#define Status XStatus
#include <X11/Xlib.h>
#undef Status

#include "video_stream.h"
#include "ros2_time_sync.h"
#include "config_loader.h"
#include "yolov5_trt_demo.h"
#include "litehrnet_pose_trt.h"

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <thread>
#include <memory>

int main(int argc, char** argv) {
    XInitThreads();
    cv::startWindowThread(); 
    rclcpp::init(argc, argv);

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
    video::VideoStream cam1("rtsp://admin:123456@192.168.31.160:554/Streaming/Channels/101", "Cam 1",
                            "/home/ljy/project/poseDetection/config/camera_gp150-160.yaml",
                            detector, pose_estimator, config.getSkeleton(),ros_node);

    video::VideoStream cam2("rtsp://admin:123456@192.168.31.161:554/Streaming/Channels/101", "Cam 2",
                            "/home/ljy/project/poseDetection/config/camera_gp150-161.yaml",
                            detector, pose_estimator, config.getSkeleton(),ros_node);

    video::VideoStream cam3("rtsp://admin:123456@192.168.31.162:554/Streaming/Channels/101", "Cam 3",
                            "/home/ljy/project/poseDetection/config/camera_gp150-162.yaml",
                            detector, pose_estimator, config.getSkeleton(),ros_node);

    cam1.start();
    cam2.start();
    cam3.start();

    sync::TimeSyncNode sync;
    sync.start();

    while (rclcpp::ok()) std::this_thread::sleep_for(std::chrono::seconds(1));

    cam1.stop();
    cam2.stop();
    cam3.stop();
    sync.stop();
    rclcpp::shutdown();
    return 0;
}
