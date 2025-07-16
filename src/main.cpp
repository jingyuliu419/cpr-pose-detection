#define Status XStatus
#include <X11/Xlib.h>
#undef Status
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <functional>  // <- for std::ref and std::bind if needed
#include <unordered_map>  // ✅ 提供 std::unordered_map
#include <string>         // ✅ 提供 std::string

#include "yolov5_trt_detector.h"
#include "video_stream.h"
#include "ui_display_thread.h"
#include "ros2_time_sync.h"

#include "litehrnet_pose_trt.h"
#include "rtmpose_trt.h"
#include "charuco_camera_manager.h"
#include "multi_cam_calibrator.h"
#include "camera_calibrator.h"
#include "Triangulator.h"

struct CamInfo {
    int device_index;
    std::string calib_yaml;
    std::string window;
};

static void printBanner() {
    std::cout <<
    "=====================================================\n"
    "  Multi-Cam Pose / Depth Demo – 3 × USB + TensorRT   \n"
    "  OpenCV-4.6 | 单机 + Rig 外参                       \n"
    "=====================================================\n";
}

int main(int argc, char **argv) {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
    XInitThreads();
    cv::startWindowThread();
    rclcpp::init(argc, argv);
    printBanner();

    auto node = std::make_shared<rclcpp::Node>("multi_cam_node");
std::unordered_map<std::string, int> map{
    {"cam0", 1},
    {"cam1", 4},
    {"cam2", 6}
};

std::vector<CamInfo> cams = {
    {map["cam0"], "/home/ljy/project/poseDetection/config/camera_gp01.yml", "Cam-1"},
    {map["cam2"], "/home/ljy/project/poseDetection/config/camera_gp23.yml", "Cam-2"},
    {map["cam1"], "/home/ljy/project/poseDetection/config/camera_gp67.yml", "Cam-3"}};


    auto detector = std::make_shared<detectPerson::YOLOv5TRTDetector>();
    detector->initConfig("/home/ljy/project/poseDetection/models/engine/yolov5s.engine", 0.6f, 0.6f);
    auto pose_estimator = std::make_shared<posetiny::RTMPoseTRT>(
        "/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.engine");

    // const std::size_t REQUIRED_CAMS = cams.size();   // == 3
    auto triangulator = std::make_shared<Triangulator>(2);

    std::vector<std::shared_ptr<charuco::CameraManager>> cam_mgrs;
    for (auto const &c : cams) {
        std::cout<<"_______"<<c.device_index<<"_______\n";
        auto mgr = std::make_shared<charuco::CameraManager>(
            std::vector<std::string>{std::to_string(c.device_index)});

        calib::CameraCalibrator calib(c.calib_yaml);
        mgr->setCameraMatrix(calib.cameraMatrix());
        mgr->setDistCoeffs(cv::Mat());
        mgr->setExtrinsics(calib.rotationMatrix(), calib.translationVector());

        cam_mgrs.push_back(mgr);
    }

    std::vector<std::shared_ptr<video::VideoStream>> streams;
    for (size_t i = 0; i < cams.size(); ++i) {
        auto stream = std::make_shared<video::VideoStream>(
            cams[i].device_index, "", cams[i].window, cams[i].calib_yaml,
            detector, pose_estimator, cam_mgrs[i], triangulator);

        stream->setNode(node);
        stream->start();
        streams.emplace_back(stream);
    }

    const std::string rig_yaml = "/home/ljy/project/poseDetection/config/rig_extrinsics.yaml";
    bool need_rig = std::any_of(cam_mgrs.begin(), cam_mgrs.end(),
                                 [](auto &m) { return !m->hasRigExtrinsics(); });

    if (need_rig) {
        auto board = cv::aruco::CharucoBoard::create(
            6, 6, 0.03f, 0.022f,
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100));

        charuco::MultiCamCalibrator calib(cam_mgrs, board);
        std::cout << "[RigCalib]  采集 40 组同步帧…\n";

        for (int k = 0; k < 40;) {
            std::vector<cv::Mat> sync;
            bool got_all = true;
            for (auto &s : streams) {
                cv::Mat f = s->lastFrame().clone();
                if (f.empty()) {
                    got_all = false;
                    break;
                }
                sync.emplace_back(std::move(f));
            }
            if (!got_all) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            calib.pushSyncFrames(sync);
            ++k;
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }

        std::cout << "[RigCalib]  start标定，可能需要几分钟…\n";
        // 替换原先的 calib.solveAndSave(rig_yaml)
        auto exts = calib.solveAndSave(rig_yaml);
        std::vector<int> cam_ids;
        for (size_t i = 0; i < exts.size(); ++i) {
            cam_mgrs[i]->setRigExtrinsicsRaw(exts[i].R, exts[i].t, cams[i].device_index);
            cam_ids.push_back(cams[i].device_index);
        }
        charuco::CameraManager::saveAllRigExtrinsicsToYaml(rig_yaml, cam_mgrs, cam_ids); // ✅ 静态调用方式


        std::cout << "[RigCalib]  end…\n";
        for (size_t i = 0; i < cam_mgrs.size(); ++i)
            cam_mgrs[i]->setRigExtrinsics(exts[i].R, exts[i].t);
    } else {
        for (auto& mgr : cam_mgrs)
            mgr->loadRigExtrinsicsFromYaml(rig_yaml);
    }

    cv::Size frame_size(1280, 720);
    std::thread ui_thread([&]() {
        ui::UiThreadFunc(streams, frame_size);
    });

    timesync::TimeSyncNode time_sync;
    time_sync.start();

    while (rclcpp::ok()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (auto &s : streams) s->stop();
    time_sync.stop();
    rclcpp::shutdown();
    return 0;
}
