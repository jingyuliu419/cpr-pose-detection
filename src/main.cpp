#define Status XStatus
#include <X11/Xlib.h>
#undef Status

#include "video_stream.h"
#include "ui_display_thread.h"
#include "ros2_time_sync.h"
#include "yolov5_trt_detector.h"
#include "litehrnet_pose_trt.h"
#include "rtmpose_trt.h"
#include "charuco_camera_manager.h"
#include "multi_cam_calibrator.h"
#include "camera_calibrator.h"  // 

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <algorithm>

struct CamInfo {
    std::string rtsp, calib_yaml, window;
};

static void printBanner() {
    std::cout <<
    "=====================================================\n"
    "  Multi-Cam Pose / Depth Demo – 3 × RTSP + TensorRT  \n"
    "  OpenCV-4.6 | 单机 + Rig 外参                       \n"
    "=====================================================\n";
}

int main(int argc, char** argv) {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
    XInitThreads();  cv::startWindowThread();
    rclcpp::init(argc, argv);      printBanner();

    std::vector<CamInfo> cams = {
        {"rtsp://admin:123456@192.168.31.160:554/Streaming/Channels/101?tcp",
         "/home/ljy/project/poseDetection/config/camera_gp150-160.yaml", "Cam-1"},
        {"rtsp://admin:123456@192.168.31.161:554/Streaming/Channels/101?tcp",
         "/home/ljy/project/poseDetection/config/camera_gp150-161.yaml", "Cam-2"},
        {"rtsp://admin:123456@192.168.31.162:554/Streaming/Channels/101?tcp",
         "/home/ljy/project/poseDetection/config/camera_gp150-162.yaml", "Cam-3"}
    };

    auto detector = std::make_shared<detectPerson::YOLOv5TRTDetector>();
    detector->initConfig("/home/ljy/project/poseDetection/models/engine/yolov5s.engine", 0.5f, 0.5f);
    auto pose_estimator = std::make_shared<posetiny::RTMPoseTRT>(
        "/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.engine");

    std::vector<std::shared_ptr<charuco::CameraManager>> cam_mgrs;
    
    for (auto const& c : cams) {
        auto mgr = std::make_shared<charuco::CameraManager>(std::vector<std::string>{c.rtsp});

        // ✅ 使用张正友标定好的 YAML 加载内参
        auto calibrator = calib::CameraCalibrator(c.calib_yaml);
        
        mgr->setCameraMatrix(calibrator.cameraMatrix());
        std::cout<<"________________________"<<std::endl;
        mgr->setDistCoeffs(calibrator.distCoeffs());
        mgr->setExtrinsics(calibrator.rotationMatrix(), calibrator.translationVector());

        cam_mgrs.push_back(mgr);
        
        std::cout << "K type: " << calibrator.cameraMatrix().type() << std::endl;
        std::cout << "R type: " << calibrator.rotationMatrix().type() << std::endl;
        std::cout << "T type: " << calibrator.translationVector().type() << std::endl;

    }
    
    std::vector<std::shared_ptr<video::VideoStream>> streams;
    for (size_t i = 0; i < cams.size(); ++i) {
        streams.emplace_back(std::make_shared<video::VideoStream>(
            cams[i].rtsp, cams[i].window, cams[i].calib_yaml,
            detector, pose_estimator, cam_mgrs[i]));
        streams.back()->start();
    }

    const std::string rig_yaml = "/home/ljy/project/poseDetection/config/rig_extrinsics.yaml";
    bool need_rig = std::any_of(cam_mgrs.begin(), cam_mgrs.end(),
                                [](auto& m) { return !m->hasRigExtrinsics(); });

    if (need_rig) {
        auto board = cv::aruco::CharucoBoard::create(
            6, 6, 0.03f, 0.022f,
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100));

        charuco::MultiCamCalibrator calib(cam_mgrs, board);
        std::cout << "[RigCalib]  采集 40 组同步帧…\n";

        for (int k = 0; k < 40;) {
            std::vector<cv::Mat> sync;
            bool got_all = true;
            for (auto& s : streams) {
                cv::Mat f = s->lastFrame().clone();
                if (f.empty()) { got_all = false; break; }
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
        auto exts = calib.solveAndSave(rig_yaml);
        std::cout << "[RigCalib]  end…\n";
        for (size_t i = 0; i < cam_mgrs.size(); ++i)
            cam_mgrs[i]->setRigExtrinsics(exts[i].R, exts[i].t);
    } else {
        cv::FileStorage fs(rig_yaml, cv::FileStorage::READ);
        for (size_t i = 0; i < cam_mgrs.size(); ++i) {
            cv::Mat R, t; fs["R" + std::to_string(i)] >> R;
                        fs["t" + std::to_string(i)] >> t;
            cam_mgrs[i]->setRigExtrinsics(R, t);
        }
    }

    cv::Size frame_size(640, 480);
    std::thread ui_thread(ui::UiThreadFunc, std::ref(streams), frame_size);
    sync::TimeSyncNode sync; sync.start();

    while (rclcpp::ok()) std::this_thread::sleep_for(std::chrono::seconds(1));

    for (auto& s : streams) s->stop();
    sync.stop();
    rclcpp::shutdown();
    return 0;
}
