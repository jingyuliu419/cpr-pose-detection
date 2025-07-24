// =========================  main.cpp  =========================
#define Status XStatus
#include <X11/Xlib.h>
#undef Status
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

#include "yolov5_trt_detector.h"
#include "video_stream.h"
#include "ui_display_thread.h"
#include "ros2_time_sync.h"
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
    "  Multi‑Cam Pose / Depth Demo – 3 × USB + TensorRT   \n"
    "=====================================================\n";
}

int main(int argc,char** argv)
{
    /* ---------- 基础初始化 ---------- */
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS","rtsp_transport;tcp",1);
    XInitThreads();
    cv::startWindowThread();
    rclcpp::init(argc,argv);
    printBanner();
    auto node = std::make_shared<rclcpp::Node>("multi_cam_node");

    /* ---------- 相机 → 设备号 ---------- */
    std::unordered_map<std::string,int> dev {
        {"cam0",0},{"cam1",4},{"cam2",6}
    };
    std::vector<CamInfo> cams = {
        {dev["cam0"],"/home/ljy/project/poseDetection/config/camera_gp01.yml","Cam‑0"},
        {dev["cam1"],"/home/ljy/project/poseDetection/config/camera_gp23.yml","Cam‑1"},
        {dev["cam2"],"/home/ljy/project/poseDetection/config/camera_gp67.yml","Cam‑2"}
    };

    /* ---------- Detector / Pose / Triangulator ---------- */
    auto detector = std::make_shared<detectPerson::YOLOv5TRTDetector>();
    detector->initConfig("/home/ljy/project/poseDetection/models/engine/yolov5s.engine",0.6f,0.6f);
    auto pose     = std::make_shared<posetiny::RTMPoseTRT>(
        "/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.engine");
    auto triang   = std::make_shared<Triangulator>(2);

    /* ---------- CameraManagers（K,D） ---------- */
    std::vector<std::shared_ptr<charuco::CameraManager>> mgrs;
    for(auto& c:cams){
        auto mgr = std::make_shared<charuco::CameraManager>(
                       std::vector<std::string>{std::to_string(c.device_index)});
        calib::CameraCalibrator calib(c.calib_yaml);
        mgr->setCameraMatrix(calib.cameraMatrix());
        mgr->setDistCoeffs (calib.distCoeffs());
        mgrs.push_back(mgr);
    }

    /* ---------- VideoStreams ---------- */
    std::vector<std::shared_ptr<video::VideoStream>> streams;
    for(size_t i=0;i<cams.size();++i){
        auto s = std::make_shared<video::VideoStream>(
                     cams[i].device_index,"",cams[i].window,
                     cams[i].calib_yaml,detector,pose,mgrs[i],triang);
        s->setNode(node);
        s->start();
        streams.push_back(s);
    }

    /* ---------- Rig 外参处理 ---------- */
    const std::string rig_yaml="/home/ljy/project/poseDetection/config/rig_extrinsics.yaml";
    bool need_rig = std::any_of(mgrs.begin(),mgrs.end(),
                                [](auto&m){return !m->hasRigExtrinsics();});

    if(need_rig){
        auto board = cv::aruco::CharucoBoard::create(
            6,6,0.03f,0.022f,
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100));

        charuco::MultiCamCalibrator calib(mgrs,board);
        std::cout<<"[RigCalib] Capturing 40 sync frames…\n";
        for(int k=0;k<40;){
            std::vector<cv::Mat> sync;
            bool ok=true;
            for(auto&s:streams){
                cv::Mat f=s->lastFrame().clone();
                if(f.empty()){ ok=false;break; }
                sync.emplace_back(std::move(f));
            }
            if(!ok){ std::this_thread::sleep_for(std::chrono::milliseconds(50));continue;}
            calib.pushSyncFrames(sync); ++k;
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
        std::cout<<"[RigCalib] Solving…\n";
        calib.solveAndSave(rig_yaml);      // 写入 YAML
        std::cout<<"[RigCalib] Done.\n";
    }

    /* ★ 始终重新加载 YAML，写入 R/t ★ */
    for(auto&m:mgrs) m->loadRigExtrinsicsFromYaml(rig_yaml);

    /* ---------- 自检 ---------- */
    for(size_t i=0;i<mgrs.size();++i){
        auto&m=mgrs[i];
        std::cout<<"cam "<<cams[i].device_index
                 <<" | K:"<<!m->K().empty()
                 <<" R:"<<!m->rotationMatrix().empty()
                 <<" t:"<<!m->translationVector().empty()
                 <<std::endl;
    }

    /* ---------- UI 线程 ---------- */
    cv::Size win_size(1280,720);
    std::thread ui_thr([&](){ ui::UiThreadFunc(streams,win_size); });

    /* ---------- ROS2 时钟同步 ---------- */
    timesync::TimeSyncNode ts; ts.start();

    /* ---------- 主循环 ---------- */
    while(rclcpp::ok()) std::this_thread::sleep_for(std::chrono::seconds(1));

    /* ---------- 退出清理 ---------- */
    for(auto&s:streams) s->stop();
    ts.stop();
    rclcpp::shutdown();
    return 0;
}
