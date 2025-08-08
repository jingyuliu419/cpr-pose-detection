#include "video_stream.h"
#include "nv12_cuda.cuh"          // 自定义 CUDA kernel
#include <libswscale/swscale.h>
#include <opencv2/cudaarithm.hpp>
#include <fstream>
#include "ui_display_thread.h" 
#include "rtmpose_trt.h"
#include "charuco_camera_manager.h"
#include <unistd.h>
#include <sys/syscall.h>
#include "ui_display_thread.h"
#include "yolov5_trt_detector.h"
#include "EMAFilter2D.h"

using namespace video;

/* ---------- ctor / dtor ---------- */
using namespace video;
VideoStream::VideoStream(int cam_id,
                         const std::string& url,
                         const std::string& win,
                         const std::string& calib_yaml,
                         std::shared_ptr<detectPerson::YOLOv5TRTDetector> det,
                         std::shared_ptr<posetiny::RTMPoseTRT> pose,
                         std::shared_ptr<charuco::CameraManager> cam_mgr,
                         std::shared_ptr<Triangulator> triangulator)
    : url_(url),
      window_name_(win),
      detector_(std::move(det)),
      pose_model_(std::move(pose)),
      camera_mgr_(std::move(cam_mgr)),
      triangulator_(std::move(triangulator)),
      cam_id_(cam_id)
{
    use_usb_camera_ = url_.empty();             // ✅ 自动判断是否为 USB 摄像头
    device_id_ = cam_id_;                       // ✅ 保存设备编号

    calibrator_   = std::make_unique<calib::CameraCalibrator>(calib_yaml);
    detector_ctx_ = detector_->createContext();
    pose_ctx_     = pose_model_->createContext();
}



VideoStream::~VideoStream() { stop(); }

/* ---------- life cycle ---------- */
void VideoStream::start() {
    if (running_) return;
    running_ = true;
    // 启动 camera manager 的采集线程（可忽略回调）
    camera_mgr_->start([](const cv::Mat&, int){});
    // VideoStream->enableWorldCoord(true);
    capture_thread_ = std::thread(&VideoStream::captureLoop, this);
    infer_thread_   = std::thread(&VideoStream::inferenceLoop, this);
}

void VideoStream::stop() {
    running_ = false;
    cap_cv_.notify_all();
    disp_cv_.notify_all();
    if (capture_thread_.joinable()) capture_thread_.join();
    if (infer_thread_.joinable())   infer_thread_.join();
}

/* ---------- capture loop ---------- */
#include "ros2_time_sync.h"
#include "std_msgs/msg/int64.hpp"  // 添加头文件引用
#include <rclcpp/rclcpp.hpp>

// 订阅同步时间戳的消息
void VideoStream::captureLoop() {
    use_usb_camera_ = url_.empty();

    if (use_usb_camera_) {
        // ===== USB 相机采集逻辑 =====
        cv::VideoCapture cap(cam_id_, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 60);

        // 禁用自动曝光、对焦等
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
        cap.set(cv::CAP_PROP_FOCUS, 0);
        cap.set(cv::CAP_PROP_AUTO_WB, 0);

        if (!cap.isOpened()) {
            std::cerr << "[Error] Failed to open USB camera with index = " << cam_id_ << "\n";
            running_ = false;
            return;
        }

        std::cout << ">>> Using USB Camera (index=" << cam_id_ << ") <<<\n";

        try {
            while (running_) {
                cv::Mat bgr;
                if (!cap.read(bgr) || bgr.empty()) {
                    std::cerr << "[Warn] USB camera frame empty or read failed\n";
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                // 获取当前系统时间作为采集时间戳（STEADY_CLOCK 推荐用于同步）
                rclcpp::Time stamp = rclcpp::Clock(RCL_STEADY_TIME).now();

                {
                    std::lock_guard<std::mutex> lock(last_mtx_);
                    last_frame_ = bgr.clone();  // 更新最后一帧
                }

                // 入队图像和时间戳
                std::unique_lock<std::mutex> lock(cap_mutex_);
                if (capture_queue_.size() >= 1) capture_queue_.pop();
                capture_queue_.emplace(bgr.clone(), stamp);  // 👈 带上准确时间
                cap_cv_.notify_one();
            }
        } catch (const std::exception& e) {
            std::cerr << "[Fatal] Exception in USB captureLoop(): " << e.what() << std::endl;
        }

        cap.release();
        return;
    }
}


void VideoStream::writeProjected3DToFile(const std::string& filename, const rclcpp::Time& stamp, const cv::Point3f& p3d) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(3)
             << stamp.seconds() << ","
             << p3d.x << "," << p3d.y << "," << p3d.z << "\n";
    }
}


#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sys/syscall.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include <algorithm>
using FrameStamp = std::pair<cv::Mat, rclcpp::Time>;

constexpr std::array<std::pair<int, int>, 12> kEdges {{
    {5,7}, {6,8}, {7,9}, {8,10}
}};
static const float KPT_TH = 0.35f; // Keypoint threshold
static const int AXIS_THICK = 3; // Axis thickness for drawing
static const cv::Scalar X_COLOR( 0, 0, 255);
static const cv::Scalar Y_COLOR( 0, 255, 0);
static const cv::Scalar Z_COLOR(255, 0, 0);
const std::string pose_log_path_ = "/home/ljy/project/poseDetection/build/triangulated_pose_log.csv";

void VideoStream::inferenceLoop() {
    constexpr int MAX_TIME_DIFF_MS = 30;  // Time threshold in milliseconds

    while (running_) {
        try {
            // Get the next frame
            std::unique_lock<std::mutex> lk(cap_mutex_);
            cap_cv_.wait(lk, [&]{ return !capture_queue_.empty() || !running_; });
            if (!running_) break;
            FrameStamp fs = std::move(capture_queue_.front());
            capture_queue_.pop();
            lk.unlock();
            
            cv::Mat frame = std::move(fs.first);
            rclcpp::Time stamp = fs.second;
            if (frame.empty()) continue;
            std::vector<detectPerson::DetectResult> dets;
            auto start = std::chrono::high_resolution_clock::now();
            detector_->detect(frame, dets, *detector_ctx_);

            for (const auto& d : dets) {
                if (d.classId != 0) continue;
                const cv::Rect box = d.box & cv::Rect(0, 0, frame.cols, frame.rows);
                if (box.empty()) continue;

                // Extract region of interest (ROI) for person
                cv::Mat roi = frame(box).clone();
                if (roi.empty()) continue;

                // Get keypoints for the person
                std::vector<cv::Point2f> kpts;
                std::vector<float> confs;
                pose_model_->infer(roi, kpts, confs, *pose_ctx_);

                if (kpts.size() != confs.size() || kpts.empty()) continue;

                // Create array for global keypoints
                std::array<cv::Point2f, 17> gpts;
                gpts.fill({-1.f, -1.f});
                for (size_t i = 0; i < std::min<size_t>(17, kpts.size()); ++i) {
                    if (confs[i] >= KPT_TH) {
                        cv::Point2f global_pt = kpts[i] + cv::Point2f(box.x, box.y);
                        gpts[i] = ema_pool[i].update(global_pt);  // Apply EMA filtering
                    }
                }

                // Connect keypoints with lines to form the skeleton
                for (auto [u, v] : kEdges) {
                    if (gpts[u].x >= 0 && gpts[v].x >= 0)
                        cv::line(frame, gpts[u], gpts[v], {0, 0, 255}, 5);
                }

                // Get wrist keypoint for triangulation
                constexpr int WRIST = 9;
                const auto& wpt = gpts[WRIST];

                if (use_world_coord_ && wpt.x >= 0 && wpt.y >= 0) {
                    const auto& K = camera_mgr_->K();
                    const auto& R = camera_mgr_->rotationMatrix();
                    const auto& t = camera_mgr_->translationVector();

                    if (K.empty() || R.empty() || t.empty()) {
                        std::cerr << "[Error] Empty matrix in projection: cam_id = " << cam_id_ << "\n";
                        continue;
                    }

                    // 从2D图像坐标恢复相机坐标系下3D射线，并转换到世界坐标
                    cv::Mat pt2d = (cv::Mat_<double>(3, 1) << wpt.x, wpt.y, 1.0);
                    cv::Mat pt_cam = K.inv() * pt2d;  // 相机坐标系下方向向量
                    pt_cam = pt_cam / cv::norm(pt_cam);  // 归一化

                    // 将方向向量变换到世界坐标系
                    cv::Mat dir_world = R.t() * pt_cam;     // 世界系方向
                    cv::Mat cam_center = -R.t() * t;        // 世界坐标下相机位置


                    // 使用单位射线 + 相机中心作为估计点（例如向前延伸1米）
                    cv::Mat pt_world = cam_center + dir_world;

                    // 构造世界坐标下点
                    cv::Point3f p3d(pt_world.at<double>(0), pt_world.at<double>(1), pt_world.at<double>(2));

                    // ⏺️ 更新上一帧记录（每个相机独立）
                    static std::map<int, cv::Point3f> last_p3d_map;
                    last_p3d_map[cam_id_] = p3d;

                    // 欧式距离
                    float move_dist = std::sqrt(p3d.x * p3d.x + p3d.y * p3d.y + p3d.z * p3d.z);

                    // 单位方向向量 & 夹角余弦
                    cv::Mat ray_dir_world = dir_world / cv::norm(dir_world);
                    cv::Mat world_z = (cv::Mat_<double>(3, 1) << 0, 0, 1);
                    double cos_theta = ray_dir_world.dot(world_z);
                    cos_theta = std::clamp(cos_theta, -1.0, 1.0);  // 防止 acos 出现 NaN

                    // 投影压深估计
                    double est_depth = (0.423600-p3d.x)  + 11.5230  * cos_theta+( -1.9806); // 或者其他非线性因子
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::ifstream tchk("/home/ljy/project/poseDetection/build/full_pipeline_log.csv");
                    bool t_exists = tchk.good();
                    tchk.close();
                    std::ofstream tlog("/home/ljy/project/poseDetection/build/full_pipeline_log.csv", std::ios::app);
                    if (tlog.is_open()) {
                        if (!t_exists) tlog << "total_pipeline_ms\n";
                        tlog << duration << "\n";
                    }
                    // 可视化
                    cv::Point text_pos = wpt + cv::Point2f(10, -10);
                    cv::putText(frame,
                        cv::format("EstDepth: %.2f cm | Angle: %.1f°", est_depth * 100.0, std::acos(cos_theta) * 180.0 / CV_PI),
                        text_pos, cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 255, 255}, 3);

                    // 保存
                    std::string save_path = "/home/ljy/project/poseDetection/build/cam" + std::to_string(cam_id_) + "_projected_log.csv";
                    std::ofstream fout(save_path, std::ios::app);
                    if (fout.is_open()) {
                        using namespace std::chrono;
                        auto now = high_resolution_clock::now();
                        int64_t timestamp_ms = duration_cast<milliseconds>(now.time_since_epoch()).count();
                        float conf_score = (WRIST < confs.size()) ? confs[WRIST] : -1.0f;

                        fout << std::fixed << std::setprecision(6)
                             << timestamp_ms << "," << p3d.x << "," << p3d.y << "," << p3d.z << ","
                             << move_dist << "," << est_depth << "," << cos_theta << "," << conf_score << "\n";

                    }
                }

            }
            // Draw axes and update the display
            camera_mgr_->drawOriginAxesUnified(frame, cam_id_);
            std::lock_guard<std::mutex> disp_lock(disp_mutex_);
            while (!display_queue_.empty()) display_queue_.pop();
            display_queue_.push(frame.clone());
            ui::notifyUI();

        } catch (const std::exception &e) {
            std::cerr << "[Fatal] inferenceLoop: " << e.what() << '\n';
        }
    }
}

void VideoStream::run_video_inference(const std::string& engine_path) {
    auto pose_model = std::make_shared<posetiny::RTMPoseTRT>(engine_path);
    video::VideoStream stream;
    stream.setPoseModel(pose_model);
    stream.inferenceLoop();
}
void VideoStream::setPoseModel(const std::shared_ptr<posetiny::RTMPoseTRT>& model)
{
    pose_model_ = model;
    pose_ctx_.reset();                     // 先清理旧 ctx

    if (pose_model_) {
        pose_ctx_ = pose_model_->createContext();
        std::cout << "[VideoStream] Pose model set. "
                  << "Input="  << pose_model_->inputWidth() << "x" << pose_model_->inputHeight()
                  << ", K="    << pose_model_->numKeypoints() << std::endl;
    } else {
        std::cerr << "[VideoStream] Warning: pose_model is nullptr\n";
    }
}


