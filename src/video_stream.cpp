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
    bool use_gpu_ = true;
    use_usb_camera_ = url_.empty();

    // 创建 ROS2 节点并订阅时间同步消息
    auto node = std::make_shared<rclcpp::Node>("capture_node");
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr time_sync_sub = node->create_subscription<std_msgs::msg::Int64>(
        "/time_sync_topic", 10, [this](const std_msgs::msg::Int64::SharedPtr msg) {
            // 处理从 TimeSyncNode 发布过来的时间戳
            synchronized_timestamp_ = rclcpp::Time(msg->data);  // 将时间戳存储到 synchronized_timestamp_
        });

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread ros_thread([&executor]() { executor.spin(); });

    if (use_usb_camera_) {
        // ===== USB 相机采集逻辑 =====
        cv::VideoCapture cap(cam_id_, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);   // 降低分辨率
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);  // 降低分辨率
        cap.set(cv::CAP_PROP_FPS, 30);  // 限制帧率为30 FPS

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

                {
                    std::lock_guard<std::mutex> lock(last_mtx_);
                    last_frame_ = bgr.clone();  // 仅克隆一次
                }

                // 使用同步的时间戳
                rclcpp::Time stamp = synchronized_timestamp_;  // 使用同步的时间戳
                if (stamp == rclcpp::Time(0)) {
                    // std::cerr << "[Warn] Time stamp not synchronized yet, skipping frame\n";
                    continue;  // 如果时间戳还没有同步，跳过此帧
                }

                std::unique_lock<std::mutex> lock(cap_mutex_);
                if (capture_queue_.size() >= 2) capture_queue_.pop();  // 保持队列大小为2
                capture_queue_.emplace(bgr.clone(), stamp);
                cap_cv_.notify_one();
            }
        } catch (const std::exception& e) {
            std::cerr << "[Fatal] Exception in USB captureLoop(): " << e.what() << std::endl;
        }

        cap.release();
        return;
    }
}


#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sys/syscall.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>

using FrameStamp = std::pair<cv::Mat, rclcpp::Time>;

constexpr std::array<std::pair<int, int>, 12> kEdges {{
    {5,7}, {6,8}, {7,9}, {8,10}
}};
static const float KPT_TH = 0.10f; // Keypoint threshold
static const int AXIS_THICK = 3; // Axis thickness for drawing
static const cv::Scalar X_COLOR( 0, 0, 255);
static const cv::Scalar Y_COLOR( 0, 255, 0);
static const cv::Scalar Z_COLOR(255, 0, 0);
const std::string pose_log_path_ = "/home/ljy/project/poseDetection/build/triangulated_pose_log.csv";

void VideoStream::inferenceLoop() {
    constexpr int MAX_TIME_DIFF_MS = 5;  // Time threshold in milliseconds

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
                constexpr int WRIST = 10;
                const auto& wpt = gpts[WRIST];

                if (use_world_coord_ && wpt.x >= 0 && wpt.y >= 0 && triangulator_) {
                    const auto& K = camera_mgr_->K();
                    const auto& R = camera_mgr_->rotationMatrix();
                    const auto& t = camera_mgr_->translationVector();
                    
                    // Ensure matrices are valid
                    if (K.empty() || R.empty() || t.empty()) {
                        std::cerr << "[Error] Empty matrix in triangulation: "
                                  << "cam_id = " << cam_id_ << "\n";
                        continue;
                    }

                    // Check if the time difference exceeds the threshold
                    std::lock_guard<std::mutex> lock(mtx_);
                    if (!window_.empty() && std::abs(stamp.nanoseconds() - window_.back().stamp.nanoseconds()) > 30 * 1000000) {
                        window_.clear();  // Clear window if time difference exceeds threshold
                    }
                    
                    // Push 2D keypoint to triangulator
                    triangulator_->push2DKeypoint(cam_id_, stamp, wpt, K, R, t);

                    // Perform triangulation if ready
                    if (auto p3d = triangulator_->triangulateIfReady()) {
                        cv::Point text_pos = wpt + cv::Point2f(10, -10);
                        cv::putText(frame,
                            cv::format("[%.2f %.2f %.2f]", p3d->x, p3d->y, p3d->z),
                            text_pos,
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, {0,0,255}, 8);
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


