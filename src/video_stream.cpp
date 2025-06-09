#include<video_stream.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <rclcpp/rclcpp.hpp>
#include "camera_calibrator.h" 
#include "yolov5_trt_demo.h"
#include "litehrnet_pose_trt.h"
#include "trt_logger.h" 
#include "camera_calibrator.h"
#include "yolov5_trt_demo.h"
#include "litehrnet_pose_trt.h"
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
extern "C"{
    #include<libavformat/avformat.h>
    #include<libavcodec/avcodec.h>
    #include<libswscale/swscale.h>
}
    // video_stream.cpp

namespace video{
    std::mutex pose_mutex_;
    std::mutex imshow_mutex_;

    VideoStream::VideoStream(const std::string& url,
                            const std::string& window_name,
                            const std::string& calib_path,
                            std::shared_ptr<detectPerson::YOLOv5TRTDetector> detector,
                            std::shared_ptr<pose::LiteHRNetTRT> pose,
                            const std::vector<std::pair<int, int>>& skeleton,
                            std::shared_ptr<rclcpp::Node> ros_node)
        : url_(url), window_name_(window_name), detector_(detector), pose_(pose), skeleton_(skeleton),ros_node_(ros_node){
        calibrator_ = std::make_unique<calib::CameraCalibrator>(calib_path);
    }


    VideoStream::~VideoStream() {
        stop();
    }
void VideoStream::start() {
    running_ = true;
    capture_thread_ = std::thread(&VideoStream::captureLoop, this);
    inference_thread_ = std::thread(&VideoStream::inferenceLoop, this);
    display_thread_ = std::thread(&VideoStream::displayLoop, this);
}

void VideoStream::stop() {
    running_ = false;
    cap_cv_.notify_all();
    disp_cv_.notify_all();
    if (capture_thread_.joinable()) capture_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
    if (display_thread_.joinable()) display_thread_.join();
}

// void VideoStream::run() {


//     avformat_network_init();
//     AVFormatContext* fmt_ctx = nullptr;
//     AVDictionary* options = nullptr;
//     // 保证用的是 TCP 连接（更稳定）
//     av_dict_set(&options, "rtsp_transport", "tcp", 0);
//     av_dict_set(&options, "flush_packets", "1", 0); // 强制解码最新帧
//     av_dict_set(&options, "rtbufsize", "100000", 0); // 100KB 缓冲

//     // 关闭缓冲
//     av_dict_set(&options, "fflags", "nobuffer", 0);
//     av_dict_set(&options, "flags", "low_delay", 0);
//     av_dict_set(&options, "max_delay", "5", 0);      // 500ms
//     av_dict_set(&options, "reorder_queue_size", "0", 0);

//     avformat_open_input(&fmt_ctx, url_.c_str(), nullptr, &options);
//     avformat_find_stream_info(fmt_ctx, nullptr);

//     int index = -1;
//     for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
//         if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
//             index = i; break;
//         }
//     }

//     AVCodecParameters* codecpar = fmt_ctx->streams[index]->codecpar;
//     const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
//     AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
//     avcodec_parameters_to_context(codec_ctx, codecpar);
//     avcodec_open2(codec_ctx, codec, nullptr);

//     SwsContext* sws = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
//                                      codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24,
//                                      SWS_BILINEAR, nullptr, nullptr, nullptr);

//     AVFrame* frame = av_framecv::COLOR_YUV2BGR_NV12_alloc();
//     AVPacket* packet = av_packet_alloc();
//     cv::Mat image(codec_ctx->height, codec_ctx->width, CV_8UC3);
//     uint8_t* dest[4] = { image.data, nullptr, nullptr, nullptr };
//     int linesize[4] = { static_cast<int>(image.step), 0, 0, 0 };
//     cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
//     cv::resizeWindow(window_name_, codec_ctx->width, codec_ctx->height);

//     while (running_) {
//         AVPacket tmp;
//         bool got_packet = false;

//         while (av_read_frame(fmt_ctx, &tmp) >= 0) {
//             if (tmp.stream_index != index) {
//                 av_packet_unref(&tmp);
//                 continue;
//             }
//             av_packet_unref(packet);            // 释放旧的
//             av_packet_ref(packet, &tmp);        // 替换为新的
//             av_packet_unref(&tmp);              // 释放临时包
//             got_packet = true;

//             // 💡 加这个 break，避免一直读光缓存
//             break;
//         }

//         if (!got_packet) continue;


//         if (avcodec_send_packet(codec_ctx, packet) == 0 && avcodec_receive_frame(codec_ctx, frame) == 0) {
//             // rclcpp::Time ros_time = ros_node_->now();  // 获取当前 ROS 时间
//             sws_scale(sws, frame->data, frame->linesize, 0, codec_ctx->height, dest, linesize);
//             auto undistorted = calibrator_->undistort(image);
//             // bool drawn = false;

//             // std::vector<detectPerson::DetectResult> detections;
//             // detector_->detect(undistorted, detections);

//             // for (const auto& dr : detections) {
//             //     if (dr.classId != 0) continue;
//             //     cv::Rect person_box = dr.box & cv::Rect(0, 0, undistorted.cols, undistorted.rows);
//             //     cv::Mat roi = undistorted(person_box).clone();
//             //     std::cout << "[DEBUG] Processed frame with ROI size: " << roi.cols << "x" << roi.rows << std::endl;

//             //     if (roi.empty() || roi.cols < 10 || roi.rows < 10 || roi.cols / roi.rows > 10 || roi.rows / roi.cols > 10) {
//             //         std::cerr << "[WARNING] Skipped invalid ROI: (" << roi.cols << "x" << roi.rows << ")" << std::endl;
//             //         continue;
//             //     }
//             //     std::vector<pose::Keypoint> keypoints;
//             //     {
//             //         std::lock_guard<std::mutex> lock(pose_mutex_);
//             //         pose_->infer(roi, keypoints);
//             //     }
//             //     std::cout << "[DEBUG] keypoints detected: " << keypoints.size() << std::endl;

//             //     for (const auto& kp : keypoints) {
//             //         cv::circle(undistorted, kp.pt + cv::Point2f(person_box.x, person_box.y), 3, {0, 0, 255}, -1);
//             //     }

//             //     for (const auto& [i, j] : skeleton_) {
//             //         if (i >= keypoints.size() || j >= keypoints.size()) continue;
//             //         if (keypoints[i].score > 0.3f && keypoints[j].score > 0.3f) {
//             //             cv::line(undistorted, keypoints[i].pt + cv::Point2f(person_box.x, person_box.y),
//             //                      keypoints[j].pt + cv::Point2f(person_box.x, person_box.y),
//             //                      cv::Scalar(0, 255, 255), 2);
//             //         }
//             //     }

//             //     cv::rectangle(undistorted, person_box, cv::Scalar(0, 255, 0), 2);
//             //     drawn = true;
//             // }
//             // std::cout << "Frame Timestamp: " << ros_time.seconds() << "s" << std::endl;
//             // static double last = ros_time.seconds();
//             // double delta = ros_time.seconds() - last;
//             // last = ros_time.seconds();
//             // std::cout << "[延迟间隔] " << delta << " 秒" << std::endl;

//             static int frame_count = 0;
//             if (frame_count++ % 5 == 0) {
//                 cv::imshow(window_name_, undistorted);
//                 if (cv::waitKey(1) == 'q') break;
//             }

//         }
//             av_packet_unref(packet);
        
//     }

//     av_frame_free(&frame);
//     av_packet_free(&packet);
//     sws_freeContext(sws);
//     avcodec_free_context(&codec_ctx);
//     avformat_close_input(&fmt_ctx);
//     avformat_network_deinit();
// }
void VideoStream::captureLoop() {
    avformat_network_init();
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "max_delay", "5", 0);

    avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &options);
    avformat_find_stream_info(fmt_ctx_, nullptr);

    int video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_, codecpar);
    codec_ctx_->pix_fmt = AV_PIX_FMT_NV12;
    avcodec_open2(codec_ctx_, codec, nullptr);

    av_frame_ = av_frame_alloc();
    av_packet_ = av_packet_alloc();

    const int w = codec_ctx_->width;
    const int h = codec_ctx_->height;

    while (running_) {
        AVPacket pkt;
        bool got_packet = false;

        while (av_read_frame(fmt_ctx_, &pkt) >= 0) {
            if (pkt.stream_index == video_stream_index) {
                av_packet_unref(av_packet_);
                av_packet_ref(av_packet_, &pkt);
                av_packet_unref(&pkt);
                got_packet = true;
                break;
            }
            av_packet_unref(&pkt);
        }

        if (!got_packet) continue;

        if (avcodec_send_packet(codec_ctx_, av_packet_) == 0 &&
            avcodec_receive_frame(codec_ctx_, av_frame_) == 0) {

            cv::Mat nv12(h + h / 2, w, CV_8UC1);
            for (int i = 0; i < h; ++i)
                memcpy(nv12.ptr(i), av_frame_->data[0] + i * av_frame_->linesize[0], w);
            for (int i = 0; i < h / 2; ++i)
                memcpy(nv12.ptr(i + h), av_frame_->data[1] + i * av_frame_->linesize[1], w);

            cv::Mat bgr;
            cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);

            {
                std::unique_lock<std::mutex> lock(cap_mutex_);
                if (capture_queue_.size() >= 2) capture_queue_.pop();
                capture_queue_.push(bgr.clone());
            }
            cap_cv_.notify_one();
        }
    }

    av_frame_free(&av_frame_);
    av_packet_free(&av_packet_);
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    avformat_network_deinit();
}






void VideoStream::inferenceLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(cap_mutex_);
        cap_cv_.wait(lock, [this] { return !capture_queue_.empty() || !running_; });
        if (!running_) break;
        cv::Mat frame = capture_queue_.front(); capture_queue_.pop();
        lock.unlock();

        auto undistorted = calibrator_->undistort(frame);

        // if (detector_ && pose_) 推理等逻辑...
        // 可选：画关键点、框、骨架...

        {
            std::lock_guard<std::mutex> dlock(disp_mutex_);
            display_queue_.push(undistorted);
        }
        disp_cv_.notify_one();
    }
}
void VideoStream::displayLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(disp_mutex_);
        disp_cv_.wait(lock, [this] { return !display_queue_.empty() || !running_; });
        if (!running_) break;
        cv::Mat img = display_queue_.front(); display_queue_.pop();
        lock.unlock();

        std::lock_guard<std::mutex> ig(imshow_mutex_);
        cv::imshow(window_name_, img);
        if (cv::waitKey(1) == 'q') break;
    }
}


}