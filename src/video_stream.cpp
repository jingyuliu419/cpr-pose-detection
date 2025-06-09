#include<video_stream.h>
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
                            const std::vector<std::pair<int, int>>& skeleton)
        : url_(url), window_name_(window_name), detector_(detector), pose_(pose), skeleton_(skeleton) {
        calibrator_ = std::make_unique<calib::CameraCalibrator>(calib_path);
    }


    VideoStream::~VideoStream() {
        stop();
    }
    void VideoStream::start(){
        running_=true;
        worker_=std::thread(&VideoStream::run,this);
    }
    void VideoStream::stop(){
        running_=false;
        if(worker_.joinable())worker_.join();
    }
void VideoStream::run() {


    avformat_network_init();
    AVFormatContext* fmt_ctx = nullptr;
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "max_delay", "1000000", 0);  // 单位 us

    avformat_open_input(&fmt_ctx, url_.c_str(), nullptr, &options);
    avformat_find_stream_info(fmt_ctx, nullptr);

    int index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            index = i; break;
        }
    }

    AVCodecParameters* codecpar = fmt_ctx->streams[index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    avcodec_open2(codec_ctx, codec, nullptr);

    SwsContext* sws = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                     codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    cv::Mat image(codec_ctx->height, codec_ctx->width, CV_8UC3);
    uint8_t* dest[4] = { image.data, nullptr, nullptr, nullptr };
    int linesize[4] = { static_cast<int>(image.step), 0, 0, 0 };
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_, codec_ctx->width, codec_ctx->height);

    while (running_) {
        if (av_read_frame(fmt_ctx, packet) >= 0 && packet->stream_index == index) {
            if (avcodec_send_packet(codec_ctx, packet) == 0 && avcodec_receive_frame(codec_ctx, frame) == 0) {
                sws_scale(sws, frame->data, frame->linesize, 0, codec_ctx->height, dest, linesize);
                auto undistorted = calibrator_->undistort(image);
                bool drawn = false;

                std::vector<detectPerson::DetectResult> detections;
                detector_->detect(undistorted, detections);

                for (const auto& dr : detections) {
                    if (dr.classId != 0) continue;
                    cv::Rect person_box = dr.box & cv::Rect(0, 0, undistorted.cols, undistorted.rows);
                    cv::Mat roi = undistorted(person_box).clone();
                    std::cout << "[DEBUG] Processed frame with ROI size: " << roi.cols << "x" << roi.rows << std::endl;

                    if (roi.empty() || roi.cols < 10 || roi.rows < 10 || roi.cols / roi.rows > 10 || roi.rows / roi.cols > 10) {
                        std::cerr << "[WARNING] Skipped invalid ROI: (" << roi.cols << "x" << roi.rows << ")" << std::endl;
                        continue;
                    }
                    std::vector<pose::Keypoint> keypoints;
                    {
                        std::lock_guard<std::mutex> lock(pose_mutex_);
                        pose_->infer(roi, keypoints);
                    }
                    std::cout << "[DEBUG] keypoints detected: " << keypoints.size() << std::endl;

                    for (const auto& kp : keypoints) {
                        cv::circle(undistorted, kp.pt + cv::Point2f(person_box.x, person_box.y), 3, {0, 0, 255}, -1);
                    }

                    for (const auto& [i, j] : skeleton_) {
                        if (i >= keypoints.size() || j >= keypoints.size()) continue;
                        if (keypoints[i].score > 0.3f && keypoints[j].score > 0.3f) {
                            cv::line(undistorted, keypoints[i].pt + cv::Point2f(person_box.x, person_box.y),
                                     keypoints[j].pt + cv::Point2f(person_box.x, person_box.y),
                                     cv::Scalar(0, 255, 255), 2);
                        }
                    }

                    cv::rectangle(undistorted, person_box, cv::Scalar(0, 255, 0), 2);
                    drawn = true;
                }
                // {
                //     std::lock_guard<std::mutex> lock(imshow_mutex_);
                    cv::imshow(window_name_, undistorted);
                //     if (cv::waitKey(1) == 'q') break;
                // }
            }
            av_packet_unref(packet);
        }
    }

    av_frame_free(&frame);
    av_packet_free(&packet);
    sws_freeContext(sws);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    avformat_network_deinit();
}
}