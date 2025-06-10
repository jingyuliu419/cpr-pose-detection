#include "video_stream.h"
#include "nv12_cuda.cuh"          // 自定义 CUDA kernel
#include <libswscale/swscale.h>
#include <opencv2/cudaarithm.hpp>
#include "ui_display_thread.h" 
using namespace video;

/* ---------- ctor / dtor ---------- */
VideoStream::VideoStream(const std::string& url,
                         const std::string& win,
                         const std::string& calib_yaml,
                         std::shared_ptr<detectPerson::YOLOv5TRTDetector> det,
                         std::shared_ptr<pose::LiteHRNetTRT> pose,
                         const std::vector<std::pair<int,int>>& skel,
                         std::shared_ptr<rclcpp::Node> ros_node)
    : url_(url), window_name_(win),
      detector_(std::move(det)),
      pose_model_(std::move(pose)),
      skeleton_(skel),
      ros_node_(std::move(ros_node)) {

    calibrator_    = std::make_unique<calib::CameraCalibrator>(calib_yaml);
    detector_ctx_  = detector_->createContext();
    pose_ctx_      = pose_model_->createContext();
}

VideoStream::~VideoStream() { stop(); }

/* ---------- life cycle ---------- */
void VideoStream::start() {
    if (running_) return;
    running_ = true;
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
void VideoStream::captureLoop() {
        bool use_gpu_=true;
        avformat_network_init();
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "fflags", "nobuffer", 0);
        av_dict_set(&options, "flags", "low_delay", 0);
        av_dict_set(&options, "max_delay", "50000", 0);

        if (avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &options) != 0) {
            std::cerr << "Failed to open RTSP stream!" << std::endl;
            return;
        }

        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            std::cerr << "Failed to find stream info!" << std::endl;
            return;
        }

        int video_stream_index = -1;
        for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
            if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                break;
            }
        }

        if (video_stream_index == -1) {
            std::cerr << "No video stream found!" << std::endl;
            return;
        }

        AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_index]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            std::cerr << "Unsupported codec!" << std::endl;
            return;
        }

        codec_ctx_ = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx_, codecpar);
        codec_ctx_->pix_fmt = AV_PIX_FMT_NV12;
        codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            std::cerr << "Failed to open codec!" << std::endl;
            return;
        }

        av_frame_ = av_frame_alloc();
        av_packet_ = av_packet_alloc();

        const int w = codec_ctx_->width;
        const int h = codec_ctx_->height;

        std::cout << (use_gpu_ ? ">>> Using GPU mode <<<" : ">>> Using CPU mode <<<") << std::endl;

        cv::Mat bgr;

        while (running_) {
            AVPacket pkt;
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;

            bool got_packet = false;
            while (av_read_frame(fmt_ctx_, &pkt) >= 0) {
                if (pkt.stream_index == video_stream_index) {
                    av_packet_unref(av_packet_);
                    av_packet_ref(av_packet_, &pkt);
                    got_packet = true;
                }
                av_packet_unref(&pkt);
                if (got_packet) break;
            }

            if (!got_packet) continue;

            if (avcodec_send_packet(codec_ctx_, av_packet_) == 0) {
                while (avcodec_receive_frame(codec_ctx_, av_frame_) == 0) {
                    if (av_frame_->format == -1) {
                        std::cerr << "Invalid frame format!" << std::endl;
                        continue;
                    }

                    if (av_frame_->format != AV_PIX_FMT_NV12) {
                        SwsContext* sws_ctx = sws_getContext(
                            w, h, (AVPixelFormat)av_frame_->format,
                            w, h, AV_PIX_FMT_NV12,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

                        AVFrame* nv12_frame = av_frame_alloc();
                        nv12_frame->format = AV_PIX_FMT_NV12;
                        nv12_frame->width = w;
                        nv12_frame->height = h;
                        av_frame_get_buffer(nv12_frame, 0);

                        sws_scale(sws_ctx,
                            av_frame_->data, av_frame_->linesize, 0, h,
                            nv12_frame->data, nv12_frame->linesize);

                        av_frame_unref(av_frame_);
                        av_frame_move_ref(av_frame_, nv12_frame);
                        av_frame_free(&nv12_frame);
                        sws_freeContext(sws_ctx);
                    }

                    cv::Mat nv12(h + h / 2, w, CV_8UC1);
                    for (int y = 0; y < h; y++) {
                        memcpy(nv12.ptr(y), av_frame_->data[0] + y * av_frame_->linesize[0], w);
                    }
                    for (int y = 0; y < h / 2; y++) {
                        memcpy(nv12.ptr(h + y), av_frame_->data[1] + y * av_frame_->linesize[1], w);
                    }

                    if (use_gpu_) {
                        nv12_to_bgr_gpu(nv12.data, bgr, w, h);
                    } else {
                        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
                    }

                    if (!bgr.empty()) {
                        std::unique_lock<std::mutex> lock(cap_mutex_);
                        if (capture_queue_.size() >= 2) capture_queue_.pop();
                        capture_queue_.push(bgr.clone());
                        cap_cv_.notify_one();
                    }
                }
            }
        }

        av_frame_free(&av_frame_);
        av_packet_free(&av_packet_);
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&fmt_ctx_);
        avformat_network_deinit();
}


// LiteHRNet / COCO 的 17-keypoints 连线对
/* ---------- inference loop (优化版) ---------- */
void VideoStream::inferenceLoop() {
    /* coco17 常用骨架 (依需求可外移到 header) */
    static constexpr std::array<std::pair<int,int>, 19> kEdges {{
        {0,1},{0,2},{1,3},{2,4},{0,5},{0,6},{5,7},{7,9},
        {6,8},{8,10},{5,6},{5,11},{6,12},{11,12},
        {11,13},{13,15},{12,14},{14,16},{15,16}
    }};

    while (running_) {
        /* ① 取最新帧 ------------------------------------- */
        std::unique_lock<std::mutex> lk(cap_mutex_);
        cap_cv_.wait(lk, [this]{ return !capture_queue_.empty() || !running_; });
        if (!running_) break;
        cv::Mat frame = std::move(capture_queue_.front());
        capture_queue_.pop();
        lk.unlock();

        /* ② 畸变校正 + 检测 ------------------------------ */
        cv::Mat undist = calibrator_->undistort(frame);

        std::vector<detectPerson::DetectResult> dets;
        detector_->detect(undist, dets, *detector_ctx_);

        /* ③ 逐人姿态推理 ------------------------------ */
        for (const auto& d : dets) {
            if (d.classId != 0) continue;        // 只保留 person
            cv::Rect box = d.box & cv::Rect(0, 0, undist.cols, undist.rows);
            if (box.empty()) continue;

            cv::rectangle(undist, box, {0,255,0}, 2);

            cv::Mat roi = undist(box).clone();   // 推理用局部
            if (roi.empty()) continue;

            std::vector<pose::Keypoint> kps;
            pose_model_->infer(roi, kps, *pose_ctx_);

            /* 全局坐标关键点容器，-1 表示该点无效 */
            std::array<cv::Point2f,17> pts{};
            pts.fill({-1.f, -1.f});

            for (size_t i = 0; i < kps.size() && i < 17; ++i) {
                const auto& kp = kps[i];
                cv::Point2f pt = kp.pt + cv::Point2f(box.x, box.y);
                cv::circle(undist, pt, 3, {0,0,255}, -1);
                pts[i] = pt;
            }

            for (auto [u,v] : kEdges) {
                const cv::Point2f& p = pts[u];
                const cv::Point2f& q = pts[v];
                if (p.x >= 0 && q.x >= 0)
                    cv::line(undist, p, q, {255,0,0}, 2);
            }
        }

        /* ④ 将最新结果交给 UI -------------------------- */
        {
            std::lock_guard<std::mutex> g(disp_mutex_);
            while (display_queue_.size() > 0)    // 仅保留 1 帧
                display_queue_.pop();
            display_queue_.push(std::move(undist));
        }
        ui::notifyUI();                          // 唤醒 UI 线程
    }
}

