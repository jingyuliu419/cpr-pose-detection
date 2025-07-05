#include "video_stream.h"
#include "nv12_cuda.cuh"          // 自定义 CUDA kernel
#include <libswscale/swscale.h>
#include <opencv2/cudaarithm.hpp>
#include "ui_display_thread.h" 
#include "rtmpose_trt.h"
#include "charuco_camera_manager.h"
using namespace video;

/* ---------- ctor / dtor ---------- */
using namespace video;

VideoStream::VideoStream(const std::string& url,
                         const std::string& win,
                         const std::string& calib_yaml,
                         std::shared_ptr<detectPerson::YOLOv5TRTDetector> det,
                         std::shared_ptr<posetiny::RTMPoseTRT> pose,
                         std::shared_ptr<charuco::CameraManager> cam_mgr)
    : url_(url), window_name_(win),
      detector_(std::move(det)),
      pose_model_(std::move(pose)),
      camera_mgr_(std::move(cam_mgr))
    {
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
                        {
                            std::lock_guard<std::mutex> lock(last_mtx_);
                            last_frame_ = bgr.clone();          // ← 先 clone 再保存，避免野指针
                        }

                        std::unique_lock<std::mutex> lock(cap_mutex_);
                        if (capture_queue_.size() >= 2) capture_queue_.pop();
                        capture_queue_.push(std::move(bgr));    // 不再 clone，队列里就用原 Mat
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
// video_stream_cuda.cpp

void VideoStream::decodeWithCudaLoophard() {
    const bool use_gpu = true;
    avformat_network_init();

    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "fflags", "nobuffer", 0);
    av_dict_set(&options, "flags", "low_delay", 0);
    av_dict_set(&options, "max_delay", "50000", 0);

    if (avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &options) != 0) {
        std::cerr << "[ERROR] RTSP open failed\n";
        return;
    }
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        std::cerr << "[ERROR] No stream info\n";
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
        std::cerr << "[ERROR] No video stream\n";
        return;
    }

    AVBufferRef* hw_device = nullptr;
    if (av_hwdevice_ctx_create(&hw_device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        std::cerr << "[ERROR] CUDA device init failed\n";
        return;
    }

    const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        std::cerr << "[ERROR] CUVID decoder not found\n";
        return;
    }

    AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_index]->codecpar;
    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_, codecpar);
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device);
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        std::cerr << "[ERROR] Decoder open failed\n";
        return;
    }

    av_frame_ = av_frame_alloc();
    av_packet_ = av_packet_alloc();
    int w = codec_ctx_->width;
    int h = codec_ctx_->height;
    std::cout << (use_gpu ? "[INFO] GPU mode" : "[INFO] CPU mode") << std::endl;

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

        if (avcodec_send_packet(codec_ctx_, av_packet_) < 0) continue;

        while (avcodec_receive_frame(codec_ctx_, av_frame_) == 0) {
            AVFrame* frame = av_frame_;
            AVFrame* nv12_frame = nullptr;
            SwsContext* sws_ctx = nullptr;

            if (frame->format != AV_PIX_FMT_NV12) {
                sws_ctx = sws_getContext(w, h, (AVPixelFormat)frame->format,
                                         w, h, AV_PIX_FMT_NV12,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                nv12_frame = av_frame_alloc();
                nv12_frame->format = AV_PIX_FMT_NV12;
                nv12_frame->width = w;
                nv12_frame->height = h;
                av_frame_get_buffer(nv12_frame, 0);
                sws_scale(sws_ctx, frame->data, frame->linesize, 0, h,
                          nv12_frame->data, nv12_frame->linesize);
                frame = nv12_frame;
            }

            cv::Mat nv12(h + h / 2, w, CV_8UC1);
            for (int y = 0; y < h; ++y)
                memcpy(nv12.ptr(y), frame->data[0] + y * frame->linesize[0], w);
            for (int y = 0; y < h / 2; ++y)
                memcpy(nv12.ptr(h + y), frame->data[1] + y * frame->linesize[1], w);

            if (use_gpu) {
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

            if (nv12_frame) av_frame_free(&nv12_frame);
            if (sws_ctx) sws_freeContext(sws_ctx);
        }
    }

    av_frame_free(&av_frame_);
    av_packet_free(&av_packet_);
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&fmt_ctx_);
    avformat_network_deinit();
}


/***************  video_stream.cpp  *****************/
#define WRIST_IDX 9        // 根据你的关键点顺序调整
static const float KPT_TH = 0.01f;          // 关键点置信阈值
static const cv::Scalar X_COLOR(  0,   0, 255);   // +X 红
static const cv::Scalar Y_COLOR(  0, 255,   0);   // +Y 绿
static const cv::Scalar Z_COLOR(255,   0,   0);   // +Z 蓝

void VideoStream::inferenceLoop()
{
    constexpr std::array<std::pair<int,int>,12> kEdges {{
        {5,11},{6,12},{5,6},{5,7},{6,8},
        {7,9},{8,10}
    }};
    constexpr float KPT_TH = 0.25f;
    constexpr int   AXIS_THICK = 3;

    /* ---- λ: 画世界原点 + 坐标轴 ---- */
    auto drawOriginAxes = [&](cv::Mat& img)
    {
        if (!camera_mgr_->hasValidExtrinsics()) return;

        const cv::Mat& K  = camera_mgr_->K();
        const cv::Mat& rv = camera_mgr_->rvec();
        const cv::Mat& tv = camera_mgr_->tvec();

        std::vector<cv::Point3f> pts3 = {
            {0,0,0},     {0.10f,0,0},   // +X 10 cm
            {0,0.10f,0}, {0,0,0.10f}    // +Y / +Z
        };
        std::vector<cv::Point2f> pts2;
        cv::projectPoints(pts3, rv, tv, K, cv::noArray(), pts2);

        cv::drawMarker(img, pts2[0], cv::Scalar(0,255,255), cv::MARKER_CROSS, 12, 2);
        cv::arrowedLine(img, pts2[0], pts2[1], cv::Scalar(  0,  0,255), AXIS_THICK); // X 红
        cv::arrowedLine(img, pts2[0], pts2[2], cv::Scalar(  0,255,  0), AXIS_THICK); // Y 绿
        cv::arrowedLine(img, pts2[0], pts2[3], cv::Scalar(255,  0,  0), AXIS_THICK); // Z 蓝
    };
    /* ----------------------------------- */

    while (running_)
    {
        /* ---------- 1. 取帧 ---------- */
        std::unique_lock<std::mutex> lk(cap_mutex_);
        cap_cv_.wait(lk, [this]{ return !capture_queue_.empty() || !running_; });
        if (!running_) break;
        cv::Mat frame = std::move(capture_queue_.front());
        capture_queue_.pop();
        lk.unlock();

        /* ---------- 2. 去畸变 ---------- */
        cv::Mat undist = camera_mgr_->undistort(calibrator_->undistort(frame));

        /* ---------- 3. 检测 ---------- */
        std::vector<detectPerson::DetectResult> dets;
        detector_->detect(undist, dets, *detector_ctx_);

        for (auto const& d : dets)
        {
            if (d.classId != 0) continue;
            cv::Rect box = d.box & cv::Rect(0,0,undist.cols,undist.rows);
            if (box.empty()) continue;

            cv::rectangle(undist, box, {0,255,0}, 2);

            cv::Mat roi = undist(box).clone();
            if (roi.empty()) continue;

            /* ---- 关键点 ---- */
            std::vector<cv::Point2f> kpts; std::vector<float> confs;
            pose_model_->infer(roi, kpts, confs, *pose_ctx_);

            std::array<cv::Point2f,17> gpts; gpts.fill({-1,-1});
            for (size_t i=0;i<kpts.size() && i<17;++i)
                if (confs[i] >= KPT_TH)
                {
                    gpts[i] = kpts[i] + cv::Point2f(box.x,box.y);
                    cv::circle(undist, gpts[i], 4, {0,0,255}, -1);
                }

            /* ---- 世界坐标 (手腕) ---- */
            constexpr int WRIST = 9;
            if (gpts[WRIST].x >= 0)
            {
                const float depth_z = 0.60f;                      // 经验深度
                cv::Point3f pw = camera_mgr_->pixel2world(gpts[WRIST], depth_z);

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << '(' << pw.x << ", " << pw.y << ", " << pw.z << ')';
                cv::putText(undist, oss.str(), gpts[WRIST],
                            cv::FONT_HERSHEY_SIMPLEX, 2,
                            {255,0,0}, 6);
            }

            /* ---- 画骨架 ---- */
            for (auto [u,v]: kEdges)
                if (gpts[u].x >= 0 && gpts[v].x >= 0)
                    cv::line(undist, gpts[u], gpts[v], {0,0,255}, 5);
        }

        /* ---------- 4. 原点坐标轴 ---------- */
        drawOriginAxes(undist);

        /* ---------- 5. 推 UI ---------- */
        {
            std::lock_guard<std::mutex> g(disp_mutex_);
            while (!display_queue_.empty()) display_queue_.pop();
            display_queue_.push(std::move(undist));
        }
        ui::notifyUI();
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


