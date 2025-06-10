#include "video_stream.h"
#include "nv12_cuda.cuh"          // 自定义 CUDA kernel
#include <libswscale/swscale.h>
#include <opencv2/cudaarithm.hpp>

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
static const std::vector<std::pair<int,int>> kEdges = {
    {0,1},{1,2},{2,3},{3,4},        // 头 → 右臂
    {0,5},{5,6},{6,7},              // 头 → 左臂
    {0,8},{8,9},{9,10},             // 头 → 右腿
    {0,11},{11,12},{12,13},         // 头 → 左腿
    {8,11},                         // 髋部横连
    {5,8}, {6,11}                   // 跨体两条辅助线（可选）
};

/* ---------- inference loop ---------- */
void VideoStream::inferenceLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lk(cap_mutex_);
        cap_cv_.wait(lk, [this]{ return !capture_queue_.empty() || !running_; });
        if (!running_) break;
        cv::Mat frame = capture_queue_.front(); capture_queue_.pop();
        lk.unlock();

        auto undist = calibrator_->undistort(frame);

        std::vector<detectPerson::DetectResult> dets;
        detector_->detect(undist, dets, *detector_ctx_);

        for (auto& d : dets) {
            cv::Rect box = d.box & cv::Rect(0,0,undist.cols,undist.rows);
            if (box.empty()) continue;

            cv::rectangle(undist, box, {0,255,0}, 2);       // 1️⃣ 画框

            cv::Mat roi = undist(box).clone();
            if (roi.empty()) continue;

            std::vector<pose::Keypoint> kps;
            pose_model_->infer(roi, kps, *pose_ctx_);

            // 17 关键点坐标初始化为无效
            std::vector<cv::Point2f> pts17(17, {-1,-1});    

            for (auto& kp : kps) {
                cv::Point2f pt = kp.pt + cv::Point2f(box.x, box.y);
                cv::circle(undist, pt, 3, {0,0,255}, -1);   // 2️⃣ 画点
                int idx = &kp - &kps[0];                    // kps 内部顺序 == id
                if (idx < 17) pts17[idx] = pt;              // 保存全图坐标
            }

            // 3️⃣ 画骨架连线
            for (auto&& e : kEdges) {
                const cv::Point2f& p = pts17[e.first];
                const cv::Point2f& q = pts17[e.second];
                if (p.x>=0 && q.x>=0)
                    cv::line(undist, p, q, {255,0,0}, 2);
            }
        }


        {
            std::lock_guard<std::mutex> g(disp_mutex_);
            display_queue_.push(undist);
        }
        disp_cv_.notify_one();
    }
}
