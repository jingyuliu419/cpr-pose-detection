/*****************************************************
 * File: litehrnet_pose_trt.cpp
 *****************************************************/
#include "litehrnet_pose_trt.h"
#include "trt_logger.h"

#include <cassert>
#include <fstream>
#include <iostream>

namespace pose {

/* ---------- ctor / dtor ---------- */
LiteHRNetTRT::LiteHRNetTRT(const std::string& engine_file) {
    /* 反序列化 TensorRT 引擎（线程安全，只执行一次） */
    std::ifstream in(engine_file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open engine: " + engine_file);
    in.seekg(0, std::ios::end);
    const size_t sz = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<char> engine_data(sz);
    in.read(engine_data.data(), sz);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), sz);
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

    /* 解析维度 / binding index（共享常量） */
    input_index_  = engine_->getBindingIndex("input");
    output_index_ = engine_->getBindingIndex("output");

    const auto in_dims = engine_->getBindingDimensions(input_index_);
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];

    const auto out_dims = engine_->getBindingDimensions(output_index_);
    output_c_ = out_dims.d[1];
    output_h_ = out_dims.d[2];
    output_w_ = out_dims.d[3];

    std::cout << "[LiteHRNet] output dims = "
              << output_c_ << "×" << output_h_ << "×" << output_w_ << '\n';
}

LiteHRNetTRT::~LiteHRNetTRT() {
    if (engine_)  engine_->destroy();
    if (runtime_) runtime_->destroy();
}

/* ---------- context factory ---------- */
std::shared_ptr<LiteHRNetTRT::Context> LiteHRNetTRT::createContext() {
    auto deleter = [this](Context* ctx) {
        cudaStreamSynchronize(ctx->stream);
        if (ctx->input_buf)  cudaFree(ctx->input_buf);
        if (ctx->output_buf) cudaFree(ctx->output_buf);
        if (ctx->ctx)        ctx->ctx->destroy();
        if (ctx->stream)     cudaStreamDestroy(ctx->stream);
        delete ctx;
    };

    std::shared_ptr<Context> ctx(new Context, deleter);

    /* per-thread ExecutionContext & Stream */
    ctx->ctx = engine_->createExecutionContext();
    if (!ctx->ctx) throw std::runtime_error("createExecutionContext failed");
    cudaStreamCreate(&ctx->stream);

    const size_t in_bytes  = input_w_ * input_h_ * 3 * sizeof(__half);
    const size_t out_bytes = output_c_ * output_h_ * output_w_ * sizeof(__half);
    cudaMalloc(&ctx->input_buf,  in_bytes);
    cudaMalloc(&ctx->output_buf, out_bytes);

    ctx->output.resize(output_c_ * output_h_ * output_w_);
    return ctx;
}

/* ---------- public infer ---------- */
void LiteHRNetTRT::infer(const cv::Mat& img,
                         std::vector<Keypoint>& kps,
                         Context& ctx) {
    if (img.empty() || img.cols < 10 || img.rows < 10) return;

    /* 1. 预处理 → GPU */
    preprocess(img, ctx.input_buf, ctx.stream);

    /* 2. TensorRT 推理 */
    void* bindings[2]{ctx.input_buf, ctx.output_buf};
    ctx.ctx->enqueueV2(bindings, ctx.stream, nullptr);

    /* 3. 拷回 Host */
    const size_t out_bytes = output_c_ * output_h_ * output_w_ * sizeof(__half);
    cudaMemcpyAsync(ctx.output.data(), ctx.output_buf,
                    out_bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    /* 4. 后处理 */
    postprocess(ctx.output.data(), kps, img.size());
}

/* ---------- preprocess ---------- */
void LiteHRNetTRT::preprocess(const cv::Mat& img, void* gpu_input,
                              cudaStream_t stream) const {
    cv::Mat resized, rgb, f32;
    cv::resize(img, resized, {input_w_, input_h_});
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32F);

    /* mean / std 归一化 */
    const float mean[3]{123.675f, 116.28f, 103.53f};
    const float stdv[3]{58.395f, 57.12f, 57.375f};
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    for (int i = 0; i < 3; ++i) ch[i] = (ch[i] - mean[i]) / stdv[i];
    cv::merge(ch, f32);

    /* HWC → CHW → half */
    std::vector<__half> half_input(3 * input_h_ * input_w_);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_h_; ++y) {
            const float* src = f32.ptr<float>(y) + c;
            for (int x = 0; x < input_w_; ++x)
                half_input[c * input_h_ * input_w_ + y * input_w_ + x] =
                    __float2half(src[x * 3]);
        }
    }
    const size_t bytes = half_input.size() * sizeof(__half);
    cudaMemcpyAsync(gpu_input, half_input.data(), bytes,
                    cudaMemcpyHostToDevice, stream);
}

/* ---------- postprocess ---------- */
void LiteHRNetTRT::postprocess(void* out_ptr, std::vector<Keypoint>& kps,
                               const cv::Size& roi_sz) const {
    const __half* out = static_cast<const __half*>(out_ptr);
    kps.clear();

    const float sx = static_cast<float>(roi_sz.width)  / output_w_;
    const float sy = static_cast<float>(roi_sz.height) / output_h_;
    const float thr = 0.4f;

    for (int k = 0; k < output_c_; ++k) {
        float best{-1.f}; int idx{-1};
        for (int i = 0; i < output_h_ * output_w_; ++i) {
            float v = __half2float(out[k * output_h_ * output_w_ + i]);
            if (v > best) { best = v; idx = i; }
        }
        if (best < thr) continue;

        int x = idx % output_w_, y = idx / output_w_;
        kps.push_back({{x * sx, y * sy}, best});
    }
}

} // namespace pose
