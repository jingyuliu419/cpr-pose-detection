// File: rtmpose_trt.cpp
#include "rtmpose_trt.h"
#include "trt_logger.h"

#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace posetiny {

RTMPoseTRT::RTMPoseTRT(const std::string& engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) throw std::runtime_error("Cannot open engine: " + engine_path);

    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");

    std::cout << "[RTMPose] Engine bindings:" << std::endl;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        std::cout << "  [" << i << "] " << engine_->getBindingName(i)
                  << (engine_->bindingIsInput(i) ? " (input)" : " (output)") << std::endl;
    }

    // 替换为实际的 binding 名（例如 mmdeploy 默认是 input / output.simcc_x / output.simcc_y）
    input_index_     = engine_->getBindingIndex("input");
    output_x_index_ = engine_->getBindingIndex("output");   // SimCC-x
    output_y_index_ = engine_->getBindingIndex("506");      // SimCC-y

    if (input_index_ == -1 || output_x_index_ == -1 || output_y_index_ == -1) {
        throw std::runtime_error("Binding index not found. Check input/output names in engine.");
    }

    auto in_dims = engine_->getBindingDimensions(input_index_);
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];

    auto out_x_dims = engine_->getBindingDimensions(output_x_index_);
    num_kpts_ = out_x_dims.d[1];
    simcc_w_ = out_x_dims.d[2];

    auto out_y_dims = engine_->getBindingDimensions(output_y_index_);
    simcc_h_ = out_y_dims.d[2];

    std::cout << "[RTMPose] simcc_x: " << num_kpts_ << " x " << simcc_w_ 
              << ", simcc_y: " << num_kpts_ << " x " << simcc_h_ << std::endl;
}



RTMPoseTRT::~RTMPoseTRT() {
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
}

std::shared_ptr<RTMPoseTRT::Context> RTMPoseTRT::createContext() {
    auto deleter = [this](Context* ctx) {
        cudaStreamSynchronize(ctx->stream);
        cudaFree(ctx->input);
        cudaFree(ctx->output_x);
        cudaFree(ctx->output_y);
        if (ctx->context) ctx->context->destroy();
        cudaStreamDestroy(ctx->stream);
        delete ctx;
    };

    std::shared_ptr<Context> ctx(new Context, deleter);
    ctx->context = engine_->createExecutionContext();
    if (!ctx->context) throw std::runtime_error("Failed to create ExecutionContext");

    cudaStreamCreate(&ctx->stream);

    size_t in_bytes = 3 * input_h_ * input_w_ * sizeof(float);
    size_t out_x_bytes = num_kpts_ * simcc_w_ * sizeof(float);
    size_t out_y_bytes = num_kpts_ * simcc_h_ * sizeof(float);

    cudaMalloc(reinterpret_cast<void**>(&ctx->input),    in_bytes);
    cudaMalloc(reinterpret_cast<void**>(&ctx->output_x), out_x_bytes);
    cudaMalloc(reinterpret_cast<void**>(&ctx->output_y), out_y_bytes);


    ctx->simcc_x.resize(num_kpts_ * simcc_w_);
    ctx->simcc_y.resize(num_kpts_ * simcc_h_);

    return ctx;
}

void RTMPoseTRT::infer(const cv::Mat& img,
                       std::vector<cv::Point2f>& kpts,
                       std::vector<float>& confs,
                       Context& ctx)
{
    preprocess(img, ctx.input, ctx.stream);

    // void* bindings[3] = { ctx.input, ctx.output_x, ctx.output_y };
    void* bindings[3];
    bindings[input_index_]   = ctx.input;
    bindings[output_x_index_] = ctx.output_x;
    bindings[output_y_index_] = ctx.output_y;


    ctx.context->enqueueV2(bindings, ctx.stream, nullptr);

    cudaMemcpyAsync(ctx.simcc_x.data(), ctx.output_x,
                    ctx.simcc_x.size()*sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaMemcpyAsync(ctx.simcc_y.data(), ctx.output_y,
                    ctx.simcc_y.size()*sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    postprocess(ctx.simcc_x, ctx.simcc_y, kpts, confs, img.size());
}


void RTMPoseTRT::preprocess(const cv::Mat& img, void* gpu_input, cudaStream_t stream) const {
    cv::Mat resized, rgb, f32;
    cv::resize(img, resized, {input_w_, input_h_});
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

    const float mean[3] = {123.675f / 255.f, 116.28f / 255.f, 103.53f / 255.f};
    const float std[3]  = {58.395f / 255.f, 57.12f / 255.f, 57.375f / 255.f};

    std::vector<cv::Mat> channels(3);
    cv::split(f32, channels);
    for (int i = 0; i < 3; ++i) channels[i] = (channels[i] - mean[i]) / std[i];
    cv::merge(channels, f32);

    std::vector<float> chw(3 * input_h_ * input_w_);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < input_h_; ++y)
            for (int x = 0; x < input_w_; ++x)
                chw[c * input_h_ * input_w_ + y * input_w_ + x] = f32.at<cv::Vec3f>(y, x)[c];

    cudaMemcpyAsync(gpu_input, chw.data(), chw.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
}

void RTMPoseTRT::postprocess(const std::vector<float>& simcc_x,
                             const std::vector<float>& simcc_y,
                             std::vector<cv::Point2f>& kpts,
                             std::vector<float>&       confs,
                             const cv::Size& bbox_sz) const
{
    const float sx = static_cast<float>(bbox_sz.width)  / simcc_w_;
    const float sy = static_cast<float>(bbox_sz.height) / simcc_h_;

    kpts.clear();  confs.clear();
    for (int i = 0; i < num_kpts_; ++i) {
        const float* x_ptr = simcc_x.data() + i*simcc_w_;
        const float* y_ptr = simcc_y.data() + i*simcc_h_;

        int x_idx = std::max_element(x_ptr, x_ptr+simcc_w_) - x_ptr;
        int y_idx = std::max_element(y_ptr, y_ptr+simcc_h_) - y_ptr;

        /* 简单置信度：两轴最大值平均，可按需替换 softmax */
        float conf = 0.5f * (x_ptr[x_idx] + y_ptr[y_idx]);

        kpts.emplace_back(x_idx * sx, y_idx * sy);
        confs.emplace_back(conf);
    }
}


} // namespace posetiny
