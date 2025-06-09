// File: litehrnet_pose_trt.cpp
#include "litehrnet_pose_trt.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include "trt_logger.h"

namespace pose {

LiteHRNetTRT::LiteHRNetTRT(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    assert(file.good());
    file.seekg(0, file.end);
    int size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    assert(engine_);
    context_ = engine_->createExecutionContext();
    assert(context_);

    input_index_ = engine_->getBindingIndex("input");
    output_index_ = engine_->getBindingIndex("output");

    auto in_dims = engine_->getBindingDimensions(input_index_);
    input_h_ = in_dims.d[2];
    input_w_ = in_dims.d[3];

    auto out_dims = engine_->getBindingDimensions(output_index_);
    output_c_ = out_dims.d[1];
    output_h_ = out_dims.d[2];
    output_w_ = out_dims.d[3];

    std::cout << "Output dims: " << output_c_ << " x " << output_h_ << " x " << output_w_ << std::endl;

    cudaMalloc(&buffers_[input_index_], input_h_ * input_w_ * 3 * sizeof(__half));
    cudaMalloc(&buffers_[output_index_], output_c_ * output_h_ * output_w_ * sizeof(__half));

    output_.resize(output_c_ * output_h_ * output_w_);
    cudaStreamCreate(&stream_);
}

LiteHRNetTRT::~LiteHRNetTRT() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    if (buffers_[0]) cudaFree(buffers_[0]);
    if (buffers_[1]) cudaFree(buffers_[1]);
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
}

void LiteHRNetTRT::preprocess(const cv::Mat& image, void* gpu_input) {
    cv::Mat resized, rgb, float_img;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32F);

    float mean[3] = {123.675f, 116.28f, 103.53f};
    float std[3]  = {58.395f, 57.12f, 57.375f};

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int i = 0; i < 3; ++i)
        channels[i] = (channels[i] - mean[i]) / std[i];
    cv::merge(channels, float_img);

    float* tmp = reinterpret_cast<float*>(malloc(3 * input_h_ * input_w_ * sizeof(float)));
    std::vector<cv::Mat> chw;
    for (int i = 0; i < 3; ++i)
        chw.emplace_back(input_h_, input_w_, CV_32F, tmp + i * input_h_ * input_w_);
    cv::split(float_img, chw);

    std::vector<__half> half_input(3 * input_h_ * input_w_);
    for (int i = 0; i < 3 * input_h_ * input_w_; ++i)
        half_input[i] = __float2half(tmp[i]);

    free(tmp);
    cudaMemcpyAsync(gpu_input, half_input.data(), half_input.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
}

void LiteHRNetTRT::postprocess(void* output_ptr, std::vector<Keypoint>& keypoints, const cv::Size& roi_size) {
    __half* output = reinterpret_cast<__half*>(output_ptr);
    keypoints.clear();

    float scale_x = static_cast<float>(roi_size.width) / output_w_;
    float scale_y = static_cast<float>(roi_size.height) / output_h_;
    float threshold = 0.4f;

    for (int k = 0; k < output_c_; ++k) {
        float max_val = -1;
        int max_idx = 0;
        for (int i = 0; i < output_h_ * output_w_; ++i) {
            float val = __half2float(output[k * output_h_ * output_w_ + i]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        if (max_val < threshold) continue;

        int x = max_idx % output_w_;
        int y = max_idx / output_w_;

        Keypoint kp;
        kp.pt = cv::Point2f(x * scale_x, y * scale_y);
        kp.score = max_val;
        keypoints.push_back(kp);
    }
}

void LiteHRNetTRT::infer(const cv::Mat& image, std::vector<Keypoint>& keypoints) {
    if (image.empty() || image.cols < 10 || image.rows < 10 ||
        image.cols / image.rows > 10 || image.rows / image.cols > 10) {
        std::cerr << "[ERROR] LiteHRNetTRT::infer received invalid image (" 
                  << image.cols << "x" << image.rows << ")" << std::endl;
        return;
    }

    preprocess(image, buffers_[input_index_]);
    context_->enqueueV2(buffers_, stream_, nullptr);
    cudaMemcpyAsync(output_.data(), buffers_[output_index_], 
                    output_c_ * output_h_ * output_w_ * sizeof(__half), 
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    postprocess(output_.data(), keypoints, image.size());
}


} // namespace pose
