#pragma once
#include <vector>
#include <string>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

namespace pose {

struct Keypoint {
    cv::Point2f pt;
    float score;
};

class LiteHRNetTRT {
public:
    LiteHRNetTRT(const std::string& engine_file);
    ~LiteHRNetTRT();

    void infer(const cv::Mat& image, std::vector<Keypoint>& keypoints);
    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }

private:
    void preprocess(const cv::Mat& image, void* gpu_input); // <-- 修改为 void*
    void postprocess(void* output_ptr, std::vector<Keypoint>& keypoints, const cv::Size& roi_size);


    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    void* buffers_[2]{nullptr, nullptr};
    int input_index_{};
    int output_index_{};

    int input_w_{};
    int input_h_{};
    int output_c_{};
    int output_h_{};
    int output_w_{};

    cudaStream_t stream_{};
    std::vector<float> output_;
};

} // namespace pose
