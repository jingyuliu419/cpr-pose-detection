#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace detectPersonInt8 {

struct DetectResult {
    int classId;  // 替换 id
    float score;  // 替换 conf
    cv::Rect box;
};


class YOLOv5Int8TRTDetector {
public:
    struct Context {
        nvinfer1::IExecutionContext* ctx{nullptr};
        cudaStream_t stream{nullptr};
        void* input_dev{nullptr};
        void* output_dev{nullptr};
        std::vector<float> output_host;
    };

    YOLOv5Int8TRTDetector();
    ~YOLOv5Int8TRTDetector();

    /* load engine */
    void init(const std::string& engineFile,
              float confT = 0.25f, float scoreT = 0.25f);

    /* single‑thread helper */
    void detect(cv::Mat& img, std::vector<DetectResult>& results);

    /* user‑supplied context */
    std::shared_ptr<Context> createContext();
    void detect(cv::Mat& img, std::vector<DetectResult>& results, Context& ctx);

private:
    static size_t vol(const nvinfer1::Dims& d);
    static size_t dtypeSize(nvinfer1::DataType t);

    std::shared_ptr<Context> default_ctx_;
    nvinfer1::IRuntime*    runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    int inW_{}, inH_{};
    int outW_{}, outH_{};
    size_t inBytes_{}, outBytes_{};
    float conf_th_{0.25f};
    float score_th_{0.25f};
};

} // namespace detectPersonInt8