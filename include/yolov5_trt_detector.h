#pragma once
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace detectPerson {

struct DetectResult {
    int classId;
    float conf;
    cv::Rect box;
};

class YOLOv5TRTDetector {
public:
    YOLOv5TRTDetector();
    ~YOLOv5TRTDetector();

    void initConfig(const std::string& engineFile,
                    float confThreshold = 0.5f,
                    float scoreThreshold = 0.5f);

    /* —— 每路相机独享 —— */
    struct Context {
        nvinfer1::IExecutionContext* ctx{nullptr};
        cudaStream_t stream{nullptr};
        void* input_buf{nullptr};
        void* output_buf{nullptr};
        std::vector<float> prob;
    };
    std::shared_ptr<Context> createContext();          // 线程安全

    void detect(cv::Mat& frame, std::vector<DetectResult>& res,
                Context& ctx);                         // 带 Context 的推理

    int inputWidth()  const { return input_w; }
    int inputHeight() const { return input_h; }

private:
    /* shared */
    nvinfer1::IRuntime*    runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};

    int input_w{}, input_h{};
    int output_w{}, output_h{};
    float conf_threshold{}, score_threshold{};
};

}  // namespace detectPerson
