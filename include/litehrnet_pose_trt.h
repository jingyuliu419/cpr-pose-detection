/*****************************************************
 * File: litehrnet_pose_trt.h
 *****************************************************/
#pragma once
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace pose {

struct Keypoint {
    cv::Point2f pt;
    float score;
};

class LiteHRNetTRT {
public:
    explicit LiteHRNetTRT(const std::string& engine_file);
    ~LiteHRNetTRT();                                       // 仅销毁 engine/runtime

    /*—— 每路相机调用一次 ——*/
    struct Context {
        nvinfer1::IExecutionContext* ctx{nullptr};
        cudaStream_t stream{nullptr};
        void* input_buf{nullptr};
        void* output_buf{nullptr};
        std::vector<__half> output;                        // Host 侧缓存
    };
    std::shared_ptr<Context> createContext();              // 线程安全

    /*—— 推理 ——*/
    void infer(const cv::Mat& img, std::vector<Keypoint>& kps,
               Context& ctx);

    int input_width()  const { return input_w_; }
    int input_height() const { return input_h_; }

private:
    /* helper */
    void preprocess(const cv::Mat& img, void* gpu_input,
                    cudaStream_t stream) const;
    void postprocess(void* output_ptr, std::vector<Keypoint>& kps,
                     const cv::Size& roi_size) const;

    /* shared between contexts */
    nvinfer1::IRuntime*      runtime_{nullptr};
    nvinfer1::ICudaEngine*   engine_{nullptr};

    /* dims / index */
    int input_w_{};
    int input_h_{};
    int output_c_{};
    int output_h_{};
    int output_w_{};
    int input_index_{};
    int output_index_{};
};

} // namespace pose
