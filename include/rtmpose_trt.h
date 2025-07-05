#ifndef RTMPOSE_TRT_H
#define RTMPOSE_TRT_H

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace posetiny {

class RTMPoseTRT {
public:
    // 上下文结构体，用于管理推理状态
    struct Context {
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t stream = nullptr;
        float* input = nullptr;        // 输入数据（设备内存）
        float* output_x = nullptr;     // simcc_x 输出（设备内存）
        float* output_y = nullptr;     // simcc_y 输出（设备内存）
        std::vector<float> simcc_x;    // simcc_x 主机数据
        std::vector<float> simcc_y;    // simcc_y 主机数据
    };

    // 构造函数：加载 TensorRT 引擎
    explicit RTMPoseTRT(const std::string& engine_path);
    
    // 析构函数
    ~RTMPoseTRT();
    
    // 创建推理上下文
    std::shared_ptr<Context> createContext();
    
    // 执行推理
    void infer(const cv::Mat&                    img,
               std::vector<cv::Point2f>&         kpts,
               std::vector<float>&               confs,
               Context&                          ctx);
    
    // 获取输入尺寸
    int inputHeight() const { return input_h_; }
    int inputWidth() const { return input_w_; }
    
    // 获取关键点数量
    int numKeypoints() const { return num_kpts_; }

private:
    // 预处理函数
    void preprocess(const cv::Mat& img, 
                    void* gpu_input, 
                    cudaStream_t stream) const;
    
    // 后处理函数

    void postprocess(const std::vector<float>& simcc_x,
                    const std::vector<float>& simcc_y,
                    std::vector<cv::Point2f>& kpts,
                    std::vector<float>&       confs,
                    const cv::Size&           bbox_size) const;

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    
    int input_index_ = -1;      // 输入绑定索引
    int output_x_index_ = -1;   // simcc_x 输出索引
    int output_y_index_ = -1;   // simcc_y 输出索引
    
    int input_h_ = 0;           // 输入高度
    int input_w_ = 0;           // 输入宽度
    int num_kpts_ = 0;          // 关键点数量
    int simcc_w_ = 0;           // simcc_x 宽度
    int simcc_h_ = 0;           // simcc_y 高度
};

} // namespace posetiny

#endif // RTMPOSE_TRT_H