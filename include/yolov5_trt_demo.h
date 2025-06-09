#pragma once
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

namespace detectPerson {

    // 检测结果结构体，包含类别ID、置信度和目标框
    struct DetectResult {
        int classId;      // 类别ID
        float conf;       // 置信度
        cv::Rect box;     // 目标框
    };

    // YOLOv5 TensorRT 推理类
    class YOLOv5TRTDetector {
    public:
        YOLOv5TRTDetector();  // 在类中声明构造函数

        // 初始化推理引擎配置
        void initConfig(const std::string& engineFile, float confThreshold, float scoreThreshold);

        // 对输入帧进行检测，输出检测结果
        void detect(cv::Mat& frame, std::vector<DetectResult>& result);

        // 析构函数，释放资源
        ~YOLOv5TRTDetector();

    private:
        float conf_threshold = 0.25f;   // 置信度阈值
        float score_threshold = 0.25f;  // 分数阈值
        int input_h = 640;              // 输入高度
        int input_w = 640;              // 输入宽度
        int output_h;                   // 输出高度
        int output_w;                   // 输出宽度

        nvinfer1::IRuntime* runtime = nullptr;                // TensorRT 运行时
        nvinfer1::ICudaEngine* engine = nullptr;              // TensorRT 引擎
        nvinfer1::IExecutionContext* context = nullptr;       // 推理上下文
        std::vector<void*> buffers;                            // 输入输出缓冲区
        std::vector<float> prob;                               // 存储推理输出概率
        cudaStream_t stream;                                   // CUDA 流
    };
}