#include "yolov5_trt_demo.h"
#include <NvInferRuntime.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "trt_logger.h"

namespace detectPerson {


YOLOv5TRTDetector::YOLOv5TRTDetector() = default;

YOLOv5TRTDetector::~YOLOv5TRTDetector() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for (void* buffer : buffers) {
        if (buffer) cudaFree(buffer);
    }
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}

void YOLOv5TRTDetector::initConfig(const std::string& engineFile, float confThreshold, float scoreThreshold) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engineFile);
    }
    std::vector<char> trtModelStream((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");

    engine = runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size());
    if (!engine) throw std::runtime_error("Failed to deserialize engine");

    context = engine->createExecutionContext();
    if (!context) throw std::runtime_error("Failed to create execution context");

    int input_index = engine->getBindingIndex("images");
    int output_index = engine->getBindingIndex("output0");

    if (input_index == -1 || output_index == -1) {
        throw std::runtime_error("Invalid binding index for inputs/outputs");
    }

    nvinfer1::Dims input_shape = engine->getBindingDimensions(input_index);
    nvinfer1::Dims output_shape = engine->getBindingDimensions(output_index);

    if (input_shape.nbDims < 4 || output_shape.nbDims < 3) {
        throw std::runtime_error("Unexpected input/output tensor dimensions");
    }

    input_h = input_shape.d[2];
    input_w = input_shape.d[3];
    output_h = output_shape.d[1];
    output_w = output_shape.d[2];

    buffers.resize(engine->getNbBindings(), nullptr);

    size_t input_size = input_h * input_w * 3 * sizeof(float);
    size_t output_size = output_h * output_w * sizeof(float);

    if (cudaMalloc(&buffers[input_index], input_size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate input buffer");
    }
    if (cudaMalloc(&buffers[output_index], output_size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate output buffer");
    }

    prob.resize(output_h * output_w);
    cudaStreamCreate(&stream);

    conf_threshold = confThreshold;
    score_threshold = scoreThreshold;
}

void YOLOv5TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results) {
    if (frame.empty()) {
        throw std::runtime_error("Input frame is empty");
    }

    int64 start = cv::getTickCount();

    int w = frame.cols, h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
        frame.copyTo(image(cv::Rect(0, 0, w, h)));
    /*
    input_w, input_h 是模型的输入尺寸（如 640×640）

    x_factor, y_factor 是从原图到模型输入尺寸的缩放比例

    用于后续把预测框坐标从模型尺寸还原到原始尺寸
    */
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat tensor = cv::dnn::blobFromImage(image, 1.0f / 255.f, cv::Size(input_w, input_h), cv::Scalar(), true);

    cudaMemcpyAsync(buffers[engine->getBindingIndex("images")], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (!context->enqueueV2(buffers.data(), stream, nullptr)) {
        throw std::runtime_error("TensorRT inference failed");
    }
    cudaMemcpyAsync(prob.data(), buffers[engine->getBindingIndex("output0")], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    cv::Mat det_output(output_h, output_w, CV_32F, prob.data());
    for (int i = 0; i < det_output.rows; ++i) {
        float confidence = det_output.at<float>(i, 4);
        if (confidence < conf_threshold) continue;

        cv::Mat scores = det_output.row(i).colRange(5, output_w);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

        // 只保留 person 类（classId == 0）
        if (score > score_threshold && classIdPoint.x == 0) {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);

            int x = static_cast<int>((cx - 0.5f * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5f * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);

            boxes.emplace_back(x, y, width, height);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(score));
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, 0.45f, indexes);
    for (int idx : indexes) {
        DetectResult dr{ classIds[idx], confidences[idx], boxes[idx] };
        cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 1);
        cv::rectangle(frame,
                      cv::Point(boxes[idx].tl().x, boxes[idx].tl().y - 20),
                      cv::Point(boxes[idx].br().x, boxes[idx].tl().y),
                      cv::Scalar(0, 0, 255), -1);
        results.emplace_back(dr);
    }

    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS:%.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2);
}

} // namespace detectPerson
