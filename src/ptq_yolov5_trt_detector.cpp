#include "quant_yolov5_trt_detector.h"
#include "trt_logger.h"

#include <fstream>
#include <stdexcept>
#include <thread>
#include <iostream>

using namespace detectPersonInt8;

size_t YOLOv5Int8TRTDetector::vol(const nvinfer1::Dims& d) {
    size_t v = 1; for (int i = 0; i < d.nbDims; ++i) v *= d.d[i]; return v;
}

size_t YOLOv5Int8TRTDetector::dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: throw std::runtime_error("unknown dtype");
    }
}

YOLOv5Int8TRTDetector::YOLOv5Int8TRTDetector() = default;
YOLOv5Int8TRTDetector::~YOLOv5Int8TRTDetector() {
    default_ctx_.reset();
    if (engine_)  engine_->destroy();
    if (runtime_) runtime_->destroy();
}

void YOLOv5Int8TRTDetector::init(const std::string& engineFile,
                                 float confT, float scoreT) {
    std::ifstream f(engineFile, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + engineFile);
    f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz); f.read(buf.data(), sz);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_  = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_) throw std::runtime_error("deserialize failed");

    int inIdx  = engine_->getBindingIndex("images");
    int outIdx = engine_->getBindingIndex("output0");

    auto inDims  = engine_->getBindingDimensions(inIdx);
    auto outDims = engine_->getBindingDimensions(outIdx);

    inH_  = inDims.d[2]; inW_  = inDims.d[3];
    outH_ = outDims.d[1]; outW_ = outDims.d[2];

    conf_th_  = confT; score_th_ = scoreT;
    inBytes_  = vol(inDims)  * dtypeSize(engine_->getBindingDataType(inIdx));
    outBytes_ = vol(outDims) * dtypeSize(engine_->getBindingDataType(outIdx));
}

std::shared_ptr<YOLOv5Int8TRTDetector::Context> YOLOv5Int8TRTDetector::createContext() {
    auto del = [](Context* c){
        if (c->stream) cudaStreamSynchronize(c->stream);
        if (c->input_dev)  cudaFree(c->input_dev);
        if (c->output_dev) cudaFree(c->output_dev);
        if (c->ctx) c->ctx->destroy();
        if (c->stream) cudaStreamDestroy(c->stream);
        delete c;
    };
    std::shared_ptr<Context> ctx(new Context, del);

    ctx->ctx = engine_->createExecutionContext();
    cudaStreamCreate(&ctx->stream);
    cudaMalloc(&ctx->input_dev,  inBytes_);
    cudaMalloc(&ctx->output_dev, outBytes_);
    ctx->output_host.resize(outBytes_ / sizeof(float));
    return ctx;
}

void YOLOv5Int8TRTDetector::detect(cv::Mat& img,
                                   std::vector<DetectResult>& res,
                                   Context& ctx) {
    res.clear(); 
    if (img.empty()) return;

    int w = img.cols, h = img.rows; int s = std::max(w, h);
    cv::Mat sq(s, s, CV_8UC3, cv::Scalar(0,0,0));
    img.copyTo(sq(cv::Rect(0,0,w,h)));

    float xf = sq.cols / static_cast<float>(inW_);
    float yf = sq.rows / static_cast<float>(inH_);

    static thread_local cv::Mat blob;
    blob = cv::dnn::blobFromImage(sq, 1.f / 255.f, {inW_, inH_}, {}, true, false);

    void* bindings[2];
    bindings[engine_->getBindingIndex("images")]  = ctx.input_dev;
    bindings[engine_->getBindingIndex("output0")] = ctx.output_dev;

    cudaMemcpyAsync(ctx.input_dev, blob.ptr<float>(), inBytes_,
                    cudaMemcpyHostToDevice, ctx.stream);
    ctx.ctx->enqueueV2(bindings, ctx.stream, nullptr);
    cudaMemcpyAsync(ctx.output_host.data(), ctx.output_dev, outBytes_,
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    cv::Mat out(outH_, outW_, CV_32F, ctx.output_host.data());
    std::vector<cv::Rect> boxes; std::vector<float> scores;

    for (int i = 0; i < out.rows; ++i) {
        float obj = out.at<float>(i, 4);
        if (obj < conf_th_) continue;

        cv::Mat cls = out.row(i).colRange(5, outW_);
        cv::Point idPt; double best;
        cv::minMaxLoc(cls, nullptr, &best, nullptr, &idPt);
        if (best < score_th_) continue;

        float cx = out.at<float>(i, 0), cy = out.at<float>(i, 1);
        float ow = out.at<float>(i, 2), oh = out.at<float>(i, 3);
        int x  = static_cast<int>((cx - 0.5f * ow) * xf);
        int y  = static_cast<int>((cy - 0.5f * oh) * yf);
        int ww = static_cast<int>(ow * xf);
        int hh = static_cast<int>(oh * yf);

        boxes.emplace_back(x, y, ww, hh);
        scores.push_back(static_cast<float>(best));
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_th_, 0.45f, keep);
    for (int idx : keep)
        res.push_back({0, scores[idx], boxes[idx]});
}
