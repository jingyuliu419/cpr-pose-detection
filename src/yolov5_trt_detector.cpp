#include "yolov5_trt_detector.h"
#include "trt_logger.h"

#include <fstream>
#include <stdexcept>

using namespace detectPerson;

/* ---------- ctor / dtor ---------- */
YOLOv5TRTDetector::YOLOv5TRTDetector() = default;

YOLOv5TRTDetector::~YOLOv5TRTDetector() {
    if (engine)  engine->destroy();
    if (runtime) runtime->destroy();
}

/* ---------- init ---------- */
void YOLOv5TRTDetector::initConfig(const std::string& engineFile,
                                   float confT, float scoreT) {
    std::ifstream fin(engineFile, std::ios::binary);
    if (!fin) throw std::runtime_error("open engine failed: " + engineFile);
    fin.seekg(0, std::ios::end);
    const size_t sz = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::vector<char> buf(sz);
    fin.read(buf.data(), sz);

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) throw std::runtime_error("createInferRuntime failed");
    engine = runtime->deserializeCudaEngine(buf.data(), sz);
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed");

    int inIdx  = engine->getBindingIndex("images");
    int outIdx = engine->getBindingIndex("output0");

    const auto inDims  = engine->getBindingDimensions(inIdx);
    const auto outDims = engine->getBindingDimensions(outIdx);

    input_h = inDims.d[2]; input_w = inDims.d[3];
    output_h = outDims.d[1]; output_w = outDims.d[2];

    conf_threshold  = confT;
    score_threshold = scoreT;
}

/* ---------- context factory ---------- */
std::shared_ptr<YOLOv5TRTDetector::Context>
YOLOv5TRTDetector::createContext() {
    auto deleter = [](Context* c) {
        if (c->stream)  cudaStreamSynchronize(c->stream);
        if (c->input_buf)  cudaFree(c->input_buf);
        if (c->output_buf) cudaFree(c->output_buf);
        if (c->ctx) c->ctx->destroy();
        if (c->stream) cudaStreamDestroy(c->stream);
        delete c;
    };
    std::shared_ptr<Context> ctx(new Context, deleter);

    ctx->ctx = engine->createExecutionContext();
    if (!ctx->ctx) throw std::runtime_error("createExecutionContext failed");
    cudaStreamCreate(&ctx->stream);

    const size_t inBytes  = input_h * input_w * 3 * sizeof(float);
    const size_t outBytes = output_h * output_w * sizeof(float);
    cudaMalloc(&ctx->input_buf,  inBytes);
    cudaMalloc(&ctx->output_buf, outBytes);

    ctx->prob.resize(output_h * output_w);
    return ctx;
}

/* ---------- detect ---------- */
void YOLOv5TRTDetector::detect(cv::Mat& frame,
                               std::vector<DetectResult>& results,
                               Context& ctx) {
    results.clear();
    if (frame.empty()) return;

    /* 1. 填充为正方形 */
    int w = frame.cols, h = frame.rows;
    int _max = std::max(w, h);
    cv::Mat square = cv::Mat::zeros(_max, _max, CV_8UC3);
    frame.copyTo(square(cv::Rect(0, 0, w, h)));

    const float xf = square.cols / static_cast<float>(input_w);
    const float yf = square.rows / static_cast<float>(input_h);

    /* 2. HWC → CHW blob */
    cv::Mat blob = cv::dnn::blobFromImage(
        square, 1.f / 255.f, {input_w, input_h}, cv::Scalar(), true, false);

    /* 3. 推理 */
    void* bindings[2];
    bindings[engine->getBindingIndex("images")]  = ctx.input_buf;
    bindings[engine->getBindingIndex("output0")] = ctx.output_buf;

    const size_t inBytes = input_h * input_w * 3 * sizeof(float);
    cudaMemcpyAsync(ctx.input_buf, blob.ptr<float>(),
                    inBytes, cudaMemcpyHostToDevice, ctx.stream);
    ctx.ctx->enqueueV2(bindings, ctx.stream, nullptr);

    const size_t outBytes = output_h * output_w * sizeof(float);
    cudaMemcpyAsync(ctx.prob.data(), ctx.output_buf,
                    outBytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    /* 4. 解析输出 */
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    cv::Mat outMat(output_h, output_w, CV_32F, ctx.prob.data());

    for (int i = 0; i < outMat.rows; ++i) {
        float conf = outMat.at<float>(i, 4);
        if (conf < conf_threshold) continue;

        cv::Mat cls = outMat.row(i).colRange(5, output_w);
        cv::Point idPt; double best;
        minMaxLoc(cls, nullptr, &best, nullptr, &idPt);
        if (best < score_threshold || idPt.x != 0) continue;   // 只要 person

        float cx = outMat.at<float>(i, 0), cy = outMat.at<float>(i, 1);
        float ow = outMat.at<float>(i, 2), oh = outMat.at<float>(i, 3);

        int x = static_cast<int>((cx - 0.5f * ow) * xf);
        int y = static_cast<int>((cy - 0.5f * oh) * yf);
        int ww = static_cast<int>(ow * xf);
        int hh = static_cast<int>(oh * yf);

        boxes.emplace_back(x, y, ww, hh);
        scores.push_back(static_cast<float>(best));
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, 0.45f, keep);
    for (int idx : keep)
        results.push_back({0, scores[idx], boxes[idx]});
}
