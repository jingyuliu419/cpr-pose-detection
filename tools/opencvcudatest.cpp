#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <vector>

void testCudaColorConversion(int code, const std::string& name) {
    try {
        // 构造一个简单的测试图像
        cv::Mat input(480, 640, CV_8UC1, cv::Scalar(128));
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);
        cv::cuda::cvtColor(gpu_input, gpu_output, code);
        std::cout << "[OK] Supported: " << name << " (" << code << ")" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "[FAIL] Not supported: " << name << " (" << code << ")" << std::endl;
    }
}

int main() {
    std::vector<std::pair<int, std::string>> codes = {
        {cv::COLOR_YUV2BGR_NV12, "COLOR_YUV2BGR_NV12"},
        {cv::COLOR_YUV2RGB_NV12, "COLOR_YUV2RGB_NV12"},
        {cv::COLOR_YUV2BGR_I420, "COLOR_YUV2BGR_I420"},
        {cv::COLOR_BGR2GRAY,     "COLOR_BGR2GRAY"},
        {cv::COLOR_GRAY2BGR,     "COLOR_GRAY2BGR"},
        {cv::COLOR_BGR2YUV,      "COLOR_BGR2YUV"},     // CPU-only
        {cv::COLOR_YUV2BGR_YV12, "COLOR_YUV2BGR_YV12"}  // unsupported
    };

    for (const auto& code : codes) {
        testCudaColorConversion(code.first, code.second);
    }

    return 0;
}
