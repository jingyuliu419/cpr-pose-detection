#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

int main() {
    std::cout << "CUDA devices: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

#if CV_VERSION_MAJOR >= 4
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
#endif

    return 0;
}
