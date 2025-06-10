#pragma once
#include <opencv2/core.hpp>

void nv12_to_bgr_gpu(const uint8_t* nv12_data, cv::Mat& dst, int width, int height);
