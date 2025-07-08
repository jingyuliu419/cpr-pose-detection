#pragma once
#include "video_stream.h"
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace ui {

/* 全局条件变量：VideoStream 推新帧时唤醒 UI 线程 */
extern std::mutex              ui_mtx;
extern std::condition_variable ui_cv;
inline void notifyUI() { ui_cv.notify_one(); }

/* UI 渲染线程主体 */
void UiThreadFunc(std::vector<std::shared_ptr<video::VideoStream>> streams,
                  cv::Size target_size = {1280, 720});   // 默认窗口大小

}  // namespace ui
