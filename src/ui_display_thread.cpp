// ui_display_thread.cpp -------------------------------------------------
#include "ui_display_thread.h"
#include <chrono>
#include "video_stream.h"
#include <thread>
namespace ui {

std::mutex              ui_mtx;
std::condition_variable ui_cv;

void UiThreadFunc(std::vector<std::shared_ptr<video::VideoStream>> streams,
                  cv::Size target_size)
{
    for (auto &s : streams) {
        if (!s) continue;
        cv::namedWindow(s->name(), cv::WINDOW_NORMAL);
        cv::resizeWindow(s->name(), target_size.width, target_size.height);
    }

    while (true) {
        // 等待推帧，50 ms 超时，保证 UI 流畅
        std::unique_lock<std::mutex> lk(ui_mtx);
        ui_cv.wait_for(lk, std::chrono::milliseconds(20));
        lk.unlock();

        bool drew = false;
        for (auto &s : streams) {
            if (!s) continue;
            cv::Mat img;
            {
                std::lock_guard<std::mutex> g(s->imshow_mutex_);
                while (s->display_queue_.size() > 1) s->display_queue_.pop();
                if (!s->display_queue_.empty()) {
                    img = std::move(s->display_queue_.front());
                    s->display_queue_.pop();
                }
            }
            if (img.empty()) continue;

            if (img.cols != target_size.width || img.rows != target_size.height)
                cv::resize(img, img, target_size, 0, 0, cv::INTER_AREA);
            cv::imshow(s->name(), img);
            drew = true;
        }

        if (!drew) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (int k=cv::waitKey(1); k==27 || k=='q') break; // ESC / q 退出
    }
}

} // namespace ui
