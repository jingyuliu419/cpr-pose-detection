#include "ui_display_thread.h"
#include <opencv2/opencv.hpp>
#include <chrono>

namespace ui {

/* 定义全局同步对象 */
std::mutex              ui_mtx;
std::condition_variable ui_cv;

/* ---------------- UI 线程实现 ---------------- */
void UiThreadFunc(std::vector<std::shared_ptr<video::VideoStream>> streams,
                  cv::Size target_size)
{
    /* 创建窗口并设置统一大小 */
    for (auto& s : streams) {
        cv::namedWindow(s->name(), cv::WINDOW_NORMAL);
        cv::resizeWindow(s->name(), target_size.width, target_size.height);
    }

    while (true) {
        /* ① 等待 VideoStream 推帧：最长 50 ms 超时，避免 UI 掉帧 */
        std::unique_lock<std::mutex> lk(ui_mtx);
        ui_cv.wait_for(lk, std::chrono::milliseconds(50));
        lk.unlock();

        bool has_frame = false;

        /* ② 一次性批量刷新所有流 */
        for (auto& s : streams) {
            if (!s) {
                std::cerr << "[UI] Warning: Null VideoStream pointer!" << std::endl;
                continue;  // 忽略空指针
            }
            cv::Mat img;
            {
                std::lock_guard<std::mutex> g(s->imshow_mutex_);

                /* 只保留最新帧，丢弃历史 */
                while (s->display_queue_.size() > 1)
                    s->display_queue_.pop();

                if (!s->display_queue_.empty()) {
                    img = std::move(s->display_queue_.front());
                    s->display_queue_.pop();
                }
            }

            if (!img.empty()) {
                has_frame = true;

                /* ③ 如需要统一分辨率，做缩放（INTER_AREA 性能最佳） */
                if (img.cols != target_size.width ||
                    img.rows != target_size.height)
                {
                    cv::resize(img, img, target_size, 0, 0,
                               cv::INTER_AREA);
                }
                cv::imshow(s->name(), img);
            }
        }

        /* ④ 若这一轮完全无帧，则小睡 5 ms，降 CPU 占用 */
        if (!has_frame)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        /* ⑤ 处理键盘事件 */
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;      // Esc / q 退出
    }
}

}  // namespace ui
