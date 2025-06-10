#include "ui_display_thread.h"
#include <opencv2/opencv.hpp>

void UiThreadFunc(std::vector<std::shared_ptr<video::VideoStream>> streams) {
    while (true) {
        for (auto& s : streams) {
            cv::Mat img;
            {
                std::lock_guard<std::mutex> g(s->imshow_mutex_);
                if (!s->display_queue_.empty()) {
                    img = s->display_queue_.front();
                    s->display_queue_.pop();
                }
            }
            if (!img.empty()) cv::imshow(s->name(), img);
        }
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }
}
