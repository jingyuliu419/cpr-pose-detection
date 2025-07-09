#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <optional>
#include <mutex>

class Triangulator {
public:
    Triangulator(std::size_t required_cam);
    void push2DKeypoint(int cam_id,
                        const rclcpp::Time& stamp,
                        const cv::Point2f& pt,
                        const cv::Mat& K,
                        const cv::Mat& R,
                        const cv::Mat& t);

    std::optional<cv::Point3f> triangulateIfReady();

private:
    struct TimedKeypoint {
        int cam_id;
        rclcpp::Time stamp;
        cv::Point2f keypoint;
        cv::Mat P;
    };

    static constexpr size_t MAX_WINDOW_SIZE = 50;
    static constexpr int64_t MAX_SYNC_NS = 30000000;  // 30ms

    std::deque<TimedKeypoint> window_;
    std::size_t required_cam_;
    std::mutex mtx_;
};

#endif
