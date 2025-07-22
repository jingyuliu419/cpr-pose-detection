// Triangulator.h -------------------------------------------------------------
#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <vector>
#include "KalmanFilter1D.h"

class Triangulator {
public:
    explicit Triangulator(std::size_t required_cam = 2);  // 默认要求 2 路即可同步

    void push2DKeypoint(int cam_id,
                        const rclcpp::Time &stamp,
                        const cv::Point2f &pt,
                        const cv::Mat &K,
                        const cv::Mat &R,
                        const cv::Mat &t);

    std::optional<cv::Point3f> triangulateIfReady();

private:
    struct TimedKeypoint {
        int cam_id;
        rclcpp::Time stamp;
        cv::Point2f keypoint;
        cv::Mat P;                 ///< 3×4 投影矩阵 (CV_64F)
    };

    // helpers
    static cv::Mat makeProjection(const cv::Mat &K, const cv::Mat &R, const cv::Mat &t);
    static void    log_to_csv(const std::vector<rclcpp::Time> &stamps,
                              const std::vector<int> &cam_ids,
                              const std::string &path = "sync_error_log.csv");

    static cv::Point3f linearTriangulateN(const std::vector<cv::Mat> &Ps,
                                          const std::vector<cv::Point2f> &xs);
    static cv::Point3f refineLM(const std::vector<cv::Mat> &Ps,
                                const std::vector<cv::Point2f> &xs,
                                cv::Point3f X0,
                                int iters = 5);

    const std::size_t required_cam_;   ///< 同步所需最小相机数（≥2 推荐）
    // std::deque<TimedKeypoint> window_;
    std::unordered_map<int, std::deque<TimedKeypoint>> window_by_cam_;

    KalmanFilter1D kf_x_, kf_y_, kf_z_;  // 每个轴一个滤波器

    static constexpr std::size_t MAX_WINDOW_SIZE = 100;
    static constexpr int64_t MAX_WINDOW_DURATION_NS = 20000000;  // 300ms
    static constexpr int64_t     MAX_SYNC_NS     = 20000000; // 60 ms

    std::mutex mtx_;
    std::mt19937 rng_;
    bool kf_initialized_ = false;
};

#endif // TRIANGULATOR_H