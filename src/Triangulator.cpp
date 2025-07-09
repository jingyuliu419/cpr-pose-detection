#include "Triangulator.h"
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <map>
#include <algorithm>
#include <iostream>

namespace {
cv::Mat makeProjection(const cv::Mat &K, const cv::Mat &R, const cv::Mat &t) {
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    return K * Rt;
}

void log_to_csv(const std::vector<rclcpp::Time>& stamps, const std::string& path = "sync_error_log.csv") {
    if (stamps.empty()) return;
    std::ofstream fout(path, std::ios::app);
    if (!fout.is_open()) return;

    auto minmax = std::minmax_element(stamps.begin(), stamps.end(),
        [](const auto& a, const auto& b) {
            return a.nanoseconds() < b.nanoseconds();
        });

    int64_t min_ns = minmax.first->nanoseconds();
    int64_t max_ns = minmax.second->nanoseconds();
    int64_t drift_ns = max_ns - min_ns;

    fout << min_ns << "," << max_ns << "," << drift_ns << "\n";
}

} // namespace

Triangulator::Triangulator(std::size_t required_cam)
    : required_cam_(required_cam) {}

void Triangulator::push2DKeypoint(int cam_id,
                                  const rclcpp::Time& stamp,
                                  const cv::Point2f& pt,
                                  const cv::Mat& K,
                                  const cv::Mat& R,
                                  const cv::Mat& t)
{
    std::lock_guard<std::mutex> lock(mtx_);
    cv::Mat P = makeProjection(K, R, t);

    // optional debug:
    // if (!window_.empty()) {
    //     auto drift = std::abs((stamp - window_.front().stamp).nanoseconds());
    //     std::cerr << "[Triangulator] cam_id = " << cam_id
    //               << ", time drift = " << drift * 1e-6 << " ms\n";
    // }

    window_.emplace_back(TimedKeypoint{cam_id, stamp, pt, P});

    if (window_.size() > MAX_WINDOW_SIZE)
        window_.pop_front();
}

std::optional<cv::Point3f> Triangulator::triangulateIfReady() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (window_.size() < required_cam_) return std::nullopt;

    for (size_t i = 0; i < window_.size(); ++i) {
        rclcpp::Time ref_time = window_[i].stamp;
        std::map<int, TimedKeypoint> group{{window_[i].cam_id, window_[i]}};

        for (size_t j = i + 1; j < window_.size(); ++j) {
            auto delta = std::abs((window_[j].stamp - ref_time).nanoseconds());
            if (delta < MAX_SYNC_NS) {
                group[window_[j].cam_id] = window_[j];
            }
        }

        if (group.size() >= required_cam_) {
            std::vector<int> cam_ids;
            for (const auto& [id, _] : group)
                cam_ids.push_back(id);
            std::sort(cam_ids.begin(), cam_ids.end());

            std::vector<rclcpp::Time> group_stamps;
            for (int id : cam_ids)
                group_stamps.push_back(group[id].stamp);
            log_to_csv(group_stamps);

            std::vector<cv::Mat> Ps;
            std::vector<cv::Point2f> pts;
            for (const auto& [_, ob] : group) {
                Ps.push_back(ob.P);
                pts.push_back(ob.keypoint);
            }

            if (Ps.size() < 2) return std::nullopt;

            cv::Mat pt_3d_homo;
            std::vector<cv::Point2f> pt1 = {pts[0]};
            std::vector<cv::Point2f> pt2 = {pts[1]};
            cv::triangulatePoints(Ps[0], Ps[1], pt1, pt2, pt_3d_homo);


            cv::Point3f pt_3d(
                pt_3d_homo.at<float>(0) / pt_3d_homo.at<float>(3),
                pt_3d_homo.at<float>(1) / pt_3d_homo.at<float>(3),
                pt_3d_homo.at<float>(2) / pt_3d_homo.at<float>(3));

            return pt_3d;
        }
    }

    return std::nullopt;
}
