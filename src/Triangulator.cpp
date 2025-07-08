// -----------------------------------------------------------------------------
// Triangulator.cpp
// -----------------------------------------------------------------------------

#include "Triangulator.h"

namespace {
inline cv::Mat makeProjection(const cv::Mat &K,
                              const cv::Mat &R,
                              const cv::Mat &t) {
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);       // [R|t] 3×4
    return K * Rt;               // 3×4
}
} // namespace

Triangulator::Triangulator(std::size_t required_cam)
    : required_cam_(required_cam) {}

void Triangulator::push2DKeypoint(int /*cam_id*/,
                                  const rclcpp::Time &stamp,
                                  const cv::Point2f &pt,
                                  const cv::Mat &K,
                                  const cv::Mat &R,
                                  const cv::Mat &t) {
    Observation ob{makeProjection(K, R, t), pt};
    obs_[stamp].emplace_back(std::move(ob));
}

std::optional<cv::Point3f> Triangulator::triangulateIfReady() {
    for (auto it = obs_.begin(); it != obs_.end();) {
        ObsVec &vec = it->second;
        if (vec.size() >= required_cam_) {
            // 取前两次观测进行线性三角测距
            cv::Mat pt1(2, 1, CV_64F), pt2(2, 1, CV_64F);
            pt1.at<double>(0) = vec[0].pt.x;
            pt1.at<double>(1) = vec[0].pt.y;
            pt2.at<double>(0) = vec[1].pt.x;
            pt2.at<double>(1) = vec[1].pt.y;

            cv::Mat X_h;
            cv::triangulatePoints(vec[0].P, vec[1].P, pt1, pt2, X_h);

            if (X_h.cols == 0) {  // 失败，丢弃并继续
                it = obs_.erase(it);
                continue;
            }
            cv::Mat X = X_h.col(0);
            X /= X.at<double>(3); // 齐次归一化

            cv::Point3f res{
                static_cast<float>(X.at<double>(0)),
                static_cast<float>(X.at<double>(1)),
                static_cast<float>(X.at<double>(2))};

            obs_.erase(it);
            return res;
        }
        ++it;  // 继续检查下一时间戳
    }
    return std::nullopt;
}
