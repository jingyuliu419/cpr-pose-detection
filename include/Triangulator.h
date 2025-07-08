// Triangulator.h
#pragma once

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <optional>
#include <map>
#include <vector>

/**
 * @brief 轻量级多相机关键点三角测距器
 */
class Triangulator {
public:
    /**
     * @param required_cam 至少多少个相机观测后才计算（默认 2）
     */
    explicit Triangulator(std::size_t required_cam = 2);

    /**
     * 推送单帧关键点
     * @param cam_id   相机 ID（仅用于调试，可不使用）
     * @param stamp    同步时间戳（确保不同相机帧对应同一时刻）
     * @param pt       图像坐标（去畸变后）
     * @param K        相机内参矩阵
     * @param R        相机旋转矩阵 (world→cam)
     * @param t        相机平移向量 (world→cam)
     */
    void push2DKeypoint(int cam_id,
                        const rclcpp::Time &stamp,
                        const cv::Point2f &pt,
                        const cv::Mat &K,
                        const cv::Mat &R,
                        const cv::Mat &t);

    /**
     * 若观测数量满足要求则立即三角测距
     * @return 3D 点（世界坐标系），否则 std::nullopt
     */
    std::optional<cv::Point3f> triangulateIfReady();

private:
    struct Observation {
        cv::Mat P;          //!< 投影矩阵 K*[R|t]
        cv::Point2f pt;     //!< 图像点
    };

    using ObsVec = std::vector<Observation>;

    std::map<rclcpp::Time, ObsVec> obs_; //!< 按时间戳聚合观测
    std::size_t required_cam_;
};