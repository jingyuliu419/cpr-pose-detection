#pragma once
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <atomic>
#include "std_msgs/msg/int64.hpp"  // 添加这个头文件

namespace timesync {

class TimeSyncNode {
public:
    TimeSyncNode();
    ~TimeSyncNode();

    void start();
    void stop();

    std::shared_ptr<rclcpp::Node> get_node() const;

private:
    void loop();

    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Int64>> time_publisher_;  // 添加 time_publisher_ 声明
    std::thread sync_thread_;          // 时间同步线程
    std::thread executor_thread_;      // Executor spin线程
    std::atomic<bool> running_;
    rclcpp::Time synchronized_timestamp_;  // 添加 synchronized_timestamp_ 变量
};

} // namespace timesync
