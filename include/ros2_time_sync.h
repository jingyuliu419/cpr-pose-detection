#pragma once
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <atomic>

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
    std::thread sync_thread_;          // 时间同步线程
    std::thread executor_thread_;      // Executor spin线程
    std::atomic<bool> running_;
};

} // namespace timesync
