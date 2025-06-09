#include "ros2_time_sync.h"
#include <iostream>
#include <rclcpp/executors/multi_threaded_executor.hpp>

namespace sync {

TimeSyncNode::TimeSyncNode()
    : node_(std::make_shared<rclcpp::Node>("time_sync_node")),
      running_(false) {}

TimeSyncNode::~TimeSyncNode() {
    stop();
}

void TimeSyncNode::start() {
    running_ = true;
    // 启动 executor 线程
    executor_thread_ = std::thread([this]() {
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node_);
        executor.spin();
    });

    // 启动主同步逻辑线程
    sync_thread_ = std::thread(&TimeSyncNode::loop, this);
}

void TimeSyncNode::stop() {
    running_ = false;
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
    if (node_->get_node_base_interface()->get_context()->is_valid()) {
        rclcpp::shutdown();
    }
    if (executor_thread_.joinable()) {
        executor_thread_.join();
    }
}

void TimeSyncNode::loop() {
    while (running_) {
        rclcpp::Time now = node_->now();
        std::cout << "ROS2 Time: " << now.seconds() << "s\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

std::shared_ptr<rclcpp::Node> TimeSyncNode::get_node() const {
    return node_;
}

} // namespace sync
