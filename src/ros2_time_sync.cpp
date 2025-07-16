#include "ros2_time_sync.h"
#include <iostream>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include "std_msgs/msg/int64.hpp"  // 添加这个头文件

namespace timesync {

TimeSyncNode::TimeSyncNode()
    : node_(std::make_shared<rclcpp::Node>("time_sync_node")),
      running_(false),
      time_publisher_(node_->create_publisher<std_msgs::msg::Int64>("time_sync_topic", 10)) {}  // 初始化发布者

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
        // 获取当前的时间（精确到纳秒）
        rclcpp::Time now = node_->now(); // 默认精度为纳秒

        // 获取当前时间的纳秒数
        int64_t nanoseconds = now.nanoseconds();

        // 创建消息并发布
        auto time_msg = std::make_shared<std_msgs::msg::Int64>(); // 使用 Int64 来存储纳秒时间
        time_msg->data = nanoseconds;  // 发布当前的 ROS2 时间（纳秒）

        time_publisher_->publish(*time_msg);  // 发布到 ROS2 网络

        // 输出当前的时间戳，确保时间同步的高精度
        // std::cout << "Time sync published: " << nanoseconds << " ns" << std::endl;


        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    }
}

std::shared_ptr<rclcpp::Node> TimeSyncNode::get_node() const {
    return node_;
}

} // namespace timesync
