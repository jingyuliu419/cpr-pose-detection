#include "ros2_time_sync.h"
#include<iostream>
namespace sync{
    TimeSyncNode::TimeSyncNode()
        : node_(std::make_shared<rclcpp::Node>("time_sync_node")), running_(false) {}

    TimeSyncNode::~TimeSyncNode() {
        stop();
    }

    void TimeSyncNode::start() {
        running_ = true;
        thread_ = std::thread(&TimeSyncNode::loop, this);
    }

    void TimeSyncNode::stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void TimeSyncNode::loop() {
        while(running_){
            rclcpp::Time now=node_->now();
            std::cout<<"ROS2 Time: "<<now.seconds()<<"s\n";
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
    std::shared_ptr<rclcpp::Node> TimeSyncNode::get_node() const {
        return node_;
    }
}