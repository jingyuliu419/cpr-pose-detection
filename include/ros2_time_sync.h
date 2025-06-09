#pragma once
#include<rclcpp/rclcpp.hpp>
#include<thread>
#include<atomic>
namespace sync{
    class TimeSyncNode{
        public:
            TimeSyncNode();
            ~TimeSyncNode();
            void start();
            void stop();
            std::shared_ptr<rclcpp::Node> get_node() const;
        private:
            void loop();
            std::shared_ptr<rclcpp::Node>node_;
            std::thread thread_;
            std::atomic<bool>running_;
    };
}//namespace sync