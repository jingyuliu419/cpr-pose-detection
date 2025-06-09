#include "config_loader.h"
#include <iostream>  
#include <opencv2/core.hpp>
#include <stdexcept>

namespace config {

PoseConfig::PoseConfig(const std::string& class_path, const std::string& keypoint_path) {
    loadClasses(class_path);
    loadKeypointsAndSkeleton(keypoint_path);
}

const std::vector<std::string>& PoseConfig::getClasses() const {
    return classes_;
}

const std::map<int, std::string>& PoseConfig::getKeypoints() const {
    return keypoints_;
}

const std::vector<std::pair<int, int>>& PoseConfig::getSkeleton() const {
    return skeleton_;
}

void PoseConfig::loadClasses(const std::string& path) {
    std::cout << "[DEBUG] Loading classes from: " << path << std::endl;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ERROR] Cannot open classes YAML: " << path << std::endl;
        throw std::runtime_error("Cannot open classes YAML: " + path);
    }

    cv::FileNode n = fs["names"];
    if (n.empty()) {
        std::cerr << "[ERROR] No 'names' node in: " << path << std::endl;
        throw std::runtime_error("No 'names' node found in classes YAML: " + path);
    }
    if (n.type() != cv::FileNode::SEQ) {
        std::cerr << "[ERROR] 'names' node is not a sequence." << std::endl;
        throw std::runtime_error("Invalid format in classes.yaml");
    }
   
    for (const auto& node : n) {
        std::string cls;
        node >> cls;
        classes_.push_back(cls);
    }
    fs.release();
    // std::cout << "[DEBUG] Loaded " << classes_.size() << " classes." << std::endl;
    // for (const auto& c : classes_)
    //     std::cout << " - " << c << std::endl;

}


void PoseConfig::loadKeypointsAndSkeleton(const std::string& path) {
    std::cout << "[DEBUG] Loading keypoints from: " << path << std::endl;


    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ERROR] Cannot open keypoint YAML: " << path << std::endl;
        throw std::runtime_error("Cannot open keypoint YAML: " + path);
    }

    // cv::FileNode kp_node = fs["keypoints"];
    // if (kp_node.empty()) {
    //     std::cerr << "[ERROR] No 'keypoints' node in: " << path << std::endl;
    //     throw std::runtime_error("Missing 'keypoints' node");
    // }

    // for (const auto& kv : kp_node) {
    //     int id = std::stoi(kv.name());
    //     std::string name;
    //     kv >> name;
    //     keypoints_[id] = name;
    // }

    cv::FileNode sk_node = fs["skeleton"];
    if (sk_node.empty()) {
        std::cerr << "[ERROR] No 'skeleton' node in: " << path << std::endl;
        throw std::runtime_error("Missing 'skeleton' node");
    }

    for (const auto& line : sk_node) {
        if (line.size() != 2) continue;
        int i = (int)line[0], j = (int)line[1];
        skeleton_.emplace_back(i, j);
    }

    fs.release();
    // std::cout << "[DEBUG] Loaded " << keypoints_.size() << " keypoints." << std::endl;
    // for (const auto& [id, name] : keypoints_)
    //     std::cout << " - " << id << ": " << name << std::endl;

    // std::cout << "[DEBUG] Loaded " << skeleton_.size() << " skeleton bones." << std::endl;
    // for (const auto& [i, j] : skeleton_)
    //     std::cout << " - [" << i << ", " << j << "]" << std::endl;

}

}  // namespace config
