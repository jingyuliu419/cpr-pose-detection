// include/config_loader.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <opencv2/core.hpp>

namespace config {

/**
 * @brief PoseConfig 用于加载 COCO keypoints 和类名配置
 */
class PoseConfig {
public:
    PoseConfig(const std::string& class_path, const std::string& keypoint_path);

    const std::vector<std::string>& getClasses() const;
    const std::map<int, std::string>& getKeypoints() const;
    const std::vector<std::pair<int, int>>& getSkeleton() const;

private:
    std::vector<std::string> classes_;
    std::map<int, std::string> keypoints_;
    std::vector<std::pair<int, int>> skeleton_;

    void loadClasses(const std::string& path);
    void loadKeypointsAndSkeleton(const std::string& path);
};

} // namespace config
