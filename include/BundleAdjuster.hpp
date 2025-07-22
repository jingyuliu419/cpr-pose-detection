#pragma once

#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include <vector>

namespace vision {

class BundleAdjuster {
public:
    BundleAdjuster() = default;

    /**
     * @brief 进行单点 Bundle Adjustment 优化
     * @param Ps 投影矩阵（每帧一个）
     * @param xs 每帧对应的像素点观测值
     * @param init_3d 初始估计的三维点
     * @return 优化后的三维点
     */
    cv::Point3f optimize(const std::vector<cv::Mat>& Ps,
                         const std::vector<cv::Point2f>& xs,
                         const cv::Point3f& init_3d);

private:
    struct ReprojResidual {
        ReprojResidual(const cv::Mat& P, const cv::Point2f& obs)
            : obs_(obs) {
            P.convertTo(P_, CV_64F);
        }

        template <typename T>
        bool operator()(const T* const point, T* residuals) const {
            T X[4] = { point[0], point[1], point[2], T(1.0) };
            T proj[3] = { T(0), T(0), T(0) };

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j)
                    proj[i] += T(P_.at<double>(i, j)) * X[j];

            T u = proj[0] / proj[2];
            T v = proj[1] / proj[2];

            residuals[0] = u - T(obs_.x);
            residuals[1] = v - T(obs_.y);
            return true;
        }

        static ceres::CostFunction* Create(const cv::Mat& P, const cv::Point2f& obs) {
            return new ceres::AutoDiffCostFunction<ReprojResidual, 2, 3>(
                new ReprojResidual(P, obs));
        }

        cv::Mat P_;
        cv::Point2f obs_;
    };
};

inline cv::Point3f BundleAdjuster::optimize(const std::vector<cv::Mat>& Ps,
                                            const std::vector<cv::Point2f>& xs,
                                            const cv::Point3f& init_3d)
{
    double pt[3] = { init_3d.x, init_3d.y, init_3d.z };

    ceres::Problem problem;
    for (size_t i = 0; i < Ps.size(); ++i) {
        ceres::CostFunction* cost_function = ReprojResidual::Create(Ps[i], xs[i]);
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), pt);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return { static_cast<float>(pt[0]),
             static_cast<float>(pt[1]),
             static_cast<float>(pt[2]) };
}

}  // namespace vision
