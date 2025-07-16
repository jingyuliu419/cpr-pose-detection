// Triangulator.cpp -----------------------------------------------------------
#include "Triangulator.h"
#include "KalmanFilter1D.h"
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <iomanip>

using namespace std::chrono;

// ---------------- helpers ----------------
cv::Mat Triangulator::makeProjection(const cv::Mat &K,const cv::Mat &R,const cv::Mat &t)
{
    cv::Mat Kd,Rd,td;
    K.convertTo(Kd, CV_64F);
    R.convertTo(Rd, CV_64F);
    t.convertTo(td, CV_64F);

    cv::Mat Rt; cv::hconcat(Rd, td, Rt);
    return Kd * Rt;
}

void Triangulator::log_to_csv(const std::vector<rclcpp::Time>& stamps,
                              const std::vector<int>& cam_ids,
                              const std::string& path)
{
    if (stamps.empty() || stamps.size() != cam_ids.size()) return;

    std::ofstream fout(path, std::ios::app);
    if (!fout.is_open()) return;

    // 写 CSV 表头（仅第一次写时建议使用，可额外加判断避免重复）
    static bool header_written = false;
    if (!header_written) {
        fout << "baseline_ns";
        for (std::size_t i = 0; i < cam_ids.size(); ++i)
            fout << ",cam" << cam_ids[i];
        fout << "\n";
        header_written = true;
    }

    // baseline_ns 为最小时间戳（单位 ns）
    auto[min_it, _] = std::minmax_element(stamps.begin(), stamps.end(),
                                          [](auto& a, auto& b) {
                                              return a.nanoseconds() < b.nanoseconds();
                                          });
    int64_t baseline_ns = min_it->nanoseconds();

    fout << baseline_ns;
    for (std::size_t i = 0; i < stamps.size(); ++i)
        fout << "," << stamps[i].nanoseconds();
    fout << "\n";
}


cv::Point3f Triangulator::linearTriangulateN(const std::vector<cv::Mat>& Ps,
                                             const std::vector<cv::Point2f>& xs)
{
    const int n = static_cast<int>(Ps.size());
    cv::Mat A(2*n, 4, CV_64F);
    for(int i=0;i<n;++i){
        cv::Mat P64; Ps[i].convertTo(P64, CV_64F);
        double u = xs[i].x, v = xs[i].y;
        A.row(2*i  ) = u*P64.row(2) - P64.row(0);
        A.row(2*i+1) = v*P64.row(2) - P64.row(1);
    }
    cv::SVD svd(A, cv::SVD::FULL_UV);
    cv::Mat Xh = svd.vt.row(3).t();
    Xh /= Xh.at<double>(3);
    return {static_cast<float>(Xh.at<double>(0)),
            static_cast<float>(Xh.at<double>(1)),
            static_cast<float>(Xh.at<double>(2))};
}

cv::Point3f Triangulator::refineLM(const std::vector<cv::Mat>& Ps,const std::vector<cv::Point2f>& xs,cv::Point3f X0,int iters)
{
    cv::Mat X=(cv::Mat_<double>(3,1)<<X0.x,X0.y,X0.z);
    for(int it=0;it<iters;++it){
        cv::Mat J(2*Ps.size(),3,CV_64F), r(2*Ps.size(),1,CV_64F);
        for(std::size_t i=0;i<Ps.size();++i){
            cv::Mat P64; Ps[i].convertTo(P64,CV_64F);
            cv::Mat proj = P64 * (cv::Mat_<double>(4,1)<<X.at<double>(0),X.at<double>(1),X.at<double>(2),1.0);
            double u = proj.at<double>(0)/proj.at<double>(2);
            double v = proj.at<double>(1)/proj.at<double>(2);
            r.at<double>(2*i  ) = xs[i].x - u;
            r.at<double>(2*i+1) = xs[i].y - v;
            const double eps=1e-4;
            for(int d=0; d<3; ++d){
                cv::Mat Xeps=X.clone(); Xeps.at<double>(d)+=eps;
                cv::Mat proj2 = P64 * (cv::Mat_<double>(4,1)<<Xeps.at<double>(0),Xeps.at<double>(1),Xeps.at<double>(2),1.0);
                double u2=proj2.at<double>(0)/proj2.at<double>(2);
                double v2=proj2.at<double>(1)/proj2.at<double>(2);
                J.at<double>(2*i  ,d) = (xs[i].x-u2)/eps;
                J.at<double>(2*i+1,d) = (xs[i].y-v2)/eps;
            }
        }
        cv::Mat H=J.t()*J, g=J.t()*r, dX; cv::solve(H,g,dX,cv::DECOMP_SVD);
        X += dX; if(cv::norm(dX)<1e-6) break;
    }
    return {static_cast<float>(X.at<double>(0)),static_cast<float>(X.at<double>(1)),static_cast<float>(X.at<double>(2))};
}

Triangulator::Triangulator(std::size_t required_cam):required_cam_(required_cam),rng_(std::random_device{}()){}

// void Triangulator::push2DKeypoint(int cam_id,const rclcpp::Time& stamp,const cv::Point2f& pt,const cv::Mat& K,const cv::Mat& R,const cv::Mat& t)
// {
//     std::lock_guard<std::mutex> lk(mtx_);
//     // 剔除时间差小于10ms 且像素距离小于2的近似重复帧
//     // if (!window_.empty()) {
//     //     const auto& last = window_.back();
//     //     double dt = std::abs((stamp - last.stamp).nanoseconds()) / 1e6;
//     //     double dist = cv::norm(pt - last.keypoint);
//     //     if (dt < 10 && dist < 2.0) return;
//     // }
//     window_.push_back({cam_id, stamp, pt, makeProjection(K,R,t)});
//     if(window_.size()>MAX_WINDOW_SIZE) window_.pop_front();
// }
void Triangulator::push2DKeypoint(int cam_id,
                                  const rclcpp::Time& stamp,
                                  const cv::Point2f& pt,
                                  const cv::Mat& K,
                                  const cv::Mat& R,
                                  const cv::Mat& t)
{
    std::lock_guard<std::mutex> lk(mtx_);
    TimedKeypoint kp{cam_id, stamp, pt, makeProjection(K, R, t)};
    auto& cam_queue = window_by_cam_[cam_id];

    // 近重复剔除
    if (!cam_queue.empty()) {
        const auto& last = cam_queue.back();
        if (std::abs((stamp - last.stamp).nanoseconds() * 1e-6) < 10.0 &&
            cv::norm(pt - last.keypoint) < 2.0)
            return;
    }

    cam_queue.push_back(kp);

    // 清理超时帧
    while (!cam_queue.empty() &&
           (stamp.nanoseconds() - cam_queue.front().stamp.nanoseconds()) > MAX_WINDOW_DURATION_NS)
        cam_queue.pop_front();
}

std::optional<cv::Point3f> Triangulator::triangulateIfReady()
{
    std::lock_guard<std::mutex> lk(mtx_);

    // Step 1: 聚合所有相机视角的缓存数据
    std::vector<TimedKeypoint> buf;
    for (auto& [id, queue] : window_by_cam_) {
        buf.insert(buf.end(), queue.begin(), queue.end());
    }

    if (buf.size() < required_cam_) return std::nullopt;

    // Step 2: 按时间排序
    std::sort(buf.begin(), buf.end(),
              [](auto& a, auto& b) {
                  return a.stamp.nanoseconds() < b.stamp.nanoseconds();
              });

    for (std::size_t i = 0; i < buf.size(); ++i) {
        rclcpp::Time t0 = buf[i].stamp;
        std::map<int, TimedKeypoint> group;

        // Step 3: 在时间窗口内聚合每个视角的最近一帧
        for (const auto& kp : buf) {
            int64_t dt = std::llabs((kp.stamp - t0).nanoseconds());
            if (dt > MAX_SYNC_NS) continue;

            // 只保留每个相机最接近 t0 的关键点
            if (!group.count(kp.cam_id) ||
                dt < std::llabs((group[kp.cam_id].stamp - t0).nanoseconds())) {
                group[kp.cam_id] = kp;
            }
        }

        if (group.size() < required_cam_) continue;

        // Step 4: 拆出用于三角测量的 Ps, xs
        std::vector<cv::Mat> Ps;
        std::vector<cv::Point2f> xs;
        std::vector<rclcpp::Time> stamps;
        std::vector<int> ids;

        for (const auto& [id, ob] : group) {
            Ps.push_back(ob.P);
            xs.push_back(ob.keypoint);
            stamps.push_back(ob.stamp);
            ids.push_back(id);
        }

        log_to_csv(stamps, ids);

        const int n = Ps.size();
        if (n < 2) continue;

        const int maxIter = 50;
        const double thresh = 3.0;
        int bestInl = 0;
        cv::Point3f bestX;
        std::uniform_int_distribution<int> uni(0, n - 1);

        // Step 5: RANSAC 随机采样
        for (int it = 0; it < maxIter; ++it) {
            int a = uni(rng_), b = uni(rng_);
            while (b == a) b = uni(rng_);
            cv::Point3f Xhyp = linearTriangulateN({Ps[a], Ps[b]}, {xs[a], xs[b]});

            int inl = 0;
            for (int k = 0; k < n; ++k) {
                cv::Mat proj = Ps[k] * (cv::Mat_<double>(4, 1) << Xhyp.x, Xhyp.y, Xhyp.z, 1.0);
                double u = proj.at<double>(0) / proj.at<double>(2);
                double v = proj.at<double>(1) / proj.at<double>(2);
                if (cv::norm(cv::Point2f(u, v) - xs[k]) < thresh) ++inl;
            }
            if (inl > bestInl) {
                bestInl = inl;
                bestX = Xhyp;
            }
        }

        // Step 6: 使用内点集 refine
        std::vector<cv::Mat> Ps_in;
        std::vector<cv::Point2f> xs_in;
        for (int k = 0; k < n; ++k) {
            cv::Mat proj = Ps[k] * (cv::Mat_<double>(4, 1) << bestX.x, bestX.y, bestX.z, 1.0);
            double u = proj.at<double>(0) / proj.at<double>(2);
            double v = proj.at<double>(1) / proj.at<double>(2);
            if (cv::norm(cv::Point2f(u, v) - xs[k]) < thresh) {
                Ps_in.push_back(Ps[k]);
                xs_in.push_back(xs[k]);
            }
        }

        if (Ps_in.size() < 2) continue;

        RCLCPP_INFO(rclcpp::get_logger("Triangulator"),
                    "Triangulated with %ld cameras, inliers = %ld",
                    Ps.size(), Ps_in.size());

        // Step 7: refine + 滤波
        cv::Point3f rawX = refineLM(Ps_in, xs_in, linearTriangulateN(Ps_in, xs_in), 5);
        cv::Point3f X;
        X.x = kf_x_.update(rawX.x);
        X.y = kf_y_.update(rawX.y);
        X.z = kf_z_.update(rawX.z);

        // Step 8: 记录输出
        auto now = std::chrono::system_clock::now();
        int64_t ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        std::ofstream fout("/home/ljy/project/poseDetection/build/triangulated_xyz_log.csv", std::ios::app);
        if (fout.is_open()) {
            fout << ms_since_epoch << ","
                 << std::fixed << std::setprecision(6)
                 << X.x << "," << X.y << "," << X.z << "\n";
        }

        return X;
    }

    return std::nullopt;
}



