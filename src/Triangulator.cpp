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

    cv::Mat Rt; cv::hconcat(Rd, td, Rt);          // 3×4, CV_64F
    return Kd * Rt;                               // CV_64F × CV_64F → CV_64F
}

void Triangulator::log_to_csv(const std::vector<rclcpp::Time>& stamps,
                              const std::vector<int>& cam_ids,
                              const std::string& path)
{
    if (stamps.empty() || stamps.size()!=cam_ids.size()) return;
    std::ofstream fout(path, std::ios::app);
    if(!fout.is_open()) return;

    auto[min_it,max_it]=std::minmax_element(stamps.begin(),stamps.end(),[](auto&a,auto&b){return a.nanoseconds()<b.nanoseconds();});
    int64_t min_ns=min_it->nanoseconds(), max_ns=max_it->nanoseconds();
    fout<<min_ns<<","<<max_ns<<","<<(max_ns-min_ns);
    for(std::size_t i=0;i<stamps.size();++i) fout<<","<<cam_ids[i]<<":"<<stamps[i].nanoseconds();
    fout<<"\n";
}

cv::Point3f Triangulator::linearTriangulateN(const std::vector<cv::Mat>& Ps,
                                             const std::vector<cv::Point2f>& xs)
{
    const int n = static_cast<int>(Ps.size());
    cv::Mat A(2*n, 4, CV_64F);
    for(int i=0;i<n;++i){
        cv::Mat P64; Ps[i].convertTo(P64, CV_64F);
        double u = xs[i].x, v = xs[i].y;
        P64.row(2).convertTo(P64.row(2), CV_64F);
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

void Triangulator::push2DKeypoint(int cam_id,const rclcpp::Time& stamp,const cv::Point2f& pt,const cv::Mat& K,const cv::Mat& R,const cv::Mat& t)
{
    // std::cout<<"__________________________1"<<std::endl;
    std::lock_guard<std::mutex> lk(mtx_);
    // std::cout<<cam_id<<"\t"<<stamp.nanoseconds()<<"\t"<<pt.x<<","<<pt.y<<"\t"<<K.size()<<","<<R.size()<<","<<t.size()<<std::endl;
    window_.push_back({cam_id, stamp, pt, makeProjection(K,R,t)});
    if(window_.size()>MAX_WINDOW_SIZE) window_.pop_front();
}

std::optional<cv::Point3f> Triangulator::triangulateIfReady()
{
    static float baseline_y=0.f; static bool baseline_set=false; static std::vector<float> baseline_samples;
    constexpr float PRESS_MIN=-1.0f, PRESS_MAX=1.0f; constexpr std::size_t BASELINE_SAMPLE_COUNT=10;

    std::lock_guard<std::mutex> lk(mtx_);
    if(window_.size()<required_cam_) return std::nullopt;

    std::vector<TimedKeypoint> buf(window_.begin(),window_.end());
    std::sort(buf.begin(),buf.end(),[](auto&a,auto&b){return a.stamp.nanoseconds()<b.stamp.nanoseconds();});

    for(std::size_t i=0;i<buf.size();++i){
        rclcpp::Time t0=buf[i].stamp; std::map<int,TimedKeypoint> group;
        for(std::size_t j=i;j<buf.size();++j){
            const auto& kp=buf[j]; int64_t dt=std::llabs((kp.stamp-t0).nanoseconds()); if(dt>MAX_SYNC_NS) break;
            if(!group.count(kp.cam_id) || dt < std::llabs((group[kp.cam_id].stamp - t0).nanoseconds())) group[kp.cam_id]=kp;
        }
        if(group.size()<required_cam_) continue;

        std::vector<cv::Mat> Ps; std::vector<cv::Point2f> xs; std::vector<rclcpp::Time> stamps; std::vector<int> ids;
        for(auto [id,ob]:group){ Ps.push_back(ob.P); xs.push_back(ob.keypoint); stamps.push_back(ob.stamp); ids.push_back(id);} log_to_csv(stamps,ids);
        const int n=Ps.size(); if(n<2) continue;

        const int maxIter=50; const double thresh=3.0; int bestInl=0; cv::Point3f bestX; std::uniform_int_distribution<int> uni(0,n-1);
        for(int it=0; it<maxIter; ++it){ int a=uni(rng_), b=uni(rng_); while(b==a) b=uni(rng_);
            cv::Point3f Xhyp=linearTriangulateN({Ps[a],Ps[b]},{xs[a],xs[b]}); int inl=0;
            for(int k=0;k<n;++k){ cv::Mat proj=Ps[k]*(cv::Mat_<double>(4,1)<<Xhyp.x,Xhyp.y,Xhyp.z,1.0);
                double u=proj.at<double>(0)/proj.at<double>(2), v=proj.at<double>(1)/proj.at<double>(2);
                if(cv::norm(cv::Point2f(u,v)-xs[k])<thresh) ++inl; }
            if(inl>bestInl){ bestInl=inl; bestX=Xhyp; }
        }
        if(bestInl<static_cast<int>(required_cam_)) continue;
        std::vector<cv::Mat> Ps_in; std::vector<cv::Point2f> xs_in;
        for(int k=0;k<n;++k){ cv::Mat proj=Ps[k]*(cv::Mat_<double>(4,1)<<bestX.x,bestX.y,bestX.z,1.0);
            double u=proj.at<double>(0)/proj.at<double>(2), v=proj.at<double>(1)/proj.at<double>(2);
            if(cv::norm(cv::Point2f(u,v)-xs[k])<thresh){ Ps_in.push_back(Ps[k]); xs_in.push_back(xs[k]); }}
        cv::Point3f X=refineLM(Ps_in,xs_in,linearTriangulateN(Ps_in,xs_in),5);
        static KalmanFilter1D kf_depth(1e-4f, 2e-2f);          // 参数可再调
        float y_cur = kf_depth.update(X.y);                    // 滤波后的 y
        // float y_cur=X.y;
        if(!baseline_set){ baseline_samples.push_back(y_cur); if(baseline_samples.size()>=BASELINE_SAMPLE_COUNT){ baseline_y=std::accumulate(baseline_samples.begin(),baseline_samples.end(),0.f)/baseline_samples.size(); baseline_set=true;} }
        else{
            float depth=baseline_y-y_cur; if(depth>=PRESS_MIN && depth<=PRESS_MAX){ double tsec=duration<double>(steady_clock::now().time_since_epoch()).count(); std::ofstream fout("/home/ljy/project/poseDetection/build/press_depth_log.csv",std::ios::app); if(fout.is_open()) fout<<std::fixed<<std::setprecision(9)<<tsec<<","<<X.x<<","<<y_cur<<","<<X.z<<","<<depth<<"\n"; }
        }
        return X;
    }
    return std::nullopt;
}
