// ==============================
// src/inferenceLoop_with_online_baseline.cpp 关键修改
// ==============================
#include "inferenceLoop_with_online_baseline.h"
#include <algorithm>
#include <limits>
#include <cmath>

MovingMean::MovingMean(size_t w) : w_(std::max<size_t>(1, w)), sum_(0.0) {}
double MovingMean::add(double x){ buf_.push_back(x); sum_+=x; if(buf_.size()>w_){ sum_-=buf_.front(); buf_.pop_front(); } return sum_/double(buf_.size()); }
void   MovingMean::clear(){ buf_.clear(); sum_=0.0; }
size_t MovingMean::size() const { return buf_.size(); }

FlatWindow::FlatWindow(size_t w, double threshold) : w_(std::max<size_t>(1, w)), th_(threshold), sum_(0.0), count_(0) {}
std::tuple<bool,double,double> FlatWindow::push(double x){
    const size_t idx = count_++;
    dq_.emplace_back(x, idx); sum_ += x;
    while(!minq_.empty() && minq_.back().first >= x) minq_.pop_back();
    while(!maxq_.empty() && maxq_.back().first <= x) maxq_.pop_back();
    minq_.emplace_back(x, idx); maxq_.emplace_back(x, idx);
    const size_t oldest_allowed = (idx >= w_-1) ? (idx-(w_-1)) : 0;
    while(!dq_.empty()   && dq_.front().second   < oldest_allowed) popFront();
    while(!minq_.empty() && minq_.front().second < oldest_allowed) minq_.pop_front();
    while(!maxq_.empty() && maxq_.front().second < oldest_allowed) maxq_.pop_front();
    const bool full = dq_.size() == w_;
    const double range = full ? (maxq_.front().first - minq_.front().first) : std::numeric_limits<double>::infinity();
    const double mean  = full ? (sum_/double(dq_.size())) : 0.0;
    return {full, range, mean};
}
void FlatWindow::reset(){ dq_.clear(); minq_.clear(); maxq_.clear(); sum_=0.0; count_=0; }
void FlatWindow::popFront(){ if(dq_.empty()) return; sum_ -= dq_.front().first; dq_.pop_front(); }

// --- BaselineEstimator ---
BaselineEstimator::BaselineEstimator()
: depth_smooth(3), flat(300, 0.03), rel_cm_smooth(15),
  baseline_found(false), baseline_m(0.0), last_range_mm(0.0),
  baseline_start_ts(0), baseline_end_ts(0) {}

void BaselineEstimator::reset(){
    baseline_found=false; baseline_m=0.0; last_range_mm=0.0;
    baseline_start_ts=baseline_end_ts=0;
    depth_smooth.clear(); flat.reset(); rel_cm_smooth.clear(); tsq.clear();
}

// returns (has_output, rel_depth_cm, rel_depth_cm_smooth)
std::tuple<bool,double,double> BaselineEstimator::ingest(uint64_t ts_ms, double est_depth_m){
    // keep ts ring buffer aligned to flat.window()
    tsq.push_back(ts_ms);
    if (tsq.size() > flat.window()) tsq.pop_front();

    const double d_s = depth_smooth.add(est_depth_m);

    if (!baseline_found){
        auto [full, range, mean] = flat.push(d_s);
        if (full && range < flat.threshold()){
            baseline_found  = true;
            baseline_m      = mean;
            baseline_end_ts = tsq.back();
            baseline_start_ts = tsq.front();              // <-- 正确的起止时间
            last_range_mm   = range * 1000.0;
        }
        return {false, 0.0, 0.0};
    }

    const double rel_cm   = std::abs(d_s - baseline_m) * 100.0;
    const double rel_cm_s = rel_cm_smooth.add(rel_cm);
    return {true, rel_cm, rel_cm_s};
}
