// ==============================
// include/inferenceLoop_with_online_baseline.h 仅追加成员声明
// ==============================
#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <tuple>
#include <cstdint>

class MovingMean {
public:
    explicit MovingMean(size_t w);
    double add(double x);
    void clear();
    size_t size() const;
private:
    size_t w_;
    std::deque<double> buf_;
    double sum_;
};

class FlatWindow {
public:
    FlatWindow(size_t w, double threshold);
    std::tuple<bool,double,double> push(double x);
    void reset();

    inline size_t  window()    const { return w_; }
    inline double  threshold() const { return th_; }

private:
    void popFront();
    size_t w_;
    double th_;
    std::deque<std::pair<double,size_t>> dq_, minq_, maxq_;
    double sum_;
    size_t count_;
};

struct BaselineEstimator {
    MovingMean depth_smooth;   // w=3
    FlatWindow flat;           // w=300, th=0.03 (m)
    MovingMean rel_cm_smooth;  // w=5

    // === add: track timestamps of the last `flat.window()` samples ===
    std::deque<uint64_t> tsq;  // ms, ring buffer aligned to FlatWindow size

    bool baseline_found;
    double baseline_m;
    double last_range_mm;
    uint64_t baseline_start_ts, baseline_end_ts;

    BaselineEstimator();
    void reset();
    std::tuple<bool,double,double> ingest(uint64_t ts_ms, double est_depth_m);
};
