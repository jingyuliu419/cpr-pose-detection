#include "multi_cam_calibrator.h"
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>      // ✅ 关键：修复 cv::cvtColor 错误
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <iostream>
#include <map>  
#ifndef USE_STEREO_CHARUCO   // 0 = 全版本兼容
#define USE_STEREO_CHARUCO 0
#endif

namespace charuco {

using std::vector;

/* ---------- ctor ---------- */
MultiCamCalibrator::MultiCamCalibrator(
        vector<std::shared_ptr<CameraManager>>& cams,
        cv::Ptr<cv::aruco::CharucoBoard> board)
    : cams_(cams), board_(std::move(board))
{
    buf_.resize(cams_.size());
}

/* ---------- Charuco 检测 ---------- */
bool MultiCamCalibrator::detectCharuco(const cv::Mat& frame, Shot2D& out)
{
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    auto dict = board_->dictionary;
    cv::aruco::detectMarkers(frame, dict, corners, ids);

    if (ids.empty()) {
        std::cout << "[WARN] detectMarkers() failed.\n";
        return false;
    }

    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    cv::aruco::interpolateCornersCharuco(corners, ids, gray, board_,
                                         out.imgPts, out.ids);

    return !out.imgPts.empty();
}


/* ---------- 缓冲同步帧 ---------- */
void MultiCamCalibrator::pushSyncFrames(const std::vector<cv::Mat>& frames)
{
    for (size_t i = 0; i < cams_.size(); ++i) {
        Shot2D shot;
        bool success = detectCharuco(frames[i], shot);

        if (!success) {
            std::cout << "[WARN] Camera #" << i << " failed to detect Charuco corners\n";
        } else {
            std::cout << "[INFO] Camera #" << i << " detected " << shot.ids.size() << " corners\n";
        }

        if (buf_.size() <= i) buf_.emplace_back();
        buf_[i].shots.push_back(shot);
        buf_[i].imgSize = frames[i].size();
    }

    std::cout << "[INFO] Captured sync group #" << buf_[0].shots.size() << "\n";
}


/* ---------- 求外参 ---------- */
static cv::Mat vecAverage(const vector<cv::Mat>& vecs)
{
    cv::Mat mean = cv::Mat::zeros(vecs[0].size(), CV_64F);
    for (auto& v:vecs) mean += v;
    mean /= (double)vecs.size();
    return mean;
}

vector<RigExtrinsics>MultiCamCalibrator::solveAndSave(const std::string& rig_yaml)
{   
    for (size_t i = 0; i < buf_.size(); ++i) {
        std::cout << "[DEBUG] Camera #" << i << " collected " << buf_[i].shots.size() << " frames.\n";
        if (buf_[i].shots.size() < 10)
            std::cerr << "[ERROR] Camera #" << i << " has too few valid shots, calibration may fail!\n";
    }
    const size_t N = cams_.size();
    vector<RigExtrinsics> exts(N);
    exts[0].R = cv::Mat::eye(3,3,CV_64F);   // master
    exts[0].t = cv::Mat::zeros(3,1,CV_64F);
    if (buf_[0].shots.empty()) {
        std::cerr << "[ERROR] No valid shots in master camera.\n";
        return exts;  // 返回空外参
    }
    /* 预取棋盘 3D 角点（Charuco 内部给的是 float） */
    vector<cv::Point3f> objCorners;
    for (auto& p : board_->chessboardCorners)
        objCorners.push_back(p);
    
    for (size_t cam=1; cam<N; ++cam)
    {
        const cv::Mat& Ki = cams_[cam]->K();
        cv::Mat Di = cams_[cam]->getDist();   // 新增 getDist()

        vector<cv::Mat> rvecs, tvecs;

        size_t shots = std::min(buf_[0].shots.size(), buf_[cam].shots.size());
        for (size_t k=0;k<shots;++k)
        {
            /* 根据 id 对齐同一 3D 点 */

            std::map<int,cv::Point2f> map0;
            for (size_t i=0;i<buf_[0].shots[k].ids.size();++i)
                map0[ buf_[0].shots[k].ids[i] ] = buf_[0].shots[k].imgPts[i];

            vector<cv::Point3f> obj;
            vector<cv::Point2f> img_i;

            for (size_t j=0;j<buf_[cam].shots[k].ids.size();++j)
            {
                int id = buf_[cam].shots[k].ids[j];
                auto it = map0.find(id);
                if (it!=map0.end())
                {
                    obj.push_back( objCorners[id] );
                    img_i.push_back( buf_[cam].shots[k].imgPts[j] );
                }
            }

            if (obj.size() < 4) {
                std::cerr << "[solvePnP] Cam " << cam << ", frame " << k
                          << ": too few matched points (" << obj.size() << ")\n";
                continue;
            }

            try {
                cv::Mat rvec, tvec;
                bool success = cv::solvePnP(obj, img_i, Ki, Di, rvec, tvec,
                                            false, cv::SOLVEPNP_ITERATIVE);
                if (!success) {
                    std::cerr << "[solvePnP] Cam " << cam << ", frame " << k
                              << ": solvePnP failed to converge\n";
                    continue;
                }
                rvecs.push_back(rvec);
                tvecs.push_back(tvec);
            } catch (const cv::Exception& e) {
                std::cerr << "[solvePnP] Cam " << cam << ", frame " << k
                          << ": exception during solvePnP:\n" << e.what() << '\n';
            }
        }

        if (rvecs.empty()){
            std::cerr<<"[MultiCamCalib] Cam"<<cam<<" solvePnP failed\n";
            continue;
        }

        cv::Mat r_mean = vecAverage(rvecs);
        cv::Mat t_mean = vecAverage(tvecs);

        cv::Mat R;
        cv::Rodrigues(r_mean, R);

        exts[cam].R = R.clone();
        exts[cam].t = t_mean.clone();
    }

    /* 写 YAML */
    if (!rig_yaml.empty())
    {
        cv::FileStorage fs(rig_yaml, cv::FileStorage::WRITE);
        for (size_t i=0;i<N;++i){
            fs << ("R"+std::to_string(i)) << exts[i].R;
            fs << ("t"+std::to_string(i)) << exts[i].t;
        }
        fs.release();
        std::cout<<"[MultiCamCalib] Rig 外参写入 "<<rig_yaml<<'\n';
    }
    return exts;
}


} // namespace charuco
