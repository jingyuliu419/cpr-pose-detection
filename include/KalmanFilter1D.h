// KalmanFilter1D.hpp  —— 头文件放在 include/filters 目录
#pragma once
class KalmanFilter1D {
public:
    KalmanFilter1D(float q = 1e-4f,  // 过程噪声协方差
                   float r = 1e-2f,  // 观测噪声协方差
                   float p0 = 1.f)   // 初值误差
        : Q(q), R(r), P(p0), X(0.f) {}

    float update(float z) {
        // 预测：X(k|k-1) = X(k-1)
        P += Q;                     // 误差协方差预测
        // 更新
        float K = P / (P + R);      // 卡尔曼增益
        X += K * (z - X);           // 状态校正
        P *= (1.f - K);             // 协方差更新
        return X;
    }
private:
    float Q, R;    // 噪声协方差
    float P;       // 估计误差协方差
    float X;       // 当前位置估计
};
