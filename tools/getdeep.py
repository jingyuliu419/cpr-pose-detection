import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. 读取三相机数据 ===
df1 = pd.read_csv("cam0_projected_log.csv", header=None, names=["time", "fps", "x1", "y1", "z1", "d1"])
df2 = pd.read_csv("cam4_projected_log.csv", header=None, names=["time", "fps", "x2", "y2", "z2", "d2"])
df3 = pd.read_csv("cam6_projected_log.csv", header=None, names=["time", "fps", "x3", "y3", "z3", "d3"])

# === 2. 对齐时间戳（保留两位小数） ===
df1["round_time"] = df1["time"].round(2)
df2["round_time"] = df2["time"].round(2)
df3["round_time"] = df3["time"].round(2)

# === 3. 合并三相机同步帧 ===
merged = df1.merge(df2, on="round_time").merge(df3, on="round_time")

# === 4. 计算三相机原始空间距离 ===
merged["dist_cam0"] = np.sqrt(merged["x1"]**2 + merged["y1"]**2 + merged["z1"]**2)
merged["dist_cam2"] = np.sqrt(merged["x2"]**2 + merged["y2"]**2 + merged["z2"]**2)
merged["dist_cam6"] = np.sqrt(merged["x3"]**2 + merged["y3"]**2 + merged["z3"]**2)

# === 5. 残差加权融合函数 ===
# === 5. 残差加权融合函数（Camera 2 权重大两倍）===
def residual_weighted_fusion(p1, p2, p3):
    d12 = np.linalg.norm(p1 - p2)
    d13 = np.linalg.norm(p1 - p3)
    d23 = np.linalg.norm(p2 - p3)

    if d12 <= d13 and d12 <= d23:
        # Camera 0 (p1) + Camera 2 (p2)
        w1 = 1
        w2 = 2 # cam2 加权更高
        return (w1 * p1 + w2 * p2) / (w1 + w2)

    elif d13 <= d12 and d13 <= d23:
        # Camera 0 (p1) + Camera 6 (p3)
        w1 = 1
        w3 = 1
        return (w1 * p1 + w3 * p3) / (w1 + w3)

    else:
        # Camera 2 (p2) + Camera 6 (p3)
        w2 = 2  # cam2 加权更高
        w3 = 1
        return (w2 * p2 + w3 * p3) / (w2 + w3)

# === 5. 简单平均融合函数 ===
def average_fusion(p1, p2, p3):
    return (p1 + p2 + p3) / 3

# === 6. 简单平均融合 + 计算距离 ===
fused_points = []
for row in merged.itertuples():
    p1 = np.array([row.x1, row.y1, row.z1])
    p2 = np.array([row.x2, row.y2, row.z2])
    p3 = np.array([row.x3, row.y3, row.z3])
    fused = average_fusion(p1, p2, p3)
    fused_points.append(fused)

fused_points = np.array(fused_points)
merged["dist_fused"] = np.linalg.norm(fused_points, axis=1)

# === 7. 绘图 ===
plt.figure(figsize=(12, 6))
plt.plot(merged["round_time"], merged["dist_cam0"], label="Camera 0", alpha=0.5)
plt.plot(merged["round_time"], merged["dist_cam2"], label="Camera 2", alpha=0.5)
plt.plot(merged["round_time"], merged["dist_cam6"], label="Camera 6", alpha=0.5)
plt.plot(merged["round_time"], merged["dist_fused"], label="Fused (Residual Weighted)", color='black', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Distance from Origin (m)")
plt.title("Camera Trajectories and Residual-Weighted Fusion")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
