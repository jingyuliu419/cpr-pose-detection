import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ 配置 ------------------
CSV = "/home/ljy/project/poseDetection/build/press_depth_log.csv"
# ------------------------------------------

# ---------- 读取 CSV ----------
cols = ["t", "x", "y", "z"]
df = pd.read_csv(CSV, header=None, names=cols, usecols=range(len(cols)))
P = df[["x", "y", "z"]].values

# ---------- 1. SVD 拟合平面 → 法向 ----------
centroid = P.mean(axis=0)
_, _, vh = np.linalg.svd(P - centroid)
normal = vh[-1] / np.linalg.norm(vh[-1])
if ((P - centroid) @ normal).mean() < 0:
    normal = -normal

# ---------- 2. 构造旋转矩阵 normal → Y ----------
Y_axis = np.array([0.0, 1.0, 0.0])
angle = np.arccos(np.clip(np.dot(normal, Y_axis), -1.0, 1.0))
print(f"Tilt angle: {np.degrees(angle):.2f}°")
axis = np.cross(normal, Y_axis)
axis_norm = np.linalg.norm(axis)
if axis_norm < 1e-8:
    R = np.eye(3)
else:
    axis /= axis_norm
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# ---------- 3. 旋转并取 Y' ----------
P_rot = (R @ P.T).T
Y_corr = P_rot[:, 1]
df["y_corr"] = Y_corr

# ---------- 4. 绘制轨迹 ----------
plt.figure()
plt.plot(df["t"], df["y_corr"], label="Y' (after tilt compensation)")
plt.xlabel("t (s)")
plt.ylabel("Y' along board normal (m)")
plt.title("Y' trajectory after tilt compensation")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 5. 打印 Y' ----------
print("Y' values (m):")
print(df[["t", "y_corr"]].to_string(index=False))


# 读取 CSV 文件
CSV = "/home/ljy/project/poseDetection/build/press_depth_log.csv"
cols = ["t", "x", "y", "z"]
df = pd.read_csv(CSV, header=None, names=cols, usecols=range(len(cols)))

# 计算相邻帧的 y 坐标差异
df["y_diff"] = df["y"].diff().abs()

# 找到最大 y 方向的位移
# 找到最大 y 方向的位移，并使用倾斜角修正
# max_move_distance = df["y_diff"].max() * np.sin(angle)  # 直接使用角度的弧度制值
max_move_distance = df["y_diff"].max()
# 输出最大位移
print(f"The maximum movement distance (in Y direction) is: {max_move_distance:.6f} m")

