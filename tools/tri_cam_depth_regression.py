import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# === 1. OpenCV 加载外参 rig_extrinsics.yaml ===
def load_all_extrinsics_cv2(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    R0 = fs.getNode("R0").mat()
    t0 = fs.getNode("t0").mat()
    R1 = fs.getNode("R1").mat()
    t1 = fs.getNode("t1").mat()
    R2 = fs.getNode("R2").mat()
    t2 = fs.getNode("t2").mat()
    fs.release()
    return (R0, t0), (R1, t1), (R2, t2)

# === 2. OpenCV 加载内参（camera_gp*.yml） ===
def load_intrinsics_cv2(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    fs.release()
    return K

# === 3. 构造投影矩阵 ===
def get_projection_matrix(K, R, T):
    RT = np.hstack((R, T))  # 3x4
    return K @ RT

# === 4. 两视图三角测量 ===
def triangulate_two_views(P1, P2, pt1, pt2):
    pt1 = pt1.reshape(2, 1)
    pt2 = pt2.reshape(2, 1)
    X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
    X = X_h[:3] / X_h[3]
    return X.flatten()

# === 5. 读取三相机像素点数据 ===
df0 = pd.read_csv("/home/ljy/project/poseDetection/build/cam0_projected_log.csv", names=["time", "u0", "v0"])
df2 = pd.read_csv("/home/ljy/project/poseDetection/build/cam2_projected_log.csv", names=["time", "u2", "v2"])
df6 = pd.read_csv("/home/ljy/project/poseDetection/build/cam6_projected_log.csv", names=["time", "u6", "v6"])

# 对齐时间戳
df0["round_time"] = df0["time"].round(2)
df2["round_time"] = df2["time"].round(2)
df6["round_time"] = df6["time"].round(2)
merged = df0.merge(df2, on="round_time").merge(df6, on="round_time")

# === 6. 加载外参 ===
(R0, t0), (R2, t2), (R6, t6) = load_all_extrinsics_cv2("/home/ljy/project/poseDetection/config/rig_extrinsics.yaml")

# === 7. 加载内参 ===
K0 = load_intrinsics_cv2("/home/ljy/project/poseDetection/config/camera_gp01.yml")  # cam0
K2 = load_intrinsics_cv2("/home/ljy/project/poseDetection/config/camera_gp23.yml")  # cam2
K6 = load_intrinsics_cv2("/home/ljy/project/poseDetection/config/camera_gp67.yml")  # cam6

# === 8. 构造投影矩阵 ===
P0 = get_projection_matrix(K0, R0, t0)
P2 = get_projection_matrix(K2, R2, t2)
P6 = get_projection_matrix(K6, R6, t6)

# === 9. 三角测量融合 ===
fused_points = []
for row in merged.itertuples():
    pt0 = np.array([row.u0, row.v0])
    pt2 = np.array([row.u2, row.v2])
    pt6 = np.array([row.u6, row.v6])

    # 三个组合
    X02 = triangulate_two_views(P0, P2, pt0, pt2)
    X06 = triangulate_two_views(P0, P6, pt0, pt6)
    X26 = triangulate_two_views(P2, P6, pt2, pt6)

    # 融合为平均点
    fused = (X02 + X06 + X26) / 3
    fused_points.append(fused)

fused_points = np.array(fused_points)
merged["fused_distance"] = np.linalg.norm(fused_points, axis=1)

# === 10. 可选保存3D坐标 ===
np.savetxt("fused_3d_points.csv", fused_points, delimiter=",", header="x,y,z", comments='')

# === 11. 绘图展示 ===
plt.figure(figsize=(12, 6))
plt.plot(merged["round_time"], merged["fused_distance"], label="Triangulated Distance", color='black')
plt.xlabel("Time (s)")
plt.ylabel("Distance from Origin (m)")
plt.title("3-Camera Triangulated Distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
