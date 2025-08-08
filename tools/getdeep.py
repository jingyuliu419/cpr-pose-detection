import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. 加载三个相机的数据 ===
file_cam0 = "/home/ljy/project/poseDetection/build/cam0_projected_log.csv"
file_cam4 = "/home/ljy/project/poseDetection/build/cam5_projected_log.csv"
file_cam6 = "/home/ljy/project/poseDetection/build/cam8_projected_log.csv"

df_cam0 = pd.read_csv(file_cam0, header=None, names=["timestamp", "move_dist", "conf_score"])
df_cam4 = pd.read_csv(file_cam4, header=None, names=["timestamp", "move_dist", "conf_score"])
df_cam6 = pd.read_csv(file_cam6, header=None, names=["timestamp", "move_dist", "conf_score"])

# === 2. 加入相机编号信息（可选）===
df_cam0["camera"] = "cam0"
df_cam4["camera"] = "cam4"
df_cam6["camera"] = "cam6"

# === 3. 合并为一个大表 ===
df_all = pd.concat([df_cam0, df_cam4, df_cam6], ignore_index=True)

# === 4. 简单平均融合：对同一 timestamp 的 move_dist 求平均 ===
df_avg = (
    df_all.groupby("timestamp")
    .agg(avg_move_dist=("move_dist", "mean"))
    .reset_index()
)

# === 5. 绘制原始曲线 + 平均融合曲线 ===
plt.figure(figsize=(12, 6))

# 原始三个相机曲线（加虚线与透明度）
plt.plot(df_cam0["timestamp"], df_cam0["move_dist"], label="Cam 0", linestyle='--', alpha=0.5)
plt.plot(df_cam4["timestamp"], df_cam4["move_dist"], label="Cam 1", linestyle='--', alpha=0.5)
plt.plot(df_cam6["timestamp"], df_cam6["move_dist"], label="Cam 2", linestyle='--', alpha=0.5)

# 融合曲线（实线）
plt.plot(df_avg["timestamp"], df_avg["avg_move_dist"], label="Average Distance", color="blue", linewidth=2)

# === 6. 图形美化 ===
plt.xlabel("Timestamp (ms)")
plt.ylabel("Euclidean Distance")
plt.title("Original and Averaged Distance over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === 7. 保存图像与 CSV ===
curve_path = "multi_camera_fusion_simple_average.png"
csv_path = "simple_average_move_distance.csv"

plt.savefig(curve_path, dpi=300)
print(f"✅ 曲线图已保存: {os.path.abspath(curve_path)}")

df_avg.to_csv(csv_path, index=False)
print(f"✅ 简单平均融合结果已保存: {os.path.abspath(csv_path)}")

# 显示图
plt.show()
