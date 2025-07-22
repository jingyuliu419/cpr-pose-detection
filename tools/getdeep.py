import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --------- 1. 读取数据 ---------
df = pd.read_csv("/home/ljy/project/poseDetection/build/triangulated_xyz_log.csv", header=None, names=["time", "x", "y", "z"])
df = df[np.isfinite(df["z"])]  # 过滤无效数据

z_vals = df["z"].values
time_vals = df["time"].values

# --------- 2. 自动检测稳定段 ---------
window_size = 20
min_std = float("inf")
baseline_start = 0

for i in range(len(z_vals) - window_size):
    std = np.std(z_vals[i:i + window_size])
    if std < min_std:
        min_std = std
        baseline_start = i

baseline_z = np.mean(z_vals[baseline_start:baseline_start + window_size])
print(f"📌 稳定段起点: 第 {baseline_start} 帧")
print(f"📌 估计稳定基线 Z 值: {baseline_z:.5f} m")

# --------- 3. 相对压深（手掌误差修正 -0.05） ---------
baseline_z = -0.73760  # 手掌误差修正
df["depth"] = baseline_z - df["z"]
df["depth"] = df["depth"].clip(lower=0)

# --------- 4. 描述性统计 ---------
print("\n📊 压深（以基线为原点）描述性统计：")
print(df["depth"].describe())

# --------- 5. 落在 3cm ~ 8cm 的占比 ---------
mask_3_8 = df["depth"].between(0.03, 0.08)
rate = mask_3_8.sum() / len(df)
print(f"\n📌 落在 3cm ~ 8cm 范围内的压深帧数：{mask_3_8.sum()}，占比：{rate*100:.2f}%")

# --------- 6. KDE 估计 & 分布图 ---------
depth_vals = df["depth"].values
kde = gaussian_kde(depth_vals)
depth_grid = np.linspace(0, depth_vals.max(), 1000)
mode_depth = depth_grid[np.argmax(kde(depth_grid))]

plt.figure(figsize=(8, 5))
plt.hist(depth_vals, bins=80, color='lightcoral', edgecolor='black', alpha=0.6, density=True, label="Histogram")
plt.plot(depth_grid, kde(depth_grid), color='darkred', lw=2, label="KDE")
plt.axvline(mode_depth, color='green', linestyle='--', label=f'Mode: {mode_depth*100:.1f} cm')
plt.axvspan(0.03, 0.08, color='yellow', alpha=0.3, label="3cm ~ 8cm")
plt.xlabel("Compression Depth (m)")
plt.ylabel("Density")
plt.title("Depth Distribution (with baseline=0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("depth_distribution_rebased.png", dpi=300)
plt.close()
print("✅ 已保存压深分布图（设基线为 0）为 depth_distribution_rebased.png")

# --------- 7. 平滑压深随时间变化曲线 ---------
depth_smooth = gaussian_filter1d(depth_vals, sigma=5)

plt.figure(figsize=(12, 5))
plt.plot(df["time"], depth_vals, color='gray', lw=0.5, alpha=0.5, label='Raw Depth')
plt.plot(df["time"], depth_smooth, color='blue', lw=1.5, label='Smoothed Depth')
plt.xlabel("Time")
plt.ylabel("Depth (m)")
plt.title("Smoothed Compression Depth Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("depth_smoothed_curve.png", dpi=300)
plt.close()
print("✅ 已保存平滑压深曲线图为 depth_smoothed_curve.png")

# --------- 8. 峰值检测（压深波峰） ---------
# 动态设定 height 阈值为 90% 分位数的 30%，防止过高阈值导致 0 个峰
adaptive_threshold = np.percentile(depth_smooth, 90) * 0.3
adaptive_threshold = max(adaptive_threshold, 0.001)  # 保底阈值

peaks, _ = find_peaks(depth_smooth, height=adaptive_threshold, distance=30)
peak_depths = depth_smooth[peaks]

# 可视化峰值检测结果
plt.figure(figsize=(12, 5))
plt.plot(df["time"], depth_smooth, label="Smoothed Depth", color="blue")
plt.plot(df["time"].iloc[peaks], peak_depths, "rx", label="Detected Peaks")
plt.xlabel("Time")
plt.ylabel("Depth (m)")
plt.title("Detected Compression Peaks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compression_peaks.png", dpi=300)
plt.close()
print(f"✅ 检测到 {len(peaks)} 个峰值，已保存峰值图为 compression_peaks.png")

# --------- 9. 峰值 KDE 分布分析 ---------
if len(peak_depths) >= 2:
    kde_peak = gaussian_kde(peak_depths)
    peak_grid = np.linspace(0, peak_depths.max(), 1000)
    mode_peak = peak_grid[np.argmax(kde_peak(peak_grid))]

    plt.figure(figsize=(8, 5))
    plt.hist(peak_depths, bins=60, color='lightblue', edgecolor='black', alpha=0.6, density=True, label="Peak Histogram")
    plt.plot(peak_grid, kde_peak(peak_grid), color='blue', lw=2, label="KDE of Peaks")
    plt.axvline(mode_peak, color='green', linestyle='--', label=f'Peak Mode: {mode_peak*100:.1f} cm')
    plt.axvspan(0.03, 0.08, color='yellow', alpha=0.3, label="3cm ~ 8cm")
    plt.xlabel("Peak Compression Depth (m)")
    plt.ylabel("Density")
    plt.title("Distribution of Peak Compression Depths")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("peak_depth_distribution.png", dpi=300)
    plt.close()

    print(f"📌 峰值压深的众数为：{mode_peak*100:.2f} cm")
    print("✅ 已保存峰值压深分布图为 peak_depth_distribution.png")
else:
    print("⚠️ 未检测到足够的峰值，跳过 KDE 分布分析和图像保存。")
