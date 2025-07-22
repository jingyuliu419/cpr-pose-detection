import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==== 参数设置 ====
vcc = 5.0                  # 传感器供电电压
full_range_mm = 600.0      # 满量程对应位移（mm）
target_column = 6          # 电压数据在第6列（索引5）

# ==== 1. 读取数据 ====
file_path = "/home/ljy/project/poseDetection/tools/SerialLog_20250722_205734.log"  # 替换为你的路径
with open(file_path, "r") as f:
    lines = f.readlines()

# 仅保留数据行
data_lines = [line.strip().split(",") for line in lines if line.startswith("2025")]
timestamps = [line[0] for line in data_lines]
voltages = [float(line[target_column]) for line in data_lines]

# ==== 动态 reference_voltage ====
reference_voltage = np.mean([float(line[target_column]) for line in data_lines[:100]])

# ==== 2. 电压 -> 深度转换 ====
depths = [(reference_voltage - v) / vcc * full_range_mm for v in voltages]

# ==== 3. 峰值检测（寻找局部最大按压）====
# 使用负号查找最深按压（即最大正深度）
depths_array = np.array(depths)
peaks, _ = signal.find_peaks(depths_array, distance=10, height=1)  # 最小距离10帧，最小按压1mm

# ==== 4. 保存峰值电压与深度 ====
peak_data = {
    "Timestamp": [timestamps[i] for i in peaks],
    "Voltage": [voltages[i] for i in peaks],
    "Depth_mm": [depths[i] for i in peaks]
}
df_peaks = pd.DataFrame(peak_data)
df_peaks.to_csv("peak_voltage_depth.csv", index=False)

# ==== 5. 统计深度区间频次 ====
bins = [0, 1, 2, 3, 4, 5, 6]
labels = ["0-1 cm", "1-2 cm", "2-3 cm", "3-4 cm", "4-5 cm", "5-6 cm"]
df_peaks["Depth_Range"] = pd.cut(df_peaks["Depth_mm"], bins=bins, labels=labels, right=False)
depth_stats = df_peaks["Depth_Range"].value_counts().sort_index()

# ==== 6. 输出结果 ====
print("✔ 峰值个数：", len(df_peaks))
print("✔ 深度区间统计：")
print(depth_stats)

# ==== 7. 可选：绘图并保存 ====
plt.figure(figsize=(12, 6))
plt.plot(depths_array, label="Depth")
plt.plot(peaks, depths_array[peaks], "rx", label="Peaks")
plt.legend()
plt.title("Detected Depth Peaks")
plt.xlabel("Sample Index")
plt.ylabel("Depth (mm)")
plt.grid()
plt.tight_layout()
plt.savefig("depth_peaks_plot.png", dpi=300)  # 保存为 PNG 文件
plt.show()

