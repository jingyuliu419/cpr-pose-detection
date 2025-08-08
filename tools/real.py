import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from datetime import datetime

# ==== 参数设置 ====
vcc = 5.0                  # 传感器供电电压
full_range_mm = 600.0      # 满量程对应位移（mm）
target_column = 5          # 电压数据在第6列（索引5）

# ==== 1. 读取数据 ====
file_path = "/home/ljy/project/poseDetection/tools/SerialLog_20250724_160853.log"
with open(file_path, "r") as f:
    lines = f.readlines()

# 仅保留以时间戳开头的数据行（如"2025"）
data_lines = [line.strip().split(",") for line in lines if line.startswith("2025")]
timestamps = [line[0] for line in data_lines]
voltages = [float(line[target_column]) for line in data_lines]

# ==== 2. 动态参考电压 ====
reference_voltage = np.mean([float(line[target_column]) for line in data_lines[:20]])

# ==== 3. 电压 → 深度（单位：米） ====
depths = [(reference_voltage - v) / vcc * full_range_mm / 1000 for v in voltages]
depths_array = np.array(depths)

# ==== 4. 峰值检测 ====
peaks, _ = signal.find_peaks(depths_array, distance=10, height=0.01)
peak_depths = depths_array[peaks]

# ==== 5. 转换时间为 UNIX 时间戳 ====
def to_unix_float_and_ms(ts_str):
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    unix_sec = dt.timestamp()
    unix_ms = int(unix_sec * 1000)
    return unix_sec, unix_ms

peak_times_unix = []
peak_times_ms = []
for i in peaks:
    ts_sec, ts_ms = to_unix_float_and_ms(timestamps[i])
    peak_times_unix.append(ts_sec)
    peak_times_ms.append(ts_ms)

# ==== 6. 保存所有峰值数据 ====
df_peaks_all = pd.DataFrame({
    "Timestamp_ms": peak_times_ms,
    "Depth_m": peak_depths
})

df_peaks_all.to_csv("peak_voltage_depth_all.csv", index=False)
print("✅ 所有峰值数据已保存为 peak_voltage_depth_all.csv")

# ==== 7. 可选统计：深度区间（单位仍按 cm） ====
bins = [0, 10, 20, 30, 40, 50, 60]
labels = ["0-1 cm", "1-2 cm", "2-3 cm", "3-4 cm", "4-5 cm", "5-6 cm"]
df_peaks_all["Depth_Range"] = pd.cut(df_peaks_all["Depth_m"] * 1000, bins=bins, labels=labels, right=False)
depth_stats = df_peaks_all["Depth_Range"].value_counts().sort_index()

print("✔ 深度区间统计：")
print(depth_stats)

# ==== 8. 绘图 ====
plt.figure(figsize=(12, 6))
plt.plot(depths_array, label="Depth (m)")
plt.plot(peaks, depths_array[peaks], "rx", label="All Peaks")
plt.legend()
plt.title("Detected Depth Peaks (All)")
plt.xlabel("Sample Index")
plt.ylabel("Depth (m)")
plt.grid()
plt.tight_layout()
plt.savefig("depth_peaks_plot_all.png", dpi=300)
plt.show()
