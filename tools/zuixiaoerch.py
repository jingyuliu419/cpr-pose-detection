import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# === 1. 加载数据 ===
df_visual = pd.read_csv("/home/ljy/project/poseDetection/build/final_fused_result.csv")  # timestamp, weighted_move_dist
df_pressure = pd.read_csv("/home/ljy/project/poseDetection/build/pressure_log.csv")          # timestamp, pressure
df_truth = pd.read_csv("/home/ljy/project/poseDetection/build/peak_voltage_depth_all.csv")       # Timestamp_ms, Depth_m

# === 2. 时间戳对齐（视觉、压力 round(-1)，真值减偏差后再 round(-1)） ===
time_offset = 35897  # 单位：ms
df_visual["time"] = df_visual["timestamp"].round(-1)
df_pressure["time"] = df_pressure["timestamp"].round(-1)
df_truth["time"] = (df_truth["Timestamp_ms"] - time_offset).round(-1)

# === 3. 合并三表格 ===
df_merged = df_truth.merge(df_visual[["time", "weighted_move_dist"]], on="time") \
                    .merge(df_pressure[["time", "pressure"]], on="time")

# === 4. 构建特征与标签（使用 -视觉距离+1.4 和压力）===
X = np.column_stack([-(df_merged["weighted_move_dist"].values) + 1.4, df_merged["pressure"].values])
y = df_merged["Depth_m"].values  # 真值

# === 5. 最小二乘拟合 ===
model = LinearRegression()
model.fit(X, y)
a, b = model.coef_
r = model.intercept_
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# === 6. 打印结果 ===
print("=== 最小二乘拟合结果 ===")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"r = {r:.6f}")
print(f"RMSE = {rmse:.6f}")

# === 7. 可视化 ===
plt.figure(figsize=(12, 6))
plt.plot(df_merged["time"], y, label="Ground Truth (Depth_m)", linewidth=2)
plt.plot(df_merged["time"], y_pred, label="Fitted Prediction", linestyle="--")
plt.xlabel("Timestamp (ms)")
plt.ylabel("Distance (m)")
plt.title("Truth vs Fitted Model with Time Offset Correction")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("truth_vs_fitted_with_time_offset.png", dpi=300)
plt.show()
