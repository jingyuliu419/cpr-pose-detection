import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("/home/ljy/project/poseDetection/build/sync_error_log.csv", header=None)

# 添加 drift_ns 列（假设在最后一列）
df["drift_ns"] = df.iloc[:, -1]

# ====== 图一：整体漂移分布 ======
plt.figure(figsize=(8, 5))
plt.hist(df["drift_ns"], bins=50)
plt.xlabel("Time Drift (ns)")
plt.ylabel("Count")
plt.title("Sync Time Drift Distribution (All)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== 图二：漂移大于 4e7 ns 的帧分析 ======
drift_threshold = 3e7
df_large_drift = df[df["drift_ns"] > drift_threshold]

print(f"Total entries: {len(df)}, Drift > 30ms: {len(df_large_drift)} ({100 * len(df_large_drift)/len(df):.2f}%)")

if not df_large_drift.empty:
    plt.figure(figsize=(8, 5))
    plt.hist(df_large_drift["drift_ns"], bins=30, color='red')
    plt.xlabel("Time Drift (ns)")
    plt.ylabel("Count")
    plt.title("Sync Drift > 30ms")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No entries with drift > 30ms.")
