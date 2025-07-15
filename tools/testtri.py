import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 为每行字符串
with open("/home/ljy/project/poseDetection/build/sync_error_log.csv") as f:
    lines = f.readlines()

# 拆分并提取字段
records = []
max_cams = 0

for line in lines:
    parts = line.strip().split(",")
    min_ts = int(parts[0])
    max_ts = int(parts[1])
    drift_ns = int(parts[2])
    cam_info = parts[3:]

    cams = {}
    for item in cam_info:
        if ':' in item:
            cid, ts = item.split(":")
            cams[f"cam{cid}"] = int(ts)
    max_cams = max(max_cams, len(cams))
    records.append({**cams, "drift_ns": drift_ns})

# 转为 DataFrame（不同 cam 列自动补 NaN）
df = pd.DataFrame(records)

# ========== 图一：漂移分布 ==========
plt.figure(figsize=(8, 5))
plt.hist(df["drift_ns"].dropna(), bins=50)
plt.xlabel("Time Drift (ns)")
plt.ylabel("Count")
plt.title("Sync Time Drift Distribution (All)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 图二：各相机时间线 ==========
plt.figure(figsize=(10, 6))
for col in df.columns:
    if col.startswith("cam"):
        plt.plot(df[col], label=col, alpha=0.7)
plt.xlabel("Frame Index")
plt.ylabel("Timestamp (ns)")
plt.title("Camera Timestamp Trace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 图三：异常漂移分布 ==========
threshold = 25000000  # 30 ms
df_large_drift = df[df["drift_ns"] > threshold]
print(f"Total: {len(df)}, Drift > 25ms: {len(df_large_drift)} ({len(df_large_drift) / len(df) * 100:.2f}%)")

if not df_large_drift.empty:
    plt.figure(figsize=(8, 5))
    plt.hist(df_large_drift["drift_ns"], bins=10, color='red')
    plt.xlabel("Time Drift (ns)")
    plt.ylabel("Count")
    plt.title("Sync Drift > 30ms")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No entries drift > 30ms.")
