import pandas as pd
import matplotlib.pyplot as plt

# Read full pipeline inference log
csv_path = "/home/ljy/project/poseDetection/build/full_pipeline_log.csv"
df = pd.read_csv(csv_path)

# Convert to float and remove first 20 frames
times = pd.to_numeric(df["total_pipeline_ms"], errors='coerce').dropna().values
times = times[20:]

# Calculate average inference time
avg_time = times.mean()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(range(len(times)), times, label="Full Pipeline Inference Time", color="tab:purple")

# Add average line
plt.axhline(avg_time, color='red', linestyle='--', label=f"Average: {avg_time:.2f} ms")

# Format
plt.title("Full Pipeline Inference Time per Frame (Trimmed First 20)", fontsize=14)
plt.xlabel("Frame Index (starting from 21)")
plt.ylabel("Total Pipeline Time (ms)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save as vector graphic
plt.savefig("full_pipeline_inference_time_trimmed.svg")
plt.show()
