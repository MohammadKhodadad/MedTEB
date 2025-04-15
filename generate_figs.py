import os
import pandas as pd
import matplotlib.pyplot as plt

# === Ensure figs/ directory exists ===
os.makedirs("figs", exist_ok=True)

# === Load the data ===
df = pd.read_csv("final_model_summary.csv")

# === Convert columns to numeric if needed ===
df["AvgAcrossTaskTypes"] = pd.to_numeric(df["AvgAcrossTaskTypes"], errors="coerce")
df["EvalTime"] = pd.to_numeric(df["EvalTime"], errors="coerce")

# === Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(df["EvalTime"], df["AvgAcrossTaskTypes"], color="blue", alpha=0.8)

# Add labels
for i, row in df.iterrows():
    plt.text(row["EvalTime"] + 1, row["AvgAcrossTaskTypes"], row["model_name"], fontsize=8)

plt.xlabel("Evaluation Time (s)")
plt.ylabel("AvgType (Mean across Task Types)")
plt.title("Evaluation Time vs AvgType")
plt.grid(True)

# === Save ===
output_path = "figs/eval_time_vs_avgtype.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Figure saved to {output_path}")
