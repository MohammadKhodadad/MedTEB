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






# Disease by system with distance‐based outlier removal
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Paths
csv_path = "/home/skyfury/projects/def-mahyarh/skyfury/medteb/MedTEB/data/clustering/wiki_disease_by_system_dataset.csv"
fig_dir  = "figs"
os.makedirs(fig_dir, exist_ok=True)

# 2. Load data
df        = pd.read_csv(csv_path)
texts     = df["text"].tolist()
labels    = df["label"].tolist()

# 3. Embed texts
model     = SentenceTransformer('skyfury/CTMEDGTE-cl14-step_8000')
embeddings= model.encode(texts, show_progress_bar=True)

# 4. t-SNE projection
tsne      = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
emb_2d    = tsne.fit_transform(embeddings)

# 5. Build DataFrame for filtering
plot_df = pd.DataFrame(emb_2d, columns=["x","y"])
plot_df["label"] = labels

def drop_furthest(group, frac=0.1):
    # compute centroid
    cx, cy = group.x.mean(), group.y.mean()
    # compute distances
    dists  = np.hypot(group.x - cx, group.y - cy)
    # threshold at the (1−frac) percentile
    thresh = np.percentile(dists, 100 * (1 - frac))
    return group[dists <= thresh]

# apply per-label filtering
clean_df = plot_df.groupby("label", group_keys=False).apply(drop_furthest)

# re-extract cleaned coordinates & labels
emb_clean = clean_df[["x","y"]].values
labels_clean = clean_df["label"].tolist()

# 6. Map labels to colors
unique_labels = sorted(set(labels_clean))
label_to_int  = {lab: i for i, lab in enumerate(unique_labels)}
color_ids     = [label_to_int[lab] for lab in labels_clean]

# 7. Plot (transparent background, smaller dots)
plt.figure(figsize=(10, 8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('none')

scatter = ax.scatter(
    emb_clean[:, 0],
    emb_clean[:, 1],
    c=color_ids,
    s=20,
    alpha=0.7
)

# ensure legend colors match exactly
cmap = scatter.cmap
norm = scatter.norm

handles = []
for lab, idx in label_to_int.items():
    color = cmap(norm(idx))
    handles.append(
        plt.Line2D([], [], marker="o", linestyle="",
                   label=lab,
                   markerfacecolor=color,
                   markersize=6,
                   alpha=0.7)
    )
plt.legend(handles=handles, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.title("t-SNE of Wiki Disease Terms by System\n(top 10% furthest per label removed)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()

# 8. Save figure
out_path = os.path.join(fig_dir, "tsne_wiki_disease_by_system_filtered.pdf")
plt.savefig(out_path, format='pdf')
out_path = os.path.join(fig_dir, "tsne_wiki_disease_by_system_filtered.jpg")
plt.savefig(out_path, format='jpg')
plt.close()

print(f"Saved t-SNE plot to {out_path}")







