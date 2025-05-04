# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # === Ensure figs/ directory exists ===
# os.makedirs("figs", exist_ok=True)

# # === Load the data ===
# df = pd.read_csv("final_model_summary.csv")

# # === Convert columns to numeric if needed ===
# df["AvgAcrossTaskTypes"] = pd.to_numeric(df["AvgAcrossTaskTypes"], errors="coerce")
# df["EvalTime"] = pd.to_numeric(df["EvalTime"], errors="coerce")

# # === Plot ===
# plt.figure(figsize=(10, 6))
# plt.scatter(df["EvalTime"], df["AvgAcrossTaskTypes"], color="blue", alpha=0.8)

# # Add labels
# for i, row in df.iterrows():
#     plt.text(row["EvalTime"] + 1, row["AvgAcrossTaskTypes"], row["model_name"], fontsize=8)

# plt.xlabel("Evaluation Time (s)")
# plt.ylabel("AvgType (Mean across Task Types)")
# plt.title("Evaluation Time vs AvgType")
# plt.grid(True)

# # === Save ===
# output_path = "figs/eval_time_vs_avgtype.png"
# plt.tight_layout()
# plt.savefig(output_path, dpi=300)
# print(f"Figure saved to {output_path}")






# # Disease by system with distance‐based outlier removal
# import os
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # 1. Paths
# csv_path = "/home/skyfury/projects/def-mahyarh/skyfury/medteb/MedTEB/data/clustering/wiki_disease_by_system_dataset.csv"
# fig_dir  = "figs"
# os.makedirs(fig_dir, exist_ok=True)

# # 2. Load data
# df        = pd.read_csv(csv_path)
# texts     = df["text"].tolist()
# labels    = df["label"].tolist()

# # 3. Embed texts
# model     = SentenceTransformer('skyfury/CTMEDGTE-cl15-step_8000')
# embeddings= model.encode(texts, show_progress_bar=True)

# # 4. t-SNE projection
# tsne      = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
# emb_2d    = tsne.fit_transform(embeddings)

# # 5. Build DataFrame for filtering
# plot_df = pd.DataFrame(emb_2d, columns=["x","y"])
# plot_df["label"] = labels

# def drop_furthest(group, frac=0.05):
#     # compute centroid
#     cx, cy = group.x.mean(), group.y.mean()
#     # compute distances
#     dists  = np.hypot(group.x - cx, group.y - cy)
#     # threshold at the (1−frac) percentile
#     thresh = np.percentile(dists, 100 * (1 - frac))
#     return group[dists <= thresh]

# # apply per-label filtering
# clean_df = plot_df.groupby("label", group_keys=False).apply(drop_furthest)

# # re-extract cleaned coordinates & labels
# emb_clean = clean_df[["x","y"]].values
# labels_clean = clean_df["label"].tolist()

# # 6. Map labels to colors
# unique_labels = sorted(set(labels_clean))
# label_to_int  = {lab: i for i, lab in enumerate(unique_labels)}
# color_ids     = [label_to_int[lab] for lab in labels_clean]

# # 7. Plot (transparent background, smaller dots)
# plt.figure(figsize=(10, 8), facecolor='white')
# ax = plt.gca()
# ax.set_facecolor('none')

# scatter = ax.scatter(
#     emb_clean[:, 0],
#     emb_clean[:, 1],
#     c=color_ids,
#     s=20,
#     alpha=0.7
# )

# # ensure legend colors match exactly
# cmap = scatter.cmap
# norm = scatter.norm

# handles = []
# for lab, idx in label_to_int.items():
#     color = cmap(norm(idx))
#     handles.append(
#         plt.Line2D([], [], marker="o", linestyle="",
#                    label=lab,
#                    markerfacecolor=color,
#                    markersize=6,
#                    alpha=0.7)
#     )
# plt.legend(handles=handles, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

# plt.title("t‑SNE Visualization of Wikipedia Disease Embeddings by Body System")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.tight_layout()

# # 8. Save figure
# out_path = os.path.join(fig_dir, "tsne_wiki_disease_by_system_filtered.pdf")
# plt.savefig(out_path, format='pdf')
# out_path = os.path.join(fig_dir, "tsne_wiki_disease_by_system_filtered.jpg")
# plt.savefig(out_path, format='jpg')
# plt.close()

# print(f"Saved t-SNE plot to {out_path}")





# """
# make_mteb_summary.py
# ────────────────────
# Collect remote MTEB results for a list of models and build a
# model × {classification, clustering, pair_classification, retrieval} table.

# Requires:
#     pip install datasets huggingface_hub pandas tqdm
# """
# from __future__ import annotations
# import re
# import pandas as pd
# from tqdm.auto import tqdm
# from datasets import load_dataset

# # ----------------------------------------------------------------------
# # 1) Slug → pretty name  (extend freely)
# # ----------------------------------------------------------------------
# name_mapping = {
#     "BAAI/bge-base-en-v1.5":                     "BAAI Bge Base En V1.5",
#     "BASF-AI/chem-embed-text-v1":                "BASF AI Chem Embed Text V1",
#     "allenai/scibert_scivocab_uncased":          "AllenAI Scibert Scivocab Uncased",
#     "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
#                                                  "BioNLP BlueBERT (PubMed+MIMIC)",
#     "emilyalsentzer/Bio_ClinicalBERT":           "Bio ClinicalBERT",
#     "google-bert/bert-base-uncased":             "BERT Base Uncased",
#     "hsila/run1-med-nomic":                      "Hsila Med‑Nomic",
#     "intfloat/e5-base":                          "E5 Base",
#     "kamalkraj/BioSimCSE-BioLinkBERT-BASE":      "BioSimCSE BioLinkBERT",
#     "malteos/scincl":                            "SciNCL",
#     "medicalai/ClinicalBERT":                    "ClinicalBERT",
#     "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
#                                                  "BiomedBERT (Abstract+FT)",
#     "nomic-ai/nomic-embed-text-v1":              "Nomic Embed‑Text v1",
#     "nomic-ai/nomic-embed-text-v1-unsupervised": "Nomic Embed‑Text v1 (unsup)",
#     "sentence-transformers/all-MiniLM-L6-v2":    "All‑MiniLM‑L6‑v2",
#     "sentence-transformers/all-mpnet-base-v2":   "All‑MPNet‑Base‑v2",
#     "skyfury/CTMEDGTE-cl15-step_8000":           "Skyfury CTMEDGTE cl‑15 @ 8k",
#     "thenlper/gte-base":                         "GTE Base",
#     "abhinand/MedEmbed-base-v0.1":               "MedEmbed Base v0.1",
# }

# # ----------------------------------------------------------------------
# # 2) Heuristic ‑– map a dataset name to a coarse task‑type
# # ----------------------------------------------------------------------
# def infer_task_type(ds: str) -> str | None:
#     ds = ds.lower()
#     if "clustering" in ds:
#         return "clustering"
#     if "retrieval" in ds or "reranking" in ds or "bitext" in ds:
#         return "retrieval"
#     if ("pc" in ds) or ("pairclassification" in ds) or ("sts" in ds):
#         return "pair_classification"
#     if "classification" in ds or "sentiment" in ds:
#         return "classification"
#     return None  # ignore anything we can’t categorise


# # ----------------------------------------------------------------------
# # 3) Gather one model’s table of scores directly from HF
# # ----------------------------------------------------------------------
# def get_model_summary(model_slug: str) -> dict[str, float]:
#     """
#     Returns a dict with averaged scores for each task‑type.

#     Notes
#     -----
#     • We load the `mteb/results` dataset with the *model*‑specific
#       config (owner__model).  
#     • Every row in that split is one metric on one dataset/language.
#       We:
#          a) infer the coarse task‑type from the dataset name;
#          b) keep only rows we can categorise;
#          c) average the `score` column per task‑type.
#     """
#     cfg_name = model_slug.replace("/", "__")      # e.g.  BAAI/bge‑…  →  BAAI__bge‑…
#     ds = load_dataset("mteb/results", name=cfg_name, split="test", streaming=False)
#     df = ds.to_pandas()

#     # Categorise tasks
#     df["task_type"] = df["mteb_dataset_name"].apply(infer_task_type)
#     df = df.dropna(subset=["task_type"])

#     # Average all numeric scores within each task‑type
#     summary = (
#         df.groupby("task_type")["score"]
#           .mean()
#           .round(3)                 # keep only 3 decimals for clarity
#           .to_dict()
#     )
#     return summary


# # ----------------------------------------------------------------------
# # 4) Build the final DataFrame
# # ----------------------------------------------------------------------
# rows = []

# for slug, pretty in tqdm(name_mapping.items(), desc="Fetching models"):
#     try:
#         scores = get_model_summary(slug)
#     except Exception as exc:
#         print(f"⚠️  {slug}: {exc}")
#         continue

#     rows.append(
#         {
#             "model_name":          pretty,
#             "classification":      scores.get("classification", float("nan")),
#             "clustering":          scores.get("clustering", float("nan")),
#             "pair_classification": scores.get("pair_classification", float("nan")),
#             "retrieval":           scores.get("retrieval", float("nan")),
#         }
#     )

# df = (
#     pd.DataFrame(rows)
#       .set_index("model_name")
#       .sort_index()
# )

# print(df)
# # Optional: df.to_csv("mteb_summary_remote.csv")




import os
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------------------------------------------------
# 1) I/O ----------------------------------------------------------------
os.makedirs("figs", exist_ok=True)                       # ensure output dir
df = pd.read_csv("final_model_summary.csv")

# ----------------------------------------------------------------------
# 2) Prep ---------------------------------------------------------------
# Set the model names as index (adjust if your CSV already uses an index)
if "model_name" in df.columns:
    df = df.set_index("model_name")

# Make sure all numeric cols are numbers
num_cols = [
    "classification",
    "clustering",
    "pair_classification",
    "retrieval",
]

def just_mean(x):
    """Take '0.69 ± 0.22' ↦ '0.69'."""
    if isinstance(x, str):
        m = re.match(r"\s*([0-9.]+)", x)
        return float(m.group(1)) if m else float("nan")
    return x

df[num_cols] = df[num_cols].applymap(just_mean)


models_by_domain = {
    "medical": [
        "Abhinand MedEmbed Base",
        "BioNLP Bluebert PubMed Mimic Uncased L‑12 H‑768 A‑12",
        "EmilyAlsentzer Bio ClinicalBERT",
        "Kamalkraj BioSimCSE BioLinkBERT Base",
        "Malteos SciNCL",
        "MedicalAI ClinicalBERT",
        "Microsoft BiomedNLP BiomedBERT Base Uncased Abstract Fulltext",
        "Skyfury CTMEDGTE Cl15 Step 8000",
    ],
    "non_medical": [
        "BAAI Bge Base En V1.5",
        "AllenAI Scibert Scivocab Uncased",
        "Google BERT Base Uncased",
        "Intfloat E5 Base",
        "Nomic AI Nomic Embed Text V1",
        "Nomic AI Nomic Embed Text V1 Unsupervised",
        "Sentence-Transformers All MiniLM L6 V2",
        "Sentence-Transformers All MPNet Base V2",
        "Thenlper GTE Base",
    ],
}
my_model = "Skyfury CTMEDGTE Cl15 Step 8000"             # highlight this one
domain_colour = {"medical": "tab:blue", "non_medical": "tab:orange"}

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3) Lower‑triangle without empty top row / right col -------------------
var_order = num_cols                         # fixed order you gave
n_vars = len(var_order)                      # = 4 → need 3 × 3 grid

fig, axes = plt.subplots(
    n_vars - 1, n_vars - 1,
    figsize=(4.6 * (n_vars - 1), 4.6 * (n_vars - 1)),
    sharex=False, sharey=False,
    constrained_layout=True,
)

for row in range(n_vars - 1):                # rows 0..2  → y‑vars 1..3
    y_var = var_order[row + 1]               # shift by +1
    for col in range(n_vars - 1):            # cols 0..2 → x‑vars 0..2
        x_var = var_order[col]
        ax = axes[row, col]

        # plot only if we are in (strict) lower triangle: col ≤ row
        if col <= row:
            # ── scatter for each domain ────────────────────────────────
            for domain, models in models_by_domain.items():
                mask = df.index.isin(models)
                ax.scatter(
                    df.loc[mask, x_var],
                    df.loc[mask, y_var],
                    label=domain if (row, col) == (0, 0) else "",
                    c=domain_colour[domain],
                    s=60,
                    edgecolors="k",
                    alpha=0.8,
                )

            # ── highlight *your* checkpoint ────────────────────────────
            if my_model in df.index:
                ax.scatter(
                    df.loc[my_model, x_var],
                    df.loc[my_model, y_var],
                    marker="*",
                    s=220,
                    c="red",
                    edgecolors="k",
                    label="Ours" if (row, col) == (0, 0) else "",
                    zorder=10,
                )

            # ── axis labels only on left column / bottom row ───────────
            if col == 0:
                ax.set_ylabel(y_var.replace("_", " ").title())
            if row == n_vars - 2:
                ax.set_xlabel(x_var.replace("_", " ").title())

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            # ── Pearson correlation r ----------------------------------
            valid = df[[x_var, y_var]].dropna()
            if len(valid) > 1:
                r, _ = pearsonr(valid[x_var], valid[y_var])
                ax.text(
                    0.05, 0.95, f"r = {r:.2f}",
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, fontweight="bold",
                )
        else:
            # hide upper‑triangle cells
            ax.set_visible(False)

# ----------------------------------------------------------------------
# 4) legend outside, title, save (unchanged) ----------------------------
fig.subplots_adjust(right=0.80)              # free 20 % width for legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(.90, 0.95),
    frameon=True,
)
fig.suptitle("Pairwise Correlations of Model Metrics", fontsize=16, y=1.06)

fig.savefig("figs/pairwise_correlations.pdf", bbox_inches="tight")
fig.savefig("figs/pairwise_correlations.jpg", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figures written to figs/pairwise_correlations.[pdf|jpg]")
