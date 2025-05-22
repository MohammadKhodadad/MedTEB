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
model     = SentenceTransformer('skyfury/CTMEDGTE-cl15-step_8000')
embeddings= model.encode(texts, show_progress_bar=True)

# 4. t-SNE projection
tsne      = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
emb_2d    = tsne.fit_transform(embeddings)

# 5. Build DataFrame for filtering
plot_df = pd.DataFrame(emb_2d, columns=["x","y"])
plot_df["label"] = labels

def drop_furthest(group, frac=0.07):
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
plt.figure(figsize=(10, 10), facecolor='white')
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
# plt.legend(handles=handles, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")


plt.legend(
    handles=handles,
    # title="Label",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),  # adjust the vertical offset as needed
    ncol=len(handles),            # all entries in one row
    frameon=True,
    framealpha=1,
    edgecolor='black',
    fontsize=8
)

plt.title("t‑SNE Visualization of Wikipedia Disease Embeddings by Body System")
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










































# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from adjustText import adjust_text  # pip install adjustText if needed

# # Filepaths
# perf_path = 'final_source_model_summary.csv'
# time_path = 'final_model_summary_with_time.csv'

# # Load data
# perf = pd.read_csv(perf_path)
# time_df = pd.read_csv(time_path)

# # Buckets & colors (as before)
# models_by_bucket = {
#     "medical_contrastive": [
#         "Abhinand MedEmbed Base", "Kamalkraj BioSimCSE BioLinkBERT Base",
#         "Malteos SciNCL", "Skyfury CTMEDGTE Cl15 Step 8000",
#     ],
#     "medical_non_contrastive": [
#         "BioNLP Bluebert PubMed Mimic Uncased L-12 H-768 A-12",
#         "EmilyAlsentzer Bio ClinicalBERT", "MedicalAI ClinicalBERT",
#         "Microsoft BiomedNLP BiomedBERT Base Uncased Abstract Fulltext",
#     ],
#     "non_medical_contrastive": [
#         "BAAI Bge Base En V1.5", "Intfloat E5 Base",
#         "Nomic AI Nomic Embed Text V1", "Nomic AI Nomic Embed Text V1 Unsupervised",
#         "Sentence-Transformers All MiniLM L6 V2", "Sentence-Transformers All MPNet Base V2",
#         "Thenlper GTE Base",
#     ],
#     "non_medical_non_contrastive": [
#         "AllenAI Scibert Scivocab Uncased", "Google BERT Base Uncased",
#     ],
# }
# bucket_colour = {
#     "medical_contrastive":        "tab:blue",
#     "medical_non_contrastive":    "tab:cyan",
#     "non_medical_contrastive":    "tab:orange",
#     "non_medical_non_contrastive":"tab:green",
# }

# # Our model name
# our_model = "Skyfury CTMEDGTE Cl15 Step 8000"

# # Pre-merge time
# df_full = perf.merge(time_df, on='model_name')
# df_full['bucket'] = df_full['model_name'].apply(
#     lambda m: next((b for b, ms in models_by_bucket.items() if m in ms), None)
# )

# # Define the three pairs to plot
# pairs = [
#     ('MIMIC-IV', 'Wikipedia'),
#     ('PubMed', 'Wikipedia'),
#     ('PubMed', 'MIMIC-IV'),
# ]

# # Noise level for jitter
# noise_level = 0.002

# # Output directory
# out_dir = "figs"
# os.makedirs(out_dir, exist_ok=True)

# for x_col, y_col in pairs:
#     # Compute averages for this pair
#     df = df_full.copy()
#     df['x_avg'] = df[x_col]
#     df['y_avg'] = df[y_col]
    
#     # Rename our model entry
#     df['label'] = df['model_name'].replace({our_model: 'Ours'})
    
#     # Start plotting
#     plt.figure(figsize=(16, 12))
#     texts = []
    
#     # Scatter all buckets
#     for bucket, group in df.groupby('bucket'):
#         plt.scatter(
#             group['x_avg'], group['y_avg'],
#             s=150,
#             label=bucket.replace('_', ' ').title(),
#             color=bucket_colour[bucket],
#             alpha=0.7,
#             edgecolors='w', linewidth=0.5
#         )
#         for _, row in group.iterrows():
#             x_j = row['x_avg'] + np.random.uniform(-noise_level, noise_level)
#             y_j = row['y_avg'] + np.random.uniform(-noise_level, noise_level)
#             texts.append(plt.text(x_j, y_j, row['label'], fontsize=8))
    
#     # Highlight “Ours”
#     ours = df[df['label'] == 'Ours'].iloc[0]
#     plt.scatter(
#         ours['x_avg'], ours['y_avg'],
#         s=250, marker='*', color='red',
#         edgecolors='k', linewidth=1.2, label='Ours'
#     )
    
#     # Adjust text
#     adjust_text(texts)
    
#     # Labels, legend, grid, title
#     plt.xlabel(f'Performance on {x_col}')
#     plt.ylabel(f'Performance on {y_col}')
#     plt.title(f'{x_col} vs {y_col}: The Effect of Training Data and Objective on Perforamcne')
#     plt.legend(title='Model Bucket', loc='best', frameon=True)
#     plt.grid(True, linestyle='--', alpha=0.4)
#     plt.tight_layout()
    
#     # Save
#     stem = f"style_vs_{x_col.lower().replace('-', '').replace(' ', '')}_{y_col.lower()}"
#     plt.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
#     plt.savefig(os.path.join(out_dir, f"{stem}.jpg"), dpi=300, bbox_inches="tight")
#     plt.close()






































# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # Filepath
# perf_path = 'final_model_summary.csv'

# # Load data
# perf = pd.read_csv(perf_path)

# # Tasks to visualize
# tasks = ['classification', 'clustering', 'pair_classification', 'retrieval']

# # Clean the ± values: keep only the numeric part before the ±
# for task in tasks:
#     perf[task] = (
#         perf[task]
#         .astype(str)
#         .str.split('±')
#         .str[0]
#         .astype(float)
#     )

# # Buckets & colors
# models_by_bucket = {
#     "Medical Contrastive": [
#         "Abhinand MedEmbed Base", "Kamalkraj BioSimCSE BioLinkBERT Base",
#         "Malteos SciNCL", "Skyfury CTMEDGTE Cl15 Step 8000",
#     ],
#     "Medical Non-Contrastive": [
#         "BioNLP Bluebert PubMed Mimic Uncased L-12 H-768 A-12",
#         "EmilyAlsentzer Bio ClinicalBERT", "MedicalAI ClinicalBERT",
#         "Microsoft BiomedNLP BiomedBERT Base Uncased Abstract Fulltext",
#     ],
#     "Non-Medical Contrastive": [
#         "BAAI Bge Base En V1.5", "Intfloat E5 Base",
#         "Nomic AI Nomic Embed Text V1", "Nomic AI Nomic Embed Text V1 Unsupervised",
#         "Sentence-Transformers All MiniLM L6 V2", "Sentence-Transformers All MPNet Base V2",
#         "Thenlper GTE Base",
#     ],
#     "Non-Medical Non-Contrastive": [
#         "AllenAI Scibert Scivocab Uncased", "Google BERT Base Uncased",
#     ],
# }
# bucket_colour = {
#     "Medical Contrastive":        "tab:blue",
#     "Medical Non-Contrastive":    "tab:cyan",
#     "Non-Medical Contrastive":    "tab:orange",
#     "Non-Medical Non-Contrastive":"tab:green",
# }

# # Our model
# our_model = "Skyfury CTMEDGTE Cl15 Step 8000"

# # Create output directory
# os.makedirs('figs', exist_ok=True)

# # Create figure and axes
# fig, axes = plt.subplots(len(tasks), len(tasks), figsize=(16, 16))

# # Plot scatter for each pair
# for i, y_task in enumerate(tasks):
#     for j, x_task in enumerate(tasks[::-1]): #???
#         ax = axes[i, j]
#         # Plot buckets
#         for bucket, models in models_by_bucket.items():
#             subset = perf[perf['model_name'].isin(models)]
#             ax.scatter(
#                 subset[x_task],
#                 subset[y_task],
#                 color=bucket_colour[bucket],
#                 s=50,
#                 alpha=0.7
#             )
#         # Highlight our model
#         our_data = perf[perf['model_name'] == our_model]
#         ax.scatter(
#             our_data[x_task],
#             our_data[y_task],
#             color='red',
#             marker='*',
#             s=200
#         )
#         # Axis labels and ticks
#         if i == len(tasks)-1:
#             ax.set_xlabel(x_task, fontsize=9)
#         else:
#             ax.set_xticklabels([])
#         if j == 0:
#             ax.set_ylabel(y_task, fontsize=9)
#         else:
#             ax.set_yticklabels([])
#         ax.tick_params(axis='both', which='major', labelsize=7)

# # Create custom legend handles
# legend_handles = []
# legend_labels = []
# for bucket, color in bucket_colour.items():
#     legend_handles.append(Line2D([], [], marker='o', color=color, linestyle='None', markersize=8))
#     legend_labels.append(bucket)
# # Our model handle
# legend_handles.append(Line2D([], [], marker='*', color='red', linestyle='None', markersize=12))
# legend_labels.append("Our Model")

# # Add horizontal legend bar below with box
# legend = fig.legend(
#     legend_handles,
#     legend_labels,
#     loc='lower center',
#     ncol=len(legend_handles),
#     bbox_to_anchor=(0.5, 0.02),
#     frameon=True,
#     framealpha=1,
#     edgecolor='black',
#     fontsize=10
# )

# plt.tight_layout(rect=[0, 0.07, 1, 1])

# # Save figure
# output_path = 'figs/pairwise_scatter.png'
# fig.savefig(output_path, dpi=300)

# output_path = 'figs/pairwise_scatter.pdf'
# fig.savefig(output_path, dpi=300)
# plt.close(fig)

# print(f"Figure saved to {output_path}")
