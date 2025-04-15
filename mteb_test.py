from sentence_transformers import SentenceTransformer
from mteb import MTEB
import mteb
import json
import os

# ====== Step 1: Load your sentence embedding model ======
# You can change this to your own local model or HuggingFace model
# model_name_or_path = "skyfury/CTMEDGTE-cl9-step_8500"  # Change this if needed
model_name_or_path = "skyfury/CTMEDGTE-cl12-step_8500"  # Change this if needed
model = SentenceTransformer(model_name_or_path)

# ====== Step 2: Define the task you want to evaluate ======
tasks = mteb.get_benchmark("MTEB(Medical)")

# ====== Step 3: Run the evaluation ======
evaluation = MTEB(tasks=tasks)
output_dir = f"results/{tasks[0]}/{model_name_or_path.split('/')[-1]}"

evaluation.run(model, output_folder=output_dir)

# ====== Step 4: Load and display the result ======
results_file = os.path.join(output_dir, "results.json")

if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)
        print("===== Evaluation Results =====")
        print(json.dumps(results, indent=2))
else:
    print("Results file not found.")