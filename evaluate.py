import os
import json
import glob
import pandas as pd

def load_json_files():
    base_dirs = [
        'data/classification/results',
        'data/clustering/results',
        'data/pair_classification/results',
        'data/retrieval/results'
    ]

    json_data = []

    for base_dir in base_dirs:
        # Use glob to find all JSON files in subdirectories
        json_files = glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True)

        for json_file in json_files:
            if any(keyword in json_file for keyword in ['classification', 'clustering', 'pair_classification', 'retrieval']):
                # Open and load the JSON file
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        name=json_file.split('/')[-3]
                        data['model_name']=name
                        json_data.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

    return json_data

def read_results(data):
    rows=[]
    for item in data:
        task_name = item.get("task_name", "")
        model_name = item.get("model_name", "")
        scores = item.get("scores", {})
        test_scores = scores.get("test", [])

        # Determine the task type and extract the relevant metric
        if len(test_scores)>0:
            test_scores=test_scores[0]
            if "clustering" in task_name:
                metric = test_scores.get("v_measure")
                task_type = "clustering"
            elif "classification" in task_name and "pair_classification" not in task_name:
                metric = test_scores.get("f1")
                task_type = "classification"
            elif "pair_classification" in task_name:
                metric = test_scores.get("similarity_f1")
                task_type = "pair_classification"
            elif "retrieval" in task_name:
                metric = test_scores.get("ndcg_at_10")
                task_type = "retrieval"
            else:
                continue  # Skip if task type is not recognized
        else:
            continue

        # Append the row to the list
        rows.append({"task_name": task_name, "task_type": task_type, "metric": metric,'model_name': model_name})

# Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    return df



# Load all the relevant JSON data
all_json_data = load_json_files()
df= read_results(all_json_data)
df.to_csv("results.csv")

grouped = df.groupby(["task_type", "model_name"])["metric"].agg(["mean", "std"]).reset_index()

# Create the 'mean ± std' column
grouped["mean ± std"] = grouped["mean"].round(2).astype(str) + " ± " + grouped["std"].round(2).astype(str)

# Pivot table with model_name as index and task_type as columns
pivot_table = grouped.pivot(index="model_name", columns="task_type", values="mean ± std")

pivot_table.to_csv('grouped_results.csv')

print(f"Loaded {len(df)} JSON files containing relevant data.")
# print(df.iloc[0])
for col in df.columns:
    print(f'Number of values in column {col}: ',len(df[col].unique()))
print(df['task_name'].unique())