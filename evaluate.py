import os
import json
import glob
import pandas as pd
# pairclassification_medmcqa_pair_classification
# pairclassification_medqa_pair_classification
# pairclassification_clinical_trials_officialTitle_vsdetailedDescription
# pairclassification_clinical_trials_officialTitle_vsprimaryOutcomes
# retrieval_wiki_diseases_dataset

removed_tasks = """
clustering_wiki_broad_medical_topics_dataset
clustering_mimiciv_general_Neurological_vs_Musculoskeletal_Conditions
clustering_pmc_types_of_interventions_dataset
clustering_pubmed_inflammation_signaling_metabolism_immunity_dataset
clustering_pmc_medical_imaging_types_dataset
clustering_mimiciv_general_Abdominal_vs_Thoracic_Conditions
""".split('\n')+ \
"""classification_wiki_syndromes_and_symptoms_dataset
classification_wiki_viral_vs_bacterial_dataset
classification_wiki_syndromes_dataset
clustering_wiki_syndromes_dataset
clustering_wiki_special_populations_and_focused_fields_dataset""".split('\n') 

removed_tasks += \
"""
classification_pmc_diagnostic_vs_therapeutic_dataset
classification_pmc_medical_imaging_types_dataset
classification_pmc_types_of_interventions_dataset
classification_pubmed_chronic_infectious_genetic_autoimmune_dataset
classification_pubmed_treatment_prevention_dataset
classification_pubmed_inflammation_signaling_metabolism_immunity_dataset""".split('\n') 

removed_tasks += \
"""
classification_wiki_cardiovascular_vs_digestive_dataset
classification_pubmed_hypertension_diabetes_cancer_alzheimers_influenza_dataset
clustering_mimiciv_specific_Cancer_Types
pairclassification_clinical_trials_officialTitle_vsdetailedDescription
retrieval_Clinical Trials""".split('\n')


# removed_tasks += \
# """
# retrieval_clinical_trials_officialTitle_vsdetailedDescription

# """.split('\n')
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
        if task_name in removed_tasks:
            continue
        model_name = item.get("model_name", "")
        scores = item.get("scores", {})
        test_scores = scores.get("test", [])
        evaluation_time = item.get("evaluation_time", None)
        # Determine the task type and extract the relevant metric
        if len(test_scores)>0:
            test_scores=test_scores[0]
            if "clustering" in task_name:
                metric = test_scores.get("v_measure")
                task_type = "clustering"
            elif "classification" in task_name and not ("pair_classification"  in task_name or "pairclassification" in task_name):
                metric = test_scores.get("f1")
                task_type = "classification"
            elif ("pair_classification" in task_name) or ("pairclassification" in task_name):
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
        rows.append({"task_name": task_name, "task_type": task_type, "metric": metric,'model_name': model_name,'evaluation_time':evaluation_time})

# Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    
    return df



# Load all the relevant JSON data
all_json_data = load_json_files()
# df= read_results(all_json_data)
# df.to_csv("results.csv")

# grouped = df.groupby(["task_type", "model_name"])["metric"].agg(["mean", "std"]).reset_index()

# # Create the 'mean ± std' column
# grouped["mean ± std"] = grouped["mean"].round(2).astype(str) + " ± " + grouped["std"].round(2).astype(str)

# # Pivot table with model_name as index and task_type as columns
# pivot_table = grouped.pivot(index="model_name", columns="task_type", values="mean ± std")

# pivot_table.to_csv('grouped_results.csv')

# print(f"Loaded {len(df)} JSON files containing relevant data.")
# # print(df.iloc[0])
# for col in df.columns:
#     print(f'Number of values in column {col}: ',len(df[col].unique()))
# print(df['task_name'].unique())

# for task_type in df.task_type.unique():
#     print(f"Tasks in task_type {task_type}: {len(df[df.task_type==task_type].task_name.unique())}")
# for task_name in df.task_name.unique():
#     print(f'{task_name}\nAverage:{df[df.task_name==task_name]["metric"].mean()}\nOurs:{df[(df.task_name==task_name) & (df.model_name=="skyfury__CTMEDGTE-cl9-step_8500")]["metric"].item()}\nGTE:{df[(df.task_name==task_name) & (df.model_name=="thenlper__gte-base")]["metric"].item()}\n')
# print(df.task_type.value_counts())

# latex_table = pivot_table.to_latex(index=True)

# print(latex_table)

df = read_results(all_json_data)
df.to_csv("results.csv")

# Group and calculate mean ± std
grouped = df.groupby(["task_type", "model_name"])["metric"].agg(["mean", "std"]).reset_index()
grouped["mean ± std"] = grouped["mean"].round(2).astype(str) + " ± " + grouped["std"].round(2).astype(str)

# Pivot table of mean ± std
pivot_table = grouped.pivot(index="model_name", columns="task_type", values="mean ± std")

# === Compute Model-Wise Averages ===
# 1. Average across task *types* (4 values)
task_type_avg = grouped.groupby("model_name")["mean"].mean().round(3).rename("AvgAcrossTaskTypes")

# 2. Average across all individual *tasks* (e.g., 54 tasks)
task_name_avg = df.groupby("model_name")["metric"].mean().round(3).rename("AvgAcrossAllTasks")

# Combine both into a DataFrame
model_averages = pd.concat([task_type_avg, task_name_avg], axis=1)

# Merge with pivot table


final_table = pivot_table.copy()
final_table["AvgAcrossTaskTypes"] = model_averages["AvgAcrossTaskTypes"].astype(str)
final_table["AvgAcrossAllTasks"] = model_averages["AvgAcrossAllTasks"].astype(str)
# 3. Average evaluation time per model
eval_time_avg = df.groupby("model_name")["evaluation_time"].mean().round(2).rename("EvalTime")

# Add to final summary
final_table["EvalTime"] = eval_time_avg.astype(str)

# Save to CSV
final_table.to_csv("final_model_summary.csv")

# LaTeX output if needed
latex_table = final_table.to_latex(index=True)
print(latex_table)