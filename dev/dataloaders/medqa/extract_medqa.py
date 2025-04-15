import os
import pandas as pd
from datasets import load_dataset

def create_medqa_pair_classification(output_file="../data/medqa_pair_classification.csv"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the MedQA dataset from Hugging Face
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

    # Initialize lists for pair classification
    sentence1 = []  # Questions
    sentence2 = []  # Options
    labels = []     # 1 for correct, 0 for incorrect

    # Iterate over each split in the dataset (train, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            question = example.get("question", None)
            correct_answer_idx = example.get("answer_idx", None)  # Index of the correct answer (0-based)
            options = example.get("options", None)

            # Create pairs for each option
            if correct_answer_idx and options and question:
                for key in options.keys():
                    sentence1.append(question)
                    sentence2.append(options[key])
                    labels.append(1 if key == correct_answer_idx else 0)

    # Create a DataFrame with the pair classification data
    df = pd.DataFrame({
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": labels
    })

    # Save the DataFrame to a CSV file
    df=df.dropna()
    if len(df)>16384:
        df=df.sample(16384)
    df.to_csv(output_file, index=False)
    
    print(f"Pair classification data saved to {output_file}")

if __name__ == "__main__":
    # Create and save the pair classification dataset for MedQA
    create_medqa_pair_classification()
