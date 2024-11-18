import os
import pandas as pd
from datasets import load_dataset

def download_and_save_medmcqa(output_file="../data/medmcqa_data.csv"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the MedMCQA dataset from Hugging Face
    dataset = load_dataset("openlifescienceai/medmcqa")

    # Initialize lists to store the data
    ids = []
    questions = []
    opa = []
    opb = []
    opc = []
    opd = []
    correct_option = []
    choice_type = []
    explanations = []
    subject_names = []
    topic_names = []

    # Iterate over each split in the dataset (train, validation, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            ids.append(example.get("id", ""))
            questions.append(example.get("question", ""))
            opa.append(example.get("opa", ""))
            opb.append(example.get("opb", ""))
            opc.append(example.get("opc", ""))
            opd.append(example.get("opd", ""))
            correct_option.append(example.get("cop", ""))
            choice_type.append(example.get("choice_type", ""))
            explanations.append(example.get("exp", ""))
            subject_names.append(example.get("subject_name", ""))
            topic_names.append(example.get("topic_name", ""))

    # Create a DataFrame with the collected data
    df = pd.DataFrame({
        "id": ids,
        "question": questions,
        "opa": opa,
        "opb": opb,
        "opc": opc,
        "opd": opd,
        "correct_option": correct_option,
        "choice_type": choice_type,
        "explanation": explanations,
        "subject_name": subject_names,
        "topic_name": topic_names
    })

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"MedMCQA data saved to {output_file}")

if __name__ == "__main__":
    # Download and save the MedMCQA dataset
    download_and_save_medmcqa()
