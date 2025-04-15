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


def medmc_qa_create_pair_classification_data(output_file="../data/medmcqa_pair_classification.csv"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the MedMCQA dataset from Hugging Face
    dataset = load_dataset("openlifescienceai/medmcqa")

    # Initialize lists for pair classification
    sentence1 = []  # Questions
    sentence2 = []  # Options
    labels = []     # 1 for correct, 0 for incorrect

    # Iterate over each split in the dataset (train, validation, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            question = example.get("question", "")
            correct_option = example.get("cop", "")  # Correct option index (1-based)

            # Create pairs for each option
            for i, option_text in enumerate([example.get("opa", ""), example.get("opb", ""), 
                                             example.get("opc", ""), example.get("opd", "")], start=0):
                if option_text:  # Ensure the option text is not empty
                    sentence1.append(question)
                    sentence2.append(option_text)
                    labels.append(1 if str(i) == str(correct_option) else 0)

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


def medmc_qa_create_retrieval_dataset(output_file="../data/medmcqa_retrieval_dataset.csv"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the MedMCQA dataset from Hugging Face
    dataset = load_dataset("openlifescienceai/medmcqa")

    # Initialize lists for retrieval
    queries = []      # Questions
    documents = []    # Explanations

    # Iterate over each split in the dataset (train, validation, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            question = example.get("question", "")
            explanation = example.get("exp", "")

            # Ensure both question and explanation are not empty
            if question and explanation:
                queries.append(question)
                documents.append(explanation)

    # Create a DataFrame with the retrieval data
    df = pd.DataFrame({
        "query": queries,
        "corpus": documents
    })

    # Save the DataFrame to a CSV file
    df=df.dropna()
    if len(df)>16384:
        df=df.sample(16384)
    df.to_csv(output_file, index=False)
    print(f"Retrieval dataset saved to {output_file}")


if __name__ == "__main__":
    # download_and_save_medmcqa()
    # create_pair_classification_data()
    create_retrieval_dataset()
