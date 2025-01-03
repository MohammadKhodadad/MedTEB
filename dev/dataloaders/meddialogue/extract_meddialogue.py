import os
import pandas as pd
from datasets import load_dataset

def save_meddialog_dataframe(output_file="../data/meddialog_data.csv"):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the MedDialog dataset from Hugging Face
    dataset = load_dataset("bigbio/meddialog", trust_remote_code=True)

    # Initialize lists to store patient and doctor utterances
    patient_utterances = []
    doctor_utterances = []

    # Iterate over each split in the dataset (train, validation, test)
    for split in dataset:
        print(f"Processing split: {split}")
        for example in dataset[split]:
            utterances = example.get("utterances", {})
            speakers = utterances.get("speaker", [])
            texts = utterances.get("utterance", [])

            # Initialize variables to hold conversation segments
            patient_text = ""
            doctor_text = ""

            # Separate the utterances based on the speaker
            for speaker, text in zip(speakers, texts):
                if speaker == 0:  # Assuming 0 indicates the patient
                    patient_text += text + " "
                elif speaker == 1:  # Assuming 1 indicates the doctor
                    doctor_text += text + " "

            # Add the collected utterances to the lists
            patient_utterances.append(patient_text.strip())
            doctor_utterances.append(doctor_text.strip())

    # Create a DataFrame with patient and doctor columns
    df = pd.DataFrame({
        "patient": patient_utterances,
        "doctor": doctor_utterances
    })

    # Save the DataFrame to a CSV file
    df=df.dropna()
    if len(df)>8192:
        df=df.sample(8192)
    df.to_csv(output_file, index=False)
    print(f"MedDialog data saved to {output_file}")

if __name__ == "__main__":
    # Save the MedDialog data to a CSV file
    save_meddialog_dataframe()
