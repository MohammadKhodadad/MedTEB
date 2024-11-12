from datasets import load_dataset

# Load the MedDialog dataset from Hugging Face
dataset = load_dataset("bigbio/meddialog",trust_remote_code=True)

# Display the first few examples from the dataset
print(dataset)