import pandas as pd
import json
import tqdm


def mimic_create_classification_data(data_address='../data/discharge_processed.csv',
                               cols={"class1":['small bowel obstruction'], "class2":['acute appendicitis'], "class3":['acute cholecystitis']},
                               output_dir="../data/mimic_classification_data.json"):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Convert 'Discharge Diagnosis' to lowercase
    data['Discharge Diagnosis'] = data['Discharge Diagnosis'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    data['text']= 'Chief Complaint: '+data['Chief Complaint']+ ' History of Present Illness: '+ data['History of Present Illness']
    classification_data={'text':[],'label':[]}

    for class_name, classes in cols.items():
        class_data = data[data['Discharge Diagnosis'].apply(lambda x: x in [class_.lower() for class_ in classes])]
        # print(class_data.shape)
        for _, row in class_data.iterrows():
            # Create a dictionary of relevant fields
            classification_data['text'].append(row['text'])
            classification_data['label'].append(class_name)
    classification_data=pd.DataFrame(classification_data)
    classification_data=classification_data.dropna()
    if len(classification_data)>8192:
        classification_data=classification_data.sample(8192)
    classification_data[['text','label']].to_csv(output_dir)
    # # Initialize the output structure
    # classification_data = {col: [] for col in cols.keys()}
    
    # # Filter relevant columns that can be used for classification
    # relevant_columns = [
    #     'Admission Date', 'Brief Hospital Course', 'Chief Complaint',
    #     'Discharge Condition', 'Discharge Date', 'Discharge Diagnosis',
    #     'Discharge Instructions', 'History of Present Illness',
    #     'Major Surgical or Invasive Procedure', 'Past Medical History',
    #     'Pertinent Results', 'Physical Exam', 'Sex', 'Social History'
    # ]
    
    # # Iterate over each class and collect the relevant data
    # for class_name, classes in cols.items():
        
    #     class_data = data[data['Discharge Diagnosis'].apply(lambda x: x in [class_.lower() for class_ in classes])]
    #     # print(class_data.shape)
    #     for _, row in class_data.iterrows():
    #         # Create a dictionary of relevant fields
    #         entry = {field: row[field] for field in relevant_columns if field in row}
    #         classification_data[class_name].append(entry)
    
    # # Save the structured data to a JSON file
    # with open(output_dir, 'w', encoding='utf-8') as file:
    #     json.dump(classification_data, file, indent=4)

    # print(f"Data saved to {output_dir}")


import pandas as pd

def mimic_create_readmission_dataset(data_address='../data/discharge_processed_v_3.csv',
                               readmission_days=120,output_dir='../data/discharge_processed_with_readmission.csv'):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Convert dates to datetime format
    data['admittime'] = pd.to_datetime(data['admittime'], errors='coerce')
    data['dischtime'] = pd.to_datetime(data['dischtime'], errors='coerce')
    
    # Sort by subject_id and Admission Date
    data = data.sort_values(by=['subject_id_x', 'admittime'])
    
    # Initialize a new column for readmission
    data['Readmission'] = 0
    
    # Iterate over each patient to check for readmissions
    for subject_id in tqdm.tqdm(data['subject_id_x'].unique()):
        patient_data = data[data['subject_id_x'] == subject_id]
        for i in range(len(patient_data) - 1):
            discharge_date = patient_data.iloc[i]['dischtime']
            next_admission_date = patient_data.iloc[i + 1]['admittime']
            # Check if the readmission occurred within the specified number of days
            # print((next_admission_date - discharge_date).days)
            if (next_admission_date - discharge_date).days <= readmission_days:
                data.loc[patient_data.index[i], 'Readmission'] = 1
    
    # Select relevant columns for the model
    relevant_columns = [
        'Chief Complaint', 'Brief Hospital Course', 'History of Present Illness',
        'Past Medical History', 'Discharge Instructions', 'Discharge Condition', 'Readmission'
    ]
    
    # Filter the data to include only relevant columns
    readmission_data = data[relevant_columns].dropna()
    c_0= (readmission_data['Readmission']==0).sum()
    c_1= (readmission_data['Readmission']==1).sum()
    readmission_data=pd.concat([readmission_data[readmission_data.Readmission==0].sample(min(c_0,c_1)),readmission_data[readmission_data.Readmission==1].sample(min(c_0,c_1))],axis=0)
    readmission_data['text']='Chief Complaint: '+readmission_data['Chief Complaint']+ ' History of Present Illness: '+ readmission_data['History of Present Illness']
    readmission_data['label']=readmission_data['Readmission']
    readmission_data=readmission_data[['text','label']]
    readmission_data=readmission_data.dropna()
    if len(readmission_data)>8192:
        readmission_data=readmission_data.sample(8192)
    readmission_data.to_csv(output_dir)
    
    print(readmission_data.label.value_counts())
    print(f"Data saved to {output_dir}")
    return readmission_data


import pandas as pd

def mimic_create_retrieval_dataset(data_address='../data/discharge_processed.csv', query_col='Chief Complaint', corpus_col='History of Present Illness',
                             output_dir='../data/discharge_retrieval_dataset.csv'):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Select relevant columns for the retrieval task
    # Example: 'Chief Complaint' as the query and 'Brief Hospital Course' as the corpus
    retrieval_data = data[[query_col, corpus_col]].dropna()
    
    # Rename columns to 'query' and 'corpus'
    retrieval_data.columns = ['query', 'corpus']
    
    # Save the retrieval dataset to CSV
    retrieval_data=retrieval_data.dropna()
    if len(retrieval_data)>16384:
        retrieval_data=retrieval_data.sample(16384)
    retrieval_data.to_csv(output_dir, index=False)
    
    print(f"Retrieval dataset saved to {output_dir}")
    return retrieval_data
    

import pandas as pd
import itertools
import random

import pandas as pd

import pandas as pd
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util

def sample_hard_negatives(
    texts: list[str],
    labels: list[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 32,
    seed: int = 42
) -> list[int]:
    """
    For each text in `texts`, this function computes its embedding and finds the top_k+1 
    most similar texts (including itself). It then filters the candidates by ensuring that 
    the candidate text has a different label (if labels are provided). Finally, for each 
    text, a single candidate index is randomly selected from the remaining candidates.
    
    Args:
        texts: List of texts to embed.
        labels: List of labels corresponding to each text. If provided, negatives are restricted
                to those with a different label than the current text.
        model_name: Name of the SentenceTransformer model to use.
        top_k: Number of top similar candidates to consider (excluding self-match).
        seed: Random seed for reproducibility.
        
    Returns:
        A list of indices (one per text) corresponding to a hard negative candidate.
    """
    # Load model and compute embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    # Compute semantic search results for each embedding. We request top_k+1 so that we can
    # remove the self-match later.
    hits = util.semantic_search(embeddings, embeddings, top_k=top_k + 1)
    
    random.seed(seed)
    hard_negatives = []
    
    # For each text, process its list of hits
    for i, hit_list in enumerate(hits):
        # Exclude the self-match (where corpus_id == query index i)
        candidates = [h["corpus_id"] for h in hit_list if h["corpus_id"] != i]
        # If labels are provided, further filter candidates to ensure a different label
        if labels is not None:
            candidates = [j for j in candidates if labels[j] != labels[i]]
        # If no candidate remains after filtering, fall back to using any candidate (with different index)
        if not candidates:
            candidates = [j for j in range(len(texts)) if j != i and (labels is None or labels[j] != labels[i])]
            if not candidates:
                # As a final fallback, simply use the original candidate list from semantic search
                candidates = [h["corpus_id"] for h in hit_list if h["corpus_id"] != i]
        hard_negatives.append(random.choice(candidates))
    
    return hard_negatives

def mimic_create_pair_classification_dataset(data_address='../data/discharge_processed.csv',
                                             col1='Chief Complaint',
                                             col2='Discharge Diagnosis',
                                             output_file='../data/mimic_pair_classification.csv',
                                             num_samples=1000):
    """
    Create a pair classification dataset for the MIMIC discharge data.
    
    This function loads the dataset, samples a fixed number of rows, and creates positive pairs 
    (by pairing each row with itself) and negative pairs. For negative pairs, a "hard negative" is
    selected using a BERT-based embedding model that returns one candidate from the top 32 most 
    semantically similar texts that have a different label.
    
    Args:
        data_address: Path to the CSV file with discharge data.
        col1: Name of the first column (e.g., 'Chief Complaint').
        col2: Name of the second column (e.g., 'Discharge Diagnosis').
        output_file: File path where the generated dataset will be saved.
        num_samples: Number of rows to sample from the dataset.
        
    Returns:
        A pandas DataFrame containing the pair classification dataset.
    """
    # Load the data and keep only the necessary columns. Drop rows with missing values.
    data = pd.read_csv(data_address)
    data = data[[col1, col2]].dropna()
    
    # Convert col2 (e.g., 'Discharge Diagnosis') to lowercase for uniformity.
    data[col2] = data[col2].str.lower()
    
    # Sample a specified number of rows from the full dataset.
    sampled_data = data.sample(num_samples, replace=False).reset_index(drop=True)
    
    # Extract texts and labels for negative sampling.
    texts = sampled_data[col2].astype(str).tolist()
    # Here, we use the discharge diagnosis as the label.
    labels = texts.copy()
    
    # Get hard negative indices using the helper function.
    hard_neg_indices = sample_hard_negatives(texts, labels=labels, top_k=128, seed=42)
    
    # Initialize lists for positive and negative pairs.
    positive_pairs = []
    negative_pairs = []
    
    # Iterate over the sampled rows and build pairs.
    for i, row in sampled_data.iterrows():
        # Create a positive pair (label 1)
        positive_pairs.append((row[col1], row[col2], 1))
        # Create a negative pair (label 0) by pairing with the hard negative candidate.
        neg_text = texts[hard_neg_indices[i]]
        negative_pairs.append((row[col1], neg_text, 0))
    
    # Combine positive and negative pairs.
    pairs = positive_pairs + negative_pairs
    
    # Convert to a DataFrame.
    pair_df = pd.DataFrame(pairs, columns=['sentence1', 'sentence2', 'label'])
    pair_df = pair_df.dropna()
    
    # Optionally, restrict the total number of examples.
    if len(pair_df) > 16384:
        pair_df = pair_df.sample(16384)
    
    # Save to CSV.
    pair_df.to_csv(output_file, index=False)
    print(f"Pair classification dataset saved to {output_file}")
    
    return pair_df







if __name__ == "__main__":
    mimic_create_classification_data()
    # readmission_dataset = create_readmission_dataset()
    # print(readmission_dataset.Readmission.value_counts())
    # retrieval_data= create_retrieval_dataset()
    # print(retrieval_data.head())
    # pair_classification_data = mimic_create_pair_classification_dataset()
    # print(pair_classification_data.head())


