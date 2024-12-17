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
    data['label']=data['Discharge Diagnosis']
    data[['text','label']].to_csv(output_dir)
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
    readmission_data['text']='Chief Complaint: '+readmission_data['Chief Complaint']+ ' History of Present Illness: '+ readmission_data['History of Present Illness']
    readmission_data['label']=readmission_data['Readmission']
    readmission_data=readmission_data[['text','label']]
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
    retrieval_data.to_csv(output_dir, index=False)
    
    print(f"Retrieval dataset saved to {output_dir}")
    return retrieval_data
    

import pandas as pd
import itertools
import random

import pandas as pd

import pandas as pd

def mimic_create_pair_classification_dataset(data_address='../data/discharge_processed.csv',
                                       col1='Chief Complaint',
                                       col2='Discharge Diagnosis',
                                       output_file='../data/mimic_pair_classification.csv',
                                       num_samples=1000):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Ensure col1 and col2 are present and drop rows with missing values
    data = data[[col1, col2]].dropna()
    
    # Convert col2 (e.g., 'Discharge Diagnosis') to lowercase for uniformity
    data[col2] = data[col2].str.lower()

    # Sample a specified number of rows
    sampled_data = data.sample(num_samples, replace=False)

    # Initialize lists for positive and negative pairs
    positive_pairs = []
    negative_pairs = []

    # Iterate over the sampled rows to create positive and negative pairs
    for _, row in sampled_data.iterrows():
        # Create a positive pair by pairing the row with itself
        positive_pairs.append((row[col1], row[col2], 1))  # Label 1 for positive pair

        # Find a row with a different label (different 'Discharge Diagnosis') for a negative pair
        different_label_row = data[data[col2] != row[col2]].sample(1, replace=False)
        if not different_label_row.empty:
            negative_pairs.append((row[col1], different_label_row.iloc[0][col2], 0))  # Label 0 for negative pair

    # Combine positive and negative pairs
    pairs = positive_pairs + negative_pairs

    # Convert to a DataFrame
    pair_df = pd.DataFrame(pairs, columns=[f'sentence1', f'sentence2', 'label'])
    # Save to CSV
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


