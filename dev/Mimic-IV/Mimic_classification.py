import pandas as pd
import json
import tqdm
def create_classification_data(data_address='../data/discharge_processed.csv',
                               cols=['small bowel obstruction', 'acute appendicitis', 'acute cholecystitis'],
                               output_dir="../data/mimic_classification_data.json"):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Convert 'Discharge Diagnosis' to lowercase
    data['Discharge Diagnosis'] = data['Discharge Diagnosis'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Initialize the output structure
    classification_data = {col: [] for col in cols}
    
    # Filter relevant columns that can be used for classification
    relevant_columns = [
        'Admission Date', 'Brief Hospital Course', 'Chief Complaint',
        'Discharge Condition', 'Discharge Date', 'Discharge Diagnosis',
        'Discharge Instructions', 'History of Present Illness',
        'Major Surgical or Invasive Procedure', 'Past Medical History',
        'Pertinent Results', 'Physical Exam', 'Sex', 'Social History'
    ]
    
    # Iterate over each class and collect the relevant data
    for col in cols:
        class_data = data[data['Discharge Diagnosis'].str.contains(col, na=False)]
        for _, row in class_data.iterrows():
            # Create a dictionary of relevant fields
            entry = {field: row[field] for field in relevant_columns if field in row}
            classification_data[col].append(entry)
    
    # Save the structured data to a JSON file
    with open(output_dir, 'w', encoding='utf-8') as file:
        json.dump(classification_data, file, indent=4)

    print(f"Data saved to {output_dir}")


import pandas as pd

def create_readmission_dataset(data_address='../data/discharge_processed.csv',
                               readmission_days=120,output_dir='../data/discharge_processed_with_readmission.csv'):
    # Load the data
    data = pd.read_csv(data_address)
    
    # Convert dates to datetime format
    data['admitdate'] = pd.to_datetime(data['admitdate'], errors='coerce')
    data['dischdate'] = pd.to_datetime(data['dischdate'], errors='coerce')
    
    # Sort by subject_id and Admission Date
    data = data.sort_values(by=['subject_id', 'admitdate'])
    
    # Initialize a new column for readmission
    data['Readmission'] = 0
    
    # Iterate over each patient to check for readmissions
    for subject_id in tqdm.tqdm(data['subject_id'].unique()):
        patient_data = data[data['subject_id'] == subject_id]
        for i in range(len(patient_data) - 1):
            discharge_date = patient_data.iloc[i]['dischdate']
            next_admission_date = patient_data.iloc[i + 1]['admitdate']
            # Check if the readmission occurred within the specified number of days
            print((next_admission_date - discharge_date).days)
            if (next_admission_date - discharge_date).days <= readmission_days:
                data.loc[patient_data.index[i], 'Readmission'] = 1
    
    # Select relevant columns for the model
    relevant_columns = [
        'Chief Complaint', 'Brief Hospital Course', 'History of Present Illness',
        'Past Medical History', 'Discharge Instructions', 'Discharge Condition', 'Readmission'
    ]
    
    # Filter the data to include only relevant columns
    readmission_data = data[relevant_columns].dropna()
    readmission_data.to_csv(output_dir)
    print(f"Data saved to {output_dir}")
    return readmission_data


if __name__ == "__main__":
    # create_classification_data()
    readmission_dataset = create_readmission_dataset()
    print(readmission_dataset.Readmission.value_counts())

