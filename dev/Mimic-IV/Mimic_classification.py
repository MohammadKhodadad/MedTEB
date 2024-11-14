import pandas as pd
import json

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

if __name__ == "__main__":
    create_classification_data()