import os
from dataloaders.wikipedia.extract_wikipedia_documents import wiki_fetch_data_from_categories
from dataloaders.pubmed.extract_pubmed_papers import pubmed_fetch_and_save_articles_by_category
from dataloaders.pmc.extract_pmc_papers import pmc_fetch_and_save_articles_by_category
import random

from dataloaders.mimiciv.Mimic_datasets import mimic_create_readmission_dataset, mimic_create_classification_data


categories_ = ["clustering", "classification"]



print("WIKI TASKS...")
clustering_tasks = {
    "Disease by System": [
        "Cardiovascular diseases",
        "Respiratory diseases",
        "Neurological disorders",
        "Musculoskeletal disorders",
        "Digestive diseases"
    ],
    # "Cardiovascular vs Respiratory": [
    #     "Cardiovascular diseases",
    #     "Respiratory diseases"
    # ],
    # "Cardiovascular vs Digestive": [
    #     "Cardiovascular diseases",
    #     "Digestive diseases"
    # ],
    # "Syndromes and Symptoms": [ # Add signs
    #     "Syndromes",
    #     "Symptoms"
    # ],
    "Syndromes":[
        "Syndromes affecting the eye",
        "Syndromes affecting blood",
        "Syndromes affecting the nervous system",
        "Syndromes affecting hearing"
    ],
    # "Pain vs Fever":[
    #     "Pain",
    #     "Fever",
    # ],
    # "Infection vs Cancer":[
    #     "Infectious diseases",
    #     "Cancer"
    # ],
    # "Viral vs Bacterial":[
    #     "Viral diseases",
    #     "Bacterial diseases",
    # ],
    "Broad Medical Topics": [
        "Diseases and disorders",
        "Therapy",
        "Symptoms",
        "Pathology"
    ],
    # "Core Medical Specialties": [
    #     "Cardiology",
    #     "Neurology",
    #     "Gastroenterology",
    #     "Endocrinology",
    #     "Pulmonology"
    # ],
    "Surgical Specialties": [
        "Orthopedics",
        "Ophthalmology",
        "Gynaecology",
        "Neurosurgery",
        "Transplantation medicine"
    ],
    "Diagnostic and Interdisciplinary Fields": [
        "Radiology",
        "Pathology",
        "Nuclear medicine",
        "Anesthesiology",
        "Intensive care medicine"
    ],
    "Preventive and Community Health": [
        "Public health",
        "Preventive medicine",
        "Occupational medicine",
        "Sports medicine",
        "Tropical medicine"
    ],
    "Special Populations and Focused Fields": [
        "Pediatrics",
        "Geriatrics",
        "Men's health",
        "Women's health",
        "Addiction medicine"
    ]
}


# Fetch and save data for each clustering task

for task_name, categories in clustering_tasks.items():
    # Randomly select one word
    cat_ = random.choice(categories_)
    cat_='clustering'
    output_file = f"../data/{cat_}/wiki_{task_name.replace(' ', '_').lower()}_dataset.csv"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"\nProcessing task: {task_name}")
        data = wiki_fetch_data_from_categories(categories, max_pages_per_category=1000, output_dir=output_file, max_depth=1)
        for category in data.keys():
            print(category,':',len(data[category]))

print("PubMED TASKS...")




tasks = {
    "Hypertension_Diabetes_Cancer_Alzheimers_Influenza": {
        "Hypertension": "hypertension",
        "Diabetes": "diabetes",
        "Cancer": "cancer",
        "Alzheimer's disease": "alzheimer's disease",
        "Influenza": "influenza"
    },
    # "Treatment_Prevention": {
    #     "Treatment": "treatment",
    #     "Prevention": "prevention"
    # },
    "Chronic_Infectious_Genetic_Autoimmune": {
        "Chronic diseases": "chronic disease",
        "Infectious diseases": "infectious disease",
        "Genetic disorders": "genetic disorder",
        "Autoimmune disorders": "autoimmune disorder"
    },
    "Inflammation_Signaling_Metabolism_Immunity": {
        "Inflammation": "inflammation",
        "Cellular signaling": "cellular signaling",
        "Metabolism": "metabolism",
        "Immunity": "immunity"
    }
}

for task_name, categories in tasks.items():
    cat_ = random.choice(categories_)
    cat_='clustering'
    output_file = f"../data/{cat_}/pubmed_{task_name.replace(' ', '_').lower()}_dataset.csv"

    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"Downloading data for task: {task_name}")
        data = pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=1000, output_file=output_file)
        for category in data.keys():
            print(category,':',len(data[category]))


pmc_tasks = {
    "Medical_Imaging_Types": {
        "MRI": "MRI",
        "CT scan": "CT scan",
        "Ultrasound": "Ultrasound",
        "X-ray": "X-ray",
        "PET scan": "PET scan"
    },
    # "Diagnostic_vs_Therapeutic": {
    #     "Diagnostic": "diagnostic",
    #     "Therapeutic": "therapeutic"
    # },
    "Types_of_Interventions": {
        "Drug-based": "drug-based",
        "Device-based": "device-based",
        "Behavioral": "behavioral",
        "Surgical": "surgical"
    }
}



for task_name, categories in pmc_tasks.items():
    cat_ = random.choice(categories_)
    cat_='clustering'
    output_file = f"../data/{cat_}/pmc_{task_name.replace(' ', '_').lower()}_dataset.csv"

    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"Downloading data for task: {task_name}")
        data = pmc_fetch_and_save_articles_by_category(categories, max_articles_per_category=1000, output_file=output_file)
        for category in data.keys():
            print(category,':',len(data[category]))




# # Readmission

# for t in [3,7,30]:
#     cat_ = random.choice(categories_)
#     cat_='clustering'
#     output_file=f'../data/{cat_}/mimiciv_readmission_{t}_days.csv'
#     if os.path.exists(output_file):
#         print(f"{output_file} already exists")
#     else:
#         mimic_create_readmission_dataset(data_address='./dataloaders/data/discharge_processed_v_3.csv',
#                                    readmission_days=t,output_dir=output_file)
        



specific_classification_tasks = {
    "Infectious_Diseases": {
        "Respiratory": ["Pneumonia", "COPD exacerbation"],
        "Soft_Tissue": ["Cellulitis"],
        "Urinary_Tract": ["Urinary tract infection"],
        "Biliary_System": ["Choledocholithiasis"]
    },
    "Abdominal_Conditions": {
        "Obstructions": ["Small bowel obstruction", "Partial small bowel obstruction"],
        "Inflammatory": ["Acute appendicitis", "Acute cholecystitis", "Diverticulitis", "Pancreatitis"],
        "Pain": ["Abdominal pain", "Chest pain"]
    },
    # "Musculoskeletal_Disorders": {
    #     "Osteoarthritis": ["Right knee osteoarthritis", "Left knee osteoarthritis", "Right hip osteoarthritis", "Left hip osteoarthritis"],
    #     "Stenosis": ["Lumbar stenosis", "Cervical stenosis"]
    # },
    # "Neurological_Conditions": {
    #     "Seizure_Disorders": ["Epilepsy", "Seizure", "Seizures"],
    #     "Critical_Neurology": ["Acute ischemic stroke", "CNS lymphoma", "Syncope"]
    # },
    "Cancer_Types": {
        "Breast_Cancer": ["Right breast cancer"],
        "Prostate_Cancer": ["PROSTATE CANCER"],
        "Other": ["AML", "Peripheral Arterial Disease"]
    },
    # "Mortality_and_Risk_Factors": {
    #     "Deceased_Status": ["Deceased", "Expired", "Patient expired"],
    #     "Obesity": ["Morbid obesity"]
    # }
}

for task_name, categories in specific_classification_tasks.items():
    cat_ = random.choice(categories_)
    cat_='clustering'
    output_file=f'../data/{cat_}/mimiciv_specific_{task_name}.csv'
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        mimic_create_classification_data(data_address='./dataloaders/data/discharge_processed_v_3.csv',cols=categories,output_dir=output_file)

general_classification_tasks = {
    "Infectious_vs_NonInfectious_Conditions": {
        "Infectious_Conditions": ["Pneumonia", "Cellulitis", "Urinary tract infection", "Choledocholithiasis", "COPD exacerbation"],
        "NonInfectious_Conditions": ["Small bowel obstruction", "Acute appendicitis", "Pancreatitis", "Right knee osteoarthritis", "Epilepsy"]
    },
    "Neurological_vs_Musculoskeletal_Conditions": {
        "Neurological_Conditions": ["Seizure", "Acute ischemic stroke", "Syncope", "CNS lymphoma"],
        "Musculoskeletal_Conditions": ["Right knee osteoarthritis", "Left knee osteoarthritis", "Lumbar stenosis", "Cervical stenosis"]
    },
    "Abdominal_vs_Thoracic_Conditions": {
        "Abdominal_Conditions": ["Acute appendicitis", "Acute cholecystitis", "Diverticulitis", "Small bowel obstruction"],
        "Thoracic_Conditions": ["Pneumonia", "COPD exacerbation", "Chest pain"]
    },
    "Cancer_vs_Chronic_Conditions": {
        "Cancer": ["Right breast cancer", "PROSTATE CANCER", "AML"],
        "Chronic_Conditions": ["Peripheral Arterial Disease", "Morbid obesity", "Lumbar stenosis", "COPD exacerbation"]
    },
    "Mortality_vs_Survivable_Conditions": { # Improve it. https://www.england.nhs.uk/ourwork/clinical-policy/sepsis/nationalearlywarningscore/ https://www.phcmedstaff.ca/catch-them-before-they-crash-implementation-of-the-national-early-warning-score-news/
        "Mortality_Conditions": ["Deceased", "Expired", "Patient expired", "Acute ischemic stroke"],
        "Survivable_Conditions": ["Abdominal pain", "Right knee osteoarthritis", "Epilepsy", "Pneumonia"]
    }
}


for task_name, categories in general_classification_tasks.items():
    cat_ = random.choice(categories_)
    cat_='clustering'
    output_file=f'../data/{cat_}/mimiciv_general_{task_name}.csv'
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        mimic_create_classification_data(data_address='./dataloaders/data/discharge_processed_v_3.csv',cols=categories,output_dir=output_file)