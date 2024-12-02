import os
from dataloaders.wikipedia.extract_wikipedia_documents import wiki_fetch_data_from_categories
from dataloaders.pubmed.extract_pubmed_papers import pubmed_fetch_and_save_articles_by_category
from dataloaders.pmc.extract_pmc_papers import pmc_fetch_and_save_articles_by_category






print("WIKI TASKS...")
clustering_tasks = {
    "Disease Clustering by System": [
        "Cardiovascular diseases",
        "Respiratory diseases",
        "Neurological disorders",
        "Musculoskeletal disorders",
        "Digestive diseases"
    ],
    "Cardiovascular vs Respiratory": [
        "Cardiovascular diseases",
        "Respiratory diseases"
    ],
    "Cardiovascular vs Digestive": [
        "Cardiovascular diseases",
        "Digestive diseases"
    ],
    "Syndromes and Symptoms": [ # Add signs
        "Syndromes",
        "Symptoms"
    ],
    "Syndromes":[
        "Syndromes affecting the eye",
        "Syndromes affecting blood",
        "Syndromes affecting the nervous system",
        "Syndromes affecting hearing"
    ],
    "Pain vs Fever":[
        "Pain",
        "Fever",
    ],
    "Infection vs Cancer":[
        "Infectious diseases",
        "Cancer"
    ],
    "Viral vs Bacterial":[
        "Viral diseases",
        "Bacterial diseases",
    ],
    "Broad Medical Topics": [
        "Diseases and disorders",
        "Therapy",
        "Symptoms",
        "Pathology"
    ],
    "Core Medical Specialties": [
        "Cardiology",
        "Neurology",
        "Gastroenterology",
        "Endocrinology",
        "Pulmonology"
    ],
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
    output_file = f"../data/clustering/wiki_{task_name.replace(' ', '_').lower()}_dataset.json"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"\nProcessing task: {task_name}")
        data = wiki_fetch_data_from_categories(categories, max_pages_per_category=500, output_dir=output_file, max_depth=1)
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
    "Treatment_Prevention": {
        "Treatment": "treatment",
        "Prevention": "prevention"
    },
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

    output_file = f"../data/clustering/pubmed_{task_name.replace(' ', '_').lower()}_dataset.json"

    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"Downloading data for task: {task_name}")
        data = pubmed_fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file=output_file)
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
    "Diagnostic_vs_Therapeutic": {
        "Diagnostic": "diagnostic",
        "Therapeutic": "therapeutic"
    },
    "Types_of_Interventions": {
        "Drug-based": "drug-based",
        "Device-based": "device-based",
        "Behavioral": "behavioral",
        "Surgical": "surgical"
    }
}



for task_name, categories in tasks.items():

    output_file = f"../data/clustering/pmc_{task_name.replace(' ', '_').lower()}_dataset.json"

    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"Downloading data for task: {task_name}")
        data = pmc_fetch_and_save_articles_by_category(categories, max_articles_per_category=500, output_file=output_file)
        for category in data.keys():
            print(category,':',len(data[category]))
