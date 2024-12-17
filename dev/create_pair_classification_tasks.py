import os

from dataloaders.wikipedia.extract_wikipedia_documents import wiki_create_pair_classification_data
from dataloaders.mimiciv.Mimic_datasets import mimic_create_pair_classification_dataset
from dataloaders.medmcqa.extract_medmcqa_dataset import medmc_qa_create_pair_classification_data
from dataloaders.medqa.extract_medqa import create_medqa_pair_classification
from dataloaders.pubmed.extract_pubmed_papers import pubmed_create_pair_classification_data

output_file=f"../data/pair_classification/wiki_diseases_dataset.csv"
if os.path.exists(output_file):
    print(f"{output_file} already exists")
else:
    print(f"Downloading data for task: Diseases and disorders by system")
    wiki_create_pair_classification_data(["Diseases and disorders by system"],10,max_depth=1,output_file=output_file)



tasks=[('Chief Complaint','Discharge Diagnosis'),('History of Present Illness','Chief Complaint'),
       ('Discharge Instructions','History of Present Illness'),('Chief Complaint','Discharge Instructions'),
       ('History of Present Illness','Discharge Diagnosis'),('Discharge Instructions','Chief Complaint')]


for t1,t2 in tasks:

    output_file=f"../data/pair_classification/mimic_{t1.replace(' ','_')}_vs_{t2.replace(' ','_')}.csv"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
    else:
        print(f"Creating data for task: {t1} vs {t2}")
        mimic_create_pair_classification_dataset('./dataloaders/data/discharge_processed_v_3.csv',t1,t2,output_file,10)


output_file=f"../data/pair_classification/medmcqa_pair_classification.csv"
if os.path.exists(output_file):
    print(f"{output_file} already exists")
else:
    print(f"Loading Data for task: medmcqa pairclassification")
    medmc_qa_create_pair_classification_data(output_file)





output_file=f"../data/pair_classification/medqa_pair_classification.csv"
if os.path.exists(output_file):
    print(f"{output_file} already exists")
else:
    print(f"Loading Data for task: medqa pairclassification")
    create_medqa_pair_classification(output_file)


tasks = {
    "medicine": {
        "medicine": "medicine",
    }}
output_file=f"../data/pair_classification/pubmed_pair_classification.csv"
if os.path.exists(output_file):
    print(f"{output_file} already exists")
else:
    print(f"Loading Data for task: pubmed pairclassification")
    pubmed_create_pair_classification_data(tasks,10,output_file)


