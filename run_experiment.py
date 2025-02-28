from utils.classification import create_classification_task
from utils.clustering import create_clustering_task
from utils.pair_classification import create_pair_classification_task
from utils.retrieval import create_retrieval_task
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os

model_names = [
    "bert-base-uncased",
    "BASF-AI/chem-embed-text-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base",
    "medicalai/ClinicalBERT",
    "nomic-ai/nomic-embed-text-v1",
    "emilyalsentzer/Bio_ClinicalBERT",
    "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    "malteos/scincl",
    "thenlper/gte-base",
    "allenai/scibert_scivocab_uncased",
    "BAAI/bge-base-en-v1.5",
    "skyfury/CTMEDBERT_CLS_Encoder",
    "skyfury/CTMEDBERT_CLS_Encoder2",
    "skyfury/CTMEDBERT_CLS_Encoder3",
    "skyfury/CTMEDBERT_CLS_Encoder4",
    "skyfury/CTMEDE5-cl1-step_8000",
    "skyfury/CTMEDGTE-cl1-step_18000",
    "skyfury/CTMEDGTE-cl2-step_12000",
    "skyfury/CTMEDGTE2_encoder",
    "skyfury/CTMEDGTE-cl2-step_22000",
    "skyfury/CTMEDGTE-cl3-step_25000",
    "skyfury/CTMEDGTE-cl4-step_28000",
    "skyfury/CTMEDGTE-cl4-step_44000",
    "skyfury/CTMEDGTE5_encoder",
    "skyfury/CTMEDGTE-cl5-step_22000",
    "skyfury/CTMEDGTE-cl6-step_20000",
    "skyfury/CTMEDGTE6_encoder",
    "skyfury/CTMEDGTE-cl7-step_6000",
    "skyfury/CTMEDGTE7_encoder"


]
# model_names = [
#     "intfloat/e5-base",
# ]
# Clustering
os.chdir('data/clustering')
addresses=os.listdir('.')
for address in addresses:
    if '.csv' in address:
        try:
            task = create_clustering_task(address)
            mteb = MTEB(tasks=[task])
        except:
            print(f"Error in task: {task}")
            continue
        for name in model_names:
            try:
                # model = get_model(name)
                model = SentenceTransformer(name,trust_remote_code=True)
                results = mteb.run(model)
            except:
                print(f"Error with model: {name}")

# Classification

os.chdir('../classification')
addresses=os.listdir('.')
# for address in addresses:
#     if '.csv' in address:
#         task = create_classification_task(address)
#         mteb = MTEB(tasks=[task])
#         for name in model_names:
#             # model = get_model(name)
#             model = SentenceTransformer(name)
#             results = mteb.run(model)
for address in addresses:
    if '.csv' in address:
        try:
            task = create_classification_task(address)
            mteb = MTEB(tasks=[task])
        except:
            print(f"Error in task: {task}")
            continue
        for name in model_names:
            try:
                # model = get_model(name)
                model = SentenceTransformer(name,trust_remote_code=True)
                results = mteb.run(model)
            except:
                print(f"Error with model: {name}")

os.chdir('../pair_classification')
# Pair Classification 
addresses=os.listdir('.')
for address in addresses:
    if '.csv' in address:
        try:
            task = create_pair_classification_task(address)
            mteb = MTEB(tasks=[task])
        except:
            print(f"Error in task: {task}")
            continue
        for name in model_names:
            try:
                # model = get_model(name)
                model = SentenceTransformer(name,trust_remote_code=True)
                results = mteb.run(model)
            except:
                print(f"Error with model: {name}")

os.chdir('../retrieval')
# Retrieval 
addresses=os.listdir('.')
for address in addresses:
    if '.csv' in address:
        try:
            task = create_retrieval_task(address)
            mteb = MTEB(tasks=[task])
        except:
            print(f"Error in task: {task}")
            continue
        for name in model_names:
            try:
                # model = get_model(name)
                model = SentenceTransformer(name,trust_remote_code=True)
                results = mteb.run(model)
            except:
                print(f"Error with model: {name}")
        