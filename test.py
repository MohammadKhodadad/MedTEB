from utils.classification import create_classification_task
from utils.clustering import create_clustering_task
from utils.pair_classification import create_pair_classification_task
from utils.retrieval import create_retrieval_task
from mteb import MTEB, get_model
from sentence_transformers import SentenceTransformer
import os
import glob

model_names = [
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base",
    "medicalai/ClinicalBERT",
    "nomic-ai/nomic-embed-text-v1",
    "emilyalsentzer/Bio_ClinicalBERT",
    "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    "malteos/scincl"
]
model_names = [
    "intfloat/e5-base",
]
# Clustering
os.chdir('data/clustering')
addresses=os.listdir('.')
for address in addresses:
    if '.csv' in address:
        task = create_clustering_task(address)
        mteb = MTEB(tasks=[task])
        for name in model_names:
            model = SentenceTransformer(name)
            results = mteb.run(model)