from utils.classification import create_classification_task
from utils.clustering import create_clustering_task
from utils.pair_classification import create_pair_classification_task
from utils.retrieval import create_retrieval_task
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os
import pandas as pd

data=pd.read_csv('/home/skyfury/projects/def-mahyarh/skyfury/medteb/MedTEB/data/pair_classification/mimic_Chief_Complaint_vs_Discharge_Diagnosis.csv')
print(data.label.value_counts())
for col in data.columns:
    print(col,':')
    print(data[col].duplicated().sum())
task = create_pair_classification_task('/home/skyfury/projects/def-mahyarh/skyfury/medteb/MedTEB/data/pair_classification/mimic_Chief_Complaint_vs_Discharge_Diagnosis.csv')
mteb = MTEB(tasks=[task])
# model = get_model(name)
model = SentenceTransformer("skyfury/CTMEDBERT_CLS_Encoder2",trust_remote_code=True)
results = mteb.run(model)
print(results)