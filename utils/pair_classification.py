from __future__ import annotations
from typing import Any
import datasets
import json
import pandas as pd
from datasets import Dataset
from datasets import Dataset, DatasetDict
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from mteb import MTEB, get_model
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
def create_pair_classification_task(data_address):
    class CustomAbsTaskPairClassification(AbsTaskPairClassification):
        name='pairclassification_'+data_address.split('/')[-1].replace('.csv','')
        metadata = TaskMetadata(
            name=name,
            dataset={"path": "",
                    "revision": " ",

            },
            description="",
            reference="https://errors.pydantic.dev/2.9/v/url_parsing",
            type="PairClassification",
            category="s2p",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="cosine_f1",
            date=None,
            domains=None,
            task_subtypes=None,
            license=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="""""",
            descriptive_stats={}
        )


        def load_data(self, **kwargs: Any) -> None:
            """Load dataset from HuggingFace hub or use sample data"""
            if self.data_loaded:
                return
            data = pd.read_csv(data_address)
            
            self.dataset = {
                "test": [{'sentence1':[],'sentence2':[],'labels':[]}]}
            for i in range(len(data)):
                    row = data.iloc[i]
                    self.dataset["test"][0]['sentence1'] .append(row['sentence1'])
                    self.dataset["test"][0]['sentence2'] .append(row['sentence2'])
                    self.dataset["test"][0]['labels'] .append(row['label'])
            self.data_loaded = True

        def dataset_transform(self):
            pass
            self.data_loaded = True

        def dataset_transform(self):
            pass
    return CustomAbsTaskPairClassification()
if __name__ == "__main__":
    task = create_pair_classification_task('..\\data\\pair_classification\\mimic_Chief_Complaint_vs_Discharge_Diagnosis.csv')
    mteb = MTEB(tasks=[task])
    # model = get_model("bert-base-uncased")
    # model = get_model("sentence-transformers/all-MiniLM-L6-v2")
    # model = get_model('intfloat/e5-base')
    # model = get_model("medicalai/ClinicalBERT")
    # model = get_model("nomic-ai/nomic-embed-text-v1")
    model = get_model("emilyalsentzer/Bio_ClinicalBERT")
    # model = get_model('kamalkraj/BioSimCSE-BioLinkBERT-BASE')
    # model = get_model('malteos/scincl')
    
    # model = HuggingFaceEmbedder('nomic-ai/nomic-embed-text-v1')
    # model = HuggingFaceEmbedder('intfloat/e5-base')
    # transformer= models.Transformer('intfloat/e5-base-v2')
    # pooling_model = models.Pooling(
    #     transformer.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=False,  # Use CLS token for pooling
    #     pooling_mode_mean_tokens=True,
    #     pooling_mode_max_tokens=False
    # )

    # transformer= models.Transformer('/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/contrastive/step_45000',tokenizer_name_or_path ='bert-base-uncased')
    # pooling_model = models.Pooling(
    #     transformer.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=True,  # Use CLS token for pooling
    #     pooling_mode_mean_tokens=False,
    #     pooling_mode_max_tokens=False
    # )
    # model = SentenceTransformer(modules=[transformer, pooling_model])
    results = mteb.run(model)
    print("Evaluation results:", results)
    # for task in results:
    #   print(task)