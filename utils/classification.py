from __future__ import annotations
from typing import Any
import datasets
import json
import pandas as pd
from datasets import Dataset
from datasets import Dataset, DatasetDict
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from mteb import MTEB, get_model
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import train_test_split
def create_classification_task(data_address):
    class CustomAbsTaskClassification(AbsTaskClassification):
        metadata = TaskMetadata(
            name="PubMedTitleAbsClassification",
            dataset={"path": "",
                    "revision": " ",

            },
            description="",
            reference="https://errors.pydantic.dev/2.9/v/url_parsing",
            type="Classification",
            category="p2p",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="f1",
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
                "test": {'text':[],'label':[]},
                "train":{'text':[],'label':[]}}
            train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

            # Populate the train dataset
            for _, row in train_data.iterrows():
                self.dataset["train"]['text'].append(row['text'])
                self.dataset["train"]['label'].append(row['label'])

            # Populate the test dataset
            for _, row in test_data.iterrows():
                self.dataset["test"]['text'].append(row['text'])
                self.dataset["test"]['label'].append(row['label'])
            self.data_loaded = True

        def dataset_transform(self):
            pass
            self.data_loaded = True

        def dataset_transform(self):
            pass
    return CustomAbsTaskClassification()
if __name__ == "__main__":
    task = create_classification_task('..\\data\\clustering\\wiki_broad_medical_topics_dataset.csv')
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