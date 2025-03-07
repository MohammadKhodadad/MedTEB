from __future__ import annotations
from typing import Any
import datasets
import pandas as pd
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata
import pandas as pd
from mteb import MTEB, get_model
def create_clustering_task(data_address):
    class CustomAbsTaskClustering(AbsTaskClustering):
        name='clustering_'+data_address.split('/')[-1].replace('.csv','')
        metadata = TaskMetadata(
            name=name,
            dataset={"path": "",
                    "revision": " ",

            },
            description="",
            reference="https://errors.pydantic.dev/2.9/v/url_parsing",
            type="Clustering",
            category="p2p",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="v_measure",
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
            data=data.dropna()
            self.dataset = {
                "test": [{'sentences':[],'labels':[]}]}
            for i in range(len(data)):
                    row = data.iloc[i]
                    self.dataset["test"][0]['sentences'] .append(row['text'])
                    self.dataset["test"][0]['labels'] .append(row['label'])
            self.data_loaded = True

        def dataset_transform(self):
            pass
            self.data_loaded = True

        def dataset_transform(self):
            pass
    return CustomAbsTaskClustering()
if __name__ == "__main__":
    task = create_clustering_task('..\\data\\clustering\\wiki_broad_medical_topics_dataset.csv')
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