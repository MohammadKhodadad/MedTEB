from sentence_transformers import SentenceTransformer, models
from utils.classification import create_classification_task
from utils.clustering import create_clustering_task
from utils.pair_classification import create_pair_classification_task
from utils.retrieval import create_retrieval_task
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import os


# Step 1: Load the transformer model
model_name = "your-username/CTMEDBERT-step60000"
transformer = models.Transformer(model_name)
name='CTMEDBERT'
# Step 2: Define a pooling layer that uses the CLS token
pooling = models.Pooling(
    word_embedding_dimension=transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=False,  # Disable mean pooling
    pooling_mode_cls_token=True,     # Enable CLS token pooling
    pooling_mode_max_tokens=False    # Disable max pooling
)

# Step 3: Combine the transformer and pooling into a SentenceTransformer model
model = SentenceTransformer(modules=[transformer, pooling])


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
        try:
            results = mteb.run(model)
        except:
            print(f"Error with model: {name}")

# Classification

os.chdir('../classification')
addresses=os.listdir('.')
for address in addresses:
    if '.csv' in address:
        try:
            task = create_classification_task(address)
            mteb = MTEB(tasks=[task])
        except:
            print(f"Error in task: {task}")
            continue
        try:
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
        try:
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
        try:
            results = mteb.run(model)
        except:
            print(f"Error with model: {name}")
        