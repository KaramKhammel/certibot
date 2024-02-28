from utils.load_data import load_data
from utils.rag import *
from utils.load_models import *


# importing data
path = '/home/paperspace/certibot/data/certibot-data.jsonl'
dataset = load_data(path)


api_key = "a344a187-8b52-422d-97c2-96628ef67ef6"

index_name = 'certibot-rag'

embedding_dim = 1024
batch_size = 4

# initialize index
vectorstore = init_index(
    api_key=api_key,
    index_name=index_name,
    embedding_dim=embedding_dim
)

# import embedding model
embedding_checkpoint = "intfloat/multilingual-e5-large-instruct"
embedding = LoadEmbeddingModel(embedding_checkpoint)


# upsert data to vectorstore
upsert_to_index(
    index=vectorstore,
    embedding=embedding,
    dataset=dataset,
    batch_size=batch_size
)