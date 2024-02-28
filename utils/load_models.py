import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import NoReturn, Any, List, Dict, Optional


def LoadEmbeddingModel(checkpoint: str):
    embedding = HuggingFaceEmbeddings(model_name=checkpoint)
    return embedding


def LoadTokenizer(checkpoint: str, kwargs: Optional[Dict]=None):
    if kwargs == None:
        kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        **kwargs
    )
    return tokenizer

def LoadModel(checkpoint: str, kwargs: Optional[Dict]=None):
    if kwargs == None:
        kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    **kwargs
    )

    return model

