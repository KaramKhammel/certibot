import torch
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import NoReturn, Any, List, Dict, Optional


def LoadEmbeddingModel(checkpoint: str):
    embedding = HuggingFaceEmbeddings(checkpoint)
    return embedding


def LoadTokenizer(checkpoint: str, kwargs: Optional[Dict]):
    if kwargs == None:
        kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        **kwargs
    )
    return tokenizer

def LoadModel(checkpoint: str, kwargs: Optional[dict]):
    if kwargs == None:
        kwargs = {}
        
    model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    **kwargs
    )

    return model

