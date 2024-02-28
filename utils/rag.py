import pinecone
import time

from typing import NoReturn, Any, List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from pinecone.data import Index
from datasets import Dataset


def init_index(api_key: str, index_name: str, embedding_dim:int):
    pc = pinecone.Pinecone(api_key=api_key)

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pc.list_indexes().names():

        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=embedding_dim,
            metric='cosine',
            spec=pinecone.PodSpec('gcp-starter')
        )

        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # connect to index
    index = pc.Index(index_name)
    
    return index


def upsert_to_index(index, embedding, dataset:Dataset, batch_size:int) -> NoReturn:
    data = dataset.to_pandas()
    for i in range(0, len(data), batch_size):
        end = min(len(data), i + batch_size)
        batch = data[i:end]

        ids = batch['index']
        texts = batch['context']

        embeds = embedding.embed_documents(texts)
        
        metadata = [{
            'context': x['context'],
            'topic': x['topic']
            } for _, x in batch.iterrows()]

        vectors = list(zip(ids, embeds, metadata))
        index.upsert(vectors=vectors)
    

def retrieve(
        query: str,
        embedding: HuggingFaceEmbeddings,
        index: Index,
        top_k: int,
) -> List[str]:
    
    """
    Embeds a user query, retrieves top_k relevant contexts and returns them for
    use by the LLM.
    """

    query_embed = embedding.embed_query(query)
    res = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    contexts = [x['metadata']['context'] for x in res['matches']]

    return contexts




def generate(
        query: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        contexts: List[str],
        history: List[Dict],
        template_instruction: str,
        device: Any
) -> str:
    
    # format retrieved contexts
    context = ""
    for i, text in enumerate(contexts):
        context += f'*{i}: {text}\n\n'
    
    history_prompt = ""
    if history != []:
        for exchange in history:
            history_prompt += f"Q: {exchange['query']}\nA: {exchange['response']}\n"

    # prompt template
    prompt = f"""Instruction du système
    {template_instruction}
    ---------------------------------------------------------------------
    Historique de la conversation:
    {history_prompt}

    Contextes pour inspiration:
    {context}
    
    Question: {query}
    Crée une réponse informative sans recopier la requête. Montre que tu as compris la demande en apportant une réponse précise et pertinente. Réponds en Français. Ne donne pas de références aux contextes.[/INST]
    Réponse:</s>
    """
    eos = '[/INST]'
    
        
    # tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # generate answer
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        temperature=2,
        max_new_tokens=1024,
        )
    
    # decode the output
    output = tokenizer.decode(outputs[0].to(device), skip_special_tokens=True)

    # keep the generated part
    idx = output.index(eos) + len(eos)
    answer = output[idx:].strip()

    if 'Réponse:' in answer:
        answer = answer.replace('Réponse:', '').strip()

    return answer

