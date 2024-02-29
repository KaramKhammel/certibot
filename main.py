from flask import Flask, request, render_template, jsonify, session
from flask_session import Session

import torch
import warnings

from utils.load_data import load_data
from utils.rag import *
from utils.load_models import *

warnings.filterwarnings('ignore')


app = Flask(__name__)

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'


# importing data
path = '/home/paperspace/certibot/data/certibot-data.jsonl'
dataset = load_data(path)


# initialize index
api_key = "a344a187-8b52-422d-97c2-96628ef67ef6"
index_name = 'certibot-rag'
embedding_dim = 1024

vectorstore = init_index(
    api_key=api_key,
    index_name=index_name,
    embedding_dim=embedding_dim
)


# import embedding model
embedding_checkpoint = "intfloat/multilingual-e5-large-instruct"
model_checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"

model_kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto'}

embedding = LoadEmbeddingModel(embedding_checkpoint)
tokenizer = LoadTokenizer(model_checkpoint)
model = LoadModel(model_checkpoint, model_kwargs)


# upsert data to vectorstore
batch_size = 4
upsert = True

if upsert:
    upsert_to_index(
        index=vectorstore,
        embedding=embedding,
        dataset=dataset,
        batch_size=batch_size
    )


prompt_template= """<s>[INST]Ce qui suit est une conversation entre une IA et un humain.
Tu es un assistant pour les clients de CertiDeal. CertiDeal vends des smartphones reconditionnés.
Ci-dessous est une requête d'un utilisateur ainsi que quelques contextes pertinents.
Réponds à la question en fonction des informations contenues dans ces contexts.
Les réponses doivent être précises et relativement courtes.
Si tu ne trouve pas la réponse à la question, dis "Je ne sais pas"."""


@app.route('/')
def home():
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_reponse():
    user_input = request.form['user_input']
    contexts = retrieve(
        query=user_input,
        embedding=embedding,
        index=vectorstore,
        top_k=4
    )

    # print contexts
    print('---------------------------------------------------------------------')
    print('Contexts:')
    print('---------------------------------------------------------------------')
    for context in contexts:
        print(context)
        print('---------------------------------------------------------------------')

    # retrieve conversation history
    history = session.get('history', [])

    response = generate(
        query=user_input,
        tokenizer=tokenizer,
        model=model,
        contexts=contexts,
        history=history,
        template_instruction=prompt_template,
        device=device
    )


    history.append({'query': user_input, 'response': response})
    session['history'] = history
    
    return jsonify({'response': response})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)