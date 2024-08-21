#See https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/

import json
from langchain_caai.caai_emb_client import caai_emb_client
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings

with open('../config.json') as user_file:
    config = json.load(user_file)

llm_api_key = config['llm_api_key']
llm_api_base = config['llm_api_base']
llm_api_base_local = config['llm_api_base_local']

embeddings = caai_emb_client(
    model="",
    api_key=llm_api_key,
    api_url=llm_api_base,
    max_batch_size=100
)

'''
embeddings = OpenAIEmbeddings(
    model="/models/Meta-Llama-3.1-8B-Instruct",
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_base,
)
'''


def process_query(query):

    raw_documents = TextLoader('state_of_the_union.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, embeddings)

    #Similarity search
    docs = db.similarity_search(query)
    ss = docs[0].page_content

    #Similarity search by vector
    embedding_vector = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(embedding_vector)
    ssv = docs[0].page_content

    return ss, ssv

if __name__ == '__main__':

    query = "What did the president say about Ketanji Brown Jackson"
    ss, ssv = process_query(query)
    print('text:', query)
    print('Similarity search:[', ss, ']')
    print('Similarity search by vector:[', ssv, ']')

