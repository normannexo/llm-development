from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings 
import os

mistral_api_key = os.environ["MISTRAL_API_KEY"]

vector_store = Chroma(persist_directory="./chroma_langchain_db", collection_name="greens", embedding_function=MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key))      

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)

docs = retriever.invoke("What say the greens on migration?")

print(docs) 