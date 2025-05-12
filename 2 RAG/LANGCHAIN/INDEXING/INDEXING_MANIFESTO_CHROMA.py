from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting RAG indexing process")
logging.info("Loading environment variables")
load_dotenv()  # Loads variables from .env into environment

file_path = "./DATA/EXAMPLEFILES/PDF/20210311_Grundsatzprogramm_EN.pdf"
persist_directory = "./DATA/EXAMPLEFILES/CHROMA/chroma_langchain_db"

try:
    mistral_api_key = os.environ["MISTRAL_API_KEY"]
    logging.info("Successfully loaded Mistral API key")
except KeyError:
    logging.error("Failed to load Mistral API key from environment variables")
    raise


logging.info("Importing Chroma vector store")
from langchain_chroma import Chroma

collection_name = "greens"
logging.info(f"Using collection name: {collection_name}")


logging.info(f"Attempting to load PDF from: {file_path}")

if not os.path.exists(file_path):
    logging.error(f"File not found: {file_path}")
    raise FileNotFoundError(f"The file {file_path} does not exist.")

start_time = time.time()
logging.info("Loading PDF document")
loader = PyPDFLoader(file_path)
pages = loader.load()
loading_time = time.time() - start_time

logging.info(f"PDF loaded successfully in {loading_time:.2f} seconds")
logging.info(f"Number of pages: {len(pages)}")
print(f"Number of pages: {len(pages)}")

logging.info("Splitting document into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
    chunk_overlap=200)
logging.info(f"Using chunk size: 1000, chunk overlap: 200")

start_time = time.time()
documents = text_splitter.split_documents(pages)
splitting_time = time.time() - start_time

logging.info(f"Document split into {len(documents)} chunks in {splitting_time:.2f} seconds")
print(f"Created {len(documents)} document chunks")

logging.info("Initializing embedding model")
embeddings_model = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
logging.info("Using Mistral embedding model: mistral-embed")

logging.info("Creating vector store and embedding documents")
start_time = time.time()
db = Chroma.from_documents(
    documents, 
    embedding=embeddings_model, 
    collection_name=collection_name, 
    persist_directory=persist_directory
)
embedding_time = time.time() - start_time
logging.info(f"Documents embedded and stored in {embedding_time:.2f} seconds")
print(f"Vector database created with {len(documents)} embedded chunks")

logging.info("Creating retriever from vector store")
retriever = db.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
logging.info("Retriever created with MMR search type, k=1, fetch_k=5")

# Example query to test retrieval
query = "What say the greens on migration?"
logging.info(f"Testing retriever with query: '{query}'")
start_time = time.time()
results = retriever.invoke(query)
retrieval_time = time.time() - start_time

logging.info(f"Retrieved {len(results)} documents in {retrieval_time:.2f} seconds")
print(f"\nQuery: {query}")
print(f"Retrieved {len(results)} relevant document chunks in {retrieval_time:.2f} seconds")

# Print a preview of the first result
if results:
    print("\nPreview of first result:")
    preview = results[0].page_content[:200] + "..." if len(results[0].page_content) > 200 else results[0].page_content
    print(preview)

logging.info("RAG indexing process completed successfully")