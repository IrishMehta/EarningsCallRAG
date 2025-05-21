# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Configuration ---
# Embedding model name (from Hugging Face)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM model name (from Hugging Face)
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# Hugging Face Hub API Token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Data & Storage Configuration ---
# Directory containing source transcript files
DATA_DIR = "data"
# Directory to save the FAISS vector store index
VECTOR_STORE_DIR = "vector_store"
# FAISS index file name
FAISS_INDEX_NAME = "finance_index.faiss"

# --- Text Splitting Configuration ---
# Parameters for RecursiveCharacterTextSplitter
CHUNK_SIZE = 5000  # Max characters per chunk
CHUNK_OVERLAP = 75  # Overlap between consecutive chunks

# --- RAG Configuration ---
# Number of relevant documents to retrieve from vector store
K_RETRIEVED_DOCS = 5
# Confidence score threshold for filtering retrieved docs
CONFIDENCE_THRESHOLD = 0.70

# Maximum L2 distance score for a document to be considered relevant in confidence calculation
# Lower distance means more similar. Adjust based on testing.
DISTANCE_THRESHOLD = 1.0

# --- API Configuration ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- Validation ---
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please set it in the .env file.")

print("Configuration loaded.")
print(f" - Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f" - LLM Model: {LLM_MODEL_NAME}")
print(f" - Data Directory: {DATA_DIR}")
print(f" - Vector Store Directory: {VECTOR_STORE_DIR}")