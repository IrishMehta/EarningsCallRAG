"""
Configuration settings for the Finance RAG system.

This module defines all configuration parameters used throughout the application,
including model settings, data paths, and API configuration. Environment variables
are loaded from a .env file for sensitive information like API tokens.

The configuration is organized into several sections:
- Model Configuration: Embedding and LLM model settings
- Data & Storage: File paths and storage locations
- Text Processing: Document chunking parameters
- RAG Settings: Retrieval and confidence thresholds
- API Configuration: Server settings
"""

import os
import logging
from dotenv import load_dotenv

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Model Configuration ---
# Embedding model for converting text to vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Language model for generating responses
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# API token for accessing Hugging Face models
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Data & Storage Configuration ---
# Directory containing source documents (PDFs, TXTs)
DATA_DIR = "data"
# Directory for storing the FAISS vector index
VECTOR_STORE_DIR = "vector_store"
# Name of the FAISS index file (without extension)
FAISS_INDEX_NAME = "finance_index.faiss"

# --- Text Splitting Configuration ---
# Maximum number of characters per text chunk
CHUNK_SIZE = 5000
# Number of characters to overlap between consecutive chunks
CHUNK_OVERLAP = 75

# --- RAG Configuration ---
# Number of most relevant documents to retrieve for each query
K_RETRIEVED_DOCS = 5
# Minimum confidence score required for a response to be considered reliable
CONFIDENCE_THRESHOLD = 0.70
# Maximum L2 distance for a document to be considered relevant
# Lower distance = higher similarity. Adjust based on testing.
DISTANCE_THRESHOLD = 1.0

# --- API Configuration ---
# Host address for the FastAPI server
API_HOST = "0.0.0.0"
# Port number for the FastAPI server
API_PORT = 8000

# --- Validation ---
if not HUGGINGFACEHUB_API_TOKEN:
    logger.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found in environment variables. "
        "Please set it in the .env file."
    )

# Log configuration summary
logger.info("Configuration loaded", extra={
    "embedding_model": EMBEDDING_MODEL_NAME,
    "llm_model": LLM_MODEL_NAME,
    "data_dir": DATA_DIR,
    "vector_store_dir": VECTOR_STORE_DIR,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "k_retrieved_docs": K_RETRIEVED_DOCS,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "distance_threshold": DISTANCE_THRESHOLD,
    "api_host": API_HOST,
    "api_port": API_PORT
})