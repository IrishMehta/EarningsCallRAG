# src/embedding_store.py

"""
Embedding and vector store management for the Finance RAG system.

This module handles:
- Embedding model initialization and caching
- Vector store creation and persistence
- Document embedding and indexing
"""

import os
import time
import logging
from typing import List, Optional

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Use updated import if needed
from langchain.docstore.document import Document

# Import configuration and data processing functions
from . import config
from . import data_processing

# Configure module logger
logger = logging.getLogger(__name__)

# --- Global Variables ---
# Cache the embedding model instance for efficiency
embedding_model = None

# --- Functions ---

def get_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Initialize and return a Hugging Face embedding model.
    
    This function implements singleton pattern to cache the model instance
    globally and avoid repeated initialization.
    
    Args:
        model_name: Name of the Hugging Face model to use
        
    Returns:
        An instance of HuggingFaceEmbeddings
        
    Note:
        Currently configured to use CPU. Consider adding GPU support
        by setting model_kwargs={'device': 'cuda'} if available.
    """
    global embedding_model
    if embedding_model is None:
        logger.info("Initializing embedding model", extra={
            "model": model_name
        })
        
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}  # FAISS prefers non-normalized

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Embedding model initialized successfully")
        
    return embedding_model

def create_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    save_path: str = os.path.join(config.VECTOR_STORE_DIR, config.FAISS_INDEX_NAME)
) -> FAISS:
    """
    Create or load a FAISS vector store from document chunks.
    
    This function attempts to load an existing vector store from disk.
    If loading fails or no store exists, it creates a new one from the
    provided document chunks and saves it to disk.
    
    Args:
        chunks: List of LangChain Document objects to embed
        embedding_model: Initialized HuggingFaceEmbeddings model
        save_path: Path to save/load the FAISS index
        
    Returns:
        An instance of the FAISS vector store
        
    Raises:
        ValueError: If chunks list is empty
        RuntimeError: If vector store creation/loading fails
    """
    vector_store: Optional[FAISS] = None
    index_directory = os.path.dirname(save_path)

    # Try to load existing index
    if os.path.exists(save_path):
        logger.info("Loading existing vector store", extra={
            "path": save_path
        })
        
        start_time = time.time()
        try:
            vector_store = FAISS.load_local(
                folder_path=index_directory,
                embeddings=embedding_model,
                index_name=config.FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True
            )
            load_time = time.time() - start_time
            
            logger.info("Vector store loaded successfully", extra={
                "duration_seconds": round(load_time, 2)
            })
        except Exception as e:
            logger.error("Failed to load existing vector store", extra={
                "error": str(e)
            }, exc_info=True)
            vector_store = None

    # Create new index if loading failed or no index exists
    if vector_store is None:
        if not chunks:
            logger.error("Cannot create vector store with empty chunks")
            raise ValueError("Cannot create vector store with empty document chunks.")
            
        logger.info("Creating new vector store", extra={
            "num_chunks": len(chunks),
            "model": config.EMBEDDING_MODEL_NAME
        })
        
        start_time = time.time()
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        embed_time = time.time() - start_time
        
        logger.info("Document chunks embedded", extra={
            "duration_seconds": round(embed_time, 2)
        })

        # Ensure save directory exists
        if not os.path.exists(index_directory):
            logger.info("Creating vector store directory", extra={
                "directory": index_directory
            })
            os.makedirs(index_directory)

        # Save the new vector store
        logger.info("Saving vector store", extra={
            "path": save_path
        })
        
        start_time = time.time()
        vector_store.save_local(folder_path=index_directory, index_name=config.FAISS_INDEX_NAME)
        save_time = time.time() - start_time
        
        logger.info("Vector store saved successfully", extra={
            "duration_seconds": round(save_time, 2)
        })

    if not isinstance(vector_store, FAISS):
        logger.error("Vector store creation/loading failed")
        raise RuntimeError("Failed to create or load the vector store.")

    return vector_store

# --- Main execution block for testing ---
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting embedding and vector store test")

    # 1. Load and Split Documents
    logger.info("Loading and splitting documents")
    loaded_docs = data_processing.load_documents()
    if not loaded_docs:
        logger.error("No documents loaded")
        exit(1)
        
    chunked_docs = data_processing.split_documents(loaded_docs)
    if not chunked_docs:
        logger.error("No chunks created")
        exit(1)
        
    logger.info("Documents processed", extra={
        "num_docs": len(loaded_docs),
        "num_chunks": len(chunked_docs)
    })

    # 2. Initialize Embedding Model
    logger.info("Initializing embedding model")
    embeddings = get_embedding_model()
    logger.info("Embedding model ready")

    # 3. Create or Load Vector Store
    logger.info("Creating/loading vector store")
    vector_db = create_vector_store(chunked_docs, embeddings)
    logger.info("Vector store ready")

    # 4. Perform Test Search
    test_query = "take rate quick commerce"
    logger.info("Performing test search", extra={
        "query": test_query
    })
    
    try:
        results_with_scores = vector_db.similarity_search_with_score(
            test_query,
            k=config.K_RETRIEVED_DOCS
        )

        logger.info("Search completed", extra={
            "num_results": len(results_with_scores)
        })
        
        if results_with_scores:
            for i, (doc, score) in enumerate(results_with_scores):
                logger.info(f"Result {i+1}", extra={
                    "score": round(score, 4),
                    "source": doc.metadata.get('source', 'N/A'),
                    "content_preview": doc.page_content[:300]
                })
        else:
            logger.warning("No results found for test query")

    except Exception as e:
        logger.error("Test search failed", extra={
            "error": str(e)
        }, exc_info=True)

    logger.info("Embedding and vector store test complete")