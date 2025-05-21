# src/embedding_store.py

import os
import time
from typing import List, Optional

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Use updated import if needed
from langchain.docstore.document import Document

# Import configuration and data processing functions
from . import config
from . import data_processing

# --- Global Variables ---
# Cache the embedding model instance for efficiency
embedding_model = None

# --- Functions ---

def get_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Initializes and returns the Hugging Face embedding model.
    Caches the model instance globally to avoid reloading.

    Args:
        model_name: The name of the Hugging Face model to use. Defaults to config.EMBEDDING_MODEL_NAME.

    Returns:
        An instance of HuggingFaceEmbeddings.
    """
    global embedding_model
    if embedding_model is None:
        print(f"Initializing embedding model: {model_name}...")
        # Specify 'cpu' or 'cuda' if needed.
        # Consider adding model_kwargs={'device': 'cuda'} if GPU is available
        model_kwargs = {'device': 'cpu'} # Default to CPU
        # You can also specify encode_kwargs={'normalize_embeddings': True/False} if needed
        encode_kwargs = {'normalize_embeddings': False} # FAIIS usually prefers non-normalized

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model initialized.")
    return embedding_model

def create_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    save_path: str = os.path.join(config.VECTOR_STORE_DIR, config.FAISS_INDEX_NAME)
) -> FAISS:
    """
    Creates or loads a FAISS vector store from document chunks.

    If a FAISS index exists at `save_path`, it loads it. Otherwise,
    it creates a new index from the provided chunks and saves it to `save_path`.

    Args:
        chunks: A list of LangChain Document objects (text chunks).
        embedding_model: The initialized HuggingFaceEmbeddings model.
        save_path: The path to save/load the FAISS index. Defaults to config settings.

    Returns:
        An instance of the FAISS vector store.
    """
    vector_store: Optional[FAISS] = None
    index_directory = os.path.dirname(save_path)

    # Check if the index directory and file exist
    if os.path.exists(save_path):
        print(f"Loading existing FAISS vector store from: {save_path}")
        start_time = time.time()
        try:
            # FAISS.load_local requires the folder path and the embedding function
            # allow_dangerous_deserialization=True is needed for loading some older FAISS indexes or custom embeddings
            vector_store = FAISS.load_local(
                folder_path=index_directory, # Pass the directory
                embeddings=embedding_model,
                index_name=config.FAISS_INDEX_NAME, # Pass the index name (without extension)
                allow_dangerous_deserialization=True # Be aware of security implications if index source is untrusted
            )
            load_time = time.time() - start_time
            print(f"Successfully loaded vector store in {load_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Attempting to create a new one...")
            vector_store = None # Reset to ensure creation

    # If loading failed or index doesn't exist, create a new one
    if vector_store is None:
        if not chunks:
             raise ValueError("Cannot create vector store with empty document chunks.")
        print("No existing vector store found or loading failed. Creating a new one...")
        print(f"Embedding {len(chunks)} chunks using {config.EMBEDDING_MODEL_NAME}...")
        start_time = time.time()
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        embed_time = time.time() - start_time
        print(f"Finished embedding chunks in {embed_time:.2f} seconds.")

        # Ensure the save directory exists
        if not os.path.exists(index_directory):
            print(f"Creating directory: {index_directory}")
            os.makedirs(index_directory)

        # Save the newly created vector store
        print(f"Saving vector store to: {save_path}")
        start_time = time.time()
        vector_store.save_local(folder_path=index_directory, index_name=config.FAISS_INDEX_NAME)
        save_time = time.time() - start_time
        print(f"Successfully saved vector store in {save_time:.2f} seconds.")

    if not isinstance(vector_store, FAISS):
         raise RuntimeError("Failed to create or load the vector store.")

    return vector_store

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Testing Embedding and Vector Store Creation ---")

    # 1. Load and Split Documents (using functions from data_processing)
    print("\nStep 1: Loading and Splitting Documents...")
    loaded_docs = data_processing.load_documents()
    if not loaded_docs:
        print("No documents loaded. Exiting test.")
        exit()
    chunked_docs = data_processing.split_documents(loaded_docs)
    if not chunked_docs:
        print("No chunks created. Exiting test.")
        exit()
    print("Document loading and splitting complete.")

    # 2. Initialize Embedding Model
    print("\nStep 2: Initializing Embedding Model...")
    embeddings = get_embedding_model()
    print("Embedding model ready.")

    # 3. Create or Load Vector Store
    print("\nStep 3: Creating/Loading Vector Store...")
    vector_db = create_vector_store(chunked_docs, embeddings)
    print("Vector store is ready.")

    # 4. Perform a Test Search
    print("\nStep 4: Performing a Test Similarity Search...")
    test_query = "take rate quick commerce"
    print(f"Test Query: '{test_query}'")
    try:
        # Use similarity_search_with_score to also get relevance scores
        results_with_scores = vector_db.similarity_search_with_score(
            test_query,
            k=config.K_RETRIEVED_DOCS # Retrieve top K results defined in config
        )

        print(f"\nFound {len(results_with_scores)} relevant chunks:")
        if results_with_scores:
            for i, (doc, score) in enumerate(results_with_scores):
                print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
                print(f"Content Snippet: {doc.page_content[:300]}...") # Show beginning of chunk
        else:
            print("No relevant chunks found for the test query.")

    except Exception as e:
        print(f"An error occurred during the test search: {e}")

    print("\n--- Embedding and Vector Store Test Complete ---")