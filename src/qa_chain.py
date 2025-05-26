# src/qa_chain.py

"""
Core QA chain implementation for the Finance RAG system.

This module handles the initialization and management of the RAG components:
- Language Model (LLM) initialization and caching
- Vector store loading and management
- QA chain creation and configuration
- Confidence score calculation
"""

import os
import time
import math
import logging
from typing import Dict, Any, Tuple, List, Optional

# LangChain components
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# Import project modules
from . import config
from . import embedding_store
from . import data_processing

# Configure module logger
logger = logging.getLogger(__name__)

# --- Global Variables ---
llm_model = None
vector_store_instance = None
qa_chain_instance = None
retriever_instance = None

# --- LLM Initialization ---
def get_llm(
    model_name: str = config.LLM_MODEL_NAME,
    api_token: Optional[str] = config.HUGGINGFACEHUB_API_TOKEN,
    temperature: float = 0.1,
    max_tokens: int = 200
) -> HuggingFaceEndpoint:
    """
    Initialize and return a Hugging Face Endpoint LLM instance.
    
    This function implements singleton pattern to cache the model instance
    globally and avoid repeated initialization.
    
    Args:
        model_name: Name of the model in Hugging Face Hub
        api_token: Hugging Face Hub API token for authentication
        temperature: Controls randomness (0.0 = deterministic)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        An instance of HuggingFaceEndpoint
        
    Raises:
        ValueError: If API token is not provided
    """
    global llm_model
    if llm_model is None:
        logger.info("Initializing LLM", extra={
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        if not api_token:
            logger.error("Missing API token for LLM initialization")
            raise ValueError("Hugging Face Hub API token is required.")

        llm_model = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_token,
            temperature=temperature,
            max_new_tokens=max_tokens,
            task="text-generation"
        )
        logger.info("LLM initialized successfully")
        
    return llm_model

# --- Vector Store Loading ---
def load_vector_store(
    persist_directory: str = config.VECTOR_STORE_DIR,
    index_name: str = config.FAISS_INDEX_NAME
) -> FAISS:
    """
    Load the FAISS vector store from disk.
    
    This function implements singleton pattern to cache the vector store
    instance globally and avoid repeated loading.
    
    Args:
        persist_directory: Directory containing the FAISS index
        index_name: Name of the FAISS index file (without extension)
        
    Returns:
        An instance of the loaded FAISS vector store
        
    Raises:
        FileNotFoundError: If the vector store index cannot be found
        Exception: For other loading errors
    """
    global vector_store_instance
    if vector_store_instance is None:
        index_path = os.path.join(persist_directory, f"{index_name}.faiss")
        logger.info("Loading vector store", extra={
            "directory": persist_directory,
            "index_name": index_name
        })

        if not os.path.exists(index_path):
            logger.error("Vector store index not found", extra={
                "path": index_path
            })
            raise FileNotFoundError(
                f"FAISS index file not found at {index_path}. "
                f"Please run 'python -m src.embedding_store' first to create it."
            )

        try:
            logger.info("Loading embedding model")
            embeddings = embedding_store.get_embedding_model()
            
            logger.info("Loading FAISS index from disk")
            start_time = time.time()
            vector_store_instance = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            load_time = time.time() - start_time
            
            logger.info("Vector store loaded successfully", extra={
                "duration_seconds": round(load_time, 2)
            })
        except Exception as e:
            logger.error("Vector store loading failed", extra={
                "error": str(e)
            }, exc_info=True)
            raise

    return vector_store_instance

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are a helpful AI assistant specialized in answering questions based on financial documents (earnings call transcripts and reports).
Use the following pieces of context derived from these documents to answer the question at the end.
If you don't find the answer in the provided context, just say that you cannot find the answer in the documents. Do not try to make up an answer.
Keep the answer very concise and relevant to the question.

Context:
{context}

Question: {question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --- QA Chain Creation ---
def create_qa_chain(
    llm: HuggingFaceEndpoint,
    vector_store: FAISS,
    k_docs: int = config.K_RETRIEVED_DOCS,
    custom_prompt: Optional[PromptTemplate] = QA_PROMPT
) -> Tuple[RetrievalQA, VectorStoreRetriever]:
    """
    Create the RetrievalQA chain and retriever.
    
    This function implements singleton pattern to cache the chain and retriever
    instances globally and avoid repeated creation.
    
    Args:
        llm: The initialized language model instance
        vector_store: The loaded FAISS vector store instance
        k_docs: Number of documents to retrieve
        custom_prompt: Optional custom prompt template
        
    Returns:
        A tuple containing:
        - RetrievalQA chain instance
        - VectorStoreRetriever instance
    """
    global qa_chain_instance, retriever_instance
    if qa_chain_instance is None or retriever_instance is None:
        logger.info("Creating retriever", extra={
            "k_docs": k_docs
        })
        
        # Create the retriever
        retriever_instance = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k_docs}
        )

        logger.info("Creating RetrievalQA chain")
        # Configure the chain
        chain_type_kwargs = {}
        if custom_prompt:
            chain_type_kwargs["prompt"] = custom_prompt

        qa_chain_instance = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_instance,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        logger.info("RetrievalQA chain created successfully")

    return qa_chain_instance, retriever_instance

# --- Confidence Score Calculation ---
def calculate_confidence_score(
    query: str,
    retriever: VectorStoreRetriever,
    distance_threshold: float = config.DISTANCE_THRESHOLD,
    k_for_scoring: int = config.K_RETRIEVED_DOCS
) -> float:
    """
    Calculate confidence score based on document similarity.
    
    The confidence score is calculated using the average L2 distance of
    retrieved documents that fall within the distance threshold.
    
    Args:
        query: The user's query
        retriever: The vector store retriever instance
        distance_threshold: Maximum L2 distance to consider a document relevant
        k_for_scoring: Number of documents to retrieve for scoring
        
    Returns:
        A confidence score between 0.0 and 1.0 (higher is better)
    """
    logger.info("Calculating confidence score", extra={
        "query": query,
        "k_docs": k_for_scoring,
        "distance_threshold": distance_threshold
    })

    try:
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(
            query, k=k_for_scoring
        )
    except Exception as e:
        logger.error("Similarity search failed", extra={
            "error": str(e)
        }, exc_info=True)
        return 0.0

    if not docs_with_scores:
        logger.warning("No documents retrieved for scoring")
        return 0.0

    # Filter documents based on distance threshold
    relevant_docs_scores = [
        score for doc, score in docs_with_scores if score < distance_threshold
    ]

    logger.info("Documents retrieved for scoring", extra={
        "total_docs": len(docs_with_scores),
        "relevant_docs": len(relevant_docs_scores)
    })

    if not relevant_docs_scores:
        logger.warning("No relevant documents found within threshold")
        return 0.0

    # Calculate average distance of relevant documents
    average_distance = sum(relevant_docs_scores) / len(relevant_docs_scores)
    logger.debug("Average distance calculated", extra={
        "average_distance": round(average_distance, 4)
    })

    # Convert distance to confidence score (0-1, higher is better)
    confidence = 1.0 / (1.0 + average_distance)
    
    logger.info("Confidence score calculated", extra={
        "confidence": round(confidence, 4)
    })
    
    return confidence

def get_answer(query: str) -> Optional[Dict[str, Any]]:
    """
    Process a query through the RAG system and return the answer.
    
    This function orchestrates the entire QA process:
    1. Ensures LLM and vector store are initialized
    2. Creates/retrieves the QA chain
    3. Processes the query
    4. Calculates confidence score
    5. Returns formatted response
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
        - query: Original query
        - answer: Generated answer
        - confidence_score: Calculated confidence score
        - source_documents: List of source documents used
        
    Raises:
        Exception: If any step in the process fails
    """
    logger.info("Processing query", extra={
        "query": query
    })
    
    try:
        # Ensure components are initialized
        llm = get_llm()
        vector_store = load_vector_store()
        
        # Get or create QA chain
        qa_chain, retriever = create_qa_chain(llm, vector_store)
        
        # Process query
        start_time = time.time()
        result = qa_chain({"query": query})
        process_time = time.time() - start_time
        
        logger.info("Query processed", extra={
            "duration_seconds": round(process_time, 2)
        })
        
        # Log source documents for debugging
        if result.get("source_documents"):
            for idx, doc in enumerate(result["source_documents"]):
                logger.debug(f"Source document {idx + 1}", extra={
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "speaker": doc.metadata.get("speaker", "N/A"),
                    "content_length": len(doc.page_content) if doc.page_content else 0,
                    "content_preview": doc.page_content[:200] if doc.page_content else "NO CONTENT"
                })
        
        # Calculate confidence score
        confidence = calculate_confidence_score(query, retriever)
        
        # Format response
        response = {
            "query": query,
            "answer": result["result"],
            "confidence_score": confidence,
            "source_documents": result.get("source_documents", [])
        }
        
        logger.info("Answer generated successfully", extra={
            "confidence_score": round(confidence, 4),
            "num_sources": len(result.get("source_documents", []))
        })
        
        return response
        
    except Exception as e:
        logger.error("Query processing failed", extra={
            "error": str(e)
        }, exc_info=True)
        raise

def init_qa_components():
    """Initialize all QA system components."""
    global vector_store_instance, llm_model, qa_chain_instance, retriever_instance
    
    try:
        # Initialize LLM
        llm_model = get_llm()
        
        # Load or create vector store
        vector_store_instance = load_vector_store()
        
        # Get or create QA chain
        qa_chain_instance, retriever_instance = create_qa_chain(llm_model, vector_store_instance)
        
        logger.info("QA components initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize QA components", extra={
            "error": str(e)
        }, exc_info=True)
        raise

def reload_vector_store() -> None:
    """
    Reload the vector store with all documents in the data directory.
    This should be called after new files are uploaded.
    """
    global vector_store_instance
    
    try:
        # Load all documents from the data directory
        documents = data_processing.load_documents()
        
        if not documents:
            logger.warning("No documents found to load into vector store")
            return
            
        # Create new vector store
        vector_store_instance = FAISS.from_documents(
            documents,
            embedding_store.get_embedding_model()
        )
        
        # Save the updated vector store
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
        vector_store_instance.save_local(
            folder_path=config.VECTOR_STORE_DIR,
            index_name=config.FAISS_INDEX_NAME
        )
        
        logger.info("Vector store reloaded successfully", extra={
            "num_documents": len(documents)
        })
        
    except Exception as e:
        logger.error("Failed to reload vector store", extra={
            "error": str(e)
        }, exc_info=True)
        raise

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Testing QA Chain ---")

    # Ensure API token is set in environment or .env file
    if not config.HUGGINGFACEHUB_API_TOKEN:
        print("ERROR: HUGGINGFACEHUB_API_TOKEN not found. Please set it in .env file.")
    else:
        # Example queries to test
        test_queries = [
            # "What was Swiggy's revenue growth last quarter?",
            "Summarize the key financial highlights for Q3 FY25 for Zomato",
            # "What did the CEO say about future strategy?",
            "Who is the CFO of Swiggy?",
            "What is the combined sentiment regarding the food delivery segment from both Swiggy and Zomato?",
            # "Explain the concept of Adjusted EBITDA mentioned in the documents." # Concept query
        ]

        for query in test_queries:
            print(f"\n=================================================")
            print(f"Testing Query: {query}")
            print(f"=================================================")

            response = get_answer(query)

            if response:
                print("\n--- RESULT ---")
                print(f"Query: {response['query']}")
                print(f"\nAnswer: {response['answer']}")
                print(f"\nConfidence Score: {response['confidence_score']:.4f}")

                print("\nSource Documents Used:")
                if response['source_documents']:
                     for i, doc in enumerate(response['source_documents']):
                         source = doc.metadata.get('source', 'N/A')
                         page = doc.metadata.get('page_number', 'N/A')
                         speaker = doc.metadata.get('speaker', 'N/A')
                         print(f"  - Doc {i+1}: (Source: {source}, Page: {page}, Speaker: {speaker})")
                         # print(f"    Content: {repr(doc.page_content[:150])}...") # Uncomment for snippet
                else:
                     print("  - No source documents returned by the chain.")
            else:
                print("\n--- FAILED TO GET RESPONSE ---")

        print("\n--- QA Chain Test Complete ---")