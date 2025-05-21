# src/qa_chain.py

import os
import time
import math
from typing import Dict, Any, Tuple, List, Optional

# LangChain components
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# Import project modules
from . import config
from . import embedding_store # Need this to get embedding model for loading vector store

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
    Initializes and returns the Hugging Face Endpoint LLM instance.
    Caches the model instance globally.

    Args:
        model_name: Name of the model in Hugging Face Hub.
        api_token: Hugging Face Hub API token.
        temperature: Controls randomness (0.0 = deterministic).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        An instance of MyHuggingFaceEndpoint (customized HuggingFaceEndpoint).
    """
    global llm_model
    if llm_model is None:
        print(f"Initializing LLM: {model_name}...")
        if not api_token:
            raise ValueError("Hugging Face Hub API token is required.")

    llm_model = HuggingFaceEndpoint(
        repo_id=model_name,  # Choose one from above
        huggingfacehub_api_token=api_token,
        temperature=temperature,
        max_new_tokens=max_tokens,
        # do_sample=True,  # Only standard parameters
        task="text-generation",  # Specify task for clarity
    )
    print("LLM initialized.")
    return llm_model


# --- Vector Store Loading ---

def load_vector_store(
    persist_directory: str = config.VECTOR_STORE_DIR,
    index_name: str = config.FAISS_INDEX_NAME
) -> FAISS:
    """
    Loads the FAISS vector store from the specified directory.
    Assumes the index has already been created. Caches globally.

    Args:
        persist_directory: Directory containing the FAISS index.
        index_name: Name of the FAISS index file (without extension).

    Returns:
        An instance of the loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the vector store index cannot be found.
    """
    global vector_store_instance
    if vector_store_instance is None:
        index_path = os.path.join(persist_directory, f"{index_name}.faiss")
        print(f"Attempting to load vector store from: {persist_directory} with index name: {index_name}")

        if not os.path.exists(index_path):
             raise FileNotFoundError(
                 f"FAISS index file not found at {index_path}. "
                 f"Please run 'python -m src.embedding_store' first to create it."
             )

        try:
            print("Loading embedding model for vector store...")
            embeddings = embedding_store.get_embedding_model() # Get cached embedding model
            print("Loading FAISS index from disk...")
            start_time = time.time()
            vector_store_instance = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True # Required for HF embeddings potentially
            )
            load_time = time.time() - start_time
            print(f"Vector store loaded successfully in {load_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise # Re-raise the exception after logging

    return vector_store_instance

# --- Prompt Template ---

# Define a prompt template to guide the LLM
PROMPT_TEMPLATE = """
You are a helpful AI assistant specialized in answering questions based on financial documents (earnings call transcripts and reports).
Use the following pieces of context derived from these documents to answer the question at the end.
If you don't find the answer in the provided context, just say that you cannot find the answer in the documents. Do not try to make up an answer.
Keep the answer concise and relevant to the question.

Context:
{context}

Question: {question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# --- QA Chain Creation ---

def create_qa_chain(
    llm: HuggingFaceEndpoint,
    vector_store: FAISS,
    k_docs: int = config.K_RETRIEVED_DOCS,
    custom_prompt: Optional[PromptTemplate] = QA_PROMPT
) -> Tuple[RetrievalQA, VectorStoreRetriever]:
    """
    Creates the RetrievalQA chain and the retriever.

    Args:
        llm: The initialized language model instance.
        vector_store: The loaded FAISS vector store instance.
        k_docs: The number of documents to retrieve.
        custom_prompt: Optional custom prompt template.

    Returns:
        A tuple containing the RetrievalQA chain instance and the retriever instance.
    """
    global qa_chain_instance, retriever_instance
    if qa_chain_instance is None or retriever_instance is None:
        print(f"Creating retriever to fetch top {k_docs} documents...")
        # Create the retriever
        retriever_instance = vector_store.as_retriever(
            search_type="similarity", # Use standard similarity search
            search_kwargs={'k': k_docs}
            # Note: We perform a separate search for confidence scores later
        )

        print("Creating RetrievalQA chain...")
        # Configure the chain specifics
        chain_type_kwargs = {}
        if custom_prompt:
             chain_type_kwargs["prompt"] = custom_prompt

        qa_chain_instance = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Stuffs all retrieved docs into the context
            retriever=retriever_instance,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True # Essential for seeing what context was used
        )
        print("RetrievalQA chain created.")

    return qa_chain_instance, retriever_instance

# --- Confidence Score Calculation ---

def calculate_confidence_score(
    query: str,
    retriever: VectorStoreRetriever,
    distance_threshold: float = config.DISTANCE_THRESHOLD,
    k_for_scoring: int = config.K_RETRIEVED_DOCS # Use same k for consistency
) -> float:
    """
    Calculates a confidence score based on the L2 distance of retrieved documents.

    Retrieves documents, filters them by the distance threshold, and calculates
    a score based on the average distance of the filtered documents.

    Args:
        query: The user's query.
        retriever: The vector store retriever instance.
        distance_threshold: Maximum L2 distance to consider a document relevant.
        k_for_scoring: Number of documents to retrieve for scoring.

    Returns:
        A confidence score between 0.0 and 1.0 (higher is better).
    """
    print(f"\nCalculating confidence score for query: '{query}'")
    print(f" - Retrieving top {k_for_scoring} docs with scores...")
    print(f" - Distance threshold for relevance: {distance_threshold}")

    # Retrieve documents with their L2 distance scores
    try:
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(
            query, k=k_for_scoring
        )
    except Exception as e:
        print(f"Error during similarity search for confidence score: {e}")
        return 0.0 # Return lowest confidence on error

    if not docs_with_scores:
        print(" - No documents retrieved for scoring.")
        return 0.0

    # Filter documents based on the distance threshold
    relevant_docs_scores = [
        score for doc, score in docs_with_scores if score < distance_threshold
    ]

    print(f" - Found {len(relevant_docs_scores)} documents within distance threshold.")
    # Print scores for debugging
    # print(f"   - Scores: {[round(s, 4) for _, s in docs_with_scores]}")
    # print(f"   - Relevant Scores (< {distance_threshold}): {[round(s, 4) for s in relevant_docs_scores]}")


    if not relevant_docs_scores:
        print(" - No relevant documents found within threshold.")
        return 0.0 # No documents met the criteria

    # Calculate the average distance of relevant documents
    average_distance = sum(relevant_docs_scores) / len(relevant_docs_scores)
    print(f" - Average distance of relevant documents: {average_distance:.4f}")

    # Convert average distance to a confidence score (0-1, higher is better)
    # Using formula: 1 / (1 + average_distance)
    # This maps distance [0, inf) to confidence (1, 0] -> approximately [1, 0]
    # Small distance -> high confidence (close to 1)
    # Large distance -> low confidence (close to 0)
    confidence = 1.0 / (1.0 + average_distance)

    # Alternative: Clamp linear conversion: max(0.0, 1.0 - average_distance / distance_threshold)
    # confidence = max(0.0, 1.0 - average_distance / distance_threshold)

    print(f" - Calculated confidence score: {confidence:.4f}")
    return confidence


# --- Main Query Function ---

def get_answer(query: str) -> Optional[Dict[str, Any]]:
    """
    Loads components (if needed), runs the QA chain, calculates confidence,
    and returns the results.

    Args:
        query: The user's question.

    Returns:
        A dictionary containing 'answer', 'source_documents', and 'confidence_score',
        or None if an error occurs.
    """
    try:
        # Ensure all components are loaded/created
        llm = get_llm()
        vector_store = load_vector_store()
        chain, retriever = create_qa_chain(llm, vector_store)

        print(f"\nInvoking QA chain for query: '{query}'")
        start_time = time.time()
        # Run the QA chain
        result = chain.invoke({"query": query})
        chain_time = time.time() - start_time
        print(f"QA chain invocation took {chain_time:.2f} seconds.")

        # Calculate confidence score using the same retriever
        confidence = calculate_confidence_score(query, retriever)

        # Structure the final output
        final_result = {
            "query": query,
            "answer": result.get("result", "No answer generated."),
            # Process source documents for better display if needed
            "source_documents": result.get("source_documents", []),
            "confidence_score": confidence
        }
        return final_result

    except FileNotFoundError as e:
         print(f"ERROR: {e}")
         print("Please ensure the vector store index exists. Run src.embedding_store first.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred in get_answer: {e}")
        import traceback
        print(traceback.format_exc())
        return None

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