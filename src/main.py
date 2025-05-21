# src/main.py

import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging

# Import the core QA logic and config
from . import qa_chain
from . import config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="Finance RAG Q&A Bot",
    description="API for querying financial documents using Retrieval-Augmented Generation.",
    version="1.0.0"
)

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    """ Defines the expected structure for incoming query requests. """
    query: str = Field(..., example="What was the revenue growth last quarter?")
    # Optional: Add more parameters like user_id, session_id if needed later

class SourceDocumentModel(BaseModel):
    """ Defines the structure for source document metadata in the response. """
    source: Optional[str] = None
    page_number: Optional[Any] = None # Allow Any type for page number flexibility
    speaker: Optional[str] = None
    # Add other relevant metadata fields if available, e.g., 'start_index'
    content_snippet: Optional[str] = None # Optional: Include a snippet of the source text

class QueryResponse(BaseModel):
    """ Defines the structure for the API response. """
    query: str
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0) # Ensure score is between 0 and 1
    source_documents: List[SourceDocumentModel] = []

# --- Global State / Initialization ---
# Load components on startup to avoid loading per request
# This assumes qa_chain handles caching internally now
try:
    logger.info("Initializing QA system components...")
    start_init = time.time()
    # Trigger loading/creation of LLM, Vector Store, Chain, Retriever
    qa_chain.get_llm()
    qa_chain.load_vector_store()
    # Chain creation depends on LLM and VS, call get_answer once lightly?
    # Or explicitly call create_qa_chain if needed outside get_answer
    # For now, rely on get_answer to initialize chain on first call if not already done
    end_init = time.time()
    logger.info(f"QA system components initialized (or confirmed loaded) in {end_init - start_init:.2f} seconds.")
except FileNotFoundError as e:
    logger.error(f"CRITICAL ERROR during startup: {e}")
    logger.error("Please ensure the vector store index exists. Run 'python -m src.embedding_store' first.")
    # Exit if core components can't load? Or let requests fail?
    # For simplicity, we'll let requests fail if components aren't ready.
except Exception as e:
    logger.error(f"CRITICAL ERROR during startup: {e}", exc_info=True)
    # Allow app to start but log the error


# --- Middleware for Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Received request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request finished: {response.status_code} in {process_time:.4f}s")
        return response
    except Exception as e:
         process_time = time.time() - start_time
         logger.error(f"Request failed: {e}", exc_info=True)
         # Return a generic server error response
         return JSONResponse(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             content={"detail": "Internal Server Error"},
             headers={"X-Process-Time": str(process_time)},
         )


# --- API Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Accepts a user query, processes it using the RAG QA chain,
    and returns the answer, confidence score, and source document metadata.
    """
    logger.info(f"Processing query: '{request.query}'")
    start_process = time.time()

    try:
        # Call the core QA logic from qa_chain module
        result = qa_chain.get_answer(request.query)

        if result is None:
            logger.error("QA chain failed to return a result.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the query using the QA chain."
            )

        process_time = time.time() - start_process
        logger.info(f"Query processed successfully in {process_time:.2f} seconds.")

        # Format source documents for the response model
        formatted_sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                metadata = doc.metadata or {} # Ensure metadata exists
                formatted_sources.append(
                    SourceDocumentModel(
                        source=metadata.get("source"),
                        page_number=metadata.get("page_number", metadata.get("page")), # Try both keys
                        speaker=metadata.get("speaker"),
                        # Optionally add a snippet
                        # content_snippet=doc.page_content[:100] + "..." if doc.page_content else ""
                    )
                )

        # Prepare the final response
        response_data = QueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            source_documents=formatted_sources
        )
        return response_data

    except HTTPException as http_exc:
         # Re-raise HTTPExceptions (like validation errors)
         raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the query: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """ Basic health check endpoint. """
    # Can be expanded later to check component readiness
    return {"status": "ok"}

# --- Main Execution ---
# This block allows running the app directly using `python -m src.main`
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Access the API documentation at http://{config.API_HOST}:{config.API_PORT}/docs")
    uvicorn.run(
        "src.main:app", # Reference the app object within the module
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False, # Set reload=True for development (watches for code changes)
        log_level="info" # Control uvicorn's logging level
    )