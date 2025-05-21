# src/main.py

"""
FastAPI application for the Finance RAG Q&A Bot.

This module implements a REST API that provides question-answering capabilities
for financial documents using Retrieval-Augmented Generation (RAG). It handles
query processing, response formatting, and system health monitoring.
"""

import time
import json
import logging
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

# Import the core QA logic and config
from . import qa_chain
from . import config

# --- Logging Setup ---
class StructuredLogFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Console handler with structured formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(StructuredLogFormatter())
logger.addHandler(console_handler)

# File handler for persistent logging
file_handler = logging.FileHandler("logs/finance_rag.log")
file_handler.setFormatter(StructuredLogFormatter())
logger.addHandler(file_handler)

# --- Application Setup ---
app = FastAPI(
    title="Finance RAG Q&A Bot",
    description="API for querying financial documents using Retrieval-Augmented Generation.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.
    
    Attributes:
        query: The user's question about financial documents
    """
    query: str = Field(..., example="What was the revenue growth last quarter?")
    # Optional: Add more parameters like user_id, session_id if needed later

class SourceDocumentModel(BaseModel):
    """
    Model representing a source document used in generating the answer.
    
    Attributes:
        source: The document identifier or source name
        page_number: The page number in the source document
        speaker: The speaker in case of transcript documents
        content_snippet: A relevant excerpt from the source document
    """
    source: Optional[str] = None
    page_number: Optional[Any] = None # Allow Any type for page number flexibility
    speaker: Optional[str] = None
    # Add other relevant metadata fields if available, e.g., 'start_index'
    content_snippet: Optional[str] = None # Optional: Include a snippet of the source text

class QueryResponse(BaseModel):
    """
    Response model containing the answer and supporting information.
    
    Attributes:
        query: The original user query
        answer: The generated answer
        confidence_score: A score between 0 and 1 indicating answer reliability
        source_documents: List of source documents used to generate the answer
    """
    query: str
    answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0) # Ensure score is between 0 and 1
    source_documents: List[SourceDocumentModel] = []

# --- Global State / Initialization ---
# Load components on startup to avoid loading per request
# This assumes qa_chain handles caching internally now
try:
    logger.info("Initializing QA system components", extra={
        "component": "startup",
        "action": "initialization"
    })
    start_init = time.time()
    # Trigger loading/creation of LLM, Vector Store, Chain, Retriever
    qa_chain.get_llm()
    qa_chain.load_vector_store()
    # Chain creation depends on LLM and VS, call get_answer once lightly?
    # Or explicitly call create_qa_chain if needed outside get_answer
    # For now, rely on get_answer to initialize chain on first call if not already done
    end_init = time.time()
    logger.info("QA system components initialized", extra={
        "component": "startup",
        "action": "initialization_complete",
        "duration_seconds": round(end_init - start_init, 2)
    })
except FileNotFoundError as e:
    logger.error("Vector store initialization failed", extra={
        "component": "startup",
        "error": str(e),
        "action": "vector_store_init"
    })
    # Exit if core components can't load? Or let requests fail?
    # For simplicity, we'll let requests fail if components aren't ready.
except Exception as e:
    logger.error("Critical startup error", extra={
        "component": "startup",
        "error": str(e),
        "action": "general_init"
    }, exc_info=True)
    # Allow app to start but log the error


# --- Middleware for Logging ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware for logging HTTP requests and responses.
    
    Logs request details, processing time, and any errors that occur.
    """
    request_id = str(time.time())  # Simple request ID for tracking
    start_time = time.time()
    
    logger.info("Request received", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_host": request.client.host if request.client else None
    })
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info("Request completed", extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_seconds": round(process_time, 4)
        })
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error("Request failed", extra={
            "request_id": request_id,
            "error": str(e),
            "duration_seconds": round(process_time, 4)
        }, exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error"},
            headers={"X-Process-Time": str(process_time)},
        )


# --- API Endpoint ---
@app.get("/")
async def get_frontend():
    """Serve the frontend interface."""
    return FileResponse("frontend/index.html")

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Process a user query using the RAG system.
    
    Args:
        request: The query request containing the user's question
        
    Returns:
        QueryResponse containing the answer, confidence score, and source documents
        
    Raises:
        HTTPException: If there's an error processing the query
    """
    request_id = str(time.time())
    logger.info("Processing query", extra={
        "request_id": request_id,
        "query": request.query
    })
    
    start_process = time.time()
    try:
        # Call the core QA logic from qa_chain module
        result = qa_chain.get_answer(request.query)

        if result is None:
            logger.error("QA chain returned no result", extra={
                "request_id": request_id,
                "query": request.query
            })
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the query using the QA chain."
            )

        process_time = time.time() - start_process
        logger.info("Query processed successfully", extra={
            "request_id": request_id,
            "duration_seconds": round(process_time, 2),
            "confidence_score": result.get("confidence_score")
        })

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
        logger.error("Query processing error", extra={
            "request_id": request_id,
            "query": request.query,
            "error": str(e)
        }, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the query: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns:
        Dict containing the current system status
    """
    # Can be expanded later to check component readiness
    return {"status": "ok"}

# --- Main Execution ---
# This block allows running the app directly using `python -m src.main`
if __name__ == "__main__":
    logger.info("Starting FastAPI server", extra={
        "host": config.API_HOST,
        "port": config.API_PORT
    })
    print(f"Access the chatbot interface at http://{config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "src.main:app", # Reference the app object within the module
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False, # Set reload=True for development (watches for code changes)
        log_level="info" # Control uvicorn's logging level
    )