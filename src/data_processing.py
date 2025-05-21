"""
Document processing and text splitting for the Finance RAG system.

This module handles:
- Document loading from various file formats (PDF, TXT)
- Text cleaning and normalization
- Semantic splitting based on speaker patterns
- Character-based text chunking
"""

import os
import glob
import re
import logging
from typing import List, Optional

# LangChain document loaders and text splitters
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration (using dummy for testing)
class DummyConfig:
    """Temporary configuration class for testing."""
    DATA_DIR = "."
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
config = DummyConfig()

# --- Constants ---
PDF_LOADER_MODE = "single"  # Load whole PDF as one Document

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".pdf": lambda path: UnstructuredPDFLoader(path, mode=PDF_LOADER_MODE),
}

# Regex to identify speaker lines in transcripts
SPEAKER_PATTERN = r'(?m)^\s*([A-Z][a-zA-Z .&\-]+:|Q:|A:|Question:|Answer:|Moderator:)'

# Configure module logger
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Performs the following operations:
    1. Replaces tabs with spaces
    2. Normalizes paragraph breaks
    3. Collapses multiple spaces
    4. Removes page numbers
    5. Strips whitespace from lines
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned and normalized text string
    """
    if not isinstance(text, str):
        return ""
        
    original_len = len(text)
    
    # Apply cleaning operations
    text = text.replace('\t', ' ')
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \xa0]{2,}', ' ', text)  # Collapse multiple spaces
    text = re.sub(r'(?m)^\s*Page \d+ of \d+\s*$\n?', '', text)  # Remove page numbers
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    text = text.strip()
    
    cleaned_len = len(text)
    if cleaned_len != original_len:
        logger.debug("Text cleaned", extra={
            "original_length": original_len,
            "cleaned_length": cleaned_len,
            "reduction_percent": round((1 - cleaned_len/original_len) * 100, 2)
        })
        
    return text

def transcript_semantic_split(doc: Document) -> List[Document]:
    """
    Split a transcript document into chunks based on speaker patterns.
    
    This function identifies speaker turns in transcript-like documents
    and splits them into separate chunks while preserving speaker metadata.
    
    Args:
        doc: The document to split
        
    Returns:
        List of Document chunks, each containing a speaker turn with metadata
    """
    content = doc.page_content
    if not content:
        logger.warning("Empty document content")
        return []

    # Split content on speaker patterns
    parts = re.split(SPEAKER_PATTERN, content)
    chunks = []
    base_metadata = doc.metadata.copy()

    # Process initial content before first speaker
    initial_content = parts[0].strip()
    if initial_content and not initial_content.lower().startswith("ladies and gentlemen") and len(initial_content) > 10:
        chunk_metadata = base_metadata.copy()
        chunks.append(Document(page_content=initial_content, metadata=chunk_metadata))

    # Process speaker turns
    for i in range(1, len(parts), 2):
        speaker = parts[i].strip()
        text_following_speaker = parts[i+1].strip() if (i+1 < len(parts)) else ""
        chunk_content = f"{speaker}\n{text_following_speaker}".strip()

        if chunk_content and not chunk_content.lower().startswith("ladies and gentlemen") and len(chunk_content.split('\n', 1)[-1]) > 5:
            chunk_metadata = base_metadata.copy()
            chunk_metadata['speaker'] = speaker.replace(':','').strip()
            chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))

    # Handle documents with no speaker patterns
    if not chunks and content and not content.lower().startswith("ladies and gentlemen"):
        logger.info("No speaker patterns found, using original content", extra={
            "source": doc.metadata.get('source', 'N/A')
        })
        return [Document(page_content=content, metadata=base_metadata)]
    elif not chunks:
        logger.warning("No meaningful semantic chunks found after filtering", extra={
            "source": doc.metadata.get('source', 'N/A')
        })
        return []

    logger.info("Document semantically split", extra={
        "source": doc.metadata.get('source', 'N/A'),
        "num_chunks": len(chunks)
    })
    return chunks

def load_documents(source_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Load documents from the specified directory.
    
    Supports loading from:
    - Text files (.txt)
    - PDF files (.pdf) in single-document mode
    
    Args:
        source_dir: Directory containing the documents to load
        
    Returns:
        List of loaded and cleaned Document objects
    """
    logger.info("Loading documents", extra={
        "directory": source_dir,
        "pdf_mode": PDF_LOADER_MODE
    })
    
    # Find all supported files
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        abs_source_dir = os.path.abspath(source_dir)
        search_pattern = os.path.join(abs_source_dir, f"**/*{ext}")
        logger.debug("Searching for files", extra={
            "pattern": search_pattern
        })
        all_files.extend(glob.glob(search_pattern, recursive=True))

    abs_source_dir_norm = os.path.normpath(abs_source_dir)
    all_files = [f for f in all_files if os.path.normpath(os.path.dirname(f)).startswith(abs_source_dir_norm)]

    logger.info("Files found", extra={
        "total_files": len(all_files)
    })
    
    if not all_files:
        logger.warning("No supported files found", extra={
            "directory": abs_source_dir
        })
        return []

    # Load and process each file
    documents = []
    for file_path in all_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in SUPPORTED_EXTENSIONS:
            loader_factory = SUPPORTED_EXTENSIONS[file_ext]
            loader_name = 'UnstructuredPDFLoader' if file_ext == '.pdf' else loader_factory.__name__
            
            logger.info("Loading file", extra={
                "file": os.path.basename(file_path),
                "loader": loader_name
            })
            
            try:
                loader = loader_factory(file_path)
                loaded_docs = loader.load()
                processed_docs = []

                for i, doc in enumerate(loaded_docs):
                    if not doc.page_content or not doc.page_content.strip():
                        logger.warning("Skipping empty document", extra={
                            "file": os.path.basename(file_path),
                            "section": i+1
                        })
                        continue

                    # Add source metadata and clean text
                    doc.metadata["source"] = os.path.basename(file_path)
                    original_len = len(doc.page_content)
                    doc.page_content = clean_text(doc.page_content)
                    cleaned_len = len(doc.page_content)

                    if not doc.page_content:
                        logger.warning("Document became empty after cleaning", extra={
                            "file": os.path.basename(file_path),
                            "section": i+1,
                            "original_length": original_len
                        })
                        continue

                    processed_docs.append(doc)

                documents.extend(processed_docs)
                logger.info("File processed", extra={
                    "file": os.path.basename(file_path),
                    "num_docs": len(processed_docs)
                })
                
            except Exception as e:
                logger.error("Error loading file", extra={
                    "file": file_path,
                    "error": str(e)
                }, exc_info=True)
        else:
            logger.warning("Unsupported file type", extra={
                "file": file_path
            })

    logger.info("Document loading complete", extra={
        "total_documents": len(documents)
    })
    return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into chunks using semantic and character-based splitting.
    
    The splitting process:
    1. Attempts semantic splitting based on speaker patterns
    2. Applies recursive character splitting to handle long chunks
    
    Args:
        documents: List of documents to split
        chunk_size: Target size for character-based chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split Document chunks
    """
    logger.info("Starting document splitting", extra={
        "num_documents": len(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })

    all_final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    for i, doc in enumerate(documents):
        source_info = doc.metadata.get('source', 'N/A')
        logger.info("Processing document", extra={
            "document_num": i+1,
            "total_documents": len(documents),
            "source": source_info
        })

        # Attempt semantic splitting
        try:
            semantic_chunks = transcript_semantic_split(doc)
        except Exception as e:
            logger.error("Semantic splitting failed", extra={
                "source": source_info,
                "error": str(e)
            }, exc_info=True)
            
            if doc.page_content:
                semantic_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata)]
                logger.info("Using original content as fallback")
            else:
                semantic_chunks = []

        if not semantic_chunks:
            logger.warning("No semantic chunks generated", extra={
                "source": source_info
            })
            continue

        # Apply character splitting
        try:
            char_split_chunks = text_splitter.split_documents(semantic_chunks)
            all_final_chunks.extend(char_split_chunks)
            
            logger.info("Character splitting complete", extra={
                "source": source_info,
                "num_chunks": len(char_split_chunks)
            })
        except Exception as e:
            logger.error("Character splitting failed", extra={
                "source": source_info,
                "error": str(e)
            }, exc_info=True)

    logger.info("Document splitting complete", extra={
        "total_chunks": len(all_final_chunks)
    })
    
    if all_final_chunks:
        logger.debug("First chunk metadata", extra={
            "metadata": all_final_chunks[0].metadata,
            "content_preview": all_final_chunks[0].page_content[:300]
        })

    return all_final_chunks

# --- Main execution block for testing ---
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data processing test")
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    data_directory = os.path.join(script_dir, 'data')
    
    # Test document loading and splitting
    docs = load_documents(data_directory)
    if docs:
        chunks = split_documents(docs)
        logger.info("Test completed", extra={
            "num_documents": len(docs),
            "num_chunks": len(chunks)
        })
    else:
        logger.error("No documents loaded for testing")