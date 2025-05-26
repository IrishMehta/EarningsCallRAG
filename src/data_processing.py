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
import shutil

# LangChain document loaders and text splitters
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from . import config


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

def get_standardized_filename(content: str, original_filename: str) -> str:
    """
    Generate a standardized filename based on document content using LLM.
    
    Args:
        content: The document content to analyze
        original_filename: Original filename for fallback
        
    Returns:
        Standardized filename in format: CompanyName_QuarterYear.ext
    """
    try:
        # Initialize LLM with specific prompt for filename generation
        llm = HuggingFaceEndpoint(
            repo_id=config.LLM_MODEL_NAME,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.1,
            max_new_tokens=50
        )
        
        # Create a prompt to extract company and quarter information
        prompt = f"""
        Based on the following document excerpt, identify the company name and the fiscal quarter/year discussed.
        Format the response exactly as: CompanyName_QxFYyy
        If you can't determine both pieces of information, respond with 'UNKNOWN'.
        Only provide the formatted name, nothing else.

        Document excerpt:
        {content[:1000]}  # Use first 1000 characters for analysis
        """
        
        # Get LLM response
        response = llm.invoke(prompt).strip()
        
        if response and response != "UNKNOWN":
            # Get the original file extension
            _, ext = os.path.splitext(original_filename)
            # Add the extension to the LLM-generated name
            new_filename = f"{response}{ext}"
            
            logger.info("Generated standardized filename", extra={
                "original": original_filename,
                "new": new_filename
            })
            
            return new_filename
        else:
            logger.warning("Could not generate standardized filename", extra={
                "original": original_filename
            })
            return original_filename
            
    except Exception as e:
        logger.error("Error generating filename", extra={
            "error": str(e),
            "original": original_filename
        })
        return original_filename

def rename_and_move_file(file_path: str, new_name: str, target_dir: str) -> str:
    """
    Rename and move a file to the target directory.
    
    Args:
        file_path: Path to the original file
        new_name: New filename
        target_dir: Target directory for the renamed file
        
    Returns:
        Path to the renamed file
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Generate the new file path
        new_path = os.path.join(target_dir, new_name)
        
        # If file with new name already exists, add a number suffix
        counter = 1
        while os.path.exists(new_path):
            base, ext = os.path.splitext(new_name)
            new_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
            counter += 1
        
        # Copy the file to new location with new name
        shutil.copy2(file_path, new_path)
        
        logger.info("File renamed and moved", extra={
            "original": file_path,
            "new": new_path
        })
        
        return new_path
        
    except Exception as e:
        logger.error("Error renaming file", extra={
            "error": str(e),
            "original": file_path,
            "new_name": new_name
        })
        return file_path

def process_and_rename_file(file_path: str, target_dir: str) -> str:
    """
    Process a file and rename it based on its content.
    
    Args:
        file_path: Path to the file to process
        target_dir: Target directory for renamed files
        
    Returns:
        Path to the processed and renamed file
    """
    try:
        # Load the file content
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            loader = UnstructuredPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        
        if not documents:
            logger.warning("No content found in file", extra={"file": file_path})
            return file_path
            
        # Combine all document content for analysis
        combined_content = "\n".join(doc.page_content for doc in documents)
        
        # Generate standardized filename
        new_name = get_standardized_filename(combined_content, os.path.basename(file_path))
        
        # Rename and move the file
        return rename_and_move_file(file_path, new_name, target_dir)
        
    except Exception as e:
        logger.error("Error processing file", extra={
            "error": str(e),
            "file": file_path
        })
        return file_path

def load_documents(source_dir_root: str = config.DATA_DIR) -> List[Document]:
    logger.info("Starting document loading and processing pipeline", extra={"source_dir_root": source_dir_root})

    processed_dir = os.path.join(source_dir_root, config.PROCESSED_DIR)
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # --- Step 1: Process raw files from the root of source_dir_root ---
    # These are files manually added to config.DATA_DIR, not yet in processed_dir.
    raw_files_found = []
    abs_source_dir_root = os.path.abspath(source_dir_root)

    # Iterate over items in the root of source_dir_root
    for item_name in os.listdir(abs_source_dir_root):
        item_path = os.path.join(abs_source_dir_root, item_name)
        # Check if it's a file and has a supported extension
        if os.path.isfile(item_path):
            file_ext = os.path.splitext(item_name)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                raw_files_found.append(item_path)
        # We explicitly ignore subdirectories like 'processed' or 'uploads' in this step,
        # as 'processed' is the target and 'uploads' are handled by the /upload endpoint.

    if raw_files_found:
        logger.info(f"Found {len(raw_files_found)} raw files in '{abs_source_dir_root}' to process into '{processed_dir}'.")
        for raw_file_path in raw_files_found:
            logger.debug(f"Processing and renaming raw file: {raw_file_path}")
            try:
                # process_and_rename_file copies the file to processed_dir with a new name
                # and returns the path to the new file in processed_dir.
                # Its return value isn't strictly needed here as we scan processed_dir later.
                process_and_rename_file(raw_file_path, processed_dir)
            except Exception as e:
                logger.error(f"Failed to process raw file {raw_file_path}", extra={"error": str(e)}, exc_info=True)
    else:
        logger.info(f"No raw files found directly in '{abs_source_dir_root}' for initial processing into '{processed_dir}'.")

    # --- Step 2: Load all documents exclusively from the processed_dir ---
    # This directory should now contain all renamed files (from raw files processed above and UI uploads).
    
    final_files_to_load = []
    for ext in SUPPORTED_EXTENSIONS:
        # Search recursively within processed_dir for supported file types
        search_pattern = os.path.join(processed_dir, f"**/*{ext}")
        found_in_processed = glob.glob(search_pattern, recursive=True)
        # Ensure we only add actual files, not directories if glob pattern is too loose
        final_files_to_load.extend(f for f in found_in_processed if os.path.isfile(f))

    # Deduplicate based on absolute paths to handle any OS-specific path variations from glob
    final_files_to_load = sorted(list(set(os.path.normpath(f) for f in final_files_to_load)))

    logger.info("Files to load from processed directory", extra={
        "directory": processed_dir,
        "total_files": len(final_files_to_load)
    })

    if not final_files_to_load:
        logger.warning("No supported files found in processed directory to load.", extra={"directory": processed_dir})
        return []

    # --- Step 3: Load and process documents from the final list ---
    documents = []
    for file_path in final_files_to_load: # file_path is now guaranteed to be from processed_dir
        file_ext = os.path.splitext(file_path)[1].lower()
        # This check is slightly redundant due to glob pattern using extensions, but ensures safety.
        if file_ext in SUPPORTED_EXTENSIONS:
            loader_factory = SUPPORTED_EXTENSIONS[file_ext]
            # Determine loader name for logging
            loader_name = 'UnstructuredPDFLoader' if file_ext == '.pdf' else loader_factory.__name__
            
            logger.info("Loading file from processed directory", extra={
                "file": os.path.basename(file_path), # This will be the renamed filename
                "full_path": file_path,
                "loader": loader_name
            })
            
            try:
                loader = loader_factory(file_path)
                loaded_docs_for_file = loader.load()
                processed_docs_for_file = []

                for i, doc in enumerate(loaded_docs_for_file):
                    if not doc.page_content or not doc.page_content.strip():
                        logger.warning("Skipping empty document section", extra={
                            "file": os.path.basename(file_path),
                            "section_index": i + 1
                        })
                        continue

                    # CRITICAL: Set source metadata to the basename of the (renamed) file from processed_dir
                    doc.metadata["source"] = os.path.basename(file_path)
                    original_len = len(doc.page_content)
                    doc.page_content = clean_text(doc.page_content)

                    if not doc.page_content:
                        logger.warning("Document section became empty after cleaning", extra={
                            "file": os.path.basename(file_path),
                            "section_index": i + 1,
                            "original_length": original_len
                        })
                        continue
                    
                    # Optionally, add the full path of the processed file to metadata for reference
                    doc.metadata["processed_file_path"] = file_path
                    processed_docs_for_file.append(doc)

                documents.extend(processed_docs_for_file)
                logger.info("File loaded and processed", extra={
                    "file": os.path.basename(file_path),
                    "num_document_objects": len(processed_docs_for_file)
                })
                
            except Exception as e:
                logger.error("Error loading file from processed directory", extra={
                    "file": file_path,
                    "error": str(e)
                }, exc_info=True)

    logger.info("Document loading complete from processed directory", extra={
        "total_document_objects": len(documents)
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