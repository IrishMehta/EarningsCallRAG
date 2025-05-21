# src/data_processing.py

import os
import glob
import re
from typing import List, Optional

# LangChain document loaders and text splitters
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration (using dummy for testing)
class DummyConfig:
    # Load entire PDF as one document to avoid splitting semantic units across pages
    DATA_DIR = "." # Or specify your data directory
    CHUNK_SIZE = 1000 # Example value (adjust as needed)
    CHUNK_OVERLAP = 150 # Example value (adjust as needed)
config = DummyConfig()

# --- Constants ---
# *** Changed PDF_LOADER_MODE to "single" ***
PDF_LOADER_MODE = "single" # Load whole PDF as one Document

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    # Use UnstructuredPDFLoader in single mode
    ".pdf": lambda path: UnstructuredPDFLoader(path, mode=PDF_LOADER_MODE),
}

# Regex to identify speaker lines (Name/Title:, Q:, A:, Moderator:)
SPEAKER_PATTERN = r'(?m)^\s*([A-Z][a-zA-Z .&\-]+:|Q:|A:|Question:|Answer:|Moderator:)'

# --- Functions ---

def clean_text(text: str) -> str:
    """ Basic text cleaning: replace tabs, normalize newlines, collapse multiple spaces, remove page numbers. """
    if not isinstance(text, str):
        return ""
    text = text.replace('\t', ' ')
    text = re.sub(r'\n\s*\n+', '\n\n', text) # Normalize paragraph breaks
    text = re.sub(r'[ \xa0]{2,}', ' ', text) # Collapse multiple spaces
    # Remove lines that look like standard page numbers (can be less reliable in 'single' mode)
    text = re.sub(r'(?m)^\s*Page \d+ of \d+\s*$\n?', '', text)
    # Remove lines that seem to be just headers/footers if needed (example)
    # text = re.sub(r'(?m)^\s*\[CompanyName\]\s*-\s*Q\d\s*FY\d+\s*$\n?', '', text)
    # Remove leading/trailing whitespace from each *remaining* line
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip()) # Ensure empty lines after stripping are removed
    return text.strip()

def transcript_semantic_split(doc: Document) -> List[Document]:
    """
    Splits a transcript document (potentially the entire PDF content)
    into chunks based on speaker pattern matches.
    """
    content = doc.page_content
    if not content:
        return []

    # Use re.split with the lookahead assertion to keep the speaker delimiter
    parts = re.split(SPEAKER_PATTERN, content)

    chunks = []
    # Use original doc's metadata as the base for all chunks derived from it
    base_metadata = doc.metadata.copy()

    # Handle potential content before the first speaker pattern match
    initial_content = parts[0].strip()
    # Filter common transcript boilerplate - check length to avoid filtering meaningful short intros
    if initial_content and not initial_content.lower().startswith("ladies and gentlemen") and len(initial_content) > 10:
         # Don't add speaker metadata for initial non-speaker content
         chunk_metadata = base_metadata.copy()
         chunks.append(Document(page_content=initial_content, metadata=chunk_metadata))

    # Process the remaining parts (speaker + their text)
    for i in range(1, len(parts), 2): # Step by 2: index i is speaker, i+1 is text
        speaker = parts[i].strip()
        text_following_speaker = parts[i+1].strip() if (i+1 < len(parts)) else ""

        # Combine speaker and their text for the chunk content
        chunk_content = f"{speaker}\n{text_following_speaker}".strip()

        # Filter boilerplate/empty chunks - check length of actual text after speaker tag
        if chunk_content and not chunk_content.lower().startswith("ladies and gentlemen") and len(chunk_content.split('\n', 1)[-1]) > 5:
            # Create new metadata for this chunk, copying from base
            chunk_metadata = base_metadata.copy()
            chunk_metadata['speaker'] = speaker.replace(':','').strip() # Add speaker to metadata
            # Note: 'page_number' metadata might be less relevant or absent in 'single' mode
            chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))

    # If the document had no speaker patterns after filtering, return the original content as one chunk
    if not chunks and content and not content.lower().startswith("ladies and gentlemen"):
         print(f"   - No speaker patterns found or all filtered. Returning original content as one section.")
         # Return the original content wrapped in a Document, keeping original metadata
         return [Document(page_content=content, metadata=base_metadata)]
    elif not chunks:
        print(f"   - No meaningful semantic chunks found after filtering.")
        return []

    print(f"   - Semantically split into {len(chunks)} speaker chunks.")
    return chunks


def load_documents(source_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Loads all supported documents (.txt, .pdf) from the specified directory.
    Loads PDFs as single documents. Cleans text and adds source metadata.
    """
    print(f"Loading documents from: {source_dir} (PDF mode: {PDF_LOADER_MODE})")
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        abs_source_dir = os.path.abspath(source_dir)
        search_pattern = os.path.join(abs_source_dir, f"**/*{ext}")
        print(f"Searching pattern: {search_pattern}")
        all_files.extend(
            glob.glob(search_pattern, recursive=True)
        )

    abs_source_dir_norm = os.path.normpath(abs_source_dir)
    all_files = [f for f in all_files if os.path.normpath(os.path.dirname(f)).startswith(abs_source_dir_norm)]

    print(f"Found {len(all_files)} supported files to load.")
    if not all_files:
        print(f"Warning: No files found matching patterns in {abs_source_dir}")

    documents = []
    for file_path in all_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in SUPPORTED_EXTENSIONS:
            loader_factory = SUPPORTED_EXTENSIONS[file_ext]
            loader_name = 'UnstructuredPDFLoader' if file_ext == '.pdf' else loader_factory.__name__
            print(f" > Loading {os.path.basename(file_path)} using {loader_name}...")
            try:
                loader = loader_factory(file_path) # Instantiates TextLoader or UnstructuredPDFLoader(..., mode='single')
                # Load typically returns a list, even in single mode it's often a list of one doc
                loaded_docs = loader.load()

                processed_docs = []
                # Even in 'single' mode, loop through result (usually just one doc for PDF)
                for i, doc in enumerate(loaded_docs):
                    if not doc.page_content or not doc.page_content.strip():
                        print(f"   - Skipping document/section {i+1} (empty content) from {os.path.basename(file_path)}")
                        continue

                    # Add source metadata BEFORE cleaning
                    doc.metadata["source"] = os.path.basename(file_path)
                    # Page number is not inherently available per-doc in 'single' mode
                    # Unstructured *might* add page numbers to element metadata if using mode='elements'
                    # Keep existing metadata, but don't try to force page numbers here

                    # Clean the text content
                    original_len = len(doc.page_content)
                    doc.page_content = clean_text(doc.page_content)
                    cleaned_len = len(doc.page_content)

                    if not doc.page_content:
                         print(f"   - Document {i+1} became empty after cleaning from {os.path.basename(file_path)}. Original length: {original_len}.")
                         continue

                    processed_docs.append(doc)

                documents.extend(processed_docs)
                print(f"   - Loaded and cleaned {len(processed_docs)} document(s) from {os.path.basename(file_path)}.")
            except Exception as e:
                print(f"   - Error loading {file_path}: {e}")
                # import traceback
                # print(traceback.format_exc())
        else:
            print(f"   - Skipping unsupported file: {file_path}")

    print(f"Finished loading. Total documents loaded: {len(documents)}")
    return documents


def split_documents(
    documents: List[Document], # Now each 'document' is likely a whole file content
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits documents (loaded potentially as whole files):
    1. Attempts semantic splitting based on speaker patterns for transcript-like content.
    2. Applies recursive character splitting to the results of step 1.
    """
    print(f"\nStarting splitting process...")
    print(f" - Character Splitter: Chunk Size={chunk_size}, Overlap={chunk_overlap}")

    all_final_chunks = []
    # Now 'documents' contains potentially large docs (whole files)
    for i, doc in enumerate(documents):
        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
        print(f" Processing document {i+1}/{len(documents)} ({source_info})")

        # Attempt semantic split on the entire document content
        try:
            semantic_chunks = transcript_semantic_split(doc)
        except Exception as e:
            print(f"   - Error during semantic splitting for {source_info}: {e}")
            # Fallback: use the original document content if semantic split fails catastrophically
            if doc.page_content:
                 semantic_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata)]
                 print(f"   - Using original document content due to semantic split error.")
            else:
                 semantic_chunks = []


        # *** Simplified Fallback Logic ***
        # If semantic split produced *any* chunks (even just one chunk of the original content), proceed.
        if not semantic_chunks:
             print(f"   - No content chunks generated from semantic splitting for {source_info}. Skipping character split.")
             continue # Skip to the next document if no semantic chunks were made

        print(f"   - Applying character splitting to {len(semantic_chunks)} semantic chunk(s)...")

        # Initialize the character splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Index relative to the start of the semantic chunk
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        # Apply character splitting to the list of semantic chunks
        # This will further split long speaker turns if they exceed chunk_size
        try:
            # split_documents preserves metadata from the input semantic_chunks
            char_split_chunks = text_splitter.split_documents(semantic_chunks)
            all_final_chunks.extend(char_split_chunks)
            print(f"   - Character splitting resulted in {len(char_split_chunks)} final chunks for this document.")
        except Exception as e:
            print(f"   - Error during character splitting for {source_info}: {e}")


    print(f"\nFinished splitting. Total final chunks created: {len(all_final_chunks)}")
    if all_final_chunks:
        print("Metadata of the first final chunk:")
        print(all_final_chunks[0].metadata) # Check 'speaker' if available, 'source'
        print("Content snippet of the first final chunk:")
        print(repr(all_final_chunks[0].page_content[:300]) + "...")

    return all_final_chunks


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Testing Data Processing (Semantic Splitting on Whole Doc) ---")

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    data_directory = os.path.join(script_dir, 'data')
    if not os.path.exists(data_directory):
        data_directory = script_dir

    sample_pdf_path = os.path.join(data_directory, 'UpdEarnings-Call-Transcript-Q3FY25.pdf')
    if not os.path.exists(sample_pdf_path):
         print(f"\nERROR: Sample PDF not found at '{sample_pdf_path}'")
         print("Please ensure 'UpdEarnings-Call-Transcript-Q3FY25.pdf' is in the script's directory or a 'data' subdirectory.")
    else:
        print(f"Sample PDF found at: {sample_pdf_path}")

    print(f"Test configured to load documents from: {os.path.abspath(config.DATA_DIR)}")

    try:
        import langchain_community
        import langchain
        from unstructured.partition.pdf import partition_pdf
    except ImportError as import_error:
        print(f"\nMissing Dependency: {import_error}")
        print("\nPlease install required libraries:")
        print("pip install langchain langchain-community \"unstructured[pdf]\"") # Ensure unstructured[pdf]
        print("# On Debian/Ubuntu, you might also need OS packages:")
        print("# sudo apt-get update && sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr")
        exit(1)

    try:
        loaded_docs = load_documents(source_dir=config.DATA_DIR)
        if loaded_docs:
            chunked_docs = split_documents(
                loaded_docs,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            if not chunked_docs:
                 print("\nWarning: No final chunks were created after splitting.")
        else:
            print("\nNo documents were loaded. Check the data directory and file permissions.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during testing: {e}")
         import traceback
         print(traceback.format_exc())

    print("\n--- Data Processing Test Complete ---")