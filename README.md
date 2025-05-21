<!-- PROJECT SHIELDS -->
[![Python][Python-shield]][Python-url]
[![FastAPI][FastAPI-shield]][FastAPI-url]
[![LangChain][LangChain-shield]][LangChain-url]
[![FAISS][FAISS-shield]][FAISS-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Finance RAG Bot</h3>

  <p align="center">
    A Retrieval-Augmented Generation (RAG) system for financial document analysis
    <br />
    <a href="#about-the-project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#getting-started">Getting Started</a>
    &middot;
    <a href="#usage">Usage</a>
    &middot;
    <a href="#features">Features</a>
    &middot;
    <a href="#roadmap">Roadmap</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#logging">Logging & Observability</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Finance RAG Bot is a powerful Retrieval-Augmented Generation system designed specifically for analyzing and querying financial documents (Currently supports earnings call transcripts). It combines the capabilities of large language models with efficient document retrieval to provide accurate and context-aware responses to financial queries.

### Built With

* [![Python][Python-shield]][Python-url]
* [![FastAPI][FastAPI-shield]][FastAPI-url]
* [![LangChain][LangChain-shield]][LangChain-url]
* [![FAISS][FAISS-shield]][FAISS-url]

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.8+
* pip (Python package manager)
* Hugging Face Hub API token

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/your_username/finance_rag_bot.git
   cd finance_rag_bot
   ```

2. Create and activate a virtual environment
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

4. Set up environment variables
   ```sh
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Deployment

The application is deployed using HuggingFace Spaces:

### HuggingFace Spaces Configuration

The project includes the following files for HuggingFace Spaces deployment:

- `src/streamlit_app.py`: Streamlit interface for the RAG system
- `requirements.txt`: Python dependencies
- `.env.example`: Example environment variables
- `README.md`: Project documentation

To deploy:
1. Fork this repository
2. Create a new Space on HuggingFace
3. Select Streamlit as the SDK
4. Point to the forked repository
5. Add your environment variables
6. Deploy!

The Streamlit interface provides:
- Real-time chat interface
- Confidence score display
- Source document attribution
- Responsive design
- Error handling
- Chat history

<!-- USAGE -->
## Usage

1. Place your financial documents in the `data/` directory
2. Run the FastAPI server:
   ```sh
   uvicorn src.main:app --reload
   ```
3. Access the API documentation at `http://localhost:8000/docs`

<!-- FEATURES -->
## Features

* **Efficient Document Processing**
  - Support for PDF and TXT files
  - Semantic splitting based on speaker patterns
  - Intelligent text chunking with overlap

* **Advanced RAG Implementation**
  - FAISS-based vector storage for fast similarity search
  - Configurable embedding models and LLM backends
  - Confidence scoring for response quality assessment
  - Powered by HuggingFace models:
    - Embedding: sentence-transformers/all-MiniLM-L6-v2 for efficient text vectorization
    - LLM: HuggingFaceH4/zephyr-7b-beta for high-quality response generation

* **Robust API**
  - FastAPI-based REST API
  - Comprehensive request/response logging
  - Health check endpoint
  - Structured error handling

* **Enhanced Observability**
  - Structured JSON logging
  - Performance metrics tracking
  - Detailed error reporting
  - Request tracing

* **Modern Web Interface**
  - Clean and responsive chat interface
  - Real-time message updates
  - Confidence score display
  - Source document attribution
  - Typing indicators
  - Error handling and user feedback

## Limitations

* **Document Processing**
  - Limited to PDF and TXT file formats
  - No support for images or tables in documents
  - Maximum document size restrictions
  - No automatic document versioning

* **RAG System**
  - Response quality depends on document quality and relevance
  - May struggle with complex financial calculations
  - Limited context window for very long documents
  - No multi-turn conversation memory
  - No functionality to process tabular numeric data

* **Performance**
  - Initial document processing can be time-consuming
  - No caching mechanism for frequent queries
  - Vector store size limitations
  - No distributed processing support

## Disclaimer

This project was developed with significant assistance from AI tools, with approximately 60% of the codebase being generated or modified by AI. While the code has been reviewed and tested, it's important to note that:

* The system may contain AI-generated code patterns and structures
* Some implementations may follow AI-suggested best practices
* Code quality and security should be thoroughly reviewed before deployment
* Regular updates and maintenance are recommended

<!-- ROADMAP -->
## Roadmap

- [x] Basic RAG implementation
- [x] FastAPI integration
- [x] FAISS vector store
- [x] Enhanced logging and observability
- [ ] Add support for more document formats
- [ ] Add functionality to parse unstructured tabular data (excel/pdf)
- [ ] Add support for numerical queries like revenue, growth etc
- [ ] Implement caching for improved performance


<!-- PROJECT STRUCTURE -->
## Project Structure

```
finance_rag_bot/
├── frontend/
│   └── index.html
├── logs/
│   └── finance_rag.log
│
├── data/                  # Directory for source documents
│   ├── transcript_1.txt
│   └── ...                # Add your documents here
│
├── vector_store/          # FAISS index storage
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_processing.py # Document loading and splitting
│   ├── embedding_store.py # Embedding and vector store ops
│   ├── qa_chain.py        # RAG chain and confidence scoring
│   └── main.py            # FastAPI application
│
├── logs/                  # Application logs
│   └── finance_rag.log    # Structured JSON logs
│
├── .env                   # Environment variables
├── .env.example          # Example environment file
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

<!-- LOGGING -->
## Logging & Observability
Logs are written to both console and file (`logs/finance_rag.log`).

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/Python-3.8+-blue.svg
[Python-url]: https://www.python.org/
[FastAPI-shield]: https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[LangChain-shield]: https://img.shields.io/badge/LangChain-FF6B6B?style=flat&logo=python&logoColor=white
[LangChain-url]: https://python.langchain.com/
[FAISS-shield]: https://img.shields.io/badge/FAISS-00A98F?style=flat&logo=python&logoColor=white
[FAISS-url]: https://github.com/facebookresearch/faiss