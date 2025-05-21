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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Finance RAG Bot is a powerful Retrieval-Augmented Generation system designed specifically for analyzing and querying financial documents. It combines the capabilities of large language models with efficient document retrieval to provide accurate and context-aware responses to financial queries.

Key Features:
* Efficient document processing and chunking
* FAISS-based vector storage for fast similarity search
* FastAPI-based REST API for easy integration
* Configurable embedding models and LLM backends
* Confidence scoring for response quality assessment

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

<!-- USAGE -->
## Usage

1. Place your financial transcripts in the `data/` directory
2. Run the FastAPI server:
   ```sh
   uvicorn src.main:app --reload
   ```
3. Access the API documentation at `http://localhost:8000/docs`

<!-- ROADMAP -->
## Roadmap

- [x] Basic RAG implementation
- [x] FastAPI integration
- [x] FAISS vector store
- [ ] Add support for more document formats
- [ ] Implement caching for improved performance
- [ ] Add user authentication
- [ ] Add batch processing capabilities
- [ ] Implement advanced confidence scoring

<!-- PROJECT STRUCTURE -->
## Project Structure

```
finance_rag_bot/
│
├── data/                  # Directory to store the source transcripts
│   ├── transcript_1.txt
│   └── ...                # Add your 5 transcript files here
│
├── vector_store/          # Directory to save the FAISS index
│
├── src/                   # Source code directory
│   ├── __init__.py
│   ├── config.py          # Configuration (e.g., model names, paths)
│   ├── data_processing.py # Functions for loading and splitting data
│   ├── embedding_store.py # Functions for embedding and vector store ops
│   ├── qa_chain.py        # Functions for setting up the RAG chain and confidence
│   └── main.py            # FastAPI application
│
├── .env                   # To store sensitive info like API keys (DO NOT COMMIT)
├── Dockerfile             # Instructions to build the Docker image
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/Python-3.8+-blue.svg
[Python-url]: https://www.python.org/
[FastAPI-shield]: https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[LangChain-shield]: https://img.shields.io/badge/LangChain-FF6B6B?style=flat&logo=python&logoColor=white
[LangChain-url]: https://python.langchain.com/
[FAISS-shield]: https://img.shields.io/badge/FAISS-00A98F?style=flat&logo=python&logoColor=white
[FAISS-url]: https://github.com/facebookresearch/faiss