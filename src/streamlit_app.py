import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import os

# Page config
st.set_page_config(
    page_title="Finance RAG Chatbot",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .confidence-score {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
    }
    .sources {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
        padding: 5px;
        background: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")

def query_rag_api(query: str) -> Dict[str, Any]:
    """Query the RAG API endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30  # Add timeout for better error handling
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Please check if the backend is running.")
        return None
    except Exception as e:
        st.error(f"Error querying API: {str(e)}")
        return None

def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display a message in the chat interface."""
    if is_user:
        st.chat_message("user").write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message["answer"])
            
            # Display confidence score
            st.markdown(
                f'<div class="confidence-score">Confidence: {message["confidence_score"]*100:.1f}%</div>',
                unsafe_allow_html=True
            )
            
            # Display sources
            sources = ", ".join([doc.get("source", "Unknown source") for doc in message["source_documents"]])
            st.markdown(
                f'<div class="sources">Sources: {sources}</div>',
                unsafe_allow_html=True
            )

# Title
st.title("Finance RAG Chatbot")
st.markdown("Ask questions about financial documents and get AI-powered responses!")

# Display chat history
for message in st.session_state.messages:
    display_message(message["content"], message["is_user"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat
    st.session_state.messages.append({"content": prompt, "is_user": True})
    display_message(prompt, True)
    
    # Show typing indicator
    with st.spinner("Thinking..."):
        # Query the API
        response = query_rag_api(prompt)
        
        if response:
            # Add bot response to chat
            st.session_state.messages.append({"content": response, "is_user": False})
            display_message(response)
        else:
            st.error("Failed to get response from the API. Please try again.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses a Retrieval-Augmented Generation (RAG) system to provide 
    accurate responses about financial documents.
    
    **Features:**
    - Real-time responses
    - Confidence scoring
    - Source attribution
    - Powered by HuggingFace models
    
    **Models Used:**
    - Embedding: sentence-transformers/all-MiniLM-L6-v2
    - LLM: HuggingFaceH4/zephyr-7b-beta
    """)
    
    # Add deployment status
    st.header("System Status")
    try:
        health_check = requests.get(f"{API_URL}/health", timeout=5)
        if health_check.status_code == 200:
            st.success("Backend API is running")
        else:
            st.error("Backend API is not responding correctly")
    except:
        st.error("Cannot connect to backend API")