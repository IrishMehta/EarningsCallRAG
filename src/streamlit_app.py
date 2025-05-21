import streamlit as st
import requests
import json
from typing import Dict, Any
import time

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

def query_rag_api(query: str) -> Dict[str, Any]:
    """Query the RAG API endpoint."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
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