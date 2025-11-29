# Streamlit Cloud entry point - Self-contained version
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import everything we need
import streamlit as st
import time
from datetime import datetime
from pathlib import Path

# Simple fallback imports
try:
    from core.config import settings
except:
    class Settings:
        openai_model = "gpt-4o-mini"
        openai_embed_model = "text-embedding-3-small"
        chunk_size = 1000
    settings = Settings()

try:
    from text.chunker import chunk_text_smart
except:
    def chunk_text_smart(text, chunk_size=1000, overlap=200):
        # Simple chunking fallback
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

try:
    from ui.loaders.file_loader import extract_text_from_upload, extract_text_from_url
except:
    def extract_text_from_upload(file):
        return {"source_name": file.name, "text": "Error loading file", "metadata": {}}
    def extract_text_from_url(url):
        return {"source_name": url, "text": "Error loading URL", "metadata": {}}

try:
    from llm_providers.openai_embed import OpenAIEmbeddings
    from llm_providers.openai_llm import OpenAILLM
    from vectorstore.faiss_store import FaissStore
    from agents.agent_builder import Agent
except Exception as e:
    st.error(f"Failed to import core modules: {e}")
    st.error("Please check that all required files are present in the repository.")
    st.info("This is likely a deployment issue. The app works locally but has import problems on Streamlit Cloud.")
    st.stop()

# Now include the main app logic inline
def _init_session():
    if "embedder" not in st.session_state:
        st.session_state.embedder = OpenAIEmbeddings()
    if "llm" not in st.session_state:
        st.session_state.llm = OpenAILLM()
    if "store" not in st.session_state:
        st.session_state.store = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_chat_history" not in st.session_state:
        st.session_state.llm_chat_history = []
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    if "last_ingest_count" not in st.session_state:
        st.session_state.last_ingest_count = 0
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "🤖 Direct LLM Chat"

_init_session()

# Simple app
st.set_page_config(
    page_title="🤖 RAG Chatbot", 
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.write("**Note:** This is a simplified version for Streamlit Cloud deployment.")

# API Key input
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password")
    if st.button("Save Key"):
        if openai_key:
            st.session_state["OPENAI_API_KEY"] = openai_key
            st.success("API key saved!")

# Simple chat
st.header("💬 Chat")
if not st.session_state.get("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API key in the sidebar first.")
else:
    query = st.text_input("Ask a question:")
    if st.button("Send") and query:
        try:
            llm = OpenAILLM(api_key=st.session_state.get("OPENAI_API_KEY"))
            response = llm.generate(query)
            st.write(f"**Response:** {response}")
        except Exception as e:
            st.error(f"Error: {e}")