# WHAT THIS MODULE DOES:
# - Streamlit UI for ingestion + chat.

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from pathlib import Path
import time
from datetime import datetime
from src.core.config import settings
from src.text.chunker import chunk_text_smart
from src.ui.loaders.file_loader import extract_text_from_upload, extract_text_from_url
from src.llm_providers.openai_embed import OpenAIEmbeddings
from src.llm_providers.openai_llm import OpenAILLM
from src.vectorstore.faiss_store import FaissStore
from src.agents.agent_builder import Agent

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
    if "last_ingest_count" not in st.session_state:
        st.session_state.last_ingest_count = 0
    if "processing" not in st.session_state:
        st.session_state.processing = False

_init_session()

def _create_store_if_missing(dim: int):
    store = st.session_state.get("store")
    if store is None:
        store = FaissStore(dim=dim)
        st.session_state.store = store
    return store

def _ingest_text(source_name: str, text: str, metadata: dict, chunk_size=None, overlap=None):
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if not text or len(text.strip()) < 50:
        st.warning("Extracted text empty or too short.")
        return 0
    chunks = chunk_text_smart(text, chunk_size=chunk_size, overlap=overlap)
    st.info(f"Split into {len(chunks)} chunks. Creating embeddings...")
    try:
        embeddings = st.session_state.embedder.embed_texts(chunks)
    except Exception as e:
        st.error(f"Embeddings failed: {e}")
        return 0
    dim = len(embeddings[0])
    store = _create_store_if_missing(dim)
    metas = []
    for i in range(len(chunks)):
        md = metadata.copy() if isinstance(metadata, dict) else {"source_name": source_name}
        md.update({"chunk_index": i})
        metas.append(md)
    ids = store.upsert(embeddings, chunks, metas)
    st.success(f"Ingested {len(ids)} chunks from {source_name}")
    st.session_state.last_ingest_count = (st.session_state.last_ingest_count or 0) + len(ids)
    st.session_state.agent = Agent(embedder=st.session_state.embedder, llm=st.session_state.llm, store=store)
    return len(ids)

st.set_page_config(
    page_title="🤖 RAG Chatbot - AI Document Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional RAG System by Shaikh Akbar Ali",
        'Get help': None,
        'Report a bug': None
    }
)

# Enhanced CSS for production-ready styling
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Header Styles */
.main-header {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Sidebar Styles */
.sidebar-section {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
}

/* Chat Message Styles */
.chat-message {
    padding: 1.25rem;
    border-radius: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

.user-message {
    background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
    border-left: 4px solid #8b5cf6;
}

.assistant-message {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid #0ea5e9;
}

/* Button Styles */
.stButton > button {
    width: 100%;
    border-radius: 0.75rem;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Input Styles */
.stTextInput > div > div > input {
    border-radius: 0.75rem;
    border: 2px solid #e2e8f0;
    padding: 0.75rem;
    font-size: 0.95rem;
}

.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Radio Button Styles */
.stRadio > div {
    background: #f8fafc;
    padding: 0.75rem;
    border-radius: 0.75rem;
    border: 1px solid #e2e8f0;
}

/* Expander Styles */
.streamlit-expanderHeader {
    background: #f8fafc;
    border-radius: 0.5rem;
    font-weight: 500;
}

/* Status Indicators */
.status-success {
    background: #dcfce7;
    color: #166534;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #22c55e;
    margin: 0.5rem 0;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f59e0b;
    margin: 0.5rem 0;
}

.status-info {
    background: #dbeafe;
    color: #1e40af;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 0.5rem 0;
}

/* Professional Card Style */
.info-card {
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
    margin: 1rem 0;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Document Intelligence System</p>', unsafe_allow_html=True)

# Professional Info Banner
st.markdown("""
<div class="info-card">
    <div style="text-align: center;">
        <h3 style="margin: 0; color: #1f2937;">🚀 Production-Ready RAG Architecture</h3>
        <p style="margin: 0.5rem 0; color: #6b7280;">Advanced Retrieval-Augmented Generation with Multi-Provider LLM Support</p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
            <span style="background: #eff6ff; color: #1d4ed8; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem;">OpenAI GPT-4o-mini</span>
            <span style="background: #f0fdf4; color: #166534; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem;">FAISS Vector DB</span>
            <span style="background: #fef7ff; color: #7c2d12; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem;">Smart Chunking</span>
            <span style="background: #fff7ed; color: #9a3412; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem;">Citation-Based RAG</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    # About Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 600;">👨‍💻 About Developer</h2>
        <h3 style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: 500;">SHAIKH AKBAR ALI</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9; font-size: 0.95rem;">AI Engineer & Data Scientist</p>
        <div style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.8;">
            <p style="margin: 0.25rem 0;">🎯 Specializing in LLM Applications</p>
            <p style="margin: 0.25rem 0;">🔬 RAG Systems & Vector Databases</p>
            <p style="margin: 0.25rem 0;">⚡ Production ML Pipelines</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Architecture Overview
    with st.expander("🏗️ System Architecture", expanded=False):
        st.markdown("""
        **RAG Pipeline Flow:**
        
        1️⃣ **Document Ingestion**
        - Multi-format support (PDF, DOCX, TXT, URLs)
        - Intelligent content extraction & cleaning
        
        2️⃣ **Text Processing**
        - Sentence-aware chunking (1000 chars)
        - 200-char overlap for context continuity
        
        3️⃣ **Embedding & Storage**
        - OpenAI text-embedding-3-small (1536D)
        - FAISS vector database with cosine similarity
        
        4️⃣ **Retrieval & Generation**
        - Semantic search for relevant chunks
        - Citation-based prompt engineering
        - GPT-4o-mini for grounded responses
        
        **Tech Stack:**
        - 🧠 **LLM**: OpenAI GPT-4o-mini
        - 🔍 **Embeddings**: text-embedding-3-small
        - 📊 **Vector DB**: FAISS (local deployment)
        - 🎨 **Frontend**: Streamlit with custom CSS
        - 🐍 **Backend**: Python with modular architecture
        """)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    nav_option = st.radio(
        "Choose Task:",
        ["🔑 API Keys", "📚 Documents", "📈 Status", "ℹ️ About"],
        horizontal=False
    )
    
    st.markdown("---")
    
    # API Keys Section
    if nav_option == "🔑 API Keys":
        st.markdown("### 🔑 API Keys Configuration")
        st.markdown("**Privacy:** Keys are stored only in your session and never saved to server.")
        
        # Initialize session state for keys
        if "api_keys_configured" not in st.session_state:
            st.session_state.api_keys_configured = False
        if "show_keys" not in st.session_state:
            st.session_state.show_keys = False
        
        # Toggle to show/hide keys
        show_keys = st.checkbox("👁️ Show keys", value=st.session_state.show_keys)
        st.session_state.show_keys = show_keys
        input_type = "default" if show_keys else "password"
        
        # API Key inputs
        openai_key = st.text_input("OpenAI API Key", 
                                  value=st.session_state.get("OPENAI_API_KEY", ""),
                                  type=input_type, key="openai_input")
        st.caption("Model: gpt-4o-mini | Embeddings: text-embedding-3-small")
        
        groq_key = st.text_input("Groq API Key", 
                                value=st.session_state.get("GROQ_API_KEY", ""),
                                type=input_type, key="groq_input")
        st.caption("Model: llama3-8b-8192")
        
        gemini_key = st.text_input("Gemini API Key", 
                                  value=st.session_state.get("GEMINI_API_KEY", ""),
                                  type=input_type, key="gemini_input")
        st.caption("Model: gemini-2.5-pro")
        
        hf_key = st.text_input("HuggingFace API Key", 
                              value=st.session_state.get("HUGGINGFACE_API_KEY", ""),
                              type=input_type, key="hf_input")
        st.caption("Model: sentence-transformers/all-mpnet-base-v2")
        
        # Buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save"):
                if openai_key: st.session_state["OPENAI_API_KEY"] = openai_key
                if groq_key: st.session_state["GROQ_API_KEY"] = groq_key
                if gemini_key: st.session_state["GEMINI_API_KEY"] = gemini_key
                if hf_key: st.session_state["HUGGINGFACE_API_KEY"] = hf_key
                st.session_state.api_keys_configured = True
                st.success("Keys saved to session!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🧪 Test"):
                if openai_key:
                    try:
                        from src.llm_providers.openai_embed import OpenAIEmbeddings
                        embedder = OpenAIEmbeddings(api_key=openai_key)
                        embedder.embed_texts(["test"])
                        st.success("OpenAI API key validated successfully!")
                    except Exception as e:
                        error_msg = str(e)
                        if "401" in error_msg:
                            st.error("Invalid OpenAI API key. Please check your key.")
                        elif "quota" in error_msg.lower():
                            st.error("OpenAI quota exceeded. Check your billing.")
                        else:
                            st.error(f"OpenAI failed: {error_msg[:100]}...")
                else:
                    st.warning("⚠️ Enter OpenAI key to test")
        
        with col3:
            if st.button("🗑️ Clear"):
                for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "HUGGINGFACE_API_KEY"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.api_keys_configured = False
                st.success("Keys cleared from session!")
                time.sleep(1)
                st.rerun()
    
    # Document Ingestion Section
    elif nav_option == "📚 Documents":
        st.markdown("### 📚 Document Ingestion")
        
        # URL Ingestion
        with st.expander("🌐 Ingest from URL", expanded=True):
            url_input = st.text_input("Enter URL", placeholder="https://example.com/article")
            ingest_url_btn = st.button("🔄 Ingest URL", disabled=st.session_state.processing)
        
        # File Upload
        with st.expander("📁 Upload Documents", expanded=True):
            uploaded = st.file_uploader(
                "Choose files", 
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=False,
                help="Supported: PDF, DOCX, TXT, MD"
            )
            ingest_file_btn = st.button("📤 Ingest File", disabled=st.session_state.processing)
        
        # Processing logic
        if ingest_url_btn and url_input and not st.session_state.processing:
            if not st.session_state.get("OPENAI_API_KEY"):
                st.error("❌ Please configure your OpenAI API key first")
            else:
                st.session_state.processing = True
                with st.spinner("🔍 Fetching and processing URL..."):
                    try:
                        res = extract_text_from_url(url_input)
                        count = _ingest_text(res["source_name"], res["text"], res["metadata"])
                        if count > 0:
                            st.success(f"✅ Successfully ingested {count} chunks!")
                            time.sleep(1)
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")
                    finally:
                        st.session_state.processing = False
                        st.rerun()
        elif ingest_url_btn and not url_input:
            st.warning("⚠️ Please enter a URL first")
        
        if ingest_file_btn and uploaded and not st.session_state.processing:
            if not st.session_state.get("OPENAI_API_KEY"):
                st.error("❌ Please configure your OpenAI API key first")
            else:
                st.session_state.processing = True
                with st.spinner(f"📖 Processing {uploaded.name}..."):
                    try:
                        res = extract_text_from_upload(uploaded)
                        count = _ingest_text(res["source_name"], res["text"], res["metadata"])
                        if count > 0:
                            st.success(f"✅ Successfully ingested {count} chunks!")
                            time.sleep(1)
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")
                    finally:
                        st.session_state.processing = False
                        st.rerun()
        elif ingest_file_btn and not uploaded:
            st.warning("⚠️ Please upload a file first")
    
    # Status Section
    elif nav_option == "📈 Status":
        st.markdown("### 📈 System Status")
        
        # API Keys Status
        st.markdown("#### 🔑 API Keys")
        keys_status = []
        for provider, key_name in [("OpenAI", "OPENAI_API_KEY"), ("Groq", "GROQ_API_KEY"), 
                                  ("Gemini", "GEMINI_API_KEY"), ("HuggingFace", "HUGGINGFACE_API_KEY")]:
            if st.session_state.get(key_name):
                st.success(f"✅ {provider} configured")
                keys_status.append(True)
            else:
                st.info(f"ℹ️ {provider} not configured")
                keys_status.append(False)
        
        st.markdown("#### 🗄️ Vector Store")
        store = st.session_state.get("store")
        
        if store and store.index.ntotal > 0:
            st.success(f"📊 **{store.index.ntotal}** vectors stored")
            st.info(f"📈 **{st.session_state.get('last_ingest_count', 0)}** total ingested")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Preview"):
                    st.session_state.show_preview = not st.session_state.get('show_preview', False)
            with col2:
                if st.button("🗑️ Clear All"):
                    st.session_state.store = None
                    st.session_state.agent = None
                    st.session_state.last_ingest_count = 0
                    st.session_state.chat_history = []
                    st.success("Vector store cleared successfully!")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("📭 No documents ingested yet")
        
        # Show preview if requested
        if st.session_state.get('show_preview', False) and store and store.index.ntotal > 0:
            st.markdown("#### 👀 Document Preview")
            preview_ids = store.id_map[:min(3, len(store.id_map))]
            for i, pid in enumerate(preview_ids):
                md = store.metadatas.get(pid, {})
                source_name = md.get('metadata', {}).get('source_name', 'Unknown')
                chunk_idx = md.get('metadata', {}).get('chunk_index', 0)
                text_preview = md.get("text", "")[:150] + "..."
                
                with st.expander(f"📄 {source_name} (chunk {chunk_idx})"):
                    st.write(text_preview)
        
        st.markdown("#### ⚙️ System Info")
        st.write(f"🤖 **LLM:** {settings.openai_model}")
        st.write(f"🔤 **Embeddings:** {settings.openai_embed_model}")
        st.write(f"📏 **Chunk Size:** {settings.chunk_size}")
        
        # Chat Status
        st.markdown("#### 💬 Chat Status")
        if st.session_state.get('last_ingest_count', 0) > 0 and st.session_state.get("OPENAI_API_KEY"):
            st.success("✅ Ready to chat!")
        elif not st.session_state.get("OPENAI_API_KEY"):
            st.warning("⚠️ Configure API key first")
        else:
            st.warning("⚠️ Ingest documents first")
    
    # About Section
    elif nav_option == "ℹ️ About":
        st.markdown("### ℹ️ Project Information")
        
        st.markdown("""
        #### 🎯 Project Overview
        This is a **production-ready RAG (Retrieval-Augmented Generation) system** that combines the power of large language models with precise document retrieval to provide accurate, cited responses.
        
        #### 🔧 Key Features
        - **Dual Chat Modes**: Direct LLM chat + Document-based RAG
        - **Multi-Format Support**: PDF, DOCX, TXT files, and web URLs
        - **Smart Processing**: Sentence-aware chunking with overlap
        - **Vector Search**: FAISS-powered semantic similarity
        - **Source Citations**: Transparent response attribution
        - **Session Security**: API keys never stored on server
        
        #### 🏗️ Technical Architecture
        
        **Ingestion Pipeline:**
        ```
        Documents → Content Extraction → Smart Chunking → 
        Embeddings → FAISS Storage → Ready for Queries
        ```
        
        **Query Pipeline:**
        ```
        User Query → Query Embedding → Similarity Search → 
        Context Assembly → LLM Generation → Cited Response
        ```
        
        #### 💡 Innovation Highlights
        - **Modular Design**: Clean separation of concerns
        - **Provider Abstraction**: Support for multiple LLM providers
        - **Local Vector DB**: FAISS for privacy and cost control
        - **Intelligent Chunking**: Preserves semantic coherence
        - **Citation System**: Maintains factual grounding
        
        #### 🚀 Production Considerations
        - **Scalability**: Designed for horizontal scaling
        - **Security**: Session-only API key storage
        - **Performance**: Optimized embedding and retrieval
        - **Reliability**: Error handling and fallback mechanisms
        - **Monitoring**: Built-in status and diagnostics
        """)
        
        st.markdown("---")
        
        # Developer Contact
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; text-align: center;">
            <h4 style="margin: 0; color: #1f2937;">👨‍💻 Developer</h4>
            <h3 style="margin: 0.5rem 0; color: #667eea;">SHAIKH AKBAR ALI</h3>
            <p style="margin: 0; color: #6b7280;">AI Engineer & Data Scientist</p>
            <p style="margin: 0.5rem 0; color: #6b7280; font-size: 0.9rem;">Specializing in LLM Applications, RAG Systems & Production ML</p>
        </div>
        """, unsafe_allow_html=True)

# Main Chat Interface
st.markdown("""
<div style="background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1rem 0;">
    <h2 style="margin: 0; color: #1f2937; display: flex; align-items: center; gap: 0.5rem;">
        💬 <span>Intelligent Chat Interface</span>
    </h2>
    <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Choose your interaction mode and start conversing with AI</p>
</div>
""", unsafe_allow_html=True)

# Chat mode selection with enhanced styling
col1, col2 = st.columns(2)
with col1:
    llm_mode = st.button(
        "🤖 Direct LLM Chat",
        help="Ask anything - General AI assistance",
        use_container_width=True
    )
with col2:
    rag_mode = st.button(
        "📚 RAG Chat (Documents)", 
        help="Query your uploaded documents",
        use_container_width=True
    )

# Initialize chat mode in session state
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "🤖 Direct LLM Chat"

if llm_mode:
    st.session_state.chat_mode = "🤖 Direct LLM Chat"
if rag_mode:
    st.session_state.chat_mode = "📚 RAG Chat (Documents)"

chat_mode = st.session_state.chat_mode

# Mode indicator
if chat_mode == "🤖 Direct LLM Chat":
    st.markdown("""
    <div class="status-info">
        <strong>🤖 Direct LLM Mode Active</strong><br>
        Ask me anything! I can help with general questions, coding, explanations, creative tasks, and more.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-success">
        <strong>📚 RAG Mode Active</strong><br>
        Ask questions about your uploaded documents. I'll provide answers with source citations.
    </div>
    """, unsafe_allow_html=True)

# Initialize separate chat histories
if "llm_chat_history" not in st.session_state:
    st.session_state.llm_chat_history = []
if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = []

# Select appropriate chat history
if chat_mode == "🤖 Direct LLM Chat":
    current_chat = st.session_state.llm_chat_history
    st.info("💡 Direct LLM mode: Ask anything! No document context needed.")
else:
    current_chat = st.session_state.rag_chat_history
    st.info("📖 RAG mode: Ask questions about your uploaded documents.")

# Chat history display
chat_container = st.container()
with chat_container:
    if not current_chat:
        if chat_mode == "🤖 Direct LLM Chat":
            st.info("👋 Ask me anything! I can help with general questions, coding, explanations, and more.")
        else:
            st.info("👋 Upload some documents first, then ask questions about them.")
    
    for i, entry in enumerate(current_chat):
        role, text, sources = entry
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>🙋‍♂️ You:</strong> {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {text}</div>', unsafe_allow_html=True)
            
            if sources:
                with st.expander(f"📁 Sources ({len(sources)} documents)"):
                    for j, s in enumerate(sources, 1):
                        source_name = s.get('source_name', 'Unknown')
                        chunk_idx = s.get('chunk_index', 0)
                        st.markdown(f"**{j}.** `{source_name}` — chunk {chunk_idx}")

# Chat input section with enhanced styling
st.markdown("""
<div style="background: #f8fafc; padding: 1rem; border-radius: 1rem; margin: 1.5rem 0; border: 1px solid #e2e8f0;">
    <h4 style="margin: 0 0 1rem 0; color: #374151;">💭 Ask Your Question</h4>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

# Check requirements based on chat mode
api_key_missing = not st.session_state.get("OPENAI_API_KEY")
if chat_mode == "🤖 Direct LLM Chat":
    ingestion_required = False
    placeholder = "Ask me anything..."
    input_label = "Ask any question:"
else:
    ingestion_required = st.session_state.get('last_ingest_count', 0) == 0
    placeholder = "Ask about your documents..." if not ingestion_required else "Please ingest documents first"
    input_label = "Ask about your documents:"

with col1:
    query = st.text_input(
        input_label,
        key="query_input",
        placeholder=placeholder,
        disabled=st.session_state.processing or (ingestion_required and chat_mode == "📚 RAG Chat (Documents)") or api_key_missing
    )

with col2:
    send_btn = st.button(
        "🚀 Send", 
        disabled=st.session_state.processing or not query.strip() or (ingestion_required and chat_mode == "📚 RAG Chat (Documents)") or api_key_missing
    )

# Show warnings based on mode
if api_key_missing:
    st.warning("⚠️ Please configure your OpenAI API key first.")
elif chat_mode == "📚 RAG Chat (Documents)" and ingestion_required:
    st.warning("⚠️ Please ingest at least one URL or file (PDF/DOCX/TXT) first for RAG mode.")

# Handle send button
if send_btn and query.strip() and not api_key_missing:
    can_proceed = True
    
    # Check mode-specific requirements
    if chat_mode == "📚 RAG Chat (Documents)" and ingestion_required:
        can_proceed = False
    elif chat_mode == "📚 RAG Chat (Documents)" and st.session_state.get("agent") is None:
        st.warning("⚠️ Please ingest some documents first.")
        can_proceed = False
    
    if can_proceed:
        st.session_state.processing = True
        with st.spinner("🤔 Thinking..."):
            try:
                if chat_mode == "🤖 Direct LLM Chat":
                    # Direct LLM chat - no RAG
                    llm = st.session_state.llm
                    answer = llm.generate(query, max_tokens=1000)
                    sources = []
                    
                    # Add to LLM chat history
                    st.session_state.llm_chat_history.append(("user", query, None))
                    st.session_state.llm_chat_history.append(("assistant", answer, sources))
                else:
                    # RAG chat - use documents
                    res = st.session_state.agent.run(query, return_raw_chunks=True)
                    answer = res.get("answer", "")
                    sources = res.get("sources", [])
                    
                    # Add to RAG chat history
                    st.session_state.rag_chat_history.append(("user", query, None))
                    st.session_state.rag_chat_history.append(("assistant", answer, sources))
                
                # Clear input and refresh
                st.session_state.query_input = ""
                
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                st.session_state.processing = False
                st.rerun()

# Enhanced Quick Actions
if current_chat:
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 1rem; margin: 1.5rem 0; border: 1px solid #e2e8f0;">
        <h4 style="margin: 0 0 1rem 0; color: #374151;">⚡ Quick Actions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear current conversation"):
            if chat_mode == "🤖 Direct LLM Chat":
                st.session_state.llm_chat_history = []
            else:
                st.session_state.rag_chat_history = []
            st.success("Chat cleared!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("📊 Show Stats", help="Display conversation statistics"):
            total_messages = len(current_chat)
            user_messages = len([m for m in current_chat if m[0] == "user"])
            mode_name = "LLM" if chat_mode == "🤖 Direct LLM Chat" else "RAG"
            st.markdown(f"""
            <div class="status-info">
                <strong>{mode_name} Chat Statistics</strong><br>
                Total Messages: {total_messages} | Your Questions: {user_messages}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.button("💾 Export Chat", help="Export conversation history"):
            chat_export = "\n".join([f"{entry[0].upper()}: {entry[1]}" for entry in current_chat])
            st.download_button(
                "📥 Download",
                chat_export,
                f"{mode_name.lower()}_chat_export.txt",
                "text/plain"
            )
    
    with col4:
        if st.button("🔄 Refresh", help="Refresh the interface"):
            st.rerun()
else:
    # Show helpful tips when no chat history
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; border: 1px solid #0ea5e9;">
        <h4 style="margin: 0; color: #0c4a6e;">💡 Getting Started</h4>
        <ul style="margin: 0.5rem 0; color: #075985;">
            <li>Configure your OpenAI API key in the sidebar</li>
            <li>For RAG mode: Upload documents first</li>
            <li>For LLM mode: Start asking questions immediately</li>
            <li>Use the navigation tabs to switch between features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

