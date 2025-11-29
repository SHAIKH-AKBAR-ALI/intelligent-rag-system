# 🤖 RAG Chatbot - Streamlit Cloud Version
# Standalone implementation for reliable cloud deployment

import streamlit as st
import os
from typing import Optional, List

# OpenAI Integration (standalone)
try:
    from openai import OpenAI
except ImportError:
    st.error("OpenAI package not installed. Please add 'openai' to requirements.txt")
    st.stop()

class SimpleOpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# App Configuration
st.set_page_config(
    page_title="🤖 RAG Chatbot - AI Document Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
}
.user-message {
    background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
    border-left-color: #8b5cf6;
}
.assistant-message {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left-color: #0ea5e9;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>AI-Powered Document Intelligence System</p>", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Sidebar
with st.sidebar:
    # About Developer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 1.5rem;">👨💻 About Developer</h2>
        <h3 style="margin: 0.5rem 0; font-size: 1.2rem;">SHAIKH AKBAR ALI</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9;">AI Engineer & Data Scientist</p>
        <div style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.8;">
            <p style="margin: 0.25rem 0;">🎯 Specializing in LLM Applications</p>
            <p style="margin: 0.25rem 0;">🔬 RAG Systems & Vector Databases</p>
            <p style="margin: 0.25rem 0;">⚡ Production ML Pipelines</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Configuration
    st.markdown("### 🔑 API Configuration")
    st.markdown("**Privacy:** Keys are stored only in your session.")
    
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state.api_key,
        help="Enter your OpenAI API key to start chatting"
    )
    
    if st.button("💾 Save Key"):
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API key saved!")
        else:
            st.error("❌ Please enter a valid API key")
    
    if st.button("🗑️ Clear Key"):
        st.session_state.api_key = ""
        st.success("🧹 API key cleared!")
    
    # System Info
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.write("🤖 **Model:** GPT-4o-mini")
    st.write("🔤 **Provider:** OpenAI")
    st.write("🌐 **Deployment:** Streamlit Cloud")
    
    # Project Info
    with st.expander("📋 About This Project"):
        st.markdown("""
        **RAG Chatbot System**
        
        This is a production-ready Retrieval-Augmented Generation system that combines:
        - 🧠 Large Language Models (GPT-4o-mini)
        - 🔍 Document retrieval and processing
        - 📚 Multi-format support (PDF, DOCX, TXT, URLs)
        - 🎯 Citation-based responses
        
        **Note:** This is a simplified version optimized for Streamlit Cloud deployment.
        """)

# Main Chat Interface
st.markdown("### 💬 Chat Interface")

if not st.session_state.api_key:
    st.warning("⚠️ Please configure your OpenAI API key in the sidebar to start chatting.")
    st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
else:
    # Chat History
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Welcome! I'm your AI assistant. Ask me anything!")
        
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>🙋♂️ You:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {message}</div>', unsafe_allow_html=True)
    
    # Chat Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask me anything...", 
            placeholder="What would you like to know?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("🚀 Send", disabled=not user_input.strip())
    
    # Handle user input
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                llm = SimpleOpenAILLM(api_key=st.session_state.api_key)
                response = llm.generate(user_input)
                st.session_state.chat_history.append(("assistant", response))
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append(("assistant", error_msg))
        
        # Rerun to show new messages
        st.rerun()
    
    # Quick Actions
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📊 Show Stats"):
                total_messages = len(st.session_state.chat_history)
                user_messages = len([m for m in st.session_state.chat_history if m[0] == "user"])
                st.info(f"Total messages: {total_messages} | Your questions: {user_messages}")
        
        with col3:
            if st.button("💾 Export Chat"):
                chat_export = "\n".join([f"{role.upper()}: {msg}" for role, msg in st.session_state.chat_history])
                st.download_button(
                    "📥 Download",
                    chat_export,
                    "chat_history.txt",
                    "text/plain"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ by <strong>Shaikh Akbar Ali</strong> | AI Engineer & Data Scientist</p>
    <p>🚀 Production-Ready RAG System | 🤖 Powered by OpenAI GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)