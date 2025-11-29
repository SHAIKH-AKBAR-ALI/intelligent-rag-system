# WHAT THIS MODULE DOES:
# - OpenAI embeddings adapter.

import os
from typing import List, Optional
try:
    import streamlit as st
except ImportError:
    st = None
try:
    from src.core.config import settings
except ImportError:
    settings = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class OpenAIEmbeddings:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Priority: passed key > session key > settings key > env key
        session_key = st.session_state.get("OPENAI_API_KEY") if st else None
        self.api_key = api_key or session_key or (settings.openai_api_key if settings else os.getenv("OPENAI_API_KEY"))
        self.model = model or (settings.openai_embed_model if settings else os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
        self.client = None
        if OpenAI is not None and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if self.client is None or not self.api_key:
            raise NotImplementedError("OpenAI not configured or installed.")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding error: {str(e).encode('ascii', 'ignore').decode('ascii')}"