# WHAT THIS MODULE DOES:
# - Groq LLM adapter.

import os
from typing import Optional
try:
    import streamlit as st
except ImportError:
    st = None
try:
    from src.core.config import settings
except ImportError:
    settings = None

try:
    from groq import Groq
except Exception:
    Groq = None

class GroqLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = None, temperature: float = 0.0):
        # Priority: passed key > session key > env key
        session_key = st.session_state.get("GROQ_API_KEY") if st else None
        self.api_key = api_key or session_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.temperature = temperature
        self.client = None
        if Groq is not None and self.api_key:
            self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        if self.client is None or not self.api_key:
            return f"[Groq not configured] Would respond to: {prompt[:100]}..."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Groq Error: {e}] Fallback response for: {prompt[:50]}..."