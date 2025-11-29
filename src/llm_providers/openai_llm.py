# WHAT THIS MODULE DOES:
# - OpenAI LLM adapter for GPT models.

import os
from typing import Optional
try:
    import streamlit as st
except ImportError:
    st = None
try:
    from core.config import settings
except ImportError:
    settings = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class OpenAILLM:
    def __init__(self, api_key: Optional[str] = None, model: str = None, temperature: float = 0.0):
        # Priority: passed key > session key > settings key > env key
        session_key = st.session_state.get("OPENAI_API_KEY") if st else None
        self.api_key = api_key or session_key or (settings.openai_api_key if settings else os.getenv("OPENAI_API_KEY"))
        self.model = model or (settings.openai_model if settings else os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.temperature = temperature
        self.client = None
        if OpenAI is not None and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        if self.client is None or not self.api_key:
            return f"[OpenAI not configured] Would respond to: {prompt[:100]}..."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error: {str(e).encode('ascii', 'ignore').decode('ascii')}] Fallback response for: {prompt[:50]}..."