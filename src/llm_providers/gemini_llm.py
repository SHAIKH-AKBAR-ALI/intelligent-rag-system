# WHAT THIS MODULE DOES:
# - Simple Gemini LLM adapter; returns fallback if not configured.

import os
from typing import Optional
from src.core.config import settings

try:
    import google.generativeai as genai
except Exception:
    genai = None

class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = None, temperature: float = 0.0):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.temperature = temperature
        if genai is not None and self.api_key:
            try:
                if hasattr(genai, "configure"):
                    genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def generate(self, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
        temp = self.temperature if temperature is None else temperature
        if genai is not None and self.api_key:
            try:
                if hasattr(genai, "responses") and hasattr(genai.responses, "create"):
                    resp = genai.responses.create(model=self.model, input=prompt, temperature=temp, max_output_tokens=max_tokens)
                    if hasattr(resp, "output_text"):
                        return resp.output_text
                    out = getattr(resp, "output", None)
                    if out:
                        texts = []
                        for o in out:
                            cont = getattr(o, "content", None) or (o.get("content") if isinstance(o, dict) else None)
                            if cont:
                                for c in cont:
                                    t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                                    if t:
                                        texts.append(t)
                        if texts:
                            return "\n".join(texts)
                    return str(resp)
                if hasattr(genai, "chat") and hasattr(genai.chat, "complete"):
                    resp = genai.chat.complete(model=self.model, messages=[{"role":"user","content":prompt}], temperature=temp)
                    return getattr(resp, "content", "") or str(resp)
            except Exception as e:
                raise RuntimeError(f"Failed Gemini LLM call: {e}")
        fallback = ("NOTE: Gemini LLM client not configured. This is a fallback response used for local testing.\n"
                    "Set GEMINI_API_KEY and install google-generativeai or google-genai.\n\n"
                    "Prompt preview:\n" + prompt[:1000])
        return fallback
