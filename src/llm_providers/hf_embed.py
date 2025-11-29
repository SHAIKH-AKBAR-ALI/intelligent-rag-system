# WHAT THIS MODULE DOES:
# - Hugging Face embeddings adapter (cloud-first, local fallback).

import os
import requests
from typing import List, Optional
from src.core.config import settings

HF_INFERENCE_URL = "https://router.huggingface.co/pipeline/feature-extraction/"

_local_st = None
try:
    from sentence_transformers import SentenceTransformer
    _local_st = SentenceTransformer
except Exception:
    _local_st = None

class HuggingFaceEmbeddings:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or settings.huggingface_api_key
        self.model = model or settings.huggingface_embed_model
        self.timeout = timeout
        self._local_model = None
        self._use_local = (self.api_key is None and _local_st is not None)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.api_key:
            return self._embed_via_hf_api(texts)
        if self._use_local or _local_st is not None:
            return self._embed_local(texts)
        raise NotImplementedError("No HF API key and sentence-transformers not installed.")

    def _embed_via_hf_api(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = HF_INFERENCE_URL + self.model
        resp = requests.post(url, headers=headers, json=texts, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and isinstance(data[0], list):
                return [list(map(float, v)) for v in data]
            raise RuntimeError(f"Unexpected HF response shape: {type(data)}")
        else:
            raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        if self._local_model is None:
            self._local_model = _local_st(self.model)
        emb = self._local_model.encode(texts, show_progress_bar=False)
        return [list(map(float, v)) for v in emb]
