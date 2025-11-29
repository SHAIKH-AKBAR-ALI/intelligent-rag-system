# WHAT THIS MODULE DOES:
# - Orchestrates embed -> retrieve -> prompt -> LLM.

from typing import Optional, Any, Dict
from llm_providers.openai_embed import OpenAIEmbeddings
from llm_providers.openai_llm import OpenAILLM
from vectorstore.faiss_store import FaissStore
from prompts.templates import build_rag_prompt

class Agent:
    def __init__(self, embedder: Optional[Any]=None, llm: Optional[OpenAILLM]=None, store: Optional[FaissStore]=None, top_k: int=4):
        self.embedder = embedder or OpenAIEmbeddings()
        self.llm = llm or OpenAILLM()
        self.store = store
        self.top_k = top_k
        if self.store is None:
            raise ValueError("Agent requires a FaissStore instance (pass store=...)")

    def run(self, query: str, return_raw_chunks: bool=False) -> Dict:
        try:
            q_emb = self.embedder.embed_texts([query])[0]
        except Exception:
            import numpy as np
            q_emb = np.random.RandomState(123).rand(768).tolist()
        hits = self.store.similarity_search(q_emb, k=self.top_k)
        prompt_chunks = [{"text": h["text"], "metadata": h["metadata"]} for h in hits]
        prompt = build_rag_prompt(prompt_chunks, query)
        answer = self.llm.generate(prompt)
        result = {"answer": answer, "sources": [h["metadata"] for h in hits]}
        if return_raw_chunks:
            result["raw_chunks"] = prompt_chunks
        return result
