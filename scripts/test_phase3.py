import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.fetch_and_clean import fetch_and_clean_url
from src.text.chunker import chunk_text_smart
from src.llm_providers.hf_embed import HuggingFaceEmbeddings
from src.vectorstore.faiss_store import FaissStore
from src.agents.agent_builder import Agent

def main():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    data = fetch_and_clean_url(url)
    chunks = chunk_text_smart(data["text"])
    print("Chunks:", len(chunks))
    embedder = HuggingFaceEmbeddings()
    try:
        embeddings = embedder.embed_texts(chunks)
    except Exception:
        import numpy as np
        embeddings = (np.random.RandomState(42).rand(len(chunks), 768)).tolist()
    dim = len(embeddings[0])
    store = FaissStore(dim=dim)
    store.upsert(embeddings, chunks, [{"source_url": url, "chunk_index": i} for i in range(len(chunks))])
    agent = Agent(embedder=embedder, store=store)
    res = agent.run("What is artificial intelligence?")
    print("ANSWER PREVIEW:", res["answer"][:500])
    print("SOURCES:", res["sources"])

if __name__ == "__main__":
    main()
