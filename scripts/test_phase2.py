import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.text.chunker import chunk_text_smart
from src.ingestion.fetch_and_clean import fetch_and_clean_url
from src.llm_providers.hf_embed import HuggingFaceEmbeddings
from src.vectorstore.faiss_store import FaissStore

def main():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    data = fetch_and_clean_url(url)
    chunks = chunk_text_smart(data["text"])
    print("Chunks:", len(chunks))
    embedder = HuggingFaceEmbeddings()
    try:
        embeddings = embedder.embed_texts(chunks)
    except Exception as e:
        print("HF embed failed:", e)
        import numpy as np
        embeddings = (np.random.RandomState(42).rand(len(chunks), 768)).tolist()
    dim = len(embeddings[0])
    store = FaissStore(dim=dim)
    store.upsert(embeddings, chunks, [{"source_url": url, "chunk_index": i} for i in range(len(chunks))])
    print("Store size:", len(store.id_map))

if __name__ == "__main__":
    main()
