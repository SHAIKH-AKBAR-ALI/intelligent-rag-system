# WHAT THIS MODULE DOES:
# - Loads .env and centralizes configuration for the project.

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    @property
    def openai_api_key(self):
        return os.getenv("OPENAI_API_KEY")

    @property
    def openai_model(self):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def openai_embed_model(self):
        return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    @property
    def groq_api_key(self):
        return os.getenv("GROQ_API_KEY")

    @property
    def groq_model(self):
        return os.getenv("GROQ_MODEL", "llama3-8b-8192")

    @property
    def gemini_api_key(self):
        return os.getenv("GEMINI_API_KEY")

    @property
    def gemini_model(self):
        return os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    @property
    def huggingface_api_key(self):
        return os.getenv("HUGGINGFACE_API_KEY")

    @property
    def huggingface_embed_model(self):
        return os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")

    @property
    def chunk_size(self):
        return int(os.getenv("CHUNK_SIZE", "1000"))

    @property
    def chunk_overlap(self):
        return int(os.getenv("CHUNK_OVERLAP", "200"))

    @property
    def faiss_index_path(self):
        return os.getenv("FAISS_INDEX_PATH", "./vectorstore/faiss.index")

    @property
    def faiss_meta_path(self):
        return os.getenv("FAISS_META_PATH", "./vectorstore/faiss_meta.json")

settings = Settings()
