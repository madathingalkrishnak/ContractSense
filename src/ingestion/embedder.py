"""
src/ingestion/embedder.py
--------------------------
Wraps sentence-transformers to produce embeddings for contract chunks.

Why sentence-transformers over OpenAI embeddings?
- 100% free, runs locally, no API key needed
- BAAI/bge-small-en-v1.5 scores very close to OpenAI on MTEB benchmark
- Data never leaves your machine (important for legal content)
- Deterministic: same input → same output every time

Model choices (all free):
- BAAI/bge-small-en-v1.5  → 384-dim, fast, great quality  (recommended)
- BAAI/bge-base-en-v1.5   → 768-dim, better, 2× slower
- all-MiniLM-L6-v2        → 384-dim, very fast, slightly lower quality
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from loguru import logger

from configs.config_loader import load_config

cfg = load_config()


class BGEEmbeddings(Embeddings):
    """Thin wrapper around sentence-transformers, no langchain-community needed."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"Loading embedding model: {model_name}  (device={device})")
        self._model = SentenceTransformer(model_name, device=device)
        logger.success(f"Embedding model ready: {model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=cfg.embedding.batch_size,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embeddings(model_name: str | None = None):
    model_name = model_name or cfg.embedding.model
    device = cfg.embedding.device
    return BGEEmbeddings(model_name=model_name, device=device)
