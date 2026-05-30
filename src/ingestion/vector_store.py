"""
src/ingestion/vector_store.py
------------------------------
Manages the ChromaDB vector store: create, populate, query, delete.

Why ChromaDB?
- Runs locally, no server needed (embedded mode)
- Supports metadata filtering (filter by contract_id, chunk_index, etc.)
- Persists to disk across sessions
- LangChain has first-class Chroma integration

ChromaDB key concepts:
- Collection: like a database table — we have one per project
- Document: a text chunk + its embedding + metadata
- Query: returns top-k most similar documents by cosine distance
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from configs.config_loader import load_config
from src.ingestion.embedder import get_embeddings

cfg = load_config()


def get_vector_store(
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    Return (or create) the ChromaDB vector store.

    ChromaDB automatically persists to disk when persist_directory is set.
    On subsequent calls it loads from disk — no need to re-embed.

    Args:
        persist_directory: Where to store the Chroma database on disk
        collection_name:   Name of the collection (like a table name)

    Returns:
        LangChain Chroma instance (implements VectorStore interface)
    """
    persist_directory = persist_directory or str(cfg.abs_path("chroma_db"))
    collection_name = collection_name or cfg.chroma.collection_name

    embeddings = get_embeddings()

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": cfg.chroma.distance_metric},
    )

    logger.info(
        f"Vector store: collection='{collection_name}'  "
        f"persist='{persist_directory}'"
    )
    return store


def add_documents(
    documents: list[Document],
    batch_size: int = 100,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """
    Embed and add documents to ChromaDB in batches.

    We batch to avoid memory issues with large document sets.
    ChromaDB deduplicates by ID — if you re-run ingestion with
    the same documents, they won't be added twice (as long as IDs match).

    Args:
        documents:         List of LangChain Documents to add
        batch_size:        How many documents to embed at once
        persist_directory: ChromaDB storage location

    Returns:
        The populated Chroma vector store
    """
    store = get_vector_store(persist_directory)

    # Generate stable IDs from source + chunk_index so re-runs don't duplicate
    def make_id(doc: Document) -> str:
        source = doc.metadata.get("file_name", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        return f"{source}::chunk_{chunk_idx:04d}"

    ids = [make_id(doc) for doc in documents]

    total = len(documents)
    logger.info(f"Adding {total} chunks to ChromaDB in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        store.add_documents(documents=batch_docs, ids=batch_ids)

        pct = min(i + batch_size, total) / total * 100
        logger.info(f"  Indexed {min(i + batch_size, total)}/{total} ({pct:.0f}%)")

    logger.success(f"Done. {total} chunks indexed in ChromaDB.")
    return store


def get_collection_stats(store: Chroma) -> dict:
    """
    Return stats about what's currently in the vector store.

    Useful for verifying ingestion worked correctly.
    """
    count = store._collection.count()
    return {
        "total_chunks": count,
        "collection_name": store._collection.name,
    }


def delete_collection(
    collection_name: str | None = None,
    persist_directory: str | Path | None = None,
) -> None:
    """
    Delete the entire collection. Useful when re-ingesting from scratch.

    WARNING: This deletes all indexed chunks — you'd need to re-run ingestion.
    """
    import chromadb

    persist_directory = persist_directory or str(cfg.abs_path("chroma_db"))
    collection_name = collection_name or cfg.chroma.collection_name

    client = chromadb.PersistentClient(path=str(persist_directory))

    try:
        client.delete_collection(name=collection_name)
        logger.warning(f"Deleted collection: '{collection_name}'")
    except Exception as e:
        logger.error(f"Could not delete collection: {e}")


def similarity_search(
    query: str,
    top_k: int | None = None,
    filter: dict | None = None,
    store: Chroma | None = None,
) -> list[Document]:
    """
    Run a similarity search against the vector store.

    Args:
        query:  The search query
        top_k:  Number of results to return
        filter: ChromaDB metadata filter, e.g. {"contract_id": "42"}
        store:  Existing store (or creates one)

    Returns:
        List of Document chunks sorted by relevance (most relevant first)

    Example:
        results = similarity_search("What is the termination clause?")
        for doc in results:
            print(doc.page_content[:200])
    """
    store = store or get_vector_store()
    top_k = top_k or cfg.rag.top_k

    results = store.similarity_search(query=query, k=top_k, filter=filter)
    return results


def similarity_search_with_scores(
    query: str,
    top_k: int | None = None,
    filter: dict | None = None,
    store: Chroma | None = None,
) -> list[tuple[Document, float]]:
    """
    Like similarity_search but also returns cosine similarity scores.

    Returns:
        List of (Document, score) tuples. Score is 0–1, higher = more similar.
    """
    store = store or get_vector_store()
    top_k = top_k or cfg.rag.top_k

    results = store.similarity_search_with_relevance_scores(
        query=query, k=top_k, filter=filter
    )

    # Filter by score threshold
    threshold = cfg.rag.score_threshold
    filtered = [(doc, score) for doc, score in results if score >= threshold]

    if not filtered:
        logger.warning(
            f"No results above threshold {threshold} for query: '{query[:60]}...'"
        )

    return filtered
