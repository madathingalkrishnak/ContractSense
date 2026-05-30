"""
src/ingestion/pipeline.py
--------------------------
Orchestrates the full ingestion pipeline:

    Raw contracts (TXT/PDF)
        │
        ▼ load_contract()
    LangChain Documents
        │
        ▼ chunk_document()
    Chunks (smaller Documents with metadata)
        │
        ▼ get_embeddings()
    Embedding vectors
        │
        ▼ ChromaDB.add_documents()
    Indexed vector store (persisted to disk)

Run this once to set up the knowledge base. After that, you only
re-run if you add new contracts or change the chunking strategy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from tqdm import tqdm

from configs.config_loader import load_config
from src.ingestion.chunker import chunk_document, print_chunk_stats
from src.ingestion.document_loader import load_contract, load_contracts_from_manifest
from src.ingestion.vector_store import add_documents, get_collection_stats, get_vector_store

cfg = load_config()


def run_ingestion_pipeline(
    manifest_path: str | Path | None = None,
    limit: int | None = None,
    strategy: str | None = None,
    reset: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        manifest_path: Path to manifest CSV (defaults to sample_manifest)
        limit:         Only ingest this many contracts (useful for testing)
        strategy:      Chunking strategy override ("recursive" | "fixed")
        reset:         If True, delete existing collection first

    Returns:
        Stats dict with ingestion summary
    """
    start_time = time.time()

    # Default to sample manifest for fast iteration
    if manifest_path is None:
        manifest_path = cfg.abs_path("data_cuad") / "sample_manifest.csv"

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run: python scripts/download_cuad.py first"
        )

    # Optionally reset the collection
    if reset:
        from src.ingestion.vector_store import delete_collection
        logger.warning("Resetting ChromaDB collection...")
        delete_collection()

    # ── Step 1: Load documents ────────────────────────────────────
    logger.info("── Step 1: Loading documents ────────────────────")
    all_docs = list(load_contracts_from_manifest(manifest_path, limit=limit))
    logger.info(f"Loaded {len(all_docs)} contracts")

    # ── Step 2: Chunk documents ───────────────────────────────────
    logger.info("── Step 2: Chunking documents ───────────────────")
    all_chunks = []
    for doc in tqdm(all_docs, desc="Chunking"):
        chunks = chunk_document(doc, strategy=strategy)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    print_chunk_stats(all_chunks)

    # ── Step 3: Embed + store in ChromaDB ────────────────────────
    logger.info("── Step 3: Embedding and indexing ───────────────")
    store = add_documents(all_chunks)

    # ── Step 4: Verify ───────────────────────────────────────────
    stats = get_collection_stats(store)
    elapsed = time.time() - start_time

    # ── Step 5: Save ingestion metadata ──────────────────────────
    metadata_path = cfg.abs_path("data_processed") / "ingestion_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    ingestion_meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_path": str(manifest_path),
        "num_contracts": len(all_docs),
        "num_chunks": len(all_chunks),
        "chunking_strategy": strategy or cfg.ingestion.chunking_strategy,
        "chunk_size": cfg.ingestion.chunk_size,
        "chunk_overlap": cfg.ingestion.chunk_overlap,
        "embedding_model": cfg.embedding.model,
        "collection_name": cfg.chroma.collection_name,
        "elapsed_seconds": round(elapsed, 1),
        "chroma_count": stats["total_chunks"],
    }

    metadata_path.write_text(json.dumps(ingestion_meta, indent=2))
    logger.info(f"Ingestion metadata saved → {metadata_path}")

    # ── Summary ───────────────────────────────────────────────────
    logger.success(
        f"\n── Ingestion Complete ────────────────────────────────\n"
        f"  Contracts ingested : {len(all_docs)}\n"
        f"  Chunks indexed     : {len(all_chunks)}\n"
        f"  Chroma collection  : '{stats['collection_name']}' "
        f"({stats['total_chunks']} vectors)\n"
        f"  Time elapsed       : {elapsed:.1f}s\n"
        f"──────────────────────────────────────────────────────"
    )

    return ingestion_meta


def verify_ingestion(query: str = "What is the termination clause?") -> None:
    """
    Quick sanity check: run a test query against the vector store.
    Call this after ingestion to confirm everything works.
    """
    from src.ingestion.vector_store import similarity_search_with_scores

    logger.info(f"Running verification query: '{query}'")
    results = similarity_search_with_scores(query, top_k=3)

    if not results:
        logger.error("No results returned — ingestion may have failed!")
        return

    print(f"\n── Top {len(results)} results for: '{query}' ─────────────────")
    for i, (doc, score) in enumerate(results):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"     Source: {doc.metadata.get('file_name', 'unknown')}")
        print(f"     Chunk:  {doc.metadata.get('chunk_index', '?')}/{doc.metadata.get('total_chunks', '?')}")
        print(f"     Text:   {doc.page_content[:250].strip()}...")
    print("─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ContractSense ingestion pipeline")
    parser.add_argument("--manifest", type=str, help="Path to manifest CSV")
    parser.add_argument("--limit", type=int, help="Limit number of contracts")
    parser.add_argument("--strategy", type=str, default="recursive", help="Chunking strategy")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB first")
    parser.add_argument("--verify", action="store_true", help="Run verification query after ingestion")
    args = parser.parse_args()

    run_ingestion_pipeline(
        manifest_path=args.manifest,
        limit=args.limit,
        strategy=args.strategy,
        reset=args.reset,
    )

    if args.verify:
        verify_ingestion()
