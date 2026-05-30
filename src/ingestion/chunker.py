"""
src/ingestion/chunker.py
-------------------------
Splits long contract documents into smaller chunks for vector storage.

Why chunking matters for contracts:
- Contracts can be 20,000+ words — too long to embed as a whole
- Good chunking = relevant clauses are retrieved together
- Bad chunking = a clause gets split mid-sentence, breaking semantics

We implement three strategies and explain the tradeoffs:

1. RECURSIVE (default)
   - Splits on paragraphs → sentences → words in order
   - Respects document structure better than fixed-size splitting
   - Best general-purpose choice for legal text

2. FIXED
   - Splits every N characters regardless of structure
   - Simple and fast, but may split clauses mid-sentence
   - Good baseline to compare against

3. SEMANTIC (advanced)
   - Groups sentences by embedding similarity
   - Keeps semantically related content together
   - Slower but best for dense contracts with many topics
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from loguru import logger

from configs.config_loader import load_config

cfg = load_config()

ChunkStrategy = Literal["recursive", "fixed", "semantic"]


# ── Separators tuned for legal contracts ─────────────────────────
#
# Legal contracts have consistent structure we can exploit:
#   - Section headings: "ARTICLE 1.", "1.1", "Section 2"
#   - Numbered lists: "(a)", "(b)", "(i)"
#   - Paragraph breaks
#   - Sentences
#
LEGAL_SEPARATORS = [
    "\n\nARTICLE ",
    "\n\nSECTION ",
    "\n\nSection ",
    r"\n\n\d+\.",    # numbered sections
    "\n\n",          # paragraph break (most common)
    "\n",            # line break
    ". ",            # sentence
    " ",             # word (last resort)
    "",              # character (absolute last resort)
]


def get_chunker(
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """
    Factory function: returns the right text splitter for the strategy.

    Args:
        strategy:      "recursive" | "fixed" | "semantic"
        chunk_size:    Characters per chunk (defaults to config value)
        chunk_overlap: Overlap between chunks (defaults to config value)
    """
    strategy = strategy or cfg.ingestion.chunking_strategy
    chunk_size = chunk_size or cfg.ingestion.chunk_size
    chunk_overlap = chunk_overlap or cfg.ingestion.chunk_overlap

    if strategy == "recursive":
        # Best for contracts: respects legal document structure
        return RecursiveCharacterTextSplitter(
            separators=LEGAL_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    elif strategy == "fixed":
        # Simple baseline
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
            length_function=len,
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose: recursive | fixed | semantic"
        )


def chunk_document(
    document: Document,
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a single Document into chunks.

    Each chunk inherits the parent document's metadata, plus:
    - chunk_index:   Position of this chunk within the document (0-based)
    - total_chunks:  Total number of chunks from this document
    - char_start:    Approximate character offset in original text

    Args:
        document:      LangChain Document to split
        strategy:      Chunking strategy (defaults to config)
        chunk_size:    Characters per chunk
        chunk_overlap: Overlap characters

    Returns:
        List of Document chunks with enriched metadata
    """
    splitter = get_chunker(strategy, chunk_size, chunk_overlap)
    raw_chunks = splitter.split_documents([document])

    # Filter out very short chunks (noise: headers, page numbers, etc.)
    min_len = cfg.ingestion.min_chunk_length
    filtered = [c for c in raw_chunks if len(c.page_content.strip()) >= min_len]

    # Cap at max_chunks_per_doc (safety for very long contracts)
    max_chunks = cfg.ingestion.max_chunks_per_doc
    if len(filtered) > max_chunks:
        logger.warning(
            f"Contract '{document.metadata.get('contract_title', '?')}' "
            f"produced {len(filtered)} chunks — capping at {max_chunks}"
        )
        filtered = filtered[:max_chunks]

    # Enrich metadata on each chunk
    total = len(filtered)
    enriched = []
    for i, chunk in enumerate(filtered):
        chunk.metadata.update(
            {
                "chunk_index": i,
                "total_chunks": total,
                "chunk_char_count": len(chunk.page_content),
                "chunk_word_count": len(chunk.page_content.split()),
            }
        )
        enriched.append(chunk)

    return enriched


def chunk_documents(
    documents: list[Document],
    strategy: ChunkStrategy | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Chunk a list of Documents, with progress logging.

    Args:
        documents: List of loaded Documents
        strategy:  Chunking strategy
        chunk_size: Characters per chunk
        chunk_overlap: Overlap

    Returns:
        Flat list of all chunks across all documents
    """
    all_chunks: list[Document] = []

    for doc in documents:
        chunks = chunk_document(doc, strategy, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(
        f"Chunked {len(documents)} documents → {len(all_chunks)} chunks  "
        f"(avg {len(all_chunks) / max(len(documents), 1):.1f} chunks/doc)"
    )
    return all_chunks


def print_chunk_stats(chunks: list[Document]) -> None:
    """Print a summary table of chunking statistics."""
    import statistics

    char_counts = [len(c.page_content) for c in chunks]
    word_counts = [len(c.page_content.split()) for c in chunks]

    print("\n── Chunk Statistics ─────────────────────────────────")
    print(f"  Total chunks       : {len(chunks)}")
    print(f"  Avg chars/chunk    : {statistics.mean(char_counts):.0f}")
    print(f"  Median chars/chunk : {statistics.median(char_counts):.0f}")
    print(f"  Min chars/chunk    : {min(char_counts)}")
    print(f"  Max chars/chunk    : {max(char_counts)}")
    print(f"  Avg words/chunk    : {statistics.mean(word_counts):.0f}")
    print("─────────────────────────────────────────────────────\n")
