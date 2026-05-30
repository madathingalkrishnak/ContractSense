"""
src/ingestion/document_loader.py
---------------------------------
Loads raw contract files (TXT or PDF) into LangChain Document
objects with rich metadata.

Key design decisions:
- We support both .txt (CUAD) and .pdf (real-world use)
- Metadata is first-class: every chunk tracks its source
- We clean contracts before chunking (remove noise, normalize whitespace)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fitz  # PyMuPDF
from langchain_core.documents import Document
from loguru import logger


# ── Text cleaning ────────────────────────────────────────────────


def clean_contract_text(text: str) -> str:
    """
    Clean raw contract text.

    Problems we fix:
    - Multiple blank lines (common in PDFs) → single newline
    - Page header/footer artifacts (e.g. "Page 1 of 23")
    - Excessive whitespace within lines
    - Null bytes or non-printable characters
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove page number artifacts like "Page 1 of 23" or "- 1 -"
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)

    # Collapse excessive whitespace within lines (keep single spaces)
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    # Collapse 3+ consecutive blank lines into 2
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ── Loaders ──────────────────────────────────────────────────────


def load_txt(file_path: Path, metadata: dict | None = None) -> Document:
    """Load a plain-text contract file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = clean_contract_text(text)

    base_meta = {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_type": "txt",
        "contract_title": file_path.stem,
    }
    if metadata:
        base_meta.update(metadata)

    return Document(page_content=text, metadata=base_meta)


def load_pdf(file_path: Path, metadata: dict | None = None) -> Document:
    """
    Load a PDF contract using PyMuPDF (fitz).

    PyMuPDF is better than pdfplumber for contracts because:
    - Handles complex layouts with tables
    - Faster
    - Better at preserving reading order
    """
    doc = fitz.open(str(file_path))
    pages_text = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")   # plain text extraction
        pages_text.append(page_text)

    doc.close()

    full_text = "\n\n".join(pages_text)
    full_text = clean_contract_text(full_text)

    base_meta = {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_type": "pdf",
        "contract_title": file_path.stem,
        "page_count": len(pages_text),
    }
    if metadata:
        base_meta.update(metadata)

    return Document(page_content=full_text, metadata=base_meta)


def load_contract(file_path: Path | str, metadata: dict | None = None) -> Document:
    """
    Auto-detect file type and load the contract.

    Args:
        file_path: Path to .txt or .pdf contract
        metadata:  Extra metadata to attach (e.g. contract_id, category)

    Returns:
        LangChain Document with full text and metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Contract not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return load_txt(file_path, metadata)
    elif suffix == ".pdf":
        return load_pdf(file_path, metadata)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")


def load_contracts_from_manifest(
    manifest_path: Path | str,
    limit: int | None = None,
) -> Iterator[Document]:
    """
    Generator that yields Documents one by one from a manifest CSV.

    The manifest CSV must have columns:
        - file_path   (relative to project root)
        - contract_id
        - title

    Using a generator keeps memory usage flat regardless of dataset size.

    Args:
        manifest_path: Path to manifest CSV
        limit:         If set, only load this many contracts
    """
    import pandas as pd

    manifest_path = Path(manifest_path)
    root = Path(__file__).parent.parent.parent

    df = pd.read_csv(manifest_path)
    if limit is not None:
        df = df.head(limit)

    logger.info(f"Loading {len(df)} contracts from manifest: {manifest_path.name}")

    for _, row in df.iterrows():
        file_path = root / row["file_path"]
        metadata = {
            "contract_id": str(row["contract_id"]),
            "contract_title": row["title"],
        }

        try:
            doc = load_contract(file_path, metadata)
            yield doc
        except Exception as e:
            logger.warning(f"Skipping {file_path.name}: {e}")
            continue
