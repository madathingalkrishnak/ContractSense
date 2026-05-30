"""
tests/test_ingestion.py
------------------------
Unit tests for the ingestion pipeline.

Run: pytest tests/test_ingestion.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from langchain_core.documents import Document

from src.ingestion.document_loader import clean_contract_text, load_contract
from src.ingestion.chunker import chunk_document, get_chunker


# ── Fixtures ──────────────────────────────────────────────────────

SAMPLE_CONTRACT = """
SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of January 1, 2024,
by and between Acme Corp ("Client") and TechVendor Inc ("Service Provider").

ARTICLE 1. SERVICES
1.1 Service Provider shall perform the following services: software development,
    maintenance, and support as described in Exhibit A.

1.2 Service Provider shall commence services on February 1, 2024, and shall
    complete all services by December 31, 2024.

ARTICLE 2. PAYMENT
2.1 Client shall pay Service Provider $10,000 per month, due on the first
    business day of each month.

2.2 Late payments shall accrue interest at 1.5% per month.

ARTICLE 3. TERMINATION
3.1 Either party may terminate this Agreement with 30 days written notice.

3.2 Client may terminate for cause immediately upon written notice if Service
    Provider materially breaches this Agreement.

ARTICLE 4. GOVERNING LAW
4.1 This Agreement shall be governed by the laws of the State of California.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
first written above.
""".strip()


@pytest.fixture
def sample_doc():
    return Document(
        page_content=SAMPLE_CONTRACT,
        metadata={
            "source": "test_contract.txt",
            "file_name": "test_contract.txt",
            "contract_title": "Services Agreement",
        },
    )


@pytest.fixture
def tmp_contract_file(tmp_path):
    """Create a temporary contract file for testing."""
    contract_file = tmp_path / "test_agreement.txt"
    contract_file.write_text(SAMPLE_CONTRACT)
    return contract_file


# ── Tests: document_loader ────────────────────────────────────────

class TestDocumentLoader:

    def test_clean_text_removes_page_numbers(self):
        dirty = "Some text\nPage 1 of 23\nMore text"
        cleaned = clean_contract_text(dirty)
        assert "Page 1 of 23" not in cleaned
        assert "Some text" in cleaned

    def test_clean_text_collapses_blank_lines(self):
        dirty = "Line 1\n\n\n\n\n\nLine 2"
        cleaned = clean_contract_text(dirty)
        # Should have max 2 consecutive blank lines
        assert "\n\n\n\n" not in cleaned

    def test_clean_text_normalizes_whitespace(self):
        dirty = "Word1    Word2    Word3"
        cleaned = clean_contract_text(dirty)
        assert "Word1 Word2 Word3" in cleaned

    def test_load_txt_file(self, tmp_contract_file):
        doc = load_contract(tmp_contract_file)
        assert isinstance(doc, Document)
        assert "Services Agreement" in doc.page_content
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["file_name"] == "test_agreement.txt"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_contract(Path("/nonexistent/file.txt"))

    def test_load_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "contract.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_contract(bad_file)

    def test_metadata_passthrough(self, tmp_contract_file):
        extra_meta = {"contract_id": "42", "category": "SaaS"}
        doc = load_contract(tmp_contract_file, metadata=extra_meta)
        assert doc.metadata["contract_id"] == "42"
        assert doc.metadata["category"] == "SaaS"


# ── Tests: chunker ────────────────────────────────────────────────

class TestChunker:

    def test_recursive_chunking(self, sample_doc):
        chunks = chunk_document(sample_doc, strategy="recursive")
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_fixed_chunking(self, sample_doc):
        chunks = chunk_document(sample_doc, strategy="fixed")
        assert len(chunks) > 0

    def test_chunks_have_metadata(self, sample_doc):
        chunks = chunk_document(sample_doc)
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i

    def test_chunks_inherit_parent_metadata(self, sample_doc):
        chunks = chunk_document(sample_doc)
        for chunk in chunks:
            assert chunk.metadata["file_name"] == "test_contract.txt"
            assert chunk.metadata["contract_title"] == "Services Agreement"

    def test_min_chunk_length_filter(self, sample_doc):
        chunks = chunk_document(sample_doc, chunk_size=50)
        for chunk in chunks:
            # All chunks should be at least min_chunk_length chars
            from configs.config_loader import load_config
            cfg = load_config()
            assert len(chunk.page_content) >= cfg.ingestion.min_chunk_length

    def test_chunk_size_respected(self, sample_doc):
        chunk_size = 200
        chunks = chunk_document(sample_doc, chunk_size=chunk_size)
        # Most chunks should be near the target size (some may be smaller at boundaries)
        oversized = [c for c in chunks if len(c.page_content) > chunk_size * 1.5]
        assert len(oversized) == 0, f"Found {len(oversized)} oversized chunks"

    def test_content_coverage(self, sample_doc):
        """All words in original should appear in at least one chunk."""
        chunks = chunk_document(sample_doc)
        all_chunk_text = " ".join(c.page_content for c in chunks)

        # Key terms that must be preserved
        key_terms = ["Services Agreement", "ARTICLE 1", "PAYMENT", "TERMINATION"]
        for term in key_terms:
            assert term in all_chunk_text, f"Term '{term}' lost during chunking"

    def test_invalid_strategy(self, sample_doc):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_document(sample_doc, strategy="invalid_strategy")
