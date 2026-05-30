"""
src/rag/chain.py
-----------------
The core RAG (Retrieval Augmented Generation) chain.

RAG flow:
    User question
        │
        ▼ embed_query()
    Query vector
        │
        ▼ ChromaDB.similarity_search()
    Retrieved chunks (context)
        │
        ▼ format_prompt(question + context)
    Prompt
        │
        ▼ Ollama LLM
    Answer

This is what every serious LLM application at DoorDash, Airbnb,
Dropbox etc. builds first — the baseline RAG system that the
evaluation framework then measures and improves.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

from configs.config_loader import load_config
from src.ingestion.vector_store import get_vector_store, similarity_search_with_scores
from src.rag.llm_client import get_chat_llm

cfg = load_config()


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class RAGResponse:
    """
    Structured output from the RAG chain.

    We use a dataclass (not just a dict) because:
    - Type hints help with downstream eval code
    - Easy to serialize to JSON for logging
    - Clear contract for what every RAG response contains
    """
    question: str
    answer: str
    retrieved_chunks: list[Document]
    retrieval_scores: list[float]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """Concatenated text of all retrieved chunks."""
        return "\n\n---\n\n".join(
            [f"[Chunk {i+1}]\n{doc.page_content}"
             for i, doc in enumerate(self.retrieved_chunks)]
        )

    @property
    def source_files(self) -> list[str]:
        """Unique source files that contributed to the answer."""
        return list({
            doc.metadata.get("file_name", "unknown")
            for doc in self.retrieved_chunks
        })

    def to_dict(self) -> dict:
        """Serialize to dict for logging / eval."""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context_text,
            "retrieved_chunks": [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in zip(self.retrieved_chunks, self.retrieval_scores)
            ],
            "source_files": self.source_files,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ── Prompt templates ──────────────────────────────────────────────

# Single-turn prompt: one question + retrieved context → answer
SINGLE_TURN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", cfg.rag.system_prompt),
    ("human", """Here are the relevant contract excerpts:

{context}

---
Question: {question}

Answer based strictly on the excerpts above. If you cannot find the answer, say so."""),
])


# Multi-turn prompt: maintains conversation history
MULTI_TURN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", cfg.rag.system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Here are the relevant contract excerpts:

{context}

---
Question: {question}"""),
])


# ── Context formatter ─────────────────────────────────────────────


def format_context(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Design decisions:
    - Number each chunk so the LLM can reference them ("See Excerpt 2")
    - Include source file name for traceability
    - Separate chunks with --- to prevent blending
    """
    if not chunks:
        return "No relevant contract excerpts found."

    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("file_name", "unknown")
        clause_hint = chunk.metadata.get("contract_title", "")
        header = f"[Excerpt {i+1}] Source: {source}"
        if clause_hint:
            header += f" | {clause_hint}"
        parts.append(f"{header}\n{chunk.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ── Main RAG class ────────────────────────────────────────────────


class ContractRAG:
    """
    The core RAG system for contract question answering.

    Usage:
        rag = ContractRAG()
        response = rag.ask("What is the payment term?")
        print(response.answer)
        print(response.source_files)

    The class handles:
    - Retrieval from ChromaDB
    - Context formatting
    - Prompt construction
    - LLM generation
    - Response packaging with full traceability
    """

    def __init__(
        self,
        contract_filter: dict | None = None,
        model: str | None = None,
        top_k: int | None = None,
    ):
        """
        Args:
            contract_filter: Optional ChromaDB filter to restrict retrieval
                             to specific contracts. E.g. {"contract_id": "5"}
            model:           LLM model override
            top_k:           Number of chunks to retrieve
        """
        self.store = get_vector_store()
        self.llm = get_chat_llm(model=model)
        self.contract_filter = contract_filter
        self.top_k = top_k or cfg.rag.top_k

        # Build the chain: retriever → formatter → prompt → LLM → parser
        self._parser = StrOutputParser()

    def retrieve(self, question: str) -> tuple[list[Document], list[float]]:
        """
        Retrieve the most relevant chunks for a question.

        Returns:
            Tuple of (chunks, scores) where scores are cosine similarities
        """
        results = similarity_search_with_scores(
            query=question,
            top_k=self.top_k,
            filter=self.contract_filter,
            store=self.store,
        )

        if not results:
            return [], []

        chunks, scores = zip(*results)
        return list(chunks), list(scores)

    def ask(
        self,
        question: str,
        chat_history: list | None = None,
    ) -> RAGResponse:
        """
        Ask a question and get a grounded answer from the contracts.

        Args:
            question:     The user's question
            chat_history: List of previous (question, answer) tuples
                          for multi-turn conversation

        Returns:
            RAGResponse with answer, sources, and latency
        """
        t_start = time.time()

        # ── 1. Retrieve relevant chunks ───────────────────────────
        chunks, scores = self.retrieve(question)

        # ── 2. Format context ─────────────────────────────────────
        context = format_context(chunks)

        # ── 3. Build prompt ───────────────────────────────────────
        if chat_history:
            # Multi-turn: include history
            from langchain_core.messages import HumanMessage, AIMessage
            history_messages = []
            for human_msg, ai_msg in chat_history:
                history_messages.append(HumanMessage(content=human_msg))
                history_messages.append(AIMessage(content=ai_msg))

            prompt_value = MULTI_TURN_PROMPT.format_messages(
                context=context,
                question=question,
                chat_history=history_messages,
            )
        else:
            # Single-turn
            prompt_value = SINGLE_TURN_PROMPT.format_messages(
                context=context,
                question=question,
            )

        # ── 4. Generate answer ────────────────────────────────────
        response = self.llm.invoke(prompt_value)
        answer = response.content if hasattr(response, "content") else str(response)

        latency_ms = (time.time() - t_start) * 1000

        logger.debug(
            f"RAG: '{question[:60]}...' → "
            f"{len(chunks)} chunks retrieved, "
            f"{latency_ms:.0f}ms"
        )

        return RAGResponse(
            question=question,
            answer=answer.strip(),
            retrieved_chunks=chunks,
            retrieval_scores=scores,
            latency_ms=latency_ms,
            metadata={
                "model": cfg.llm.model,
                "top_k": self.top_k,
                "contract_filter": self.contract_filter,
            },
        )

    def ask_batch(self, questions: list[str]) -> list[RAGResponse]:
        """
        Ask multiple questions. Used by the eval framework.
        """
        responses = []
        for question in questions:
            try:
                responses.append(self.ask(question))
            except Exception as e:
                logger.error(f"Failed on question '{question[:60]}': {e}")
                # Return empty response so eval doesn't crash
                responses.append(RAGResponse(
                    question=question,
                    answer="[ERROR] Failed to generate answer",
                    retrieved_chunks=[],
                    retrieval_scores=[],
                    latency_ms=0,
                    metadata={"error": str(e)},
                ))
        return responses


# ── Quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.rag.llm_client import assert_ollama_ready

    assert_ollama_ready()

    rag = ContractRAG()
    test_questions = [
        "What is the governing law of this contract?",
        "What are the payment terms?",
        "What happens if either party breaches the contract?",
    ]

    print("\n── ContractRAG Quick Test ──────────────────────────────")
    for q in test_questions:
        resp = rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer[:300]}")
        print(f"   Sources: {resp.source_files}")
        print(f"   Latency: {resp.latency_ms:.0f}ms")
    print("──────────────────────────────────────────────────────\n")
