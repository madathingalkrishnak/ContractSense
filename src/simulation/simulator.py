"""
src/simulation/simulator.py
-----------------------------
Generates multi-turn synthetic conversations between a persona (LLM-as-user)
and the RAG system (our chatbot).

This is the core innovation from DoorDash's engineering blog:
instead of testing with static question lists, we simulate realistic
conversations where the "user" adapts to the chatbot's responses —
including asking follow-ups, expressing confusion, and pushing back.

Architecture:
    PersonaAgent (LLM-as-user)
         │ generates questions
         ▼
    ContractRAG (our system under test)
         │ generates answers
         ▼
    PersonaAgent (evaluates answer, generates follow-up)
         │
         ▼ [repeat for N turns]
    Conversation transcript → EvalDataset
"""
from __future__ import annotations

import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm

from configs.config_loader import load_config, PersonaConfig
from src.rag.chain import ContractRAG, RAGResponse
from src.simulation.question_generator import QuestionGenerator

cfg = load_config()


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class ConversationTurn:
    """One turn in a simulated conversation (question + answer + metadata)."""
    turn_index: int
    question: str
    answer: str
    retrieved_chunks: list[Document]
    retrieval_scores: list[float]
    latency_ms: float
    is_followup: bool = False

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "question": self.question,
            "answer": self.answer,
            "context": "\n\n---\n\n".join(
                [c.page_content for c in self.retrieved_chunks]
            ),
            "source_files": list({
                c.metadata.get("file_name", "unknown")
                for c in self.retrieved_chunks
            }),
            "retrieval_scores": self.retrieval_scores,
            "latency_ms": self.latency_ms,
            "is_followup": self.is_followup,
        }


@dataclass
class Conversation:
    """A complete simulated conversation (multiple turns)."""
    conversation_id: str
    persona_name: str
    contract_title: str
    contract_file: str
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "persona_name": self.persona_name,
            "contract_title": self.contract_title,
            "contract_file": self.contract_file,
            "num_turns": self.num_turns,
            "total_latency_ms": self.total_latency_ms,
            "turns": [t.to_dict() for t in self.turns],
            "metadata": self.metadata,
        }

    def to_eval_format(self) -> list[dict]:
        """
        Convert to RAGAS-compatible format.

        RAGAS expects a list of dicts with:
        - question: str
        - answer: str
        - contexts: list[str]  ← retrieved chunk texts
        - ground_truth: str    ← we leave this blank (no labels)
        """
        rows = []
        for turn in self.turns:
            rows.append({
                "question": turn.question,
                "answer": turn.answer,
                "contexts": [c.page_content for c in turn.retrieved_chunks],
                "ground_truth": "",  # no ground truth labels in unsupervised eval
                "conversation_id": self.conversation_id,
                "turn_index": turn.turn_index,
                "persona": self.persona_name,
                "contract": self.contract_title,
                "latency_ms": turn.latency_ms,
                "is_followup": turn.is_followup,
            })
        return rows


# ── Simulator ─────────────────────────────────────────────────────


class ConversationSimulator:
    """
    Simulates realistic multi-turn conversations between a persona
    (LLM-as-user) and the ContractRAG system.

    Each conversation:
    1. Picks a random contract from the vector store
    2. Picks a random persona from config
    3. Generates an opening question using QuestionGenerator
    4. Runs N turns: RAG answers → persona generates follow-up
    5. Records everything as a Conversation object

    The resulting conversations become the eval dataset.
    """

    def __init__(
        self,
        rag: ContractRAG | None = None,
        question_gen: QuestionGenerator | None = None,
    ):
        self.rag = rag or ContractRAG()
        self.question_gen = question_gen or QuestionGenerator()

    def _get_contract_excerpt(self, contract_file: str) -> str:
        """Get a random excerpt from a contract for question generation."""
        # Search for chunks from this specific contract
        chunks = self.rag.store.similarity_search(
            "contract terms conditions obligations parties",
            k=10,
            filter={"file_name": contract_file},
        )

        if not chunks:
            # Fallback: get any chunks
            chunks = self.rag.store.similarity_search(
                "contract terms", k=5
            )

        # Return a random chunk as the excerpt seed
        if chunks:
            random_chunk = random.choice(chunks)
            return random_chunk.page_content
        return ""

    def _get_available_contracts(self) -> list[dict]:
        """Get list of unique contracts in the vector store."""
        # Query a few random results to discover what contracts exist
        results = self.rag.store.similarity_search(
            "agreement contract", k=50
        )

        seen = {}
        for doc in results:
            fname = doc.metadata.get("file_name", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "file_name": fname,
                    "contract_title": doc.metadata.get("contract_title", fname),
                }

        return list(seen.values())

    def simulate_conversation(
        self,
        persona: PersonaConfig | None = None,
        contract_info: dict | None = None,
        num_turns: int | None = None,
    ) -> Conversation:
        """
        Run a single simulated conversation.

        Args:
            persona:       Which persona to simulate (random if None)
            contract_info: Which contract to use (random if None)
            num_turns:     Number of back-and-forth turns

        Returns:
            A Conversation object with all turns recorded
        """
        num_turns = num_turns or cfg.simulation.turns_per_conversation

        # Pick random persona if not specified
        if persona is None:
            persona = random.choice(cfg.simulation.personas)

        # Pick random contract if not specified
        if contract_info is None:
            available = self._get_available_contracts()
            if not available:
                raise RuntimeError("No contracts found in vector store. Run ingestion first.")
            contract_info = random.choice(available)

        conv_id = str(uuid.uuid4())[:8]
        contract_file = contract_info["file_name"]
        contract_title = contract_info["contract_title"]

        logger.debug(
            f"Simulating conversation {conv_id} | "
            f"persona={persona.name} | "
            f"contract={contract_file[:40]}"
        )

        conversation = Conversation(
            conversation_id=conv_id,
            persona_name=persona.name,
            contract_title=contract_title,
            contract_file=contract_file,
            metadata={
                "persona_description": persona.description,
                "persona_focus": persona.focus,
                "num_turns_requested": num_turns,
            },
        )

        # Restrict RAG to this specific contract for focused simulation
        contract_rag = ContractRAG(
            contract_filter={"file_name": contract_file}
        )

        chat_history = []

        for turn_idx in range(num_turns):
            try:
                # ── Generate question ────────────────────────────
                if turn_idx == 0:
                    # First turn: generate opening question from contract excerpt
                    excerpt = self._get_contract_excerpt(contract_file)
                    questions = self.question_gen.generate_questions(
                        contract_excerpt=excerpt,
                        persona=persona,
                        num_questions=1,
                    )
                    question = questions[0] if questions else "What are the main terms of this contract?"
                    is_followup = False
                else:
                    # Subsequent turns: generate follow-up based on previous exchange
                    prev_turn = conversation.turns[-1]
                    question = self.question_gen.generate_followup(
                        previous_question=prev_turn.question,
                        previous_answer=prev_turn.answer,
                        persona=persona,
                        topic=", ".join(persona.focus),
                    )
                    is_followup = True

                # ── Get RAG answer ───────────────────────────────
                rag_response: RAGResponse = contract_rag.ask(
                    question=question,
                    chat_history=chat_history if turn_idx > 0 else None,
                )

                # ── Record turn ──────────────────────────────────
                turn = ConversationTurn(
                    turn_index=turn_idx,
                    question=question,
                    answer=rag_response.answer,
                    retrieved_chunks=rag_response.retrieved_chunks,
                    retrieval_scores=rag_response.retrieval_scores,
                    latency_ms=rag_response.latency_ms,
                    is_followup=is_followup,
                )
                conversation.turns.append(turn)

                # Update history for next turn
                chat_history.append((question, rag_response.answer))

            except Exception as e:
                logger.error(f"Turn {turn_idx} failed: {e}")
                break

        logger.debug(
            f"Conversation {conv_id} complete: "
            f"{len(conversation.turns)}/{num_turns} turns, "
            f"{conversation.total_latency_ms:.0f}ms total"
        )

        return conversation

    def run_simulation_batch(
        self,
        num_conversations: int | None = None,
        output_path: Path | None = None,
    ) -> list[Conversation]:
        """
        Run a full batch of simulated conversations.

        This is the main entry point — generates the full eval dataset.

        Args:
            num_conversations: How many conversations to simulate
            output_path:       Where to save the dataset as JSONL

        Returns:
            List of Conversation objects
        """
        num_conversations = num_conversations or cfg.simulation.num_conversations

        if output_path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = cfg.abs_path("evals_datasets") / f"sim_{ts}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting simulation: {num_conversations} conversations")
        conversations = []

        with open(output_path, "w") as f:
            for i in tqdm(range(num_conversations), desc="Simulating conversations"):
                try:
                    conv = self.simulate_conversation()
                    conversations.append(conv)

                    # Write to JSONL immediately (don't wait for all to finish)
                    f.write(json.dumps(conv.to_dict()) + "\n")
                    f.flush()

                except Exception as e:
                    logger.error(f"Conversation {i} failed: {e}")
                    continue

        logger.success(
            f"Simulation complete: {len(conversations)} conversations saved → {output_path}"
        )
        return conversations

    @staticmethod
    def load_conversations(path: Path) -> list[Conversation]:
        """Load conversations from a JSONL file."""
        conversations = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                # Reconstruct Conversation (without Document objects — just data)
                conv = Conversation(
                    conversation_id=data["conversation_id"],
                    persona_name=data["persona_name"],
                    contract_title=data["contract_title"],
                    contract_file=data["contract_file"],
                    metadata=data.get("metadata", {}),
                )
                conversations.append(conv)
        return conversations

    @staticmethod
    def to_eval_dataset(conversations: list[Conversation]) -> list[dict]:
        """
        Flatten all conversations into a flat list of eval rows.
        Each row is one (question, answer, contexts) triple.
        """
        rows = []
        for conv in conversations:
            rows.extend(conv.to_eval_format())
        return rows
