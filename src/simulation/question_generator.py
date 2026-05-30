"""
src/simulation/question_generator.py
--------------------------------------
Generates realistic questions about contracts for a given persona.

This is a key part of the simulation system. Instead of hand-writing
test questions, we use an LLM to generate diverse, realistic questions
that different types of users would actually ask.

Why this matters:
- Manual question sets are small and biased toward easy cases
- LLM-generated questions surface failure modes you didn't think of
- Persona-based generation ensures we test different user intents
- This is exactly what DoorDash built: synthetic conversation generation
  from historical transcripts and behavioral patterns
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.config_loader import load_config, PersonaConfig
from src.rag.llm_client import get_chat_llm

cfg = load_config()


# ── Prompts ───────────────────────────────────────────────────────

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are simulating a {persona_name}: {persona_description}.

You are reading a contract and have specific concerns about: {focus_areas}.

Your task: Generate {num_questions} realistic questions you would ask about
this contract. Make them:
1. Specific to the actual contract text provided
2. Varied in complexity (some simple, some requiring inference)
3. Natural — how a real person would ask, not like a lawyer exam
4. Focused on your persona's concerns

Return ONLY a JSON array of question strings. No other text.
Example format: ["question 1", "question 2", "question 3"]"""),
    ("human", """Contract excerpt:
{contract_excerpt}

Generate {num_questions} questions as a JSON array."""),
])

FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are simulating a {persona_name}: {persona_description}.

You just asked a question and received an answer. Generate a natural follow-up
question that a real person would ask based on this exchange.

Return ONLY the follow-up question as plain text. No JSON, no explanation."""),
    ("human", """Previous question: {previous_question}
Answer received: {previous_answer}

Contract context topic: {topic}

Generate one natural follow-up question."""),
])


# ── Question generator ────────────────────────────────────────────


class QuestionGenerator:
    """
    Generates questions for a given persona and contract excerpt.

    This is used by the ConversationSimulator to seed each simulation
    with realistic opening questions.
    """

    def __init__(self, model: str | None = None):
        self.llm = get_chat_llm(model=model)
        self._parser = StrOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate_questions(
        self,
        contract_excerpt: str,
        persona: PersonaConfig,
        num_questions: int = 5,
    ) -> list[str]:
        """
        Generate questions a persona would ask about a contract excerpt.

        Args:
            contract_excerpt: A section of contract text (not the whole contract)
            persona:          Which persona is asking
            num_questions:    How many questions to generate

        Returns:
            List of question strings

        Design note: We use retry because LLM output parsing can fail if
        the model doesn't follow the JSON format exactly. 3 attempts covers
        most cases.
        """
        # Truncate excerpt to avoid token limits
        excerpt = contract_excerpt[:2000]

        prompt_value = QUESTION_GEN_PROMPT.format_messages(
            persona_name=persona.name,
            persona_description=persona.description,
            focus_areas=", ".join(persona.focus),
            contract_excerpt=excerpt,
            num_questions=num_questions,
        )

        raw_output = self.llm.invoke(prompt_value)
        output_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)

        questions = self._parse_questions(output_text, num_questions)
        return questions

    def _parse_questions(self, output_text: str, expected: int) -> list[str]:
        """
        Parse LLM output into a list of question strings.

        We handle common failure modes:
        - Extra text before/after the JSON
        - Single quotes instead of double quotes
        - Questions not in a list format
        """
        output_text = output_text.strip()

        # Try direct JSON parse
        try:
            questions = json.loads(output_text)
            if isinstance(questions, list):
                return [str(q) for q in questions[:expected]]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from text
        import re
        json_match = re.search(r'\[.*?\]', output_text, re.DOTALL)
        if json_match:
            try:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    return [str(q) for q in questions[:expected]]
            except json.JSONDecodeError:
                pass

        # Fallback: split by newlines and clean up
        lines = [
            line.strip().lstrip("0123456789.-) ").strip('"').strip("'")
            for line in output_text.splitlines()
            if line.strip() and "?" in line
        ]

        if lines:
            logger.warning(f"JSON parse failed, extracted {len(lines)} questions from text")
            return lines[:expected]

        # Last resort: return a generic question
        logger.error(f"Could not parse questions from LLM output: {output_text[:200]}")
        return ["What are the key terms of this contract?"]

    def generate_followup(
        self,
        previous_question: str,
        previous_answer: str,
        persona: PersonaConfig,
        topic: str = "contract terms",
    ) -> str:
        """
        Generate a natural follow-up question given the previous exchange.

        This is what makes simulations feel like real conversations
        instead of disconnected Q&A pairs.
        """
        prompt_value = FOLLOWUP_PROMPT.format_messages(
            persona_name=persona.name,
            persona_description=persona.description,
            previous_question=previous_question,
            previous_answer=previous_answer[:500],  # truncate long answers
            topic=topic,
        )

        output = self.llm.invoke(prompt_value)
        followup = output.content if hasattr(output, "content") else str(output)
        return followup.strip().strip('"').strip("'")
