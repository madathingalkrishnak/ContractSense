"""
src/evaluation/llm_judge.py
-----------------------------
Implements LLM-as-Judge evaluation for RAG responses.

LLM-as-Judge is the dominant pattern in production LLMOps:
instead of hand-labeling thousands of outputs, you use a strong LLM
to score responses against a rubric.

We implement judges for:
1. FAITHFULNESS   — Is the answer grounded in retrieved context?
                    (no hallucination)
2. RELEVANCY      — Does the answer address the question asked?
3. GROUNDING      — How much of the answer comes from the context?
4. HALLUCINATION  — Does the answer introduce facts NOT in context?
5. COMPLIANCE     — Does the answer follow the system's rules?
                    (e.g. says "I cannot find this" when appropriate)

Each judge:
- Takes (question, answer, context) as input
- Returns a score (0.0–1.0) and a short explanation
- Uses temperature=0 for deterministic, reproducible judgments

This mirrors exactly what DoorDash, GitHub, and Asana described in
production: LLM-as-judge with a human-calibrated rubric.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.config_loader import load_config
from src.rag.llm_client import get_judge_llm

cfg = load_config()


# ── Data class ────────────────────────────────────────────────────


@dataclass
class JudgeScore:
    """Result from a single LLM judge evaluation."""
    metric: str
    score: float           # 0.0 – 1.0 (higher is better)
    explanation: str
    passed: bool           # did it meet the threshold?
    raw_output: str = ""   # raw LLM output for debugging

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": round(self.score, 4),
            "explanation": self.explanation,
            "passed": self.passed,
        }


# ── Judge prompt templates ────────────────────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator assessing whether an AI answer
is faithful to the provided source material. Faithful means: every claim
in the answer can be directly supported by the context provided.

Score on this scale:
1.0 = Completely faithful, every claim is in the context
0.75 = Mostly faithful, minor embellishments
0.5 = Partially faithful, some claims not in context
0.25 = Mostly unfaithful, many unsupported claims
0.0 = Completely unfaithful, answer contradicts or ignores context

Respond in EXACTLY this format:
SCORE: [number between 0 and 1]
EXPLANATION: [one sentence explaining the score]"""),
    ("human", """CONTEXT (retrieved contract excerpts):
{context}

QUESTION: {question}

ANSWER TO EVALUATE: {answer}

Evaluate faithfulness:"""),
])

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator assessing whether an AI answer
actually addresses the question asked.

Score on this scale:
1.0 = Answer fully addresses the question
0.75 = Answer addresses most of the question
0.5 = Answer partially addresses the question
0.25 = Answer barely addresses the question
0.0 = Answer does not address the question at all

Respond in EXACTLY this format:
SCORE: [number between 0 and 1]
EXPLANATION: [one sentence explaining the score]"""),
    ("human", """QUESTION: {question}

ANSWER TO EVALUATE: {answer}

Evaluate relevancy:"""),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are evaluating whether an AI answer introduces facts not present in the context.

Reply with ONLY one of these three scores:
1.0 = Answer only uses information from the context
0.5 = Answer has minor additions beyond context  
0.0 = Answer clearly invents facts not in context

Respond in EXACTLY this format:
SCORE: [1.0 or 0.5 or 0.0]
EXPLANATION: [one sentence]"""),
    ("human", """CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Score:"""),
])

COMPLIANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are evaluating whether an AI contract analyst follows its
operating rules. The rules are:
1. Must say "I cannot find this in the contract" when information is not available
2. Must cite specific clauses when possible
3. Must not speculate beyond the contract text
4. Must be precise about legal terms

Score on this scale:
1.0 = Fully compliant with all rules
0.75 = Minor compliance issue
0.5 = Noticeable compliance issue
0.25 = Major compliance issue
0.0 = Completely non-compliant

Respond in EXACTLY this format:
SCORE: [number between 0 and 1]
EXPLANATION: [one sentence explaining compliance issues if any]"""),
    ("human", """CONTEXT: {context}

QUESTION: {question}

ANSWER TO EVALUATE: {answer}

Evaluate compliance:"""),
])


# ── Judge class ───────────────────────────────────────────────────


class LLMJudge:
    """
    Runs LLM-as-Judge evaluations on RAG responses.

    Usage:
        judge = LLMJudge()
        score = judge.evaluate_faithfulness(
            question="What is the payment term?",
            answer="Payment is due in 30 days.",
            context="The contract states payment shall be made within 30 days..."
        )
        print(score.score, score.explanation)
    """

    def __init__(self, model: str | None = None):
        self.llm = get_judge_llm(model=model)

    def _parse_score(self, output: str, metric: str) -> tuple[float, str]:
        """
        Parse the SCORE and EXPLANATION from judge output.

        Returns:
            (score: float, explanation: str)
        """
        output = output.strip()

        # Extract score
        score_match = re.search(r"SCORE:\s*([0-9.]+)", output, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse score from: {output[:100]}")
                score = 0.5
        else:
            logger.warning(f"No SCORE found in {metric} judge output: {output[:100]}")
            score = 0.5

        # Extract explanation
        explanation_match = re.search(
            r"EXPLANATION:\s*(.+?)(?:\n|$)", output, re.IGNORECASE | re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = output[:200]  # fallback: use raw output

        return score, explanation

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _run_judge(
        self,
        prompt_template: ChatPromptTemplate,
        **kwargs,
    ) -> tuple[str, str]:
        """Run a judge prompt and return (raw_output, content)."""
        prompt_value = prompt_template.format_messages(**kwargs)
        response = self.llm.invoke(prompt_value)
        content = response.content if hasattr(response, "content") else str(response)
        return content, content

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeScore:
        """Is the answer grounded in the retrieved context?"""
        raw, content = self._run_judge(
            FAITHFULNESS_PROMPT,
            question=question,
            answer=answer,
            context=context[:3000],  # truncate to avoid token limits
        )
        score, explanation = self._parse_score(content, "faithfulness")
        threshold = cfg.evaluation.pass_thresholds.faithfulness

        return JudgeScore(
            metric="faithfulness",
            score=score,
            explanation=explanation,
            passed=score >= threshold,
            raw_output=raw,
        )

    def evaluate_relevancy(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> JudgeScore:
        """Does the answer address the question?"""
        raw, content = self._run_judge(
            RELEVANCY_PROMPT,
            question=question,
            answer=answer,
        )
        score, explanation = self._parse_score(content, "relevancy")
        threshold = cfg.evaluation.pass_thresholds.answer_relevancy

        return JudgeScore(
            metric="answer_relevancy",
            score=score,
            explanation=explanation,
            passed=score >= threshold,
            raw_output=raw,
        )

    def evaluate_hallucination(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeScore:
        """Does the answer introduce facts not in the context?"""
        raw, content = self._run_judge(
            HALLUCINATION_PROMPT,
            question=question,
            answer=answer,
            context=context[:3000],
        )
        score, explanation = self._parse_score(content, "hallucination")
        threshold = cfg.evaluation.pass_thresholds.hallucination_score

        return JudgeScore(
            metric="hallucination_score",
            score=score,
            explanation=explanation,
            passed=score >= threshold,
            raw_output=raw,
        )

    def evaluate_compliance(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeScore:
        """Does the answer follow the system's operating rules?"""
        raw, content = self._run_judge(
            COMPLIANCE_PROMPT,
            question=question,
            answer=answer,
            context=context[:3000],
        )
        score, explanation = self._parse_score(content, "compliance")

        return JudgeScore(
            metric="compliance",
            score=score,
            explanation=explanation,
            passed=score >= 0.75,
            raw_output=raw,
        )

    def evaluate_all(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict[str, JudgeScore]:
        """
        Run all judge metrics on a single QA pair.

        Returns:
            Dict mapping metric_name → JudgeScore
        """
        metrics_to_run = cfg.evaluation.metrics

        results = {}

        if "faithfulness" in metrics_to_run:
            results["faithfulness"] = self.evaluate_faithfulness(question, answer, context)

        if "answer_relevancy" in metrics_to_run:
            results["answer_relevancy"] = self.evaluate_relevancy(question, answer, context)

        if "hallucination_score" in metrics_to_run:
            results["hallucination_score"] = self.evaluate_hallucination(question, answer, context)

        if "compliance" in metrics_to_run:
            results["compliance"] = self.evaluate_compliance(question, answer, context)

        return results
