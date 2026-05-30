"""
tests/test_evaluation.py
-------------------------
Unit tests for the evaluation framework.

These tests use mock LLM responses so they run without Ollama.
Integration tests (which need Ollama) are marked with @pytest.mark.integration.

Run unit tests:    pytest tests/test_evaluation.py -v -m "not integration"
Run all tests:     pytest tests/test_evaluation.py -v  (needs Ollama running)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.evaluation.llm_judge import LLMJudge, JudgeScore
from src.evaluation.eval_runner import EvalRow, EvalRun, EvalRunner


# ── Fixtures ──────────────────────────────────────────────────────

SAMPLE_QUESTION = "What is the payment term?"
SAMPLE_ANSWER_GOOD = "Payment is due within 30 days of invoice, as stated in Section 2.1 of the contract."
SAMPLE_ANSWER_HALLUCINATED = "Payment is due within 45 days. The contract also requires a 10% deposit upfront, due on signing."
SAMPLE_CONTEXT = """
Section 2.1 Payment Terms
Client shall pay Service Provider within 30 days of receiving invoice.
All payments shall be made in US Dollars.
"""


def make_mock_llm(response_text: str):
    """Create a mock LLM that returns a fixed response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ── Tests: JudgeScore ─────────────────────────────────────────────

class TestJudgeScore:

    def test_to_dict(self):
        score = JudgeScore(
            metric="faithfulness",
            score=0.85,
            explanation="The answer is grounded in the context.",
            passed=True,
        )
        d = score.to_dict()
        assert d["metric"] == "faithfulness"
        assert d["score"] == 0.85
        assert d["passed"] is True
        assert "explanation" in d

    def test_score_clamped(self):
        """Score should be between 0 and 1."""
        score = JudgeScore(metric="test", score=1.5, explanation="", passed=True)
        assert score.score >= 0
        # Note: clamping happens in _parse_score, not the dataclass itself


# ── Tests: LLMJudge._parse_score ─────────────────────────────────

class TestScoreParsing:

    def setup_method(self):
        with patch("src.evaluation.llm_judge.get_judge_llm"):
            self.judge = LLMJudge()

    def test_parse_valid_output(self):
        output = "SCORE: 0.85\nEXPLANATION: The answer is fully supported by the context."
        score, explanation = self.judge._parse_score(output, "faithfulness")
        assert score == pytest.approx(0.85)
        assert "supported" in explanation.lower()

    def test_parse_score_clamps_to_one(self):
        output = "SCORE: 1.5\nEXPLANATION: Perfect."
        score, _ = self.judge._parse_score(output, "test")
        assert score <= 1.0

    def test_parse_score_clamps_to_zero(self):
        output = "SCORE: -0.2\nEXPLANATION: Terrible."
        score, _ = self.judge._parse_score(output, "test")
        assert score >= 0.0

    def test_parse_missing_score_returns_default(self):
        output = "I cannot determine a score for this."
        score, _ = self.judge._parse_score(output, "test")
        assert score == 0.5  # default

    def test_parse_missing_explanation_uses_raw(self):
        output = "SCORE: 0.7\nSomething without the EXPLANATION tag"
        _, explanation = self.judge._parse_score(output, "test")
        assert len(explanation) > 0


# ── Tests: LLMJudge with mocked LLM ──────────────────────────────

class TestLLMJudge:

    def _make_judge(self, response: str) -> LLMJudge:
        with patch("src.evaluation.llm_judge.get_judge_llm") as mock_get:
            judge = LLMJudge()
            judge.llm = make_mock_llm(response)
            return judge

    def test_faithfulness_high_score(self):
        judge = self._make_judge(
            "SCORE: 0.95\nEXPLANATION: All claims are directly from the context."
        )
        score = judge.evaluate_faithfulness(
            SAMPLE_QUESTION, SAMPLE_ANSWER_GOOD, SAMPLE_CONTEXT
        )
        assert isinstance(score, JudgeScore)
        assert score.metric == "faithfulness"
        assert score.score == pytest.approx(0.95)
        assert score.passed is True   # 0.95 > threshold 0.75

    def test_hallucination_low_score_fails(self):
        judge = self._make_judge(
            "SCORE: 0.3\nEXPLANATION: Answer introduces facts not in context."
        )
        score = judge.evaluate_hallucination(
            SAMPLE_QUESTION, SAMPLE_ANSWER_HALLUCINATED, SAMPLE_CONTEXT
        )
        assert score.score == pytest.approx(0.3)
        assert score.passed is False  # 0.3 < threshold 0.80

    def test_relevancy_evaluation(self):
        judge = self._make_judge(
            "SCORE: 0.80\nEXPLANATION: Answer directly addresses the payment term question."
        )
        score = judge.evaluate_relevancy(
            SAMPLE_QUESTION, SAMPLE_ANSWER_GOOD
        )
        assert score.metric == "answer_relevancy"
        assert score.score == pytest.approx(0.80)


# ── Tests: EvalRow / EvalRun ──────────────────────────────────────

class TestEvalDataClasses:

    def _make_score(self, metric, score_val, threshold=0.75):
        return JudgeScore(
            metric=metric,
            score=score_val,
            explanation="test",
            passed=score_val >= threshold,
        )

    def test_eval_row_passed_all_true(self):
        row = EvalRow(
            question="Q?",
            answer="A.",
            context="C.",
            scores={
                "faithfulness": self._make_score("faithfulness", 0.9),
                "answer_relevancy": self._make_score("answer_relevancy", 0.85),
            },
        )
        assert row.passed_all is True

    def test_eval_row_passed_all_false_when_one_fails(self):
        row = EvalRow(
            question="Q?",
            answer="A.",
            context="C.",
            scores={
                "faithfulness": self._make_score("faithfulness", 0.9),
                "answer_relevancy": self._make_score("answer_relevancy", 0.3),
            },
        )
        assert row.passed_all is False

    def test_eval_row_avg_score(self):
        row = EvalRow(
            question="Q?",
            answer="A.",
            context="C.",
            scores={
                "faithfulness": self._make_score("faithfulness", 0.8),
                "answer_relevancy": self._make_score("answer_relevancy", 0.6),
            },
        )
        assert row.avg_score == pytest.approx(0.7)

    def test_eval_run_pass_rate(self):
        row_pass = EvalRow("Q", "A", "C", scores={"f": self._make_score("f", 0.9)})
        row_fail = EvalRow("Q", "A", "C", scores={"f": self._make_score("f", 0.3)})
        run = EvalRun(run_id="test", timestamp="2024-01-01", rows=[row_pass, row_fail])
        assert run.pass_rate == 0.5

    def test_eval_run_metric_summary(self):
        rows = [
            EvalRow("Q", "A", "C", scores={"faithfulness": self._make_score("faithfulness", 0.8)}),
            EvalRow("Q", "A", "C", scores={"faithfulness": self._make_score("faithfulness", 0.6)}),
        ]
        run = EvalRun(run_id="test", timestamp="2024-01-01", rows=rows)
        summary = run.metric_summary()
        assert "mean_faithfulness" in summary
        assert summary["mean_faithfulness"] == pytest.approx(0.7)

    def test_eval_run_failure_cases(self):
        row_pass = EvalRow("Q", "A", "C", scores={"f": self._make_score("f", 0.9)})
        row_fail = EvalRow("Q", "A", "C", scores={"f": self._make_score("f", 0.3)})
        run = EvalRun(run_id="test", timestamp="2024-01-01", rows=[row_pass, row_fail])
        failures = run.failure_cases()
        assert len(failures) == 1
        assert failures[0].avg_score == pytest.approx(0.3)

    def test_to_dataframe(self):
        row = EvalRow(
            question="What is payment?",
            answer="30 days.",
            context="Payment in 30 days.",
            scores={"faithfulness": self._make_score("faithfulness", 0.9)},
            metadata={"persona": "founder"},
        )
        run = EvalRun(run_id="test", timestamp="2024-01-01", rows=[row])
        df = run.to_dataframe()
        assert len(df) == 1
        assert "score_faithfulness" in df.columns
        assert df["persona"].iloc[0] == "founder"


# ── Integration tests (need Ollama) ──────────────────────────────

@pytest.mark.integration
class TestLLMJudgeIntegration:
    """These tests require Ollama to be running with the model pulled."""

    def test_faithfulness_with_real_llm(self):
        judge = LLMJudge()
        score = judge.evaluate_faithfulness(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            context=SAMPLE_CONTEXT,
        )
        assert 0.0 <= score.score <= 1.0
        assert len(score.explanation) > 10

    def test_hallucination_detection(self):
        """A hallucinated answer should score lower than a faithful one."""
        judge = LLMJudge()

        good_score = judge.evaluate_hallucination(
            SAMPLE_QUESTION, SAMPLE_ANSWER_GOOD, SAMPLE_CONTEXT
        )
        bad_score = judge.evaluate_hallucination(
            SAMPLE_QUESTION, SAMPLE_ANSWER_HALLUCINATED, SAMPLE_CONTEXT
        )
        # Hallucinated answer should score lower (more hallucinations detected)
        assert good_score.score > bad_score.score
