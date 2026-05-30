"""
src/evaluation/eval_runner.py
------------------------------
Orchestrates evaluation of a full simulation dataset.

Takes a list of (question, answer, context) rows, runs all judges,
aggregates scores, logs to MLflow, and saves results to disk.

This is the "CI pipeline" component — you run this after any change
to your prompt, model, or chunking strategy to catch regressions.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import pandas as pd
from loguru import logger
from tqdm import tqdm

from configs.config_loader import load_config
from src.evaluation.llm_judge import LLMJudge, JudgeScore

cfg = load_config()


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class EvalRow:
    """Evaluation result for a single QA pair."""
    question: str
    answer: str
    context: str
    scores: dict[str, JudgeScore] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return all(s.passed for s in self.scores.values())

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer[:500],  # truncate for readability
            "context_preview": self.context[:300],
            "passed_all": self.passed_all,
            "avg_score": round(self.avg_score, 4),
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "metadata": self.metadata,
        }


@dataclass
class EvalRun:
    """Results of a complete evaluation run across all rows."""
    run_id: str
    timestamp: str
    rows: list[EvalRow] = field(default_factory=list)
    run_metadata: dict = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def pass_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.passed_all) / len(self.rows)

    def metric_summary(self) -> dict[str, float]:
        """Compute mean score per metric across all rows."""
        if not self.rows:
            return {}

        all_metrics = set()
        for row in self.rows:
            all_metrics.update(row.scores.keys())

        summary = {}
        for metric in all_metrics:
            scores = [
                row.scores[metric].score
                for row in self.rows
                if metric in row.scores
            ]
            if scores:
                summary[f"mean_{metric}"] = round(sum(scores) / len(scores), 4)
                summary[f"pass_rate_{metric}"] = round(
                    sum(1 for s in scores if s >= getattr(
                        cfg.evaluation.pass_thresholds, metric, 0.7
                    )) / len(scores), 4
                )

        summary["overall_pass_rate"] = round(self.pass_rate, 4)
        summary["num_evaluated"] = self.num_rows
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all rows to a flat DataFrame for analysis."""
        records = []
        for row in self.rows:
            record = {
                "question": row.question,
                "answer_preview": row.answer[:200],
                "passed_all": row.passed_all,
                "avg_score": row.avg_score,
                **row.metadata,
            }
            for metric, score in row.scores.items():
                record[f"score_{metric}"] = score.score
                record[f"passed_{metric}"] = score.passed
                record[f"explanation_{metric}"] = score.explanation
            records.append(record)
        return pd.DataFrame(records)

    def failure_cases(self) -> list[EvalRow]:
        """Return rows that failed any metric — useful for debugging."""
        return [r for r in self.rows if not r.passed_all]


# ── Runner ────────────────────────────────────────────────────────


class EvalRunner:
    """
    Runs evaluation on a dataset of QA rows and logs results to MLflow.

    Usage:
        runner = EvalRunner()
        eval_run = runner.run(dataset_rows, run_name="prompt_v2_test")
        print(eval_run.metric_summary())
    """

    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge or LLMJudge()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow experiment."""
        mlflow.set_tracking_uri(str(cfg.abs_path("mlflow_uri")))
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    def evaluate_row(self, row: dict) -> EvalRow:
        """
        Run all judges on a single data row.

        Args:
            row: Dict with keys: question, answer, context (or contexts list)
        """
        question = row["question"]
        answer = row["answer"]

        # Handle both "context" (str) and "contexts" (list) formats
        context = row.get("context", "")
        if not context and "contexts" in row:
            contexts = row["contexts"]
            context = "\n\n---\n\n".join(contexts) if isinstance(contexts, list) else contexts

        # Extract metadata fields (anything that's not q/a/c)
        metadata = {
            k: v for k, v in row.items()
            if k not in ("question", "answer", "context", "contexts", "ground_truth")
        }

        # Run all judges
        scores = self.judge.evaluate_all(
            question=question,
            answer=answer,
            context=context,
        )

        return EvalRow(
            question=question,
            answer=answer,
            context=context,
            scores=scores,
            metadata=metadata,
        )

    def run(
        self,
        dataset: list[dict],
        run_name: str | None = None,
        limit: int | None = None,
        save_results: bool = True,
    ) -> EvalRun:
        """
        Run full evaluation on a dataset.

        Args:
            dataset:      List of row dicts (question, answer, context)
            run_name:     MLflow run name (auto-generated if None)
            limit:        Only evaluate this many rows (for quick checks)
            save_results: Whether to save results to disk

        Returns:
            EvalRun object with all scores and summary stats
        """
        if limit:
            dataset = dataset[:limit]

        run_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
        run_name = run_name or run_id
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(
            f"Starting eval run '{run_name}': "
            f"{len(dataset)} rows, {len(cfg.evaluation.metrics)} metrics"
        )

        eval_run = EvalRun(
            run_id=run_id,
            timestamp=timestamp,
            run_metadata={
                "run_name": run_name,
                "judge_model": cfg.evaluation.judge_model,
                "num_rows": len(dataset),
                "metrics": cfg.evaluation.metrics,
            },
        )

        # ── Evaluate row by row ───────────────────────────────────
        self._setup_mlflow()

        with mlflow.start_run(run_name=run_name) as mlflow_run:
            # Log run parameters
            mlflow.log_params({
                "judge_model": cfg.evaluation.judge_model,
                "num_rows": len(dataset),
                "rag_model": cfg.llm.model,
                "chunk_size": cfg.ingestion.chunk_size,
                "top_k": cfg.rag.top_k,
                "metrics": ",".join(cfg.evaluation.metrics),
            })

            for row in tqdm(dataset, desc=f"Evaluating ({run_name})"):
                try:
                    eval_row = self.evaluate_row(row)
                    eval_run.rows.append(eval_row)
                except Exception as e:
                    logger.error(f"Failed to evaluate row: {e}")
                    continue

            # ── Compute and log summary metrics ───────────────────
            summary = eval_run.metric_summary()
            mlflow.log_metrics(summary)

            # Log failure examples as MLflow artifacts
            failures = eval_run.failure_cases()
            if failures:
                failure_sample = [f.to_dict() for f in failures[:10]]
                mlflow.log_dict(
                    {"failures": failure_sample},
                    "failure_cases.json"
                )

            mlflow_run_id = mlflow_run.info.run_id

        # ── Save results to disk ──────────────────────────────────
        if save_results:
            results_dir = cfg.abs_path("evals_runs") / run_id
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save full results as JSONL
            results_path = results_dir / "results.jsonl"
            with open(results_path, "w") as f:
                for row in eval_run.rows:
                    f.write(json.dumps(row.to_dict()) + "\n")

            # Save summary JSON
            summary_path = results_dir / "summary.json"
            summary_with_meta = {
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": timestamp,
                "mlflow_run_id": mlflow_run_id,
                **summary,
                **eval_run.run_metadata,
            }
            summary_path.write_text(json.dumps(summary_with_meta, indent=2))

            # Save DataFrame as CSV for easy exploration
            df = eval_run.to_dataframe()
            df.to_csv(results_dir / "results.csv", index=False)

            logger.success(f"Results saved → {results_dir}")

        # ── Print summary ─────────────────────────────────────────
        self._print_summary(eval_run, run_name)

        return eval_run

    def _print_summary(self, eval_run: EvalRun, run_name: str) -> None:
        summary = eval_run.metric_summary()
        failures = eval_run.failure_cases()

        print(f"\n── Eval Run: {run_name} ───────────────────────────────")
        print(f"  Rows evaluated   : {eval_run.num_rows}")
        print(f"  Overall pass rate: {eval_run.pass_rate:.1%}")
        print(f"\n  Per-metric scores:")
        for key, val in summary.items():
            if key.startswith("mean_"):
                metric = key[5:]
                mean_val = val
                pass_val = summary.get(f"pass_rate_{metric}", "?")
                print(f"    {metric:<25} mean={mean_val:.3f}  pass_rate={pass_val:.1%}")
        print(f"\n  Failures         : {len(failures)}/{eval_run.num_rows}")
        if failures:
            print(f"\n  Sample failure:")
            f = failures[0]
            print(f"    Q: {f.question[:80]}")
            print(f"    A: {f.answer[:120]}")
            worst = min(f.scores.values(), key=lambda s: s.score)
            print(f"    Lowest score: {worst.metric}={worst.score:.2f} — {worst.explanation}")
        print("──────────────────────────────────────────────────────\n")
