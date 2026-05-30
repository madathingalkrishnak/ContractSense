"""
src/evaluation/regression_check.py
------------------------------------
CI-style regression checker.

Run this script after any change to:
- The system prompt
- The LLM model
- The chunking strategy
- The embedding model
- The top_k retrieval setting

If any metric drops below its threshold, this script exits with
code 1 (like a failing test), which blocks deployment in a CI pipeline.

This is the "eval CI" pattern described by DoorDash, GitHub, and Asana:
treat LLM quality metrics like unit tests — regressions must not ship.

Usage:
    python src/evaluation/regression_check.py --dataset evals/datasets/sim_latest.jsonl
    python src/evaluation/regression_check.py --dataset evals/datasets/sim_latest.jsonl --limit 20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from loguru import logger

from configs.config_loader import load_config
from src.evaluation.eval_runner import EvalRunner

cfg = load_config()

# Hardcoded baseline thresholds (override via config.yaml)
THRESHOLDS = {
    "mean_faithfulness": cfg.evaluation.pass_thresholds.faithfulness,
    "mean_answer_relevancy": cfg.evaluation.pass_thresholds.answer_relevancy,
    "mean_hallucination_score": cfg.evaluation.pass_thresholds.hallucination_score,
    "overall_pass_rate": 0.60,   # at least 60% of rows must pass all metrics
}


def load_dataset(path: Path) -> list[dict]:
    """Load eval dataset from JSONL."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_regression_check(
    dataset_path: Path,
    run_name: str = "regression_check",
    limit: int | None = None,
) -> bool:
    """
    Run regression check. Returns True if all checks pass.

    Args:
        dataset_path: Path to JSONL dataset
        run_name:     MLflow run name
        limit:        Limit rows for quick checks

    Returns:
        True if all thresholds met, False if any regression detected
    """
    logger.info(f"Running regression check on: {dataset_path}")

    rows = load_dataset(dataset_path)
    if not rows:
        logger.error("Dataset is empty!")
        return False

    logger.info(f"Dataset size: {len(rows)} rows")

    runner = EvalRunner()
    eval_run = runner.run(
        dataset=rows,
        run_name=run_name,
        limit=limit,
        save_results=True,
    )

    summary = eval_run.metric_summary()

    # ── Check each threshold ───────────────────────────────────────
    all_passed = True
    regressions = []

    print("\n── Regression Check Results ─────────────────────────────")

    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric, 0.0)
        passed = actual >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {metric:<35} actual={actual:.3f}  threshold={threshold:.3f}")

        if not passed:
            all_passed = False
            regressions.append({
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "gap": round(threshold - actual, 4),
            })

    print("─────────────────────────────────────────────────────────")

    if all_passed:
        print("\n✓ ALL CHECKS PASSED — safe to deploy\n")
    else:
        print(f"\n✗ REGRESSION DETECTED — {len(regressions)} metric(s) below threshold:")
        for r in regressions:
            print(f"  • {r['metric']}: {r['actual']:.3f} < {r['threshold']:.3f} (gap: {r['gap']:.3f})")
        print("\nDo NOT deploy. Investigate failures and improve the system.\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContractSense regression check")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL eval dataset",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="regression_check",
        help="MLflow run name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to evaluate",
    )
    args = parser.parse_args()

    success = run_regression_check(
        dataset_path=Path(args.dataset),
        run_name=args.run_name,
        limit=args.limit,
    )

    # Exit code 1 = failure (blocks CI pipeline)
    sys.exit(0 if success else 1)
