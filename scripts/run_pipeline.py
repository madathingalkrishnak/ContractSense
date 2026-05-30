"""
scripts/run_pipeline.py
------------------------
End-to-end pipeline runner.

Runs in sequence:
    1. Ingestion    (load → chunk → embed → store)
    2. Simulation   (generate synthetic conversations)
    3. Evaluation   (score with LLM judges)
    4. Regression   (check against thresholds)

You can run all phases or individual phases using --phase flag.

Usage:
    # Full pipeline
    python scripts/run_pipeline.py

    # Only ingestion
    python scripts/run_pipeline.py --phase ingest

    # Only simulation + eval (assumes ingestion already done)
    python scripts/run_pipeline.py --phase simulate
    python scripts/run_pipeline.py --phase evaluate --dataset evals/datasets/sim_XXXXX.jsonl

    # Fast dev run (5 contracts, 5 conversations, 10 eval rows)
    python scripts/run_pipeline.py --dev
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from configs.config_loader import load_config

cfg = load_config()


def phase_ingest(args) -> None:
    """Run the ingestion pipeline."""
    from src.ingestion.pipeline import run_ingestion_pipeline, verify_ingestion

    logger.info("=" * 55)
    logger.info("PHASE 1: INGESTION")
    logger.info("=" * 55)

    manifest = cfg.abs_path("data_cuad") / (
        "sample_manifest.csv" if not args.full else "manifest.csv"
    )

    if not manifest.exists():
        logger.error(f"Manifest not found: {manifest}")
        logger.error("Run: python scripts/download_cuad.py first")
        sys.exit(1)

    run_ingestion_pipeline(
        manifest_path=manifest,
        limit=args.limit,
        reset=args.reset,
    )

    if args.verify:
        verify_ingestion()


def phase_simulate(args) -> tuple[Path, list]:
    """Run the simulation pipeline."""
    from src.simulation.simulator import ConversationSimulator

    logger.info("=" * 55)
    logger.info("PHASE 2: SIMULATION")
    logger.info("=" * 55)

    simulator = ConversationSimulator()

    num_convs = 5 if args.dev else cfg.simulation.num_conversations
    if args.limit:
        num_convs = args.limit

    conversations = simulator.run_simulation_batch(num_conversations=num_convs)

    # Build eval dataset
    dataset = ConversationSimulator.to_eval_dataset(conversations)

    # Save dataset
    ts = time.strftime("%Y%m%d_%H%M%S")
    dataset_path = cfg.abs_path("evals_datasets") / f"sim_{ts}.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(dataset_path, "w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    logger.success(f"Eval dataset saved: {dataset_path} ({len(dataset)} rows)")
    return dataset_path, dataset


def phase_evaluate(args, dataset_path: Path | None = None, dataset: list | None = None) -> None:
    """Run the evaluation pipeline."""
    import json
    from src.evaluation.eval_runner import EvalRunner

    logger.info("=" * 55)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 55)

    # Load dataset if not passed directly
    if dataset is None:
        if dataset_path is None:
            if args.dataset:
                dataset_path = Path(args.dataset)
            else:
                # Use the most recent sim file
                sim_files = sorted(cfg.abs_path("evals_datasets").glob("sim_*.jsonl"))
                if not sim_files:
                    logger.error("No simulation dataset found. Run simulation phase first.")
                    sys.exit(1)
                dataset_path = sim_files[-1]
                logger.info(f"Using latest dataset: {dataset_path.name}")

        dataset = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))

    limit = 10 if args.dev else args.limit
    runner = EvalRunner()
    ts = time.strftime("%Y%m%d_%H%M%S")
    runner.run(
        dataset=dataset,
        run_name=f"eval_{ts}",
        limit=limit,
    )


def phase_regression(args) -> None:
    """Run regression check."""
    from src.evaluation.regression_check import run_regression_check

    logger.info("=" * 55)
    logger.info("PHASE 4: REGRESSION CHECK")
    logger.info("=" * 55)

    # Find latest eval run
    eval_dirs = sorted(cfg.abs_path("evals_runs").glob("eval_*"))
    if not eval_dirs:
        logger.error("No eval runs found. Run evaluation phase first.")
        sys.exit(1)

    latest_dir = eval_dirs[-1]
    results_path = latest_dir / "results.jsonl"

    success = run_regression_check(
        dataset_path=results_path,
        run_name=f"regression_{time.strftime('%Y%m%d_%H%M%S')}",
        limit=args.limit,
    )

    if not success:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ContractSense Pipeline Runner")
    parser.add_argument(
        "--phase",
        choices=["ingest", "simulate", "evaluate", "regression", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument("--dev", action="store_true", help="Fast dev mode (small data)")
    parser.add_argument("--full", action="store_true", help="Use full CUAD dataset")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB")
    parser.add_argument("--verify", action="store_true", help="Verify ingestion with test query")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows/contracts")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path for eval phase")
    args = parser.parse_args()

    start = time.time()

    if args.phase in ("ingest", "all"):
        phase_ingest(args)

    dataset_path = None
    dataset = None

    if args.phase in ("simulate", "all"):
        dataset_path, dataset = phase_simulate(args)

    if args.phase in ("evaluate", "all"):
        phase_evaluate(args, dataset_path=dataset_path, dataset=dataset)

    if args.phase in ("regression", "all"):
        phase_regression(args)

    elapsed = time.time() - start
    logger.success(f"\nPipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
