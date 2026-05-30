# ContractSense Makefile
# ─────────────────────────────────────────────────────────────────
# Usage: make <target>

.PHONY: install setup download ingest simulate evaluate test dashboard clean

# ── Setup ─────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

setup: install
	@echo "Checking Ollama..."
	@which ollama || (echo "Install Ollama from https://ollama.ai" && exit 1)
	@ollama pull llama3.2:3b
	@echo "✓ Setup complete"

# ── Data ──────────────────────────────────────────────────────────

download:
	python scripts/download_cuad.py
	@echo "✓ CUAD dataset downloaded"

# ── Pipeline phases ───────────────────────────────────────────────

ingest:
	python src/ingestion/pipeline.py --verify
	@echo "✓ Ingestion complete"

ingest-reset:
	python src/ingestion/pipeline.py --reset --verify
	@echo "✓ Ingestion reset and complete"

simulate:
	python scripts/run_pipeline.py --phase simulate
	@echo "✓ Simulation complete"

evaluate:
	python scripts/run_pipeline.py --phase evaluate
	@echo "✓ Evaluation complete"

regression:
	python scripts/run_pipeline.py --phase regression

# ── Full runs ─────────────────────────────────────────────────────

dev:
	python scripts/run_pipeline.py --dev
	@echo "✓ Dev pipeline complete"

pipeline:
	python scripts/run_pipeline.py
	@echo "✓ Full pipeline complete"

# ── Tests ─────────────────────────────────────────────────────────

test:
	pytest tests/ -m "not integration" -v

test-all:
	pytest tests/ -v

test-coverage:
	pytest tests/ -m "not integration" --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Dashboard ─────────────────────────────────────────────────────

dashboard:
	streamlit run src/dashboard/app.py

# ── MLflow UI ─────────────────────────────────────────────────────

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5001

# ── Utilities ─────────────────────────────────────────────────────

clean-runs:
	rm -rf evals/runs/* evals/datasets/*
	@echo "✓ Eval runs cleared"

clean-db:
	rm -rf data/chroma_db
	@echo "✓ ChromaDB cleared"

clean: clean-runs clean-db
	@echo "✓ Cleaned"

lint:
	ruff check src/ tests/
	@echo "✓ Lint passed"

format:
	ruff format src/ tests/

# ── Status check ─────────────────────────────────────────────────

status:
	@echo "── ContractSense Status ──────────────────────────────"
	@python -c "from src.ingestion.vector_store import get_vector_store, get_collection_stats; s=get_vector_store(); st=get_collection_stats(s); print(f'  ChromaDB chunks: {st[\"total_chunks\"]}')" 2>/dev/null || echo "  ChromaDB: not initialized"
	@ls evals/datasets/*.jsonl 2>/dev/null | wc -l | xargs -I{} echo "  Simulation datasets: {}"
	@ls evals/runs/ 2>/dev/null | wc -l | xargs -I{} echo "  Eval runs: {}"
	@curl -s http://localhost:11434 > /dev/null 2>&1 && echo "  Ollama: running" || echo "  Ollama: not running"
	@echo "──────────────────────────────────────────────────────"
