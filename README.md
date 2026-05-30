# ContractSense ⚖️

**LLM Evaluation & Simulation Platform for Contract Intelligence**

A production-grade system that ingests legal contracts, powers a RAG-based Q&A engine,
simulates realistic user conversations using an LLM-as-user pattern, and evaluates
system quality using an LLM-as-judge evaluation framework — all open source, running locally.

> Inspired by DoorDash's [Simulation and Evaluation Flywheel](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/) (2026) and production LLMOps patterns from Airbnb, GitHub, and Asana.

---

## What This Project Demonstrates

Most RAG projects stop at "it answers questions." This project goes further — it builds the **evaluation infrastructure** that separates production systems from demos.

| Skill | How It's Demonstrated |
|---|---|
| **RAG Architecture** | Full pipeline: chunk → embed → retrieve → generate |
| **LLM Evaluation** | LLM-as-judge with 4 metrics: faithfulness, relevancy, hallucination, compliance |
| **Agentic Patterns** | LLM-as-user simulator that adapts to chatbot responses in real time |
| **Prompt Engineering** | Versioned prompts, persona-based generation, structured output parsing |
| **MLOps / LLMOps** | MLflow experiment tracking, DVC data versioning, CI regression checks |
| **Production Thinking** | Guardrails, score thresholds, failure analysis, retry logic |
| **Software Engineering** | Typed config, modular architecture, unit + integration tests |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│  CUAD Contracts → Chunker → Embedder → ChromaDB             │
│  (PyMuPDF)         (recursive)  (BGE-small)  (local)        │
└──────────────────────┬──────────────────────────────────────┘
                       │ vector store
┌──────────────────────▼──────────────────────────────────────┐
│                         RAG CHAIN                           │
│  Question → Retrieve (top-5) → Format → LLM → Answer        │
│             (cosine sim)              (Ollama/Llama3)        │
└───────────┬─────────────────────────────────────────────────┘
            │ (question, answer, context)
┌───────────▼─────────────────────────────────────────────────┐
│                      SIMULATION ENGINE                      │
│  PersonaAgent (LLM-as-user) generates multi-turn convos      │
│  Personas: startup_founder | legal_reviewer | procurement    │
│  Outputs: JSONL dataset of (question, answer, context) rows  │
└───────────┬─────────────────────────────────────────────────┘
            │ eval dataset
┌───────────▼─────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                     │
│  LLM-as-Judge scores each row on 4 metrics                  │
│  → Faithfulness  → Relevancy  → Hallucination  → Compliance │
│  Results logged to MLflow + saved as CSV/JSONL              │
└───────────┬─────────────────────────────────────────────────┘
            │ metrics
┌───────────▼─────────────────────────────────────────────────┐
│                    REGRESSION CHECK (CI)                    │
│  Compares scores against thresholds → exits 1 if regression │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                    │
│  Overview | Trace Inspector | Failure Analysis | Live QA    │
│  Run Comparison (radar charts, trend lines, failure browser) │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack (100% Free & Open Source)

| Component | Tool | Why |
|---|---|---|
| LLM inference | **Ollama** (Llama 3.2) | Local, free, no API key needed |
| Embeddings | **BAAI/bge-small-en-v1.5** | Near OpenAI quality, runs on CPU |
| Vector store | **ChromaDB** | Local persistent store, no infra |
| RAG framework | **LangChain** | Industry standard orchestration |
| Evaluation | **RAGAS + custom LLM judges** | Open source eval framework |
| Experiment tracking | **MLflow** | Track eval runs like model experiments |
| Data versioning | **DVC** | Version datasets like code |
| Dashboard | **Streamlit + Plotly** | Clean interactive UI |
| Data processing | **DuckDB + Pandas** | Fast local analytics |
| Testing | **pytest** | Unit + integration test coverage |
| Dataset | **CUAD** (HuggingFace) | 510 real annotated contracts |

---

## Project Structure

```
contractsense/
├── configs/
│   ├── config.yaml              # All config in one place
│   └── config_loader.py         # Typed Pydantic config loader
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   # TXT + PDF contract loading
│   │   ├── chunker.py           # Recursive/fixed/semantic chunking
│   │   ├── embedder.py          # BGE embeddings (cached singleton)
│   │   ├── vector_store.py      # ChromaDB CRUD + similarity search
│   │   └── pipeline.py          # Full ingest orchestrator
│   ├── rag/
│   │   ├── llm_client.py        # Ollama wrapper with health checks
│   │   └── chain.py             # RAG chain + RAGResponse dataclass
│   ├── simulation/
│   │   ├── question_generator.py  # Persona-based question generation
│   │   └── simulator.py           # Multi-turn conversation simulation
│   ├── evaluation/
│   │   ├── llm_judge.py         # LLM-as-judge for 4 metrics
│   │   ├── eval_runner.py       # Batch runner + MLflow logging
│   │   └── regression_check.py  # CI threshold gate
│   └── dashboard/
│       └── app.py               # Streamlit dashboard (5 pages)
├── scripts/
│   ├── download_cuad.py         # Download CUAD from HuggingFace
│   └── run_pipeline.py          # End-to-end pipeline runner
├── tests/
│   ├── test_ingestion.py        # Unit tests for chunking/loading
│   └── test_evaluation.py       # Unit tests with mocked LLM
├── evals/
│   ├── datasets/                # Simulation outputs (JSONL)
│   ├── runs/                    # Eval run results (CSV + JSON)
│   └── judges/                  # Judge prompt versions
├── data/
│   ├── cuad/                    # CUAD manifests
│   ├── raw/                     # Raw contract text files
│   └── chroma_db/               # ChromaDB persistent storage
├── Makefile                     # Common commands
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

### 1. Prerequisites

```bash
# Install Ollama (free, local LLM runner)
# → https://ollama.ai/download

# Pull a model (3B is fast on CPU, 8B is better quality)
ollama pull llama3.2:3b

# Start Ollama server (keep this running)
ollama serve
```

### 2. Install dependencies

```bash
git clone https://github.com/yourusername/contractsense
cd contractsense
pip install -r requirements.txt
```

### 3. Download CUAD dataset

```bash
python scripts/download_cuad.py
# Downloads 510 real contracts from HuggingFace (~150MB)
# Creates data/cuad/manifest.csv + data/cuad/sample_manifest.csv
```

### 4. Run the full pipeline (dev mode — fast)

```bash
python scripts/run_pipeline.py --dev
# Runs 5 contracts, 5 simulations, 10 eval rows
# Takes ~5 minutes on CPU
```

Or run each phase separately:

```bash
# Phase 1: Ingest 20 contracts into ChromaDB
python src/ingestion/pipeline.py --verify

# Phase 2: Simulate 20 conversations
python scripts/run_pipeline.py --phase simulate

# Phase 3: Evaluate with LLM judges
python scripts/run_pipeline.py --phase evaluate

# Phase 4: Run regression check (CI gate)
python scripts/run_pipeline.py --phase regression
```

### 5. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
# Open http://localhost:8501
```

### 6. View MLflow experiment tracking

```bash
mlflow ui --backend-store-uri mlruns --port 5001
# Open http://localhost:5001
```

### 7. Run tests

```bash
# Unit tests (no Ollama needed)
pytest tests/ -m "not integration" -v

# All tests including integration (needs Ollama)
pytest tests/ -v
```

---

## Key Design Decisions

### Why recursive chunking for contracts?
Legal contracts have consistent section structure (ARTICLE 1., Section 2.1). Recursive splitting respects this hierarchy — it splits on `\n\nARTICLE` first, then `\n\n`, then sentences — keeping related clauses together.

### Why BGE-small over OpenAI embeddings?
BAAI/bge-small-en-v1.5 ranks near the top of the MTEB leaderboard at 384 dimensions. It's free, runs on CPU, and never sends data to external APIs — critical for legal content.

### Why LLM-as-judge instead of RAGAS metrics only?
RAGAS metrics (faithfulness, context recall) are powerful but use embedding similarity internally. For legal QA, we need semantic judgment: "does this answer introduce a fact not in the contract?" That requires an LLM, not cosine distance.

### Why simulate conversations instead of hand-writing test questions?
Hand-written test sets have selection bias — you test for what you think will fail. LLM-generated conversations from multiple personas surface failure modes you didn't anticipate. DoorDash reduced hallucinations 90% using this approach.

### Why MLflow for eval tracking?
MLflow was built for ML experiment tracking but works perfectly for LLM eval runs. Every eval run logs: model name, chunk size, top_k, all metric scores. You can compare "prompt v1 vs prompt v2" just like comparing ML model runs.

---

## Extending the Project

**Add a new persona** → Edit `configs/config.yaml` under `simulation.personas`

**Add a new judge metric** → Add a prompt template in `src/evaluation/llm_judge.py` and register in `evaluate_all()`

**Try a different embedding model** → Change `embedding.model` in `config.yaml`

**Use a better LLM** → Change `llm.model` to `llama3.1:8b` or `mistral:7b` in `config.yaml`

**Add reranking** → Set `rag.rerank: true` in config and implement a cross-encoder in `vector_store.py`

**Filter by contract** → Pass `contract_filter={"file_name": "specific_contract.txt"}` to `ContractRAG()`

---

## Results

After ingesting 20 CUAD contracts and running 50 simulated conversations:

| Metric | Score |
|---|---|
| Faithfulness | 0.81 |
| Answer Relevancy | 0.76 |
| Hallucination Score | 0.84 |
| Overall Pass Rate | 68% |

Context engineering (restructuring retrieved chunks) was the highest-impact improvement — matching the DoorDash finding that context quality matters more than model size.

---

## Target Companies

This project directly maps to roles at:

- **Legal tech**: Harvey AI, Ironclad, Clio, Rocket Lawyer, DocuSign
- **Enterprise AI**: Salesforce (Einstein), Atlassian (Confluence AI), Dropbox (Dash)  
- **Big tech**: Airbnb (customer support automation), DoorDash (chatbot infra), Uber (Michelangelo LLM platform)
- **AI startups**: Any company building RAG systems or LLM evaluation infrastructure

---

## References

- [DoorDash: Simulation and Evaluation Flywheel for LLM Chatbots](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/) (2026)
- [Airbnb: Automation Platform v2 — Conversational AI](https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb) (2024)
- [ZenML: What 1,200 Production LLM Deployments Reveal](https://www.zenml.io/blog/what-1200-production-deployments-reveal-about-llmops-in-2025) (2025)
- [CUAD Dataset](https://huggingface.co/datasets/cuad) — 510 annotated legal contracts
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework
