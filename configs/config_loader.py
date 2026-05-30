"""
configs/config_loader.py
------------------------
Central config loader. Import this everywhere instead of
hardcoding paths or values.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


# ── Project root ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent


# ── Pydantic models for typed config ────────────────────────────


class PathsConfig(BaseModel):
    data_raw: str
    data_processed: str
    data_cuad: str
    evals_datasets: str
    evals_runs: str
    evals_judges: str
    chroma_db: str
    mlflow_uri: str


class IngestionConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    min_chunk_length: int
    max_chunks_per_doc: int


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int
    device: str


class ChromaConfig(BaseModel):
    collection_name: str
    distance_metric: str


class LLMConfig(BaseModel):
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout: int


class RAGConfig(BaseModel):
    top_k: int
    rerank: bool
    score_threshold: float
    system_prompt: str


class PersonaConfig(BaseModel):
    name: str
    description: str
    focus: list[str]


class SimulationConfig(BaseModel):
    num_conversations: int
    turns_per_conversation: int
    personas: list[PersonaConfig]


class EvalThresholds(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    hallucination_score: float


class EvaluationConfig(BaseModel):
    metrics: list[str]
    judge_model: str
    judge_temperature: float
    batch_size: int
    pass_thresholds: EvalThresholds


class MLflowConfig(BaseModel):
    experiment_name: str
    run_tags: dict[str, str]


class AppConfig(BaseModel):
    paths: PathsConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    llm: LLMConfig
    rag: RAGConfig
    simulation: SimulationConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig

    def abs_path(self, key: str) -> Path:
        """Return absolute path for a paths.* key."""
        rel = getattr(self.paths, key)
        return ROOT / rel


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> AppConfig:
    """
    Load and validate config.yaml once, cache the result.

    Usage:
        from configs.config_loader import load_config
        cfg = load_config()
        print(cfg.llm.model)
    """
    if config_path is None:
        config_path = str(ROOT / "configs" / "config.yaml")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(**raw)
