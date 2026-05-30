"""
src/rag/llm_client.py
----------------------
Wraps Ollama for local LLM inference.

Ollama runs open-source models (Llama, Mistral, etc.) locally.
No API key, no cost, no data leaving your machine.

Setup (one-time):
    1. Install Ollama: https://ollama.ai (free)
    2. Pull a model:   ollama pull llama3.2:3b
    3. Start server:   ollama serve  (runs on http://localhost:11434)

Model recommendations:
    - llama3.2:3b   → Fast, 4GB RAM, good for dev/eval
    - llama3.1:8b   → Better quality, 8GB RAM, good for production
    - mistral:7b    → Alternative, good instruction following
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.language_models import BaseLanguageModel
from loguru import logger

from configs.config_loader import load_config

cfg = load_config()


def check_ollama_running() -> bool:
    """Check if Ollama server is running. Returns True if reachable."""
    try:
        response = httpx.get(cfg.llm.base_url, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def check_model_available(model_name: str | None = None) -> bool:
    """Check if the specified model is pulled in Ollama."""
    model_name = model_name or cfg.llm.model
    try:
        response = httpx.get(f"{cfg.llm.base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            # Ollama model names may or may not include the tag
            return any(model_name in m for m in models)
    except Exception:
        pass
    return False


def assert_ollama_ready(model_name: str | None = None) -> None:
    """
    Assert Ollama is running and the model is available.
    Raises RuntimeError with helpful messages if not.
    """
    model_name = model_name or cfg.llm.model

    if not check_ollama_running():
        raise RuntimeError(
            f"Ollama is not running at {cfg.llm.base_url}\n"
            f"Start it with: ollama serve"
        )

    if not check_model_available(model_name):
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama.\n"
            f"Pull it with: ollama pull {model_name}"
        )

    logger.info(f"Ollama ready ✓  model={model_name}")


@lru_cache(maxsize=4)
def get_chat_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOllama:
    """
    Return a cached ChatOllama instance (chat-style LLM).

    ChatOllama uses the messages API (system/human/ai messages).
    This is what we use for the main RAG chain and simulation.

    Args:
        model:       Model name (e.g. "llama3.2:3b")
        temperature: Sampling temperature. 0 = deterministic.

    Returns:
        ChatOllama instance (implements BaseChatModel interface)
    """
    model = model or cfg.llm.model
    temperature = temperature if temperature is not None else cfg.llm.temperature

    llm = ChatOllama(
        model=model,
        base_url=cfg.llm.base_url,
        temperature=temperature,
        num_predict=cfg.llm.max_tokens,
        timeout=cfg.llm.timeout,
    )

    logger.info(f"LLM ready: {model} (temp={temperature})")
    return llm


@lru_cache(maxsize=4)
def get_judge_llm(
    model: str | None = None,
) -> ChatOllama:
    """
    Return a cached ChatOllama for the LLM-as-Judge evaluator.

    The judge uses temperature=0 for deterministic, reproducible scores.
    """
    model = model or cfg.evaluation.judge_model
    return get_chat_llm(model=model, temperature=0.0)
