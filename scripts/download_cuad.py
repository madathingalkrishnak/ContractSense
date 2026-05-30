#!/usr/bin/env python3
"""
scripts/download_cuad.py
Parses the manually downloaded CUAD JSON into contract .txt files.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

RAW_DIR   = ROOT / "data" / "raw"
CUAD_DIR  = ROOT / "data" / "cuad"
JSON_PATH = CUAD_DIR / "CUADv1.json"
SAMPLE_SIZE = 20


def download_cuad() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CUAD_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.exists():
        logger.error(
            f"File not found: {JSON_PATH}\n"
            f"Download it from:\n"
            f"  https://github.com/TheAtticusProject/cuad/raw/main/data/train_separate_questions.json\n"
            f"Save it as: data/cuad/CUADv1.json"
        )
        sys.exit(1)

    logger.info(f"Parsing {JSON_PATH}...")
    with open(JSON_PATH, encoding="utf-8") as f:
        cuad = json.load(f)

    # Structure: {"data": [{"title": ..., "paragraphs": [{"context": ...}]}]}
    contracts: dict[str, str] = {}
    for entry in cuad["data"]:
        title   = entry["title"]
        context = entry["paragraphs"][0]["context"]
        if title not in contracts:
            contracts[title] = context

    logger.info(f"Unique contracts found: {len(contracts)}")

    records = []
    for i, (title, text) in enumerate(contracts.items()):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        fp   = RAW_DIR / f"{safe}.txt"
        fp.write_text(text, encoding="utf-8")
        records.append({
            "contract_id": i,
            "title":       title,
            "file_name":   fp.name,
            "file_path":   str(fp.relative_to(ROOT)),
            "char_count":  len(text),
            "word_count":  len(text.split()),
        })

    df = pd.DataFrame(records)
    df.to_csv(CUAD_DIR / "manifest.csv", index=False)

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    sample.to_csv(CUAD_DIR / "sample_manifest.csv", index=False)

    logger.success(
        f"\n── Done ──────────────────────────────────────\n"
        f"  Total contracts : {len(df)}\n"
        f"  Avg words       : {df['word_count'].mean():.0f}\n"
        f"  Files saved to  : data/raw/\n"
        f"  Sample manifest : data/cuad/sample_manifest.csv ({len(sample)} contracts)\n"
        f"──────────────────────────────────────────────"
    )


if __name__ == "__main__":
    download_cuad()