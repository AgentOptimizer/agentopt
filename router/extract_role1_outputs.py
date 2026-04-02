#!/usr/bin/env python3
"""
Extract role1 (planner/answer) model outputs from cache.db for two-pass routing.

For HotpotQA: extracts planner outputs (1,800 entries: 200 samples × 9 models)
For MathQA: extracts first answer-model outputs (1,800 entries: 200 samples × 9 models)

These outputs are used as additional context for head_role2 during two-pass routing:
  head picks role1 from query text → role1 runs → head picks role2 from query + role1 output

Output: router/role1_outputs.json with structure:
{
  "hotpotqa": {"0": {"Claude 3 Haiku": "Ongoing: NEXT: ...", ...}, ...},
  "mathqa": {"0": {"Claude 3 Haiku": "Step 1: ...", ...}, ...}
}

Usage:
    cd agentopt/
    python -m router.extract_role1_outputs
"""

from __future__ import annotations

import base64
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

# Profile ID → display name mapping
PROFILE_MAP = {
    "58ii6j0n0zhw": "Claude 3 Haiku",
    "4ax1twcuwbfk": "Claude Haiku 4.5",
    "vqhud2pxz4wy": "Claude Opus 4.6",
    "nrqbxznvrt7p": "Kimi K2.5",
    "uj2ujdo7k1qe": "Ministral 3 8B",
    "d6kuf8xcphsl": "Qwen3 32B",
    "a6jppcyeu4ms": "Qwen3 Next 80B A3B",
    "d9uiuyipu5b2": "gpt-oss-120b",
    "fkpdj71utboq": "gpt-oss-20b",
}

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.path.join(ROOT, ".agentopt_cache", "cache.db")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "role1_outputs.json")


def _get_model_name(url: str) -> str | None:
    """Extract model display name from Bedrock URL."""
    for pid, name in PROFILE_MAP.items():
        if pid in url:
            return name
    return None


def _extract_response_text(response_bytes: bytes) -> str:
    """Extract text content from Bedrock response, filtering reasoning blocks."""
    resp = json.loads(response_bytes)
    content = resp["output"]["message"]["content"]
    # Filter out reasoning_content blocks (gpt-oss models)
    texts = []
    for block in content:
        if block.get("type") == "reasoning_content":
            continue
        if "text" in block:
            texts.append(block["text"])
    return "\n".join(texts)


def _load_hotpotqa_queries() -> list[str]:
    """Load HotpotQA queries in the same order as data.py."""
    data_path = Path(ROOT) / "benchmarks" / "HotpotQA" / "data" / "hotpot_dev_distractor_v1.json"
    if not data_path.exists():
        data_path = Path(ROOT).parent / "other_benchmarks" / "hotpot_qa" / "hotpot_dev_distractor_v1.json"

    with open(data_path, "r") as f:
        raw = json.load(f)

    items = list(raw)
    rng = random.Random(0)
    rng.shuffle(items)
    items = items[:200]

    queries = []
    for item in items:
        question = str(item.get("question", "")).strip()
        queries.append(question)
    return queries


def _load_mathqa_queries() -> list[str]:
    """Load MathQA queries in the same order as data.py."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/math_qa", split="test")
        queries = []
        for i, row in enumerate(ds):
            if i >= 200:
                break
            queries.append(f"{row['Problem']}\n\nOptions: {row['options']}")
        return queries
    except Exception:
        return [""] * 200


def _match_hotpotqa(user_text: str, queries: list[str]) -> int | None:
    """Match a planner user message to a HotpotQA sample index.

    The planner user message format is:
    "Context:\n...\n\nQuestion: {question}\nAnswer:\n\nNo prior solver output yet..."

    We match by extracting the question from the user message.
    """
    # Extract question from user message
    for marker in ["Question: ", "Question:"]:
        qstart = user_text.find(marker)
        if qstart >= 0:
            qstart += len(marker)
            # Question ends at "\nAnswer:" or "\n\n"
            qend = user_text.find("\nAnswer:", qstart)
            if qend < 0:
                qend = user_text.find("\n\n", qstart)
            if qend < 0:
                qend = min(qstart + 500, len(user_text))
            question = user_text[qstart:qend].strip()

            # Match against queries
            for idx, q in enumerate(queries):
                if q and question[:100] == q[:100]:
                    return idx
            # Fuzzy fallback
            for idx, q in enumerate(queries):
                if q and question[:50] in q[:200]:
                    return idx
            break
    return None


def _match_mathqa(user_text: str, queries: list[str]) -> int | None:
    """Match a MathQA answer user message to a sample index.

    The user message IS the query text directly (problem + options).
    """
    # Direct match on first 100 chars
    prefix = user_text[:100]
    for idx, q in enumerate(queries):
        if q and q[:100] == prefix:
            return idx
    # Fuzzy fallback
    for idx, q in enumerate(queries):
        if q and user_text[:50] in q[:200]:
            return idx
    return None


def extract_role1_outputs():
    """Scan cache.db and extract all role1 model outputs."""
    if not os.path.exists(CACHE_DB):
        print(f"ERROR: cache.db not found at {CACHE_DB}")
        sys.exit(1)

    print("Loading benchmark queries...")
    hotpotqa_queries = _load_hotpotqa_queries()
    mathqa_queries = _load_mathqa_queries()
    print(f"  HotpotQA: {len(hotpotqa_queries)} queries")
    print(f"  MathQA: {len(mathqa_queries)} queries")

    # Output structure
    outputs = {
        "hotpotqa": {},  # {sample_idx_str: {model_name: output_text}}
        "mathqa": {},
    }

    print(f"\nScanning cache.db at {CACHE_DB}...")
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.execute("SELECT data_json FROM cache")

    hotpotqa_found = 0
    mathqa_found = 0
    hotpotqa_unmatched = 0
    mathqa_unmatched = 0

    for (data_json,) in cursor:
        data = json.loads(data_json)
        rb = data.get("request_body", {})

        # Get system message
        system_text = ""
        if "system" in rb and rb["system"]:
            for s in rb["system"]:
                if "text" in s:
                    system_text += s["text"]

        # Get model name from URL
        url = rb.get("_bedrock_url", "")
        model_name = _get_model_name(url)
        if not model_name:
            continue

        # Get user message text
        user_text = ""
        msgs = rb.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content", [])
                if isinstance(content, list) and content:
                    user_text = content[0].get("text", "")
                break

        # Decode response
        try:
            resp_bytes = base64.b64decode(data["response_bytes_b64"])
            output_text = _extract_response_text(resp_bytes)
        except Exception:
            continue

        sl = system_text.lower()

        # HotpotQA planner
        if "planner" in sl and "planner/solver" in sl:
            idx = _match_hotpotqa(user_text, hotpotqa_queries)
            if idx is not None:
                idx_str = str(idx)
                if idx_str not in outputs["hotpotqa"]:
                    outputs["hotpotqa"][idx_str] = {}
                outputs["hotpotqa"][idx_str][model_name] = output_text
                hotpotqa_found += 1
            else:
                hotpotqa_unmatched += 1

        # MathQA answer (first call only — 1 message)
        elif "math problem solver" in sl and len(msgs) == 1:
            idx = _match_mathqa(user_text, mathqa_queries)
            if idx is not None:
                idx_str = str(idx)
                if idx_str not in outputs["mathqa"]:
                    outputs["mathqa"][idx_str] = {}
                outputs["mathqa"][idx_str][model_name] = output_text
                mathqa_found += 1
            else:
                mathqa_unmatched += 1

    conn.close()

    # Print stats
    print(f"\nHotpotQA planner outputs:")
    print(f"  Matched: {hotpotqa_found}")
    print(f"  Unmatched: {hotpotqa_unmatched}")
    print(f"  Samples with outputs: {len(outputs['hotpotqa'])}")
    if outputs["hotpotqa"]:
        models_per_sample = [len(v) for v in outputs["hotpotqa"].values()]
        print(f"  Models per sample: min={min(models_per_sample)}, "
              f"max={max(models_per_sample)}, "
              f"avg={sum(models_per_sample)/len(models_per_sample):.1f}")

    print(f"\nMathQA answer outputs:")
    print(f"  Matched: {mathqa_found}")
    print(f"  Unmatched: {mathqa_unmatched}")
    print(f"  Samples with outputs: {len(outputs['mathqa'])}")
    if outputs["mathqa"]:
        models_per_sample = [len(v) for v in outputs["mathqa"].values()]
        print(f"  Models per sample: min={min(models_per_sample)}, "
              f"max={max(models_per_sample)}, "
              f"avg={sum(models_per_sample)/len(models_per_sample):.1f}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(outputs, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    extract_role1_outputs()
