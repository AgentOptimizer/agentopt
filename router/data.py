"""Dataset loading and tokenization for the BERT router.

Each sample is: (benchmark_token + query_text) → score vector.
- 1-tuple benchmarks (GPQA, BFCL): 9-dim score vector
- 2-tuple benchmarks (HotpotQA, MathQA): decomposed into two 9-dim vectors
  (role1 marginal scores, role2 marginal scores)

For 2-tuple, the marginal scores are computed by averaging across the other
role. E.g., role1_score[i] = mean of combo_score[i*9+j] for all valid j.
This captures "how good is this model for role 1 on average."

Missing scores (partial coverage, especially MathQA) are tracked via a
boolean mask tensor. The loss function and evaluation metrics use this
mask to ignore missing entries — no zero-filling or sample dropping.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .config import (
    BENCHMARK_TOKENS,
    COMBOS_HOTPOTQA,
    COMBOS_MATHQA,
    MODELS_1TUPLE,
    RouterConfig,
)

# ---------------------------------------------------------------------------
# Raw query extraction per benchmark
# ---------------------------------------------------------------------------

# These functions are used only when loading from datasets directly.
# For score extraction we only need scores.json, but we also need the
# query text from the actual benchmark datasets.


def _load_gpqa_queries() -> List[str]:
    """Return raw question text for each GPQA sample (198 total)."""
    data_path = Path(__file__).parent.parent / "benchmarks" / "GPQA" / "data" / "gpqa_diamond.jsonl"
    queries = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row:
                queries.append(row["Question"])
    return queries


def _load_bfcl_queries() -> List[str]:
    """Return initial user message for each BFCL sample (200 total)."""
    data_path = (Path(__file__).parent.parent / "benchmarks" / "BFCL" /
                 "data" / "BFCL_v3_multi_turn_base.json")
    queries = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            try:
                queries.append(sample["question"][0][0]["content"])
            except (KeyError, IndexError):
                queries.append("")
            if len(queries) >= 200:
                break
    return queries


def _load_hotpotqa_queries() -> List[str]:
    """Return formatted context+question for each HotpotQA sample (200 total).

    Uses the same loading logic as the benchmark: shuffle with seed=0,
    take first 200, format context with _format_context().
    """
    import random as _random

    data_path = Path(__file__).parent.parent / "benchmarks" / "HotpotQA" / "data" / "hotpot_dev_distractor_v1.json"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent.parent / "other_benchmarks" / "hotpot_qa" / "hotpot_dev_distractor_v1.json"

    with open(data_path, "r") as f:
        raw = json.load(f)

    items = list(raw)
    rng = _random.Random(0)  # benchmark uses seed=0
    rng.shuffle(items)
    items = items[:200]

    queries = []
    for item in items:
        question = str(item.get("question", "")).strip()
        context = _format_hotpotqa_context(item.get("context"))
        queries.append(f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    return queries


def _format_hotpotqa_context(context) -> str:
    """Format HotpotQA context exactly as the benchmark does."""
    blocks = []
    if not isinstance(context, list):
        return str(context)
    for entry in context:
        if (isinstance(entry, (list, tuple)) and len(entry) == 2
                and isinstance(entry[0], str) and isinstance(entry[1], list)):
            title = entry[0].strip()
            sentences = [str(s).strip() for s in entry[1] if str(s).strip()]
            text = " ".join(sentences)
            blocks.append(f"[{title}] {text}".strip() if title else text)
        else:
            blocks.append(str(entry).strip())
    return "\n".join(b for b in blocks if b)


def _load_mathqa_queries() -> List[str]:
    """Return problem+options text for each MathQA sample (200 total).

    Loads from HuggingFace datasets library (allenai/math_qa) to match
    the benchmark's load_math_qa() behavior.
    """
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
        # Fallback: return empty strings (will still train on scores)
        return [""] * 200


_QUERY_LOADERS = {
    "gpqa": _load_gpqa_queries,
    "bfcl": _load_bfcl_queries,
    "hotpotqa": _load_hotpotqa_queries,
    "mathqa": _load_mathqa_queries,
}


def _get_combo_list(benchmark: str) -> List[str]:
    """Return the canonical ordered combo list for a benchmark."""
    if benchmark in ("gpqa", "bfcl"):
        return MODELS_1TUPLE
    elif benchmark == "hotpotqa":
        return COMBOS_HOTPOTQA
    elif benchmark == "mathqa":
        return COMBOS_MATHQA
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _decompose_2tuple_scores(
    combo_scores: torch.Tensor,
    combo_mask: torch.Tensor,
    n_models: int = 9,
    agg: str = "max",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose 81-dim combo scores into two 9-dim role marginals.

    For role1 (planner/answer), role1_score[i] = agg of combo_score[i*9+j]
    across all valid j.

    For role2 (solver/critic), role2_score[j] = agg of combo_score[i*9+j]
    across all valid i.

    agg="max": best-case score (sharper signal, captures strong pairings)
    agg="mean": average-case score (blurrier, dilutes strong pairings)

    Args:
        combo_scores: (81,) float tensor
        combo_mask: (81,) bool tensor
        n_models: number of models per role (9)
        agg: "max" or "mean"

    Returns:
        role1_scores: (9,) float
        role1_mask: (9,) bool — True if at least one combo with this role1 model has a score
        role2_scores: (9,) float
        role2_mask: (9,) bool
    """
    role1_scores = torch.zeros(n_models, dtype=torch.float32)
    role1_mask = torch.zeros(n_models, dtype=torch.bool)
    role2_scores = torch.zeros(n_models, dtype=torch.float32)
    role2_mask = torch.zeros(n_models, dtype=torch.bool)

    # Reshape to (9, 9) — row=role1, col=role2
    scores_2d = combo_scores[:n_models * n_models].view(n_models, n_models)
    mask_2d = combo_mask[:n_models * n_models].view(n_models, n_models)

    # Role 1 marginals (aggregate across role2 dimension)
    for i in range(n_models):
        valid = mask_2d[i]  # which role2 models have scores for this role1
        if valid.any():
            vals = scores_2d[i][valid]
            role1_scores[i] = vals.max().item() if agg == "max" else vals.mean().item()
            role1_mask[i] = True

    # Role 2 marginals (aggregate across role1 dimension)
    for j in range(n_models):
        valid = mask_2d[:, j]  # which role1 models have scores for this role2
        if valid.any():
            vals = scores_2d[:, j][valid]
            role2_scores[j] = vals.max().item() if agg == "max" else vals.mean().item()
            role2_mask[j] = True

    return role1_scores, role1_mask, role2_scores, role2_mask


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class RouterDataset(Dataset):
    """PyTorch dataset for the BERT router.

    Each item contains:
        - input_ids, attention_mask: tokenized "[BENCH] query_text"
        - scores: float tensor of shape (18,) — [role1(9), role2(9)]
          For 1-tuple: role1 = model scores, role2 = zeros
          For 2-tuple: role1 = marginal role1 scores, role2 = marginal role2 scores
        - mask: bool tensor of shape (18,)
        - combo_scores: float tensor of shape (81,) — original combo scores for eval
        - combo_mask: bool tensor of shape (81,) — original combo mask for eval
        - head: int — 0 for 1-tuple, 1 for 2-tuple
        - benchmark: str
        - sample_idx: int
    """

    def __init__(
        self,
        samples: List[dict],
        tokenizer,
        max_length: int = 8192,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "scores": s["scores"],
            "mask": s["mask"],
            "combo_scores": s["combo_scores"],
            "combo_mask": s["combo_mask"],
            "head": s["head"],
            "benchmark": s["benchmark"],
            "sample_idx": s["sample_idx"],
        }


# ---------------------------------------------------------------------------
# Build samples from scores.json + dataset queries
# ---------------------------------------------------------------------------


def load_samples(
    config: RouterConfig,
    scores_path: Optional[str] = None,
) -> List[dict]:
    """Load all samples across benchmarks.

    Returns a list of dicts, each with:
        text: str — "[BENCH] query_text"
        scores: Tensor of shape (18,) — decomposed [role1, role2]
        mask: Tensor of shape (18,)
        combo_scores: Tensor of shape (81,) — original combo scores
        combo_mask: Tensor of shape (81,) — original combo mask
        head: int — 0 or 1
        benchmark: str
        sample_idx: int
    """
    scores_file = scores_path or config.scores_path
    # Resolve relative to agentopt/ root
    if not os.path.isabs(scores_file):
        scores_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            scores_file,
        )

    with open(scores_file) as f:
        all_scores = json.load(f)

    n_models = config.n_models
    samples = []

    for bench in config.benchmarks:
        bench_scores = all_scores.get(bench, {})
        if not bench_scores:
            print(f"WARNING: No scores for {bench}")
            continue

        combo_list = _get_combo_list(bench)
        n_combos = len(combo_list)
        head = 0 if bench in ("gpqa", "bfcl") else 1
        token = BENCHMARK_TOKENS[bench]

        # Load query texts
        query_loader = _QUERY_LOADERS[bench]
        queries = query_loader()

        n_samples = len(bench_scores)
        if len(queries) < n_samples:
            print(f"WARNING: {bench} has {n_samples} score entries but "
                  f"only {len(queries)} queries — padding with empty strings")
            queries.extend([""] * (n_samples - len(queries)))

        for str_idx, score_dict in bench_scores.items():
            idx = int(str_idx)
            if idx >= len(queries):
                print(f"WARNING: {bench} sample {idx} has no query text")
                continue

            query = queries[idx]
            text = f"{token} {query}"

            # Build original combo score and mask vectors (81-dim max)
            combo_scores_vec = torch.zeros(n_models * n_models, dtype=torch.float32)
            combo_mask_vec = torch.zeros(n_models * n_models, dtype=torch.bool)

            for combo_idx, combo_name in enumerate(combo_list):
                if combo_name in score_dict:
                    combo_scores_vec[combo_idx] = score_dict[combo_name]
                    combo_mask_vec[combo_idx] = True

            # Build training targets (18-dim: [role1(9), role2(9)])
            train_scores = torch.zeros(n_models * 2, dtype=torch.float32)
            train_mask = torch.zeros(n_models * 2, dtype=torch.bool)

            if head == 0:
                # 1-tuple: first 9 = model scores, last 9 = zeros
                for combo_idx, combo_name in enumerate(combo_list):
                    if combo_name in score_dict:
                        train_scores[combo_idx] = score_dict[combo_name]
                        train_mask[combo_idx] = True
            else:
                # 2-tuple: decompose 81 → 9+9 marginals
                r1_scores, r1_mask, r2_scores, r2_mask = _decompose_2tuple_scores(
                    combo_scores_vec, combo_mask_vec, n_models, agg=config.agg
                )
                train_scores[:n_models] = r1_scores
                train_scores[n_models:] = r2_scores
                train_mask[:n_models] = r1_mask
                train_mask[n_models:] = r2_mask

            samples.append({
                "text": text,
                "scores": train_scores,
                "mask": train_mask,
                "combo_scores": combo_scores_vec,
                "combo_mask": combo_mask_vec,
                "head": head,
                "benchmark": bench,
                "sample_idx": idx,
            })

    return samples


def train_test_split(
    samples: List[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Stratified train/test split by benchmark.

    Ensures each benchmark has ~test_size fraction in the test set.
    """
    rng = random.Random(seed)

    # Group by benchmark
    by_bench: Dict[str, List[dict]] = {}
    for s in samples:
        by_bench.setdefault(s["benchmark"], []).append(s)

    train, test = [], []
    for bench, bench_samples in sorted(by_bench.items()):
        indices = list(range(len(bench_samples)))
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_indices = set(indices[:n_test])

        for i, s in enumerate(bench_samples):
            if i in test_indices:
                test.append(s)
            else:
                train.append(s)

    # Shuffle within splits
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
