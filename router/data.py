"""Dataset loading and tokenization for the BERT router (classification version).

Each sample produces one or more training examples with a classification label
(the tiebreaker-selected best model index) instead of score vectors.

1-tuple benchmarks (GPQA, BFCL):
    1 example: (query_text) → label (0-8, best model by accuracy>latency>cost)

2-tuple benchmarks (HotpotQA, MathQA):
    1 role1 example: (query_text) → label (0-8, best role1 from overall best combo)
    + up to 9 role2 examples: (query_text + role1_model_output) → label (0-8, best role2 for that role1)

Missing scores (partial coverage) are tracked via boolean masks for eval metrics.
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
    _ROLE_MODELS,
)

# ---------------------------------------------------------------------------
# Raw query extraction per benchmark
# ---------------------------------------------------------------------------


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
    """Return formatted context+question for each HotpotQA sample (200 total)."""
    import random as _random

    data_path = Path(__file__).parent.parent / "benchmarks" / "HotpotQA" / "data" / "hotpot_dev_distractor_v1.json"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent.parent / "other_benchmarks" / "hotpot_qa" / "hotpot_dev_distractor_v1.json"

    with open(data_path, "r") as f:
        raw = json.load(f)

    items = list(raw)
    rng = _random.Random(0)
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
    """Return problem+options text for each MathQA sample (200 total)."""
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


def _get_role2_scores_for_role1(
    combo_scores: torch.Tensor,
    combo_mask: torch.Tensor,
    role1_idx: int,
    n_models: int = 9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get 9-dim role2 scores for a specific role1 model."""
    start = role1_idx * n_models
    end = start + n_models
    return combo_scores[start:end].clone(), combo_mask[start:end].clone()


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class RouterDataset(Dataset):
    """PyTorch dataset for the BERT router (classification version).

    Each item contains:
        - input_ids, attention_mask: tokenized text
        - label: int — target class index (0-8), the tiebreaker-selected best model
        - scores: float tensor of shape (9,) — target model scores (for eval metrics)
        - mask: bool tensor of shape (9,) — True where score exists
        - combo_scores: float tensor of shape (81,) — original combo scores (for eval)
        - combo_mask: bool tensor of shape (81,) — original combo mask (for eval)
        - head: int — 0 for 1-tuple, 1 for 2-tuple role1, 2 for 2-tuple role2
        - role1_model_idx: int — which role1 model (for role2 examples), -1 otherwise
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
            "label": s["label"],
            "scores": s["scores"],
            "mask": s["mask"],
            "combo_scores": s["combo_scores"],
            "combo_mask": s["combo_mask"],
            "head": s["head"],
            "role1_model_idx": s.get("role1_model_idx", -1),
            "benchmark": s["benchmark"],
            "sample_idx": s["sample_idx"],
        }


# ---------------------------------------------------------------------------
# Build samples from scores.json + labels.json + role1_outputs.json + queries
# ---------------------------------------------------------------------------


def _load_role1_outputs(path: Optional[str] = None) -> Dict:
    """Load role1 model outputs from role1_outputs.json."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "role1_outputs.json")
    if not os.path.exists(path):
        print(f"WARNING: role1_outputs.json not found at {path}")
        print("  Run: python -m router.extract_role1_outputs")
        return {}
    with open(path) as f:
        return json.load(f)


def _load_labels() -> Dict:
    """Load tiebreaker labels from labels.json."""
    path = os.path.join(os.path.dirname(__file__), "labels.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"labels.json not found at {path}\n"
            "Run: python -m router.extract_labels"
        )
    with open(path) as f:
        return json.load(f)


def load_samples(
    config: RouterConfig,
    scores_path: Optional[str] = None,
) -> List[dict]:
    """Load all samples across benchmarks.

    For 1-tuple benchmarks: 1 training example per sample.
    For 2-tuple benchmarks: 1 role1 example + up to 9 role2 examples per sample.

    Returns a list of dicts, each with:
        text: str — tokenizer input
        label: int — tiebreaker-selected best model index (0-8)
        scores: Tensor of shape (9,) — target scores (for eval metrics only)
        mask: Tensor of shape (9,)
        combo_scores: Tensor of shape (81,) — original combo scores
        combo_mask: Tensor of shape (81,) — original combo mask
        head: int — 0 (1-tuple), 1 (2-tuple role1), 2 (2-tuple role2)
        role1_model_idx: int — -1 for 1-tuple/role1, 0-8 for role2
        benchmark: str
        sample_idx: int
    """
    scores_file = scores_path or config.scores_path
    if not os.path.isabs(scores_file):
        scores_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            scores_file,
        )

    with open(scores_file) as f:
        all_scores = json.load(f)

    # Load labels (tiebreaker-selected best model per sample)
    all_labels = _load_labels()

    # Load role1 outputs for two-pass training
    role1_outputs = _load_role1_outputs()

    n_models = config.n_models
    samples = []
    skipped_no_label = 0

    for bench in config.benchmarks:
        bench_scores = all_scores.get(bench, {})
        if not bench_scores:
            print(f"WARNING: No scores for {bench}")
            continue

        bench_labels = all_labels.get(bench, {})
        if not bench_labels:
            print(f"WARNING: No labels for {bench}")
            continue

        combo_list = _get_combo_list(bench)
        is_2tuple = bench in ("hotpotqa", "mathqa")
        token = BENCHMARK_TOKENS[bench]

        # Load query texts
        query_loader = _QUERY_LOADERS[bench]
        queries = query_loader()

        # Load role1 outputs for this benchmark
        bench_role1 = role1_outputs.get(bench, {})

        n_samples = len(bench_scores)
        if len(queries) < n_samples:
            queries.extend([""] * (n_samples - len(queries)))

        for str_idx, score_dict in bench_scores.items():
            idx = int(str_idx)
            if idx >= len(queries):
                continue

            # Get label for this sample
            sample_label = bench_labels.get(str_idx)
            if sample_label is None:
                skipped_no_label += 1
                continue

            query = queries[idx]

            # Build original combo score vectors (81-dim max)
            combo_scores_vec = torch.zeros(n_models * n_models, dtype=torch.float32)
            combo_mask_vec = torch.zeros(n_models * n_models, dtype=torch.bool)

            for combo_idx, combo_name in enumerate(combo_list):
                if combo_name in score_dict:
                    combo_scores_vec[combo_idx] = score_dict[combo_name]
                    combo_mask_vec[combo_idx] = True

            if not is_2tuple:
                # 1-tuple: single training example
                # label is int (best model index 0-8)
                label = sample_label

                scores = torch.zeros(n_models, dtype=torch.float32)
                mask = torch.zeros(n_models, dtype=torch.bool)
                for combo_idx, combo_name in enumerate(combo_list):
                    if combo_name in score_dict:
                        scores[combo_idx] = score_dict[combo_name]
                        mask[combo_idx] = True

                samples.append({
                    "text": f"{token} {query}",
                    "label": label,
                    "scores": scores,
                    "mask": mask,
                    "combo_scores": combo_scores_vec,
                    "combo_mask": combo_mask_vec,
                    "head": 0,
                    "role1_model_idx": -1,
                    "benchmark": bench,
                    "sample_idx": idx,
                })
            else:
                # 2-tuple: role1 example
                # sample_label is {"role1": int, "role2": {"0": int, ...}}
                role1_label = sample_label["role1"]
                role2_labels = sample_label.get("role2", {})

                # Build role1 scores for eval (marginal max scores)
                r1_scores = torch.zeros(n_models, dtype=torch.float32)
                r1_mask = torch.zeros(n_models, dtype=torch.bool)
                scores_2d = combo_scores_vec[:n_models * n_models].view(n_models, n_models)
                mask_2d = combo_mask_vec[:n_models * n_models].view(n_models, n_models)

                for i in range(n_models):
                    valid = mask_2d[i]
                    if valid.any():
                        r1_scores[i] = scores_2d[i][valid].max().item()
                        r1_mask[i] = True

                samples.append({
                    "text": f"{token} {query}",
                    "label": role1_label,
                    "scores": r1_scores,
                    "mask": r1_mask,
                    "combo_scores": combo_scores_vec,
                    "combo_mask": combo_mask_vec,
                    "head": 1,
                    "role1_model_idx": -1,
                    "benchmark": bench,
                    "sample_idx": idx,
                })

                # 2-tuple: role2 examples (query + role1 output → best role2)
                sample_role1 = bench_role1.get(str_idx, {})
                for model_idx, model_name in enumerate(_ROLE_MODELS):
                    output_text = sample_role1.get(model_name)
                    if output_text is None:
                        continue

                    # Get role2 label for this role1
                    r2_label = role2_labels.get(str(model_idx))
                    if r2_label is None:
                        continue

                    r2_scores, r2_mask = _get_role2_scores_for_role1(
                        combo_scores_vec, combo_mask_vec, model_idx, n_models
                    )
                    if not r2_mask.any():
                        continue

                    role2_text = f"{token} {query} [SEP] {output_text}"

                    samples.append({
                        "text": role2_text,
                        "label": r2_label,
                        "scores": r2_scores,
                        "mask": r2_mask,
                        "combo_scores": combo_scores_vec,
                        "combo_mask": combo_mask_vec,
                        "head": 2,
                        "role1_model_idx": model_idx,
                        "benchmark": bench,
                        "sample_idx": idx,
                    })

    if skipped_no_label > 0:
        print(f"WARNING: Skipped {skipped_no_label} samples with no label")

    return samples


def train_test_split(
    samples: List[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Stratified train/test split by benchmark AND sample_idx.

    All examples from the same (benchmark, sample_idx) go to the same split.
    This prevents data leakage: role2 examples for a sample can't appear
    in train if the role1 example is in test.
    """
    rng = random.Random(seed)

    # Group by (benchmark, sample_idx)
    by_key: Dict[tuple, List[dict]] = {}
    for s in samples:
        key = (s["benchmark"], s["sample_idx"])
        by_key.setdefault(key, []).append(s)

    # Group keys by benchmark for stratification
    bench_keys: Dict[str, List[tuple]] = {}
    for key in by_key:
        bench_keys.setdefault(key[0], []).append(key)

    train, test = [], []
    for bench, keys in sorted(bench_keys.items()):
        rng.shuffle(keys)
        n_test = max(1, int(len(keys) * test_size))
        test_keys = set(keys[:n_test])

        for key in keys:
            if key in test_keys:
                test.extend(by_key[key])
            else:
                train.extend(by_key[key])

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test
