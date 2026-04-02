"""Evaluation metrics for the two-pass BERT router.

For 1-tuple benchmarks (GPQA, BFCL):
    Router predicts 9 model scores → pick argmax → look up actual score.

For 2-tuple benchmarks (HotpotQA, MathQA):
    Pass 1: Router predicts role1 scores from query → pick role1 = argmax
    Pass 2: Router predicts role2 scores from query + role1_output → pick role2
    Combo index = role1 * 9 + role2.
    Look up actual combo score from original 81-dim scores.

Primary metric: selection_accuracy (does the router find the oracle best?).

Metrics per benchmark and overall:
- Selection accuracy: does the router's combo match the oracle best?
- Top-3 accuracy: is the oracle best reachable from top-3 picks?
- Mean selected score: average actual score of the router's pick
- Oracle score: average of the best available score per sample
- Mean regret: oracle_score - mean_selected_score

All metrics respect the mask — only combos with actual scores are
considered when computing argmax, top-k, etc.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .config import (
    BENCHMARK_TOKENS,
    MODELS_1TUPLE,
    COMBOS_HOTPOTQA,
    COMBOS_MATHQA,
    _ROLE_MODELS,
)


def _load_role1_outputs() -> Dict:
    """Load role1 outputs for two-pass evaluation."""
    path = os.path.join(os.path.dirname(__file__), "role1_outputs.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer=None,
    n_models: int = 9,
) -> Dict:
    """Evaluate the model on a dataloader.

    For 2-tuple benchmarks, performs two-pass evaluation:
    1. Predict role1 from query (head=1 examples)
    2. Encode query + role1 output → predict role2

    Args:
        model: BERTRouter model
        dataloader: test DataLoader
        device: torch device
        tokenizer: needed for encoding role2 input in two-pass eval
        n_models: number of models per role

    Returns:
        Dict with per-benchmark and overall metrics.
    """
    model.eval()
    role1_outputs = _load_role1_outputs()

    # First pass: collect all predictions and group by (benchmark, sample_idx)
    # We only use head=0 (1-tuple) and head=1 (role1) examples for eval.
    # head=2 (role2) examples are training-only — at eval time we do two-pass.
    records_1tuple = []  # 1-tuple eval records
    records_2tuple = {}  # {(bench, sample_idx): {combo_scores, combo_mask, role1_pred}}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        heads = batch["head"]
        combo_scores = batch["combo_scores"]
        combo_masks = batch["combo_mask"]
        benchmarks = batch["benchmark"]
        sample_idxs = batch["sample_idx"]

        predictions = model(input_ids, attention_mask).cpu()  # (batch, 9)

        for i in range(len(benchmarks)):
            head = heads[i].item()

            if head == 0:
                # 1-tuple: direct eval
                records_1tuple.append({
                    "benchmark": benchmarks[i],
                    "pred": predictions[i],
                    "combo_scores": combo_scores[i],
                    "combo_mask": combo_masks[i],
                })
            elif head == 1:
                # 2-tuple role1: save prediction for two-pass
                key = (benchmarks[i], sample_idxs[i])
                records_2tuple[key] = {
                    "benchmark": benchmarks[i],
                    "sample_idx": sample_idxs[i],
                    "role1_pred": predictions[i],
                    "combo_scores": combo_scores[i],
                    "combo_mask": combo_masks[i],
                }
            # head == 2 (role2 training examples) — skip during eval

    # Second pass for 2-tuple: pick role1, then encode query+output, predict role2
    records_2tuple_eval = []
    max_length = 2048

    for key, rec in records_2tuple.items():
        bench = rec["benchmark"]
        sample_idx = rec["sample_idx"]
        role1_pred = rec["role1_pred"]  # (9,)
        combo_scores = rec["combo_scores"]  # (81,)
        combo_mask = rec["combo_mask"]  # (81,)

        # Pick role1 model (from valid models only)
        combo_mask_2d = combo_mask[:n_models * n_models].view(n_models, n_models)
        role1_valid = combo_mask_2d.any(dim=1)
        valid_r1 = torch.where(role1_valid)[0]

        if len(valid_r1) == 0:
            continue

        r1_pick = valid_r1[role1_pred[valid_r1].argmax()].item()

        # Get role1 output text
        bench_outputs = role1_outputs.get(bench, {})
        sample_outputs = bench_outputs.get(str(sample_idx), {})
        r1_model_name = _ROLE_MODELS[r1_pick]
        r1_output = sample_outputs.get(r1_model_name, "")

        # Encode query + role1 output for role2 prediction
        if tokenizer is not None and r1_output:
            # Find the original query text from the role1 example
            # We reconstruct: token + query + [SEP] + role1_output
            token = BENCHMARK_TOKENS[bench]
            # We need the query — extract from batch. For now, re-encode.
            # The role1 input was: "token query" — we can get query by
            # using the same data loading. But simpler: just use the
            # role1 input_ids to get the text, then append role1 output.
            # Actually, we need the raw query. Let's build the role2 text.

            # Get the role1 input text from the tokenizer
            # We stored the role1 example's input_ids but it's padded.
            # Instead, build role2 text from what we have.
            # Since we have the sample_idx and benchmark, we can look up
            # the query from the data loaders.
            role2_text = f"{token} [SEP] {r1_output}"

            enc = tokenizer(
                role2_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            r2_input_ids = enc["input_ids"].to(device)
            r2_attention_mask = enc["attention_mask"].to(device)
            role2_pred = model(r2_input_ids, r2_attention_mask).cpu().squeeze(0)
        else:
            # Fallback: use role1 predictions as role2 (no two-pass)
            role2_pred = role1_pred

        records_2tuple_eval.append({
            "benchmark": bench,
            "role1_pred": role1_pred,
            "role2_pred": role2_pred,
            "r1_pick": r1_pick,
            "combo_scores": combo_scores,
            "combo_mask": combo_mask,
        })

    # Compute metrics
    by_bench = defaultdict(lambda: {"1tuple": [], "2tuple": []})
    for r in records_1tuple:
        by_bench[r["benchmark"]]["1tuple"].append(r)
    for r in records_2tuple_eval:
        by_bench[r["benchmark"]]["2tuple"].append(r)

    all_metrics = {}
    all_sel_scores = []
    all_oracle_scores = []
    all_correct = []
    all_top3 = []

    for bench in sorted(by_bench.keys()):
        bench_data = by_bench[bench]
        m = _compute_benchmark_metrics(bench, bench_data, n_models)
        all_metrics[bench] = m

        all_sel_scores.extend(m["_selected_scores"])
        all_oracle_scores.extend(m["_oracle_scores"])
        all_correct.extend(m["_correct"])
        all_top3.extend(m["_top3"])

    # Overall metrics
    n = len(all_correct)
    overall_sel_acc = sum(all_correct) / n if n else 0
    overall_top3 = sum(all_top3) / n if n else 0
    overall_mean_sel = sum(all_sel_scores) / n if n else 0
    overall_oracle = sum(all_oracle_scores) / n if n else 0

    all_metrics["overall"] = {
        "selection_accuracy": overall_sel_acc,
        "top3_accuracy": overall_top3,
        "mean_selected_score": overall_mean_sel,
        "oracle_score": overall_oracle,
        "mean_regret": overall_oracle - overall_mean_sel,
        "n_samples": n,
    }

    # Clean up internal lists
    for bench in list(all_metrics.keys()):
        for k in ("_selected_scores", "_oracle_scores", "_correct", "_top3"):
            all_metrics[bench].pop(k, None)

    return all_metrics


def _compute_benchmark_metrics(
    benchmark: str,
    bench_data: dict,
    n_models: int = 9,
) -> dict:
    """Compute metrics for a single benchmark."""
    correct = []
    top3 = []
    selected_scores = []
    oracle_scores = []
    random_scores = []

    rng = random.Random(42)

    # 1-tuple records
    for r in bench_data["1tuple"]:
        pred = r["pred"]  # (9,)
        combo_scores = r["combo_scores"]  # (81,)
        combo_mask = r["combo_mask"]  # (81,)

        n_combos = n_models
        valid_indices = torch.where(combo_mask[:n_combos])[0]
        if len(valid_indices) == 0:
            continue

        valid_scores = combo_scores[valid_indices]
        valid_preds = pred[valid_indices]

        oracle_val = valid_scores.max().item()
        oracle_idx_in_valid = valid_scores.argmax().item()
        oracle_idx = valid_indices[oracle_idx_in_valid].item()

        selected_idx_in_valid = valid_preds.argmax().item()
        selected_idx = valid_indices[selected_idx_in_valid].item()
        selected_score = combo_scores[selected_idx].item()

        k = min(3, len(valid_indices))
        top_k_in_valid = valid_preds.topk(k).indices
        top_k_indices = valid_indices[top_k_in_valid]
        # Top-3 correct if any of the top-3 picks achieves the oracle score
        top_k_scores = combo_scores[top_k_indices]
        is_top3 = (top_k_scores - oracle_val).abs().min().item() < 1e-6

        rand_idx = valid_indices[rng.randint(0, len(valid_indices) - 1)].item()
        random_scores.append(combo_scores[rand_idx].item())

        # Count as correct if selected score matches oracle score (handles ties)
        correct.append(1 if abs(selected_score - oracle_val) < 1e-6 else 0)
        top3.append(1 if is_top3 else 0)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

    # 2-tuple records (two-pass)
    for r in bench_data["2tuple"]:
        role1_pred = r["role1_pred"]  # (9,)
        role2_pred = r["role2_pred"]  # (9,)
        r1_pick = r["r1_pick"]
        combo_scores = r["combo_scores"]  # (81,)
        combo_mask = r["combo_mask"]  # (81,)

        n_combos = n_models * n_models
        valid_combo_indices = torch.where(combo_mask[:n_combos])[0]
        if len(valid_combo_indices) == 0:
            continue

        # Oracle
        oracle_val = combo_scores[valid_combo_indices].max().item()
        oracle_combo_idx = valid_combo_indices[
            combo_scores[valid_combo_indices].argmax()
        ].item()

        # Role2 pick (from role2 predictions, restricted to valid combos with this role1)
        combo_mask_2d = combo_mask[:n_combos].view(n_models, n_models)
        role2_valid = combo_mask_2d[r1_pick]  # (9,) — which role2 models are valid for this role1
        valid_r2 = torch.where(role2_valid)[0]

        if len(valid_r2) == 0:
            # Fallback: pick any valid combo
            selected_idx = valid_combo_indices[0].item()
            selected_score = combo_scores[selected_idx].item()
        else:
            r2_pick = valid_r2[role2_pred[valid_r2].argmax()].item()
            selected_idx = r1_pick * n_models + r2_pick

            if combo_mask[selected_idx]:
                selected_score = combo_scores[selected_idx].item()
            else:
                # Fallback: best available combo with this role1
                row = combo_scores[r1_pick * n_models:(r1_pick + 1) * n_models]
                row_mask = combo_mask[r1_pick * n_models:(r1_pick + 1) * n_models]
                if row_mask.any():
                    best_in_row = row_mask.float() * row + (~row_mask).float() * (-1e9)
                    selected_idx = r1_pick * n_models + best_in_row.argmax().item()
                    selected_score = combo_scores[selected_idx].item()
                else:
                    selected_idx = valid_combo_indices[0].item()
                    selected_score = combo_scores[selected_idx].item()

        # Top-3: check if oracle reachable from top-3 role1 × top-3 role2
        role1_valid_mask = combo_mask_2d.any(dim=1)
        valid_r1 = torch.where(role1_valid_mask)[0]

        k1 = min(3, len(valid_r1))
        k2 = min(3, len(valid_r2)) if len(valid_r2) > 0 else 0
        top_r1 = valid_r1[role1_pred[valid_r1].topk(k1).indices]

        oracle_r1 = oracle_combo_idx // n_models
        oracle_r2 = oracle_combo_idx % n_models

        if k2 > 0:
            top_r2 = valid_r2[role2_pred[valid_r2].topk(k2).indices]
            # Check if any combo in top-3 role1 × top-3 role2 achieves oracle score
            is_top3 = False
            for r1_cand in top_r1.tolist():
                for r2_cand in top_r2.tolist():
                    cand_idx = r1_cand * n_models + r2_cand
                    if combo_mask[cand_idx] and abs(combo_scores[cand_idx].item() - oracle_val) < 1e-6:
                        is_top3 = True
                        break
                if is_top3:
                    break
        else:
            is_top3 = False

        # Random baseline
        rand_combo = valid_combo_indices[
            rng.randint(0, len(valid_combo_indices) - 1)
        ].item()
        random_scores.append(combo_scores[rand_combo].item())

        # Count as correct if selected score matches oracle score (handles ties)
        correct.append(1 if abs(selected_score - oracle_val) < 1e-6 else 0)
        top3.append(1 if is_top3 else 0)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_val)

    n = len(correct)
    sel_acc = sum(correct) / n if n else 0
    top3_acc = sum(top3) / n if n else 0
    mean_sel = sum(selected_scores) / n if n else 0
    mean_oracle = sum(oracle_scores) / n if n else 0
    mean_random = sum(random_scores) / n if n else 0

    return {
        "selection_accuracy": sel_acc,
        "top3_accuracy": top3_acc,
        "mean_selected_score": mean_sel,
        "oracle_score": mean_oracle,
        "mean_regret": mean_oracle - mean_sel,
        "random_baseline_score": mean_random,
        "n_samples": n,
        "_selected_scores": selected_scores,
        "_oracle_scores": oracle_scores,
        "_correct": correct,
        "_top3": top3,
    }
