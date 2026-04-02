"""Training script for the BERT router (classification version).

The model has a single shared Linear(1024 -> 9) head. Training uses cross-entropy
loss against tiebreaker-selected labels (accuracy > latency > cost).

Training examples:
- 1-tuple (GPQA, BFCL): query -> label (best model index 0-8)
- 2-tuple role1 (HotpotQA, MathQA): query -> label (best role1 index 0-8)
- 2-tuple role2: query + role1_output -> label (best role2 index 0-8 for that role1)

Usage:
    cd agentopt/
    python -m router.train
    python -m router.train --epochs 5 --batch-size 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import BENCHMARK_TOKENS, RouterConfig
from .data import RouterDataset, load_samples, train_test_split
from .evaluate import evaluate
from .model import BERTRouter, masked_cross_entropy_loss


def _collate_fn(batch):
    """Custom collate: pad scores/mask to 9, combo_scores/mask to 81."""
    n_models = 9
    n_combo = 81

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    heads = torch.tensor([b["head"] for b in batch], dtype=torch.long)
    role1_model_idxs = torch.tensor(
        [b["role1_model_idx"] for b in batch], dtype=torch.long
    )

    scores = torch.zeros(len(batch), n_models)
    masks = torch.zeros(len(batch), n_models, dtype=torch.bool)
    combo_scores = torch.zeros(len(batch), n_combo)
    combo_masks = torch.zeros(len(batch), n_combo, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["scores"].size(0)
        scores[i, :n] = b["scores"]
        masks[i, :n] = b["mask"]
        nc = b["combo_scores"].size(0)
        combo_scores[i, :nc] = b["combo_scores"]
        combo_masks[i, :nc] = b["combo_mask"]

    benchmarks = [b["benchmark"] for b in batch]
    sample_idxs = [b["sample_idx"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "scores": scores,
        "mask": masks,
        "combo_scores": combo_scores,
        "combo_mask": combo_masks,
        "head": heads,
        "role1_model_idx": role1_model_idxs,
        "benchmark": benchmarks,
        "sample_idx": sample_idxs,
    }


def train(config: RouterConfig):
    """Train the BERT router with cross-entropy classification."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    torch.manual_seed(config.seed)

    # Load tokenizer and add special tokens
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens = list(BENCHMARK_TOKENS.values())
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {n_added} special tokens: {special_tokens}")

    # Load data
    print("Loading samples...")
    samples = load_samples(config)
    train_samples, test_samples = train_test_split(
        samples, test_size=config.test_size, seed=config.seed
    )

    # Print split stats
    from collections import Counter
    train_heads = Counter(s["head"] for s in train_samples)
    test_heads = Counter(s["head"] for s in test_samples)
    train_bench = Counter(s["benchmark"] for s in train_samples)
    test_bench = Counter(s["benchmark"] for s in test_samples)

    print(f"\nTrain: {len(train_samples)} examples")
    print(f"  By head: {dict(sorted(train_heads.items()))}")
    for b in sorted(train_bench):
        print(f"  {b}: {train_bench[b]}")
    print(f"Test: {len(test_samples)} examples")
    print(f"  By head: {dict(sorted(test_heads.items()))}")
    for b in sorted(test_bench):
        print(f"  {b}: {test_bench[b]}")

    # Label distribution in train
    train_labels = Counter(s["label"] for s in train_samples)
    print(f"  Train label distribution: {dict(sorted(train_labels.items()))}")

    # Count unique test samples (for eval reporting)
    test_unique = len(set((s["benchmark"], s["sample_idx"]) for s in test_samples))
    print(f"  Unique test samples: {test_unique}")

    train_dataset = RouterDataset(train_samples, tokenizer, config.max_length)
    test_dataset = RouterDataset(test_samples, tokenizer, config.max_length)

    # Gradient accumulation
    micro_batch = config.micro_batch
    accum_steps = max(1, config.batch_size // micro_batch)
    print(f"Micro batch: {micro_batch}, accumulation steps: {accum_steps}, "
          f"effective batch: {micro_batch * accum_steps}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=micro_batch,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
    )

    # Model
    print(f"\nLoading model: {config.model_name}", flush=True)
    model = BERTRouter(config.model_name, config.n_models)
    model.encoder.resize_token_embeddings(len(tokenizer))

    # Optionally freeze encoder
    if config.freeze_encoder:
        model.encoder.requires_grad_(False)
        print("Encoder FROZEN — only training linear head")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable",
          flush=True)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining: {config.epochs} epochs, {len(train_loader)} micro-batches/epoch, "
          f"{optimizer_steps_per_epoch} optimizer steps/epoch, "
          f"{total_steps} total steps, {warmup_steps} warmup",
          flush=True)

    # Training loop
    os.makedirs(config.output_dir, exist_ok=True)
    best_metric = -1.0
    best_epoch = -1

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_total = 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            # Single forward pass — shared head for all example types
            logits = model(input_ids, attention_mask)
            loss = masked_cross_entropy_loss(logits, labels, mask)
            loss = loss / accum_steps
            loss.backward()

            epoch_loss += loss.item() * accum_steps
            n_batches += 1

            # Track training accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = n_correct / max(n_total, 1)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch + 1}/{config.epochs} — "
              f"loss: {avg_loss:.4f}, train_acc: {train_acc:.1%}, "
              f"time: {elapsed:.1f}s, lr: {scheduler.get_last_lr()[0]:.2e}")

        # Evaluate
        metrics = evaluate(model, test_loader, device, tokenizer,
                           n_models=config.n_models)
        _print_metrics(metrics)

        # Save best model (by overall selection_accuracy)
        overall_sel_acc = metrics["overall"]["selection_accuracy"]
        if overall_sel_acc > best_metric:
            best_metric = overall_sel_acc
            best_epoch = epoch + 1
            save_path = os.path.join(config.output_dir, "best_model")
            model.encoder.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            torch.save({
                "head": model.head.state_dict(),
                "config": {
                    "model_name": config.model_name,
                    "n_models": config.n_models,
                },
            }, os.path.join(save_path, "heads.pt"))
            print(f"  ** Saved best model (epoch {best_epoch}, "
                  f"sel_acc={overall_sel_acc:.3f})")

        # Save metrics
        metrics_path = os.path.join(config.output_dir, f"metrics_epoch{epoch + 1}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Best: epoch {best_epoch}, "
          f"sel_acc={best_metric:.3f}")
    print(f"Model saved to: {config.output_dir}/best_model/")


def _print_metrics(metrics: dict):
    """Print evaluation metrics in a readable format."""
    for key in sorted(metrics.keys()):
        if key == "overall":
            continue
        m = metrics[key]
        print(f"  {key:>10}: sel_acc={m['selection_accuracy']:.1%}, "
              f"top3={m['top3_accuracy']:.1%}, "
              f"mean_score={m['mean_selected_score']:.3f}, "
              f"oracle={m['oracle_score']:.3f}, "
              f"regret={m['mean_regret']:.3f}")
    m = metrics["overall"]
    print(f"  {'OVERALL':>10}: sel_acc={m['selection_accuracy']:.1%}, "
          f"top3={m['top3_accuracy']:.1%}, "
          f"mean_score={m['mean_selected_score']:.3f}, "
          f"regret={m['mean_regret']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the two-pass BERT router")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Effective batch size (via gradient accumulation)")
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="Actual batch size per forward pass")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="router/checkpoints_twopass")
    parser.add_argument("--scores-path", type=str, default="router/scores.json")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze BERT encoder, only train linear head")
    parser.add_argument("--agg", type=str, default="max", choices=["max", "mean"],
                        help="2-tuple marginal aggregation: max (sharper) or mean (smoother)")
    args = parser.parse_args()

    config = RouterConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        test_size=args.test_size,
        output_dir=args.output_dir,
        scores_path=args.scores_path,
        freeze_encoder=args.freeze_encoder,
        agg=args.agg,
    )
    train(config)


if __name__ == "__main__":
    main()
