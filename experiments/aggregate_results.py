#!/usr/bin/env python3
"""
Aggregate per-sample JSONL into per-combo summaries.

Usage:
    python aggregate_results.py --jsonl results/gpqa_200_bf.jsonl [--output results/gpqa_200_bf_agg.jsonl]

Input format (per-sample):
    {"combo_name": "agent=Claude Opus 4.6", "sample_index": 0, "score": 1.0, ...}

Output format (per-combo, same as old format):
    {"model_name": "agent=Claude Opus 4.6", "accuracy": 0.90, "latency_seconds": 5.2, ...}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def aggregate(input_path: str, output_path: str | None = None) -> list[dict]:
    """Read per-sample JSONL and produce per-combo summaries."""
    combos: dict[str, list[dict]] = defaultdict(list)

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            combos[row["combo_name"]].append(row)

    results = []
    best_name = None
    best_accuracy = -1.0
    best_latency = float("inf")

    for combo_name, samples in sorted(combos.items()):
        samples.sort(key=lambda s: s["sample_index"])
        n = len(samples)
        accuracy = sum(s["score"] for s in samples) / n if n else 0.0
        avg_latency = sum(s["latency_seconds"] for s in samples) / n if n else 0.0

        # Aggregate tokens across all samples
        input_tokens: dict[str, int] = defaultdict(int)
        output_tokens: dict[str, int] = defaultdict(int)
        for s in samples:
            for model, count in s.get("input_tokens", {}).items():
                input_tokens[model] += count
            for model, count in s.get("output_tokens", {}).items():
                output_tokens[model] += count

        # Build per-datapoint results
        dp_results = []
        for s in samples:
            dp_results.append({
                "datapoint_index": s["sample_index"],
                "score": s["score"],
                "latency_seconds": s["latency_seconds"],
                "input_tokens": s.get("input_tokens", {}),
                "output_tokens": s.get("output_tokens", {}),
            })

        result = {
            "model_name": combo_name,
            "accuracy": accuracy,
            "latency_seconds": avg_latency,
            "input_tokens": dict(input_tokens),
            "output_tokens": dict(output_tokens),
            "attribute": "combination",
            "is_best": False,
            "datapoint_results": dp_results,
        }
        results.append(result)

        if accuracy > best_accuracy + 1e-9 or (
            abs(accuracy - best_accuracy) <= 1e-9 and avg_latency < best_latency
        ):
            best_accuracy = accuracy
            best_latency = avg_latency
            best_name = combo_name

    # Mark best
    for r in results:
        if r["model_name"] == best_name:
            r["is_best"] = True

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Aggregated Results from {input_path}")
    print(f"{'='*70}")
    print(f"{'Rank':>4}  {'Model':<40} {'Acc':>7} {'Lat':>8} {'Samples':>7}")
    print(f"{'-'*70}")

    sorted_results = sorted(results, key=lambda r: (-r["accuracy"], r["latency_seconds"]))
    for rank, r in enumerate(sorted_results, 1):
        n_samples = len(r["datapoint_results"])
        marker = ">>>" if r["is_best"] else "   "
        print(
            f"{marker}{rank:>2}  {r['model_name']:<40} "
            f"{r['accuracy']:>6.2%} {r['latency_seconds']:>7.2f}s {n_samples:>7}"
        )
    print(f"{'-'*70}")
    print(f"Total combos: {len(results)}")

    # Write output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nAggregated JSONL written to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-sample JSONL to per-combo summaries")
    parser.add_argument("--jsonl", required=True, help="Input per-sample JSONL file")
    parser.add_argument("--output", "-o", help="Output aggregated JSONL file (optional)")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"Error: {args.jsonl} not found", file=sys.stderr)
        sys.exit(1)

    aggregate(args.jsonl, args.output)


if __name__ == "__main__":
    main()
