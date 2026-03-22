"""
Run StreamingBruteForceModelSelector on fake streaming data.

This script does not make external API calls; it uses a deterministic fake
agent so you can validate streaming update behavior quickly.
"""

from __future__ import annotations

import argparse
from random import Random
from typing import Any, Dict, List, Sequence, Tuple

from agentopt import StreamingBruteForceModelSelector


def build_fake_stream(total_samples: int, seed: int) -> List[Tuple[Dict[str, int], int]]:
    """Create a simple synthetic stream: expected output is exactly x."""
    rng = Random(seed)
    stream: List[Tuple[Dict[str, int], int]] = []
    for _ in range(total_samples):
        x = rng.randint(0, 100)
        stream.append(({"x": x}, x))
    return stream


def agent_fn(combo: Dict[str, Any]):
    model_name = combo["agent"]

    def _run(input_data: Dict[str, int]) -> int:
        x = input_data["x"]
        # Deterministic fake model behavior:
        # - "good": always correct
        # - "noisy": occasionally wrong
        # - "bad": always off by +1
        if model_name == "good":
            return x
        if model_name == "noisy":
            return x if (x % 5 != 0) else (x + 1)
        return x + 1

    return _run


def eval_fn(expected: int, actual: int) -> float:
    return 1.0 if expected == actual else 0.0


def batches_of(
    stream: Sequence[Tuple[Dict[str, int], int]], batch_size: int
) -> List[List[Tuple[Dict[str, int], int]]]:
    return [list(stream[i : i + batch_size]) for i in range(0, len(stream), batch_size)]


def run(total: int, warm_start: int, batch_size: int, seed: int) -> None:
    if warm_start < 1:
        raise ValueError("warm_start must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if total <= warm_start:
        raise ValueError("total must be greater than warm_start")

    stream = build_fake_stream(total, seed)
    seed_dataset = stream[:warm_start]
    incoming = stream[warm_start:]

    selector = StreamingBruteForceModelSelector(
        agent_fn=agent_fn,
        models={"agent": ["good", "noisy", "bad"]},
        eval_fn=eval_fn,
        dataset=seed_dataset,
    )

    print("== Initial seed evaluation ==")
    initial_results = selector.select_best()
    print(initial_results)
    print(f"Initial best: {initial_results.get_best_combo()}")

    print("\n== Streaming updates ==")
    for i, batch in enumerate(batches_of(incoming, batch_size), 1):
        results = selector.update(batch, parallel=True, max_concurrent=20)
        best = results.get_best()
        print(
            f"Batch {i}: size={len(batch)} "
            f"best={results.get_best_combo()} "
            f"accuracy={best.accuracy:.3f} "
            f"latency={best.latency_seconds:.4f}s"
        )

    print("\n== Final cumulative ranking ==")
    print(selector.results())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test streaming model selection on fake data.")
    parser.add_argument("--total", type=int, default=60, help="Total fake samples.")
    parser.add_argument("--warm-start", type=int, default=12, help="Initial seed samples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Incoming batch size.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    run(
        total=args.total,
        warm_start=args.warm_start,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
