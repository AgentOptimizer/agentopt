"""Merge successful supplemental JSONL rows into a new lookup pickle.

The source pickle is never modified.  Existing cells are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from offline_selector_sim_v2 import SampleResult  # noqa: E402


EXPECTED_SHA256 = {
    "hotpotqa": "2e1260c986c4be0cf71fbaf31ce07d4b3fb2abbc6a599f09b246c4971d3ded8a",
    "mathqa": "10b569c8371356bfc41bdfa4e84173d5a3d4f1352340acd45f3473390c65b9be",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_successes(path: Path, benchmark: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if row.get("benchmark") != benchmark or row.get("status") != "ok":
                continue
            key = (row["configuration_id"], int(row["question_id"]))
            latest[key] = row
    return list(latest.values())


def is_missing(table: Any, question_id: int) -> bool:
    if isinstance(table, dict):
        return question_id not in table or table[question_id] is None
    return question_id >= len(table) or table[question_id] is None


def set_cell(table: Any, question_id: int, value: SampleResult) -> None:
    if isinstance(table, dict):
        table[question_id] = value
        return
    while len(table) <= question_id:
        table.append(None)
    table[question_id] = value


def set_server_latency(values: Any, question_id: int, value: float | None) -> None:
    if isinstance(values, dict):
        values[question_id] = value
        return
    while len(values) <= question_id:
        values.append(None)
    values[question_id] = value


def remaining_missing(lookup: dict[str, Any]) -> int:
    n_datapoints = len(lookup["datapoints"])
    count = 0
    for config in lookup["model_names"]:
        table = lookup["table"][config]
        count += sum(is_missing(table, question_id) for question_id in range(n_datapoints))
    return count


def merge(
    benchmark: str,
    source: Path,
    patch_jsonl: Path,
    output: Path,
) -> tuple[int, int]:
    actual_hash = sha256(source)
    expected_hash = EXPECTED_SHA256[benchmark]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Source hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with source.open("rb") as handle:
        lookup = pickle.load(handle)

    rows = load_successes(patch_jsonl, benchmark)
    filled = 0
    for row in rows:
        config = row["configuration_id"]
        question_id = int(row["question_id"])
        if config not in lookup["table"]:
            raise KeyError(f"Unknown configuration: {config}")
        table = lookup["table"][config]
        if not is_missing(table, question_id):
            raise RuntimeError(
                f"Refusing to overwrite existing cell: {config}, q={question_id}"
            )
        result = SampleResult(
            score=float(row["score"]),
            latency_seconds=float(row["latency_seconds"]),
            input_tokens={k: int(v) for k, v in row["input_tokens"].items()},
            output_tokens={k: int(v) for k, v in row["output_tokens"].items()},
            cost=float(row["cost"]),
        )
        set_cell(table, question_id, result)
        set_server_latency(
            lookup["server_latencies"][config],
            question_id,
            (
                None
                if row.get("server_latency_ms") is None
                else float(row["server_latency_ms"])
            ),
        )
        filled += 1

    remaining = remaining_missing(lookup)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
    temp.replace(output)
    return filled, remaining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(EXPECTED_SHA256), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--patch-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source.resolve() == args.output.resolve():
        raise RuntimeError("Output must differ from the source pickle")
    filled, remaining = merge(
        args.benchmark,
        args.source,
        args.patch_jsonl,
        args.output,
    )
    print(f"Filled {filled} cells; {remaining} cells remain missing.")
    print(f"Wrote new lookup: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
