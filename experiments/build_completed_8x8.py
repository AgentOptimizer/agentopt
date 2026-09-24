"""Build audited 8x8 lookup datasets from the historical runs and supplements.

The historical source pickles are never modified.  Only the eight-model
submatrix (excluding Claude 3 Haiku) is emitted, in both the original pickle
shape and a flat CSV that is convenient for inspection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from offline_selector_sim_v2 import SampleResult  # noqa: E402


MODELS_8 = [
    "Claude Haiku 4.5",
    "Claude Opus 4.6",
    "Kimi K2.5",
    "Ministral 3 8B",
    "Qwen3 32B",
    "Qwen3 Next 80B A3B",
    "gpt-oss-120b",
    "gpt-oss-20b",
]

EXPECTED_SOURCE_SHA256 = {
    "hotpotqa": "2e1260c986c4be0cf71fbaf31ce07d4b3fb2abbc6a599f09b246c4971d3ded8a",
    "mathqa": "10b569c8371356bfc41bdfa4e84173d5a3d4f1352340acd45f3473390c65b9be",
}

ROLE_NAMES = {
    "hotpotqa": ("planner", "solver"),
    "mathqa": ("answer", "critic"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(table: Any, question_id: int) -> bool:
    if isinstance(table, dict):
        return question_id not in table or table[question_id] is None
    return question_id >= len(table) or table[question_id] is None


def get_cell(table: Any, question_id: int) -> Any:
    if isinstance(table, dict):
        return table.get(question_id)
    if question_id >= len(table):
        return None
    return table[question_id]


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


def get_server_latency(values: Any, question_id: int) -> float | None:
    if isinstance(values, dict):
        return values.get(question_id)
    if question_id >= len(values):
        return None
    return values[question_id]


def load_latest_successes(
    path: Path, benchmark: str
) -> tuple[dict[tuple[str, int], dict[str, Any]], Counter[str]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if row.get("benchmark") != benchmark:
                continue
            statuses[str(row.get("status"))] += 1
            if row.get("status") != "ok":
                continue
            key = (str(row["configuration_id"]), int(row["question_id"]))
            latest[key] = row
    return latest, statuses


def expected_configurations(benchmark: str) -> list[str]:
    role1, role2 = ROLE_NAMES[benchmark]
    return [
        f"{role1}={model1} + {role2}={model2}"
        for model1 in MODELS_8
        for model2 in MODELS_8
    ]


def split_configuration(benchmark: str, configuration: str) -> tuple[str, str]:
    role1, role2 = ROLE_NAMES[benchmark]
    prefix1 = f"{role1}="
    marker = f" + {role2}="
    if not configuration.startswith(prefix1) or marker not in configuration:
        raise ValueError(f"Unexpected configuration name: {configuration}")
    first, second = configuration[len(prefix1) :].split(marker, 1)
    return first, second


def build_one(
    benchmark: str,
    source: Path,
    patch_jsonl: Path,
    output_pickle: Path,
    output_csv: Path,
) -> dict[str, Any]:
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256[benchmark]:
        raise RuntimeError(
            f"{benchmark} source hash mismatch: expected "
            f"{EXPECTED_SOURCE_SHA256[benchmark]}, got {source_hash}"
        )

    with source.open("rb") as handle:
        lookup = pickle.load(handle)

    successes, raw_statuses = load_latest_successes(patch_jsonl, benchmark)
    configs = expected_configurations(benchmark)
    source_configs = set(lookup["model_names"])
    absent = sorted(set(configs) - source_configs)
    if absent:
        raise RuntimeError(f"{benchmark} source is missing configurations: {absent}")

    subset = {
        "model_names": configs,
        "datapoints": copy.deepcopy(lookup["datapoints"]),
        "table": {config: copy.deepcopy(lookup["table"][config]) for config in configs},
        "server_latencies": {
            config: copy.deepcopy(lookup["server_latencies"][config])
            for config in configs
        },
    }

    filled = 0
    used_patch_keys: set[tuple[str, int]] = set()
    for key, row in successes.items():
        config, question_id = key
        if config not in subset["table"]:
            continue
        if not is_missing(subset["table"][config], question_id):
            raise RuntimeError(
                f"Refusing to overwrite historical cell: {config}, q={question_id}"
            )
        result = SampleResult(
            score=float(row["score"]),
            latency_seconds=float(row["latency_seconds"]),
            input_tokens={k: int(v) for k, v in row["input_tokens"].items()},
            output_tokens={k: int(v) for k, v in row["output_tokens"].items()},
            cost=float(row["cost"]),
        )
        set_cell(subset["table"][config], question_id, result)
        set_server_latency(
            subset["server_latencies"][config],
            question_id,
            None
            if row.get("server_latency_ms") is None
            else float(row["server_latency_ms"]),
        )
        used_patch_keys.add(key)
        filled += 1

    n_datapoints = len(subset["datapoints"])
    missing: list[tuple[str, int]] = []
    invalid_shapes: list[tuple[str, int]] = []
    missing_server_latency: list[tuple[str, int]] = []
    source_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for config in configs:
        model1, model2 = split_configuration(benchmark, config)
        for question_id in range(n_datapoints):
            sample = get_cell(subset["table"][config], question_id)
            if sample is None:
                missing.append((config, question_id))
                continue
            if set(vars(sample)) != {
                "score",
                "latency_seconds",
                "input_tokens",
                "output_tokens",
                "cost",
            }:
                invalid_shapes.append((config, question_id))
            server_latency = get_server_latency(
                subset["server_latencies"][config], question_id
            )
            if server_latency is None:
                missing_server_latency.append((config, question_id))
            key = (config, question_id)
            provenance = successes.get(key, {}) if key in used_patch_keys else {}
            source_kind = "supplement" if key in used_patch_keys else "historical"
            source_counts[source_kind] += 1
            records.append(
                {
                    "benchmark": benchmark,
                    "configuration_id": config,
                    ROLE_NAMES[benchmark][0] + "_model": model1,
                    ROLE_NAMES[benchmark][1] + "_model": model2,
                    "question_id": question_id,
                    "score": float(sample.score),
                    "latency_seconds": float(sample.latency_seconds),
                    "server_latency_ms": server_latency,
                    "input_tokens_total": sum(sample.input_tokens.values()),
                    "output_tokens_total": sum(sample.output_tokens.values()),
                    "input_tokens_by_model": json.dumps(
                        sample.input_tokens, sort_keys=True, separators=(",", ":")
                    ),
                    "output_tokens_by_model": json.dumps(
                        sample.output_tokens, sort_keys=True, separators=(",", ":")
                    ),
                    "cost": float(sample.cost),
                    "record_source": source_kind,
                    "model_call_count": provenance.get("model_call_count"),
                    "cached_call_count": provenance.get("cached_call_count"),
                    "max_concurrent": provenance.get("max_concurrent"),
                    "max_tool_rounds": provenance.get("max_tool_rounds"),
                }
            )

    expected_records = 64 * n_datapoints
    if missing:
        raise RuntimeError(
            f"{benchmark} 8x8 subset still has {len(missing)} missing cells; "
            f"first={missing[:5]}"
        )
    if invalid_shapes:
        raise RuntimeError(
            f"{benchmark} has {len(invalid_shapes)} incompatible SampleResult rows"
        )
    if len(records) != expected_records:
        raise RuntimeError(
            f"{benchmark} expected {expected_records} records, got {len(records)}"
        )

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    temp_pickle = output_pickle.with_suffix(output_pickle.suffix + ".tmp")
    with temp_pickle.open("wb") as handle:
        pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
    temp_pickle.replace(output_pickle)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
    temp_csv.replace(output_csv)

    capped_rows = [
        {
            "configuration_id": config,
            "question_id": question_id,
            "max_tool_rounds": successes[(config, question_id)].get("max_tool_rounds"),
        }
        for config, question_id in sorted(used_patch_keys)
        if successes[(config, question_id)].get("max_tool_rounds") is not None
    ]

    return {
        "benchmark": benchmark,
        "models": MODELS_8,
        "configurations": len(configs),
        "datapoints_per_configuration": n_datapoints,
        "records": len(records),
        "historical_records": source_counts["historical"],
        "supplemental_records": source_counts["supplement"],
        "raw_jsonl_status_rows": dict(raw_statuses),
        "missing_results": len(missing),
        "missing_server_latency": len(missing_server_latency),
        "capped_supplemental_rows": capped_rows,
        "source_pickle": str(source.relative_to(REPO_ROOT)),
        "source_sha256": source_hash,
        "patch_jsonl": str(patch_jsonl.relative_to(REPO_ROOT)),
        "patch_sha256": sha256(patch_jsonl),
        "output_pickle": str(output_pickle.relative_to(REPO_ROOT)),
        "output_pickle_sha256": sha256(output_pickle),
        "output_csv": str(output_csv.relative_to(REPO_ROOT)),
        "output_csv_sha256": sha256(output_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completed-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "missing_cells" / "completed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for benchmark in ("hotpotqa", "mathqa"):
        reports.append(
            build_one(
                benchmark=benchmark,
                source=(
                    REPO_ROOT
                    / "experiments"
                    / "results"
                    / "cache_db_results"
                    / f"{benchmark}_lookup.pkl"
                ),
                patch_jsonl=(
                    REPO_ROOT
                    / "experiments"
                    / "missing_cells"
                    / "output"
                    / f"{benchmark}.jsonl"
                ),
                output_pickle=args.completed_dir / f"{benchmark}_lookup.8x8.pkl",
                output_csv=args.completed_dir / f"{benchmark}_lookup.8x8.csv",
            )
        )

    manifest = {
        "schema_version": 1,
        "description": "Audited complete 8x8 model-combination datasets",
        "models": MODELS_8,
        "sample_result_fields": [
            "score",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "cost",
        ],
        "server_latency_unit": "milliseconds",
        "benchmarks": reports,
    }
    manifest_path = args.completed_dir / "manifest.8x8.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for report in reports:
        print(
            f"{report['benchmark']}: {report['configurations']} configurations x "
            f"{report['datapoints_per_configuration']} = {report['records']} records; "
            f"missing={report['missing_results']}"
        )
        print(f"  pickle: {report['output_pickle']}")
        print(f"  csv:    {report['output_csv']}")
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
