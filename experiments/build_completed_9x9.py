"""Build audited 9x9 lookup datasets from the complete 8x8 base and DeepSeek runs.

The historical and completed 8x8 source files are never modified.  The output
keeps the exact lookup-pickle schema used by the experiment algorithms and also
emits a flat CSV for inspection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from experiments.build_completed_8x8 import (
        MODELS_8,
        REPO_ROOT,
        ROLE_NAMES,
        SampleResult,
        get_cell,
        get_server_latency,
        is_missing,
        load_latest_successes,
        set_cell,
        set_server_latency,
        sha256,
        split_configuration,
    )
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from build_completed_8x8 import (  # type: ignore[no-redef]
        MODELS_8,
        REPO_ROOT,
        ROLE_NAMES,
        SampleResult,
        get_cell,
        get_server_latency,
        is_missing,
        load_latest_successes,
        set_cell,
        set_server_latency,
        sha256,
        split_configuration,
    )


MODELS_9 = [*MODELS_8, "DeepSeek R1"]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def expected_configurations(benchmark: str) -> list[str]:
    role1, role2 = ROLE_NAMES[benchmark]
    return [
        f"{role1}={model1} + {role2}={model2}"
        for model1 in MODELS_9
        for model2 in MODELS_9
    ]


def load_base_report(manifest_path: Path, benchmark: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = {
        report["benchmark"]: report for report in manifest.get("benchmarks", [])
    }
    if benchmark not in reports:
        raise RuntimeError(f"8x8 manifest has no {benchmark!r} report")
    return reports[benchmark]


def load_base_csv(path: Path) -> tuple[list[str], dict[tuple[str, int], dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        records: dict[tuple[str, int], dict[str, str]] = {}
        for row in reader:
            key = (row["configuration_id"], int(row["question_id"]))
            if key in records:
                raise RuntimeError(f"Duplicate 8x8 CSV row: {key}")
            records[key] = row
    return fieldnames, records


def load_adjudications(
    path: Path, benchmark: str
) -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload.get("records", []):
        if row.get("benchmark") != benchmark:
            continue
        if row.get("status") != "adjudicated_incorrect":
            raise RuntimeError(f"Unexpected adjudication status in {path}: {row}")
        if float(row.get("score", 1.0)) != 0.0:
            raise RuntimeError(f"Adjudicated failure must have score 0: {row}")
        key = (str(row["configuration_id"]), int(row["question_id"]))
        if key in records:
            raise RuntimeError(f"Duplicate adjudication row: {key}")
        records[key] = row
    return records


def build_one(
    benchmark: str,
    base_pickle: Path,
    base_csv: Path,
    base_manifest: Path,
    patch_jsonl: Path,
    adjudications_path: Path,
    output_pickle: Path,
    output_csv: Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    base_report = load_base_report(base_manifest, benchmark)
    if sha256(base_pickle) != base_report["output_pickle_sha256"]:
        raise RuntimeError(f"{benchmark} 8x8 pickle does not match its manifest")
    if sha256(base_csv) != base_report["output_csv_sha256"]:
        raise RuntimeError(f"{benchmark} 8x8 CSV does not match its manifest")

    with base_pickle.open("rb") as handle:
        base_lookup = pickle.load(handle)

    configs = expected_configurations(benchmark)
    expected_base_configs = {
        config
        for config in configs
        if "DeepSeek R1" not in split_configuration(benchmark, config)
    }
    if set(base_lookup["model_names"]) != expected_base_configs:
        raise RuntimeError(f"{benchmark} 8x8 base configuration set is unexpected")

    n_datapoints = len(base_lookup["datapoints"])
    subset = {
        "model_names": configs,
        "datapoints": copy.deepcopy(base_lookup["datapoints"]),
        "table": {},
        "server_latencies": {},
    }
    for config in configs:
        if config in base_lookup["table"]:
            subset["table"][config] = copy.deepcopy(base_lookup["table"][config])
            subset["server_latencies"][config] = copy.deepcopy(
                base_lookup["server_latencies"][config]
            )
        else:
            subset["table"][config] = {}
            subset["server_latencies"][config] = [None] * n_datapoints

    successes, raw_statuses = load_latest_successes(patch_jsonl, benchmark)
    adjudications = load_adjudications(adjudications_path, benchmark)
    resolved_rows = dict(adjudications)
    resolved_rows.update(successes)
    deepseek_configs = set(configs) - expected_base_configs
    used_patch_keys: set[tuple[str, int]] = set()
    for key, row in resolved_rows.items():
        config, question_id = key
        if config not in deepseek_configs:
            continue
        if not is_missing(subset["table"][config], question_id):
            raise RuntimeError(f"Refusing to overwrite 9x9 cell: {config}, q={question_id}")
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

    fieldnames, base_csv_records = load_base_csv(base_csv)
    expected_fields = [
        "benchmark",
        "configuration_id",
        ROLE_NAMES[benchmark][0] + "_model",
        ROLE_NAMES[benchmark][1] + "_model",
        "question_id",
        "score",
        "latency_seconds",
        "server_latency_ms",
        "input_tokens_total",
        "output_tokens_total",
        "input_tokens_by_model",
        "output_tokens_by_model",
        "cost",
        "record_source",
        "model_call_count",
        "cached_call_count",
        "max_concurrent",
        "max_tool_rounds",
    ]
    if fieldnames != expected_fields:
        raise RuntimeError(f"Unexpected 8x8 CSV schema for {benchmark}: {fieldnames}")

    missing: list[tuple[str, int]] = []
    invalid_shapes: list[tuple[str, int]] = []
    missing_server_latency: list[tuple[str, int]] = []
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
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
            if config in expected_base_configs:
                if key not in base_csv_records:
                    raise RuntimeError(f"Missing 8x8 CSV base row: {key}")
                records.append(base_csv_records[key])
                source_counts["8x8_base"] += 1
                continue

            provenance = resolved_rows[key]
            record_source = provenance.get("record_source", "deepseek_supplement")
            source_counts[record_source] += 1
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
                    "record_source": record_source,
                    "model_call_count": provenance.get("model_call_count"),
                    "cached_call_count": provenance.get("cached_call_count"),
                    "max_concurrent": provenance.get("max_concurrent"),
                    "max_tool_rounds": provenance.get("max_tool_rounds"),
                }
            )

    expected_records = 81 * n_datapoints
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"{benchmark} 9x9 dataset still has {len(missing)} missing cells; "
            f"first={missing[:10]}"
        )
    if invalid_shapes:
        raise RuntimeError(
            f"{benchmark} has {len(invalid_shapes)} incompatible SampleResult rows"
        )
    if missing_server_latency and not allow_incomplete:
        raise RuntimeError(
            f"{benchmark} has {len(missing_server_latency)} rows without server "
            f"latency; first={missing_server_latency[:10]}"
        )
    if len(records) != expected_records - len(missing):
        raise RuntimeError(
            f"{benchmark} expected {expected_records - len(missing)} resolved records, "
            f"got {len(records)}"
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
    temp_csv.replace(output_csv)

    capped_rows = [
        {
            "configuration_id": config,
            "question_id": question_id,
            "max_tool_rounds": resolved_rows[(config, question_id)].get(
                "max_tool_rounds"
            ),
        }
        for config, question_id in sorted(used_patch_keys)
        if resolved_rows[(config, question_id)].get("max_tool_rounds") is not None
    ]
    limited_reflection_rows = [
        {
            "configuration_id": config,
            "question_id": question_id,
            "mathqa_max_iterations": resolved_rows[(config, question_id)].get(
                "mathqa_max_iterations"
            ),
        }
        for config, question_id in sorted(used_patch_keys)
        if resolved_rows[(config, question_id)].get("mathqa_max_iterations")
        not in (None, 3)
    ]

    return {
        "benchmark": benchmark,
        "models": MODELS_9,
        "configurations": len(configs),
        "datapoints_per_configuration": n_datapoints,
        "records": len(records),
        "expected_records": expected_records,
        "complete": not missing,
        "base_8x8_records": source_counts["8x8_base"],
        "deepseek_supplemental_records": source_counts["deepseek_supplement"],
        "adjudicated_incorrect_records": source_counts[
            "context_overflow_adjudicated_incorrect"
        ],
        "raw_jsonl_status_rows": dict(raw_statuses),
        "missing_results": len(missing),
        "missing_result_rows": [
            {"configuration_id": config, "question_id": question_id}
            for config, question_id in missing
        ],
        "missing_server_latency": len(missing_server_latency),
        "capped_supplemental_rows": capped_rows,
        "limited_reflection_rows": limited_reflection_rows,
        "base_pickle": display_path(base_pickle),
        "base_pickle_sha256": sha256(base_pickle),
        "base_csv": display_path(base_csv),
        "base_csv_sha256": sha256(base_csv),
        "patch_jsonl": display_path(patch_jsonl),
        "patch_sha256": sha256(patch_jsonl),
        "adjudications": display_path(adjudications_path),
        "adjudications_sha256": sha256(adjudications_path),
        "output_pickle": display_path(output_pickle),
        "output_pickle_sha256": sha256(output_pickle),
        "output_csv": display_path(output_csv),
        "output_csv_sha256": sha256(output_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completed-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "missing_cells" / "completed",
    )
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments"
            / "missing_cells"
            / "adjudicated_failures.9x9.json"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write separately named partial artifacts while retaining unresolved "
            "cells as None. Strict final builds remain the default."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = REPO_ROOT / "experiments" / "missing_cells" / "completed"
    output_label = "9x9.partial" if args.allow_incomplete else "9x9"
    reports = []
    for benchmark in ("hotpotqa", "mathqa"):
        reports.append(
            build_one(
                benchmark=benchmark,
                base_pickle=base_dir / f"{benchmark}_lookup.8x8.pkl",
                base_csv=base_dir / f"{benchmark}_lookup.8x8.csv",
                base_manifest=base_dir / "manifest.8x8.json",
                patch_jsonl=(
                    REPO_ROOT
                    / "experiments"
                    / "missing_cells"
                    / "output"
                    / f"{benchmark}.jsonl"
                ),
                adjudications_path=args.adjudications,
                output_pickle=(
                    args.completed_dir / f"{benchmark}_lookup.{output_label}.pkl"
                ),
                output_csv=args.completed_dir / f"{benchmark}_lookup.{output_label}.csv",
                allow_incomplete=args.allow_incomplete,
            )
        )

    manifest = {
        "schema_version": 1,
        "description": (
            "Audited complete 9x9 model-combination datasets"
            if not args.allow_incomplete
            else "Audited partial 9x9 datasets with unresolved cells retained as None"
        ),
        "complete": all(report["complete"] for report in reports),
        "models": MODELS_9,
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
    manifest_path = args.completed_dir / f"manifest.{output_label}.json"
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
    print(f"Manifest: {display_path(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
