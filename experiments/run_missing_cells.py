"""Run only the missing HotpotQA/MathQA lookup-table cells.

The command is a dry-run unless --execute or --preflight-only is supplied.
Execution is fail-closed: it verifies the AWS account, region, application
inference profiles, and required DAPLab tags before invoking any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PLAN_PATH = REPO_ROOT / "experiments" / "missing_cells" / "plan.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "missing_cells" / "output"
EXPECTED_ACCOUNT = "920736616554"
EXPECTED_REGION = "us-east-1"


def account_from_profile_arns(arns: Iterable[str], *, region: str) -> str:
    """Validate profile ARN scope and return the single AWS account ID."""
    accounts: set[str] = set()
    for arn in arns:
        parts = arn.split(":", 5)
        if (
            len(parts) != 6
            or parts[0] != "arn"
            or parts[2] != "bedrock"
            or parts[3] != region
            or not parts[5].startswith("application-inference-profile/")
        ):
            raise RuntimeError(
                f"Expected a {region} application inference profile ARN; got {arn!r}"
            )
        accounts.add(parts[4])
    if len(accounts) != 1:
        raise RuntimeError(f"Profiles must belong to exactly one AWS account: {accounts}")
    return next(iter(accounts))


def load_plan(path: Path = PLAN_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = int(row["first_question_id"])
            end = int(row["last_question_id"])
            count = int(row["missing_count"])
            if end - start + 1 != count:
                raise ValueError(f"Non-contiguous or invalid plan row: {row}")
            rows.append({**row, "first_question_id": start, "last_question_id": end})
    return rows


def expand_plan(
    rows: Iterable[dict[str, Any]], benchmark: str
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row in rows:
        if benchmark != "all" and row["benchmark"] != benchmark:
            continue
        for question_id in range(
            row["first_question_id"], row["last_question_id"] + 1
        ):
            cells.append({**row, "question_id": question_id})
    keys = {
        (cell["benchmark"], cell["configuration_id"], cell["question_id"])
        for cell in cells
    }
    if len(keys) != len(cells):
        raise ValueError("Plan contains duplicate cells")
    return cells


def load_completed(output_dir: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    for path in output_dir.glob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                if row.get("status") == "ok":
                    completed.add(
                        (
                            row["benchmark"],
                            row["configuration_id"],
                            int(row["question_id"]),
                        )
                    )
    return completed


def preflight_aws(
    cells: list[dict[str, Any]],
    *,
    region: str,
    project_tag: str,
    billing_tag1: str,
) -> dict[str, Any]:
    if region != EXPECTED_REGION:
        raise RuntimeError(
            f"Region must be {EXPECTED_REGION!r}; received {region!r}"
        )
    if not project_tag or not billing_tag1:
        raise RuntimeError("--project-tag and --billing-tag1 are required")

    import boto3
    from benchmarks.common import _DISPLAY_NAME_TO_ARN

    model_names = sorted(
        {cell["role1_model"] for cell in cells}
        | {cell["role2_model"] for cell in cells}
    )
    missing_profiles = [name for name in model_names if name not in _DISPLAY_NAME_TO_ARN]
    if missing_profiles:
        raise RuntimeError(
            "Plan contains models without application inference profiles: "
            + ", ".join(missing_profiles)
        )

    profile_arns = [_DISPLAY_NAME_TO_ARN[name] for name in model_names]
    profile_account = account_from_profile_arns(profile_arns, region=region)

    session = boto3.Session(region_name=region)
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        # Bedrock short-term API keys intentionally cannot call STS.  The
        # profile ARN scope plus the tag checks below provide the equivalent
        # fail-closed account/resource verification for bearer-token runs.
        account = profile_account
    else:
        identity = session.client("sts").get_caller_identity()
        account = identity.get("Account")
    if account != EXPECTED_ACCOUNT:
        raise RuntimeError(
            f"Wrong AWS account: expected {EXPECTED_ACCOUNT}, got {account}"
        )

    expected_tags = {"project": project_tag, "billing-tag1": billing_tag1}
    bedrock = session.client("bedrock")
    checked_profiles: list[str] = []
    for model_name in model_names:
        arn = _DISPLAY_NAME_TO_ARN[model_name]
        response = bedrock.list_tags_for_resource(resourceARN=arn)
        actual = {
            item["key"]: item.get("value", "")
            for item in response.get("tags", [])
            if "key" in item
        }
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected_tags.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Tag preflight failed for {model_name} ({arn}): {mismatches}"
            )
        checked_profiles.append(arn)

    os.environ["AWS_DEFAULT_REGION"] = region
    return {
        "account": account,
        "region": region,
        "project": project_tag,
        "billing-tag1": billing_tag1,
        "profiles": checked_profiles,
    }


def compute_cost(
    input_tokens: dict[str, int], output_tokens: dict[str, int]
) -> float:
    from benchmarks.common import BEDROCK_PRICES

    total = 0.0
    for model in set(input_tokens) | set(output_tokens):
        price = BEDROCK_PRICES.get(model)
        if price is None:
            raise KeyError(f"No Bedrock price configured for {model}")
        total += input_tokens.get(model, 0) * price["input_price"] / 1_000_000
        total += output_tokens.get(model, 0) * price["output_price"] / 1_000_000
    return total


def is_context_length_exceeded(exc: Exception) -> bool:
    """Return True only for an explicit provider context-length rejection."""
    response = getattr(exc, "response", None)
    error_code = None
    if isinstance(response, dict):
        error_code = response.get("Error", {}).get("Code")
    if error_code not in (None, "ValidationException"):
        return False
    if error_code is None and type(exc).__name__ != "ValidationException":
        return False

    message = str(exc).lower()
    explicit_patterns = (
        "maximum context length",
        "context length exceeded",
        "context window exceeded",
        "input is too long for requested model",
        "prompt is too long",
        "too many input tokens",
    )
    return any(pattern in message for pattern in explicit_patterns)


def tracked_result_metrics(
    tracker: Any,
    *,
    data_id: str,
    wall_seconds: float,
) -> dict[str, Any]:
    """Collect exact metrics from successful responses recorded so far."""
    usage = tracker.get_usage(data_id=data_id)
    input_tokens = {model: pair[0] for model, pair in usage.items()}
    output_tokens = {model: pair[1] for model, pair in usage.items()}
    records = tracker.get_records(data_id=data_id)
    return {
        "latency_seconds": wall_seconds
        + tracker.get_cached_latency(data_id=data_id),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": compute_cost(input_tokens, output_tokens),
        "server_latency_ms": tracker.get_server_latency(data_id=data_id),
        "model_call_count": len(records),
        "cached_call_count": sum(record.cached for record in records),
    }


def context_overflow_result(
    base: dict[str, Any],
    *,
    exc: Exception,
    tracker: Any,
    data_id: str,
    wall_seconds: float,
) -> dict[str, Any]:
    """Convert a terminal context overflow into an incorrect scored result."""
    return {
        **base,
        "status": "ok",
        "score": 0.0,
        **tracked_result_metrics(
            tracker,
            data_id=data_id,
            wall_seconds=wall_seconds,
        ),
        "terminal_reason": "context_length_exceeded",
        "record_source": "context_overflow_adjudicated_incorrect",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "wall_seconds": wall_seconds,
        "rejected_call_count": 1,
        "usage_scope": "successful_responses_before_terminal_error",
    }


def load_dataset(benchmark: str) -> list[tuple[dict[str, Any], str]]:
    if benchmark == "hotpotqa":
        from benchmarks.HotpotQA.eval import load_hotpotqa_distractor

        return load_hotpotqa_distractor(
            str(
                REPO_ROOT
                / "benchmarks"
                / "HotpotQA"
                / "data"
                / "hotpot_dev_distractor_v1.json"
            ),
            limit=200,
            seed=0,
        )

    from benchmarks.MathQA.eval import load_math_qa

    selection, holdout = load_math_qa(train_split=1.0, max_samples=200)
    if holdout:
        raise RuntimeError("MathQA train_split=1.0 unexpectedly returned holdout rows")
    return selection


def build_agent(
    cell: dict[str, Any],
    *,
    max_tool_rounds: int | None = None,
    mathqa_max_iterations: int = 3,
):
    if cell["benchmark"] == "hotpotqa":
        from benchmarks.HotpotQA.eval import _hotpotqa_agent_fn_langgraph

        return _hotpotqa_agent_fn_langgraph(
            {
                "planner": cell["role1_model"],
                "solver": cell["role2_model"],
            },
            pipeline="multi",
            max_reflections=1,
        )

    from benchmarks.MathQA.eval import _mathqa_agent_fn_langgraph

    return _mathqa_agent_fn_langgraph(
        {
            "answer": cell["role1_model"],
            "critic": cell["role2_model"],
        },
        max_iterations=mathqa_max_iterations,
        max_tool_rounds=max_tool_rounds,
    )


def score_result(benchmark: str, expected: str, actual: Any) -> float:
    if benchmark == "hotpotqa":
        from benchmarks.HotpotQA.eval import _extract_text, hotpot_f1

        return float(hotpot_f1(expected, _extract_text(actual)))

    from benchmarks.MathQA.eval import eval_fn

    return float(eval_fn(expected, actual))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_cells(
    cells: list[dict[str, Any]],
    *,
    output_dir: Path,
    max_concurrent: int,
    aws_context: dict[str, Any],
    use_cache: bool = True,
    max_tool_rounds: int | None = None,
    mathqa_max_iterations: int = 3,
) -> None:
    from agentopt import LLMTracker

    datasets = {
        benchmark: load_dataset(benchmark)
        for benchmark in sorted({cell["benchmark"] for cell in cells})
    }
    for benchmark, dataset in datasets.items():
        if len(dataset) != 200:
            raise RuntimeError(f"{benchmark} dataset has {len(dataset)} rows, expected 200")

    cache_dir = REPO_ROOT / ".agentopt_cache" / "missing-cells"
    tracker = LLMTracker(
        cache=use_cache,
        cache_dir=cache_dir if use_cache else None,
    )
    tracker.start()
    try:
        total = len(cells)
        completed_count = 0

        def evaluate_cell(
            cell: dict[str, Any],
            agent: Any,
        ) -> dict[str, Any]:
            benchmark = cell["benchmark"]
            combo_id = cell["configuration_id"]
            question_id = int(cell["question_id"])
            # Match BaseModelSelector._evaluate_agent_async attribution.
            data_id = f"{combo_id}::dp_{question_id + 1}"
            started = time.time()
            base = {
                "benchmark": benchmark,
                "configuration_id": combo_id,
                "question_id": question_id,
                "role1_model": cell["role1_model"],
                "role2_model": cell["role2_model"],
                "aws_account": aws_context["account"],
                "aws_region": aws_context["region"],
                "project_tag": aws_context["project"],
                "billing_tag1": aws_context["billing-tag1"],
                "max_concurrent": max_concurrent,
                "max_tool_rounds": max_tool_rounds,
                "mathqa_max_iterations": (
                    mathqa_max_iterations if benchmark == "mathqa" else None
                ),
            }
            try:
                input_data, expected = datasets[benchmark][question_id]
                with tracker.track(data_id=data_id, combo_id=combo_id):
                    actual = agent(input_data)
                    wall_seconds = time.time() - started
                metrics = tracked_result_metrics(
                    tracker,
                    data_id=data_id,
                    wall_seconds=wall_seconds,
                )
                return {
                    **base,
                    "status": "ok",
                    "score": score_result(benchmark, expected, actual),
                    **metrics,
                }
            except Exception as exc:
                wall_seconds = time.time() - started
                if is_context_length_exceeded(exc):
                    return context_overflow_result(
                        base,
                        exc=exc,
                        tracker=tracker,
                        data_id=data_id,
                        wall_seconds=wall_seconds,
                    )
                return {
                    **base,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_seconds": wall_seconds,
                }

        key_fn = lambda cell: (cell["benchmark"], cell["configuration_id"])
        ordered_cells = sorted(cells, key=key_fn)
        for _, grouped in groupby(ordered_cells, key=key_fn):
            config_cells = list(grouped)
            agent = build_agent(
                config_cells[0],
                max_tool_rounds=max_tool_rounds,
                mathqa_max_iterations=mathqa_max_iterations,
            )
            workers = min(max_concurrent, len(config_cells))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluate_cell, cell, agent): cell
                    for cell in config_cells
                }
                for future in as_completed(futures):
                    cell = futures[future]
                    row = future.result()
                    output_path = output_dir / f"{cell['benchmark']}.jsonl"
                    append_jsonl(output_path, row)
                    completed_count += 1
                    status = row["status"]
                    if row.get("terminal_reason"):
                        status += f" ({row['terminal_reason']})"
                    print(
                        f"[{completed_count}/{total}] {cell['benchmark']} "
                        f"{cell['configuration_id']} q={cell['question_id']}: "
                        f"{status}"
                    )
            tracker.flush_cache()
    finally:
        tracker.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("all", "hotpotqa", "mathqa"),
        default="all",
    )
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--profile-map",
        type=Path,
        help="JSON display-name to application-inference-profile ARN overrides",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the LLM response cache for this run",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help=(
            "Optional per-answer cap on MathQA tool-use rounds; disabled by "
            "default to preserve the original experiment protocol"
        ),
    )
    parser.add_argument(
        "--mathqa-max-iterations",
        type=int,
        default=3,
        help=(
            "Maximum answer-critic iterations for MathQA; defaults to the "
            "original experiment value of 3"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate AWS account/profile tags without invoking a model",
    )
    parser.add_argument("--region", default=EXPECTED_REGION)
    parser.add_argument("--project-tag")
    parser.add_argument("--billing-tag1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile_map:
        os.environ["AGENTOPT_BEDROCK_PROFILE_MAP"] = str(args.profile_map.resolve())
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    if args.max_tool_rounds is not None and args.max_tool_rounds < 1:
        raise ValueError("--max-tool-rounds must be at least 1")
    if args.mathqa_max_iterations < 1:
        raise ValueError("--mathqa-max-iterations must be at least 1")
    rows = load_plan(args.plan)
    cells = expand_plan(rows, args.benchmark)
    completed = load_completed(args.output_dir) if args.output_dir.exists() else set()
    unfinished = [
        cell
        for cell in cells
        if (
            cell["benchmark"],
            cell["configuration_id"],
            int(cell["question_id"]),
        )
        not in completed
    ]
    pending = unfinished
    if args.limit is not None:
        pending = pending[: max(0, args.limit)]

    counts = Counter(cell["benchmark"] for cell in cells)
    pending_counts = Counter(cell["benchmark"] for cell in pending)
    print(f"Plan: {len(cells)} cells {dict(sorted(counts.items()))}")
    print(f"Already complete: {len(cells) - len(unfinished)}")
    print(f"Pending this invocation: {len(pending)} {dict(sorted(pending_counts.items()))}")
    print(f"Per-configuration datapoint concurrency: {args.max_concurrent}")
    if not args.execute and not args.preflight_only:
        print("Dry-run only. No AWS API or model calls were made.")
        return 0
    if not pending:
        print("Nothing to execute.")
        return 0

    aws_context = preflight_aws(
        pending,
        region=args.region,
        project_tag=args.project_tag or "",
        billing_tag1=args.billing_tag1 or "",
    )
    print(
        f"AWS preflight passed for account {aws_context['account']} "
        f"and {len(aws_context['profiles'])} profiles."
    )
    if args.preflight_only:
        print("Preflight only. No model calls were made.")
        return 0
    run_cells(
        pending,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        aws_context=aws_context,
        use_cache=not args.no_cache,
        max_tool_rounds=args.max_tool_rounds,
        mathqa_max_iterations=args.mathqa_max_iterations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
