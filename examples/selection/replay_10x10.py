"""Replay recorded QA and SCOPE matrices through AgentOpt without API calls.

The score and USD cost of each configuration-question cell come from committed
research matrices. Each cell waits 0.1 second by default; this delay is
simulated and is also the measured latency in the result. Replay bookkeeping
tokens are synthetic. A separate checkpoint stores each completed evaluation cell.
"""

from __future__ import annotations

import argparse
import contextvars
import csv
import hashlib
import json
import math
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from agentopt.model_selection.gittins import GittinsModelSelector
from agentopt.visualization.live_pareto import LiveParetoPlot
from benchmark_runner import save_json, show_progress


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "agentopt-research" / "data"
SCOPE = ("bird_mini_dev", "bird_dev", "bing_querylogs", "restaurant_test", "restaurant_valid", "stackoverflow")
LEDGER_MODEL = "__historical_cell_cost__"
# The package prices one million tokens. These units encode the recorded USD
# cost exactly enough for replay; they are not observed input token counts.
LEDGER_UNITS_PER_USD = 10**12
LEDGER_PRICE_PER_MILLION = 10**-6


def read_matrix(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][0] != "model_name":
        raise ValueError(f"Expected model_name header: {path}")
    questions = rows[0][1:]
    if questions != [f"question_{i}" for i in range(len(questions))]:
        raise ValueError(f"Question columns are not sequential: {path}")
    values = {}
    for line, row in enumerate(rows[1:], 2):
        if len(row) != len(rows[0]) or row[0] in values:
            raise ValueError(f"Invalid or duplicate configuration at {path}:{line}")
        numbers = [float(value) for value in row[1:]]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"Non-finite value at {path}:{line}")
        values[row[0]] = numbers
    return questions, values


def load_benchmark(data_root: Path, benchmark: str, max_configs=None):
    directory = data_root / "scope" / benchmark if benchmark in SCOPE else data_root / benchmark
    score_questions, scores = read_matrix(directory / "accuracy_matrix.csv")
    cost_questions, costs = read_matrix(directory / "cost_matrix_usd.csv")
    if score_questions != cost_questions or list(scores) != list(costs):
        raise ValueError("Score and cost matrices do not align")
    if (any(value < 0 or value > 1 for row in scores.values() for value in row)
            or any(value < 0 for row in costs.values() for value in row)):
        raise ValueError("Scores must be in [0, 1]; costs must be nonnegative")

    if benchmark in SCOPE:
        records = json.loads((directory / "configs.json").read_text(encoding="utf-8"))
        assignments = {row["model_name"]: row["config"] for row in records}
        if len(assignments) != len(records) or assignments.keys() != scores.keys() or not scores or not score_questions:
            raise ValueError("SCOPE configuration IDs must match the matrix rows")
        roles = tuple(assignments[next(iter(scores))])
        if not roles or any(mapping.keys() != set(roles) or
                            not all(isinstance(value, str) and value for value in mapping.values())
                            for mapping in assignments.values()):
            raise ValueError("Invalid role assignments in configs.json")
        labels = list(scores)
        if max_configs is not None and max_configs < len(labels):
            indices = sorted(random.Random(42).sample(range(len(labels)), max_configs))
            labels = [labels[i] for i in indices]
        selected = {label: {role: assignments[label][role] for role in roles} for label in labels}
        return directory, roles, {"configuration": labels}, selected, scores, costs, len(score_questions)

    if len(scores) != 100 or len(score_questions) != 200:
        raise ValueError("Expected 100 configurations and 200 questions")
    names_by_node = {}
    for label in scores:
        parts = label.split(" + ")
        if len(parts) != 2 or any("=" not in part for part in parts):
            raise ValueError(f"Cannot parse configuration: {label}")
        for part in parts:
            node, model = part.split("=", 1)
            names_by_node.setdefault(node, [])
            if model not in names_by_node[node]:
                names_by_node[node].append(model)
    if len(names_by_node) != 2 or any(len(names) != 10 for names in names_by_node.values()):
        raise ValueError("Expected two model nodes with ten candidates each")
    nodes = tuple(names_by_node)
    expected_labels = {f"{nodes[0]}={first} + {nodes[1]}={second}"
                       for first in names_by_node[nodes[0]]
                       for second in names_by_node[nodes[1]]}
    if set(scores) != expected_labels:
        raise ValueError("Matrix rows are not the complete 10x10 candidate grid")
    return directory, nodes, names_by_node, None, scores, costs, len(score_questions)


class MatrixTracker:
    """Feed historical per-cell USD to the selector during replay."""

    def __init__(self):
        self.records = []
        self.active = contextvars.ContextVar("matrix_replay_evaluation")

    def start(self):
        pass

    def stop(self):
        pass

    @contextmanager
    def track(self, data_id, combo_id):
        token = self.active.set((data_id, combo_id))
        try:
            yield SimpleNamespace()
        finally:
            self.active.reset(token)

    def record_cell(self, cost_usd):
        data_id, combo_id = self.active.get()
        self.records.append(SimpleNamespace(
            model=LEDGER_MODEL,
            prompt_tokens=round(cost_usd * LEDGER_UNITS_PER_USD),
            completion_tokens=0,
            data_id=data_id,
            combo_id=combo_id,
            cached=False,
        ))

    def get_records(self):
        return list(self.records)

    def get_usage(self, data_id=None, combo_id=None):
        units = sum(record.prompt_tokens for record in self.records
                    if (data_id is None or record.data_id == data_id)
                    and (combo_id is None or record.combo_id == combo_id))
        return {LEDGER_MODEL: (units, 0)} if units else {}

    def get_cached_latency(self, data_id):
        return 0.0


class ResumableMatrixSelector(GittinsModelSelector):
    """Use the product checkpoint implementation with a matrix identity."""

    def __init__(self, *, checkpoint, identity, **kwargs):
        super().__init__(checkpoint_path=checkpoint, checkpoint_id=identity, **kwargs)


class ScopeMatrixSelector(ResumableMatrixSelector):
    """Count one arm per recorded row and expose its actual role assignments."""

    def __init__(self, *, assignments, **kwargs):
        self.assignments = assignments
        super().__init__(**kwargs)

    def _generate_combos(self):
        for label in self._models["configuration"]:
            yield self.assignments[label]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("hotpotqa", "mathqa", *SCOPE), default="hotpotqa")
    parser.add_argument("--max-configs", type=int,
                        help="preview a reproducible subset of SCOPE configurations (default: all rows)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="optional automatic stop after this many completed cells")
    parser.add_argument("--delay-seconds", type=float, default=0.1,
                        help="simulated API time per cell (default: 0.1 second)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="maximum concurrent evaluations, as in benchmark_runner.py (default: 4)")
    parser.add_argument("--grid-size", type=int, default=129,
                        help="Gittins DP grid size (default: 129)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path,
                        help="resume from this SQLite checkpoint, or create it if absent; defaults to the output path plus .checkpoint.sqlite3")
    parser.add_argument("--live-plot", action="store_true",
                        help="show an updating estimated Pareto plot in a Matplotlib window")
    parser.add_argument("--plot-output", type=Path, default=None,
                        help="save the final Pareto plot as an image (also used by --live-plot)")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if not math.isfinite(args.delay_seconds) or args.delay_seconds < 0:
        parser.error("--delay-seconds must be nonnegative")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be positive")
    if args.max_configs is not None and (args.benchmark not in SCOPE or args.max_configs < 1):
        parser.error("--max-configs requires a SCOPE benchmark and a positive count")
    directory, nodes, candidates, assignments, scores, costs, questions = load_benchmark(
        args.data_root, args.benchmark, args.max_configs)
    combinations = len(assignments) if assignments is not None else len(scores)
    row_by_combo = {}
    if assignments is not None:
        row_by_combo = {tuple(mapping[role] for role in nodes): label for label, mapping in assignments.items()}
        if len(row_by_combo) != combinations:
            parser.error("multiple recorded rows have identical role assignments")
    tracker = MatrixTracker()
    stem = f"{args.benchmark}-10x10" if assignments is None else f"{args.benchmark}-{combinations}configs"
    output = args.output or Path(f"{stem}-replay.json")
    checkpoint = args.checkpoint or output.with_suffix(output.suffix + ".checkpoint.sqlite3")
    plot_output = args.plot_output or Path(f"{args.benchmark}-pareto.png")
    if checkpoint.resolve() in {output.resolve(), plot_output.resolve()}:
        parser.error("--checkpoint must differ from result and image paths")
    fingerprint = hashlib.sha256()
    files = ["accuracy_matrix.csv", "cost_matrix_usd.csv"]
    if assignments is not None:
        files.append("configs.json")
    for name in files:
        fingerprint.update(hashlib.sha256((directory / name).read_bytes()).digest())
    identity = json.dumps({"version": 1, "benchmark": args.benchmark, "matrix": fingerprint.hexdigest(),
                           "grid_size": args.grid_size, "seed": 42, "warm_size": 4, "batch_size": 4},
                          sort_keys=True)

    class MatrixAgent:
        def __init__(self, models):
            if assignments is None:
                self.label = " + ".join(f"{role}={models[role]}" for role in nodes)
            else:
                self.label = row_by_combo[tuple(models[role] for role in nodes)]

        def run(self, question_id):
            time.sleep(args.delay_seconds)
            tracker.record_cell(costs[self.label][question_id])
            return scores[self.label][question_id]

    selector_type = ResumableMatrixSelector if assignments is None else ScopeMatrixSelector
    selector = selector_type(
        **({} if assignments is None else {"assignments": assignments}),
        checkpoint=checkpoint, identity=identity,
        agent=MatrixAgent,
        models=candidates,
        dataset=[(question_id, None) for question_id in range(questions)],
        eval_fn=lambda expected, actual: actual,
        objective_mode="multi",
        warm_start_batch_size=4,
        batch_size=4,
        grid_size=args.grid_size,
        tracker=tracker,
        model_prices={LEDGER_MODEL: {"input_price": LEDGER_PRICE_PER_MILLION,
                                     "output_price": 0.0}},
    )
    stop = Event()
    plot = None
    if args.live_plot or args.plot_output is not None:
        plot = LiveParetoPlot(
            title=f"{args.benchmark.upper()} · {combinations:,} of {len(scores):,} configurations" if combinations < len(scores)
                  else f"{args.benchmark.upper()} · Offline replay",
            warm_total=combinations * min(4, questions),
            total_questions=questions,
            total_combinations=combinations,
            output=plot_output,
            show_window=args.live_plot,
            stop_event=stop,
        )

    def update(snapshot):
        count = snapshot.total_evaluations
        show_progress(snapshot, total_questions=questions)
        if (args.stop_after is not None and not snapshot.is_final
                and count >= args.stop_after):
            stop.set()
            if plot is not None:
                plot.finish_search(None)
            return False
        return None

    def request_stop(signum, frame):
        if stop.is_set():
            return
        stop.set()
        if plot is not None:
            plot.finish_search(None)
        print("\nStop requested. Waiting for the current batch to finish…", flush=True)

    print(f"Offline replay: {args.benchmark}, {combinations}/{len(scores)} recorded configurations, {questions} questions; "
          f"{args.delay_seconds:g} simulated seconds per cell; no model API calls.", flush=True)
    print("Each evaluation runs one model configuration on one test question and scores its output.", flush=True)
    print("Press Ctrl+C to stop after the scheduled batch; its completed results will be saved.", flush=True)
    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        if plot is not None:
            results = plot.run(selector, max_concurrent=args.max_concurrent, on_update=update)
        else:
            results = selector.select_best(parallel=True, max_concurrent=args.max_concurrent,
                                           on_update=update, stop_event=stop)
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    save_json(output, results.model_dump(mode="json"))
    print(f"Saved results to {output}", flush=True)
    print(f"Checkpoint: {checkpoint} ({selector.restored_cells:,} restored, "
          f"{len(tracker.records):,} newly evaluated)", flush=True)
    if plot is not None:
        print(f"Saved Pareto plot to {plot.output}", flush=True)
        if args.live_plot:
            show_saved_results = getattr(plot, "show_saved_results", None)
            if show_saved_results is not None:
                show_saved_results(output)
            plot.wait_for_close()


if __name__ == "__main__":
    main()
