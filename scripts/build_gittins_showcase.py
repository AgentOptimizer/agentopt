"""Reproduce the documentation's frozen-matrix Gittins example; no API calls.

Run from a checkout with numpy, scipy and matplotlib installed:
    python scripts/build_gittins_showcase.py

Only evaluation, charge accounting and snapshot export are adapted for replay.
Acquisition, priors, budget checks and recommendations use the package selector.
Full-matrix means are used only after selection, to evaluate and plot its output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentopt.model_selection.base import DatapointResult
from agentopt.model_selection.gittins import GittinsModelSelector

ASSETS = ROOT / "docs/assets/benchmarks/gittins-10x10"
SEED = 42
BUDGET_FRACTION = 0.1


class ReplayTracker:
    """No network, HTTP interception or response cache is created for replay."""

    def start(self):
        pass

    def stop(self):
        pass

    def get_records(self):
        return []


def read_matrix(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert header == ["model_name", *[f"question_{i}" for i in range(200)]]
    names = [row[0] for row in rows]
    values = np.asarray([row[1:] for row in rows], dtype=float)
    assert values.shape == (100, 200) and len(set(names)) == 100
    assert np.isfinite(values).all()
    return names, values


class MatrixReplay(GittinsModelSelector):
    def __init__(self, names, scores, costs, mode):
        self.matrix_names, self.matrix_scores, self.matrix_costs = names, scores, costs
        self.matrix_index = {name: i for i, name in enumerate(names)}
        self.history = []
        models = {}
        for name in names:
            for part in name.split(" + "):
                role, model = part.split("=", 1)
                candidates = models.setdefault(role, [])
                if model not in candidates:
                    candidates.append(model)
        assert len(models) == 2 and all(len(values) == 10 for values in models.values())
        super().__init__(
            agent=lambda models: None, models=models, eval_fn=lambda expected, actual: actual,
            dataset=[(i, None) for i in range(200)], tracker=ReplayTracker(),
            objective_mode=mode, seed=SEED, warm_start_batch_size=4, batch_size=4,
            max_search_cost_usd=float(costs.sum()) * BUDGET_FRACTION,
        )

    async def _pull(self, arm, question_ids, parallel, max_concurrent):
        index = self.matrix_index[self._names[arm]]
        charge = 0.0
        for question in question_ids:
            score = float(self.matrix_scores[index, question])
            cost = float(self.matrix_costs[index, question])
            assert 0 <= score <= 1 and cost >= 0
            self._scores[arm].append(score)
            self._costs[arm].append(cost)
            # Latency is not an objective in this showcase. Do not invent timings.
            self._latencies[arm].append(0.0)
            self._datapoints[arm].append(DatapointResult(
                datapoint_index=question, score=score, latency_seconds=0.0,
            ))
            charge += cost
        self._spent += charge
        self._evaluations += len(question_ids)
        self._positions[arm] += len(question_ids)
        return charge

    def _selection_cost_so_far(self):
        return getattr(self, "_spent", 0.0)

    def _publish_update(self, result, *, final=False):
        # Retain recommendation snapshots without expensive HTTP-record copying.
        # This changes presentation only, not any selection or stopping decision.
        result.is_final = final
        result.selection_cost = self._spent
        result.total_evaluations = self._evaluations
        recommendations = [candidate.model_dump() for candidate in result.recommendations]
        entry = dict(search_cost_usd=self._spent, evaluations=self._evaluations,
                     stop_reason=result.stop_reason, is_final=final,
                     recommendations=recommendations)
        if recommendations:
            self.history.append(entry)
        self._latest_results = result
        return result


def frontier_indices(scores, costs):
    return [i for i in range(len(scores)) if not any(
        scores[j] >= scores[i] and costs[j] <= costs[i]
        and (scores[j] > scores[i] or costs[j] < costs[i])
        for j in range(len(scores))
    )]


def chart_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "text.color": "#0f172a", "xtick.color": "#64748b", "ytick.color": "#64748b",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "agentopt-gittins-showcase",
    })


def save_figure(fig, name):
    for suffix in ("svg", "png"):
        path = ASSETS / f"{name}.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight",
                    metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


def make_figures(benchmark, scores, costs, single, multi, names):
    benchmark_title = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}[benchmark]
    mean_scores, mean_costs = scores.mean(axis=1), costs.mean(axis=1)
    exhaustive = float(costs.sum())
    index = {name: i for i, name in enumerate(names)}
    fig, ax = plt.subplots(figsize=(8.2, 4.3), layout="constrained")
    x = [row["search_cost_usd"] / exhaustive * 100 for row in single.history]
    y = [mean_scores[index[row["recommendations"][0]["model_name"]]] * 100
         for row in single.history]
    ax.step(x, y, where="post", color="#7864d9", linewidth=2.3,
            label="SDK Gittins recommendation (seed 42)")
    ax.axhline(float(mean_scores.max()) * 100, color="#209a80", linestyle="--",
               linewidth=1.5, label="Best full-matrix score (reference only)")
    ax.axvspan(0, x[0], color="#eef2f6", label="Initial sampling; no recommendation")
    ax.set(xlim=(0, max(10, x[-1]) * 1.02), xlabel="Search cost (% of exhaustive evaluation cost)",
           ylabel="Recommended configuration's full-matrix score (%)",
           title=f"{benchmark_title}: recommendation as search progresses")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    ax.legend(loc="best", fontsize=8.5, frameon=False)
    save_figure(fig, f"{benchmark}-search")

    selected = [index[row["model_name"]] for row in multi.history[-1]["recommendations"]]
    front = sorted(frontier_indices(mean_scores, mean_costs), key=lambda i: mean_costs[i])
    fig, ax = plt.subplots(figsize=(8.2, 4.3), layout="constrained")
    ax.scatter(mean_costs, mean_scores * 100, s=25, color="#bdc6d3", alpha=.8,
               label="All 100 configurations")
    ax.plot(mean_costs[front], mean_scores[front] * 100, "o-", color="#209a80",
            markersize=4, linewidth=1.4, label="Full-matrix Pareto frontier (reference)")
    ax.scatter(mean_costs[selected], mean_scores[selected] * 100, s=110,
               facecolors="none", edgecolors="#7864d9", linewidths=2,
               label=f"Multi-objective Gittins recommendations ({len(selected)})", zorder=5)
    ax.set(xscale="log", xlabel="Mean model-call cost per question (USD; log scale)",
           ylabel="Full-matrix mean score (%)",
           title=f"{benchmark_title}: quality and deployment cost")
    ax.grid(color="#e2e8f0", linewidth=.6)
    ax.legend(loc="best", fontsize=8.5, frameon=False)
    save_figure(fig, f"{benchmark}-tradeoffs")


def update_summary(manifest):
    rows = ["| Benchmark | Mode | Search cost | Evaluations | Recommendations |",
            "|:--|:--|--:|--:|--:|"]
    for benchmark, report in manifest["benchmarks"].items():
        title = {"hotpotqa": "HotpotQA", "mathqa": "MathQA"}[benchmark]
        for mode, run in report["runs"].items():
            cost = f"${run['search_cost_usd']:.4f} ({run['search_fraction']:.2%})"
            rows.append(f"| {title} | {mode.title()} | {cost} | {run['evaluations']:,} | {len(run['recommendations'])} |")
    page = ROOT / "docs/benchmark-results/index.md"
    source = page.read_text(encoding="utf-8")
    start, end = "<!-- gittins-summary-start -->", "<!-- gittins-summary-end -->"
    prefix, rest = source.split(start, 1)
    _, suffix = rest.split(end, 1)
    page.write_text(prefix + start + "\n" + "\n".join(rows) + "\n" + end + suffix,
                    encoding="utf-8", newline="\n")


def plot_saved():
    chart_style()
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["input_sha256"].items():
        assert hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest() == expected
    for benchmark in manifest["benchmarks"]:
        names, scores = read_matrix(ASSETS / "inputs" / f"{benchmark}-scores.csv")
        _, costs = read_matrix(ASSETS / "inputs" / f"{benchmark}-costs.csv")
        histories = {mode: SimpleNamespace(history=json.loads(
            (ASSETS / f"{benchmark}-{mode}-history.json").read_text(encoding="utf-8")))
            for mode in ("single", "multi")}
        make_figures(benchmark, scores, costs, histories["single"], histories["multi"], names)
    update_summary(manifest)


def main():
    chart_style()
    provenance = json.loads((ASSETS / "sources.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["input_sha256"].items():
        assert hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest() == expected
    for relative, expected in provenance["selector_source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_text(encoding="utf-8").encode()).hexdigest() == expected
    manifest = dict(provenance, seed=SEED, budget_fraction=BUDGET_FRACTION,
                    SDK_commit="d4c00e6e86cdb9814ab534981708bf0e8a0822fd",
                    batch_size=4, warm_start_batch_size=4, grid_size=129,
                    comparison="None; one seed per mode and benchmark, not an algorithm ranking.",
                    cost_definition="Sum of frozen per-cell workflow token costs; no cross-cell cache reuse. Not an invoice or runtime measurement.",
                    benchmarks={})
    for benchmark in ("hotpotqa", "mathqa"):
        names, scores = read_matrix(ASSETS / "inputs" / f"{benchmark}-scores.csv")
        cost_names, costs = read_matrix(ASSETS / "inputs" / f"{benchmark}-costs.csv")
        assert names == cost_names and (costs >= 0).all() and (costs.mean(axis=1) > 0).all() and ((scores >= 0) & (scores <= 1)).all()
        outputs, selectors = {}, {}
        for mode in ("single", "multi"):
            print(f"Replaying {benchmark} / {mode}...", flush=True)
            selector = MatrixReplay(names, scores, costs, mode)
            result = selector.select_best()
            assert selector.history and len({r.datapoint_index for r in selector._datapoints[0]}) == len(selector._datapoints[0])
            assert selector._evaluations == sum(len(values) for values in selector._scores)
            assert np.isclose(selector._spent, sum(sum(values) for values in selector._costs))
            for dps in selector._datapoints:
                assert len({r.datapoint_index for r in dps}) == len(dps)
            outputs[mode] = dict(search_cost_usd=selector._spent,
                                 search_fraction=selector._spent / float(costs.sum()),
                                 evaluations=selector._evaluations, stop_reason=result.stop_reason,
                                 recommendations=[candidate.model_dump() for candidate in result.recommendations])
            (ASSETS / f"{benchmark}-{mode}-history.json").write_text(
                "[\n" + ",\n".join(json.dumps(row, separators=(",", ":")) for row in selector.history) + "\n]\n", encoding="utf-8")
            selectors[mode] = selector
            print(f"  {selector._evaluations} evaluations, ${selector._spent:.6f}; {result.stop_reason}", flush=True)
        with (ASSETS / f"{benchmark}-configurations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["configuration", "full_matrix_mean_score", "mean_cost_usd", "questions"])
            writer.writerows((name, float(score), float(cost), 200)
                             for name, score, cost in zip(names, scores.mean(axis=1), costs.mean(axis=1)))
        make_figures(benchmark, scores, costs, selectors["single"], selectors["multi"], names)
        manifest["benchmarks"][benchmark] = dict(configurations=100, questions=200,
                                                exhaustive_cost_usd=float(costs.sum()), runs=outputs)
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    update_summary(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Rebuild figures and table from saved trajectories")
    args = parser.parse_args()
    plot_saved() if args.plot_only else main()
