"""
Base classes and result types for model selection.
"""

import asyncio
import inspect
import itertools
import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel, Field, PrivateAttr

from agentopt.proxy import LLMTracker

from ..base_models import (
    Dataset,
    EvalFn,
    ModelCandidate,
    ModelsConfig,
    validate_dataset,
)
from ..model_price import compute_price
from .objectives import (
    ObjectiveMode,
    chebyshev_scalar,
    chebyshev_weight_at,
    pareto_mask_3d,
    score_to_error,
    validate_objective_config,
)

logger = logging.getLogger(__name__)


class DatapointResult(BaseModel):
    """Metrics for a single datapoint evaluation."""

    datapoint_index: int
    score: float
    latency_seconds: float
    input_tokens: Dict[str, int] = Field(default_factory=dict)
    output_tokens: Dict[str, int] = Field(default_factory=dict)


class ModelResult(BaseModel):
    """Result of evaluating a single model combination."""

    model_name: str
    accuracy: float
    latency_seconds: float
    input_tokens: Dict[str, int] = Field(default_factory=dict)
    output_tokens: Dict[str, int] = Field(default_factory=dict)
    attribute: str
    is_best: bool = False
    is_pareto_optimal: bool = Field(
        default=False,
        description="True when this combo lies on the empirical Pareto frontier.",
    )
    datapoint_results: List[DatapointResult] = Field(default_factory=list)
    combined_objective: Optional[float] = Field(
        default=None,
        description=(
            "Per-sample mean of the combined objective "
            "(score - lambda_cost*norm_cost - lambda_latency*norm_latency). "
            "None when no lambdas were configured."
        ),
    )
    _custom_prices: Optional[Dict[str, Tuple[float, float]]] = PrivateAttr(default=None)

    @property
    def error(self) -> float:
        """Per-combo error rate ``1 - accuracy`` (for Pareto plots; scores in [0, 1])."""
        return score_to_error(self.accuracy)

    @property
    def num_samples(self) -> int:
        """Number of datapoints evaluated; falls back to 1 for failed combos."""
        return len(self.datapoint_results) or 1

    @property
    def total_input_tokens(self) -> int:
        return sum(self.input_tokens.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self.output_tokens.values())

    @property
    def price(self) -> Optional[float]:
        """Per-sample cost in USD, or ``None`` if pricing is unavailable."""
        total = compute_price(
            self.input_tokens, self.output_tokens, custom_prices=self._custom_prices
        )
        if total is None:
            return None
        return total / self.num_samples

    def __str__(self) -> str:
        tok_parts = []
        for model in sorted(set(self.input_tokens) | set(self.output_tokens)):
            i = self.input_tokens.get(model, 0)
            o = self.output_tokens.get(model, 0)
            tok_parts.append(f"{model}: {i}/{o}")
        tok_str = ", ".join(tok_parts) if tok_parts else "0/0"
        p = self.price
        price_str = f"${p:.6f}" if p is not None else "N/A"
        return (
            f"{self.model_name} (accuracy: {self.accuracy:.2%}, "
            f"latency: {self.latency_seconds:.2f}s, "
            f"tokens: {{{tok_str}}}, "
            f"price: {price_str})"
        )


class SelectionResults(BaseModel):
    """Results from model selection."""

    results: List[ModelResult] = Field(default_factory=list)
    objective_mode: Optional[ObjectiveMode] = Field(
        default=None,
        description="``weighted`` (single best) or ``pareto`` (frontier).",
    )
    selection_wall_time_seconds: Optional[float] = None
    selection_cost: Optional[float] = Field(
        default=None, description="Total selection cost in USD.",
    )

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def get_best(self, attribute: Optional[str] = None) -> Optional[ModelResult]:
        """Get the best model result, optionally filtered by attribute."""
        filtered = self.results
        if attribute:
            filtered = [r for r in self.results if r.attribute == attribute]
        best = [r for r in filtered if r.is_best]
        return best[0] if best else None

    def get_best_combo(self) -> Optional[Dict[str, str]]:
        """Get the best model combination as a dict of node_name -> model_name.

        Parses the combo_name format ``"node1=model1 + node2=model2"``.
        """
        best = self.get_best()
        if best is None:
            return None
        combo = {}
        for part in best.model_name.split(" + "):
            if "=" in part:
                node, model = part.split("=", 1)
                combo[node] = model
            else:
                combo[part] = part
        return combo

    def get_by_attribute(self, attribute: str) -> List[ModelResult]:
        """Get all results for a specific attribute."""
        return [r for r in self.results if r.attribute == attribute]

    def get_pareto_front(self) -> List[ModelResult]:
        """Return Pareto-optimal combinations (error, latency, cost all minimized).

        Uses the same deduplication and final-layer filtering as :meth:`plot_pareto`.
        """
        unique = self._comparable_results()
        if not unique:
            return []
        return [r for r in unique if r.is_pareto_optimal]

    def _comparable_results(self) -> List[ModelResult]:
        """Deduplicated results at the fullest evaluation depth."""
        seen: Dict[str, ModelResult] = {}
        for r in self.results:
            if r.model_name not in seen or (
                r.is_best and not seen[r.model_name].is_best
            ):
                seen[r.model_name] = r
        unique = list(seen.values())
        if not unique:
            return []
        if any(r.datapoint_results for r in unique):
            max_samples = max(
                r.num_samples for r in unique if r.datapoint_results
            )
            unique = [
                r for r in unique
                if r.datapoint_results and r.num_samples == max_samples
            ]
        return unique

    def export_config(
        self, output_path: str, api_key_env_vars: Optional[Dict[str, str]] = None,
    ) -> None:
        """Export the best combination as a config YAML."""
        best_combo = self.get_best_combo()
        if best_combo is None:
            raise ValueError("No best combination found to export.")

        if api_key_env_vars is None:
            api_key_env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
            }

        def _detect_provider(model_name: str) -> str:
            if "/" in model_name:
                return model_name.split("/")[0]
            lower = model_name.lower()
            if "gpt" in lower or lower.startswith("o3") or lower.startswith("o4"):
                return "openai"
            elif "claude" in lower:
                return "anthropic"
            elif "gemini" in lower:
                return "google"
            return "openai"

        def _full_model_name(model_name: str) -> str:
            if "/" in model_name:
                return model_name
            return f"{_detect_provider(model_name)}/{model_name}"

        unique_models = set(best_combo.values())
        lines = ["model_list:"]
        for model in sorted(unique_models):
            full_name = _full_model_name(model)
            provider = _detect_provider(model)
            env_var = api_key_env_vars.get(provider, "API_KEY")
            lines.append(f"  - model_name: {model}")
            lines.append(f"    litellm_params:")
            lines.append(f"      model: {full_name}")
            lines.append(f"      api_key: os.environ/{env_var}")
            lines.append("")

        lines.append("# Optimized model mapping (from agentopt):")
        for node, model in best_combo.items():
            lines.append(f"#   {node}: {model}")
        lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def to_csv(self, path: str) -> None:
        """Save results to CSV file."""
        import csv
        import json

        if not self.results:
            return

        fieldnames = [
            "model_name",
            "accuracy",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "price",
            "attribute",
            "is_best",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = result.model_dump()
                row["input_tokens"] = json.dumps(row["input_tokens"])
                row["output_tokens"] = json.dumps(row["output_tokens"])
                p = result.price
                row["price"] = f"{p:.6f}" if p is not None else ""
                writer.writerow(row)

    @staticmethod
    def _build_prefix_sums(
        r: "ModelResult",
    ) -> Tuple[List[float], List[float], Dict[str, List[int]], Dict[str, List[int]]]:
        """Build cumulative sums from datapoint_results for O(1) layer lookups.

        Returns (cum_scores, cum_latencies, cum_input_tokens, cum_output_tokens)
        where each list has length len(datapoint_results)+1 and index 0 is 0.
        """
        dps = r.datapoint_results
        n = len(dps)
        cum_scores = [0.0] * (n + 1)
        cum_lats = [0.0] * (n + 1)
        # Collect all model names first.
        all_models: set = set()
        for dp in dps:
            all_models.update(dp.input_tokens)
            all_models.update(dp.output_tokens)
        cum_in: Dict[str, List[int]] = {m: [0] * (n + 1) for m in all_models}
        cum_out: Dict[str, List[int]] = {m: [0] * (n + 1) for m in all_models}
        for i, dp in enumerate(dps):
            cum_scores[i + 1] = cum_scores[i] + dp.score
            cum_lats[i + 1] = cum_lats[i] + dp.latency_seconds
            for m in all_models:
                cum_in[m][i + 1] = cum_in[m][i] + dp.input_tokens.get(m, 0)
                cum_out[m][i + 1] = cum_out[m][i] + dp.output_tokens.get(m, 0)
        return cum_scores, cum_lats, cum_in, cum_out

    @staticmethod
    def _metrics_at(
        prefix: Tuple[
            List[float], List[float], Dict[str, List[int]], Dict[str, List[int]]
        ],
        n: int,
        custom_prices: Optional[Dict[str, Tuple[float, float]]],
    ) -> Tuple[float, float, Optional[float]]:
        """Look up accuracy, latency, per-sample price at *n* datapoints using prefix sums."""
        cum_scores, cum_lats, cum_in, cum_out = prefix
        acc = cum_scores[n] / n
        lat = cum_lats[n] / n
        agg_in = {m: vals[n] for m, vals in cum_in.items()}
        agg_out = {m: vals[n] for m, vals in cum_out.items()}
        total = compute_price(agg_in, agg_out, custom_prices=custom_prices)
        price = total / n if total is not None else None
        return acc, lat, price

    def __str__(self) -> str:
        if not self.results:
            return "No results."

        # Deduplicate by model_name, preferring entries with is_best=True.
        seen: Dict[str, "ModelResult"] = {}
        for r in self.results:
            if r.model_name not in seen or (
                r.is_best and not seen[r.model_name].is_best
            ):
                seen[r.model_name] = r
        unique = list(seen.values())

        # Determine layer boundaries from distinct num_samples values.
        sample_counts = sorted({r.num_samples for r in unique if r.datapoint_results})
        use_layers = len(sample_counts) > 1

        # Format helpers.
        def fmt_acc(v: float) -> str:
            return f"{v:.2%}"

        def fmt_lat(v: float) -> str:
            return f"{v:.2f}s"

        def fmt_price_val(p: Optional[float]) -> str:
            return f"${p:.6f}" if p is not None else "N/A"

        def fmt_price(r: "ModelResult") -> str:
            return fmt_price_val(r.price)

        marker = ">>>"
        pad = " " * len(marker)

        if not use_layers:
            # --- Flat table (single layer / non-bandit) ---
            unique.sort(key=lambda r: (-r.accuracy, r.latency_seconds))
            return self._render_flat(unique, fmt_acc, fmt_lat, fmt_price, marker, pad)

        # --- Layered table (bandit algorithms) ---
        total_combos = len(unique)
        lines: List[str] = []
        lines.append("")
        lines.append(pad + " Model Selection Results")

        # Precompute prefix sums once per result for O(1) layer lookups.
        prefix_cache = {
            id(r): self._build_prefix_sums(r) for r in unique if r.datapoint_results
        }

        for layer_idx, n_samples in enumerate(sample_counts):
            is_final = layer_idx == len(sample_counts) - 1
            layer_results = [r for r in unique if r.num_samples >= n_samples]
            # Look up metrics at this layer's sample count.
            recomputed: List[Tuple["ModelResult", float, float, Optional[float]]] = []
            for r in layer_results:
                pfx = prefix_cache.get(id(r))
                if pfx is not None:
                    acc, lat, price = self._metrics_at(pfx, n_samples, r._custom_prices)
                else:
                    acc, lat, price = r.accuracy, r.latency_seconds, r.price
                recomputed.append((r, acc, lat, price))
            recomputed.sort(key=lambda t: (-t[1], t[2]))

            # Eliminated = combos with exactly this num_samples (won't appear next layer).
            eliminated = (
                [r.model_name for r in unique if r.num_samples == n_samples]
                if not is_final
                else []
            )

            # Column widths for this layer.
            rank_h, model_h, acc_h, lat_h, price_h = (
                "Rank",
                "Model",
                "Accuracy",
                "Latency",
                "Price",
            )
            rank_w = max(len(rank_h), len(str(len(recomputed))))
            model_w = max(len(model_h), *(len(t[0].model_name) for t in recomputed),)
            acc_w = max(len(acc_h), *(len(fmt_acc(t[1])) for t in recomputed))
            lat_w = max(len(lat_h), *(len(fmt_lat(t[2])) for t in recomputed))
            price_w = max(
                len(price_h), *(len(fmt_price_val(t[3])) for t in recomputed),
            )

            def row(
                rank_s: str,
                model_s: str,
                acc_s: str,
                lat_s: str,
                price_s: str,
                best: bool,
            ) -> str:
                prefix = marker if best else pad
                return (
                    f"{prefix} {rank_s:>{rank_w}}  "
                    f"{model_s:<{model_w}}  "
                    f"{acc_s:>{acc_w}}  "
                    f"{lat_s:>{lat_w}}  "
                    f"{price_s:>{price_w}}"
                )

            header_row = row(rank_h, model_h, acc_h, lat_h, price_h, False)
            sep = pad + " " + "-" * (len(header_row) - len(pad) - 1)

            label = "Final" if is_final else f"Round {layer_idx + 1}"
            lines.append("")
            lines.append(
                f"{pad} {label} "
                f"({n_samples} datapoints, "
                f"{len(layer_results)}/{total_combos} combos):"
            )
            lines.append(sep)
            lines.append(header_row)
            lines.append(sep)

            for i, (r, acc, lat, price) in enumerate(recomputed, 1):
                is_best = r.is_best and is_final
                lines.append(
                    row(
                        str(i),
                        r.model_name,
                        fmt_acc(acc),
                        fmt_lat(lat),
                        fmt_price_val(price),
                        is_best,
                    )
                )

            lines.append(sep)

            if eliminated:
                lines.append(f"{pad} Eliminated: {', '.join(eliminated)}")

        # Best result summary.
        best_result = next((r for r in unique if r.is_best), None)
        if best_result:
            lines.append("")
            lines.append(
                f"{pad} Best: {best_result.model_name} "
                f"(accuracy: {best_result.accuracy:.2%}, "
                f"latency: {best_result.latency_seconds:.2f}s, "
                f"price: {fmt_price(best_result)})"
            )
        lines.append("")

        # Selection overhead.
        parts: List[str] = []
        if self.selection_wall_time_seconds is not None:
            parts.append(f"{self.selection_wall_time_seconds:.2f}s")
        if self.selection_cost is not None:
            parts.append(f"${self.selection_cost:.6f}")
        if parts:
            lines.append(f"{pad} Selection overhead: {', '.join(parts)}")
            lines.append("")

        return "\n".join(lines)

    def _render_flat(
        self,
        unique: List["ModelResult"],
        fmt_acc: Callable[[float], str],
        fmt_lat: Callable[[float], str],
        fmt_price: Callable[["ModelResult"], str],
        marker: str,
        pad: str,
    ) -> str:
        """Render the original flat table for non-bandit selectors."""
        rank_h, model_h, acc_h, lat_h, price_h = (
            "Rank",
            "Model",
            "Accuracy",
            "Latency",
            "Price",
        )
        rank_w = max(len(rank_h), len(str(len(unique))))
        model_w = max(len(model_h), *(len(r.model_name) for r in unique))
        acc_w = max(len(acc_h), *(len(fmt_acc(r.accuracy)) for r in unique))
        lat_w = max(len(lat_h), *(len(fmt_lat(r.latency_seconds)) for r in unique))
        price_w = max(len(price_h), *(len(fmt_price(r)) for r in unique))

        def row(
            rank_s: str, model_s: str, acc_s: str, lat_s: str, price_s: str, best: bool,
        ) -> str:
            prefix = marker if best else pad
            return (
                f"{prefix} {rank_s:>{rank_w}}  "
                f"{model_s:<{model_w}}  "
                f"{acc_s:>{acc_w}}  "
                f"{lat_s:>{lat_w}}  "
                f"{price_s:>{price_w}}"
            )

        header_row = row(rank_h, model_h, acc_h, lat_h, price_h, False)
        sep = pad + " " + "-" * (len(header_row) - len(pad) - 1)

        lines: List[str] = []
        lines.append("")
        lines.append(pad + " " + "Model Selection Results")
        lines.append(sep)
        lines.append(header_row)
        lines.append(sep)

        for i, r in enumerate(unique, 1):
            lines.append(
                row(
                    str(i),
                    r.model_name,
                    fmt_acc(r.accuracy),
                    fmt_lat(r.latency_seconds),
                    fmt_price(r),
                    r.is_best,
                )
            )

        lines.append(sep)

        pareto_results = self.get_pareto_front() if self.objective_mode == "pareto" else []
        best_result = next((r for r in unique if r.is_best), None)
        if self.objective_mode == "pareto" and pareto_results:
            lines.append("")
            lines.append(
                f"{pad} Pareto-optimal: {len(pareto_results)} combination(s) "
                f"(ideal: 0% error, 0s latency, $0)"
            )
            for r in pareto_results[:8]:
                lines.append(
                    f"{pad}   {r.model_name} — error {r.error:.2%}, "
                    f"{r.latency_seconds:.2f}s, {fmt_price(r)}"
                )
            if len(pareto_results) > 8:
                lines.append(f"{pad}   ... and {len(pareto_results) - 8} more")
        elif best_result:
            lines.append(
                f"{pad} Best: {best_result.model_name} "
                f"(accuracy: {best_result.accuracy:.2%}, "
                f"latency: {best_result.latency_seconds:.2f}s, "
                f"price: {fmt_price(best_result)})"
            )
        lines.append("")

        parts: List[str] = []
        if self.selection_wall_time_seconds is not None:
            parts.append(f"{self.selection_wall_time_seconds:.2f}s")
        if self.selection_cost is not None:
            parts.append(f"${self.selection_cost:.6f}")
        if parts:
            lines.append(f"{pad} Selection overhead: {', '.join(parts)}")
            lines.append("")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary table of all results."""
        print(self)

    # ------------------------------------------------------------------
    # Pareto frontier visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pareto_mask(
        xs: List[float], ys: List[float], x_minimize: bool, y_minimize: bool,
    ) -> List[bool]:
        """Return a boolean mask marking Pareto-optimal points.

        A point is Pareto-optimal if no other point is strictly better on both
        objectives.
        """
        n = len(xs)
        mask = [True] * n
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
                # Is j at least as good as i on both, and strictly better on one?
                x_ok = (xj <= xi) if x_minimize else (xj >= xi)
                y_ok = (yj <= yi) if y_minimize else (yj >= yi)
                x_strict = (xj < xi) if x_minimize else (xj > xi)
                y_strict = (yj < yi) if y_minimize else (yj > yi)
                if x_ok and y_ok and (x_strict or y_strict):
                    mask[i] = False
                    break
        return mask

    def plot_pareto(
        self, path: Optional[str] = None, *, show_ideal: bool = True,
    ) -> None:
        """Generate pairwise Pareto frontier plots (error, latency, price).

        Both axes are **lower is better**. The ideal wish corner is
        (0 error, 0s latency, $0) when ``show_ideal`` is True.

        Requires ``matplotlib`` (install with ``pip install agentopt-py[plot]``).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_pareto. "
                "Install it with: pip install agentopt-py[plot]"
            )

        unique = self._comparable_results()
        if not unique:
            seen: Dict[str, ModelResult] = {}
            for r in self.results:
                seen[r.model_name] = r
            unique = list(seen.values())

        # Prefer combos with price for the cost panel; keep all for error/latency.
        with_price = [r for r in unique if r.price is not None]
        plot_set = with_price if len(with_price) >= 2 else unique

        plot_set = list(plot_set)
        plot_set.sort(key=lambda r: (r.error, r.latency_seconds))

        if len(plot_set) < 2:
            print("Not enough comparable results to plot (need at least 2).")
            return

        names = [r.model_name for r in plot_set]
        errors = [r.error for r in plot_set]
        lats = [r.latency_seconds for r in plot_set]
        prices = [r.price for r in plot_set]
        has_price = [p is not None for p in prices]
        is_pareto = [r.is_pareto_optimal for r in plot_set]
        is_best = [r.is_best for r in plot_set]

        num_labels = [f"({i})" for i in range(1, len(plot_set) + 1)]

        pairs: List[Tuple[List[float], List[float], str, str]] = [
            (lats, errors, "Latency (s)", "Error"),
        ]
        if any(has_price):
            px = [p for p, ok in zip(prices, has_price) if ok]  # type: ignore[misc]
            ey = [e for e, ok in zip(errors, has_price) if ok]
            if len(px) >= 2:
                pairs.append((px, ey, "Price ($)", "Error"))

        fig = plt.figure(figsize=(7 * len(pairs), 5))
        gs = fig.add_gridspec(1, len(pairs), left=0.06, right=0.68, wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(pairs))]
        title = "Pareto Frontiers (lower is better)"
        if self.objective_mode == "weighted":
            title += " — weighted run"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for ax, (xs, ys, xlabel, ylabel) in zip(axes, pairs):
            mask = self._pareto_mask(xs, ys, True, True)

            np_x = [x for x, m in zip(xs, mask) if not m]
            np_y = [y for y, m in zip(ys, mask) if not m]
            ax.scatter(
                np_x, np_y, c="lightgray", edgecolors="gray", s=60,
                zorder=2, label="Dominated",
            )

            p_x = [x for x, m in zip(xs, mask) if m]
            p_y = [y for y, m in zip(ys, mask) if m]
            ax.scatter(
                p_x, p_y, c="steelblue", edgecolors="navy", s=80,
                zorder=3, label="Pareto-optimal",
            )

            if p_x:
                order = sorted(range(len(p_x)), key=lambda i: p_x[i])
                ax.plot(
                    [p_x[i] for i in order], [p_y[i] for i in order],
                    c="steelblue", linewidth=1.5, alpha=0.6, zorder=2,
                )

            if show_ideal:
                ax.scatter(
                    [0.0], [0.0], c="none", edgecolors="green", s=120,
                    linewidths=2, marker="o", zorder=1, label="Ideal (0, 0)",
                )

            for x, y, b, po in zip(xs, ys, is_best, is_pareto):
                if b:
                    ax.scatter(
                        [x], [y], c="gold", edgecolors="darkorange", s=140,
                        zorder=5, marker="*", label="Best (weighted)",
                    )
                elif po and self.objective_mode == "pareto":
                    ax.scatter(
                        [x], [y], facecolors="none", edgecolors="darkorange",
                        s=120, zorder=4, linewidths=1.5, label="Frontier",
                    )

            for x, y, lbl in zip(xs, ys, num_labels):
                ax.annotate(
                    lbl, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontweight="bold",
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        # External legend mapping numbers to combo names.
        legend_lines = [f"({i}) {name}" for i, name in enumerate(names, 1)]
        fig.text(
            0.72,
            0.5,
            "\n".join(legend_lines),
            fontsize=8,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightyellow",
                edgecolor="gray",
                alpha=0.9,
            ),
        )

        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Pareto plot saved to {path}")
        else:
            plt.show()
        plt.close(fig)


class BaseModelSelector(ABC):
    """Abstract base class for model selectors.

    Provide an agent class with ``__init__(self, models)`` and
    ``run(self, input_data)`` methods.  For each model combination the
    selector instantiates the class and calls ``run`` on every datapoint.
    """

    def __init__(
        self,
        agent: Any = None,
        models: ModelsConfig = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        node_descriptions: Optional[Dict[str, str]] = None,
        tracker: Optional[LLMTracker] = None,
        objective_mode: Optional[ObjectiveMode] = None,
        lambda_cost: float = 0.0,
        lambda_latency: float = 0.0,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            agent: Agent class implementing ``__init__(self, models)`` and
                ``run(self, input_data)``.  The selector will call
                ``agent(combo_dict)`` to create an instance, then call
                ``instance.run(input_data)`` for each datapoint.
                No base class is required — duck typing only.
            models: Dict mapping node names to candidate model specs, e.g.
                ``{"planner": ["gpt-4o", "gpt-4o-mini"]}`` or prebuilt
                LLM instances.
            eval_fn: Function ``(expected, actual) -> bool | float``
                (higher is better).
            dataset: Sequence of ``(input_data, expected_answer)`` pairs.
            model_prices: Optional custom pricing overrides. Maps model names
                to dicts with ``'input_price'`` and ``'output_price'`` keys
                ($/MTok).
            node_descriptions: Optional dict mapping node names to human-readable
                descriptions of what each node does, e.g.
                ``{"planner": "Decomposes queries into sub-tasks"}``.
            tracker: Optional :class:`LLMTracker` instance. If not provided,
                one is created and started automatically.
            objective_mode: Required. ``"weighted"`` — pass ``lambda_cost``
                and/or ``lambda_latency`` > 0 for a single recommended combo.
                ``"pareto"`` — omit lambdas; explore the error/latency/cost
                frontier (Chebyshev-driven matrix UCB; full eval otherwise).
            lambda_cost: Weight on normalized per-sample cost (weighted mode only).
            lambda_latency: Weight on normalized per-sample latency (weighted mode only).
        """
        if agent is None:
            raise TypeError("'agent' is required")
        if models is None or eval_fn is None or dataset is None:
            raise TypeError("'models', 'eval_fn', and 'dataset' are required")

        self.objective_mode = validate_objective_config(
            objective_mode, lambda_cost, lambda_latency,
        )
        self.lambda_cost = float(lambda_cost)
        self.lambda_latency = float(lambda_latency)

        validate_dataset(dataset)

        self._custom_prices: Optional[Dict[str, Tuple[float, float]]] = (
            {
                name: (d["input_price"], d["output_price"])
                for name, d in model_prices.items()
            }
            if model_prices
            else None
        )

        self.agent = agent
        self.eval_fn = eval_fn
        self.dataset = dataset
        self._models = models
        self._node_names = list(models.keys())
        self.model_prices = model_prices
        self.node_descriptions = node_descriptions

        # Running min/max for adaptive [0,1] normalization of cost/latency/gaps.
        self._cost_min: float = float("inf")
        self._cost_max: float = float("-inf")
        self._latency_min: float = float("inf")
        self._latency_max: float = float("-inf")
        self._error_min: float = float("inf")
        self._error_max: float = float("-inf")
        self._chebyshev_step: int = 0

        # Detect whether agent.run() is async
        run_method = getattr(agent, "run", None)
        self._is_async_run = inspect.iscoroutinefunction(run_method)

        if tracker is not None:
            self._tracker = tracker
        else:
            self._tracker = LLMTracker()
        self._tracker.start()

    # ------------------------------------------------------------------
    # Combo generation
    # ------------------------------------------------------------------

    def _generate_combos(self,) -> Generator[Dict[str, ModelCandidate], None, None]:
        """Yield all combinations as ``{node_name: model_candidate}`` dicts."""
        for combo_tuple in itertools.product(*self._models.values()):
            yield dict(zip(self._node_names, combo_tuple))

    def _all_combos(self) -> List[Dict[str, ModelCandidate]]:
        """Return all combinations as a list."""
        return list(self._generate_combos())

    @staticmethod
    def _candidate_label(candidate: ModelCandidate) -> str:
        """Return a readable, mostly stable label for a model candidate."""
        if isinstance(candidate, str):
            return candidate

        if isinstance(candidate, dict):
            for key in ("model", "model_name", "id", "name"):
                value = candidate.get(key)
                if value is not None:
                    return str(value)

        for attr in ("model", "model_name", "id", "name"):
            value = getattr(candidate, attr, None)
            if value is not None:
                return str(value)

        return f"{type(candidate).__name__}@{id(candidate):x}"

    @staticmethod
    def _combo_name(combo: Dict[str, ModelCandidate]) -> str:
        """Generate display name for a combination."""
        return " + ".join(
            f"{k}={BaseModelSelector._candidate_label(v)}" for k, v in combo.items()
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _invoke_agent(self, agent: Any, input_data: Any) -> Any:
        """Call agent.run(input_data)."""
        return agent.run(input_data)

    def _evaluate_combo(
        self,
        combo: Dict[str, ModelCandidate],
        evaluation_tasks: Dataset,
        label: str = "",
        dp_offset: int = 0,
    ) -> Tuple[List[float], List[float], List[str]]:
        """Build an agent for combo and evaluate it on tasks.

        Returns (scores, latencies, datapoint_ids).
        """
        agent = self.agent(combo)
        return self._evaluate_agent(agent, evaluation_tasks, label, dp_offset=dp_offset)

    def _evaluate_agent(
        self,
        agent: Any,
        evaluation_tasks: Dataset,
        label: str = "",
        dp_offset: int = 0,
    ) -> Tuple[List[float], List[float], List[str]]:
        """Evaluate a pre-built agent against tasks.

        Returns (scores, latencies, datapoint_ids).
        """
        scores: List[float] = []
        latencies: List[float] = []
        datapoint_ids: List[str] = []
        total = len(evaluation_tasks)

        for i, (input_data, expected_answer) in enumerate(evaluation_tasks, 1):
            dp_id = f"{label}::dp_{dp_offset + i}"
            try:
                with self._tracker.track(data_id=dp_id, combo_id=label) as session:
                    start_time = time.time()
                    actual_result = self._invoke_agent(agent, input_data)
                    wall_clock = time.time() - start_time
                    # Add back time saved by cache hits so latency
                    # reflects what a real (uncached) run would cost.
                    cached_latency = self._tracker.get_cached_latency(data_id=dp_id)
                    latency = wall_clock + cached_latency
                score = float(self.eval_fn(expected_answer, actual_result))
                scores.append(score)
                latencies.append(latency)
                datapoint_ids.append(dp_id)
            except Exception as e:
                logger.warning("[%s] sample %d/%d error: %s", label, i, total, e)

        return scores, latencies, datapoint_ids

    async def _evaluate_combo_async(
        self,
        combo: Dict[str, ModelCandidate],
        evaluation_tasks: Dataset,
        label: str = "",
        max_concurrent: int = 20,
        dp_offset: int = 0,
    ) -> Tuple[List[float], List[float], List[str]]:
        """Build an agent for combo and evaluate async.

        Returns (scores, latencies, datapoint_ids).
        """
        agent = self.agent(combo)
        return await self._evaluate_agent_async(
            agent, evaluation_tasks, label, max_concurrent, dp_offset=dp_offset
        )

    async def _evaluate_agent_async(
        self,
        agent: Any,
        evaluation_tasks: Dataset,
        label: str = "",
        max_concurrent: int = 20,
        dp_offset: int = 0,
    ) -> Tuple[List[float], List[float], List[str]]:
        """Evaluate a pre-built agent asynchronously.

        Returns (scores, latencies, datapoint_ids).
        """
        import contextvars as _ctx

        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(evaluation_tasks)
        results: List[Optional[dict]] = [None] * total

        is_async = self._is_async_run

        async def _eval_single(idx: int, input_data: Any, expected_answer: Any) -> None:
            dp_id = f"{label}::dp_{dp_offset + idx + 1}"
            async with semaphore:
                try:
                    with self._tracker.track(data_id=dp_id, combo_id=label) as session:
                        start_time = time.time()
                        if is_async:
                            actual_result = await self._invoke_agent(agent, input_data)
                        else:
                            loop = asyncio.get_running_loop()
                            ctx = _ctx.copy_context()
                            actual_result = await loop.run_in_executor(
                                None, ctx.run, self._invoke_agent, agent, input_data
                            )
                        wall_clock = time.time() - start_time
                        # Add back time saved by cache hits so latency
                        # reflects what a real (uncached) run would cost.
                        cached_latency = self._tracker.get_cached_latency(data_id=dp_id)
                        latency = wall_clock + cached_latency
                    score = float(self.eval_fn(expected_answer, actual_result))
                    results[idx] = {"score": score, "latency": latency, "dp_id": dp_id}
                except Exception as e:
                    logger.warning(
                        "[%s] sample %d/%d error: %s", label, idx + 1, total, e
                    )

        tasks = [
            _eval_single(i, inp, exp) for i, (inp, exp) in enumerate(evaluation_tasks)
        ]
        await asyncio.gather(*tasks)

        scores: List[float] = []
        latencies: List[float] = []
        datapoint_ids: List[str] = []
        for r in results:
            if r is not None:
                scores.append(r["score"])
                latencies.append(r["latency"])
                datapoint_ids.append(r["dp_id"])

        return scores, latencies, datapoint_ids

    # ------------------------------------------------------------------
    # Token tracking via agentopt.proxy
    # ------------------------------------------------------------------

    def _fetch_tokens(self, combo_id: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Return (input_tokens, output_tokens) dicts for a combo."""
        usage = self._tracker.get_usage(combo_id=combo_id)
        return self._split_tokens(usage)

    def _fetch_tokens_by_datapoint(
        self, datapoint_ids: List[str],
    ) -> Dict[str, Tuple[Dict[str, int], Dict[str, int]]]:
        """Return per-datapoint token usage as ``{dp_id: (input_tokens, output_tokens)}``."""
        result: Dict[str, Tuple[Dict[str, int], Dict[str, int]]] = {}
        for dp_id in datapoint_ids:
            usage = self._tracker.get_usage(data_id=dp_id)
            result[dp_id] = self._split_tokens(usage)
        return result

    # ------------------------------------------------------------------
    # Combined objective (cost / latency weighted)
    # ------------------------------------------------------------------

    @property
    def _has_combined_objective(self) -> bool:
        """True in weighted mode with cost/latency lambdas configured."""
        return self.objective_mode == "weighted"

    @property
    def _pareto_exploration(self) -> bool:
        return self.objective_mode == "pareto"

    @property
    def _uses_matrix_scalar_refresh(self) -> bool:
        """Matrix UCB cells need recomputation when the normalizer moves."""
        return self._has_combined_objective or self._pareto_exploration

    def _per_sample_costs(self, dp_ids: List[str]) -> List[Optional[float]]:
        """Look up per-datapoint cost from tracked tokens; ``None`` if unpriced."""
        if not dp_ids:
            return []
        by_dp = self._fetch_tokens_by_datapoint(dp_ids)
        costs: List[Optional[float]] = []
        for dp_id in dp_ids:
            in_tok, out_tok = by_dp.get(dp_id, ({}, {}))
            costs.append(
                compute_price(in_tok, out_tok, custom_prices=self._custom_prices)
            )
        return costs

    def _absorb_pareto_gaps(
        self, scores: List[float], latencies: List[float], costs: List[Optional[float]],
    ) -> None:
        """Update running min/max for error, latency, and cost gaps (Pareto mode)."""
        if not self._pareto_exploration:
            return
        for sc in scores:
            err = score_to_error(sc)
            if err < self._error_min:
                self._error_min = err
            if err > self._error_max:
                self._error_max = err
        self._absorb_observations(latencies, costs)

    def _normalized_pareto_gaps(
        self, score: float, latency: float, cost: Optional[float],
    ) -> Tuple[float, float, float]:
        """Min-max normalized (error, latency, cost) gaps in [0, 1] (ideal = 0)."""
        err = score_to_error(score)
        ne = self._minmax_norm(err, self._error_min, self._error_max)
        nl = self._minmax_norm(latency, self._latency_min, self._latency_max)
        nc = self._minmax_norm(cost, self._cost_min, self._cost_max)
        return ne, nl, nc

    def _chebyshev_cell_scalar(
        self, score: float, latency: float, cost: Optional[float],
        weights: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """Chebyshev achievement scalar for one cell (lower is better)."""
        w = weights if weights is not None else chebyshev_weight_at(self._chebyshev_step)
        ne, nl, nc = self._normalized_pareto_gaps(score, latency, cost)
        use_cost = cost is not None and math.isfinite(cost)
        return chebyshev_scalar(ne, nl, nc, w, use_cost=use_cost)

    def _advance_chebyshev_direction(self) -> None:
        self._chebyshev_step += 1

    def _absorb_observations(
        self, latencies: List[float], costs: List[Optional[float]],
    ) -> None:
        """Update the running min/max with new samples."""
        if not self._has_combined_objective and not self._pareto_exploration:
            return
        for lat in latencies:
            if lat is None or not math.isfinite(lat):
                continue
            if lat < self._latency_min:
                self._latency_min = lat
            if lat > self._latency_max:
                self._latency_max = lat
        for c in costs:
            if c is None or not math.isfinite(c):
                continue
            if c < self._cost_min:
                self._cost_min = c
            if c > self._cost_max:
                self._cost_max = c

    @staticmethod
    def _minmax_norm(value: Optional[float], lo: float, hi: float) -> float:
        """Min-max normalize to [0,1]; returns 0.0 if value/range is degenerate."""
        if value is None or not math.isfinite(value):
            return 0.0
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _combined_objective(
        self, score: float, latency: float, cost: Optional[float],
    ) -> float:
        """Per-sample combined objective, computed against the current normalizer."""
        if not self._has_combined_objective:
            return score
        norm_lat = self._minmax_norm(latency, self._latency_min, self._latency_max)
        norm_cost = self._minmax_norm(cost, self._cost_min, self._cost_max)
        return (
            score
            - self.lambda_cost * norm_cost
            - self.lambda_latency * norm_lat
        )

    def _compute_objectives(
        self,
        scores: List[float],
        latencies: List[float],
        costs: List[Optional[float]],
    ) -> List[float]:
        """Vectorized per-sample combined objectives using current normalizer."""
        if not self._has_combined_objective:
            return list(scores)
        return [
            self._combined_objective(s, lat, c)
            for s, lat, c in zip(scores, latencies, costs)
        ]

    def _mean_objective(
        self,
        scores: List[float],
        latencies: List[float],
        costs: List[Optional[float]],
    ) -> Optional[float]:
        """Mean combined objective across samples, or ``None`` if no lambdas set."""
        if not self._has_combined_objective or not scores:
            return None
        objs = self._compute_objectives(scores, latencies, costs)
        return sum(objs) / len(objs)

    def _observe_combo(
        self,
        scores: List[float],
        latencies: List[float],
        dp_ids: List[str],
    ) -> List[Optional[float]]:
        """Fetch per-sample costs and absorb them into the running normalizer.

        Returns the per-sample costs aligned with ``scores`` / ``latencies``.
        Always returns a list (possibly all ``None`` if pricing is unavailable);
        selectors can cache it alongside scores/latencies for later objective
        recomputation when the normalizer changes.
        """
        costs = self._per_sample_costs(dp_ids)
        if self._pareto_exploration:
            self._absorb_pareto_gaps(scores, latencies, costs)
        else:
            self._absorb_observations(latencies, costs)
        return costs

    def _recover_costs(self, result: ModelResult) -> List[Optional[float]]:
        """Reconstruct per-sample costs for a finalized result from its dp tokens."""
        return [
            compute_price(
                dp.input_tokens,
                dp.output_tokens,
                custom_prices=result._custom_prices,
            )
            for dp in result.datapoint_results
        ]

    def _finalize_combined_objectives(self, results: List[ModelResult]) -> None:
        """Recompute ``combined_objective`` on every result using all observed data.

        Selectors typically evaluate combos in arbitrary order. ``_build_combo_result``
        sets ``combined_objective`` against the normalizer at observation time;
        this pass rebuilds it against the **final** normalizer so that all
        results are ranked on a consistent scale.

        Idempotent and a no-op when no lambdas are configured.
        """
        if not self._has_combined_objective:
            return

        # Pass 1: absorb every result's samples so the normalizer reflects the
        # full set (cheap; observe_combo during search may have missed some
        # samples if a selector skipped it).
        cached: List[Tuple[ModelResult, List[float], List[float], List[Optional[float]]]] = []
        for r in results:
            if not r.datapoint_results:
                cached.append((r, [], [], []))
                continue
            scores = [dp.score for dp in r.datapoint_results]
            lats = [dp.latency_seconds for dp in r.datapoint_results]
            costs = self._recover_costs(r)
            self._absorb_observations(lats, costs)
            cached.append((r, scores, lats, costs))

        # Pass 2: recompute objectives against the now-final normalizer.
        for r, scores, lats, costs in cached:
            r.combined_objective = self._mean_objective(scores, lats, costs)

    def _mark_pareto_optimal(self, results: List[ModelResult]) -> None:
        """Set ``is_pareto_optimal`` on results at the fullest evaluation depth."""
        for r in results:
            r.is_pareto_optimal = False
        candidates = [
            r for r in results
            if r.datapoint_results and r.accuracy >= 0.0
        ]
        if not candidates:
            return
        max_samples = max(r.num_samples for r in candidates)
        layer = [r for r in candidates if r.num_samples == max_samples]
        errors = [r.error for r in layer]
        lats = [r.latency_seconds for r in layer]
        costs: List[Optional[float]] = [r.price for r in layer]
        mask = pareto_mask_3d(errors, lats, costs)
        names = {layer[i].model_name for i, m in enumerate(mask) if m}
        for r in results:
            if r.model_name in names and r.num_samples == max_samples:
                r.is_pareto_optimal = True

    def _finalize_selection_outcomes(
        self, results: List[ModelResult],
    ) -> SelectionResults:
        """Apply weighted best pick or Pareto marking; wrap :class:`SelectionResults`."""
        self._finalize_combined_objectives(results)
        if self.objective_mode == "pareto":
            self._mark_pareto_optimal(results)
            return SelectionResults(
                results=results, objective_mode="pareto",
            )
        best_info = self._find_best(results)
        if best_info is not None:
            best_name, _ = best_info
            for r in results:
                if r.model_name == best_name:
                    r.is_best = True
                    break
        else:
            print("\n  No combinations succeeded.")
        return SelectionResults(
            results=results, objective_mode="weighted",
        )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _make_result(self, **kwargs: Any) -> ModelResult:
        """Create a :class:`ModelResult` with instance-level custom prices."""
        result = ModelResult(**kwargs)
        result._custom_prices = self._custom_prices
        return result

    def _build_combo_result(
        self,
        combo_name: str,
        scores: List[float],
        latencies: List[float],
        dp_ids: List[str],
        costs: Optional[List[Optional[float]]] = None,
        attribute: str = "combination",
        is_best: bool = False,
    ) -> ModelResult:
        """Build a :class:`ModelResult` and absorb the samples into the normalizer.

        Centralizes the boilerplate selectors share for end-of-search result
        construction. Computes accuracy, average latency, datapoint results,
        token totals, and (when ``lambda_cost``/``lambda_latency`` are set)
        the mean combined objective using the current running normalizer.

        ``costs`` may be passed if the selector already cached per-sample costs
        during search; otherwise they are looked up from the tracker.
        """
        input_tokens, output_tokens = self._fetch_tokens(combo_name)
        accuracy, _ = self._compute_stats(scores)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        dp_results = (
            self._build_datapoint_results(scores, latencies, dp_ids) if dp_ids else []
        )
        if costs is None:
            costs = self._observe_combo(scores, latencies, dp_ids)
        elif self._pareto_exploration:
            self._absorb_pareto_gaps(scores, latencies, costs)
        else:
            self._absorb_observations(latencies, costs)
        combined = (
            self._mean_objective(scores, latencies, costs)
            if self._has_combined_objective
            else None
        )
        return self._make_result(
            model_name=combo_name,
            accuracy=accuracy,
            latency_seconds=avg_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attribute=attribute,
            is_best=is_best,
            datapoint_results=dp_results,
            combined_objective=combined,
        )

    def _build_datapoint_results(
        self, scores: List[float], latencies: List[float], datapoint_ids: List[str],
    ) -> List[DatapointResult]:
        """Build per-datapoint result objects with token attribution."""
        dp_tokens = self._fetch_tokens_by_datapoint(datapoint_ids)
        results: List[DatapointResult] = []
        for j, dp_id in enumerate(datapoint_ids):
            inp_tok, out_tok = dp_tokens.get(dp_id, ({}, {}))
            results.append(
                DatapointResult(
                    datapoint_index=j,
                    score=scores[j],
                    latency_seconds=latencies[j],
                    input_tokens=inp_tok,
                    output_tokens=out_tok,
                )
            )
        return results

    @staticmethod
    def _split_tokens(
        tokens: Dict[str, Tuple[int, int]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Split a per-model tokens dict into separate input/output dicts."""
        return (
            {k: v[0] for k, v in tokens.items()},
            {k: v[1] for k, v in tokens.items()},
        )

    @staticmethod
    def _find_best(results: List[ModelResult]) -> Optional[Tuple[str, float]]:
        """Find the best result.

        When any result has ``combined_objective`` set, picks the result with
        the highest combined objective (latency/cost tiebreakers within ties).
        Otherwise falls back to accuracy > latency > cost.
        """
        has_combined = any(r.combined_objective is not None for r in results)
        best = None
        best_primary = float("-inf")
        best_latency = float("inf")
        best_cost = float("inf")
        tol = 1e-9

        for r in results:
            primary = (
                r.combined_objective
                if has_combined and r.combined_objective is not None
                else r.accuracy
            )
            r_cost = r.price if r.price is not None else float("inf")
            if primary > best_primary + tol:
                best = r
                best_primary = primary
                best_latency = r.latency_seconds
                best_cost = r_cost
            elif (
                abs(primary - best_primary) <= tol
                and r.latency_seconds < best_latency - tol
            ):
                best = r
                best_primary = primary
                best_latency = r.latency_seconds
                best_cost = r_cost
            elif (
                abs(primary - best_primary) <= tol
                and abs(r.latency_seconds - best_latency) <= tol
                and r_cost < best_cost - tol
            ):
                best = r
                best_primary = primary
                best_latency = r.latency_seconds
                best_cost = r_cost

        return (best.model_name, best_primary) if best else None

    @staticmethod
    def _compute_stats(scores: List[float]) -> Tuple[float, float]:
        """Return (mean, sample_std) for a list of scores."""
        n = len(scores)
        if n == 0:
            return 0.0, 0.5
        mean = sum(scores) / n
        if n < 2:
            return mean, 0.5
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        return mean, math.sqrt(variance)

    @staticmethod
    def _compute_concurrency(max_concurrent: int, batch_size: int,) -> Tuple[int, int]:
        """Compute combo-level and datapoint-level concurrency limits.

        Splits ``max_concurrent`` into two tiers so the total number of
        in-flight API calls never exceeds the budget:

        * **Datapoint concurrency** (inner): how many datapoints within a
          single combo evaluate concurrently.  Set to
          ``min(max_concurrent, batch_size)`` — saturate the batch first.
        * **Combo concurrency** (outer): how many combos run in parallel.
          Set to ``max_concurrent // dp_concurrent`` — use remaining slots.

        The product ``n_combo * dp_concurrent <= max_concurrent`` always holds.

        Args:
            max_concurrent: Total concurrent API call budget across all
                combos and datapoints.
            batch_size: Number of datapoints to evaluate per combo in
                the current round.

        Returns:
            Tuple of ``(n_combo_parallel, dp_concurrent_per_combo)``.

        Examples:
            >>> BaseModelSelector._compute_concurrency(20, 5)
            (4, 5)   # 4 combos x 5 datapoints = 20 slots
            >>> BaseModelSelector._compute_concurrency(20, 100)
            (1, 20)  # batch is large, cap dp at 20, 1 combo at a time
            >>> BaseModelSelector._compute_concurrency(20, 1)
            (20, 1)  # 1 dp per combo, run 20 combos in parallel
        """
        if batch_size <= 0:
            return max(max_concurrent, 1), 1
        dp_concurrent = min(max(max_concurrent, 1), batch_size)
        n_combo = max(max_concurrent, 1) // dp_concurrent
        return n_combo, dp_concurrent

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """
        Select the best model combination.

        Calls :meth:`_run_selection` (implemented by each subclass) and
        ensures the tracker is stopped afterwards so that any disk cache
        is flushed.

        Args:
            parallel: If True, evaluate combinations concurrently.
            max_concurrent: Max total concurrent API calls across all
                combinations and datapoints.

        Returns:
            SelectionResults containing all model evaluation results.
        """
        record_offset = len(self._tracker.get_records())
        t0 = time.time()
        try:
            result = self._run_selection(parallel, max_concurrent)
        finally:
            self._tracker.stop()

        result.selection_wall_time_seconds = time.time() - t0

        # Cost: only non-cached calls made during this run
        input_tokens: Dict[str, int] = {}
        output_tokens: Dict[str, int] = {}
        for r in self._tracker.get_records()[record_offset:]:
            if r.cached:
                continue
            input_tokens[r.model] = input_tokens.get(r.model, 0) + r.prompt_tokens
            output_tokens[r.model] = output_tokens.get(r.model, 0) + r.completion_tokens
        result.selection_cost = compute_price(
            input_tokens, output_tokens, custom_prices=self._custom_prices,
        )
        if result.objective_mode is None:
            result.objective_mode = self.objective_mode

        return result

    @abstractmethod
    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Subclass hook — implement the actual selection algorithm."""
        ...
