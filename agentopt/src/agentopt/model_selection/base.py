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

from agentproxy import LLMTracker

from ..base_models import (
    Dataset,
    EvalFn,
    ModelCandidate,
    ModelsConfig,
    validate_dataset,
)
from ..model_price import compute_price

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
    datapoint_results: List[DatapointResult] = Field(default_factory=list)
    _custom_prices: Optional[Dict[str, Tuple[float, float]]] = PrivateAttr(default=None)

    @property
    def total_input_tokens(self) -> int:
        return sum(self.input_tokens.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self.output_tokens.values())

    @property
    def price(self) -> Optional[float]:
        """Total cost in USD, or ``None`` if pricing is unavailable."""
        return compute_price(
            self.input_tokens, self.output_tokens, custom_prices=self._custom_prices
        )

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

    def __str__(self) -> str:
        if not self.results:
            return "No results."

        # Deduplicate by model_name, preferring entries with is_best=True.
        seen: Dict[str, ModelResult] = {}
        for r in self.results:
            if r.model_name not in seen or (
                r.is_best and not seen[r.model_name].is_best
            ):
                seen[r.model_name] = r
        unique = list(seen.values())

        # Sort: best accuracy first, ties broken by lowest latency.
        unique.sort(key=lambda r: (-r.accuracy, r.latency_seconds))

        # Format helpers.
        def fmt_acc(v: float) -> str:
            return f"{v:.2%}"

        def fmt_lat(v: float) -> str:
            return f"{v:.2f}s"

        def fmt_price(r: ModelResult) -> str:
            p = r.price
            return f"${p:.6f}" if p is not None else "N/A"

        # Compute column widths.
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

        # Row builder.
        marker = ">>>"
        pad = " " * len(marker)

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

        best_result = next((r for r in unique if r.is_best), None)
        if best_result:
            lines.append(
                f"{pad} Best: {best_result.model_name} "
                f"(accuracy: {best_result.accuracy:.2%}, "
                f"latency: {best_result.latency_seconds:.2f}s, "
                f"price: {fmt_price(best_result)})"
            )
        lines.append("")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary table of all results."""
        print(self)


class BaseModelSelector(ABC):
    """Abstract base class for model selectors.

    Uses the factory pattern: ``agent_fn(combo_dict)`` returns a runnable
    agent for each model combination. No ModelProxy or framework adapters.
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: ModelsConfig,
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        node_descriptions: Optional[Dict[str, str]] = None,
        tracker: Optional[LLMTracker] = None,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            agent_fn: Factory function ``(combo_dict) -> agent``. Takes a dict
                mapping node names to model candidates (string names or
                framework-specific model instances) and returns a runnable
                agent.
            models: Dict mapping node names to candidate model specs, e.g.
                ``{"planner": ["gpt-4o", "gpt-4o-mini"]}`` or prebuilt
                LLM instances.
            eval_fn: Function ``(expected, actual) -> bool | float``
                (higher is better).
            dataset: Sequence of ``(input_data, expected_answer)`` pairs.
            invoke_fn: Optional callable ``(agent, input_data) -> result``.
                If not provided, the agent is called directly as
                ``agent(input_data)``.
            model_prices: Optional custom pricing overrides. Maps model names
                to dicts with ``'input_price'`` and ``'output_price'`` keys
                ($/MTok).
            node_descriptions: Optional dict mapping node names to human-readable
                descriptions of what each node does, e.g.
                ``{"planner": "Decomposes queries into sub-tasks"}``.
            tracker: Optional :class:`LLMTracker` instance. If not provided,
                one is created and started automatically.
        """
        validate_dataset(dataset)

        self._custom_prices: Optional[Dict[str, Tuple[float, float]]] = (
            {
                name: (d["input_price"], d["output_price"])
                for name, d in model_prices.items()
            }
            if model_prices
            else None
        )

        self.agent_fn = agent_fn
        self.eval_fn = eval_fn
        self.dataset = dataset
        self._models = models
        self._node_names = list(models.keys())
        self.invoke_fn = invoke_fn
        self.model_prices = model_prices
        self.node_descriptions = node_descriptions

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
        """Call agent with input_data, using invoke_fn if provided."""
        if self.invoke_fn is not None:
            return self.invoke_fn(agent, input_data)
        return agent(input_data)

    def _evaluate_combo(
        self,
        combo: Dict[str, ModelCandidate],
        evaluation_tasks: Dataset,
        label: str = "",
    ) -> Tuple[List[float], List[float], List[str]]:
        """Build an agent for combo and evaluate it on tasks.

        Returns (scores, latencies, datapoint_ids).
        """
        agent = self.agent_fn(combo)
        return self._evaluate_agent(agent, evaluation_tasks, label)

    def _evaluate_agent(
        self, agent: Any, evaluation_tasks: Dataset, label: str = "",
    ) -> Tuple[List[float], List[float], List[str]]:
        """Evaluate a pre-built agent against tasks.

        Returns (scores, latencies, datapoint_ids).
        """
        scores: List[float] = []
        latencies: List[float] = []
        datapoint_ids: List[str] = []
        total = len(evaluation_tasks)

        for i, (input_data, expected_answer) in enumerate(evaluation_tasks, 1):
            dp_id = f"{label}::dp_{i}"
            try:
                with self._tracker.track(data_id=dp_id, combo_id=label):
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
    ) -> Tuple[List[float], List[float], List[str]]:
        """Build an agent for combo and evaluate async.

        Returns (scores, latencies, datapoint_ids).
        """
        agent = self.agent_fn(combo)
        return await self._evaluate_agent_async(
            agent, evaluation_tasks, label, max_concurrent
        )

    async def _evaluate_agent_async(
        self,
        agent: Any,
        evaluation_tasks: Dataset,
        label: str = "",
        max_concurrent: int = 20,
    ) -> Tuple[List[float], List[float], List[str]]:
        """Evaluate a pre-built agent asynchronously.

        Returns (scores, latencies, datapoint_ids).
        """
        import contextvars as _ctx

        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(evaluation_tasks)
        results: List[Optional[dict]] = [None] * total

        is_async = inspect.iscoroutinefunction(
            self.invoke_fn if self.invoke_fn else getattr(agent, "__call__", None)
        )

        async def _eval_single(idx: int, input_data: Any, expected_answer: Any) -> None:
            dp_id = f"{label}::dp_{idx + 1}"
            async with semaphore:
                try:
                    with self._tracker.track(data_id=dp_id, combo_id=label):
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
    # Token tracking via agentproxy
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
    # Result helpers
    # ------------------------------------------------------------------

    def _make_result(self, **kwargs: Any) -> ModelResult:
        """Create a :class:`ModelResult` with instance-level custom prices."""
        result = ModelResult(**kwargs)
        result._custom_prices = self._custom_prices
        return result

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
        """Find best result by accuracy > latency > cost tiebreaker."""
        best = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        best_cost = float("inf")
        tol = 1e-9

        for r in results:
            r_cost = r.price if r.price is not None else float("inf")
            if r.accuracy > best_accuracy + tol:
                best = r
                best_accuracy = r.accuracy
                best_latency = r.latency_seconds
                best_cost = r_cost
            elif (
                abs(r.accuracy - best_accuracy) <= tol
                and r.latency_seconds < best_latency - tol
            ):
                best = r
                best_accuracy = r.accuracy
                best_latency = r.latency_seconds
                best_cost = r_cost
            elif (
                abs(r.accuracy - best_accuracy) <= tol
                and abs(r.latency_seconds - best_latency) <= tol
                and r_cost < best_cost - tol
            ):
                best = r
                best_accuracy = r.accuracy
                best_latency = r.latency_seconds
                best_cost = r_cost

        return (best.model_name, best.accuracy) if best else None

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

    @abstractmethod
    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """
        Select the best model combination.

        Args:
            parallel: If True, evaluate combinations concurrently.
            max_concurrent: Max concurrent API calls per combination.

        Returns:
            SelectionResults containing all model evaluation results.
        """
        ...
