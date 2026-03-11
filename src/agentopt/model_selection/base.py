"""
Base classes and result types for model selection.
"""

import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..base_models import Dataset, EvalFn, validate_dataset
from ..model_proxy import ModelProxy
from ..model_proxy.constants import validate_model_candidates

logger = logging.getLogger(__name__)


class ModelResult(BaseModel):
    """Result of evaluating a single model."""

    model_name: str
    accuracy: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    attribute: str
    is_best: bool = False

    def __str__(self) -> str:
        return (
            f"{self.model_name} (accuracy: {self.accuracy:.2%}, "
            f"latency: {self.latency_seconds:.2f}s, "
            f"in_tokens: {self.input_tokens}, out_tokens: {self.output_tokens})"
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

    def get_by_attribute(self, attribute: str) -> List[ModelResult]:
        """Get all results for a specific attribute."""
        return [r for r in self.results if r.attribute == attribute]

    def to_csv(self, path: str) -> None:
        """Save results to CSV file."""
        import csv

        if not self.results:
            return

        fieldnames = [
            "model_name",
            "accuracy",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "attribute",
            "is_best",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.model_dump())

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

        def fmt_tok(r: ModelResult) -> str:
            return f"{r.input_tokens}/{r.output_tokens}"

        # Compute column widths.
        rank_h, model_h, acc_h, lat_h, tok_h = (
            "Rank",
            "Model",
            "Accuracy",
            "Latency",
            "Tokens (in/out)",
        )
        rank_w = max(len(rank_h), len(str(len(unique))))
        model_w = max(len(model_h), *(len(r.model_name) for r in unique))
        acc_w = max(len(acc_h), *(len(fmt_acc(r.accuracy)) for r in unique))
        lat_w = max(len(lat_h), *(len(fmt_lat(r.latency_seconds)) for r in unique))
        tok_w = max(len(tok_h), *(len(fmt_tok(r)) for r in unique))

        # Row builder.
        marker = ">>>"
        pad = " " * len(marker)

        def row(
            rank_s: str, model_s: str, acc_s: str, lat_s: str, tok_s: str, best: bool
        ) -> str:
            prefix = marker if best else pad
            return (
                f"{prefix} {rank_s:>{rank_w}}  "
                f"{model_s:<{model_w}}  "
                f"{acc_s:>{acc_w}}  "
                f"{lat_s:>{lat_w}}  "
                f"{tok_s:>{tok_w}}"
            )

        header_row = row(rank_h, model_h, acc_h, lat_h, tok_h, False)
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
                    fmt_tok(r),
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
                f"in_tokens: {best_result.input_tokens}, out_tokens: {best_result.output_tokens})"
            )
        lines.append("")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary table of all results."""
        print(self)


class BaseModelSelector(ABC):
    """Abstract base class for model selectors."""

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            models: Dictionary mapping ModelProxy to list of model candidates
            eval_fn: Function (expected, actual) -> bool | float (higher is better)
            dataset: Sequence of (input_data, expected_answer) pairs, pre-loaded
                by the user.  Any object supporting len() and indexing that yields
                2-element tuples works.
            agent: Agent object for a supported framework (CrewAI, LangChain,
                LlamaIndex, OpenAI SDK).  Mutually exclusive with *invoke_fn*.
            invoke_fn: Callable for a custom agent or multi-agent chain.
                Mutually exclusive with *agent*.
        """
        if agent is None and invoke_fn is None:
            raise ValueError("Either 'agent' or 'invoke_fn' must be provided")
        if agent is not None and invoke_fn is not None:
            raise ValueError(
                "Only one of 'agent' or 'invoke_fn' should be provided, not both"
            )

        validate_dataset(dataset)
        self.agent = agent
        self.eval_fn = eval_fn
        self.dataset = dataset
        self._models = models
        self._proxies = list(models.keys())

        # Validate API keys for all candidate models.
        warnings, _ = validate_model_candidates(models)
        for warning in warnings:
            raise ValueError(warning)

        self._token_tracker = None
        if invoke_fn is not None:
            self.invoke_fn = invoke_fn
            self.is_async = inspect.iscoroutinefunction(invoke_fn)
            self._invoke_method_name = None
            self._adapter = None
            # Detect adapter from proxy's underlying model for token tracking.
            from ..model_proxy.adapter import get_adapter_for_model

            for proxy in self._proxies:
                # Check the underlying model first for a more specific match,
                # then fall back to the proxy itself.  This prevents the OpenAI
                # SDK adapter (which registers ModelProxy as a virtual subclass
                # of Model) from shadowing framework-specific adapters.
                adapter = None
                try:
                    underlying = object.__getattribute__(proxy, "_optmodel")
                    adapter = get_adapter_for_model(underlying)
                except AttributeError:
                    pass
                if adapter is None:
                    adapter = get_adapter_for_model(proxy)
                if adapter is not None:
                    self._adapter = adapter
                    tracker = adapter.create_token_tracker()
                    self._token_tracker = tracker
                    for p in self._proxies:
                        p._set_token_tracker(tracker)
                        adapter.register_with_proxy(p, None, self._proxies)
                    self.invoke_fn = adapter.wrap_invoke_fn_with_tracker(
                        self.invoke_fn, tracker
                    )
                    break
        else:
            # Auto-detect framework via adapter registry.
            from ..model_proxy.adapter import get_adapter

            adapter = get_adapter(agent)
            if adapter is None:
                raise TypeError(
                    f"Unsupported agent type: {type(agent).__name__}. "
                    "Pass 'invoke_fn' directly instead."
                )

            self._adapter = adapter
            tracker = adapter.create_token_tracker(agent)
            self._token_tracker = tracker
            self.invoke_fn = adapter.get_invoke_fn(agent)
            self.is_async = False  # all adapters wrap async internally
            self._invoke_method_name = getattr(adapter, "invoke_method_name", None)

            # Auto-register proxy ↔ agent sync for sequential evaluation.
            for proxy in self._proxies:
                proxy._set_token_tracker(tracker)
                adapter.register_with_proxy(proxy, agent, self._proxies)

    def _evaluate(
        self,
        evaluation_tasks: Dataset,
        label: str = "",
    ) -> Tuple[float, float, int, int]:
        """
        Evaluate the current state of the agent against a list of tasks.

        Args:
            evaluation_tasks: Sequence of (input_data, expected_answer) pairs
            label: Display label for progress traces.

        Returns:
            Tuple of (score, avg_latency_seconds, total_input_tokens, total_output_tokens)
        """
        total_score = 0.0
        total = len(evaluation_tasks)
        total_latency = 0.0

        for i, (input_data, expected_answer) in enumerate(evaluation_tasks, 1):
            try:
                start_time = time.time()
                if self.is_async:
                    actual_result = asyncio.run(self.invoke_fn(input_data))
                else:
                    actual_result = self.invoke_fn(input_data)
                latency = time.time() - start_time
                total_latency += latency

                score = self.eval_fn(expected_answer, actual_result)
                total_score += float(score)

            except Exception as e:
                logger.warning("[%s] sample %d/%d error: %s", label, i, total, e)

        avg_score = total_score / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0

        in_tok, out_tok = self._token_tracker.reset() if self._token_tracker else (0, 0)

        return avg_score, avg_latency, in_tok, out_tok

    def _get_model_name(self, model_obj: Any) -> str:
        """Extract model name from model object for display purposes."""
        if isinstance(model_obj, str):
            return model_obj
        elif hasattr(model_obj, "model_name"):
            return str(model_obj.model_name)
        elif hasattr(model_obj, "model"):
            return str(model_obj.model)
        else:
            return model_obj.__class__.__name__

    # ------------------------------------------------------------------
    # Parallel evaluation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_invoke_fn(agent_copy: Any, method_name: str, is_async: bool) -> Callable:
        """Create an invocation callable for a copied agent.

        For .run() (LlamaIndex), wraps in asyncio.run() with async wrapper.
        For .kickoff() and .invoke(), returns the method directly.
        """
        method = getattr(agent_copy, method_name)

        if method_name == "run":

            def _invoke(input_data):
                async def _async_run():
                    if isinstance(input_data, dict):
                        result = method(**input_data)
                    else:
                        result = method(input_data)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                return asyncio.run(_async_run())

            return _invoke
        else:
            return method

    @staticmethod
    def _evaluate_single(
        invoke_fn: Callable,
        is_async: bool,
        eval_fn: EvalFn,
        dataset: Dataset,
        token_tracker: Any = None,
        label: str = "",
    ) -> Tuple[float, float, int, int]:
        """Evaluate an agent against the dataset. Thread-safe."""
        total_score = 0.0
        total = len(dataset)
        total_latency = 0.0

        for i, (input_data, expected_answer) in enumerate(dataset, 1):
            try:
                start_time = time.time()
                if is_async:
                    actual_result = asyncio.run(invoke_fn(input_data))
                else:
                    actual_result = invoke_fn(input_data)
                latency = time.time() - start_time
                total_latency += latency
                score = eval_fn(expected_answer, actual_result)
                total_score += float(score)
            except Exception as e:
                logger.debug("[%s] sample %d/%d error: %s", label, i, total, e)

        avg_score = total_score / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0
        in_tok, out_tok = token_tracker.reset() if token_tracker is not None else (0, 0)
        return avg_score, avg_latency, in_tok, out_tok

    @staticmethod
    def _find_best(results: List[ModelResult]) -> Optional[Tuple[str, float]]:
        """Find the best result by accuracy (ties broken by latency)."""
        best = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9

        for r in results:
            if r.accuracy > best_accuracy + tol:
                best = r
                best_accuracy = r.accuracy
                best_latency = r.latency_seconds
            elif (
                abs(r.accuracy - best_accuracy) <= tol
                and r.latency_seconds < best_latency
            ):
                best = r
                best_accuracy = r.accuracy
                best_latency = r.latency_seconds

        return (best.model_name, best.accuracy) if best else None

    @abstractmethod
    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """
        Select the best model for each attribute/proxy.

        Args:
            parallel: If True, evaluate model combinations concurrently.
            max_workers: Max threads for parallel mode. Defaults to number of combinations.

        Returns:
            SelectionResults containing all model evaluation results
        """
        ...
