"""
Base classes and result types for model selection.
"""

import asyncio
import copy
import inspect
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..base_models import EvalFn
from ..model_proxy import ModelProxy

logger = logging.getLogger(__name__)

# Model name fields to check on LLM objects, in priority order
_MODEL_FIELDS = ("model", "model_name", "model_id")

# Attribute names to check for the LLM on an agent, in priority order
_AGENT_LLM_ATTRS = ("llm",)


class ModelResult(BaseModel):
    """Result of evaluating a single model."""

    model_name: str
    accuracy: float
    latency_seconds: float
    attribute: str
    is_best: bool = False


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
            "attribute",
            "is_best",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.model_dump())


class BaseModelSelector(ABC):
    """Abstract base class for model selectors."""

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: List[Tuple[Any, str]],
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            models: Dictionary mapping ModelProxy to list of model candidates
            eval_fn: Function (expected, actual) -> bool | float (higher is better)
            dataset: List of (input_data, expected_answer) tuples, pre-loaded by the user
            agent: agent class for agent framework that we support: Langchain, Langgraph, CrewAI, etc
            invoke_fn: callable for customized agent
        """
        if agent is None and invoke_fn is None:
            raise ValueError("Either 'agent' or 'invoke_fn' must be provided")
        if agent is not None and invoke_fn is not None:
            raise ValueError(
                "Only one of 'agent' or 'invoke_fn' should be provided, not both"
            )

        self.agent = agent
        self.eval_fn = eval_fn
        self.dataset = dataset
        self._models = models

        # Resolve invoke_fn from agent if not provided directly
        if invoke_fn is not None:
            self.invoke_fn = invoke_fn
            self.is_async = inspect.iscoroutinefunction(invoke_fn)
            self._invoke_method_name = None
        elif hasattr(agent, "kickoff"):
            # CrewAI agents use .kickoff()
            self.invoke_fn = agent.kickoff
            self.is_async = False
            self._invoke_method_name = "kickoff"
        elif hasattr(agent, "invoke"):
            # LangChain and LangGraph agents use .invoke()
            self.invoke_fn = agent.invoke
            self.is_async = False
            self._invoke_method_name = "invoke"
        elif hasattr(agent, "run"):
            # LlamaIndex agents use .run() (async)
            self.invoke_fn = agent.run
            self.is_async = inspect.iscoroutinefunction(agent.run)
            self._invoke_method_name = "run"
        else:
            raise TypeError(
                f"Unsupported agent type: {type(agent).__name__}. "
                "Pass 'invoke_fn' directly instead."
            )

    def _evaluate(
        self,
        evaluation_tasks: List[Tuple[Any, str]],
        label: str = "",
    ) -> Tuple[float, float]:
        """
        Evaluate the current state of the agent against a list of tasks.

        Args:
            evaluation_tasks: List of (input_data, expected_answer) tuples
            label: Display label for progress traces.

        Returns:
            Tuple of (score, avg_latency_seconds)
        """
        total_score = 0.0
        total = len(evaluation_tasks)
        total_latency = 0.0
        prefix = f"    [{label}]" if label else "   "

        for i, (input_data, expected_answer) in enumerate(evaluation_tasks, 1):
            try:
                print(f"{prefix} sample {i}/{total} ...", end="", flush=True)
                start_time = time.time()
                if self.is_async:
                    actual_result = asyncio.run(self.invoke_fn(input_data))
                else:
                    actual_result = self.invoke_fn(input_data)
                latency = time.time() - start_time
                total_latency += latency

                score = self.eval_fn(expected_answer, actual_result)
                total_score += float(score)
                print(f" score={float(score):.2f}  ({latency:.1f}s)")

            except Exception as e:
                print(f" error: {e}")

        avg_score = total_score / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0

        return avg_score, avg_latency

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
    def _clone_agent(agent: Any, llm_updates: Dict[str, Any]) -> Any:
        """Create an independent copy of the agent with LLMs replaced.

        Pydantic agents: model_copy(deep=False) — avoids deepcopy of HTTP
        clients with thread locks.
        Other agents: copy.deepcopy fallback.
        """
        if isinstance(agent, BaseModel):
            return agent.model_copy(update=llm_updates, deep=False)
        else:
            agent_copy = copy.deepcopy(agent)
            for attr, llm in llm_updates.items():
                setattr(agent_copy, attr, llm)
            return agent_copy

    @staticmethod
    def _create_variant(original_llm: Any, model_name: str) -> Any:
        """Create a fresh LLM with a different model name, preserving settings."""
        if isinstance(original_llm, BaseModel):
            target_field = next(
                (f for f in _MODEL_FIELDS if f in type(original_llm).model_fields),
                None,
            )
            if target_field:
                return original_llm.model_copy(update={target_field: model_name})
        else:
            target_field = next(
                (f for f in _MODEL_FIELDS if hasattr(original_llm, f)),
                None,
            )
            if target_field:
                new_llm = copy.deepcopy(original_llm)
                setattr(new_llm, target_field, model_name)
                return new_llm

        raise TypeError(
            f"Cannot create variant of {type(original_llm).__name__}: "
            f"no supported model field found (checked: {_MODEL_FIELDS})"
        )

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
                    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
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
        dataset: List[Tuple[Any, str]],
        label: str = "",
    ) -> Tuple[float, float]:
        """Evaluate an agent against the dataset. Thread-safe."""
        total_score = 0.0
        total = len(dataset)
        total_latency = 0.0
        prefix = f"    [{label}]" if label else "   "

        for i, (input_data, expected_answer) in enumerate(dataset, 1):
            try:
                print(f"{prefix} sample {i}/{total} ...", end="", flush=True)
                start_time = time.time()
                if is_async:
                    actual_result = asyncio.run(invoke_fn(input_data))
                else:
                    actual_result = invoke_fn(input_data)
                latency = time.time() - start_time
                total_latency += latency
                score = eval_fn(expected_answer, actual_result)
                total_score += float(score)
                print(f" score={float(score):.2f}  ({latency:.1f}s)")
            except Exception as e:
                print(f" error: {e}")

        avg_score = total_score / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0
        return avg_score, avg_latency

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

    def _detect_proxy_attrs(
        self,
        proxies: List[ModelProxy],
    ) -> Dict[int, str]:
        """Detect which agent attribute each proxy corresponds to."""
        mapping: Dict[int, str] = {}
        for proxy in proxies:
            proxy_model = proxy.get_model()
            for attr in _AGENT_LLM_ATTRS:
                if hasattr(self.agent, attr):
                    val = getattr(self.agent, attr)
                    if val is proxy or val is proxy_model:
                        mapping[id(proxy)] = attr
                        break
        return mapping

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