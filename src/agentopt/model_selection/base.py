"""
Base classes and result types for model selection.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import time
import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple

from ..base_models import EvalFn
from ..model_proxy import ModelProxy


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
        invoke_fn: Optional[callable] = None,
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
        elif hasattr(agent, "kickoff"):
            # CrewAI agents use .kickoff()
            self.invoke_fn = agent.kickoff
            self.is_async = False
        elif hasattr(agent, "invoke"):
            # LangChain and LangGraph agents use .invoke()
            self.invoke_fn = agent.invoke
            self.is_async = False
        elif hasattr(agent, "run"):
            # LlamaIndex agents use .run() (async)
            self.invoke_fn = agent.run
            self.is_async = inspect.iscoroutinefunction(agent.run)
        else:
            raise TypeError(
                f"Unsupported agent type: {type(agent).__name__}. "
                "Pass 'invoke_fn' directly instead."
            )

    def _evaluate(
        self,
        evaluation_tasks: List[Tuple[Any, str]],
    ) -> Tuple[float, float]:
        """
        Evaluate the current state of the agent against a list of tasks.

        Args:
            evaluation_tasks: List of (input_data, expected_answer) tuples

        Returns:
            Tuple of (score, avg_latency_seconds)
        """
        total_score = 0.0
        total = len(evaluation_tasks)
        total_latency = 0.0

        for input_data, expected_answer in evaluation_tasks:
            try:
                start_time = time.time()
                if self.is_async:
                    # Handle async invocation (e.g., LlamaIndex .run())
                    actual_result = asyncio.run(self.invoke_fn(input_data))
                else:
                    actual_result = self.invoke_fn(input_data)
                latency = time.time() - start_time
                total_latency += latency

                score = self.eval_fn(expected_answer, actual_result)
                # bool -> 1.0/0.0, float passes through
                total_score += float(score)

            except Exception:
                pass

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

    @abstractmethod
    def select_best(self) -> SelectionResults:
        """
        Select the best model for each attribute/proxy.

        Returns:
            SelectionResults containing all model evaluation results
        """
        ...
