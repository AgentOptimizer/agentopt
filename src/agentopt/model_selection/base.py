"""
Base classes and result types for model selection.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from ..types import AccuracyFn, EvaluationTask
from ..invoker.base import InvokerProtocol
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
        invoker: InvokerProtocol,
        models: Dict[ModelProxy, List[Any]],
        accuracy_fn: AccuracyFn,
        dataset_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            invoker: Wrapper with invoke() method for running the agent
            models: Dictionary mapping ModelProxy to list of model candidates
            accuracy_fn: Function (expected, actual) -> bool
            dataset_dir: Optional path to dataset directory
        """
        self.invoker = invoker
        self.accuracy_fn = accuracy_fn
        self.dataset_dir = dataset_dir
        self._models = models

    @abstractmethod
    def select_best(
        self,
        evaluation_tasks: Optional[List[EvaluationTask]] = None,
    ) -> SelectionResults:
        """
        Select the best model for each attribute/proxy.

        Args:
            evaluation_tasks: Optional list of tasks to evaluate on.

        Returns:
            SelectionResults containing all model evaluation results
        """
        ...
