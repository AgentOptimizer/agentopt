"""
Model selection functionality.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..model_proxy import ModelProxy
from ..load_dataset import load_dataset
from ..types import AccuracyFn, EvaluationTask
from ..invoker.base import InvokerProtocol
from .base import BaseModelSelector, ModelResult, SelectionResults


class ModelSelector(BaseModelSelector):
    """
    Selects the best model for an agent by evaluating on a dataset.
    """

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
        super().__init__(
            invoker=invoker,
            models=models,
            accuracy_fn=accuracy_fn,
            dataset_dir=dataset_dir,
        )

    def select_best(
        self,
        evaluation_tasks: Optional[List[EvaluationTask]] = None,
    ) -> SelectionResults:
        """
        Select the best model for each attribute/proxy.

        Args:
            evaluation_tasks: Optional list of tasks to evaluate on.
                             If None and dataset_dir is provided, loads from dataset_dir.
                             If both None, uses a default evaluation.

        Returns:
            SelectionResults containing all model evaluation results
        """
        # Load tasks from dataset_dir if not provided
        if evaluation_tasks is None and self.dataset_dir:
            evaluation_tasks = load_dataset(self.dataset_dir)

        all_results: List[ModelResult] = []

        # Process each proxy
        for i, (proxy, model_candidates) in enumerate(self._models.items()):
            results = self._select_for_target(
                model_candidates=model_candidates,
                display_name=f"llm_{i}" if len(self._models) > 1 else "llm",
                set_model_fn=proxy.set_model,
                evaluation_tasks=evaluation_tasks,
            )
            all_results.extend(results)

        return SelectionResults(results=all_results)

    def _select_for_target(
        self,
        model_candidates: List[Any],
        display_name: str,
        set_model_fn: Any,
        evaluation_tasks: Optional[List[EvaluationTask]],
    ) -> List[ModelResult]:
        """Select best model for a single proxy."""
        results: List[ModelResult] = []
        best_model: Optional[Any] = None
        best_score = float("inf")
        best_model_name: Optional[str] = None

        print(f"\n{'='*60}")
        print(f"Selecting best model for: {display_name}")
        print(f"{'='*60}\n")

        for model_obj in model_candidates:
            model_name = self._get_model_name(model_obj)
            set_model_fn(model_obj)

            try:
                accuracy, latency = self._evaluate_model(model_obj, evaluation_tasks)

                print(
                    f"✓ Model: {model_name}, Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
                )

                results.append(
                    ModelResult(
                        model_name=model_name,
                        accuracy=accuracy,
                        latency_seconds=latency,
                        attribute=display_name,
                        is_best=False,
                    )
                )

                score = 1 - accuracy
                if score < best_score:
                    best_score = score
                    best_model = model_obj
                    best_model_name = model_name

            except Exception as e:
                print(f"✗ Model {model_name} failed: {e}")
                results.append(
                    ModelResult(
                        model_name=model_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        attribute=display_name,
                        is_best=False,
                    )
                )
                continue

        if best_model_name and best_model is not None:
            set_model_fn(best_model)
            print(
                f"\n🏆 Best model for {display_name}: {best_model_name} (accuracy: {1-best_score:.2%})"
            )

            # Mark best model
            for result in results:
                if result.model_name == best_model_name:
                    result.is_best = True
                    break
        else:
            print(f"\n✗ No models succeeded for {display_name}")

        return results

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

    def _evaluate_model(
        self,
        model_obj: Any,
        evaluation_tasks: Optional[List[EvaluationTask]] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate a model using accuracy metric and measure latency.

        Args:
            model_obj: The model object being evaluated
            evaluation_tasks: List of tasks to evaluate on

        Returns:
            Tuple of (accuracy, latency_seconds)
        """
        if evaluation_tasks is None:
            if self.dataset_dir:
                evaluation_tasks = load_dataset(self.dataset_dir)
            else:
                evaluation_tasks = [
                    EvaluationTask(
                        messages=[{"role": "user", "content": "what is 2 + 2?"}],
                        expected_answer="4",
                    )
                ]

        correct = 0
        total = len(evaluation_tasks)
        total_latency = 0.0

        for task in evaluation_tasks:
            try:
                start_time = time.time()
                result = self.invoker.invoke(task.to_invoke_dict())
                latency = time.time() - start_time
                total_latency += latency

                expected_answer = task.expected_answer
                actual_answer = self._extract_answer(result)

                if expected_answer:
                    if self.accuracy_fn(expected_answer, actual_answer):
                        correct += 1
                else:
                    if actual_answer:
                        correct += 1

            except Exception:
                pass

        accuracy = correct / total if total > 0 else 0.0
        avg_latency = total_latency / total if total > 0 else 0.0

        return accuracy, avg_latency

    def _extract_answer(self, result: Dict[str, Any]) -> str:
        """Extract answer string from agent result."""
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return str(last_message.content)
                elif isinstance(last_message, dict):
                    return str(last_message.get("content", ""))
                else:
                    return str(last_message)
            return ""
        return str(result)
