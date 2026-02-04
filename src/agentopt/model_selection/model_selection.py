"""
Model selection functionality.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..core import bind_model
from ..load_dataset import load_dataset
from ..model_factory import normalize_models
from ..types import (
    AccuracyFn,
    AgentProtocol,
    EvaluationTask,
    ModelResult,
    ModelsConfig,
    SelectionResults,
)


class ModelSelector:
    """
    Selects the best model for an agent by evaluating on a dataset.
    """

    def __init__(
        self,
        agent: AgentProtocol,
        models: ModelsConfig,
        accuracy_fn: AccuracyFn,
        dataset_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the model selector.

        Args:
            agent: The agent instance to optimize (must implement AgentProtocol)
            models: Dictionary mapping attribute names to list of model objects or model name strings
                    e.g., {"._model": ["openai/gpt-4o-mini", "openai/gpt-4o"]}
            accuracy_fn: Function that takes (expected_answer, actual_output) and returns True if correct
            dataset_dir: Optional path to evaluation dataset directory
        """
        self.agent = agent
        self.accuracy_fn = accuracy_fn
        self.dataset_dir = dataset_dir

        # Pass through strings - each agent wrapper handles conversion
        self.models = normalize_models(models)

        # Store model objects
        self._model_objects: Dict[str, List[Any]] = {}
        for attr_name, model_objects in self.models.items():
            self._model_objects[attr_name] = model_objects

    def select_best(
        self,
        evaluation_tasks: Optional[List[EvaluationTask]] = None,
    ) -> SelectionResults:
        """
        Select the best model for each attribute.

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

        for attr_name, model_objects in self._model_objects.items():
            best_model: Optional[Any] = None
            best_score = float("inf")
            best_model_name: Optional[str] = None

            # Display name (strip leading dot for readability)
            display_name = attr_name.lstrip(".")

            print(f"\n{'='*60}")
            print(f"Selecting best model for attribute: {display_name}")
            print(f"{'='*60}\n")

            for model_obj in model_objects:
                model_name = self._get_model_name(model_obj)
                bind_model(self.agent, attr_name, model_obj)

                try:
                    accuracy, latency = self._evaluate_model(
                        model_obj, evaluation_tasks
                    )

                    print(
                        f"✓ Model: {model_name}, Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
                    )

                    all_results.append(
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
                    all_results.append(
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
                bind_model(self.agent, attr_name, best_model)
                print(
                    f"\n🏆 Best model for {display_name}: {best_model_name} (accuracy: {1-best_score:.2%})"
                )

                # Mark best model
                for result in all_results:
                    if (
                        result.attribute == display_name
                        and result.model_name == best_model_name
                    ):
                        result.is_best = True
                        break
            else:
                print(f"\n✗ No models succeeded for {display_name}")

        return SelectionResults(results=all_results)

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
                result = self.agent.invoke(task.to_invoke_dict())
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
