"""
Model selection functionality.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

from ..base_models import EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults


class BruteForceModelSelector(BaseModelSelector):
    """
    Selects the best model for an agent by evaluating on a dataset.
    """

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
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
        )

    def select_best(
        self,
    ) -> SelectionResults:
        """
        Select the best model for each attribute/proxy.

        Returns:
            SelectionResults containing all model evaluation results
        """
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())

        # Cartesian product of all model candidates across all proxies
        all_combinations = list(itertools.product(*candidate_lists))

        all_results: List[ModelResult] = []
        best_combination = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9

        print(f"\n{'='*60}")
        print(f"Brute force: {len(all_combinations)} combinations")
        print(f"{'='*60}\n")

        for combo in all_combinations:
            # Set each proxy to its model in this combination
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            for proxy, model_obj in zip(proxies, combo):
                proxy.set_model(model_obj)

            try:
                accuracy, latency = self._evaluate(self.dataset)

                print(
                    f"✓ [{combo_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
                )

                all_results.append(
                    ModelResult(
                        model_name=combo_name,
                        accuracy=accuracy,
                        latency_seconds=latency,
                        attribute="combination",
                        is_best=False,
                    )
                )

                should_update = False
                if best_combination is None:
                    should_update = True
                elif accuracy > best_accuracy + accuracy_tolerance:
                    should_update = True
                elif (
                    abs(accuracy - best_accuracy) <= accuracy_tolerance
                    and latency < best_latency
                ):
                    should_update = True

                if should_update:
                    best_accuracy = accuracy
                    best_latency = latency
                    best_combination = combo

            except Exception as e:
                print(f"✗ [{combo_name}] failed: {e}")
                all_results.append(
                    ModelResult(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        attribute="combination",
                        is_best=False,
                    )
                )

        if best_combination is not None:
            best_name = " + ".join(self._get_model_name(m) for m in best_combination)
            # Set proxies to the best combination
            for proxy, model_obj in zip(proxies, best_combination):
                proxy.set_model(model_obj)
            print(
                f"\n🏆 Best combination: {best_name} (accuracy: {best_accuracy:.2%}, latency: {best_latency:.2f}s)"
            )

            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            print("\n✗ No combinations succeeded")

        return SelectionResults(results=all_results)
