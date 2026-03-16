"""
Brute-force model selection: evaluates the Cartesian product of all
candidate models across all nodes.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn
from .base import BaseModelSelector, ModelResult, SelectionResults


class BruteForceModelSelector(BaseModelSelector):
    """
    Selects the best model combination by evaluating all combinations.

    Supports sequential and async-parallel evaluation via ``select_best()``.
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, str]], Any],
        models: Dict[str, List[str]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        super().__init__(
            agent_fn=agent_fn,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
        )

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    def _select_sequential(self) -> SelectionResults:
        all_combos = self._all_combos()

        all_results: List[ModelResult] = []
        best_combo_name = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'='*60}")
        print(f"Brute force (sequential): {len(all_combos)} combinations")
        print(f"{'='*60}\n")

        for idx, combo in enumerate(all_combos, 1):
            combo_name = self._combo_name(combo)
            print(f"  [{idx}/{len(all_combos)}] Evaluating: {combo_name}")

            try:
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, self.dataset, label=combo_name
                )
                input_tokens, output_tokens = self._fetch_tokens(combo_name)
                accuracy, _ = self._compute_stats(scores)
                latency = sum(latencies) / len(latencies) if latencies else 0.0
                dp_results = self._build_datapoint_results(scores, latencies, dp_ids)

                result = self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attribute="combination",
                    is_best=False,
                    datapoint_results=dp_results,
                )
                print(f"  {result}")
                all_results.append(result)

                if (accuracy > best_accuracy + tol) or (
                    abs(accuracy - best_accuracy) <= tol and latency < best_latency
                ):
                    best_accuracy = accuracy
                    best_latency = latency
                    best_combo_name = combo_name

            except Exception as e:
                print(f"  [{combo_name}] failed: {e}")
                all_results.append(
                    self._make_result(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        input_tokens={},
                        output_tokens={},
                        attribute="combination",
                        is_best=False,
                    )
                )

        if best_combo_name is not None:
            for result in all_results:
                if result.model_name == best_combo_name:
                    result.is_best = True
                    break
        else:
            print("\n  No combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()

        print(f"\n{'='*60}")
        print(
            f"Brute force (async): {len(all_combos)} combinations, "
            f"max {max_concurrent} concurrent per combo"
        )
        print(f"{'='*60}\n")

        async def _eval_combo(combo: Dict[str, str]) -> Tuple[str, ModelResult]:
            combo_name = self._combo_name(combo)
            print(f"  Evaluating: {combo_name}")

            scores, latencies, dp_ids = await self._evaluate_combo_async(
                combo, self.dataset, label=combo_name, max_concurrent=max_concurrent
            )
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            accuracy, _ = self._compute_stats(scores)
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            dp_results = self._build_datapoint_results(scores, latencies, dp_ids)

            result = self._make_result(
                model_name=combo_name,
                accuracy=accuracy,
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attribute="combination",
                is_best=False,
                datapoint_results=dp_results,
            )
            print(f"  {result}")
            return combo_name, result

        combo_results = await asyncio.gather(
            *[_eval_combo(c) for c in all_combos], return_exceptions=True,
        )

        all_results: List[ModelResult] = []
        for i, res in enumerate(combo_results):
            if isinstance(res, Exception):
                combo_name = self._combo_name(all_combos[i])
                print(f"  [{combo_name}] failed: {res}")
                all_results.append(
                    self._make_result(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        input_tokens={},
                        output_tokens={},
                        attribute="combination",
                        is_best=False,
                    )
                )
            else:
                _, result = res
                all_results.append(result)

        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for r in all_results:
                if r.model_name == best_name:
                    r.is_best = True
                    break
        else:
            print("\n  No combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results
