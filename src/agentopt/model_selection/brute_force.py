"""
Brute-force model selection: evaluates the Cartesian product of all
candidate models across all nodes.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults


class BruteForceModelSelector(BaseModelSelector):
    """
    Selects the best model combination by evaluating all combinations.

    Supports sequential and async-parallel evaluation via ``select_best()``.
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
        objective_mode: Optional[str] = None,
        lambda_cost: float = 0.0,
        lambda_latency: float = 0.0,
    ) -> None:
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
            objective_mode=objective_mode,
            lambda_cost=lambda_cost,
            lambda_latency=lambda_latency,
        )

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    def _select_sequential(self) -> SelectionResults:
        all_combos = self._all_combos()

        all_results: List[ModelResult] = []

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
                result = self._build_combo_result(
                    combo_name, scores, latencies, dp_ids,
                )
                print(f"  {result}")
                all_results.append(result)

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

        return self._finalize_selection_outcomes(all_results)

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()

        batch_size = len(self.dataset)
        n_combo, dp_concurrent = self._compute_concurrency(max_concurrent, batch_size)
        combo_sem = asyncio.Semaphore(n_combo)

        print(f"\n{'='*60}")
        print(
            f"Brute force (async): {len(all_combos)} combinations, "
            f"max {max_concurrent} total concurrent"
        )
        print(f"{'='*60}\n")

        async def _eval_combo(
            combo: Dict[str, ModelCandidate],
        ) -> Tuple[str, ModelResult]:
            async with combo_sem:
                combo_name = self._combo_name(combo)
                print(f"  Evaluating: {combo_name}")

                scores, latencies, dp_ids = await self._evaluate_combo_async(
                    combo, self.dataset, label=combo_name, max_concurrent=dp_concurrent
                )
                result = self._build_combo_result(
                    combo_name, scores, latencies, dp_ids,
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

        return self._finalize_selection_outcomes(all_results)
