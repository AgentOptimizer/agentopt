"""
Random-search model selection: evaluates a random subset of the Cartesian
product of candidate models across all nodes.
"""

import asyncio
import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class RandomSearchModelSelector(BaseModelSelector):
    """
    Selects the best model combination from a random subset of candidates.
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        sample_fraction: float = 0.25,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
    ) -> None:
        super().__init__(
            agent_fn=agent_fn,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
            tracker=tracker,
        )
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")
        self.sample_fraction = sample_fraction
        self.seed = seed

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    def _get_sampled_combinations(
        self,
    ) -> Tuple[List[Dict[str, ModelCandidate]], List[Dict[str, ModelCandidate]]]:
        """Return (all_combos, sampled_combos)."""
        all_combos = self._all_combos()
        total = len(all_combos)
        sample_size = max(1, math.ceil(total * self.sample_fraction))
        sample_size = min(sample_size, total)

        if sample_size == total:
            return all_combos, all_combos

        rng = random.Random(self.seed)
        sampled_indices = sorted(rng.sample(range(total), sample_size))
        sampled = [all_combos[i] for i in sampled_indices]
        return all_combos, sampled

    def _select_sequential(self) -> SelectionResults:
        all_combos, sampled = self._get_sampled_combinations()

        all_results: List[ModelResult] = []
        best_combo_name = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'='*60}")
        print(
            f"Random search (sequential): "
            f"{len(sampled)}/{len(all_combos)} combinations "
            f"({self.sample_fraction:.1%} sample)"
        )
        print(f"{'='*60}\n")

        for idx, combo in enumerate(sampled, 1):
            combo_name = self._combo_name(combo)
            print(f"  [{idx}/{len(sampled)}] Evaluating: {combo_name}")

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
                    num_samples=max(len(scores), 1),
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
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        return results

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos, sampled = self._get_sampled_combinations()

        batch_size = len(self.dataset)
        n_combo, dp_concurrent = self._compute_concurrency(max_concurrent, batch_size)
        combo_sem = asyncio.Semaphore(n_combo)

        print(f"\n{'='*60}")
        print(
            f"Random search (async): "
            f"{len(sampled)}/{len(all_combos)} combinations "
            f"({self.sample_fraction:.1%} sample), "
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
                    num_samples=max(len(scores), 1),
                    datapoint_results=dp_results,
                )
                print(f"  {result}")
                return combo_name, result

        combo_results = await asyncio.gather(
            *[_eval_combo(c) for c in sampled], return_exceptions=True,
        )

        all_results: List[ModelResult] = []
        for i, res in enumerate(combo_results):
            if isinstance(res, Exception):
                combo_name = self._combo_name(sampled[i])
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
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        return results
