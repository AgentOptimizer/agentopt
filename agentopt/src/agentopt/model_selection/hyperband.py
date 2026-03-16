"""
Hyperband model selector.

Implements the full Hyperband algorithm using dataset samples as the
resource, with successive halving as the inner loop.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class HyperbandModelSelector(BaseModelSelector):
    """Select models using the full Hyperband algorithm."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, str]], Any],
        models: Dict[str, List[str]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        reduction_factor: float = 3.0,
        max_resource: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        if reduction_factor <= 1.0:
            raise ValueError("reduction_factor must be > 1.0.")

        super().__init__(
            agent_fn=agent_fn,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
        )

        self.reduction_factor = reduction_factor
        n = len(self.dataset)
        if n == 0:
            raise ValueError("Hyperband requires a non-empty dataset.")

        if max_resource is None:
            self.max_resource = n
        else:
            if max_resource <= 0:
                raise ValueError("max_resource must be positive.")
            self.max_resource = min(max_resource, n)

        self._s_max = int(
            math.floor(math.log(self.max_resource, self.reduction_factor))
        )
        self._B = (self._s_max + 1) * self.max_resource

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    def _select_sequential(self) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        total_configs = len(all_combos)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(total_configs)}

        print(f"\n{'='*60}")
        print(
            f"Hyperband (sequential): {total_configs} combinations, "
            f"max_resource={self.max_resource}, eta={self.reduction_factor}"
        )
        print(f"  s_max={self._s_max}, B={self._B}")
        print(f"{'='*60}")

        for s in reversed(range(self._s_max + 1)):
            r_s = int(self.max_resource * (self.reduction_factor ** (-s)))
            r_s = max(1, min(r_s, self.max_resource))

            bracket_indices = list(range(total_configs))
            print(f"\nBracket s={s}: configs={len(bracket_indices)}, r_s={r_s}")

            n_i = len(bracket_indices)
            prev_r = 0

            for i in range(s + 1):
                if n_i <= 0:
                    break

                current_indices = bracket_indices[:n_i]
                r_i = int(r_s * (self.reduction_factor ** i))
                r_i = max(1, min(r_i, self.max_resource))

                if r_i <= prev_r:
                    break

                print(f"\n  Stage i={i}: n_i={n_i}, r_i={r_i}")

                batch = dataset_list[prev_r:r_i]
                if not batch:
                    break

                for idx in current_indices:
                    combo = all_combos[idx]
                    combo_name = self._combo_name(combo)
                    scores, latencies, dp_ids = self._evaluate_combo(
                        combo, batch, label=combo_name
                    )
                    combo_scores[idx].extend(scores)
                    combo_latencies[idx].extend(latencies)
                    combo_dp_ids[idx].extend(dp_ids)
                    mu = (
                        sum(combo_scores[idx]) / len(combo_scores[idx])
                        if combo_scores[idx]
                        else 0.0
                    )
                    print(f"    {combo_name}: mu={mu:.3f} (n={len(combo_scores[idx])})")

                prev_r = r_i

                if i == s:
                    break

                # Successive halving.
                means: List[Tuple[int, float]] = []
                for idx in current_indices:
                    sc = combo_scores[idx]
                    mu = sum(sc) / len(sc) if sc else 0.0
                    means.append((idx, mu))

                means.sort(key=lambda x: x[1], reverse=True)
                n_i_next = max(1, int(math.floor(n_i / self.reduction_factor)))
                new_indices = [idx for idx, _ in means[:n_i_next]]
                eliminated = set(current_indices) - set(new_indices)

                if eliminated:
                    for idx in sorted(eliminated):
                        print(f"    Eliminated: {self._combo_name(all_combos[idx])}")

                bracket_indices = new_indices
                n_i = len(bracket_indices)

        # Build final results.
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combos):
            combo_name = self._combo_name(combo)
            scores = combo_scores[idx]
            latencies = combo_latencies[idx]
            dp_ids = combo_dp_ids[idx]
            if scores:
                accuracy = sum(scores) / len(scores)
                avg_latency = sum(latencies) / len(latencies)
            else:
                accuracy, avg_latency = 0.0, 0.0
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            dp_results = (
                self._build_datapoint_results(scores, latencies, dp_ids)
                if dp_ids
                else []
            )
            all_results.append(
                self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=avg_latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attribute="combination",
                    is_best=False,
                    datapoint_results=dp_results,
                )
            )

        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            print("\n  No combinations succeeded.")

        results = SelectionResults(results=all_results)
        return results

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        total_configs = len(all_combos)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(total_configs)}

        print(f"\n{'='*60}")
        print(
            f"Hyperband (async): {total_configs} combinations, "
            f"max_resource={self.max_resource}, eta={self.reduction_factor}, "
            f"max {max_concurrent} concurrent"
        )
        print(f"  s_max={self._s_max}, B={self._B}")
        print(f"{'='*60}")

        for s in reversed(range(self._s_max + 1)):
            r_s = int(self.max_resource * (self.reduction_factor ** (-s)))
            r_s = max(1, min(r_s, self.max_resource))

            bracket_indices = list(range(total_configs))
            print(f"\nBracket s={s}: configs={len(bracket_indices)}, r_s={r_s}")

            n_i = len(bracket_indices)
            prev_r = 0

            for i in range(s + 1):
                if n_i <= 0:
                    break

                current_indices = bracket_indices[:n_i]
                r_i = int(r_s * (self.reduction_factor ** i))
                r_i = max(1, min(r_i, self.max_resource))

                if r_i <= prev_r:
                    break

                print(f"\n  Stage i={i}: n_i={n_i}, r_i={r_i}")

                batch = dataset_list[prev_r:r_i]
                if not batch:
                    break

                async def _eval_batch(
                    idx: int,
                ) -> Tuple[int, List[float], List[float], List[str]]:
                    combo = all_combos[idx]
                    combo_name = self._combo_name(combo)
                    scores, latencies, dp_ids = await self._evaluate_combo_async(
                        combo, batch, label=combo_name, max_concurrent=max_concurrent
                    )
                    return idx, scores, latencies, dp_ids

                stage_results = await asyncio.gather(
                    *[_eval_batch(idx) for idx in current_indices],
                    return_exceptions=True,
                )

                for res in stage_results:
                    if isinstance(res, Exception):
                        logger.warning("Stage evaluation error: %s", res)
                        continue
                    idx, scores, latencies, dp_ids = res
                    combo_scores[idx].extend(scores)
                    combo_latencies[idx].extend(latencies)
                    combo_dp_ids[idx].extend(dp_ids)
                    mu = (
                        sum(combo_scores[idx]) / len(combo_scores[idx])
                        if combo_scores[idx]
                        else 0.0
                    )
                    print(
                        f"    {self._combo_name(all_combos[idx])}: "
                        f"mu={mu:.3f} (n={len(combo_scores[idx])})"
                    )

                prev_r = r_i

                if i == s:
                    break

                means: List[Tuple[int, float]] = []
                for idx in current_indices:
                    sc = combo_scores[idx]
                    mu = sum(sc) / len(sc) if sc else 0.0
                    means.append((idx, mu))

                means.sort(key=lambda x: x[1], reverse=True)
                n_i_next = max(1, int(math.floor(n_i / self.reduction_factor)))
                new_indices = [idx for idx, _ in means[:n_i_next]]
                eliminated = set(current_indices) - set(new_indices)

                if eliminated:
                    for idx in sorted(eliminated):
                        print(f"    Eliminated: {self._combo_name(all_combos[idx])}")

                bracket_indices = new_indices
                n_i = len(bracket_indices)

        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combos):
            combo_name = self._combo_name(combo)
            scores = combo_scores[idx]
            latencies = combo_latencies[idx]
            dp_ids = combo_dp_ids[idx]
            if scores:
                accuracy = sum(scores) / len(scores)
                avg_latency = sum(latencies) / len(latencies)
            else:
                accuracy, avg_latency = 0.0, 0.0
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            dp_results = (
                self._build_datapoint_results(scores, latencies, dp_ids)
                if dp_ids
                else []
            )
            all_results.append(
                self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=avg_latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attribute="combination",
                    is_best=False,
                    datapoint_results=dp_results,
                )
            )

        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            print("\n  No combinations succeeded.")

        results = SelectionResults(results=all_results)
        return results
