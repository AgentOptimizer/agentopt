"""
Arm Elimination model selector.

Progressively eliminates statistically dominated model combinations.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class ArmEliminationModelSelector(BaseModelSelector):
    """Select models via successive arm elimination."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        n_initial: Optional[int] = None,
        growth_factor: float = 2.0,
        confidence: float = 1.0,
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
        n = len(self.dataset)
        if n_initial is None:
            self.n_initial = max(1, n // 10)
        else:
            self.n_initial = n_initial
        self.growth_factor = growth_factor
        self.confidence = confidence

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    def _select_sequential(self) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_total = len(dataset_list)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(len(all_combos))}
        combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(all_combos))
        }
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(len(all_combos))}
        active: Set[int] = set(range(len(all_combos)))

        print(f"\n{'='*60}")
        print(
            f"Arm elimination (sequential): {len(all_combos)} combinations, "
            f"{n_total} samples"
        )
        print(f"{'='*60}")

        offset = 0
        batch_size = self.n_initial
        round_num = 1

        while active and offset < n_total:
            batch_end = min(offset + batch_size, n_total)
            batch = dataset_list[offset:batch_end]

            print(
                f"\nRound {round_num} [samples {offset}-{batch_end}, "
                f"{len(active)} active]:"
            )

            for idx in sorted(active):
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, batch, label=combo_name
                )
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_dp_ids[idx].extend(dp_ids)
                mu, _ = self._compute_stats(combo_scores[idx])
                print(f"  {combo_name}: mu={mu:.3f} (n={len(combo_scores[idx])})")

            # Eliminate dominated combinations.
            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(combo_scores[i], combo_scores[j]):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = self._combo_name(all_combos[idx])
                    winner = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combos, combo_scores
                    )
                    print(
                        f"  Eliminated: {combo_name}"
                        + (f" (dominated by {winner})" if winner else "")
                    )
                active -= newly_eliminated
                print(f"  Survivors: {len(active)} / {len(all_combos)}")
            else:
                print(
                    f"  No eliminations. Survivors: {len(active)} / {len(all_combos)}"
                )

            if len(active) <= 1:
                break

            offset = batch_end
            batch_size = max(1, int(batch_size * self.growth_factor))
            round_num += 1

        # Build results.
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
        n_total = len(dataset_list)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(len(all_combos))}
        combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(all_combos))
        }
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(len(all_combos))}
        active: Set[int] = set(range(len(all_combos)))

        print(f"\n{'='*60}")
        print(
            f"Arm elimination (async): {len(all_combos)} combinations, "
            f"{n_total} samples, max {max_concurrent} concurrent"
        )
        print(f"{'='*60}")

        offset = 0
        batch_size = self.n_initial
        round_num = 1

        while active and offset < n_total:
            batch_end = min(offset + batch_size, n_total)
            batch = dataset_list[offset:batch_end]

            print(
                f"\nRound {round_num} [samples {offset}-{batch_end}, "
                f"{len(active)} active]:"
            )

            async def _eval_batch(
                idx: int,
            ) -> Tuple[int, List[float], List[float], List[str]]:
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = await self._evaluate_combo_async(
                    combo, batch, label=combo_name, max_concurrent=max_concurrent
                )
                return idx, scores, latencies, dp_ids

            round_results = await asyncio.gather(
                *[_eval_batch(idx) for idx in sorted(active)], return_exceptions=True,
            )

            for res in round_results:
                if isinstance(res, Exception):
                    logger.warning("Batch evaluation error: %s", res)
                    continue
                idx, scores, latencies, dp_ids = res
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_dp_ids[idx].extend(dp_ids)
                mu, _ = self._compute_stats(combo_scores[idx])
                print(
                    f"  {self._combo_name(all_combos[idx])}: "
                    f"mu={mu:.3f} (n={len(combo_scores[idx])})"
                )

            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(combo_scores[i], combo_scores[j]):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = self._combo_name(all_combos[idx])
                    winner = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combos, combo_scores
                    )
                    print(
                        f"  Eliminated: {combo_name}"
                        + (f" (dominated by {winner})" if winner else "")
                    )
                active -= newly_eliminated
                print(f"  Survivors: {len(active)} / {len(all_combos)}")
            else:
                print(
                    f"  No eliminations. Survivors: {len(active)} / {len(all_combos)}"
                )

            if len(active) <= 1:
                break

            offset = batch_end
            batch_size = max(1, int(batch_size * self.growth_factor))
            round_num += 1

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

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _is_dominated(self, scores_i: List[float], scores_j: List[float]) -> bool:
        """Return True if arm i is statistically dominated by arm j."""
        n_i, n_j = len(scores_i), len(scores_j)
        if n_i == 0 or n_j == 0:
            return False
        mu_i, std_i = self._compute_stats(scores_i)
        mu_j, std_j = self._compute_stats(scores_j)
        se_i = std_i / math.sqrt(n_i)
        se_j = std_j / math.sqrt(n_j)
        return mu_i + self.confidence * se_i < mu_j - self.confidence * se_j

    def _find_dominator_name(
        self,
        dominated_idx: int,
        active_remaining: Set[int],
        all_combos: List[Dict[str, ModelCandidate]],
        combo_scores: Dict[int, List[float]],
    ) -> Optional[str]:
        for j in active_remaining:
            if self._is_dominated(combo_scores[dominated_idx], combo_scores[j]):
                return self._combo_name(all_combos[j])
        return None
