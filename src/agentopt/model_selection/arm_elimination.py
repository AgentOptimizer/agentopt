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
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        n_initial: Optional[int] = None,
        growth_factor: float = 2.0,
        confidence: float = 1.0,
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
        combo_costs: Dict[int, List[Optional[float]]] = {
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
                    combo, batch, label=combo_name, dp_offset=offset
                )
                costs = self._observe_combo(scores, latencies, dp_ids)
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_costs[idx].extend(costs)
                combo_dp_ids[idx].extend(dp_ids)
                objs = self._compute_objectives(
                    combo_scores[idx], combo_latencies[idx], combo_costs[idx]
                )
                mu, _ = self._compute_stats(objs)
                print(f"  {combo_name}: mu={mu:.3f} (n={len(objs)})")

            # Eliminate dominated combinations on the current combined objective.
            arm_objectives = {
                idx: self._compute_objectives(
                    combo_scores[idx], combo_latencies[idx], combo_costs[idx]
                )
                for idx in active
            }
            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(
                        arm_objectives[i], arm_objectives[j]
                    ):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = self._combo_name(all_combos[idx])
                    winner = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combos, arm_objectives
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

        all_results = self._build_results(
            all_combos, combo_scores, combo_latencies, combo_costs, combo_dp_ids
        )
        return SelectionResults(
            results=all_results, objective_mode=self.objective_mode,
        )

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_total = len(dataset_list)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(len(all_combos))}
        combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(all_combos))
        }
        combo_costs: Dict[int, List[Optional[float]]] = {
            i: [] for i in range(len(all_combos))
        }
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(len(all_combos))}
        active: Set[int] = set(range(len(all_combos)))

        print(f"\n{'='*60}")
        print(
            f"Arm elimination (async): {len(all_combos)} combinations, "
            f"{n_total} samples, max {max_concurrent} total concurrent"
        )
        print(f"{'='*60}")

        offset = 0
        batch_size = self.n_initial
        round_num = 1

        while active and offset < n_total:
            batch_end = min(offset + batch_size, n_total)
            batch = dataset_list[offset:batch_end]
            current_batch_size = len(batch)
            n_combo_round, dp_concurrent_round = self._compute_concurrency(
                max_concurrent, current_batch_size
            )
            round_sem = asyncio.Semaphore(n_combo_round)

            print(
                f"\nRound {round_num} [samples {offset}-{batch_end}, "
                f"{len(active)} active]:"
            )

            async def _eval_batch(
                idx: int,
            ) -> Tuple[int, List[float], List[float], List[str]]:
                async with round_sem:
                    combo = all_combos[idx]
                    combo_name = self._combo_name(combo)
                    scores, latencies, dp_ids = await self._evaluate_combo_async(
                        combo,
                        batch,
                        label=combo_name,
                        max_concurrent=dp_concurrent_round,
                        dp_offset=offset,
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
                costs = self._observe_combo(scores, latencies, dp_ids)
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_costs[idx].extend(costs)
                combo_dp_ids[idx].extend(dp_ids)
                objs = self._compute_objectives(
                    combo_scores[idx], combo_latencies[idx], combo_costs[idx]
                )
                mu, _ = self._compute_stats(objs)
                print(
                    f"  {self._combo_name(all_combos[idx])}: "
                    f"mu={mu:.3f} (n={len(objs)})"
                )

            arm_objectives = {
                idx: self._compute_objectives(
                    combo_scores[idx], combo_latencies[idx], combo_costs[idx]
                )
                for idx in active
            }
            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(
                        arm_objectives[i], arm_objectives[j]
                    ):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = self._combo_name(all_combos[idx])
                    winner = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combos, arm_objectives
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

        all_results = self._build_results(
            all_combos, combo_scores, combo_latencies, combo_costs, combo_dp_ids
        )
        return SelectionResults(
            results=all_results, objective_mode=self.objective_mode,
        )

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
        arm_values: Dict[int, List[float]],
    ) -> Optional[str]:
        """Find the name of an arm that dominates ``dominated_idx`` on ``arm_values``."""
        for j in active_remaining:
            if self._is_dominated(arm_values[dominated_idx], arm_values[j]):
                return self._combo_name(all_combos[j])
        return None

    def _build_results(
        self,
        all_combos: List[Dict[str, ModelCandidate]],
        combo_scores: Dict[int, List[float]],
        combo_latencies: Dict[int, List[float]],
        combo_costs: Dict[int, List[Optional[float]]],
        combo_dp_ids: Dict[int, List[str]],
    ) -> List[ModelResult]:
        """Build final per-combo results, finalize the combined objective, mark best."""
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combos):
            combo_name = self._combo_name(combo)
            scores = combo_scores[idx]
            if scores:
                all_results.append(
                    self._build_combo_result(
                        combo_name,
                        scores,
                        combo_latencies[idx],
                        combo_dp_ids[idx],
                        costs=combo_costs[idx],
                    )
                )
            else:
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

        self._finalize_combined_objectives(all_results)
        if self.objective_mode == "pareto":
            self._mark_pareto_optimal(all_results)
        else:
            best_info = self._find_best(all_results)
            if best_info is not None:
                best_name, _ = best_info
                for result in all_results:
                    if result.model_name == best_name:
                        result.is_best = True
                        break
            else:
                print("\n  No combinations succeeded.")
        return all_results
