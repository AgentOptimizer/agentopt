"""
Threshold-bandit successive elimination model selector.

Classifies combinations as above/below a user-provided threshold.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class ThresholdBanditSEModelSelector(BaseModelSelector):
    """Select models via threshold-based successive elimination."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        threshold: float = 0.5,
        n_initial: Optional[int] = None,
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
        self.confidence = confidence
        self.threshold = threshold

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
            f"Threshold successive elimination (sequential): {len(all_combos)} "
            f"combinations, {n_total} samples, threshold={self.threshold}"
        )
        print(f"{'='*60}")

        offset = 0
        init_batch_size = min(self.n_initial, n_total)
        if init_batch_size > 0:
            init_batch = dataset_list[offset : offset + init_batch_size]
            print(
                f"\nInitial round [samples {offset}-{offset + init_batch_size}, "
                f"{len(active)} active]:"
            )
            for idx in sorted(active):
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, init_batch, label=combo_name
                )
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_dp_ids[idx].extend(dp_ids)
                mu, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                print(
                    f"  {combo_name}: mu={mu:.3f}, [{lcb:.3f}, {ucb:.3f}] "
                    f"(n={len(combo_scores[idx])})"
                )
            offset += init_batch_size

        round_num = 1

        while active and offset < n_total:
            batch = [dataset_list[offset]]
            offset += 1

            print(
                f"\nRound {round_num} [sample {offset}/{n_total}, "
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
                mu, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                print(
                    f"  {combo_name}: mu={mu:.3f}, [{lcb:.3f}, {ucb:.3f}] "
                    f"(n={len(combo_scores[idx])})"
                )

            newly_eliminated: Set[int] = set()
            for idx in active:
                _, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                # Classified wrt threshold, so this arm is no longer ambiguous.
                if ucb < self.threshold or lcb > self.threshold:
                    newly_eliminated.add(idx)

            if newly_eliminated:
                for idx in sorted(newly_eliminated):
                    combo_name = self._combo_name(all_combos[idx])
                    _, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                    side = "below" if ucb < self.threshold else "above"
                    print(
                        f"  Eliminated: {combo_name} "
                        f"(classified {side} threshold, "
                        f"CI=[{lcb:.3f}, {ucb:.3f}])"
                    )
                active -= newly_eliminated
                print(f"  Ambiguous survivors: {len(active)} / {len(all_combos)}")
            else:
                print(
                    f"  No eliminations. Ambiguous survivors: "
                    f"{len(active)} / {len(all_combos)}"
                )

            if not active:
                break
            round_num += 1

        return self._build_results(all_combos, combo_scores, combo_latencies, combo_dp_ids)

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
            f"Threshold successive elimination (async): {len(all_combos)} "
            f"combinations, {n_total} samples, threshold={self.threshold}, "
            f"max {max_concurrent} concurrent"
        )
        print(f"{'='*60}")

        offset = 0
        init_batch_size = min(self.n_initial, n_total)
        if init_batch_size > 0:
            init_batch = dataset_list[offset : offset + init_batch_size]
            print(
                f"\nInitial round [samples {offset}-{offset + init_batch_size}, "
                f"{len(active)} active]:"
            )

            async def _eval_initial(
                idx: int,
            ) -> Tuple[int, List[float], List[float], List[str]]:
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = await self._evaluate_combo_async(
                    combo, init_batch, label=combo_name, max_concurrent=max_concurrent
                )
                return idx, scores, latencies, dp_ids

            init_results = await asyncio.gather(
                *[_eval_initial(idx) for idx in sorted(active)], return_exceptions=True,
            )

            for res in init_results:
                if isinstance(res, Exception):
                    logger.warning("Initial batch evaluation error: %s", res)
                    continue
                idx, scores, latencies, dp_ids = res
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_dp_ids[idx].extend(dp_ids)
                mu, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                print(
                    f"  {self._combo_name(all_combos[idx])}: "
                    f"mu={mu:.3f}, [{lcb:.3f}, {ucb:.3f}] "
                    f"(n={len(combo_scores[idx])})"
                )
            offset += init_batch_size

        round_num = 1

        while active and offset < n_total:
            batch = [dataset_list[offset]]
            offset += 1

            print(
                f"\nRound {round_num} [sample {offset}/{n_total}, "
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
                mu, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                print(
                    f"  {self._combo_name(all_combos[idx])}: "
                    f"mu={mu:.3f}, [{lcb:.3f}, {ucb:.3f}] "
                    f"(n={len(combo_scores[idx])})"
                )

            newly_eliminated: Set[int] = set()
            for idx in active:
                _, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                if ucb < self.threshold or lcb > self.threshold:
                    newly_eliminated.add(idx)

            if newly_eliminated:
                for idx in sorted(newly_eliminated):
                    combo_name = self._combo_name(all_combos[idx])
                    _, lcb, ucb = self._confidence_bounds(combo_scores[idx])
                    side = "below" if ucb < self.threshold else "above"
                    print(
                        f"  Eliminated: {combo_name} "
                        f"(classified {side} threshold, "
                        f"CI=[{lcb:.3f}, {ucb:.3f}])"
                    )
                active -= newly_eliminated
                print(f"  Ambiguous survivors: {len(active)} / {len(all_combos)}")
            else:
                print(
                    f"  No eliminations. Ambiguous survivors: "
                    f"{len(active)} / {len(all_combos)}"
                )

            if not active:
                break
            round_num += 1

        return self._build_results(all_combos, combo_scores, combo_latencies, combo_dp_ids)

    def _confidence_bounds(self, scores: List[float]) -> Tuple[float, float, float]:
        n = len(scores)
        if n == 0:
            return 0.0, float("-inf"), float("inf")
        mu, std = self._compute_stats(scores)
        se = std / math.sqrt(n)
        radius = self.confidence * se
        return mu, mu - radius, mu + radius

    def _build_results(
        self,
        all_combos: List[Dict[str, ModelCandidate]],
        combo_scores: Dict[int, List[float]],
        combo_latencies: Dict[int, List[float]],
        combo_dp_ids: Dict[int, List[str]],
    ) -> SelectionResults:
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
                self._build_datapoint_results(scores, latencies, dp_ids) if dp_ids else []
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

        return SelectionResults(results=all_results)
