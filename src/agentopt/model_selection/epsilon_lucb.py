"""
Epsilon-optimal LUCB model selector.

Identifies an epsilon-optimal best model combination using confidence bounds.
"""

import asyncio
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class EpsilonLUCBModelSelector(BaseModelSelector):
    """Select models via epsilon-optimal LUCB."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        epsilon: float = 0.01,
        n_initial: int = 1,
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
        self.epsilon = max(0.0, float(epsilon))
        self.n_initial = max(1, int(n_initial))
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
        n_arms = len(all_combos)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(n_arms)}
        active: Set[int] = set(range(n_arms))

        print(f"\n{'='*60}")
        print(
            f"Epsilon-LUCB (sequential): {n_arms} combinations, "
            f"{n_total} samples, epsilon={self.epsilon}"
        )
        print(f"{'='*60}")

        offset = 0
        init_batch_size = min(self.n_initial, n_total)
        assert init_batch_size > 0
        init_batch = dataset_list[offset : offset + init_batch_size]
        for idx in range(n_arms):
            combo = all_combos[idx]
            combo_name = self._combo_name(combo)
            scores, latencies, dp_ids = self._evaluate_combo(
                combo, init_batch, label=combo_name, dp_offset=offset
            )
            combo_scores[idx].extend(scores)
            combo_latencies[idx].extend(latencies)
            combo_dp_ids[idx].extend(dp_ids)
        offset += init_batch_size

        round_num = 1
        while active and offset < n_total:
            h_idx, l_idx, h_lcb, l_ucb = self._choose_lucb_pair(active, combo_scores)
            gap = l_ucb - h_lcb
            if gap <= self.epsilon or l_idx is None:
                break

            batch = [dataset_list[offset]]
            offset += 1
            sample_idxs = [h_idx] if l_idx == h_idx else [h_idx, l_idx]

            print(
                f"\nRound {round_num} [sample {offset}/{n_total}] "
                f"h={self._combo_name(all_combos[h_idx])}, "
                f"l={self._combo_name(all_combos[l_idx])}, gap={gap:.4f}"
            )

            for idx in sample_idxs:
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, batch, label=combo_name, dp_offset=offset - 1
                )
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                combo_dp_ids[idx].extend(dp_ids)
                mu, _ = self._compute_stats(combo_scores[idx])
                print(f"  {combo_name}: mu={mu:.3f} (n={len(combo_scores[idx])})")

            round_num += 1

        return self._build_results(
            all_combos, combo_scores, combo_latencies, combo_dp_ids
        )

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_total = len(dataset_list)
        n_arms = len(all_combos)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in range(n_arms)}
        combo_dp_ids: Dict[int, List[str]] = {i: [] for i in range(n_arms)}
        active: Set[int] = set(range(n_arms))

        print(f"\n{'='*60}")
        print(
            f"Epsilon-LUCB (async): {n_arms} combinations, "
            f"{n_total} samples, epsilon={self.epsilon}, "
            f"max {max_concurrent} total concurrent"
        )
        print(f"{'='*60}")

        offset = 0
        init_batch_size = min(self.n_initial, n_total)
        assert init_batch_size > 0
        init_batch = dataset_list[offset : offset + init_batch_size]
        n_combo_init, dp_concurrent_init = self._compute_concurrency(
            max_concurrent, init_batch_size
        )
        init_combo_sem = asyncio.Semaphore(n_combo_init)

        async def _eval_initial(
            idx: int,
        ) -> Tuple[int, List[float], List[float], List[str]]:
            async with init_combo_sem:
                combo = all_combos[idx]
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = await self._evaluate_combo_async(
                    combo,
                    init_batch,
                    label=combo_name,
                    max_concurrent=dp_concurrent_init,
                    dp_offset=offset,
                )
                return idx, scores, latencies, dp_ids

        round_results = await asyncio.gather(
            *[_eval_initial(idx) for idx in range(n_arms)], return_exceptions=True,
        )
        for res in round_results:
            if isinstance(res, Exception):
                logger.warning("Initial LUCB batch evaluation error: %s", res)
                continue
            idx, scores, latencies, dp_ids = res
            combo_scores[idx].extend(scores)
            combo_latencies[idx].extend(latencies)
            combo_dp_ids[idx].extend(dp_ids)
        offset += init_batch_size

        round_num = 1
        # Per-round batch_size is always 1
        n_combo_round, dp_concurrent_round = self._compute_concurrency(
            max_concurrent, 1
        )
        round_combo_sem = asyncio.Semaphore(n_combo_round)

        while active and offset < n_total:
            h_idx, l_idx, h_lcb, l_ucb = self._choose_lucb_pair(active, combo_scores)
            gap = l_ucb - h_lcb
            if gap <= self.epsilon or l_idx is None:
                break

            batch = [dataset_list[offset]]
            offset += 1
            sample_idxs = [h_idx] if l_idx == h_idx else [h_idx, l_idx]

            print(
                f"\nRound {round_num} [sample {offset}/{n_total}] "
                f"h={self._combo_name(all_combos[h_idx])}, "
                f"l={self._combo_name(all_combos[l_idx])}, gap={gap:.4f}"
            )

            async def _eval_pair(
                idx: int,
            ) -> Tuple[int, List[float], List[float], List[str]]:
                async with round_combo_sem:
                    combo = all_combos[idx]
                    combo_name = self._combo_name(combo)
                    scores, latencies, dp_ids = await self._evaluate_combo_async(
                        combo,
                        batch,
                        label=combo_name,
                        max_concurrent=dp_concurrent_round,
                        dp_offset=offset - 1,
                    )
                    return idx, scores, latencies, dp_ids

            round_results = await asyncio.gather(
                *[_eval_pair(idx) for idx in sample_idxs], return_exceptions=True,
            )
            for res in round_results:
                if isinstance(res, Exception):
                    logger.warning("LUCB pair evaluation error: %s", res)
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

            round_num += 1

        return self._build_results(
            all_combos, combo_scores, combo_latencies, combo_dp_ids
        )

    def _choose_lucb_pair(
        self, active: Set[int], combo_scores: Dict[int, List[float]],
    ) -> Tuple[int, Optional[int], float, float]:
        stats: Dict[int, Tuple[float, float, float]] = {}
        for idx in active:
            stats[idx] = self._confidence_bounds(combo_scores[idx])

        h_idx = max(active, key=lambda i: stats[i][0])  # highest empirical mean
        if len(active) == 1:
            _, h_lcb, _ = stats[h_idx]
            return h_idx, None, h_lcb, h_lcb

        competitors = [i for i in active if i != h_idx]
        l_idx = max(competitors, key=lambda i: stats[i][2])  # highest UCB
        h_lcb = stats[h_idx][1]
        l_ucb = stats[l_idx][2]
        return h_idx, l_idx, h_lcb, l_ucb

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
                    num_samples=max(len(scores), 1),
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
