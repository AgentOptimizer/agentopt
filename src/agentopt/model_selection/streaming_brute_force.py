"""
Streaming brute-force model selection.

Evaluates every combination on incoming data batches and keeps cumulative
metrics updated over time.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate, validate_dataset
from .base import BaseModelSelector, ModelResult, SelectionResults


class StreamingBruteForceModelSelector(BaseModelSelector):
    """
    Brute-force selector that supports streaming updates.

    Usage:
    - call ``update(batch)`` as new labeled data arrives
    - call ``results()`` / ``best_combo()`` at any time
    - optional: still supports ``select_best()`` over the initial dataset
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
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
        self._all_combos: List[Dict[str, ModelCandidate]] = self._all_combos()
        self._combo_scores: Dict[int, List[float]] = {
            i: [] for i in range(len(self._all_combos))
        }
        self._combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(self._all_combos))
        }
        self._combo_dp_ids: Dict[int, List[str]] = {
            i: [] for i in range(len(self._all_combos))
        }
        self._seen_samples = 0
        self._seed_consumed = False

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        # Keep select_best behavior: evaluate the provided dataset once.
        if not self._seed_consumed:
            result = self.update(self.dataset, parallel=parallel, max_concurrent=max_concurrent)
            self._seed_consumed = True
            return result
        return self.results()

    def update(
        self, batch: Dataset, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Evaluate all combinations on a new incoming batch."""
        validate_dataset(batch)

        if parallel:
            asyncio.run(self._update_async(batch, max_concurrent=max_concurrent))
        else:
            self._update_sequential(batch)
        self._seen_samples += len(batch)
        return self.results()

    def update_one(
        self,
        input_data: Any,
        expected_answer: Any,
        parallel: bool = False,
        max_concurrent: int = 20,
    ) -> SelectionResults:
        """Convenience helper for a single incoming datapoint."""
        return self.update(
            [(input_data, expected_answer)],
            parallel=parallel,
            max_concurrent=max_concurrent,
        )

    def results(self) -> SelectionResults:
        """Return cumulative results over all streamed batches so far."""
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(self._all_combos):
            combo_name = self._combo_name(combo)
            scores = self._combo_scores[idx]
            latencies = self._combo_latencies[idx]
            dp_ids = self._combo_dp_ids[idx]

            if scores:
                accuracy = sum(scores) / len(scores)
                avg_latency = sum(latencies) / len(latencies)
            else:
                accuracy = 0.0
                avg_latency = 0.0

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

        return SelectionResults(results=all_results)

    def best_combo(self) -> Optional[Dict[str, str]]:
        """Return current best combination as node->model dict."""
        return self.results().get_best_combo()

    def _update_sequential(self, batch: Dataset) -> None:
        dp_offset = self._seen_samples
        total = len(self._all_combos)
        print(f"\nUpdating stream (sequential): {total} combinations, batch={len(batch)}")

        for idx, combo in enumerate(self._all_combos, 1):
            combo_name = self._combo_name(combo)
            print(f"  [{idx}/{total}] {combo_name}")
            scores, latencies, dp_ids = self._evaluate_combo(
                combo, batch, label=combo_name, dp_offset=dp_offset
            )
            self._combo_scores[idx - 1].extend(scores)
            self._combo_latencies[idx - 1].extend(latencies)
            self._combo_dp_ids[idx - 1].extend(dp_ids)

    async def _update_async(self, batch: Dataset, max_concurrent: int) -> None:
        dp_offset = self._seen_samples
        batch_size = len(batch)
        n_combo, dp_concurrent = self._compute_concurrency(max_concurrent, batch_size)
        combo_sem = asyncio.Semaphore(n_combo)
        total = len(self._all_combos)
        print(
            f"\nUpdating stream (async): {total} combinations, batch={batch_size}, "
            f"max {max_concurrent} total concurrent"
        )

        async def _eval_combo(
            idx: int,
            combo: Dict[str, ModelCandidate],
        ) -> Tuple[int, List[float], List[float], List[str]]:
            async with combo_sem:
                combo_name = self._combo_name(combo)
                scores, latencies, dp_ids = await self._evaluate_combo_async(
                    combo,
                    batch,
                    label=combo_name,
                    max_concurrent=dp_concurrent,
                    dp_offset=dp_offset,
                )
                return idx, scores, latencies, dp_ids

        results = await asyncio.gather(
            *[_eval_combo(idx, combo) for idx, combo in enumerate(self._all_combos)],
            return_exceptions=True,
        )
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                combo_name = self._combo_name(self._all_combos[i])
                print(f"  [{combo_name}] failed: {res}")
                continue
            idx, scores, latencies, dp_ids = res
            self._combo_scores[idx].extend(scores)
            self._combo_latencies[idx].extend(latencies)
            self._combo_dp_ids[idx].extend(dp_ids)
