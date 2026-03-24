"""
Streaming random-search model selection.

Evaluates a fixed random subset of combinations on incoming data batches and
keeps cumulative metrics updated over time.
"""

import asyncio
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate, validate_dataset
from .base import BaseModelSelector, ModelResult, SelectionResults


class StreamingRandomSearchModelSelector(BaseModelSelector):
    """
    Random-search selector that supports streaming updates.

    A subset of combinations is sampled once at initialization and reused for
    all incoming batches.
    """

    _CONVERGENCE_DELTA = 0.02
    _CONVERGENCE_PATIENCE_BATCHES = 10

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        sample_fraction: float = 0.25,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
    ) -> None:
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
        )
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")

        self.sample_fraction = sample_fraction
        self.seed = seed

        self._all_combos: List[Dict[str, ModelCandidate]] = self._all_combos()
        total = len(self._all_combos)
        sample_size = max(1, math.ceil(total * self.sample_fraction))
        sample_size = min(sample_size, total)

        if sample_size == total:
            self._sampled_indices = list(range(total))
        else:
            rng = random.Random(self.seed)
            self._sampled_indices = sorted(rng.sample(range(total), sample_size))
        self._sampled_combos = [self._all_combos[i] for i in self._sampled_indices]

        self._combo_scores: Dict[int, List[float]] = {
            i: [] for i in range(len(self._sampled_combos))
        }
        self._combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(self._sampled_combos))
        }
        self._combo_dp_ids: Dict[int, List[str]] = {
            i: [] for i in range(len(self._sampled_combos))
        }
        self._seen_samples = 0
        self._seed_consumed = False

        self._converged = False
        self._stable_batches = 0
        self._best_combo_signature: Optional[Tuple[Tuple[str, str], ...]] = None
        self._best_accuracy: Optional[float] = None

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if not self._seed_consumed:
            result = self.update(self.dataset, parallel=parallel, max_concurrent=max_concurrent)
            self._seed_consumed = True
            return result
        return self.results()

    def update(
        self, batch: Dataset, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Evaluate sampled combinations on a new incoming batch."""
        if self._converged:
            print(
                "\nStreaming random search converged; skipping new batch. "
                "Current best combo is stable."
            )
            return self.results()

        validate_dataset(batch)
        if parallel:
            asyncio.run(self._update_async(batch, max_concurrent=max_concurrent))
        else:
            self._update_sequential(batch)
        self._seen_samples += len(batch)

        results = self.results()
        self._update_convergence_state(results)
        return results

    def update_one(
        self,
        input_data: Any,
        expected_answer: Any,
        parallel: bool = False,
        max_concurrent: int = 20,
    ) -> SelectionResults:
        return self.update(
            [(input_data, expected_answer)],
            parallel=parallel,
            max_concurrent=max_concurrent,
        )

    def results(self) -> SelectionResults:
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(self._sampled_combos):
            combo_name = self._combo_name(combo)
            scores = self._combo_scores[idx]
            latencies = self._combo_latencies[idx]
            dp_ids = self._combo_dp_ids[idx]

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

        return SelectionResults(results=all_results)

    def best_combo(self) -> Optional[Dict[str, str]]:
        return self.results().get_best_combo()

    def has_converged(self) -> bool:
        return self._converged

    def should_continue(self) -> bool:
        return not self._converged

    def convergence_state(self) -> Dict[str, Any]:
        return {
            "converged": self._converged,
            "stable_batches": self._stable_batches,
            "required_stable_batches": self._CONVERGENCE_PATIENCE_BATCHES,
            "delta": self._CONVERGENCE_DELTA,
            "best_accuracy": self._best_accuracy,
            "best_combo": dict(self._best_combo_signature)
            if self._best_combo_signature is not None
            else None,
            "sampled_combinations": len(self._sampled_combos),
            "sample_fraction": self.sample_fraction,
        }

    def _update_sequential(self, batch: Dataset) -> None:
        dp_offset = self._seen_samples
        total = len(self._sampled_combos)
        print(
            f"\nUpdating stream random-search (sequential): "
            f"{total}/{len(self._all_combos)} combinations, batch={len(batch)}"
        )

        for idx, combo in enumerate(self._sampled_combos, 1):
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
        total = len(self._sampled_combos)
        print(
            f"\nUpdating stream random-search (async): "
            f"{total}/{len(self._all_combos)} combinations, batch={batch_size}, "
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
            *[
                _eval_combo(idx, combo)
                for idx, combo in enumerate(self._sampled_combos)
            ],
            return_exceptions=True,
        )
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                combo_name = self._combo_name(self._sampled_combos[i])
                print(f"  [{combo_name}] failed: {res}")
                continue
            idx, scores, latencies, dp_ids = res
            self._combo_scores[idx].extend(scores)
            self._combo_latencies[idx].extend(latencies)
            self._combo_dp_ids[idx].extend(dp_ids)

    @staticmethod
    def _combo_signature(
        combo: Optional[Dict[str, str]],
    ) -> Optional[Tuple[Tuple[str, str], ...]]:
        if combo is None:
            return None
        return tuple(sorted((str(k), str(v)) for k, v in combo.items()))

    def _update_convergence_state(self, results: SelectionResults) -> None:
        best = results.get_best()
        if best is None:
            return

        combo_sig = self._combo_signature(results.get_best_combo())
        if combo_sig is None:
            return

        current_acc = best.accuracy
        if self._best_combo_signature is None or self._best_accuracy is None:
            self._best_combo_signature = combo_sig
            self._best_accuracy = current_acc
            self._stable_batches = 0
            return

        combo_unchanged = combo_sig == self._best_combo_signature
        improvement = current_acc - self._best_accuracy

        if combo_unchanged and improvement < self._CONVERGENCE_DELTA:
            self._stable_batches += 1
        else:
            self._stable_batches = 0

        self._best_combo_signature = combo_sig
        self._best_accuracy = current_acc

        if self._stable_batches >= self._CONVERGENCE_PATIENCE_BATCHES:
            self._converged = True
            print(
                "\nStreaming random search converged: best combo stable for "
                f"{self._stable_batches} batches with < "
                f"{self._CONVERGENCE_DELTA:.0%} accuracy improvement."
            )
