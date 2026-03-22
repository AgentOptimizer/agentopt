"""
Hill-climbing model selector with random restarts.

Uses the model topology (quality / speed rankings) to define
neighbours so that each iteration makes an informed single-step move.
"""

import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from ..model_topology import get_faster_neighbor, get_higher_quality_neighbor
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults


class HillClimbingModelSelector(BaseModelSelector):
    """Select models via stochastic hill climbing with random restarts."""

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        invoke_fn: Optional[Callable] = None,
        max_iterations: int = 20,
        num_restarts: int = 3,
        patience: int = 3,
        seed: Optional[int] = None,
        batch_size: int = 1,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
        *,
        agent_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
            tracker=tracker,
            agent_fn=agent_fn,
        )
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts
        self.patience = patience
        self.batch_size = max(1, int(batch_size))

        if seed is not None:
            random.seed(seed)

        # Pre-compute all combinations for random starts.
        self._all_combo_list = self._all_combos()

        # Cache: combo_name -> (accuracy, latency, input_tokens, output_tokens, datapoint_results).
        self._eval_cache: Dict[
            str,
            Tuple[float, float, Dict[str, int], Dict[str, int], List[DatapointResult]],
        ] = {}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _random_combination(
        self, seen: Set[str]
    ) -> Optional[Dict[str, ModelCandidate]]:
        """Pick a random unseen combination, or ``None`` if all exhausted."""
        unseen = [c for c in self._all_combo_list if self._combo_name(c) not in seen]
        if unseen:
            return dict(random.choice(unseen))
        return None

    def _process_eval_result(
        self,
        combo_name: str,
        scores: List[float],
        latencies: List[float],
        dp_ids: List[str],
    ) -> Tuple[
        str, float, float, Dict[str, int], Dict[str, int], List[DatapointResult], bool
    ]:
        """Compute stats, cache, and return the standard eval tuple."""
        input_tokens, output_tokens = self._fetch_tokens(combo_name)
        accuracy, _ = self._compute_stats(scores)
        latency = sum(latencies) / len(latencies) if latencies else 0.0
        dp_results = self._build_datapoint_results(scores, latencies, dp_ids)
        self._eval_cache[combo_name] = (
            accuracy,
            latency,
            input_tokens,
            output_tokens,
            dp_results,
        )
        return (
            combo_name,
            accuracy,
            latency,
            input_tokens,
            output_tokens,
            dp_results,
            False,
        )

    def _evaluate_cached(
        self, combo: Dict[str, ModelCandidate],
    ) -> Tuple[
        str, float, float, Dict[str, int], Dict[str, int], List[DatapointResult], bool
    ]:
        """Evaluate a combo synchronously, using an in-memory cache."""
        combo_name = self._combo_name(combo)
        if combo_name in self._eval_cache:
            acc, lat, in_tok, out_tok, dp_results = self._eval_cache[combo_name]
            return combo_name, acc, lat, in_tok, out_tok, dp_results, True
        scores, latencies, dp_ids = self._evaluate_combo(
            combo, self.dataset, label=combo_name
        )
        return self._process_eval_result(combo_name, scores, latencies, dp_ids)

    async def _evaluate_cached_async(
        self, combo: Dict[str, ModelCandidate], max_concurrent: int
    ) -> Tuple[
        str, float, float, Dict[str, int], Dict[str, int], List[DatapointResult], bool
    ]:
        """Evaluate a combo asynchronously, using an in-memory cache."""
        combo_name = self._combo_name(combo)
        if combo_name in self._eval_cache:
            acc, lat, in_tok, out_tok, dp_results = self._eval_cache[combo_name]
            return combo_name, acc, lat, in_tok, out_tok, dp_results, True
        scores, latencies, dp_ids = await self._evaluate_combo_async(
            combo, self.dataset, label=combo_name, max_concurrent=max_concurrent
        )
        return self._process_eval_result(combo_name, scores, latencies, dp_ids)

    def _get_neighbors(
        self, combo: Dict[str, ModelCandidate], seen: Set[str], accuracy: float,
    ) -> List[Dict[str, ModelCandidate]]:
        """Generate neighbors with quality/speed fallback logic."""
        if accuracy < 1.0:
            neighbors = self._generate_neighbors(
                combo, seen, max_neighbors=self.batch_size, improve_quality=True,
            )
            if not neighbors:
                neighbors = self._generate_neighbors(
                    combo, seen, max_neighbors=self.batch_size, improve_quality=False,
                )
        else:
            neighbors = self._generate_neighbors(
                combo, seen, max_neighbors=self.batch_size, improve_quality=False,
            )
            if not neighbors:
                neighbors = self._generate_neighbors(
                    combo, seen, max_neighbors=self.batch_size, improve_quality=True,
                )
        return neighbors

    def _generate_neighbors(
        self,
        combo: Dict[str, ModelCandidate],
        seen: Set[str],
        max_neighbors: int,
        improve_quality: bool,
    ) -> List[Dict[str, ModelCandidate]]:
        """Generate up to *max_neighbors* unseen neighbors that differ by one node."""
        neighbors: List[Dict[str, ModelCandidate]] = []
        node_names = list(combo.keys())
        random.shuffle(node_names)

        for node in node_names:
            current = combo[node]
            if improve_quality:
                neighbor = get_higher_quality_neighbor(current, self._models[node])
            else:
                neighbor = get_faster_neighbor(current, self._models[node])
            if neighbor is None:
                continue

            new_combo = dict(combo)
            new_combo[node] = neighbor
            if self._combo_name(new_combo) in seen:
                continue

            neighbors.append(new_combo)
            if len(neighbors) >= max_neighbors:
                break

        return neighbors

    def _pick_best_neighbor(
        self,
        eval_results: List[Tuple],
        neighbors: List[Dict[str, ModelCandidate]],
        seen: Set[str],
        current_accuracy: float,
        current_latency: float,
        tol: float,
    ) -> Optional[Dict[str, ModelCandidate]]:
        """Select the best neighbor from eval results, or None if none improves."""
        best_neighbor: Optional[Dict[str, ModelCandidate]] = None
        best_n_acc = float("-inf")
        best_n_lat = float("inf")

        for neighbor, eval_result in zip(neighbors, eval_results):
            n_name, n_acc, n_lat, *_ = eval_result
            seen.add(n_name)

            if n_acc > best_n_acc + tol:
                best_neighbor, best_n_acc, best_n_lat = neighbor, n_acc, n_lat
            elif abs(n_acc - best_n_acc) <= tol and n_lat < best_n_lat:
                best_neighbor, best_n_acc, best_n_lat = neighbor, n_acc, n_lat

        if best_neighbor is None or (
            best_n_acc < current_accuracy - tol
            or (
                abs(best_n_acc - current_accuracy) <= tol
                and best_n_lat >= current_latency
            )
        ):
            return None
        return best_neighbor

    def _hc_finalize(
        self,
        all_results: List[ModelResult],
        global_best_combo: Optional[Dict[str, ModelCandidate]],
        global_best_accuracy: float,
    ) -> SelectionResults:
        """Mark best result and return SelectionResults."""
        tol = 1e-9
        if global_best_combo is not None:
            best_name = self._combo_name(global_best_combo)
            for result in all_results:
                if (
                    result.model_name == best_name
                    and abs(result.accuracy - global_best_accuracy) < tol
                ):
                    result.is_best = True
                    break
        else:
            print("\nNo combinations succeeded\n")
        return SelectionResults(results=all_results)

    # ------------------------------------------------------------------
    # Single restart (sequential)
    # ------------------------------------------------------------------

    def _hill_climb_once_sequential(
        self, seen: Set[str],
    ) -> Optional[Tuple[Dict[str, ModelCandidate], float, float, List[ModelResult]]]:
        combo = self._random_combination(seen)
        if combo is None:
            return None

        results: List[ModelResult] = []
        best_combo = dict(combo)
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9
        no_improve_count = 0

        for iteration in range(self.max_iterations):
            combo_name = self._combo_name(combo)
            seen.add(combo_name)

            (
                _,
                accuracy,
                latency,
                input_tokens,
                output_tokens,
                dp_results,
                cached,
            ) = self._evaluate_cached(combo)

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
            suffix = " (cached)" if cached else ""
            print(f"  Iter {iteration + 1}: {result}{suffix}")
            results.append(result)

            should_update = (
                best_accuracy == float("-inf")
                or accuracy > best_accuracy + tol
                or (abs(accuracy - best_accuracy) <= tol and latency < best_latency)
            )
            if should_update:
                best_accuracy, best_latency, best_combo = accuracy, latency, dict(combo)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print(
                    f"  No improvement for {self.patience} iterations. "
                    f"Converged at iteration {iteration + 1}."
                )
                break

            neighbors = self._get_neighbors(combo, seen, accuracy)
            if not neighbors:
                print(f"  No improving moves at iteration {iteration + 1}. Stopping.")
                break

            eval_results = [self._evaluate_cached(n) for n in neighbors]
            best_neighbor = self._pick_best_neighbor(
                eval_results, neighbors, seen, accuracy, latency, tol
            )
            if best_neighbor is None:
                print(
                    f"  No neighbor in batch of {len(neighbors)} improves at "
                    f"iteration {iteration + 1}. Stopping."
                )
                break

            combo = dict(best_neighbor)

        return best_combo, best_accuracy, best_latency, results

    # ------------------------------------------------------------------
    # Single restart (async)
    # ------------------------------------------------------------------

    async def _hill_climb_once_async(
        self, seen: Set[str], max_concurrent: int
    ) -> Optional[Tuple[Dict[str, ModelCandidate], float, float, List[ModelResult]]]:
        combo = self._random_combination(seen)
        if combo is None:
            return None

        results: List[ModelResult] = []
        best_combo = dict(combo)
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9
        no_improve_count = 0

        for iteration in range(self.max_iterations):
            combo_name = self._combo_name(combo)
            seen.add(combo_name)

            (
                _,
                accuracy,
                latency,
                input_tokens,
                output_tokens,
                dp_results,
                cached,
            ) = await self._evaluate_cached_async(combo, max_concurrent=max_concurrent)

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
            suffix = " (cached)" if cached else ""
            print(f"  Iter {iteration + 1}: {result}{suffix}")
            results.append(result)

            should_update = (
                best_accuracy == float("-inf")
                or accuracy > best_accuracy + tol
                or (abs(accuracy - best_accuracy) <= tol and latency < best_latency)
            )
            if should_update:
                best_accuracy, best_latency, best_combo = accuracy, latency, dict(combo)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print(
                    f"  No improvement for {self.patience} iterations. "
                    f"Converged at iteration {iteration + 1}."
                )
                break

            neighbors = self._get_neighbors(combo, seen, accuracy)
            if not neighbors:
                print(f"  No improving moves at iteration {iteration + 1}. Stopping.")
                break

            batch_size = len(self.dataset)
            n_combo_nb, dp_concurrent_nb = self._compute_concurrency(
                max_concurrent, batch_size
            )
            neighbor_sem = asyncio.Semaphore(n_combo_nb)

            async def _eval_neighbor_throttled(
                n: Dict[str, ModelCandidate],
            ) -> Tuple[str, float, float, Dict[str, int], Dict[str, int], List, bool]:
                async with neighbor_sem:
                    return await self._evaluate_cached_async(
                        n, max_concurrent=dp_concurrent_nb
                    )

            eval_results = await asyncio.gather(
                *(_eval_neighbor_throttled(n) for n in neighbors)
            )
            best_neighbor = self._pick_best_neighbor(
                eval_results, neighbors, seen, accuracy, latency, tol
            )
            if best_neighbor is None:
                print(
                    f"  No neighbor in batch of {len(neighbors)} improves at "
                    f"iteration {iteration + 1}. Stopping."
                )
                break

            combo = dict(best_neighbor)

        return best_combo, best_accuracy, best_latency, results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._run_selection_async(max_concurrent))
        return self._run_selection_sequential()

    def _run_selection_sequential(self) -> SelectionResults:
        all_results: List[ModelResult] = []
        global_best_combo: Optional[Dict[str, ModelCandidate]] = None
        global_best_accuracy = float("-inf")
        global_best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'=' * 60}")
        print(
            f"Hill climbing (sequential): {self.num_restarts} restart(s), "
            f"max {self.max_iterations} iterations each, patience {self.patience}"
        )
        print(f"{'=' * 60}\n")

        seen: Set[str] = set()
        for restart in range(self.num_restarts):
            print(f"--- Restart {restart + 1}/{self.num_restarts} ---")
            result = self._hill_climb_once_sequential(seen)
            if result is None:
                print("  All combinations exhausted. Stopping.\n")
                break
            best_combo, best_acc, best_lat, run_results = result
            all_results.extend(run_results)

            if (
                global_best_combo is None
                or best_acc > global_best_accuracy + tol
                or (
                    abs(best_acc - global_best_accuracy) <= tol
                    and best_lat < global_best_latency
                )
            ):
                global_best_accuracy = best_acc
                global_best_latency = best_lat
                global_best_combo = best_combo

        return self._hc_finalize(all_results, global_best_combo, global_best_accuracy)

    async def _run_selection_async(self, max_concurrent: int = 20,) -> SelectionResults:
        all_results: List[ModelResult] = []
        global_best_combo: Optional[Dict[str, ModelCandidate]] = None
        global_best_accuracy = float("-inf")
        global_best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'=' * 60}")
        print(
            f"Hill climbing (parallel): {self.num_restarts} restart(s), "
            f"max {self.max_iterations} iterations each, patience {self.patience}"
        )
        print(f"{'=' * 60}\n")

        seen: Set[str] = set()
        for restart in range(self.num_restarts):
            print(f"--- Restart {restart + 1}/{self.num_restarts} ---")
            result = await self._hill_climb_once_async(
                seen, max_concurrent=max_concurrent
            )
            if result is None:
                print("  All combinations exhausted. Stopping.\n")
                break
            best_combo, best_acc, best_lat, run_results = result
            all_results.extend(run_results)

            if (
                global_best_combo is None
                or best_acc > global_best_accuracy + tol
                or (
                    abs(best_acc - global_best_accuracy) <= tol
                    and best_lat < global_best_latency
                )
            ):
                global_best_accuracy = best_acc
                global_best_latency = best_lat
                global_best_combo = best_combo

        return self._hc_finalize(all_results, global_best_combo, global_best_accuracy)
