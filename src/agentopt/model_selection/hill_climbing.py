"""
Hill-climbing model selector with random restarts.

Uses the model topology (quality / speed rankings) to define
neighbours so that each iteration makes an informed single-step move.
"""

import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from ..model_price import compute_price
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
        max_iterations: int = 20,
        num_restarts: int = 3,
        patience: int = 3,
        seed: Optional[int] = None,
        batch_size: int = 1,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
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
            lambda_cost=lambda_cost,
            lambda_latency=lambda_latency,
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

    def _objective_from_dp(self, dp_results: List[DatapointResult]) -> Optional[float]:
        """Recompute the mean combined objective from cached datapoint results."""
        if not self._has_combined_objective or not dp_results:
            return None
        scores = [dp.score for dp in dp_results]
        lats = [dp.latency_seconds for dp in dp_results]
        costs = [
            compute_price(
                dp.input_tokens, dp.output_tokens, custom_prices=self._custom_prices,
            )
            for dp in dp_results
        ]
        return self._mean_objective(scores, lats, costs)

    def _primary_value(
        self, accuracy: float, dp_results: List[DatapointResult],
    ) -> float:
        """Ranking key for tiebreaks: combined objective if configured, else accuracy."""
        obj = self._objective_from_dp(dp_results)
        return obj if obj is not None else accuracy

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
        """Compute stats, absorb cost samples, cache, and return the eval tuple."""
        self._observe_combo(scores, latencies, dp_ids)
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
        current_value: float,
        current_latency: float,
        tol: float,
    ) -> Optional[Dict[str, ModelCandidate]]:
        """Select the best neighbor from eval results, or None if none improves.

        Ranks by primary value (combined objective when ``lambda_*`` are set,
        else accuracy), with latency as the tiebreaker.
        """
        best_neighbor: Optional[Dict[str, ModelCandidate]] = None
        best_n_val = float("-inf")
        best_n_lat = float("inf")

        for neighbor, eval_result in zip(neighbors, eval_results):
            n_name, n_acc, n_lat, _, _, n_dp_results, _ = eval_result
            seen.add(n_name)
            n_val = self._primary_value(n_acc, n_dp_results)

            if n_val > best_n_val + tol:
                best_neighbor, best_n_val, best_n_lat = neighbor, n_val, n_lat
            elif abs(n_val - best_n_val) <= tol and n_lat < best_n_lat:
                best_neighbor, best_n_val, best_n_lat = neighbor, n_val, n_lat

        if best_neighbor is None or (
            best_n_val < current_value - tol
            or (
                abs(best_n_val - current_value) <= tol
                and best_n_lat >= current_latency
            )
        ):
            return None
        return best_neighbor

    def _hc_finalize(
        self,
        all_results: List[ModelResult],
        global_best_combo: Optional[Dict[str, ModelCandidate]],
        global_best_value: float,
    ) -> SelectionResults:
        """Finalize combined objectives, mark the best result, return results."""
        self._finalize_combined_objectives(all_results)
        if global_best_combo is None:
            print("\nNo combinations succeeded\n")
            return SelectionResults(results=all_results)

        # Prefer the combined-objective-aware _find_best when lambdas are set;
        # otherwise honor the within-search global best to preserve the
        # original tie-breaking semantics.
        if self._has_combined_objective:
            best_info = self._find_best(all_results)
            best_name = best_info[0] if best_info else self._combo_name(global_best_combo)
        else:
            best_name = self._combo_name(global_best_combo)

        tol = 1e-9
        for result in all_results:
            if result.model_name != best_name:
                continue
            if self._has_combined_objective:
                result.is_best = True
                break
            # Accuracy-mode: match by name AND the tracked best value.
            if abs(result.accuracy - global_best_value) < tol:
                result.is_best = True
                break
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
        best_value = float("-inf")
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

            current_value = self._primary_value(accuracy, dp_results)
            should_update = (
                best_value == float("-inf")
                or current_value > best_value + tol
                or (abs(current_value - best_value) <= tol and latency < best_latency)
            )
            if should_update:
                best_value, best_latency, best_combo = current_value, latency, dict(combo)
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
                eval_results, neighbors, seen, current_value, latency, tol
            )
            if best_neighbor is None:
                print(
                    f"  No neighbor in batch of {len(neighbors)} improves at "
                    f"iteration {iteration + 1}. Stopping."
                )
                break

            combo = dict(best_neighbor)

        return best_combo, best_value, best_latency, results

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
        best_value = float("-inf")
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

            current_value = self._primary_value(accuracy, dp_results)
            should_update = (
                best_value == float("-inf")
                or current_value > best_value + tol
                or (abs(current_value - best_value) <= tol and latency < best_latency)
            )
            if should_update:
                best_value, best_latency, best_combo = current_value, latency, dict(combo)
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
                eval_results, neighbors, seen, current_value, latency, tol
            )
            if best_neighbor is None:
                print(
                    f"  No neighbor in batch of {len(neighbors)} improves at "
                    f"iteration {iteration + 1}. Stopping."
                )
                break

            combo = dict(best_neighbor)

        return best_combo, best_value, best_latency, results

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
        global_best_value = float("-inf")
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
            best_combo, best_val, best_lat, run_results = result
            all_results.extend(run_results)

            if (
                global_best_combo is None
                or best_val > global_best_value + tol
                or (
                    abs(best_val - global_best_value) <= tol
                    and best_lat < global_best_latency
                )
            ):
                global_best_value = best_val
                global_best_latency = best_lat
                global_best_combo = best_combo

        return self._hc_finalize(all_results, global_best_combo, global_best_value)

    async def _run_selection_async(self, max_concurrent: int = 20,) -> SelectionResults:
        all_results: List[ModelResult] = []
        global_best_combo: Optional[Dict[str, ModelCandidate]] = None
        global_best_value = float("-inf")
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
            best_combo, best_val, best_lat, run_results = result
            all_results.extend(run_results)

            if (
                global_best_combo is None
                or best_val > global_best_value + tol
                or (
                    abs(best_val - global_best_value) <= tol
                    and best_lat < global_best_latency
                )
            ):
                global_best_value = best_val
                global_best_latency = best_lat
                global_best_combo = best_combo

        return self._hc_finalize(all_results, global_best_combo, global_best_value)
