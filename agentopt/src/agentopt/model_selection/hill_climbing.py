"""
Hill-climbing model selector with random restarts.

Uses the model topology (quality / speed rankings) to define
neighbours so that each iteration makes an informed single-step move.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from ..model_topology import get_faster_neighbor, get_higher_quality_neighbor
from .base import BaseModelSelector, ModelResult, SelectionResults


class HillClimbingModelSelector(BaseModelSelector):
    """Select models via stochastic hill climbing with random restarts."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        max_iterations: int = 20,
        num_restarts: int = 3,
        patience: int = 3,
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
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts
        self.patience = patience

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
    # Helpers
    # ------------------------------------------------------------------

    def _random_combination(
        self, seen: Set[str]
    ) -> Optional[Dict[str, ModelCandidate]]:
        """Pick a random unseen combination, or ``None`` if all exhausted."""
        unseen = [c for c in self._all_combo_list if self._combo_name(c) not in seen]
        if unseen:
            return dict(random.choice(unseen))
        return None

    # ------------------------------------------------------------------
    # Move operators
    # ------------------------------------------------------------------

    def _try_improve_quality(
        self, combo: Dict[str, ModelCandidate], seen: Set[str]
    ) -> bool:
        """Swap a random node's model to the next-higher-quality neighbour.

        Modifies *combo* in-place. Returns ``True`` if a move was made.
        """
        node_names = list(combo.keys())
        random.shuffle(node_names)
        for node in node_names:
            current = combo[node]
            neighbor = get_higher_quality_neighbor(current, self._models[node])
            if neighbor is not None:
                old = combo[node]
                combo[node] = neighbor
                if self._combo_name(combo) not in seen:
                    return True
                combo[node] = old
        return False

    def _try_improve_speed(
        self, combo: Dict[str, ModelCandidate], seen: Set[str]
    ) -> bool:
        """Swap a random node's model to the next-faster neighbour.

        Modifies *combo* in-place. Returns ``True`` if a move was made.
        """
        node_names = list(combo.keys())
        random.shuffle(node_names)
        for node in node_names:
            current = combo[node]
            neighbor = get_faster_neighbor(current, self._models[node])
            if neighbor is not None:
                old = combo[node]
                combo[node] = neighbor
                if self._combo_name(combo) not in seen:
                    return True
                combo[node] = old
        return False

    # ------------------------------------------------------------------
    # Single restart
    # ------------------------------------------------------------------

    def _hill_climb_once(
        self, seen: Set[str]
    ) -> Optional[Tuple[Dict[str, ModelCandidate], float, float, List[ModelResult]]]:
        """Run one hill-climbing pass from a random starting point."""
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

            print(f"  Iter {iteration + 1}: {combo_name}")

            # Re-use cached result if available.
            if combo_name in self._eval_cache:
                (
                    accuracy,
                    latency,
                    input_tokens,
                    output_tokens,
                    dp_results,
                ) = self._eval_cache[combo_name]
                cached = True
            else:
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, self.dataset, label=combo_name
                )
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
                cached = False

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

            # Track best within this restart.
            should_update = False
            if best_accuracy == float("-inf"):
                should_update = True
            elif accuracy > best_accuracy + tol:
                should_update = True
            elif abs(accuracy - best_accuracy) <= tol and latency < best_latency:
                should_update = True

            if should_update:
                best_accuracy = accuracy
                best_latency = latency
                best_combo = dict(combo)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                print(
                    f"  No improvement for {self.patience} iterations. "
                    f"Converged at iteration {iteration + 1}."
                )
                break

            # Attempt a move.
            moved = False
            if accuracy < 1.0:
                moved = self._try_improve_quality(combo, seen)
            if not moved:
                moved = self._try_improve_speed(combo, seen)

            if not moved:
                print(
                    f"  No improving moves available at iteration {iteration + 1}. "
                    "Stopping."
                )
                break

        return best_combo, best_accuracy, best_latency, results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        assert (
            not parallel
        ), "HillClimbingModelSelector does not support parallel evaluation."

        all_results: List[ModelResult] = []
        global_best_combo: Optional[Dict[str, ModelCandidate]] = None
        global_best_accuracy = float("-inf")
        global_best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'=' * 60}")
        print(
            f"Hill climbing: {self.num_restarts} restart(s), "
            f"max {self.max_iterations} iterations each, "
            f"patience {self.patience}"
        )
        print(f"{'=' * 60}\n")

        seen: Set[str] = set()

        for restart in range(self.num_restarts):
            print(f"--- Restart {restart + 1}/{self.num_restarts} ---")

            result = self._hill_climb_once(seen)
            if result is None:
                print("  All combinations exhausted. Stopping.\n")
                break
            best_combo, best_acc, best_lat, run_results = result
            all_results.extend(run_results)

            should_update = False
            if global_best_combo is None:
                should_update = True
            elif best_acc > global_best_accuracy + tol:
                should_update = True
            elif (
                abs(best_acc - global_best_accuracy) <= tol
                and best_lat < global_best_latency
            ):
                should_update = True

            if should_update:
                global_best_accuracy = best_acc
                global_best_latency = best_lat
                global_best_combo = best_combo

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

        results = SelectionResults(results=all_results)
        return results
