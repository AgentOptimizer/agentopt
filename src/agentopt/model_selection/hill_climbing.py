"""
Hill-climbing model selector with random restarts.

Uses the model topology (quality / speed rankings) to define
neighbours so that each iteration makes an informed single-step move
rather than blindly exploring the full Cartesian product.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ..base_models import EvalFn
from ..model_proxy import ModelProxy
from ..model_topology import (
    normalize_model_name,
    get_higher_quality_neighbor,
    get_faster_neighbor,
)
from .base import BaseModelSelector, ModelResult, SelectionResults


class HillClimbingModelSelector(BaseModelSelector):
    """Select models via stochastic hill climbing with random restarts.

    Algorithm (per restart):
      1. Pick a random model for each proxy from its candidate list.
      2. Evaluate accuracy and latency.
      3. If accuracy < *accuracy_target* → swap a random proxy's model to the
         next-higher-quality neighbour (one step up in the topology).
      4. Elif latency > *latency_target* → swap a random proxy's model to the
         next-faster neighbour.
      5. Repeat until both targets are met, no move is possible, or
         *max_iterations* is reached.

    The best combination across all restarts is returned.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: List[Tuple[Any, str]],
        agent: Any = None,
        invoke_fn: Optional[callable] = None,
        accuracy_target: float = 0.8,
        latency_target: float = 10.0,
        max_iterations: int = 20,
        num_restarts: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
        )
        self.accuracy_target = accuracy_target
        self.latency_target = latency_target
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts

        if seed is not None:
            random.seed(seed)

        # Pre-compute candidate name lists for topology lookups.
        self._candidate_names: Dict[ModelProxy, List[str]] = {}
        for proxy, candidates in self._models.items():
            self._candidate_names[proxy] = [self._get_model_name(c) for c in candidates]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_candidate_by_name(self, proxy: ModelProxy, name: str) -> Any:
        """Find the original candidate object that matches *name*."""
        norm_name = normalize_model_name(name)
        for candidate in self._models[proxy]:
            if normalize_model_name(self._get_model_name(candidate)) == norm_name:
                return candidate
        return name  # fallback: pass the string directly

    def _apply_combination(self, proxies: List[ModelProxy], combo: List[Any]) -> None:
        for proxy, model_obj in zip(proxies, combo):
            proxy.set_model(model_obj)

    def _random_combination(self, proxies: List[ModelProxy]) -> List[Any]:
        return [random.choice(self._models[p]) for p in proxies]

    # ------------------------------------------------------------------
    # Move operators
    # ------------------------------------------------------------------

    def _try_improve_quality(self, proxies: List[ModelProxy], combo: List[Any]) -> bool:
        """Swap a random proxy's model to the next-higher-quality neighbour.

        Tries all proxies (in shuffled order) until one can move.
        Modifies *combo* in-place.  Returns ``True`` if a move was made.
        """
        indices = list(range(len(proxies)))
        random.shuffle(indices)
        for idx in indices:
            proxy = proxies[idx]
            current_name = self._get_model_name(combo[idx])
            neighbor_name = get_higher_quality_neighbor(
                current_name, self._candidate_names[proxy]
            )
            if neighbor_name is not None:
                combo[idx] = self._find_candidate_by_name(proxy, neighbor_name)
                return True
        return False

    def _try_improve_speed(self, proxies: List[ModelProxy], combo: List[Any]) -> bool:
        """Swap a random proxy's model to the next-faster neighbour.

        Tries all proxies (in shuffled order) until one can move.
        Modifies *combo* in-place.  Returns ``True`` if a move was made.
        """
        indices = list(range(len(proxies)))
        random.shuffle(indices)
        for idx in indices:
            proxy = proxies[idx]
            current_name = self._get_model_name(combo[idx])
            neighbor_name = get_faster_neighbor(
                current_name, self._candidate_names[proxy]
            )
            if neighbor_name is not None:
                combo[idx] = self._find_candidate_by_name(proxy, neighbor_name)
                return True
        return False

    # ------------------------------------------------------------------
    # Single restart
    # ------------------------------------------------------------------

    def _hill_climb_once(
        self, proxies: List[ModelProxy]
    ) -> Tuple[List[Any], float, float, List[ModelResult]]:
        """Run one hill-climbing pass from a random starting point."""
        combo = self._random_combination(proxies)
        self._apply_combination(proxies, combo)

        results: List[ModelResult] = []
        best_combo = list(combo)
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9

        for iteration in range(self.max_iterations):
            accuracy, latency = self._evaluate(self.dataset)
            combo_name = " + ".join(self._get_model_name(m) for m in combo)

            print(
                f"  Iter {iteration + 1}: [{combo_name}] "
                f"Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
            )

            results.append(
                ModelResult(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    attribute="combination",
                    is_best=False,
                )
            )

            # Track best within this restart
            should_update = False
            if best_accuracy == float("-inf"):
                should_update = True
            elif accuracy > best_accuracy + accuracy_tolerance:
                should_update = True
            elif (
                abs(accuracy - best_accuracy) <= accuracy_tolerance
                and latency < best_latency
            ):
                should_update = True

            if should_update:
                best_accuracy = accuracy
                best_latency = latency
                best_combo = list(combo)

            # Both targets met → stop early
            if accuracy >= self.accuracy_target and latency <= self.latency_target:
                print(f"  Targets met at iteration {iteration + 1}!")
                break

            # Attempt a move
            moved = False
            if accuracy < self.accuracy_target:
                moved = self._try_improve_quality(proxies, combo)
            if not moved and latency > self.latency_target:
                moved = self._try_improve_speed(proxies, combo)

            if not moved:
                print(
                    f"  No improving moves available at iteration {iteration + 1}. "
                    "Stopping."
                )
                break

            self._apply_combination(proxies, combo)

        return best_combo, best_accuracy, best_latency, results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(self) -> SelectionResults:
        proxies = list(self._models.keys())

        all_results: List[ModelResult] = []
        global_best_combo: Optional[List[Any]] = None
        global_best_accuracy = float("-inf")
        global_best_latency = float("inf")
        accuracy_tolerance = 1e-9

        print(f"\n{'=' * 60}")
        print(
            f"Hill climbing: {self.num_restarts} restart(s), "
            f"max {self.max_iterations} iterations each"
        )
        print(
            f"Targets: accuracy >= {self.accuracy_target:.2%}, "
            f"latency <= {self.latency_target:.2f}s"
        )
        print(f"{'=' * 60}\n")

        for restart in range(self.num_restarts):
            print(f"--- Restart {restart + 1}/{self.num_restarts} ---")

            best_combo, best_acc, best_lat, run_results = self._hill_climb_once(proxies)
            all_results.extend(run_results)

            # Update global best
            should_update = False
            if global_best_combo is None:
                should_update = True
            elif best_acc > global_best_accuracy + accuracy_tolerance:
                should_update = True
            elif (
                abs(best_acc - global_best_accuracy) <= accuracy_tolerance
                and best_lat < global_best_latency
            ):
                should_update = True

            if should_update:
                global_best_accuracy = best_acc
                global_best_latency = best_lat
                global_best_combo = best_combo

            # Early exit across restarts
            if best_acc >= self.accuracy_target and best_lat <= self.latency_target:
                print(f"Targets met in restart {restart + 1}. Stopping restarts.\n")
                break

        # Apply the global best and mark it in results
        if global_best_combo is not None:
            best_name = " + ".join(self._get_model_name(m) for m in global_best_combo)
            self._apply_combination(proxies, global_best_combo)
            print(
                f"Best combination: {best_name} "
                f"(accuracy: {global_best_accuracy:.2%}, "
                f"latency: {global_best_latency:.2f}s)\n"
            )

            for result in all_results:
                if (
                    result.model_name == best_name
                    and abs(result.accuracy - global_best_accuracy) < accuracy_tolerance
                ):
                    result.is_best = True
                    break
        else:
            print("\nNo combinations succeeded\n")

        return SelectionResults(results=all_results)
