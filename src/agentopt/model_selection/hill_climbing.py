"""
Hill-climbing model selector with random restarts.

Uses the model topology (quality / speed rankings) to define
neighbours so that each iteration makes an informed single-step move
rather than blindly exploring the full Cartesian product.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn
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
      3. If accuracy < 1.0, try swapping a random proxy's model to the
         next-higher-quality neighbour (accuracy is the primary objective).
      4. If accuracy is already perfect (>= 1.0) or no quality move is
         available, try the next-faster neighbour to reduce latency.
      5. Repeat until no improvement for *patience* consecutive iterations,
         no move is possible, or *max_iterations* is reached.

    Accuracy is treated as having an upper bound of 1.0.  Once every
    evaluation sample is answered correctly the algorithm switches
    entirely to latency optimisation.

    The best combination across all restarts is returned.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[callable] = None,
        max_iterations: int = 20,
        num_restarts: int = 3,
        patience: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the hill-climbing model selector.

        Args:
            models: Dictionary mapping ModelProxy to list of model candidates.
            eval_fn: Function ``(expected, actual) -> bool | float``
                (higher is better).
            dataset: Sequence of ``(input_data, expected_answer)`` pairs.
            agent: Agent instance (CrewAI, LangChain, LangGraph).
                Mutually exclusive with *invoke_fn*.
            invoke_fn: Callable for a custom agent.
                Mutually exclusive with *agent*.
            max_iterations: Maximum iterations per restart before stopping.
            num_restarts: Number of random restarts.
            patience: Stop a restart after this many consecutive iterations
                with no improvement (accuracy primary, latency tiebreaker).
            seed: Random seed for reproducibility.
        """
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
        )
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts
        self.patience = patience

        if seed is not None:
            random.seed(seed)

        # Pre-compute candidate name lists for topology lookups.
        self._candidate_names: Dict[ModelProxy, List[str]] = {}
        for proxy, candidates in self._models.items():
            self._candidate_names[proxy] = [self._get_model_name(c) for c in candidates]

        # Pre-compute all combinations (Cartesian product) once.
        import itertools

        proxies = list(self._models.keys())
        candidate_lists = [self._models[p] for p in proxies]
        self._all_combinations: List[List[Any]] = [
            list(c) for c in itertools.product(*candidate_lists)
        ]

        # Cache: combo_key -> (accuracy, latency) to avoid redundant evaluations.
        self._eval_cache: Dict[str, Tuple[float, float]] = {}

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

    def _random_combination(self, seen: Set[str]) -> Optional[List[Any]]:
        """Pick a random unseen combination, or ``None`` if all exhausted."""
        unseen = [c for c in self._all_combinations if self._combo_key(c) not in seen]
        if unseen:
            return list(random.choice(unseen))
        return None

    def _combo_key(self, combo: List[Any]) -> str:
        """Canonical string key for a combination (for dedup / caching)."""
        return " + ".join(self._get_model_name(m) for m in combo)

    # ------------------------------------------------------------------
    # Move operators
    # ------------------------------------------------------------------

    def _try_improve_quality(
        self, proxies: List[ModelProxy], combo: List[Any], seen: Set[str]
    ) -> bool:
        """Swap a random proxy's model to the next-higher-quality neighbour.

        Skips moves that would land on an already-seen combination.
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
                old = combo[idx]
                combo[idx] = self._find_candidate_by_name(proxy, neighbor_name)
                if self._combo_key(combo) not in seen:
                    return True
                combo[idx] = old  # revert, try next proxy
        return False

    def _try_improve_speed(
        self, proxies: List[ModelProxy], combo: List[Any], seen: Set[str]
    ) -> bool:
        """Swap a random proxy's model to the next-faster neighbour.

        Skips moves that would land on an already-seen combination.
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
                old = combo[idx]
                combo[idx] = self._find_candidate_by_name(proxy, neighbor_name)
                if self._combo_key(combo) not in seen:
                    return True
                combo[idx] = old  # revert, try next proxy
        return False

    # ------------------------------------------------------------------
    # Single restart
    # ------------------------------------------------------------------

    def _hill_climb_once(
        self, proxies: List[ModelProxy], seen: Set[str]
    ) -> Optional[Tuple[List[Any], float, float, List[ModelResult]]]:
        """Run one hill-climbing pass from a random starting point.

        Returns ``None`` if all combinations have already been seen.
        """
        combo = self._random_combination(seen)
        if combo is None:
            return None
        self._apply_combination(proxies, combo)

        results: List[ModelResult] = []
        best_combo = list(combo)
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9
        no_improve_count = 0

        for iteration in range(self.max_iterations):
            combo_name = self._combo_key(combo)
            seen.add(combo_name)

            print(f"  Iter {iteration + 1} assignment:")
            for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                print(f"    proxy[{i}] → {self._get_model_name(model_obj)}")

            # Re-use cached result if this combination was already evaluated.
            if combo_name in self._eval_cache:
                accuracy, latency, tokens = self._eval_cache[combo_name]
                cached = True
            else:
                accuracy, latency, tokens = self._evaluate(self.dataset)
                self._eval_cache[combo_name] = (accuracy, latency, tokens)
                cached = False

            in_tokens, out_tokens = self._split_tokens(tokens)
            result = ModelResult(
                model_name=combo_name,
                accuracy=accuracy,
                latency_seconds=latency,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                attribute="combination",
                is_best=False,
            )
            suffix = " (cached)" if cached else ""
            print(f"  Iter {iteration + 1}: {result}{suffix}")
            results.append(result)

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
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Patience-based convergence: stop if no improvement for N iterations
            if no_improve_count >= self.patience:
                print(
                    f"  No improvement for {self.patience} iterations. "
                    f"Converged at iteration {iteration + 1}."
                )
                break

            # Attempt a move: improve quality unless accuracy is already
            # perfect (>= 1.0), in which case focus on speed instead.
            moved = False
            if accuracy < 1.0:
                moved = self._try_improve_quality(proxies, combo, seen)
            if not moved:
                moved = self._try_improve_speed(proxies, combo, seen)

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
        """Run hill climbing with random restarts and return the best combination.

        Returns:
            SelectionResults with all evaluated combinations; the best one
            is marked with ``is_best=True``.  The proxies are left set to
            the best combination found.
        """
        proxies = list(self._models.keys())

        all_results: List[ModelResult] = []
        global_best_combo: Optional[List[Any]] = None
        global_best_accuracy = float("-inf")
        global_best_latency = float("inf")
        accuracy_tolerance = 1e-9

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

            result = self._hill_climb_once(proxies, seen)
            if result is None:
                print("  All combinations exhausted. Stopping.\n")
                break
            best_combo, best_acc, best_lat, run_results = result
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

        # Apply the global best and mark it in results
        if global_best_combo is not None:
            best_name = " + ".join(self._get_model_name(m) for m in global_best_combo)
            self._apply_combination(proxies, global_best_combo)

            for result in all_results:
                if (
                    result.model_name == best_name
                    and abs(result.accuracy - global_best_accuracy) < accuracy_tolerance
                ):
                    result.is_best = True
                    break
        else:
            print("\nNo combinations succeeded\n")

        results = SelectionResults(results=all_results)
        print(results)
        return results
