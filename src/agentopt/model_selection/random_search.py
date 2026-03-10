"""
Random-search model selection: evaluates a random subset of the Cartesian
product of candidate models across all proxies.
"""

import itertools
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class RandomSearchModelSelector(BaseModelSelector):
    """
    Selects the best model combination from a random subset of candidates.

    Supports both sequential (proxy-swap) and parallel (agent-clone)
    evaluation modes via the ``parallel`` parameter on ``select_best()``.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
        clone_fn: Optional[Callable] = None,
        sample_fraction: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            clone_fn=clone_fn,
            dataset=dataset,
        )
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")

        self.sample_fraction = sample_fraction
        self.seed = seed

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """
        Select the best model combination from a random subset.

        Args:
            parallel: If True, clone the agent per sampled combination and
                evaluate concurrently with a thread pool. If False (default),
                swap proxies in-place and evaluate sequentially.
            max_workers: Max threads for parallel mode. Defaults to the
                number of sampled combinations.

        Returns:
            SelectionResults containing all sampled model evaluation results.
        """
        if parallel:
            try:
                return self._select_parallel(max_workers)
            except RuntimeError as e:
                logger.warning(
                    "Parallel evaluation failed, falling back to sequential: %s", e
                )

        return self._select_sequential()

    def _get_sampled_combinations(
        self,
    ) -> Tuple[List[ModelProxy], List[tuple], List[tuple]]:
        """Return proxies, full search space, and sampled combinations."""
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))

        total_combinations = len(all_combinations)
        sample_size = max(1, math.ceil(total_combinations * self.sample_fraction))
        sample_size = min(sample_size, total_combinations)

        if sample_size == total_combinations:
            sampled_combinations = all_combinations
        else:
            rng = random.Random(self.seed)
            sampled_indices = sorted(rng.sample(range(total_combinations), sample_size))
            sampled_combinations = [all_combinations[idx] for idx in sampled_indices]

        return proxies, all_combinations, sampled_combinations

    # ------------------------------------------------------------------
    # Sequential evaluation (proxy swap in-place)
    # ------------------------------------------------------------------

    def _select_sequential(self) -> SelectionResults:
        proxies, all_combinations, sampled_combinations = (
            self._get_sampled_combinations()
        )

        all_results: List[ModelResult] = []
        best_combination = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9

        print(f"\n{'='*60}")
        print(
            "Random search (sequential): "
            f"{len(sampled_combinations)}/{len(all_combinations)} combinations "
            f"({self.sample_fraction:.1%} sample)"
        )
        print(f"{'='*60}\n")

        for idx, combo in enumerate(sampled_combinations, 1):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            for proxy, model_obj in zip(proxies, combo):
                proxy.set_model(model_obj)

            print(f"  [{idx}/{len(sampled_combinations)}] Evaluating: {combo_name}")
            for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                print(f"    proxy[{i}] → {self._get_model_name(model_obj)}")
            try:
                accuracy, latency = self._evaluate(self.dataset, label=combo_name)

                result = ModelResult(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    attribute="combination",
                    is_best=False,
                )
                print(f"  {result}")
                all_results.append(result)

                should_update = False
                if best_combination is None:
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
                    best_combination = combo

            except Exception as e:
                print(f"  [{combo_name}] failed: {e}")
                all_results.append(
                    ModelResult(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        attribute="combination",
                        is_best=False,
                    )
                )

        if best_combination is not None:
            best_name = " + ".join(self._get_model_name(m) for m in best_combination)
            for proxy, model_obj in zip(proxies, best_combination):
                proxy.set_model(model_obj)
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results

    # ------------------------------------------------------------------
    # Parallel evaluation (agent clones + thread pool)
    # ------------------------------------------------------------------

    def _select_parallel(
        self,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        proxies, all_combinations, sampled_combinations = (
            self._get_sampled_combinations()
        )

        if max_workers is None:
            max_workers = len(sampled_combinations)

        invoke_method_name = self._invoke_method_name
        if invoke_method_name is None and self.clone_fn is None:
            raise RuntimeError(
                "Parallel mode requires either an agent= or a clone_fn= alongside "
                "invoke_fn=. Pass clone_fn= to enable parallel evaluation for "
                "custom multi-agent chains."
            )

        from ..model_proxy.adapter import get_adapter

        adapter = get_adapter(self.agent) if self.agent is not None else None

        print(f"\n{'='*60}")
        print(
            "Random search (parallel): "
            f"{len(sampled_combinations)}/{len(all_combinations)} combinations "
            f"({self.sample_fraction:.1%} sample), {max_workers} workers"
        )
        print(f"{'='*60}\n")

        # Phase 1: Create agent/invoke_fn copies (serial — cloning must not race).
        print(
            f"  Cloning agents for {len(sampled_combinations)} sampled combinations ..."
        )
        tasks: List[Tuple[str, tuple, Any, bool]] = []

        for idx, combo in enumerate(sampled_combinations, 1):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            print(
                f"    clone {idx}/{len(sampled_combinations)}: {combo_name}",
                flush=True,
            )
            for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                print(f"      proxy[{i}] → {self._get_model_name(model_obj)}")

            try:
                if self.clone_fn is not None:
                    model_map = dict(zip(proxies, combo))
                    fresh_invoke = self.clone_fn(model_map)
                    tasks.append((combo_name, combo, fresh_invoke, self.is_async))
                else:
                    if adapter is not None:
                        agent_copy = adapter.clone_for_parallel(
                            self.agent, proxies, combo, self._get_model_name
                        )
                    else:
                        import copy

                        agent_copy = copy.deepcopy(self.agent)
                    invoke_fn = (
                        adapter.get_invoke_fn(agent_copy)
                        if adapter is not None
                        else self._make_invoke_fn(
                            agent_copy, invoke_method_name, self.is_async
                        )
                    )
                    tasks.append((combo_name, combo, invoke_fn, False))
            except Exception as e:
                logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)
                continue

        if not tasks:
            raise RuntimeError("All clone attempts failed. Cannot evaluate any models.")

        # Phase 2: Evaluate in parallel.
        print(
            f"\n  Evaluating {len(tasks)} sampled combinations across {max_workers} workers ..."
        )
        all_results: List[ModelResult] = []
        future_to_info: Dict[Any, Tuple[str, tuple]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for combo_name, combo, invoke_fn, is_async_flag in tasks:
                future = executor.submit(
                    self._evaluate_single,
                    invoke_fn,
                    is_async_flag,
                    self.eval_fn,
                    self.dataset,
                    label=combo_name,
                )
                future_to_info[future] = (combo_name, combo)

            for future in as_completed(future_to_info):
                combo_name, combo = future_to_info[future]
                try:
                    accuracy, latency = future.result()
                    result = ModelResult(
                        model_name=combo_name,
                        accuracy=accuracy,
                        latency_seconds=latency,
                        attribute="combination",
                        is_best=False,
                    )
                    print(f"  {result}")
                    all_results.append(result)
                except Exception as e:
                    print(f"  [{combo_name}] failed: {e}")
                    all_results.append(
                        ModelResult(
                            model_name=combo_name,
                            accuracy=0.0,
                            latency_seconds=0.0,
                            attribute="combination",
                            is_best=False,
                        )
                    )

        # Phase 3: Determine best and set proxies.
        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for combo_name, combo, _, _ in tasks:
                if combo_name == best_name:
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)
                    break
            for r in all_results:
                if r.model_name == best_name:
                    r.is_best = True
                    break
        else:
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results
