"""
Hyperband model selector.

Implements the full Hyperband algorithm using dataset samples as the
resource, with successive halving as the inner loop.

This selector is designed to be simple for end users:

  - Uses ``len(dataset)`` as the maximum resource per configuration.
  - Requires only a reduction factor (``reduction_factor`` / η).
  - Automatically constructs multiple brackets with different trade-offs
    between the number of configurations and per-configuration budget.
"""

import itertools
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults, _CACHE_SENTINEL

logger = logging.getLogger(__name__)


class HyperbandModelSelector(BaseModelSelector):
    """Select models using the full Hyperband algorithm.

    Hyperband treats the number of dataset samples as the resource ``r``.
    For each bracket, it:

      1. Samples a subset of model combinations.
      2. Runs successive halving across several stages, where each stage
         increases the number of samples per surviving combination and
         prunes the worst-performing ones.

    Parameters
    ----------
    reduction_factor:
        Hyperband's η (> 1). Controls both how aggressively we prune and
        how quickly the resource per configuration grows across stages.
        Typical values are in the range [2, 4].
    max_resource:
        Maximum resource (number of samples) per configuration. If not
        provided, defaults to ``len(dataset)``.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
        cache: Optional["EvalCache"] = _CACHE_SENTINEL,
        reduction_factor: float = 3.0,
        max_resource: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        if reduction_factor <= 1.0:
            raise ValueError("reduction_factor (η) must be > 1.0.")

        super().__init__(
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            agent=agent,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
            cache=cache,
        )

        self.reduction_factor = reduction_factor

        n = len(self.dataset)
        if n == 0:
            raise ValueError("Hyperband requires a non-empty dataset.")

        if max_resource is None:
            self.max_resource = n
        else:
            if max_resource <= 0:
                raise ValueError("max_resource must be positive.")
            self.max_resource = min(max_resource, n)

        # Precompute Hyperband schedule constants.
        self._s_max = int(
            math.floor(math.log(self.max_resource, self.reduction_factor))
        )
        self._B = (self._s_max + 1) * self.max_resource

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """Run Hyperband and return results.

        Args:
            parallel: If True, evaluate active combinations concurrently
                within each stage of each bracket. Requires either
                ``agent=`` or ``clone_fn=`` to be set.
            max_workers: Thread-pool size for parallel mode.

        Returns:
            SelectionResults with all evaluated combinations.
        """
        if parallel:
            try:
                return self._select_parallel(max_workers)
            except RuntimeError as e:
                logger.warning(
                    "Parallel evaluation failed, falling back to sequential: %s", e
                )
        return self._select_sequential()

    # ------------------------------------------------------------------
    # Sequential implementation
    # ------------------------------------------------------------------

    def _select_sequential(self) -> SelectionResults:
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))
        dataset_list = list(self.dataset)

        total_configs = len(all_combinations)

        combo_scores: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in range(total_configs)}
        combo_tokens: Dict[int, Dict[str, Tuple[int, int]]] = {
            i: {} for i in range(total_configs)
        }

        print(f"\n{'='*60}")
        print(
            f"Hyperband (sequential): {total_configs} combinations, "
            f"max_resource={self.max_resource}, η={self.reduction_factor}"
        )
        print(f"  s_max={self._s_max}, B={self._B}")
        print(f"{'='*60}")

        # Iterate brackets from most aggressive to least (standard Hyperband).
        for s in reversed(range(self._s_max + 1)):
            # Number of configurations (theoretical) and initial resource for this bracket.
            n_s = int(
                math.ceil(
                    (self._B / self.max_resource) * (self.reduction_factor**s) / (s + 1)
                )
            )
            r_s = int(self.max_resource * (self.reduction_factor ** (-s)))
            r_s = max(1, min(r_s, self.max_resource))

            # In this implementation we evaluate all configurations in every bracket
            # (no subsampling). n_s is used only for logging/intuition.
            bracket_indices = list(range(total_configs))

            print(
                f"\nBracket s={s}: n_s={n_s} (using all {len(bracket_indices)} configs), "
                f"r_s={r_s}"
            )

            # Successive halving within this bracket.
            n_i = len(bracket_indices)
            r_i = r_s
            prev_r = 0

            for i in range(s + 1):
                if n_i <= 0:
                    break

                current_indices = bracket_indices[:n_i]
                r_i = int(r_s * (self.reduction_factor**i))
                r_i = max(1, min(r_i, self.max_resource))

                if r_i <= prev_r:
                    # No additional resource to allocate; skip further stages.
                    break

                print(
                    f"\n  Stage i={i}: n_i={n_i}, r_i={r_i} (prev_r={prev_r}), "
                    f"configs={len(current_indices)}"
                )

                batch = dataset_list[prev_r:r_i]
                if not batch:
                    print("    No additional samples available for this stage.")
                    break

                # Evaluate each active configuration on the new batch slice.
                for idx in current_indices:
                    combo = all_combinations[idx]
                    combo_name = " + ".join(self._get_model_name(m) for m in combo)
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)

                    scores, latencies, tokens = self._evaluate_sequential(
                        batch, label=combo_name
                    )
                    combo_scores[idx].extend(scores)
                    combo_latencies[idx].extend(latencies)
                    self._merge_tokens(combo_tokens[idx], tokens)

                    mu = (
                        sum(combo_scores[idx]) / len(combo_scores[idx])
                        if combo_scores[idx]
                        else 0.0
                    )
                    print(f"    {combo_name}: μ={mu:.3f} (n={len(combo_scores[idx])})")

                prev_r = r_i

                if i == s:
                    # Last stage for this bracket — no further pruning.
                    break

                # Rank current configurations by mean score and keep top fraction.
                means: List[Tuple[int, float]] = []
                for idx in current_indices:
                    scores = combo_scores[idx]
                    mu = sum(scores) / len(scores) if scores else 0.0
                    means.append((idx, mu))

                means.sort(key=lambda x: x[1], reverse=True)
                n_i_next = int(math.floor(n_i / self.reduction_factor))
                n_i_next = max(1, n_i_next)

                new_indices = [idx for idx, _ in means[:n_i_next]]
                eliminated = set(current_indices) - set(new_indices)

                if eliminated:
                    for idx in sorted(eliminated):
                        combo_name = " + ".join(
                            self._get_model_name(m) for m in all_combinations[idx]
                        )
                        print(f"    Eliminated: {combo_name}")
                else:
                    print("    No eliminations in this stage.")

                bracket_indices = new_indices
                n_i = len(bracket_indices)

        # Build final results across all configurations.
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combinations):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            scores = combo_scores[idx]
            latencies = combo_latencies[idx]
            if scores:
                accuracy = sum(scores) / len(scores)
                avg_latency = sum(latencies) / len(latencies)
            else:
                accuracy, avg_latency = 0.0, 0.0
            in_tokens, out_tokens = self._split_tokens(combo_tokens[idx])
            all_results.append(
                self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=avg_latency,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    attribute="combination",
                    is_best=False,
                )
            )

        # Mark best and restore proxies.
        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
            for idx, combo in enumerate(all_combinations):
                if " + ".join(self._get_model_name(m) for m in combo) == best_name:
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)
                    break
        else:
            print("\n  No combinations succeeded.")

        results = SelectionResults(results=all_results)
        print(results)
        return results

    # ------------------------------------------------------------------
    # Parallel implementation
    # ------------------------------------------------------------------

    def _select_parallel(self, max_workers: Optional[int] = None) -> SelectionResults:
        from ..model_proxy.adapter import get_adapter
        from ..model_proxy.builders import build_llm
        from ..model_proxy.model_copy import clone_model_spec
        from ..model_proxy.token_tracking import TokenAccumulator

        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))
        dataset_list = list(self.dataset)

        total_configs = len(all_combinations)

        adapter = get_adapter(self.agent) if self.agent is not None else self._adapter

        if max_workers is None:
            max_workers = total_configs

        print(f"\n{'='*60}")
        print(
            f"Hyperband (parallel): {total_configs} combinations, "
            f"max_resource={self.max_resource}, η={self.reduction_factor}, "
            f"{max_workers} workers"
        )
        print(f"  s_max={self._s_max}, B={self._B}")
        print(f"{'='*60}")

        if self.agent is not None:
            # Agent-based path: clone per combination (serial cloning).
            print(f"\n  Cloning agents for {total_configs} combinations ...")
            tasks: Dict[int, Tuple[str, tuple, Callable, bool, Any]] = {}
            for idx, combo in enumerate(all_combinations):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(f"    clone {idx+1}/{total_configs}: {combo_name}", flush=True)
                try:
                    if adapter is not None:
                        agent_copy = adapter.clone_for_parallel(
                            self.agent, proxies, combo, self._get_model_name
                        )
                        invoke_fn = adapter.get_invoke_fn(agent_copy)
                    else:
                        import copy

                        agent_copy = copy.deepcopy(self.agent)
                        invoke_fn = self._make_invoke_fn(
                            agent_copy, self._invoke_method_name, self.is_async
                        )
                    tracker = (
                        adapter.create_token_tracker(agent_copy)
                        if adapter is not None
                        else None
                    )
                    tasks[idx] = (combo_name, combo, invoke_fn, False, tracker)
                except Exception as e:
                    logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)

            if not tasks:
                raise RuntimeError(
                    "All clone attempts failed. Cannot evaluate any models."
                )
        else:
            # invoke_fn path: prepare task entries (thread-local will be set per round).
            tasks = {}
            for idx, combo in enumerate(all_combinations):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                tasks[idx] = (combo_name, combo, self.invoke_fn, self.is_async, None)

        combo_scores: Dict[int, List[float]] = {i: [] for i in tasks}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in tasks}
        combo_tokens: Dict[int, Dict[str, Tuple[int, int]]] = {i: {} for i in tasks}

        use_thread_local = self.agent is None

        def _eval_batch_thread_local(
            combo: tuple,
            batch: List[Tuple[Any, Any]],
            label: str,
        ) -> Tuple[List[float], List[float], Dict[str, Tuple[int, int]]]:
            """Set thread-local models, evaluate batch, then clean up."""
            tracker = TokenAccumulator()
            for proxy, model_spec in zip(proxies, combo):
                if isinstance(model_spec, str):
                    model_name = self._get_model_name(model_spec)
                    current_model = object.__getattribute__(proxy, "_optmodel")
                    fresh_model = build_llm(model_name, current_model)
                    if fresh_model is None:
                        raise RuntimeError(
                            f"Cannot build model '{model_name}' for parallel evaluation."
                        )
                else:
                    fresh_model = clone_model_spec(model_spec)
                proxy._set_thread_model(fresh_model, tracker)
            try:
                return self._evaluate_thread_safe(
                    self.invoke_fn,
                    self.is_async,
                    self.eval_fn,
                    batch,
                    token_tracker=tracker,
                    label=label,
                    cache=self._cache,
                    proxies=proxies,
                )
            finally:
                for proxy in proxies:
                    proxy._clear_thread_model()

        # Iterate brackets from most aggressive to least (standard Hyperband).
        for s in reversed(range(self._s_max + 1)):
            n_s = int(
                math.ceil(
                    (self._B / self.max_resource) * (self.reduction_factor**s) / (s + 1)
                )
            )
            r_s = int(self.max_resource * (self.reduction_factor ** (-s)))
            r_s = max(1, min(r_s, self.max_resource))

            # Use all valid task indices for every bracket (no subsampling).
            bracket_indices = list(tasks.keys())

            print(
                f"\nBracket s={s}: n_s={n_s} (using all {len(bracket_indices)} configs), "
                f"r_s={r_s}"
            )

            n_i = len(bracket_indices)
            r_i = r_s
            prev_r = 0

            for i in range(s + 1):
                if n_i <= 0:
                    break

                current_indices = bracket_indices[:n_i]
                r_i = int(r_s * (self.reduction_factor**i))
                r_i = max(1, min(r_i, self.max_resource))

                if r_i <= prev_r:
                    break

                print(
                    f"\n  Stage i={i}: n_i={n_i}, r_i={r_i} (prev_r={prev_r}), "
                    f"configs={len(current_indices)}"
                )

                batch = dataset_list[prev_r:r_i]
                if not batch:
                    print("    No additional samples available for this stage.")
                    break

                # Evaluate active combinations in parallel on current batch slice.
                future_to_idx: Dict[Any, int] = {}
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(current_indices))
                ) as executor:
                    for idx in current_indices:
                        combo_name, combo, invoke_fn, is_async_flag, tracker = tasks[
                            idx
                        ]
                        if use_thread_local:
                            future = executor.submit(
                                _eval_batch_thread_local, combo, batch, combo_name
                            )
                        else:
                            future = executor.submit(
                                self._evaluate_thread_safe,
                                invoke_fn,
                                is_async_flag,
                                self.eval_fn,
                                batch,
                                tracker,
                                combo_name,
                                cache=self._cache,
                                proxies=proxies,
                            )
                        future_to_idx[future] = idx

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        combo_name = tasks[idx][0]
                        try:
                            scores, latencies, tokens = future.result()
                            combo_scores[idx].extend(scores)
                            combo_latencies[idx].extend(latencies)
                            self._merge_tokens(combo_tokens[idx], tokens)
                            mu = (
                                sum(combo_scores[idx]) / len(combo_scores[idx])
                                if combo_scores[idx]
                                else 0.0
                            )
                            print(
                                f"    {combo_name}: μ={mu:.3f} (n={len(combo_scores[idx])})"
                            )
                        except Exception as e:
                            logger.warning(
                                "[%s] batch evaluation error: %s", combo_name, e
                            )

                prev_r = r_i

                if i == s:
                    break

                # Rank current configurations by mean score and keep top fraction.
                means: List[Tuple[int, float]] = []
                for idx in current_indices:
                    scores = combo_scores[idx]
                    mu = sum(scores) / len(scores) if scores else 0.0
                    means.append((idx, mu))

                means.sort(key=lambda x: x[1], reverse=True)
                n_i_next = int(math.floor(n_i / self.reduction_factor))
                n_i_next = max(1, n_i_next)

                new_indices = [idx for idx, _ in means[:n_i_next]]
                eliminated = set(current_indices) - set(new_indices)

                if eliminated:
                    for idx in sorted(eliminated):
                        combo_name = tasks[idx][0]
                        print(f"    Eliminated: {combo_name}")
                else:
                    print("    No eliminations in this stage.")

                bracket_indices = new_indices
                n_i = len(bracket_indices)

        # Build results.
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combinations):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            scores = combo_scores.get(idx, [])
            latencies = combo_latencies.get(idx, [])
            if scores:
                accuracy = sum(scores) / len(scores)
                avg_latency = sum(latencies) / len(latencies)
            else:
                accuracy, avg_latency = 0.0, 0.0
            in_tokens, out_tokens = self._split_tokens(combo_tokens.get(idx, {}))
            all_results.append(
                self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=avg_latency,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    attribute="combination",
                    is_best=False,
                )
            )

        # Mark best and restore proxies.
        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
            for idx, combo in enumerate(all_combinations):
                if " + ".join(self._get_model_name(m) for m in combo) == best_name:
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)
                    break
        else:
            print("\n  No combinations succeeded.")

        results = SelectionResults(results=all_results)
        print(results)
        return results

    # ------------------------------------------------------------------
    # Batch evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_tokens(
        accumulated: Dict[str, Tuple[int, int]],
        new_tokens: Dict[str, Tuple[int, int]],
    ) -> None:
        """Merge new_tokens into accumulated token dict in-place."""
        for model, (inp, out) in new_tokens.items():
            if model in accumulated:
                prev_inp, prev_out = accumulated[model]
                accumulated[model] = (prev_inp + inp, prev_out + out)
            else:
                accumulated[model] = (inp, out)
