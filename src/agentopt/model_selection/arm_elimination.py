"""
Arm Elimination model selector.

Progressively eliminates statistically dominated model combinations,
reducing total API calls compared to brute-force evaluation.

Algorithm:
  1. Evaluate all combinations on an initial batch of samples.
  2. Eliminate any combination i whose upper confidence bound falls below
     another combination j's lower confidence bound:
         μ_i + c·SE_i  <  μ_j − c·SE_j
     where SE_i = σ_i / √n_i (standard error of the mean).
  3. Evaluate survivors on a larger next batch.
  4. Repeat until the dataset is exhausted or one combination remains.
"""

import itertools
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults, _CACHE_SENTINEL

logger = logging.getLogger(__name__)


class ArmEliminationModelSelector(BaseModelSelector):
    """Select models via successive arm elimination.

    Instead of evaluating every combination on the full dataset, this
    selector runs multiple rounds with growing batch sizes.  After each
    round it eliminates combinations whose upper confidence interval lies
    entirely below another combination's lower confidence interval.

    Parameters
    ----------
    n_initial:
        Number of dataset samples to use in the first round.
        Defaults to ``max(1, len(dataset) // 10)`` (targets ≥ 10 rounds).
    growth_factor:
        Multiplier applied to the batch size after each round (default 2.0).
        With 1 000 samples and ``n_initial=100`` the schedule is
        100 → 200 → 400 → 300 (remaining).
    confidence:
        Number of standard errors used for each side of the confidence
        interval (default 1.0, i.e. ±1 SE).  Larger values make
        elimination more conservative (harder to drop a combination).
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
        cache: Optional["EvalCache"] = _CACHE_SENTINEL,
        n_initial: Optional[int] = None,
        growth_factor: float = 2.0,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            agent=agent,
            invoke_fn=invoke_fn,
            cache=cache,
        )
        n = len(self.dataset)
        if n_initial is None:
            self.n_initial = max(1, n // 10)
        else:
            self.n_initial = n_initial
        self.growth_factor = growth_factor
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """Run arm elimination and return results.

        Args:
            parallel: If True, evaluate active combinations concurrently
                within each round.
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
        n_total = len(dataset_list)

        combo_scores: Dict[int, List[float]] = {
            i: [] for i in range(len(all_combinations))
        }
        combo_latencies: Dict[int, List[float]] = {
            i: [] for i in range(len(all_combinations))
        }
        combo_tokens: Dict[int, Dict[str, Tuple[int, int]]] = {
            i: {} for i in range(len(all_combinations))
        }
        active: Set[int] = set(range(len(all_combinations)))

        print(f"\n{'='*60}")
        print(
            f"Arm elimination (sequential): {len(all_combinations)} combinations, "
            f"{n_total} samples"
        )
        print(
            f"  n_initial={self.n_initial}, growth_factor={self.growth_factor}, "
            f"confidence={self.confidence}"
        )
        print(f"{'='*60}")

        if self.n_initial >= n_total:
            print(
                f"  Warning: n_initial ({self.n_initial}) >= dataset size ({n_total}). "
                "Only one round possible — consider a larger dataset or a smaller n_initial."
            )

        offset = 0
        batch_size = self.n_initial
        round_num = 1

        while active and offset < n_total:
            batch_end = min(offset + batch_size, n_total)
            batch = dataset_list[offset:batch_end]

            print(
                f"\nRound {round_num} [samples {offset}–{batch_end}, "
                f"{len(active)} active combination(s)]:"
            )

            for idx in sorted(active):
                combo = all_combinations[idx]
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                for proxy, model_obj in zip(proxies, combo):
                    proxy.set_model(model_obj)

                scores, latencies, tokens = self._evaluate_sequential(
                    batch, label=combo_name
                )
                combo_scores[idx].extend(scores)
                combo_latencies[idx].extend(latencies)
                for model, (in_t, out_t) in tokens.items():
                    prev_in, prev_out = combo_tokens[idx].get(model, (0, 0))
                    combo_tokens[idx][model] = (prev_in + in_t, prev_out + out_t)

                mu, _ = self._compute_stats(combo_scores[idx])
                print(f"  {combo_name}: μ={mu:.3f} (n={len(combo_scores[idx])})")

            # Eliminate dominated combinations.
            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(combo_scores[i], combo_scores[j]):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = " + ".join(
                        self._get_model_name(m) for m in all_combinations[idx]
                    )
                    winner_name = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combinations, combo_scores
                    )
                    print(
                        f"  Eliminated: {combo_name}"
                        + (f" (dominated by {winner_name})" if winner_name else "")
                    )
                active -= newly_eliminated
                print(f"  Survivors: {len(active)} / {len(all_combinations)}")
            else:
                print(
                    f"  No eliminations. Survivors: {len(active)} / {len(all_combinations)}"
                )

            if len(active) <= 1:
                break

            offset = batch_end
            batch_size = max(1, int(batch_size * self.growth_factor))
            round_num += 1

        # Build results for all combinations.
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
                ModelResult(
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
        from ..model_proxy.token_tracking import TokenAccumulator

        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))
        dataset_list = list(self.dataset)
        n_total = len(dataset_list)

        adapter = get_adapter(self.agent) if self.agent is not None else self._adapter

        if max_workers is None:
            max_workers = len(all_combinations)

        print(f"\n{'='*60}")
        print(
            f"Arm elimination (parallel): {len(all_combinations)} combinations, "
            f"{n_total} samples, {max_workers} workers"
        )
        print(
            f"  n_initial={self.n_initial}, growth_factor={self.growth_factor}, "
            f"confidence={self.confidence}"
        )
        print(f"{'='*60}")

        if self.n_initial >= n_total:
            print(
                f"  Warning: n_initial ({self.n_initial}) >= dataset size ({n_total}). "
                "Only one round possible — consider a larger dataset or a smaller n_initial."
            )

        if self.agent is not None:
            # Agent-based path: clone per combination.
            print(f"\n  Cloning agents for {len(all_combinations)} combinations ...")
            tasks: Dict[int, Tuple[str, tuple, Callable, bool]] = {}
            for idx, combo in enumerate(all_combinations):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(
                    f"    clone {idx+1}/{len(all_combinations)}: {combo_name}",
                    flush=True,
                )
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
                    tasks[idx] = (combo_name, combo, invoke_fn, False)
                except Exception as e:
                    logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)

            if not tasks:
                raise RuntimeError("All clone attempts failed.")
        else:
            # invoke_fn path: prepare task entries (thread-local will be set per round).
            tasks = {}
            for idx, combo in enumerate(all_combinations):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                tasks[idx] = (combo_name, combo, self.invoke_fn, self.is_async)

        combo_scores: Dict[int, List[float]] = {i: [] for i in tasks}
        combo_latencies: Dict[int, List[float]] = {i: [] for i in tasks}
        combo_tokens: Dict[int, Dict[str, Tuple[int, int]]] = {i: {} for i in tasks}
        active: Set[int] = set(tasks.keys())

        offset = 0
        batch_size = self.n_initial
        round_num = 1

        use_thread_local = self.agent is None

        def _eval_batch_thread_local(
            combo: tuple,
            batch: List[Tuple[Any, Any]],
            label: str,
        ) -> Tuple[List[float], List[float], Dict[str, Tuple[int, int]]]:
            """Set thread-local models, evaluate batch, then clean up."""
            tracker = TokenAccumulator()
            for proxy, model_spec in zip(proxies, combo):
                model_name = self._get_model_name(model_spec)
                current_model = object.__getattribute__(proxy, "_optmodel")
                fresh_model = build_llm(model_name, current_model)
                if fresh_model is None:
                    raise RuntimeError(
                        f"Cannot build model '{model_name}' for parallel evaluation."
                    )
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

        while active and offset < n_total:
            batch_end = min(offset + batch_size, n_total)
            batch = dataset_list[offset:batch_end]

            print(
                f"\nRound {round_num} [samples {offset}–{batch_end}, "
                f"{len(active)} active combination(s)]:"
            )

            future_to_idx: Dict[Any, int] = {}
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(active))
            ) as executor:
                for idx in active:
                    combo_name, combo, invoke_fn, is_async_flag = tasks[idx]
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
                            None,
                            combo_name,
                            self._cache,
                            proxies,
                        )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    combo_name = tasks[idx][0]
                    try:
                        scores, latencies, tokens = future.result()
                        combo_scores[idx].extend(scores)
                        combo_latencies[idx].extend(latencies)
                        for model, (in_t, out_t) in tokens.items():
                            prev_in, prev_out = combo_tokens[idx].get(model, (0, 0))
                            combo_tokens[idx][model] = (
                                prev_in + in_t,
                                prev_out + out_t,
                            )
                        mu, _ = self._compute_stats(combo_scores[idx])
                        print(
                            f"  {combo_name}: μ={mu:.3f} (n={len(combo_scores[idx])})"
                        )
                    except Exception as e:
                        logger.warning("[%s] batch evaluation error: %s", combo_name, e)

            # Eliminate dominated combinations.
            newly_eliminated: Set[int] = set()
            for i in active:
                for j in active:
                    if i != j and self._is_dominated(combo_scores[i], combo_scores[j]):
                        newly_eliminated.add(i)
                        break

            if newly_eliminated:
                for idx in newly_eliminated:
                    combo_name = tasks[idx][0]
                    winner_name = self._find_dominator_name(
                        idx, active - newly_eliminated, all_combinations, combo_scores
                    )
                    print(
                        f"  Eliminated: {combo_name}"
                        + (f" (dominated by {winner_name})" if winner_name else "")
                    )
                active -= newly_eliminated
                print(f"  Survivors: {len(active)} / {len(all_combinations)}")
            else:
                print(
                    f"  No eliminations. Survivors: {len(active)} / {len(all_combinations)}"
                )

            if len(active) <= 1:
                break

            offset = batch_end
            batch_size = max(1, int(batch_size * self.growth_factor))
            round_num += 1

        # Build results.
        all_results: List[ModelResult] = []
        for idx, combo in enumerate(all_combinations):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            scores = combo_scores.get(idx, [])
            latencies = combo_latencies.get(idx, [])
            accuracy, _ = self._compute_stats(scores)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            in_tokens, out_tokens = self._split_tokens(combo_tokens.get(idx, {}))
            all_results.append(
                ModelResult(
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
    # Statistical helpers
    # ------------------------------------------------------------------

    def _is_dominated(
        self,
        scores_i: List[float],
        scores_j: List[float],
    ) -> bool:
        """Return True if arm i is statistically dominated by arm j.

        Domination condition (upper CI of i < lower CI of j):
            μ_i + c·SE_i  <  μ_j − c·SE_j
        where SE = σ / √n.
        """
        n_i, n_j = len(scores_i), len(scores_j)
        if n_i == 0 or n_j == 0:
            return False
        mu_i, std_i = self._compute_stats(scores_i)
        mu_j, std_j = self._compute_stats(scores_j)
        se_i = std_i / math.sqrt(n_i)
        se_j = std_j / math.sqrt(n_j)
        return mu_i + self.confidence * se_i < mu_j - self.confidence * se_j

    def _find_dominator_name(
        self,
        dominated_idx: int,
        active_remaining: Set[int],
        all_combinations: List[tuple],
        combo_scores: Dict[int, List[float]],
    ) -> Optional[str]:
        """Return the display name of the combination that dominates *dominated_idx*."""
        for j in active_remaining:
            if self._is_dominated(combo_scores[dominated_idx], combo_scores[j]):
                return " + ".join(self._get_model_name(m) for m in all_combinations[j])
        return None
