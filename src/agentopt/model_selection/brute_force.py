"""
Brute-force model selection: evaluates the Cartesian product of all
candidate models across all proxies.
"""

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults, _CACHE_SENTINEL

logger = logging.getLogger(__name__)


class BruteForceModelSelector(BaseModelSelector):
    """
    Selects the best model for an agent by evaluating on a dataset.

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
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        cache: Optional["EvalCache"] = _CACHE_SENTINEL,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
            model_prices=model_prices,
            cache=cache,
        )

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """
        Select the best model combination.

        Args:
            parallel: If True, clone the agent per combination and evaluate
                concurrently with a thread pool. If False (default), swap
                proxies in-place and evaluate sequentially.
            max_workers: Max threads for parallel mode. Defaults to the
                number of combinations.

        Returns:
            SelectionResults containing all model evaluation results.
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
    # Sequential evaluation (proxy swap in-place)
    # ------------------------------------------------------------------

    def _select_sequential(self) -> SelectionResults:
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))

        all_results: List[ModelResult] = []
        best_combination = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9

        print(f"\n{'='*60}")
        print(f"Brute force (sequential): {len(all_combinations)} combinations")
        print(f"{'='*60}\n")

        for idx, combo in enumerate(all_combinations, 1):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            for proxy, model_obj in zip(proxies, combo):
                proxy.set_model(model_obj)

            print(f"  [{idx}/{len(all_combinations)}] Evaluating: {combo_name}")
            for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                print(f"    proxy[{i}] → {self._get_model_name(model_obj)}")
            try:
                scores, latencies, tokens = self._evaluate_sequential(
                    self.dataset, label=combo_name
                )
                accuracy, _ = self._compute_stats(scores)
                latency = sum(latencies) / len(latencies) if latencies else 0.0
                in_tokens, out_tokens = self._split_tokens(tokens)

                result = self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
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
                    self._make_result(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        input_tokens={},
                        output_tokens={},
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
            print("\n  No combinations succeeded")

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
        from ..model_proxy.adapter import get_adapter
        from ..model_proxy.builders import build_llm
        from ..model_proxy.token_tracking import TokenAccumulator

        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))

        if max_workers is None:
            max_workers = len(all_combinations)

        adapter = get_adapter(self.agent) if self.agent is not None else self._adapter

        print(f"\n{'='*60}")
        print(
            f"Brute force (parallel): {len(all_combinations)} combinations, "
            f"{max_workers} workers"
        )
        print(f"{'='*60}\n")

        if self.agent is not None:
            # Agent-based path: clone per combination (serial cloning).
            print(f"  Cloning agents for {len(all_combinations)} combinations ...")
            tasks: List[Tuple[str, tuple, Any, bool, Any]] = []

            for idx, combo in enumerate(all_combinations, 1):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(
                    f"    clone {idx}/{len(all_combinations)}: {combo_name}", flush=True
                )
                for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                    print(f"      proxy[{i}] → {self._get_model_name(model_obj)}")

                try:
                    if adapter is not None:
                        agent_copy = adapter.clone_for_parallel(
                            self.agent, proxies, combo, self._get_model_name
                        )
                    else:
                        import copy

                        agent_copy = copy.deepcopy(self.agent)
                    tracker = (
                        adapter.create_token_tracker(agent_copy)
                        if adapter is not None
                        else None
                    )
                    invoke_fn = (
                        adapter.get_invoke_fn(agent_copy)
                        if adapter is not None
                        else self._make_invoke_fn(
                            agent_copy, self._invoke_method_name, self.is_async
                        )
                    )
                    tasks.append((combo_name, combo, invoke_fn, False, tracker))
                except Exception as e:
                    logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)
                    continue

            if not tasks:
                raise RuntimeError("All clone attempts failed.")

            print(
                f"\n  Evaluating {len(tasks)} combinations across {max_workers} workers ..."
            )
            all_results: List[ModelResult] = []
            future_to_info: Dict[Any, Tuple[str, tuple]] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for combo_name, combo, invoke_fn, is_async_flag, tracker in tasks:
                    future = executor.submit(
                        self._evaluate_thread_safe,
                        invoke_fn,
                        is_async_flag,
                        self.eval_fn,
                        self.dataset,
                        token_tracker=tracker,
                        label=combo_name,
                        cache=self._cache,
                        proxies=proxies,
                    )
                    future_to_info[future] = (combo_name, combo)

                for future in as_completed(future_to_info):
                    combo_name, combo = future_to_info[future]
                    try:
                        scores, latencies, tokens = future.result()
                        accuracy, _ = self._compute_stats(scores)
                        latency = sum(latencies) / len(latencies) if latencies else 0.0
                        in_tokens, out_tokens = self._split_tokens(tokens)
                        result = self._make_result(
                            model_name=combo_name,
                            accuracy=accuracy,
                            latency_seconds=latency,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            attribute="combination",
                            is_best=False,
                        )
                        print(f"  {result}")
                        all_results.append(result)
                    except Exception as e:
                        print(f"  [{combo_name}] failed: {e}")
                        all_results.append(
                            self._make_result(
                                model_name=combo_name,
                                accuracy=0.0,
                                latency_seconds=0.0,
                                input_tokens={},
                                output_tokens={},
                                attribute="combination",
                                is_best=False,
                            )
                        )
        else:
            # invoke_fn path: use thread-local model overrides on proxies.
            print(f"  Preparing {len(all_combinations)} combinations ...")
            for idx, combo in enumerate(all_combinations, 1):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(
                    f"    combo {idx}/{len(all_combinations)}: {combo_name}", flush=True
                )
                for i, (proxy, model_obj) in enumerate(zip(proxies, combo)):
                    print(f"      proxy[{i}] → {self._get_model_name(model_obj)}")

            print(
                f"\n  Evaluating {len(all_combinations)} combinations across "
                f"{max_workers} workers ..."
            )
            all_results = []
            future_to_info = {}

            def _eval_with_thread_local(
                combo: tuple,
                proxies: List[ModelProxy],
                invoke_fn: Any,
                is_async: bool,
                eval_fn: Any,
                dataset: Any,
                get_model_name: Any,
                label: str,
            ) -> Tuple[float, float, Dict]:
                """Set thread-local models, evaluate, then clean up."""
                tracker = TokenAccumulator()
                for proxy, model_spec in zip(proxies, combo):
                    model_name = get_model_name(model_spec)
                    current_model = object.__getattribute__(proxy, "_optmodel")
                    fresh_model = build_llm(model_name, current_model)
                    if fresh_model is None:
                        raise RuntimeError(
                            f"Cannot build model '{model_name}' for parallel evaluation."
                        )
                    proxy._set_thread_model(fresh_model, tracker)
                try:
                    return self._evaluate_thread_safe(
                        invoke_fn,
                        is_async,
                        eval_fn,
                        dataset,
                        token_tracker=tracker,
                        label=label,
                        cache=self._cache,
                        proxies=proxies,
                    )
                finally:
                    for proxy in proxies:
                        proxy._clear_thread_model()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for combo in all_combinations:
                    combo_name = " + ".join(self._get_model_name(m) for m in combo)
                    future = executor.submit(
                        _eval_with_thread_local,
                        combo,
                        proxies,
                        self.invoke_fn,
                        self.is_async,
                        self.eval_fn,
                        self.dataset,
                        self._get_model_name,
                        combo_name,
                    )
                    future_to_info[future] = (combo_name, combo)

                for future in as_completed(future_to_info):
                    combo_name, combo = future_to_info[future]
                    try:
                        scores, latencies, tokens = future.result()
                        accuracy, _ = self._compute_stats(scores)
                        latency = sum(latencies) / len(latencies) if latencies else 0.0
                        in_tokens, out_tokens = self._split_tokens(tokens)
                        result = self._make_result(
                            model_name=combo_name,
                            accuracy=accuracy,
                            latency_seconds=latency,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            attribute="combination",
                            is_best=False,
                        )
                        print(f"  {result}")
                        all_results.append(result)
                    except Exception as e:
                        print(f"  [{combo_name}] failed: {e}")
                        all_results.append(
                            self._make_result(
                                model_name=combo_name,
                                accuracy=0.0,
                                latency_seconds=0.0,
                                input_tokens={},
                                output_tokens={},
                                attribute="combination",
                                is_best=False,
                            )
                        )

        # Determine best and set proxies.
        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for combo in all_combinations:
                if " + ".join(self._get_model_name(m) for m in combo) == best_name:
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)
                    break
            for r in all_results:
                if r.model_name == best_name:
                    r.is_best = True
                    break
        else:
            print("\n  No combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results
