"""
Brute-force model selection: evaluates the Cartesian product of all
candidate models across all proxies.
"""

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base_models import EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults

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
        dataset: List[Tuple[Any, str]],
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
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
            self._sync_agent_models(proxies, combo)

            print(f"  [{idx}/{len(all_combinations)}] Evaluating: {combo_name}")
            try:
                accuracy, latency = self._evaluate(self.dataset, label=combo_name)

                print(
                    f"  [{combo_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
                )

                all_results.append(
                    ModelResult(
                        model_name=combo_name,
                        accuracy=accuracy,
                        latency_seconds=latency,
                        attribute="combination",
                        is_best=False,
                    )
                )

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
            self._sync_agent_models(proxies, best_combination)
            print(
                f"\n  Best: {best_name} "
                f"(accuracy: {best_accuracy:.2%}, latency: {best_latency:.2f}s)"
            )
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            print("\n  No combinations succeeded")

        return SelectionResults(results=all_results)

    # ------------------------------------------------------------------
    # Parallel evaluation (agent clones + thread pool)
    # ------------------------------------------------------------------

    def _select_parallel(
        self,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))

        if max_workers is None:
            max_workers = len(all_combinations)

        # Detect which agent attribute each proxy corresponds to
        proxy_to_attr = self._detect_proxy_attrs(proxies)

        # Resolve the invoke method name (kickoff / invoke / run)
        invoke_method_name = self._invoke_method_name
        if invoke_method_name is None:
            raise RuntimeError(
                "Parallel mode requires an agent (not invoke_fn). "
                "Pass agent= instead of invoke_fn=."
            )

        print(f"\n{'='*60}")
        print(
            f"Brute force (parallel): {len(all_combinations)} combinations, "
            f"{max_workers} workers"
        )
        print(f"{'='*60}\n")

        # Phase 1: Create agent copies (sequential — cloning must be serial)
        print(f"  Cloning agents for {len(all_combinations)} combinations ...")
        tasks: List[Tuple[str, tuple, Any, bool]] = []
        for idx, combo in enumerate(all_combinations, 1):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            print(f"    clone {idx}/{len(all_combinations)}: {combo_name}", flush=True)

            llm_updates: Dict[str, Any] = {}
            for proxy, model_spec in zip(proxies, combo):
                attr = proxy_to_attr.get(id(proxy))
                if attr is None:
                    continue
                if isinstance(model_spec, str):
                    fresh = self._create_variant(proxy.get_model(), model_spec)
                else:
                    fresh = model_spec
                llm_updates[attr] = fresh

            try:
                agent_copy = self._clone_agent(self.agent, llm_updates)
            except Exception as e:
                logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)
                continue

            # For CrewAI Crews, also update each sub-agent's LLM on the clone.
            if hasattr(agent_copy, "agents"):
                n_proxies = len(proxies)
                n_agents = len(agent_copy.agents)
                if n_proxies == 1:
                    model_spec = combo[0]
                    model_name = (
                        model_spec
                        if isinstance(model_spec, str)
                        else self._get_model_name(model_spec)
                    )
                    for ag in agent_copy.agents:
                        self._set_agent_model(ag, model_name)
                elif n_proxies == n_agents:
                    for ag, model_spec in zip(agent_copy.agents, combo):
                        model_name = (
                            model_spec
                            if isinstance(model_spec, str)
                            else self._get_model_name(model_spec)
                        )
                        self._set_agent_model(ag, model_name)

            invoke_fn = self._make_invoke_fn(
                agent_copy, invoke_method_name, self.is_async
            )
            tasks.append((combo_name, combo, invoke_fn, self.is_async))

        if not tasks:
            raise RuntimeError("All clone attempts failed. Cannot evaluate any models.")

        # Phase 2: Evaluate in parallel
        print(
            f"\n  Evaluating {len(tasks)} combinations across {max_workers} workers ..."
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
                    print(
                        f"  [{combo_name}] Accuracy: {accuracy:.2%}, "
                        f"Latency: {latency:.2f}s"
                    )
                    all_results.append(
                        ModelResult(
                            model_name=combo_name,
                            accuracy=accuracy,
                            latency_seconds=latency,
                            attribute="combination",
                            is_best=False,
                        )
                    )
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

        # Phase 3: Determine best and set proxies
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
                    print(
                        f"\n  Best: {best_name} "
                        f"(accuracy: {r.accuracy:.2%}, latency: {r.latency_seconds:.2f}s)"
                    )
                    break
        else:
            print("\n  No combinations succeeded")

        return SelectionResults(results=all_results)
