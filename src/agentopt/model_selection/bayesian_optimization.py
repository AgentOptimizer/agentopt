"""
Bayesian optimization model selector.

Uses a BoTorch MixedSingleTaskGP with categorical inputs to iteratively
select promising combinations via Expected Improvement on accuracy.
"""

import asyncio
import itertools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, DatapointResult, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class BayesianSelectionResults(SelectionResults):
    """Known spend is a lower bound when a cell has an unmeasured charge."""

    unmeasured_failed_attempts: int = 0


def _require_botorch() -> None:
    """Raise if botorch/torch are not installed."""
    try:
        import torch  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Bayesian optimization requires optional dependencies: "
            """Install with `pip install "agentopt-py[bayesian]"`"""
        ) from e


class BayesianOptimizationModelSelector(BaseModelSelector):
    """Select models via Bayesian optimization.

    ``max_search_cost_usd`` optionally caps observed search spending. Each
    configuration is evaluated in full; a scheduled configuration or parallel
    acquisition batch finishes before the cap is checked again.
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        batch_size: int = 1,
        sample_fraction: float = 0.25,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
        lambda_cost: float = 0.0,
        lambda_latency: float = 0.0,
        max_search_cost_usd: Optional[float] = None,
    ) -> None:
        _require_botorch()
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
            lambda_cost=lambda_cost,
            lambda_latency=lambda_latency,
            max_search_cost_usd=max_search_cost_usd,
        )
        self.batch_size = max(1, int(batch_size))
        if not 0 < float(sample_fraction) <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")
        self.sample_fraction = float(sample_fraction)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _bo_setup(self) -> Tuple:
        """Shared BO setup: imports, combo enumeration, iteration counts."""
        import torch
        import math
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        node_names = self._node_names
        candidate_lists = [self._models[n] for n in node_names]
        n_nodes = len(node_names)
        n_choices = [len(c) for c in candidate_lists]
        all_index_combos = list(itertools.product(*[range(n) for n in n_choices]))
        total_combos = len(all_index_combos)

        # Default budget: evaluate ~sample_fraction of all combinations in total
        # (including the initial random evaluations).
        sample_budget = min(
            total_combos, max(1, int(math.ceil(total_combos * self.sample_fraction))),
        )
        # MixedSingleTaskGP fitting needs at least a couple points; if we can,
        # ensure the budget is >= 2.
        if total_combos >= 2:
            sample_budget = max(2, sample_budget)

        # Always use 2*(n_nodes+1) initial random evaluations (clamped by the
        # total budget and search-space size).
        n_initial_random = min(sample_budget, 2 * (n_nodes + 1))
        bo_samples = max(0, sample_budget - n_initial_random)
        n_iterations = max(0, int(math.ceil(bo_samples / self.batch_size)))

        return (
            torch,
            LogExpectedImprovement,
            fit_gpytorch_mll,
            MixedSingleTaskGP,
            ExactMarginalLogLikelihood,
            node_names,
            candidate_lists,
            n_nodes,
            all_index_combos,
            total_combos,
            n_initial_random,
            n_iterations,
            sample_budget,
        )

    def _bo_index_combo_to_dict(
        self,
        combo: Tuple[int, ...],
        node_names: List[str],
        candidate_lists: List[List[ModelCandidate]],
        n_nodes: int,
    ) -> Dict[str, ModelCandidate]:
        return {node_names[i]: candidate_lists[i][combo[i]] for i in range(n_nodes)}

    def _bo_record_result(
        self,
        combo: Dict[str, ModelCandidate],
        accuracy: float,
        latency: float,
        input_tokens: Dict[str, int],
        output_tokens: Dict[str, int],
        dp_results: List[DatapointResult],
        all_results: List[ModelResult],
    ) -> ModelResult:
        result = self._make_result(
            model_name=self._combo_name(combo), combo=combo,
            accuracy=accuracy,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attribute="combination",
            is_best=False,
            datapoint_results=dp_results,
        )
        all_results.append(result)
        return result

    def _bo_fit_and_acquire(
        self,
        torch_mod: Any,
        LogExpectedImprovement: Any,
        fit_gpytorch_mll: Any,
        MixedSingleTaskGP: Any,
        ExactMarginalLogLikelihood: Any,
        X_list: List[List[int]],
        Y_list: List[float],
        n_nodes: int,
        all_index_combos: List[Tuple[int, ...]],
        evaluated: Set[Tuple[int, ...]],
    ) -> Optional[List[Tuple[int, ...]]]:
        """Fit GP, compute EI, return top-k batch of unseen combos or None."""
        from botorch.models.transforms.outcome import Standardize  # type: ignore[reportMissingImports]

        train_X = torch_mod.tensor(X_list, dtype=torch_mod.float64)
        train_Y = torch_mod.tensor(Y_list, dtype=torch_mod.float64).unsqueeze(-1)
        cat_dims = list(range(n_nodes))

        # Standardize the objective values for more stable GP fitting.
        # For acquisition, BoTorch will handle mapping back to the original scale.
        outcome_transform = Standardize(m=1)
        model = MixedSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            outcome_transform=outcome_transform,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = train_Y.max().item()

        unseen = [c for c in all_index_combos if c not in evaluated]
        if not unseen:
            return None

        cand_X = torch_mod.tensor([list(c) for c in unseen], dtype=torch_mod.float64,)
        acq = LogExpectedImprovement(model=model, best_f=best_f)
        with torch_mod.no_grad():
            ei = acq(cand_X.unsqueeze(1))

        k = min(self.batch_size, len(unseen))
        topk = ei.reshape(len(unseen)).topk(k=k).indices.tolist()
        return [unseen[i] for i in topk]

    def _bo_finalize(self, all_results: List[ModelResult]) -> SelectionResults:
        all_results = [result.model_copy(deep=True) for result in all_results]
        for result in all_results:
            result.is_best = False
        self._finalize_combined_objectives(all_results)
        best_info = self._find_best([r for r in all_results if len(r.datapoint_results) == len(self.dataset)])
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            logger.warning("No successful evaluations.")
        return BayesianSelectionResults(results=all_results, unmeasured_failed_attempts=self._live_unknown_charge_attempts())

    def _unknown_cost_stop(self):
        return self._search_cost_limit is not None and self._live_unknown_charge_attempts() > 0

    def _search_budget_stop(self):
        return "search_cost_unknown" if self._unknown_cost_stop() else super()._search_budget_stop()

    def _selection_cost_so_far(self):
        records = self._tracker.get_records()[self._selection_record_offset:]
        return self._cost_of_selection_records(
            r for r in records if not self._charge_unknown(r) and not getattr(r, "request_not_sent", False)
        )

    def _bo_datapoints(self, scores, latencies, dp_ids):
        # The evaluation helpers retain the original one-based ID even when a
        # preceding question fails. Result indices are zero-based.
        indices = [int(dp_id.rsplit("::dp_", 1)[1]) - 1 for dp_id in dp_ids]
        return self._build_datapoint_results(scores, latencies, dp_ids, datapoint_indices=indices)

    def _bo_target_from_result(self, result: ModelResult) -> float:
        """BO target: combined objective if lambdas set, else accuracy."""
        if not self._has_combined_objective or not result.datapoint_results:
            return result.accuracy
        scores = [dp.score for dp in result.datapoint_results]
        lats = [dp.latency_seconds for dp in result.datapoint_results]
        from ..model_price import compute_price

        costs = [
            compute_price(
                dp.input_tokens, dp.output_tokens, custom_prices=self._custom_prices,
            )
            for dp in result.datapoint_results
        ]
        obj = self._mean_objective(scores, lats, costs)
        return obj if obj is not None else result.accuracy

    def _bo_refresh_targets(
        self, X_list: List[List[int]], Y_list: List[float],
        results_by_index: Dict[Tuple[int, ...], ModelResult],
    ) -> None:
        """Recompute all Y_list entries against the current normalizer.

        Called before each GP fit so the surrogate is trained on objectives
        normalized with all observations to date.
        """
        if not self._has_combined_objective:
            return
        for i, x in enumerate(X_list):
            key = tuple(x)
            result = results_by_index.get(key)
            if result is None:
                continue
            Y_list[i] = self._bo_target_from_result(result)

    def _bo_eval_single(
        self,
        combo: Tuple[int, ...],
        node_names: List[str],
        candidate_lists: List[List[ModelCandidate]],
        n_nodes: int,
        evaluate_fn: Callable,
        X_list: List[List[int]],
        Y_list: List[float],
        all_results: List[ModelResult],
        results_by_index: Dict[Tuple[int, ...], ModelResult],
        label: str,
    ) -> bool:
        """Evaluate a single combo, record results. Returns True on success."""
        combo_dict = self._bo_index_combo_to_dict(
            combo, node_names, candidate_lists, n_nodes
        )
        combo_name = self._combo_name(combo_dict)
        try:
            accuracy, latency, input_tokens, output_tokens, dp_results = evaluate_fn(
                combo
            )
            result = self._bo_record_result(
                combo_dict,
                accuracy,
                latency,
                input_tokens,
                output_tokens,
                dp_results,
                all_results,
            )
            if len(dp_results) != len(self.dataset):
                logger.warning("[%s] incomplete configuration; excluding it from BO targets and recommendations", combo_name)
                return False
            X_list.append(list(combo))
            results_by_index[tuple(combo)] = result
            Y_list.append(self._bo_target_from_result(result))
            print(f"  {label}{result}")
            return True
        except Exception as e:
            logger.warning("[%s] [%s] failed: %s", label.strip(), combo_name, e)
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
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._run_selection_async(max_concurrent))
        return self._run_selection_sequential()

    # ------------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------------

    def _run_selection_sequential(self) -> SelectionResults:
        (
            torch_mod,
            LogExpectedImprovement,
            fit_gpytorch_mll,
            MixedSingleTaskGP,
            ExactMarginalLogLikelihood,
            node_names,
            candidate_lists,
            n_nodes,
            all_index_combos,
            total_combos,
            n_initial_random,
            n_iterations,
            sample_budget,
        ) = self._bo_setup()

        evaluated: Set[Tuple[int, ...]] = set()
        X_list: List[List[int]] = []
        Y_list: List[float] = []
        all_results: List[ModelResult] = []
        results_by_index: Dict[Tuple[int, ...], ModelResult] = {}

        def evaluate_combo(combo: Tuple[int, ...]) -> Tuple:
            combo_dict = self._bo_index_combo_to_dict(
                combo, node_names, candidate_lists, n_nodes
            )
            combo_name = self._combo_name(combo_dict)
            scores, latencies, dp_ids = self._evaluate_agent(
                self.agent(combo_dict), self.dataset, label=combo_name, stop_before_cell=self._unknown_cost_stop
            )
            self._observe_combo(scores, latencies, dp_ids)
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            accuracy, _ = self._compute_stats(scores)
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            dp_results = self._bo_datapoints(scores, latencies, dp_ids)
            return accuracy, latency, input_tokens, output_tokens, dp_results

        print(f"\n{'='*60}")
        print(
            f"Bayesian optimization (sequential): {total_combos} combinations, "
            f"{n_initial_random} random + {n_iterations} BO iterations"
        )
        print(f"{'='*60}\n")

        # 1) Initial random evaluations
        initial_pool = list(all_index_combos)
        random.shuffle(initial_pool)
        for idx in range(min(n_initial_random, len(initial_pool))):
            if self._search_budget_stop():
                break
            if sample_budget is not None and len(evaluated) >= sample_budget:
                break
            combo = initial_pool[idx]
            if combo in evaluated:
                continue
            evaluated.add(combo)
            self._bo_eval_single(
                combo,
                node_names,
                candidate_lists,
                n_nodes,
                evaluate_combo,
                X_list,
                Y_list,
                all_results,
                results_by_index,
                label="",
            )
            self._publish_update(self._bo_finalize(all_results))

        # 2) Bayesian optimization loop
        for it in range(n_iterations):
            if self._search_budget_stop():
                break
            if sample_budget is not None and len(evaluated) >= sample_budget:
                break
            if len(X_list) < 2:
                remaining = [c for c in all_index_combos if c not in evaluated]
                if not remaining:
                    break
                if sample_budget is not None and len(evaluated) >= sample_budget:
                    break
                combo = random.choice(remaining)
                evaluated.add(combo)
                self._bo_eval_single(
                    combo,
                    node_names,
                    candidate_lists,
                    n_nodes,
                    evaluate_combo,
                    X_list,
                    Y_list,
                    all_results,
                    results_by_index,
                    label="",
                )
                self._publish_update(self._bo_finalize(all_results))
                continue

            self._bo_refresh_targets(X_list, Y_list, results_by_index)
            batch = self._bo_fit_and_acquire(
                torch_mod,
                LogExpectedImprovement,
                fit_gpytorch_mll,
                MixedSingleTaskGP,
                ExactMarginalLogLikelihood,
                X_list,
                Y_list,
                n_nodes,
                all_index_combos,
                evaluated,
            )
            if batch is None:
                break

            if sample_budget is not None:
                remaining = sample_budget - len(evaluated)
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            for j, combo in enumerate(batch, 1):
                if self._search_budget_stop():
                    break
                evaluated.add(combo)
                self._bo_eval_single(
                    combo,
                    node_names,
                    candidate_lists,
                    n_nodes,
                    evaluate_combo,
                    X_list,
                    Y_list,
                    all_results,
                    results_by_index,
                    label=f"[BO {it+1}/{n_iterations} | {j}/{len(batch)}] ",
                )
                self._publish_update(self._bo_finalize(all_results))

        result = self._bo_finalize(all_results)
        result.stop_reason = self._search_budget_stop() or (
            "all_arms_completed" if len(evaluated) == total_combos else "question_budget"
        )
        return result

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def _run_selection_async(self, max_concurrent: int = 20,) -> SelectionResults:
        (
            torch_mod,
            LogExpectedImprovement,
            fit_gpytorch_mll,
            MixedSingleTaskGP,
            ExactMarginalLogLikelihood,
            node_names,
            candidate_lists,
            n_nodes,
            all_index_combos,
            total_combos,
            n_initial_random,
            n_iterations,
            sample_budget,
        ) = self._bo_setup()

        evaluated: Set[Tuple[int, ...]] = set()
        X_list: List[List[int]] = []
        Y_list: List[float] = []
        all_results: List[ModelResult] = []
        results_by_index: Dict[Tuple[int, ...], ModelResult] = {}

        async def evaluate_combo(
            combo: Tuple[int, ...], dp_concurrent: int = max_concurrent,
        ) -> Tuple:
            combo_dict = self._bo_index_combo_to_dict(
                combo, node_names, candidate_lists, n_nodes
            )
            combo_name = self._combo_name(combo_dict)
            scores, latencies, dp_ids = await self._evaluate_agent_async(
                self.agent(combo_dict),
                self.dataset,
                label=combo_name,
                max_concurrent=dp_concurrent,
                stop_before_cell=self._unknown_cost_stop,
            )
            self._observe_combo(scores, latencies, dp_ids)
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            accuracy, _ = self._compute_stats(scores)
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            dp_results = self._bo_datapoints(scores, latencies, dp_ids)
            return accuracy, latency, input_tokens, output_tokens, dp_results

        async def eval_and_record(combo: Tuple[int, ...], label: str,) -> bool:
            combo_dict = self._bo_index_combo_to_dict(
                combo, node_names, candidate_lists, n_nodes
            )
            combo_name = self._combo_name(combo_dict)
            (
                accuracy,
                latency,
                input_tokens,
                output_tokens,
                dp_results,
            ) = await evaluate_combo(combo)
            result = self._bo_record_result(
                combo_dict,
                accuracy,
                latency,
                input_tokens,
                output_tokens,
                dp_results,
                all_results,
            )
            if len(dp_results) != len(self.dataset):
                logger.warning("[%s] incomplete configuration; excluding it from BO targets and recommendations", combo_name)
                return False
            X_list.append(list(combo))
            results_by_index[tuple(combo)] = result
            Y_list.append(self._bo_target_from_result(result))
            print(f"  {label}{result}")
            return True

        print(f"\n{'='*60}")
        print(
            f"Bayesian optimization (parallel): {total_combos} combinations, "
            f"{n_initial_random} random + {n_iterations} BO iterations"
        )
        print(f"{'='*60}\n")

        # 1) Initial random evaluations
        initial_pool = list(all_index_combos)
        random.shuffle(initial_pool)
        for idx in range(min(n_initial_random, len(initial_pool))):
            if self._search_budget_stop():
                break
            if sample_budget is not None and len(evaluated) >= sample_budget:
                break
            combo = initial_pool[idx]
            if combo in evaluated:
                continue
            evaluated.add(combo)
            combo_dict = self._bo_index_combo_to_dict(
                combo, node_names, candidate_lists, n_nodes
            )
            combo_name = self._combo_name(combo_dict)
            try:
                await eval_and_record(combo, label="")
            except Exception as e:
                logger.warning("[init] [%s] failed: %s", combo_name, e)
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
            self._publish_update(self._bo_finalize(all_results))

        # 2) Bayesian optimization loop
        for it in range(n_iterations):
            if self._search_budget_stop():
                break
            if sample_budget is not None and len(evaluated) >= sample_budget:
                break
            if len(X_list) < 2:
                remaining = [c for c in all_index_combos if c not in evaluated]
                if not remaining:
                    break
                if sample_budget is not None and len(evaluated) >= sample_budget:
                    break
                combo = random.choice(remaining)
                evaluated.add(combo)
                combo_dict = self._bo_index_combo_to_dict(
                    combo, node_names, candidate_lists, n_nodes
                )
                combo_name = self._combo_name(combo_dict)
                try:
                    await eval_and_record(combo, label="")
                except Exception as e:
                    logger.warning("[random] [%s] failed: %s", combo_name, e)
                self._publish_update(self._bo_finalize(all_results))
                continue

            self._bo_refresh_targets(X_list, Y_list, results_by_index)
            batch = self._bo_fit_and_acquire(
                torch_mod,
                LogExpectedImprovement,
                fit_gpytorch_mll,
                MixedSingleTaskGP,
                ExactMarginalLogLikelihood,
                X_list,
                Y_list,
                n_nodes,
                all_index_combos,
                evaluated,
            )
            if batch is None:
                break

            if sample_budget is not None:
                remaining = sample_budget - len(evaluated)
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            if self._search_budget_stop():
                break
            for combo in batch:
                evaluated.add(combo)

            dp_batch_size = len(self.dataset)
            n_combo_bo, dp_concurrent_bo = self._compute_concurrency(
                max_concurrent, dp_batch_size
            )
            bo_combo_sem = asyncio.Semaphore(n_combo_bo)

            async def _throttled_eval_combo(c: Tuple[int, ...]) -> Optional[Tuple]:
                async with bo_combo_sem:
                    if self._unknown_cost_stop():
                        return None
                    return await evaluate_combo(c, dp_concurrent=dp_concurrent_bo)

            batch_results = await asyncio.gather(
                *(_throttled_eval_combo(combo) for combo in batch),
                return_exceptions=True,
            )

            for j, (combo, res) in enumerate(zip(batch, batch_results), 1):
                if res is None:
                    continue
                combo_dict = self._bo_index_combo_to_dict(
                    combo, node_names, candidate_lists, n_nodes
                )
                combo_name = self._combo_name(combo_dict)
                if isinstance(res, Exception):
                    logger.warning("[BO] [%s] failed: %s", combo_name, res)
                    continue
                accuracy, latency, input_tokens, output_tokens, dp_results = res
                result = self._bo_record_result(
                    combo_dict,
                    accuracy,
                    latency,
                    input_tokens,
                    output_tokens,
                    dp_results,
                    all_results,
                )
                if len(dp_results) != len(self.dataset):
                    logger.warning("[%s] incomplete configuration; excluding it from BO targets and recommendations", combo_name)
                    continue
                X_list.append(list(combo))
                results_by_index[tuple(combo)] = result
                Y_list.append(self._bo_target_from_result(result))
                print(f"  [BO {it+1}/{n_iterations} | {j}/{len(batch)}] {result}")
            self._publish_update(self._bo_finalize(all_results))

        result = self._bo_finalize(all_results)
        result.stop_reason = self._search_budget_stop() or (
            "all_arms_completed" if len(evaluated) == total_combos else "question_budget"
        )
        return result
