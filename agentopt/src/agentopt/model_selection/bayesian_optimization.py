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


def _require_botorch() -> None:
    """Raise if botorch/torch are not installed."""
    try:
        import torch  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Bayesian optimization requires optional dependencies: "
            """Install with `pip install "agentopt[bayesian]"`"""
        ) from e


class BayesianOptimizationModelSelector(BaseModelSelector):
    """Select models via Bayesian optimization."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        n_iterations: Optional[int] = None,
        n_initial_random: Optional[int] = None,
        batch_size: int = 1,
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
        _require_botorch()
        self.n_iterations = n_iterations
        self.n_initial_random = n_initial_random
        self.batch_size = max(1, int(batch_size))

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        assert (
            not parallel
        ), "BayesianOptimizationModelSelector does not support parallel evaluation."

        import torch
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

        if self.n_initial_random is None:
            n_initial_random = min(2 * (n_nodes + 1), total_combos)
        else:
            n_initial_random = self.n_initial_random

        if self.n_iterations is None:
            n_iterations = max(0, int(0.2 * total_combos))
        else:
            n_iterations = max(0, self.n_iterations)

        evaluated: Set[Tuple[int, ...]] = set()
        X_list: List[List[int]] = []
        Y_list: List[float] = []
        all_results: List[ModelResult] = []

        def index_combo_to_dict(combo: Tuple[int, ...]) -> Dict[str, ModelCandidate]:
            return {node_names[i]: candidate_lists[i][combo[i]] for i in range(n_nodes)}

        def evaluate_combo(
            combo: Tuple[int, ...]
        ) -> Tuple[float, float, Dict[str, int], Dict[str, int], List[DatapointResult]]:
            combo_dict = index_combo_to_dict(combo)
            combo_name = self._combo_name(combo_dict)
            scores, latencies, dp_ids = asyncio.run(
                self._evaluate_combo_async(
                    combo_dict,
                    self.dataset,
                    label=combo_name,
                    max_concurrent=max_concurrent,
                )
            )
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            accuracy, _ = self._compute_stats(scores)
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            dp_results = self._build_datapoint_results(scores, latencies, dp_ids)
            return accuracy, latency, input_tokens, output_tokens, dp_results

        print(f"\n{'='*60}")
        print(
            f"Bayesian optimization: {total_combos} combinations, "
            f"{n_initial_random} random + {n_iterations} BO iterations"
        )
        print(f"{'='*60}\n")

        # 1) Initial random evaluations
        initial_pool = list(all_index_combos)
        random.shuffle(initial_pool)
        for idx in range(min(n_initial_random, len(initial_pool))):
            combo = initial_pool[idx]
            if combo in evaluated:
                continue
            evaluated.add(combo)
            combo_dict = index_combo_to_dict(combo)
            combo_name = self._combo_name(combo_dict)
            try:
                (
                    accuracy,
                    latency,
                    input_tokens,
                    output_tokens,
                    dp_results,
                ) = evaluate_combo(combo)
                X_list.append(list(combo))
                Y_list.append(accuracy)
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
                all_results.append(result)
                print(f"  {result}")
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

        # 2) Bayesian optimization loop
        for it in range(n_iterations):
            if len(X_list) < 2:
                remaining = [c for c in all_index_combos if c not in evaluated]
                if not remaining:
                    break
                combo = random.choice(remaining)
                evaluated.add(combo)
                combo_dict = index_combo_to_dict(combo)
                combo_name = self._combo_name(combo_dict)
                try:
                    (
                        accuracy,
                        latency,
                        input_tokens,
                        output_tokens,
                        dp_results,
                    ) = evaluate_combo(combo)
                    X_list.append(list(combo))
                    Y_list.append(accuracy)
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
                    all_results.append(result)
                    print(f"  {result}")
                except Exception as e:
                    logger.warning("[random] [%s] failed: %s", combo_name, e)
                continue

            # Fit GP
            train_X = torch.tensor(X_list, dtype=torch.float64)
            train_Y = torch.tensor(Y_list, dtype=torch.float64).unsqueeze(-1)
            cat_dims = list(range(n_nodes))

            model = MixedSingleTaskGP(
                train_X=train_X, train_Y=train_Y, cat_dims=cat_dims,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_f = train_Y.max().item()

            unseen = [c for c in all_index_combos if c not in evaluated]
            if not unseen:
                break

            cand_X = torch.tensor([list(c) for c in unseen], dtype=torch.float64,)

            acq = LogExpectedImprovement(model=model, best_f=best_f)
            with torch.no_grad():
                ei = acq(cand_X.unsqueeze(1))

            # Pick a batch of candidates to evaluate this iteration.
            k = min(self.batch_size, len(unseen))
            topk = ei.squeeze(-1).topk(k=k).indices.tolist()
            batch = [unseen[i] for i in topk]

            for j, combo in enumerate(batch, 1):
                evaluated.add(combo)
                combo_dict = index_combo_to_dict(combo)
                combo_name = self._combo_name(combo_dict)
                try:
                    (
                        accuracy,
                        latency,
                        input_tokens,
                        output_tokens,
                        dp_results,
                    ) = evaluate_combo(combo)
                    X_list.append(list(combo))
                    Y_list.append(accuracy)
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
                    all_results.append(result)
                    print(
                        f"  [BO {it+1}/{n_iterations} | {j}/{len(batch)}] {result}"
                    )
                except Exception as e:
                    logger.warning("[BO] [%s] failed: %s", combo_name, e)

        # 3) Determine best
        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
        else:
            logger.warning("No successful evaluations.")

        results = SelectionResults(results=all_results)
        return results
