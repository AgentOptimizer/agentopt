"""
Bayesian optimization model selector.

Uses a BoTorch `MixedSingleTaskGP` with categorical inputs (one dimension per
proxy model choice) to iteratively select promising combinations via Expected
Improvement (EI) on accuracy.
"""

import itertools
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults


logger = logging.getLogger(__name__)


def _require_botorch() -> None:
    """Raise if botorch/torch are not installed."""
    try:
        import torch  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Bayesian optimization requires optional dependencies: "
            "Install with `uv pip install -e '.[bayesian]'` or "
            "`pip install agentopt[bayesian]`"
        ) from e


class BayesianOptimizationModelSelector(BaseModelSelector):
    """Select models via Bayesian optimization.

    This selector treats each proxy's model choice as a categorical variable and
    fits a BoTorch `MixedSingleTaskGP` over the discrete combination space. It
    first evaluates a handful of random combinations, then alternates between
    refitting the GP and selecting the next combination according to Expected
    Improvement (EI) on accuracy.

    Parameters
    ----------
    n_iterations:
        Number of Bayesian optimization steps performed after the initial
        random evaluations.
    n_initial_random:
        Number of randomly chosen combinations evaluated before fitting the GP
        for the first time.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Any] = None,
        n_iterations: Optional[int] = None,
        n_initial_random: Optional[int] = None,
    ) -> None:
        """Initialize the Bayesian optimization model selector."""
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
        )
        _require_botorch()
        # If left as ``None``, effective defaults are chosen inside ``select_best``
        # based on the number of combinations.
        self.n_iterations = n_iterations
        self.n_initial_random = n_initial_random

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        """Run Bayesian optimization and return results.

        Parameters
        ----------
        parallel:
            Accepted for API compatibility with other selectors. Currently
            ignored — Bayesian optimization always runs sequentially.
        max_workers:
            Unused placeholder for future parallel support.

        Returns
        -------
        SelectionResults
            Container with all evaluated combinations and the best one flagged
            via ``is_best=True``.
        """
        import torch
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        n_proxies = len(proxies)
        n_choices = [len(c) for c in candidate_lists]
        all_combinations = list(itertools.product(*[range(n) for n in n_choices]))
        total_combos = len(all_combinations)

        # Choose effective defaults if the user did not override them.
        if self.n_initial_random is None:
            # Common BO heuristic: ~2 * (dim + 1) random points.
            n_initial_random = min(2 * (n_proxies + 1), total_combos)
        else:
            n_initial_random = self.n_initial_random

        if self.n_iterations is None:
            # By default, evaluate ~20% of the total combination space via BO.
            n_iterations = int(0.2 * total_combos)
        else:
            n_iterations = self.n_iterations

        n_iterations = max(0, n_iterations)

        # Index -> (combo tuple), set of already evaluated (as tuple)
        evaluated: Set[Tuple[int, ...]] = set()
        X_list: List[List[int]] = []
        Y_list: List[float] = []
        all_results: List[ModelResult] = []

        def combo_to_models(combo: Tuple[int, ...]) -> Tuple[Any, ...]:
            return tuple(candidate_lists[i][combo[i]] for i in range(n_proxies))

        def set_proxies_from_combo(combo: Tuple[int, ...]) -> None:
            for proxy, model_obj in zip(proxies, combo_to_models(combo)):
                proxy.set_model(model_obj)

        def evaluate_combo(combo: Tuple[int, ...]) -> Tuple[float, float]:
            set_proxies_from_combo(combo)
            return self._evaluate(self.dataset)

        logger.info(
            "Bayesian optimization: %d combinations, %d random + %d BO iterations",
            total_combos,
            n_initial_random,
            n_iterations,
        )

        # 1) Initial random evaluations
        initial_pool = list(all_combinations)
        random.shuffle(initial_pool)
        for idx in range(min(n_initial_random, len(initial_pool))):
            combo = initial_pool[idx]
            if combo in evaluated:
                continue
            evaluated.add(combo)
            combo_name = " + ".join(
                self._get_model_name(m) for m in combo_to_models(combo)
            )
            try:
                accuracy, latency = evaluate_combo(combo)
                logger.info(
                    "[init] [%s] Accuracy: %.3f, Latency: %.3fs",
                    combo_name,
                    accuracy,
                    latency,
                )
                X_list.append(list(combo))
                Y_list.append(accuracy)
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
                logger.warning("[init] [%s] failed: %s", combo_name, e)
                all_results.append(
                    ModelResult(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        attribute="combination",
                        is_best=False,
                    )
                )

        # 2) Bayesian optimization loop
        for it in range(n_iterations):
            if len(X_list) < 2:
                # Need at least 2 points to fit GP; add more random.
                remaining = [c for c in all_combinations if c not in evaluated]
                if not remaining:
                    break
                combo = random.choice(remaining)
                evaluated.add(combo)
                combo_name = " + ".join(
                    self._get_model_name(m) for m in combo_to_models(combo)
                )
                try:
                    accuracy, latency = evaluate_combo(combo)
                    logger.info(
                        "[random] [%s] Accuracy: %.3f, Latency: %.3fs",
                        combo_name,
                        accuracy,
                        latency,
                    )
                    X_list.append(list(combo))
                    Y_list.append(accuracy)
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
                    logger.warning("[random] [%s] failed: %s", combo_name, e)
                continue

            # Fit MixedSingleTaskGP (all dimensions categorical)
            train_X = torch.tensor(X_list, dtype=torch.float64)
            train_Y = torch.tensor(Y_list, dtype=torch.float64).unsqueeze(-1)
            cat_dims = list(range(n_proxies))

            model = MixedSingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Best observed accuracy (for EI)
            best_f = train_Y.max().item()

            # Candidate set: all unseen combinations
            unseen = [c for c in all_combinations if c not in evaluated]
            if not unseen:
                break
            candidates = unseen

            cand_X = torch.tensor(
                [list(c) for c in candidates],
                dtype=torch.float64,
            )

            acq = LogExpectedImprovement(model=model, best_f=best_f)
            with torch.no_grad():
                ei = acq(cand_X.unsqueeze(1))

            best_cand_idx = ei.argmax().item()
            combo = candidates[best_cand_idx]
            evaluated.add(combo)
            combo_name = " + ".join(
                self._get_model_name(m) for m in combo_to_models(combo)
            )
            try:
                accuracy, latency = evaluate_combo(combo)
                logger.info(
                    "[BO %d/%d] [%s] Accuracy: %.3f, Latency: %.3fs",
                    it + 1,
                    n_iterations,
                    combo_name,
                    accuracy,
                    latency,
                )
                X_list.append(list(combo))
                Y_list.append(accuracy)
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
                logger.warning("[BO] [%s] failed: %s", combo_name, e)

        # 3) Determine best and set proxies
        accuracy_tolerance = 1e-9
        best_combination = None
        best_accuracy = float("-inf")
        best_latency = float("inf")

        for result in all_results:
            if result.accuracy > best_accuracy + accuracy_tolerance:
                best_accuracy = result.accuracy
                best_latency = result.latency_seconds
                best_combination = result.model_name
            elif (
                abs(result.accuracy - best_accuracy) <= accuracy_tolerance
                and result.latency_seconds < best_latency
            ):
                best_latency = result.latency_seconds
                best_combination = result.model_name

        if best_combination is not None:
            # Find the combo that corresponds to best_combination and set proxies.
            for result in all_results:
                if result.model_name == best_combination:
                    result.is_best = True
                    break
            for combo in all_combinations:
                name = " + ".join(
                    self._get_model_name(m) for m in combo_to_models(combo)
                )
                if name == best_combination:
                    set_proxies_from_combo(combo)
                    break
            logger.info(
                "Best combination: %s (accuracy=%.3f, latency=%.3fs)",
                best_combination,
                best_accuracy,
                best_latency,
            )
        else:
            logger.warning("No successful evaluations.")

        return SelectionResults(results=all_results)
