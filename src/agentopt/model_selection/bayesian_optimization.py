"""
Bayesian optimization model selection using BoTorch MixedSingleTaskGP.

All variables are categorical (model choice per proxy). The metric optimized is Accuracy.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ..base_models import EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults


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
    """
    Selects the best model for each proxy using Bayesian optimization with
    BoTorch MixedSingleTaskGP. All variables are categorical (model choice per proxy).
    The objective is to maximize Accuracy.
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: List[Tuple[Any, str]],
        agent: Any = None,
        invoke_fn: Optional[callable] = None,
        n_iterations: int = 50,
        n_initial_random: int = 5,
    ) -> None:
        """
        Initialize the Bayesian optimization model selector.

        Args:
            models: Dictionary mapping ModelProxy to list of model candidates.
            eval_fn: Function (expected, actual) -> bool | float (higher is better).
            dataset: List of (input_data, expected_answer) tuples.
            agent: Agent for supported frameworks (Langchain, Langgraph, CrewAI, etc.).
            invoke_fn: Custom invoke callable if not using agent's default.
            n_iterations: Total number of acquisition steps (excluding initial random).
            n_initial_random: Number of random combinations to evaluate before fitting GP.
        """
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
        )
        _require_botorch()
        self.n_iterations = n_iterations
        self.n_initial_random = n_initial_random

    def select_best(self) -> SelectionResults:
        """
        Select the best model for each proxy via Bayesian optimization.

        Uses MixedSingleTaskGP (all dimensions categorical), maximizes Accuracy,
        and suggests next points with Expected Improvement over unseen combinations.

        Returns:
            SelectionResults containing all evaluated combinations and the best.
        """
        import itertools
        import random

        import torch
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        n_proxies = len(proxies)
        n_choices = [len(c) for c in candidate_lists]
        all_combos = list(itertools.product(*[range(n) for n in n_choices]))
        total_combos = len(all_combos)

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

        print(f"\n{'='*60}")
        print(
            f"Bayesian optimization: {total_combos} combinations, "
            f"{self.n_initial_random} random + {self.n_iterations} BO iterations"
        )
        print(f"{'='*60}\n")

        # 1) Initial random evaluations
        initial_pool = list(all_combos)
        random.shuffle(initial_pool)
        for idx in range(min(self.n_initial_random, len(initial_pool))):
            combo = initial_pool[idx]
            if combo in evaluated:
                continue
            evaluated.add(combo)
            combo_name = " + ".join(
                self._get_model_name(m) for m in combo_to_models(combo)
            )
            try:
                accuracy, latency = evaluate_combo(combo)
                print(
                    f"✓ [init] [{combo_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
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
                print(f"✗ [init] [{combo_name}] failed: {e}")
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
        for it in range(self.n_iterations):
            if len(X_list) < 2:
                # Need at least 2 points to fit GP; add more random
                remaining = [c for c in all_combos if c not in evaluated]
                if not remaining:
                    break
                combo = random.choice(remaining)
                evaluated.add(combo)
                combo_name = " + ".join(
                    self._get_model_name(m) for m in combo_to_models(combo)
                )
                try:
                    accuracy, latency = evaluate_combo(combo)
                    print(
                        f"✓ [random] [{combo_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
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
                    print(f"✗ [random] [{combo_name}] failed: {e}")
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

            # Candidate set: all unseen combinations (or sample if huge)
            unseen = [c for c in all_combos if c not in evaluated]
            if not unseen:
                break
            if len(unseen) > 2000:
                # Subsample candidates for EI evaluation
                random.shuffle(unseen)
                candidates = unseen[:2000]
            else:
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
                print(
                    f"✓ [BO {it+1}/{self.n_iterations}] [{combo_name}] "
                    f"Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s"
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
                print(f"✗ [BO] [{combo_name}] failed: {e}")

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
            # Find the combo that corresponds to best_combination and set proxies
            for result in all_results:
                if result.model_name == best_combination:
                    result.is_best = True
                    break
            # Set proxies to best: we need to find the combo from the name or from best accuracy
            for combo in all_combos:
                name = " + ".join(
                    self._get_model_name(m) for m in combo_to_models(combo)
                )
                if name == best_combination:
                    set_proxies_from_combo(combo)
                    break
            print(
                f"\n🏆 Best combination: {best_combination} "
                f"(accuracy: {best_accuracy:.2%}, latency: {best_latency:.2f}s)"
            )
        else:
            print("\n✗ No successful evaluations")

        return SelectionResults(results=all_results)
