"""
Matrix UCB exploration for model selection (banditeval-style).

Each **combination** (model combo) × **datapoint** (dataset item / question) cell is a
possible evaluation. These selectors pick *which datapoint index* to run next via UCB
over the partially observed matrix—unlike arm-elimination / LUCB-style selectors here,
which advance through the dataset in order.

Plain UCB follows :func:`upper_confidence_bound_exploration` from banditeval; the
low-rank factored variant follows
:func:`upper_confidence_bound_exploration_low_rank_factorization`.

``select_best(..., max_concurrent=k)`` caps how many **(combination, datapoint)** cells
are chosen per UCB step and how many of those evaluations run at once. The
``parallel`` flag on ``select_best`` is **ignored** for these selectors (unlike others,
where it toggles multi-combo parallelism and ``max_concurrent`` splits combo vs
datapoint slots).

``observation_budget_fraction`` (default ``1.0``) limits how much of the grid is
observed; below ``1.0`` the run stops once that fraction (ceiling) of cells is filled.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


def _resolve_observation_budget_fraction(
    observation_budget_fraction: float,
    sample_fraction: Optional[float],
) -> float:
    """Match RandomSearch/Bayesian: ``sample_fraction`` overrides ``observation_budget_fraction``."""
    if sample_fraction is not None:
        s = float(sample_fraction)
        if not 0 < s <= 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")
        return min(1.0, s)
    o = float(observation_budget_fraction)
    if o <= 0:
        raise ValueError("observation_budget_fraction must be positive")
    return min(1.0, o)


def _resolve_warmup_percentage(
    warmup_percentage: float,
    warmup_fraction: Optional[float],
) -> float:
    """Optional alias ``warmup_fraction`` for ``warmup_percentage`` (matrix UCB-LRF)."""
    if warmup_fraction is not None:
        w = float(warmup_fraction)
        if not 0 < w < 1:
            raise ValueError("warmup_fraction must be in the range (0, 1).")
        return w
    return float(warmup_percentage)


def _target_observation_count(
    n_combos: int, n_datapoints: int, observation_budget_fraction: float,
) -> int:
    """Stop after this many observed cells (ceiling of fraction × grid size; full grid if ≥ 1)."""
    total = n_combos * n_datapoints
    if observation_budget_fraction >= 1.0:
        return total
    return max(1, int(math.ceil(observation_budget_fraction * total)))


def _np_filled_count(observed: np.ndarray) -> int:
    return int(np.sum(~np.isnan(observed)))


def _ucb_plain_next_batch(
    observed: np.ndarray,
    a: float,
    max_cells: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Next batch of matrix cells: shape ``(2, k)`` rows [combo_idx, datapoint_idx], or ``None``."""
    n_combos, n_datapoints = observed.shape
    bounds = np.full(n_combos, np.inf, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mus = np.nanmean(observed, axis=1)
    counts = np.sum(~np.isnan(observed), axis=1)
    mask = counts > 0
    bounds[mask] = mus[mask] + np.sqrt(a / counts[mask])
    fully_observed_combo = counts == n_datapoints
    bounds[fully_observed_combo] = -np.inf
    if bool(np.all(fully_observed_combo)):
        return None
    best_combo = int(np.argmax(bounds))
    unobserved_dp = np.where(np.isnan(observed[best_combo]))[0]
    n_unobserved = int(unobserved_dp.size)
    if n_unobserved <= 0:
        return None
    k = min(max(max_cells, 1), n_unobserved)
    pick = rng.permutation(n_unobserved)[:k]
    dps = unobserved_dp[pick]
    combos = np.full(k, best_combo, dtype=np.int64)
    return np.stack([combos, dps.astype(np.int64)])


def _build_selection_results(
    selector: BaseModelSelector,
    all_combos: List[Dict[str, ModelCandidate]],
    cell_data: Dict[Tuple[int, int], Tuple[float, float, str]],
    n_datapoints: int,
) -> SelectionResults:
    all_results: List[ModelResult] = []
    for combo_idx, combo in enumerate(all_combos):
        combo_name = selector._combo_name(combo)
        scores: List[float] = []
        latencies: List[float] = []
        dp_ids: List[str] = []
        for datapoint_idx in range(n_datapoints):
            t = cell_data.get((combo_idx, datapoint_idx))
            if t:
                scores.append(t[0])
                latencies.append(t[1])
                dp_ids.append(t[2])
        if scores:
            accuracy = sum(scores) / len(scores)
            avg_latency = sum(latencies) / len(latencies)
        else:
            accuracy, avg_latency = 0.0, 0.0
        input_tokens, output_tokens = selector._fetch_tokens(combo_name)
        dp_results = (
            selector._build_datapoint_results(scores, latencies, dp_ids)
            if dp_ids
            else []
        )
        all_results.append(
            selector._make_result(
                model_name=combo_name,
                accuracy=accuracy,
                latency_seconds=avg_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attribute="combination",
                is_best=False,
                datapoint_results=dp_results,
            )
        )

    best_info = selector._find_best(all_results)
    if best_info is not None:
        best_name, _ = best_info
        for result in all_results:
            if result.model_name == best_name:
                result.is_best = True
                break
    else:
        print("\n  No combinations succeeded.")

    return SelectionResults(results=all_results)


def _record_cells(
    cell_data: Dict[Tuple[int, int], Tuple[float, float, str]],
    combo_idx: int,
    datapoint_idx: int,
    scores: List[float],
    latencies: List[float],
    dp_ids: List[str],
) -> None:
    if not scores:
        cell_data[(combo_idx, datapoint_idx)] = (
            0.0,
            0.0,
            f"missing::{combo_idx}:{datapoint_idx}",
        )
        return
    cell_data[(combo_idx, datapoint_idx)] = (scores[0], latencies[0], dp_ids[0])


class MatrixUCBModelSelector(BaseModelSelector):
    """UCB on the full **combination × datapoint** matrix (row means + exploration bonus).

    Selection always proceeds in batches of matrix cells; only
    ``select_best(..., max_concurrent=...)`` matters (``parallel`` is ignored).

    ``observation_budget_fraction`` or, equivalently, ``sample_fraction`` (same meaning
    as in :class:`RandomSearchModelSelector` / Bayesian: fraction of the search budget —
    here, **fraction of matrix cells** to observe) caps evaluations. ``1.0`` fills the
    full grid; ``0.1`` stops after about 10% of cells. If both are passed, ``sample_fraction``
    wins.
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        a: float = 1.0,
        observation_budget_fraction: float = 1.0,
        sample_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
    ) -> None:
        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
        )
        self.a = float(a)
        self.observation_budget_fraction = _resolve_observation_budget_fraction(
            observation_budget_fraction, sample_fraction,
        )
        self._rng = np.random.default_rng(seed)

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        """Run matrix UCB.

        The ``parallel`` argument is ignored. ``max_concurrent`` limits each
        step's (combination, datapoint) batch size and how many evaluations run
        at once.
        """
        return super().select_best(parallel=False, max_concurrent=max_concurrent)

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        return asyncio.run(self._select_async(max_concurrent))

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_datapoints = len(dataset_list)
        n_combos = len(all_combos)
        observed = np.full((n_combos, n_datapoints), np.nan, dtype=np.float64)
        cell_data: Dict[Tuple[int, int], Tuple[float, float, str]] = {}
        mc = max(max_concurrent, 1)
        target_n = _target_observation_count(
            n_combos, n_datapoints, self.observation_budget_fraction,
        )
        total_cells = n_combos * n_datapoints

        print(f"\n{'='*60}")
        print(
            f"Matrix UCB (async): {n_combos} combinations × {n_datapoints} datapoints, "
            f"a={self.a}, max {mc} concurrent"
            + (
                f", observe up to {target_n}/{total_cells} cells "
                f"({self.observation_budget_fraction:.0%} budget)"
                if target_n < total_cells
                else ""
            )
        )
        print(f"{'='*60}\n")

        sem = asyncio.Semaphore(mc)

        async def _one(
            combo_i: int, dp_i: int,
        ) -> Tuple[int, int, List[float], List[float], List[str]]:
            async with sem:
                combo = all_combos[combo_i]
                name = self._combo_name(combo)
                return (
                    combo_i,
                    dp_i,
                    *await self._evaluate_combo_async(
                        combo,
                        [dataset_list[dp_i]],
                        label=name,
                        max_concurrent=1,
                        dp_offset=dp_i,
                    ),
                )

        step = 0
        while True:
            filled = _np_filled_count(observed)
            if filled >= target_n:
                break
            mc_step = min(mc, target_n - filled)
            if mc_step <= 0:
                break
            batch = _ucb_plain_next_batch(observed, self.a, mc_step, self._rng)
            if batch is None:
                break
            combo_row, dp_row = batch[0], batch[1]
            step += 1
            print(
                f"Step {step}: {len(dp_row)} cells — "
                f"combination {int(combo_row[0])}, datapoint indices {dp_row.tolist()}"
            )
            outs = await asyncio.gather(
                *[_one(int(ci), int(di)) for ci, di in zip(combo_row, dp_row)],
                return_exceptions=True,
            )
            for res in outs:
                if isinstance(res, Exception):
                    logger.warning("Matrix UCB async cell error: %s", res)
                    continue
                combo_i, dp_i, sc, lat, ids = res
                _record_cells(cell_data, combo_i, dp_i, sc, lat, ids)
                if sc:
                    observed[combo_i, dp_i] = sc[0]

        return _build_selection_results(self, all_combos, cell_data, n_datapoints)


class MatrixUCBLRFModelSelector(BaseModelSelector):
    """Matrix UCB with low-rank factorization uncertainty (ensemble ALS), per banditeval.

    Warmup and main-phase cell batches use ``select_best(..., max_concurrent=...)``.
    Cell budget: ``observation_budget_fraction`` or ``sample_fraction`` (see
    :class:`MatrixUCBModelSelector`). Warmup threshold: ``warmup_percentage`` or
    ``warmup_fraction`` — random probes until this **fraction of the full grid** is
    observed, then LRF+UCB (banditeval-style).
    """

    def __init__(
        self,
        agent: Any = None,
        models: Dict[str, List[ModelCandidate]] = None,
        eval_fn: EvalFn = None,
        dataset: Dataset = None,
        rank: int = 1,
        ensemble_size: int = 64,
        warmup_percentage: float = 0.05,
        warmup_fraction: Optional[float] = None,
        regularizer_weight: float = 0.1,
        drop_probability: float = 0.05,
        iterations: int = 10,
        eta: float = 5.0,
        device: str = "cpu",
        observation_budget_fraction: float = 1.0,
        sample_fraction: Optional[float] = None,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        tracker=None,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "MatrixUCBLRFModelSelector requires PyTorch. "
                'Install with: pip install "agentopt-py[ucb_lrf]"'
            ) from err
        try:
            import einops  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "MatrixUCBLRFModelSelector requires einops (for low-rank fitting). "
                'Install with: pip install "agentopt-py[ucb_lrf]"'
            ) from err

        from .matrix_ucb_factorization import Factorization  # local import

        super().__init__(
            agent=agent,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            model_prices=model_prices,
            tracker=tracker,
        )
        self.Factorization = Factorization
        self.rank = int(rank)
        self.ensemble_size = int(ensemble_size)
        self.warmup_percentage = _resolve_warmup_percentage(
            warmup_percentage, warmup_fraction,
        )
        self.regularizer_weight = float(regularizer_weight)
        self.drop_probability = float(drop_probability)
        self.iterations = int(iterations)
        self.eta = float(eta)
        self.device = device
        self.observation_budget_fraction = _resolve_observation_budget_fraction(
            observation_budget_fraction, sample_fraction,
        )
        self._seed = seed

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential(max_concurrent)

    def _warmup_batch(
        self, observed_t: "torch.Tensor", max_cells: int,
    ) -> "torch.Tensor":
        import torch

        combo_idx, datapoint_idx = torch.where(observed_t.isnan())
        n_nan = combo_idx.numel()
        k = min(max(max_cells, 1), n_nan)
        perm = torch.randperm(n_nan, device=observed_t.device)[:k]
        return torch.stack([combo_idx[perm], datapoint_idx[perm]])

    def _lrf_next_batch(
        self, observed_t: "torch.Tensor", max_cells: int,
    ) -> Optional["torch.Tensor"]:
        import torch

        n_combos, n_datapoints = observed_t.shape
        mc = max(max_cells, 1)
        obs_frac = 1.0 - observed_t.isnan().sum().item() / observed_t.numel()
        if obs_frac < self.warmup_percentage:
            return self._warmup_batch(observed_t, mc)

        bounds = torch.full(
            (n_combos,), float("inf"), device=observed_t.device, dtype=observed_t.dtype
        )
        fac = self.Factorization(
            n_combos,
            n_datapoints,
            self.rank,
            self.ensemble_size,
            regularizer_weight=self.regularizer_weight,
            drop_probability=self.drop_probability,
        ).to(observed_t.device, dtype=observed_t.dtype)
        fac.fit(observed_t, iterations=self.iterations)
        matrix_approximation = fac()
        entry_mus = matrix_approximation.mean(0)
        entry_stds = matrix_approximation.std(0)
        entry_stds = torch.nan_to_num(entry_stds, nan=0.0, posinf=0.0, neginf=0.0)
        entry_ucb = entry_mus + self.eta * entry_stds
        observed_mask = ~observed_t.isnan()
        entry_ucb = entry_ucb.clone()
        entry_ucb[observed_mask] = observed_t[observed_mask]
        combo_ucb = entry_ucb.mean(1)
        counts = observed_mask.sum(1)
        mask = counts > 0
        bounds[mask] = combo_ucb[mask]
        fully_observed = counts == n_datapoints
        bounds[fully_observed] = float("-inf")
        if bool(fully_observed.all()):
            return None
        best_combo = int(torch.argmax(bounds).item())
        combo_entry_stds = entry_stds[best_combo].clone()
        combo_entry_stds[observed_mask[best_combo]] = -1.0
        n_unobs = int((~observed_mask[best_combo]).sum().item())
        k = min(mc, max(n_unobs, 0))
        if k <= 0:
            return None
        _, top_dp = torch.topk(combo_entry_stds, k, largest=True)
        return torch.stack(
            [
                torch.full((k,), best_combo, device=observed_t.device),
                top_dp,
            ]
        )

    def _select_sequential(self, max_concurrent: int = 20) -> SelectionResults:
        import torch

        if self.warmup_percentage < 0.05:
            warnings.warn(
                "Low warmup percentage can significantly degrade performance.",
                UserWarning,
                stacklevel=2,
            )
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_datapoints = len(dataset_list)
        n_combos = len(all_combos)
        if self._seed is not None:
            torch.manual_seed(int(self._seed))

        observed_t = torch.full(
            (n_combos, n_datapoints),
            float("nan"),
            device=self.device,
            dtype=torch.float64,
        )
        cell_data: Dict[Tuple[int, int], Tuple[float, float, str]] = {}
        mc = max(max_concurrent, 1)
        target_n = _target_observation_count(
            n_combos, n_datapoints, self.observation_budget_fraction,
        )
        total_cells = n_combos * n_datapoints

        print(f"\n{'='*60}")
        print(
            f"Matrix UCB-LRF (sequential): {n_combos} combinations × "
            f"{n_datapoints} datapoints, max_concurrent={mc} (cells per step), "
            f"rank={self.rank}, eta={self.eta}"
            + (
                f", observe up to {target_n}/{total_cells} cells "
                f"({self.observation_budget_fraction:.0%} budget)"
                if target_n < total_cells
                else ""
            )
        )
        print(f"{'='*60}\n")

        step = 0
        while True:
            filled = int((~observed_t.isnan()).sum().item())
            if filled >= target_n:
                break
            mc_step = min(mc, target_n - filled)
            if mc_step <= 0:
                break
            batch = self._lrf_next_batch(observed_t, mc_step)
            if batch is None:
                break
            combos, dps = batch[0].tolist(), batch[1].tolist()
            step += 1
            print(
                f"Step {step}: {len(dps)} cells — "
                f"combination {combos[0]}, datapoint indices {dps}"
            )
            for combo_i, dp_i in zip(combos, dps):
                combo = all_combos[int(combo_i)]
                name = self._combo_name(combo)
                sc, lat, ids = self._evaluate_combo(
                    combo,
                    [dataset_list[int(dp_i)]],
                    label=name,
                    dp_offset=int(dp_i),
                )
                _record_cells(cell_data, int(combo_i), int(dp_i), sc, lat, ids)
                if sc:
                    observed_t[int(combo_i), int(dp_i)] = float(sc[0])

        return _build_selection_results(self, all_combos, cell_data, n_datapoints)

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        import torch

        if self.warmup_percentage < 0.05:
            warnings.warn(
                "Low warmup percentage can significantly degrade performance.",
                UserWarning,
                stacklevel=2,
            )
        all_combos = self._all_combos()
        dataset_list = list(self.dataset)
        n_datapoints = len(dataset_list)
        n_combos = len(all_combos)
        if self._seed is not None:
            torch.manual_seed(int(self._seed))

        observed_t = torch.full(
            (n_combos, n_datapoints),
            float("nan"),
            device=self.device,
            dtype=torch.float64,
        )
        cell_data: Dict[Tuple[int, int], Tuple[float, float, str]] = {}
        mc = max(max_concurrent, 1)
        target_n = _target_observation_count(
            n_combos, n_datapoints, self.observation_budget_fraction,
        )
        total_cells = n_combos * n_datapoints

        print(f"\n{'='*60}")
        print(
            f"Matrix UCB-LRF (async): {n_combos} combinations × {n_datapoints} "
            f"datapoints, max {mc} concurrent"
            + (
                f", observe up to {target_n}/{total_cells} cells "
                f"({self.observation_budget_fraction:.0%} budget)"
                if target_n < total_cells
                else ""
            )
        )
        print(f"{'='*60}\n")

        sem = asyncio.Semaphore(mc)

        async def _one(
            combo_i: int, dp_i: int,
        ) -> Tuple[int, int, List[float], List[float], List[str]]:
            async with sem:
                combo = all_combos[combo_i]
                name = self._combo_name(combo)
                return (
                    combo_i,
                    dp_i,
                    *await self._evaluate_combo_async(
                        combo,
                        [dataset_list[dp_i]],
                        label=name,
                        max_concurrent=1,
                        dp_offset=dp_i,
                    ),
                )

        step = 0
        while True:
            filled = int((~observed_t.isnan()).sum().item())
            if filled >= target_n:
                break
            mc_step = min(mc, target_n - filled)
            if mc_step <= 0:
                break
            batch = self._lrf_next_batch(observed_t, mc_step)
            if batch is None:
                break
            combos, dps = batch[0].tolist(), batch[1].tolist()
            step += 1
            print(
                f"Step {step}: {len(dps)} cells — "
                f"combination {combos[0]}, datapoint indices {dps}"
            )
            outs = await asyncio.gather(
                *[_one(int(ci), int(di)) for ci, di in zip(combos, dps)],
                return_exceptions=True,
            )
            for res in outs:
                if isinstance(res, Exception):
                    logger.warning("Matrix UCB-LRF async cell error: %s", res)
                    continue
                combo_i, dp_i, sc, lat, ids = res
                _record_cells(cell_data, combo_i, dp_i, sc, lat, ids)
                if sc:
                    observed_t[combo_i, dp_i] = float(sc[0])

        return _build_selection_results(self, all_combos, cell_data, n_datapoints)
