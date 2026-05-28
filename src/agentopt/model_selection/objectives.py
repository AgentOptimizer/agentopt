"""Multi-objective configuration helpers (weighted scalar vs Pareto exploration)."""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence, Tuple

ObjectiveMode = Literal["weighted", "pareto"]

# Fixed tradeoff directions on the (error, latency, cost) simplex — not user-facing.
CHEBYSHEV_WEIGHTS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
    (0.25, 0.25, 0.5),
    (0.25, 0.5, 0.25),
)


def validate_objective_config(
    objective_mode: Optional[str],
    lambda_cost: float,
    lambda_latency: float,
) -> ObjectiveMode:
    """Validate ``objective_mode`` and ``lambda_*``; return normalized mode."""
    if objective_mode is None:
        raise ValueError(
            "objective_mode is required: use 'weighted' (with lambda_cost and/or "
            "lambda_latency > 0) or 'pareto' (omit lambdas; returns a Pareto frontier)."
        )
    mode = str(objective_mode).strip().lower()
    if mode not in ("weighted", "pareto"):
        raise ValueError(
            f"objective_mode must be 'weighted' or 'pareto', got {objective_mode!r}."
        )
    lc = float(lambda_cost)
    ll = float(lambda_latency)
    if lc < 0 or ll < 0:
        raise ValueError("lambda_cost and lambda_latency must be non-negative")
    if mode == "weighted":
        if lc <= 0.0 and ll <= 0.0:
            raise ValueError(
                "objective_mode='weighted' requires lambda_cost > 0 and/or "
                "lambda_latency > 0."
            )
        return "weighted"
    if lc > 0.0 or ll > 0.0:
        raise ValueError(
            "objective_mode='pareto' does not accept lambda_cost or "
            "lambda_latency; use objective_mode='weighted' instead."
        )
    return "pareto"


def score_to_error(score: float) -> float:
    """Map eval score (higher is better, typically in [0, 1]) to error (lower is better)."""
    s = float(score)
    if not math.isfinite(s):
        return 1.0
    s = max(0.0, min(1.0, s))
    return 1.0 - s


def chebyshev_scalar(
    norm_error: float,
    norm_latency: float,
    norm_cost: float,
    weights: Tuple[float, float, float],
    *,
    use_cost: bool,
) -> float:
    """Weighted Chebyshev achievement scalar (lower is better)."""
    we, wl, wc = weights
    terms = [we * norm_error, wl * norm_latency]
    if use_cost:
        terms.append(wc * norm_cost)
    else:
        # Renormalize error/latency weights when cost is absent.
        s = we + wl
        if s > 0:
            terms = [we / s * norm_error, wl / s * norm_latency]
    return max(terms)


def chebyshev_weight_at(step: int) -> Tuple[float, float, float]:
    """Rotate through the fixed weight grid."""
    grid = CHEBYSHEV_WEIGHTS
    return grid[int(step) % len(grid)]


def pareto_mask_3d(
    errors: Sequence[float],
    latencies: Sequence[float],
    costs: Sequence[Optional[float]],
) -> List[bool]:
    """Nondominated mask for minimize (error, latency, cost).

    When cost is ``None`` for a point, only error and latency are used for
    dominance. Points without price are not compared on cost.
    """
    n = len(errors)
    mask = [True] * n
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            ci, cj = costs[i], costs[j]
            use_cost = ci is not None and cj is not None
            e_ok = errors[j] <= errors[i]
            l_ok = latencies[j] <= latencies[i]
            e_strict = errors[j] < errors[i]
            l_strict = latencies[j] < latencies[i]
            if use_cost:
                c_ok = cj <= ci
                c_strict = cj < ci
                if e_ok and l_ok and c_ok and (e_strict or l_strict or c_strict):
                    mask[i] = False
                    break
            elif e_ok and l_ok and (e_strict or l_strict):
                mask[i] = False
                break
    return mask
