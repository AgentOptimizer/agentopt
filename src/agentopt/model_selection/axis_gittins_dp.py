"""Required-completion Gittins boundaries for an exact objective-axis ray.

On an axis, utility is the posterior mean of the active objective. There is
no division by a zero direction component and no terminal uncertainty
penalty. With centered mean ``x = mean - reference - retirement_value``,
the scalar retirement problem is

``V_H(x) = max(0, x)``,
``Q_n(x) = -cost + E[V_(n+1)(x + Normal(0, v_n - v_(n+1)))]``,
``V_n(x) = max(0, Q_n(x))``.

Its index is ``mean - reference - b_n``, where ``Q_n(b_n) = 0``. This
module uses only NumPy and SciPy. Gaussian expectations integrate a convex
piecewise-linear value function analytically, including its zero left tail
and unit-slope right tail. Each state grid includes the exact stopping kink;
the one-step boundary is consequently exact up to root-solving precision.
Earlier boundaries have the usual state-interpolation error, controlled by
``grid_size``. Root brackets expand as needed and never clip the index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _positive_scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    result = float(array)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _integer(value: int, name: str, minimum: int) -> int:
    if not isinstance(value, (int, float, np.integer, np.floating)) or (
        not math.isfinite(float(value)) or int(value) != value or value < minimum
    ):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def scalar_posterior_variance_schedule(
    initial_var: float, obs_noise_var: float, horizon: int
) -> np.ndarray:
    """Gaussian posterior variances at stages ``0..horizon``."""
    initial = _positive_scalar(initial_var, "initial_var")
    noise = _positive_scalar(obs_noise_var, "obs_noise_var")
    horizon = _integer(horizon, "horizon", 0)
    stages = np.arange(horizon + 1, dtype=np.float64)
    return 1.0 / (1.0 / initial + stages / noise)


def _normal_positive_part(mean: np.ndarray, std: float) -> np.ndarray:
    """Return ``E[max(0, Normal(mean, std**2))]`` without tail truncation."""
    from scipy.special import ndtr

    mean = np.asarray(mean, dtype=np.float64)
    if std == 0.0:
        return np.maximum(0.0, mean)
    # Use the negative-mean formula in both tails and add the positive mean
    # afterwards. This avoids subtracting two large positive-tail quantities.
    z = -np.abs(mean) / std
    tail = std * (np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) + z * ndtr(z))
    return np.maximum(mean, 0.0) + np.maximum(tail, 0.0)


def _hinge_expectation(
    points: np.ndarray | float,
    knots: np.ndarray,
    slope_changes: np.ndarray,
    std: float,
) -> np.ndarray:
    """Integrate ``sum_j slope_changes[j] * max(0, x-knots[j])``."""
    centered = np.asarray(points)[..., None] - knots
    return np.sum(_normal_positive_part(centered, std) * slope_changes, axis=-1)


@dataclass(frozen=True)
class AxisGittinsBoundaryTable:
    """Scalar stopping roots for stages ``0..H``; the terminal root is zero."""

    effective_pull_cost: float
    initial_var: float
    obs_noise_var: float
    horizon: int
    grid_size: int
    boundaries: np.ndarray

    def __post_init__(self) -> None:
        for name in ("effective_pull_cost", "initial_var", "obs_noise_var"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        object.__setattr__(self, "horizon", _integer(self.horizon, "horizon", 0))
        object.__setattr__(self, "grid_size", _integer(self.grid_size, "grid_size", 17))
        boundaries = np.asarray(self.boundaries, dtype=np.float64).copy()
        if boundaries.shape != (self.horizon + 1,) or not np.all(np.isfinite(boundaries)):
            raise ValueError("boundaries must be finite and have shape (horizon + 1,)")
        if boundaries[-1] != 0.0:
            raise ValueError("the terminal boundary must be zero")
        boundaries.setflags(write=False)
        object.__setattr__(self, "boundaries", boundaries)

    def boundary(self, stage: int) -> float:
        try:
            stage = _integer(stage, "stage", 0)
        except ValueError as error:
            raise IndexError(f"stage must lie in [0, {self.horizon}]") from error
        if stage > self.horizon:
            raise IndexError(f"stage must lie in [0, {self.horizon}]")
        return float(self.boundaries[stage])

    def index(self, stage: int, mean: float, reference: float = 0.0) -> float:
        """Index in active-objective units; inactive coordinates play no role."""
        if not math.isfinite(mean) or not math.isfinite(reference):
            raise ValueError("mean and reference must be finite")
        return float(mean - reference - self.boundary(stage))


def build_axis_gittins_boundary_table(
    *,
    effective_pull_cost: float,
    initial_var: float,
    obs_noise_var: float,
    horizon: int,
    grid_size: int = 129,
) -> AxisGittinsBoundaryTable:
    """Build a boundary schedule for one active objective and positive cost.

    ``initial_var`` is the posterior variance at the start of this schedule,
    after any prior-fitting warmup. ``horizon`` counts subsequent pulls needed
    to complete the arm. The schedule uses scalar inputs deliberately: neither
    the unused objective's mean nor its variance belongs in an axis index.
    """
    from scipy.optimize import brentq

    cost = _positive_scalar(effective_pull_cost, "effective_pull_cost")
    initial = _positive_scalar(initial_var, "initial_var")
    noise = _positive_scalar(obs_noise_var, "obs_noise_var")
    horizon = _integer(horizon, "horizon", 0)
    grid_size = _integer(grid_size, "grid_size", 17)
    variances = scalar_posterior_variance_schedule(initial, noise, horizon)
    total_std = math.sqrt(max(0.0, float(variances[0] - variances[-1])))
    scale = max(total_std, cost)
    scaled_cost = cost / scale
    transition_stds = np.sqrt(np.maximum(0.0, variances[:-1] - variances[1:])) / scale
    remaining_stds = np.sqrt(np.maximum(0.0, variances - variances[-1])) / scale
    boundaries = np.zeros(horizon + 1, dtype=np.float64)
    # Terminal value max(0,x) is an exact, unbounded linear hinge.
    knots = np.array([0.0])
    changes = np.array([1.0])
    fractions = np.linspace(0.0, 1.0, grid_size) ** 1.5

    for stage in range(horizon - 1, -1, -1):
        std = float(transition_stds[stage])
        remaining_cost = (horizon - stage) * scaled_cost

        def continuation(x: np.ndarray | float) -> np.ndarray:
            return _hinge_expectation(x, knots, changes, std) - scaled_cost

        # Finishing without further retirement decisions guarantees expected
        # payoff x - remaining_cost, so this is a natural upper bracket.
        lower = min(float(knots[0]), 0.0) - 8.0 * max(float(remaining_stds[stage]), std)
        upper = max(remaining_cost, float(knots[-1]), 0.0) + scaled_cost
        span = max(1.0, upper - lower)
        for _ in range(64):
            if float(continuation(lower)) < 0.0:
                break
            lower -= span
            span *= 2.0
        else:
            raise ArithmeticError("could not bracket the axis boundary from below")
        span = max(1.0, upper - lower)
        for _ in range(64):
            if float(continuation(upper)) > 0.0:
                break
            upper += span
            span *= 2.0
        else:
            raise ArithmeticError("could not bracket the axis boundary from above")
        root = brentq(
            continuation,
            lower,
            upper,
            xtol=4.0 * np.finfo(float).eps,
            rtol=4.0 * np.finfo(float).eps,
        )
        boundaries[stage] = root * scale
        if stage == 0:
            continue

        # Retain the root as an exact zero-value knot, then resolve curvature
        # above it. At eight cumulative standard deviations the continuation
        # value has its linear asymptote x - remaining_cost to roundoff.
        right = max(root + 8.0 * float(remaining_stds[stage]), remaining_cost + 8.0 * float(remaining_stds[stage]))
        right = max(right, root + 1e-6)
        points = root + (right - root) * fractions
        values = np.maximum(0.0, continuation(points))
        values[0] = 0.0
        slopes = np.diff(values) / np.diff(points)
        if np.any(np.diff(slopes) < -1e-9) or np.any(slopes < -1e-9) or np.any(slopes > 1.0 + 1e-9):
            raise ArithmeticError("axis continuation lost convexity or unit-slope bounds")
        slopes = np.maximum.accumulate(np.clip(slopes, 0.0, 1.0))
        # Extend by zero below the root and by a slope-one line above the
        # grid; edge-value padding would corrupt positive-tail expectations.
        changes = np.diff(np.r_[0.0, slopes, 1.0])
        knots = points

    return AxisGittinsBoundaryTable(cost, initial, noise, horizon, grid_size, boundaries)


class AxisGittinsBoundaryCache:
    """In-memory cache keyed by all scalar posterior and solver parameters."""

    def __init__(self) -> None:
        self._tables: dict[tuple[float, float, float, int, int], AxisGittinsBoundaryTable] = {}
        self._memory_hits = 0

    def get(
        self,
        *,
        effective_pull_cost: float,
        initial_var: float,
        obs_noise_var: float,
        horizon: int,
        grid_size: int = 129,
    ) -> AxisGittinsBoundaryTable:
        key = (
            _positive_scalar(effective_pull_cost, "effective_pull_cost"),
            _positive_scalar(initial_var, "initial_var"),
            _positive_scalar(obs_noise_var, "obs_noise_var"),
            _integer(horizon, "horizon", 0),
            _integer(grid_size, "grid_size", 17),
        )
        if key not in self._tables:
            self._tables[key] = build_axis_gittins_boundary_table(
                effective_pull_cost=key[0],
                initial_var=key[1],
                obs_noise_var=key[2],
                horizon=key[3],
                grid_size=key[4],
            )
        else:
            self._memory_hits += 1
        return self._tables[key]

    def stats_snapshot(self) -> dict[str, int]:
        return {"builds": len(self._tables), "memory_hits": self._memory_hits}

    def __len__(self) -> int:
        return len(self._tables)


__all__ = [
    "AxisGittinsBoundaryCache",
    "AxisGittinsBoundaryTable",
    "build_axis_gittins_boundary_table",
    "scalar_posterior_variance_schedule",
]
