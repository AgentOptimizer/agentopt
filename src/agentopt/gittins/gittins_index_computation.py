# Copyright 2025 Theo Brown
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Functions for computing the Gittins Index of Gaussian random walks."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar, UInt

from .q_estimation import tabular_q_estimate


def compute_gittins_for_random_walk(
    current_stage_index: UInt[Scalar, ""],
    current_stage_value: Float[Scalar, ""],
    transition_stds: Float[Array, " n_transitions"],
    transition_costs: Float[Array, " n_transitions"],
    n_points: UInt[Scalar, ""],
) -> Float[Scalar, ""]:
    """
    Finds the Gittins index for the current state of a Gaussian random walk.

     Stage 0                          Stage t                                   Stage T
    |-------|                   |--------------------|                   |--------------------|
    |  x₀   | -----> ... -----> | xₜ ~ N(xₜ₋₁, σₜ₋₁) | -----> ... -----> | x_T ~ N(x_{T-1}, σ_{T-1}) |
    |-------|   c₀        c_{t-1} |--------------------|  cₜ         c_{T-1} |--------------------|
       θ₀                                θₜ                                        θ_T

    Args:
        current_stage_index: Index for the current stage in the following arrays.
        current_stage_value: Value of the walk at the current stage (e.g. posterior mean μ_t).
        transition_stds: Array of standard deviations for Gaussian stage transitions (σ₀, ..., σ_{T-1}), where transition_stds[i] is the std of the transition from stage i to stage i+1.
        transition_costs: Array of future stage transition costs (c₀, ..., c_{T-1}), where transition_costs[i] is the cost for transitioning from stage i to stage i+1.
        n_points: Grid size for tabular Q (see tabular_q_estimate).

    Returns:
        The Gittins index for the current state.
    """
    Q_t, _, s_grid = tabular_q_estimate(
        transition_stds=transition_stds,
        transition_costs=transition_costs,
        current_stage_index=current_stage_index,
        n_points=n_points,
        method="m_diff",
        k=5.0,
    )
    idx = jnp.searchsorted(Q_t, 0.0)
    root = s_grid[jnp.clip(idx, 0, s_grid.shape[0] - 1)]

    return current_stage_value - root
