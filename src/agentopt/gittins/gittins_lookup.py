"""Precompute Gittins root lookup tables for Gaussian random-walk indices.

This module separates the expensive dynamic program (DP) from the online policy loop.
Given transition standard deviations and costs, the DP produces stage-wise Q-functions
Q_t(s). The Gittins "root" r_t is defined by Q_t(r_t) = 0, and the index is
Γ_t(s_t) = s_t - r_t.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar, UInt

from q_estimation import tabular_q_estimate


def _compute_roots_for_random_walk(
    *,
    transition_stds: Float[Array, " n_transitions"],
    transition_costs: Float[Array, " n_transitions"],
    n_points: int,
) -> tuple[Float[Array, " n_points"], Float[Array, " n_transitions+1"]]:
    """Return `(s_grid, roots)` for all stages 0..T.

    Notes:
    - `tabular_q_estimate` returns `Q_future` as the carry *before* each transition update
      in the backward scan, ordered as stages 1..T. We append the stage-0 carry (`Q_current`)
      to form a stage-major table `[Q_0; Q_1; ...; Q_T]`.
    - We then find each root by `searchsorted(Q_t, 0.0)` on the shared grid.
    """
    Q0, Q_future, s_grid = tabular_q_estimate(
        transition_stds=transition_stds,
        transition_costs=transition_costs,
        current_stage_index=jnp.uint32(0),
        n_points=n_points,
        method="m_diff",
        k=5.0,
    )
    # Q_future has length n_transitions and corresponds to stages 1..T (terminal is stage T).
    # Stack to get stage-major Q table with shape (T+1, n_points).
    Q_table = jnp.concatenate([Q0[jnp.newaxis, :], Q_future], axis=0)

    def root_from_Q(Q_t: Float[Array, " n_points"]) -> Float[Scalar, ""]:
        idx = jnp.searchsorted(Q_t, 0.0)
        return s_grid[jnp.clip(idx, 0, s_grid.shape[0] - 1)]

    roots = jax.vmap(root_from_Q)(Q_table)
    return s_grid, roots


def compute_roots_lookup_table(
    *,
    transition_stds: Float[Array, " n_transitions"],
    costs_per_arm: Float[Array, " n_arms"] | Float[Scalar, ""],
    n_points: int,
) -> Float[Array, " n_arms n_transitions+1"]:
    """Compute per-arm root lookup tables for stages 0..T.

    `costs_per_arm` can be either:
    - a scalar: treated as a single shared cost (returns shape (1, T+1))
    - an `(n_arms,)` vector: one cost per arm (returns shape (n_arms, T+1))
    """
    n_transitions = int(transition_stds.shape[0])

    n_points = int(n_points)
    costs_per_arm_arr = jnp.asarray(costs_per_arm, dtype=jnp.float32).reshape(-1)

    # `n_points` must be static for `jnp.linspace` inside the DP grid construction.
    _roots_for_random_walk_jit = jax.jit(
        _compute_roots_for_random_walk, static_argnames=("n_points",)
    )

    def roots_for_one(cost: Float[Scalar, ""]) -> Float[Array, " n_transitions+1"]:
        transition_costs = jnp.full((n_transitions,), cost, dtype=jnp.float32)
        _, roots = _roots_for_random_walk_jit(
            transition_stds=transition_stds,
            transition_costs=transition_costs,
            n_points=n_points,
        )
        return roots

    return jax.vmap(roots_for_one)(costs_per_arm_arr)

