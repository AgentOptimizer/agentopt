"""Gittins-index matrix exploration primitives (Gaussian conjugate / normal–normal).

Algorithm logic lives in the modules below; AgentOpt wiring (offline sim / selectors)
imports from here without modifying those implementations.
"""

from .gittins_policy import (
    evaluate_gittins_stopping_rules,
    gittins_index_exploration,
    gittins_post_pull_update,
)

__all__ = [
    "gittins_index_exploration",
    "gittins_post_pull_update",
    "evaluate_gittins_stopping_rules",
]
