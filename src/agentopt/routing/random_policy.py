"""Random routing policy — uniform pick from a fixed candidate pool.

One policy per file, mirroring :mod:`agentopt.model_selection`.  New
policies (length-based, classifier-backed, bandit, …) land in sibling
modules.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from .base import RouteContext, Router


class RandomRouter(Router):
    """Uniform random pick from a fixed pool of candidate models.

    Useful as a baseline and for randomised A/B comparisons.  Pass
    ``seed`` for reproducible decisions across runs.
    """

    POLICY_NAME = "random"

    def __init__(self, candidates: Sequence[str], seed: Optional[int] = None,) -> None:
        if not candidates:
            raise ValueError("RandomRouter requires at least one candidate")
        self._candidates = tuple(candidates)
        self._seed = seed
        self._rng = random.Random(seed)

    def route(self, ctx: RouteContext) -> Optional[str]:
        return self._rng.choice(self._candidates)

    def _config_kwargs(self) -> dict:
        cfg = {"candidates": list(self._candidates)}
        if self._seed is not None:
            cfg["seed"] = self._seed
        return cfg
