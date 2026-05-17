"""Concrete routing policies.

v1 ships a single baseline policy, :class:`RandomRouter`.  Additional
policies (length-based, classifier-backed, bandit) land in this module
without disturbing the abstraction in ``base.py``.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from .base import RouteContext, RouteDecision, Router


class RandomRouter(Router):
    """Uniform random pick from a fixed pool of candidate models.

    Useful as a baseline and for randomised A/B comparisons.  Pass
    ``seed`` for reproducible decisions across runs.
    """

    def __init__(self, candidates: Sequence[str], seed: Optional[int] = None,) -> None:
        assert len(candidates) > 0, "RandomRouter requires at least one candidate"
        self._candidates = tuple(candidates)
        self._rng = random.Random(seed)

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        return RouteDecision(model=self._rng.choice(self._candidates))
