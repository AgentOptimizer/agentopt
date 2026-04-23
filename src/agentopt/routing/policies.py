"""Concrete routing policies.

v1 ships a single baseline policy, :class:`RandomRouter`.  Additional
policies (length-based, bandit, classifier-backed) land in this module
without disturbing the abstraction in ``base.py``.
"""

import random
from typing import Optional, Sequence

from .base import RouteContext, RouteDecision, Router


class RandomRouter(Router):
    """Uniform random pick from a fixed pool of candidate models."""

    def __init__(
        self, model_candidates: Sequence[str], seed: Optional[int] = None,
    ) -> None:
        assert len(model_candidates) > 0, "RandomRouter requires at least one model"
        self._models = tuple(model_candidates)
        self._rng = random.Random(seed)

    def route(self, ctx: RouteContext) -> Optional[RouteDecision]:
        return RouteDecision(model=self._rng.choice(self._models))
