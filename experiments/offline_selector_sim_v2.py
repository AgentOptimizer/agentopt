"""Compatibility types used by the historical lookup-table pickles.

The original builder script is no longer present on this branch, but its
pickles reference ``offline_selector_sim_v2.SampleResult``. Keep this module
small and stable so the old data can be loaded without executing old code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SampleResult:
    score: float
    latency_seconds: float
    input_tokens: Dict[str, int] = field(default_factory=dict)
    output_tokens: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


LookupTable = Dict[str, Dict[int, SampleResult]]
