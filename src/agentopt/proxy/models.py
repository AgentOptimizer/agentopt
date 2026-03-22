"""Data models for agentopt.proxy."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CallRecord:
    """Record of a single intercepted LLM API call."""

    # Attribution (from ContextVars)
    data_id: Optional[str]
    combo_id: Optional[str]
    agent_id: Optional[str]

    # LLM call metrics
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    # Full fidelity
    request_url: str
    request_body: Dict[str, Any] = field(default_factory=dict)
    response_body: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    cached: bool = False
