"""Shared token accumulator and usage extraction for all framework adapters."""

import inspect
import threading
from typing import Any, Tuple


def extract_usage(response: Any) -> Tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a framework LLM response.

    Recognizes:
    - LangChain ``AIMessage.usage_metadata`` (keys: input_tokens, output_tokens)
    - OpenAI Agents SDK ``ModelResponse.usage`` (attrs: input_tokens, output_tokens)
    - Generic ``response.usage`` dict/object with prompt_tokens/completion_tokens

    Returns (0, 0) if no recognized usage format is found.
    """
    if response is None:
        return (0, 0)

    # LangChain AIMessage: usage_metadata is a dict
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict) and meta:
        return (
            meta.get("input_tokens", 0),
            meta.get("output_tokens", 0),
        )

    # OpenAI Agents SDK / generic: usage object with attributes
    usage = getattr(response, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0) or getattr(
            usage, "completion_tokens", 0
        )
        if in_tok or out_tok:
            return (in_tok, out_tok)
        # usage might be a dict
        if isinstance(usage, dict):
            return (
                usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
                usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            )

    return (0, 0)


class TokenAccumulator:
    """Thread-safe, resettable token counter.

    All framework adapters write into this via ``add()``, and consumers
    read + reset via ``reset()``.  Replaces per-adapter accumulator classes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token counts (thread-safe)."""
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def reset(self) -> Tuple[int, int]:
        """Return accumulated counts and reset to zero."""
        with self._lock:
            result = (self.input_tokens, self.output_tokens)
            self.input_tokens = 0
            self.output_tokens = 0
            return result
