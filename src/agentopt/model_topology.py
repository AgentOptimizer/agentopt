"""
Model topology: general-impression rankings of LLM models by quality and speed.

These rankings encode a rough ordering (not benchmark-precise) so that search
algorithms like hill climbing can define "neighbors" in model space.
"""

from typing import List, Optional

# ---------------------------------------------------------------------------
# Quality ranking – ordered from HIGHEST quality to LOWEST quality.
# Index 0 = best quality.  Uses OpenRouter model naming.
# ---------------------------------------------------------------------------
QUALITY_RANKING: List[str] = [
    # OpenAI reasoning
    "openai/o3",
    "openai/o4-mini",
    "openai/o3-mini",
    # OpenAI GPT
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-4.1",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-nano",
    # Anthropic
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-haiku",
    # Google
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash",
    "google/gemini-2.0-flash-lite",
]

# ---------------------------------------------------------------------------
# Speed ranking – ordered from FASTEST to SLOWEST.
# Index 0 = fastest.  Uses OpenRouter model naming.
# ---------------------------------------------------------------------------
SPEED_RANKING: List[str] = [
    # Small / lite models first
    "openai/gpt-4.1-nano",
    "google/gemini-2.0-flash-lite",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4.1-mini",
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/o4-mini",
    "openai/o3-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-5.1",
    "google/gemini-2.5-pro",
    "openai/gpt-5.2",
    "anthropic/claude-opus-4-20250514",
    "openai/o3",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _index_in_ranking(ranking: List[str], name: str) -> Optional[int]:
    """Return the index of *name* in *ranking*, or ``None`` if absent."""
    for i, entry in enumerate(ranking):
        if entry == name:
            return i
    return None


def _sort_candidates(candidates: List[str], ranking: List[str]) -> List[str]:
    """Sort *candidates* according to *ranking* order.

    Known models appear first (in ranking order).  Unknown models are
    appended at the end, preserving their original relative order.
    """
    known: list[tuple[int, str]] = []
    unknown: list[str] = []
    for c in candidates:
        idx = _index_in_ranking(ranking, c)
        if idx is not None:
            known.append((idx, c))
        else:
            unknown.append(c)
    known.sort(key=lambda x: x[0])
    return [name for _, name in known] + unknown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_higher_quality_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    """Return the next higher-quality candidate, or ``None`` if at the top.

    Candidates are sorted by :data:`QUALITY_RANKING` (best-first).  The
    returned value is the candidate one position *before* (= higher quality)
    the current model in that sorted order.
    """
    sorted_cands = _sort_candidates(candidates, QUALITY_RANKING)

    try:
        idx = sorted_cands.index(current)
    except ValueError:
        return None

    if idx == 0:
        return None  # already highest quality
    return sorted_cands[idx - 1]


def get_faster_neighbor(current: str, candidates: List[str]) -> Optional[str]:
    """Return the next faster candidate, or ``None`` if already the fastest.

    Candidates are sorted by :data:`SPEED_RANKING` (fastest-first).  The
    returned value is the candidate one position *before* (= faster) the
    current model in that sorted order.
    """
    sorted_cands = _sort_candidates(candidates, SPEED_RANKING)

    try:
        idx = sorted_cands.index(current)
    except ValueError:
        return None

    if idx == 0:
        return None  # already fastest
    return sorted_cands[idx - 1]
