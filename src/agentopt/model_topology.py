"""
Model topology: general-impression rankings of LLM models by quality and speed.

These rankings encode a rough ordering (not benchmark-precise) so that search
algorithms like hill climbing can define "neighbors" in model space.
"""

from typing import List, Optional

# Provider prefixes stripped during normalization.
_PROVIDER_PREFIXES = ("openai/", "anthropic/", "google/")


def normalize_model_name(name: str) -> str:
    """Strip provider prefix to get a canonical model name.

    Examples:
        >>> normalize_model_name("openai/gpt-4o")
        'gpt-4o'
        >>> normalize_model_name("gpt-4o-mini")
        'gpt-4o-mini'
    """
    for prefix in _PROVIDER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


# ---------------------------------------------------------------------------
# Quality ranking – ordered from HIGHEST quality to LOWEST quality.
# Index 0 = best quality.
# ---------------------------------------------------------------------------
QUALITY_RANKING: List[str] = [
    # OpenAI reasoning
    "o3",
    "o4-mini",
    "o3-mini",
    # OpenAI GPT
    "gpt-5.2",
    "gpt-5.1",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    # Anthropic
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3.7-sonnet",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "claude-3-haiku",
    # Google
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# ---------------------------------------------------------------------------
# Speed ranking – ordered from FASTEST to SLOWEST.
# Index 0 = fastest.
# ---------------------------------------------------------------------------
SPEED_RANKING: List[str] = [
    # Small / lite models first
    "gpt-4.1-nano",
    "gemini-2.0-flash-lite",
    "gpt-4o-mini",
    "claude-3-haiku",
    "claude-3.5-haiku",
    "gpt-4.1-mini",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gpt-4o",
    "gpt-4.1",
    "o4-mini",
    "o3-mini",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-sonnet-4-20250514",
    "gpt-5.1",
    "gemini-2.5-pro",
    "gpt-5.2",
    "claude-opus-4-20250514",
    "o3",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _index_in_ranking(ranking: List[str], name: str) -> Optional[int]:
    """Return the index of *name* in *ranking*, or ``None`` if absent."""
    normalized = normalize_model_name(name)
    for i, entry in enumerate(ranking):
        if entry == normalized:
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
    norm_current = normalize_model_name(current)
    norm_sorted = [normalize_model_name(c) for c in sorted_cands]

    try:
        idx = norm_sorted.index(norm_current)
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
    norm_current = normalize_model_name(current)
    norm_sorted = [normalize_model_name(c) for c in sorted_cands]

    try:
        idx = norm_sorted.index(norm_current)
    except ValueError:
        return None

    if idx == 0:
        return None  # already fastest
    return sorted_cands[idx - 1]
