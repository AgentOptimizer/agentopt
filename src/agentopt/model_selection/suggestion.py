"""LLM-assisted model suggestion utilities.

This module provides an optional router that asks an LLM to rank model
candidates for a specific task before running full evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from ..model_proxy.constants import MODEL_FIELDS


@dataclass(frozen=True)
class ModelSuggestion:
    """Single router suggestion."""

    model_name: str
    score: float
    reason: str


class LLMModelSuggester:
    """Ask an LLM router to rank model candidates for a task."""

    def __init__(
        self,
        router_model: str = "gpt-4o-mini",
        client: Any | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.router_model = router_model
        self.temperature = temperature
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def suggest(
        self,
        task: str,
        candidates: Sequence[Any] | Iterable[Any],
        top_k: int = 3,
        context: Dict[str, Any] | None = None,
    ) -> List[ModelSuggestion]:
        """Return top-k model suggestions for a task."""
        candidate_list = list(candidates)
        candidate_names = [self._candidate_name(c) for c in candidate_list]
        if not candidate_names:
            return []

        user_prompt = self._build_prompt(task, candidate_names, context)
        response = self.client.chat.completions.create(
            model=self.router_model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a model-routing assistant. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)

        suggestions: List[ModelSuggestion] = []
        seen = set()
        for item in payload.get("suggestions", []):
            name = str(item.get("model_name", "")).strip()
            if not name or name not in candidate_names or name in seen:
                continue
            seen.add(name)
            score = float(item.get("score", 0.0))
            reason = str(item.get("reason", "")).strip()
            suggestions.append(
                ModelSuggestion(
                    model_name=name,
                    score=max(0.0, min(1.0, score)),
                    reason=reason,
                )
            )

        # Fallback if router returns malformed/partial ranking.
        if len(suggestions) < min(top_k, len(candidate_names)):
            for name in candidate_names:
                if name in seen:
                    continue
                suggestions.append(
                    ModelSuggestion(
                        model_name=name,
                        score=0.0,
                        reason="Fallback candidate (router did not rank it).",
                    )
                )
                seen.add(name)
                if len(suggestions) >= min(top_k, len(candidate_names)):
                    break

        suggestions.sort(key=lambda s: s.score, reverse=True)
        return suggestions[: min(top_k, len(candidate_names))]

    def shortlist_candidates(
        self,
        task: str,
        candidates: Sequence[Any] | Iterable[Any],
        top_k: int = 3,
        context: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Return candidate objects corresponding to the top-k suggestions."""
        candidate_list = list(candidates)
        name_to_candidate: Dict[str, Any] = {}
        for candidate in candidate_list:
            name = self._candidate_name(candidate)
            if name not in name_to_candidate:
                name_to_candidate[name] = candidate

        suggestions = self.suggest(
            task=task,
            candidates=candidate_list,
            top_k=top_k,
            context=context,
        )
        return [
            name_to_candidate[s.model_name]
            for s in suggestions
            if s.model_name in name_to_candidate
        ]

    @staticmethod
    def _candidate_name(candidate: Any) -> str:
        if isinstance(candidate, str):
            return candidate
        for field in MODEL_FIELDS:
            value = getattr(candidate, field, None)
            if value is not None:
                return str(value)
        return type(candidate).__name__

    @staticmethod
    def _build_prompt(
        task: str,
        candidate_names: List[str],
        context: Dict[str, Any] | None = None,
    ) -> str:
        context_json = json.dumps(context or {}, ensure_ascii=True)
        candidates_json = json.dumps(candidate_names, ensure_ascii=True)
        return (
            "Given the task and candidate models, rank the best candidates.\n"
            "Task:\n"
            f"{task}\n\n"
            "Optional context (quality/latency/cost priorities, domain, past outcomes):\n"
            f"{context_json}\n\n"
            "Candidate models:\n"
            f"{candidates_json}\n\n"
            "Return JSON only with this schema:\n"
            '{\"suggestions\":[{\"model_name\":\"<one candidate>\",\"score\":0.0,\"reason\":\"...\"}]}\n'
            "Only use model_name values from the candidate list."
        )
