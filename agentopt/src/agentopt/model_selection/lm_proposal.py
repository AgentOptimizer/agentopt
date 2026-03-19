"""
LLM-proposal model selector.

This selector asks a proposer LLM to suggest which model combinations are worth
evaluating, then evaluates only that subset (+ deterministic fallback combos).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..base_models import Dataset, EvalFn, ModelCandidate
from .base import BaseModelSelector, ModelResult, SelectionResults

logger = logging.getLogger(__name__)


class _ProposalResponse(BaseModel):
    """Structured response schema for proposer LLM output."""

    combinations: List[List[int]] = Field(default_factory=list)


class LMProposalModelSelector(BaseModelSelector):
    """Model selector where an LLM proposes promising combinations first."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        proposer_model: str = "gpt-4o-mini",
        proposer_client: Any = None,
        objective: str = "accuracy_then_latency",
        max_combinations: int = 12,
        min_include_baselines: int = 1,
        exploration_fraction: float = 0.2,
        dataset_preview_size: int = 5,
        seed: Optional[int] = None,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        super().__init__(
            agent_fn=agent_fn,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
        )
        if max_combinations < 1:
            raise ValueError("max_combinations must be >= 1.")
        if min_include_baselines < 0:
            raise ValueError("min_include_baselines must be >= 0.")
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in [0, 1].")
        if dataset_preview_size < 1:
            raise ValueError("dataset_preview_size must be >= 1.")

        self.proposer_model = proposer_model
        self.objective = objective
        self.max_combinations = max_combinations
        self.min_include_baselines = min_include_baselines
        self.exploration_fraction = exploration_fraction
        self.dataset_preview_size = dataset_preview_size
        self.seed = seed

        # Introspection hooks for users.
        self.last_proposal_stats: Dict[str, Any] = {}
        self._combo_source: Dict[str, str] = {}
        self._proposal_cache: Dict[str, List[Tuple[int, ...]]] = {}

        if proposer_client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "LMProposalModelSelector requires `openai` unless proposer_client is supplied. "
                    "Install with: pip install openai"
                ) from e
            proposer_client = OpenAI()
        self.proposer_client = proposer_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            return asyncio.run(self._select_async(max_concurrent))
        return self._select_sequential()

    # ------------------------------------------------------------------
    # Proposal construction
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _dataset_preview(self) -> List[Dict[str, Any]]:
        preview: List[Dict[str, Any]] = []
        for input_data, expected in list(self.dataset)[: self.dataset_preview_size]:
            preview.append(
                {"input": self._safe_json(input_data), "expected": str(expected)}
            )
        return preview

    def _all_index_combos(self) -> List[Tuple[int, ...]]:
        return list(itertools_product_ranges([len(v) for v in self._models.values()]))

    def _index_combo_to_combo(
        self, idx_combo: Tuple[int, ...],
    ) -> Dict[str, ModelCandidate]:
        return {
            node: self._models[node][idx]
            for node, idx in zip(self._node_names, idx_combo)
        }

    def _proposal_cache_key(self, preview: List[Dict[str, Any]]) -> str:
        payload = {
            "objective": self.objective,
            "max_combinations": self.max_combinations,
            "proposer_model": self.proposer_model,
            "node_names": self._node_names,
            "candidates": [
                [self._candidate_label(c) for c in self._models[node]]
                for node in self._node_names
            ],
            "preview": preview,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_prompt(self, preview: List[Dict[str, Any]]) -> str:
        proxies = []
        for node_idx, node in enumerate(self._node_names):
            proxies.append(
                {
                    "node_name": node,
                    "proxy_order": node_idx,
                    "candidates": [
                        {"index": idx, "name": self._candidate_label(c)}
                        for idx, c in enumerate(self._models[node])
                    ],
                }
            )

        return (
            "Choose promising model combinations to evaluate for a multi-node agent.\n"
            f"Objective: {self.objective}\n\n"
            "Nodes and candidate models (ordered):\n"
            f"{json.dumps(proxies, ensure_ascii=True)}\n\n"
            "Dataset preview (input + expected):\n"
            f"{json.dumps(preview, ensure_ascii=True)}\n\n"
            "Return combinations in the schema field `combinations`.\n"
            "Rules:\n"
            "- Use integer indices only.\n"
            "- Keep node order exactly as provided.\n"
            "- Rank best combinations first.\n"
        )

    def _parse_proposed_indices(
        self, combos: Sequence[Sequence[int]], all_index_combos: Sequence[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        valid_space = set(all_index_combos)
        valid: List[Tuple[int, ...]] = []
        seen = set()
        n_nodes = len(self._node_names)

        for combo in combos:
            if not isinstance(combo, (list, tuple)) or len(combo) != n_nodes:
                continue
            try:
                idx_combo = tuple(int(v) for v in combo)
            except (TypeError, ValueError):
                continue
            if idx_combo not in valid_space or idx_combo in seen:
                continue
            seen.add(idx_combo)
            valid.append(idx_combo)
        return valid

    def _ask_proposer(
        self, all_index_combos: Sequence[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        preview = self._dataset_preview()
        key = self._proposal_cache_key(preview)
        if key in self._proposal_cache:
            return list(self._proposal_cache[key])

        prompt = self._build_prompt(preview)
        try:
            # Prefer structured parsing so output format is guaranteed by schema.
            if (
                hasattr(self.proposer_client, "beta")
                and hasattr(self.proposer_client.beta, "chat")
                and hasattr(self.proposer_client.beta.chat, "completions")
                and hasattr(self.proposer_client.beta.chat.completions, "parse")
            ):
                response = self.proposer_client.beta.chat.completions.parse(
                    model=self.proposer_model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise model-selection proposer. "
                                "Return valid output that matches the schema."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format=_ProposalResponse,
                )
                parsed = response.choices[0].message.parsed
                combos = parsed.combinations if parsed is not None else []
                proposed = self._parse_proposed_indices(combos, all_index_combos)
            else:
                # Fallback for custom clients: keep behavior but still parse through pydantic.
                response = self.proposer_client.chat.completions.create(
                    model=self.proposer_model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": "Return JSON with key `combinations` only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                if isinstance(content, str):
                    payload = json.loads(content)
                else:
                    payload = json.loads(str(content))
                parsed = _ProposalResponse.model_validate(payload)
                proposed = self._parse_proposed_indices(
                    parsed.combinations, all_index_combos
                )
        except Exception as e:
            logger.warning("LM proposer call failed; falling back to heuristics: %s", e)
            proposed = []

        self._proposal_cache[key] = list(proposed)
        return proposed

    def _baseline_indices(
        self, all_index_combos: Sequence[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        if not all_index_combos:
            return []
        lengths = [len(self._models[n]) for n in self._node_names]
        base = tuple(0 for _ in lengths)
        high = tuple(max(0, n - 1) for n in lengths)

        baselines = [base, high]
        for i, n in enumerate(lengths):
            if n <= 1:
                continue
            combo = [0 for _ in lengths]
            combo[i] = n - 1
            baselines.append(tuple(combo))
        # Keep only valid entries.
        space = set(all_index_combos)
        return [b for b in baselines if b in space]

    def _select_index_combos(
        self,
    ) -> Tuple[List[Dict[str, ModelCandidate]], List[Dict[str, ModelCandidate]]]:
        all_index_combos = self._all_index_combos()
        all_combos = [self._index_combo_to_combo(c) for c in all_index_combos]
        total = len(all_index_combos)

        if total <= self.max_combinations:
            self.last_proposal_stats = {
                "total_space": total,
                "selected_total": total,
                "proposed_valid": 0,
                "baseline_selected": 0,
                "exploration_selected": 0,
                "random_fill_selected": 0,
                "full_search": True,
            }
            self._combo_source = {self._combo_name(c): "full_search" for c in all_combos}
            return all_combos, all_combos

        selected: List[Tuple[int, ...]] = []
        source_by_idx: Dict[Tuple[int, ...], str] = {}

        def add(idx_combo: Tuple[int, ...], source: str) -> None:
            if idx_combo in source_by_idx:
                return
            if len(selected) >= self.max_combinations:
                return
            selected.append(idx_combo)
            source_by_idx[idx_combo] = source

        for combo in self._baseline_indices(all_index_combos):
            if len([1 for s in source_by_idx.values() if s == "baseline"]) >= min(
                self.min_include_baselines, self.max_combinations
            ):
                break
            add(combo, "baseline")

        proposed = self._ask_proposer(all_index_combos)
        for combo in proposed:
            add(combo, "proposed")

        rng = random.Random(self.seed)
        remaining = [c for c in all_index_combos if c not in source_by_idx]
        slots_left = self.max_combinations - len(selected)
        explore_target = min(
            slots_left,
            max(0, math.ceil(self.max_combinations * self.exploration_fraction)),
            len(remaining),
        )
        if explore_target > 0:
            for combo in rng.sample(remaining, explore_target):
                add(combo, "exploration")
            remaining = [c for c in remaining if c not in source_by_idx]

        slots_left = self.max_combinations - len(selected)
        if slots_left > 0 and remaining:
            for combo in rng.sample(remaining, min(slots_left, len(remaining))):
                add(combo, "random_fill")

        sampled = [self._index_combo_to_combo(c) for c in selected]
        self._combo_source = {
            self._combo_name(combo): source_by_idx[idx]
            for idx, combo in zip(selected, sampled)
        }
        self.last_proposal_stats = {
            "total_space": total,
            "selected_total": len(sampled),
            "proposed_valid": len(proposed),
            "baseline_selected": sum(1 for s in source_by_idx.values() if s == "baseline"),
            "exploration_selected": sum(
                1 for s in source_by_idx.values() if s == "exploration"
            ),
            "random_fill_selected": sum(
                1 for s in source_by_idx.values() if s == "random_fill"
            ),
            "full_search": False,
        }
        return all_combos, sampled

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _select_sequential(self) -> SelectionResults:
        all_combos, sampled = self._select_index_combos()

        all_results: List[ModelResult] = []
        best_combo_name = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        tol = 1e-9

        print(f"\n{'='*60}")
        print(
            f"LM proposal (sequential): "
            f"{len(sampled)}/{len(all_combos)} combinations "
            f"(max={self.max_combinations})"
        )
        print(f"{'='*60}\n")

        for idx, combo in enumerate(sampled, 1):
            combo_name = self._combo_name(combo)
            print(f"  [{idx}/{len(sampled)}] Evaluating: {combo_name}")

            try:
                scores, latencies, dp_ids = self._evaluate_combo(
                    combo, self.dataset, label=combo_name
                )
                input_tokens, output_tokens = self._fetch_tokens(combo_name)
                accuracy, _ = self._compute_stats(scores)
                latency = sum(latencies) / len(latencies) if latencies else 0.0
                dp_results = self._build_datapoint_results(scores, latencies, dp_ids)

                result = self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attribute="combination",
                    is_best=False,
                    datapoint_results=dp_results,
                )
                print(f"  {result}")
                all_results.append(result)

                if (accuracy > best_accuracy + tol) or (
                    abs(accuracy - best_accuracy) <= tol and latency < best_latency
                ):
                    best_accuracy = accuracy
                    best_latency = latency
                    best_combo_name = combo_name

            except Exception as e:
                print(f"  [{combo_name}] failed: {e}")
                all_results.append(
                    self._make_result(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        input_tokens={},
                        output_tokens={},
                        attribute="combination",
                        is_best=False,
                    )
                )

        if best_combo_name is not None:
            for result in all_results:
                if result.model_name == best_combo_name:
                    result.is_best = True
                    break
            source = self._combo_source.get(best_combo_name, "unknown")
            self.last_proposal_stats["best_source"] = source
            self.last_proposal_stats["proposer_hit"] = source == "proposed"
        else:
            print("\n  No sampled combinations succeeded")

        return SelectionResults(results=all_results)

    async def _select_async(self, max_concurrent: int = 20) -> SelectionResults:
        all_combos, sampled = self._select_index_combos()

        print(f"\n{'='*60}")
        print(
            f"LM proposal (async): "
            f"{len(sampled)}/{len(all_combos)} combinations "
            f"(max={self.max_combinations}), max {max_concurrent} concurrent per combo"
        )
        print(f"{'='*60}\n")

        async def _eval_combo(
            combo: Dict[str, ModelCandidate],
        ) -> Tuple[str, ModelResult]:
            combo_name = self._combo_name(combo)
            print(f"  Evaluating: {combo_name}")
            scores, latencies, dp_ids = await self._evaluate_combo_async(
                combo, self.dataset, label=combo_name, max_concurrent=max_concurrent
            )
            input_tokens, output_tokens = self._fetch_tokens(combo_name)
            accuracy, _ = self._compute_stats(scores)
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            dp_results = self._build_datapoint_results(scores, latencies, dp_ids)
            result = self._make_result(
                model_name=combo_name,
                accuracy=accuracy,
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attribute="combination",
                is_best=False,
                datapoint_results=dp_results,
            )
            print(f"  {result}")
            return combo_name, result

        combo_results = await asyncio.gather(
            *[_eval_combo(c) for c in sampled], return_exceptions=True,
        )

        all_results: List[ModelResult] = []
        for i, res in enumerate(combo_results):
            if isinstance(res, Exception):
                combo_name = self._combo_name(sampled[i])
                print(f"  [{combo_name}] failed: {res}")
                all_results.append(
                    self._make_result(
                        model_name=combo_name,
                        accuracy=0.0,
                        latency_seconds=0.0,
                        input_tokens={},
                        output_tokens={},
                        attribute="combination",
                        is_best=False,
                    )
                )
            else:
                _, result = res
                all_results.append(result)

        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for r in all_results:
                if r.model_name == best_name:
                    r.is_best = True
                    break
            source = self._combo_source.get(best_name, "unknown")
            self.last_proposal_stats["best_source"] = source
            self.last_proposal_stats["proposer_hit"] = source == "proposed"
        else:
            print("\n  No sampled combinations succeeded")

        return SelectionResults(results=all_results)


def itertools_product_ranges(lengths: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    """Cartesian product over index ranges of each length."""
    ranges = [range(n) for n in lengths]
    if not ranges:
        return []
    import itertools

    return itertools.product(*ranges)
