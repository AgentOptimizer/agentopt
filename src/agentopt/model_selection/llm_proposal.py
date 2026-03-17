"""
LLM-proposal model selection: an LLM proposes which combinations to test.

This selector uses a lightweight proposer LLM (default: gpt-4o-mini) to rank
candidate combinations before expensive evaluation. It then evaluates only the
proposed subset plus baseline/exploration fallbacks.
"""

import hashlib
import itertools
import json
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..base_models import Dataset, EvalFn
from ..model_proxy import ModelProxy
from .base import BaseModelSelector, ModelResult, SelectionResults, _CACHE_SENTINEL

logger = logging.getLogger(__name__)


class LMProposalModelSelector(BaseModelSelector):
    """
    Model selector that asks an LLM which combinations are worth evaluating.

    Input contract is the same as other selectors:
      - models={proxy: [candidate1, candidate2, ...]}
      - dataset
      - invoke_fn or agent
      - eval_fn
    """

    def __init__(
        self,
        models: Dict[ModelProxy, List[Any]],
        eval_fn: EvalFn,
        dataset: Dataset,
        agent: Any = None,
        invoke_fn: Optional[Callable] = None,
        cache: Optional["EvalCache"] = _CACHE_SENTINEL,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        proposer_model: str = "gpt-4o-mini",
        proposer_client: Any = None,
        proposer_temperature: float = 0.0,
        max_combinations: int = 12,
        min_include_baselines: int = 1,
        exploration_fraction: float = 0.2,
        dataset_preview_size: int = 5,
        objective: str = "accuracy_then_latency",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            models=models,
            eval_fn=eval_fn,
            agent=agent,
            invoke_fn=invoke_fn,
            dataset=dataset,
            model_prices=model_prices,
            cache=cache,
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
        self.proposer_temperature = proposer_temperature
        self.max_combinations = max_combinations
        self.min_include_baselines = min_include_baselines
        self.exploration_fraction = exploration_fraction
        self.dataset_preview_size = dataset_preview_size
        self.objective = objective
        self.seed = seed
        self._proposal_cache: Dict[str, List[Tuple[int, ...]]] = {}
        self.last_proposal_stats: Dict[str, Any] = {}
        self._last_combo_source_by_name: Dict[str, str] = {}

        if proposer_client is None:
            from openai import OpenAI

            proposer_client = OpenAI()
        self.proposer_client = proposer_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_best(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        if parallel:
            try:
                return self._select_parallel(max_workers)
            except RuntimeError as e:
                logger.warning(
                    "Parallel evaluation failed, falling back to sequential: %s", e
                )
        return self._select_sequential()

    # ------------------------------------------------------------------
    # Proposal logic
    # ------------------------------------------------------------------

    def _get_candidate_name(self, candidate: Any) -> str:
        return self._get_model_name(candidate)

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
                {
                    "input": self._safe_json(input_data),
                    "expected": str(expected),
                }
            )
        return preview

    def _proposal_cache_key(
        self,
        candidate_lists: Sequence[Sequence[Any]],
        preview: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "objective": self.objective,
            "max_combinations": self.max_combinations,
            "proposer_model": self.proposer_model,
            "candidates": [
                [self._get_candidate_name(c) for c in candidate_list]
                for candidate_list in candidate_lists
            ],
            "preview": preview,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_baseline_index_combos(
        self, candidate_lists: Sequence[Sequence[Any]]
    ) -> List[Tuple[int, ...]]:
        if not candidate_lists:
            return []
        n = len(candidate_lists)
        baselines: List[Tuple[int, ...]] = []

        # Baseline 1: first candidate for every proxy.
        baselines.append(tuple(0 for _ in range(n)))

        # Baseline 2: last candidate for every proxy.
        baselines.append(tuple(len(c) - 1 for c in candidate_lists))

        # Baseline 3+: vary one proxy at a time from baseline 1.
        for i, cands in enumerate(candidate_lists):
            if len(cands) <= 1:
                continue
            idxs = [0 for _ in range(n)]
            idxs[i] = len(cands) - 1
            baselines.append(tuple(idxs))
        return baselines

    def _build_proposer_prompt(
        self,
        candidate_lists: Sequence[Sequence[Any]],
        preview: List[Dict[str, Any]],
    ) -> str:
        proxies_spec = []
        for proxy_idx, candidate_list in enumerate(candidate_lists):
            proxies_spec.append(
                {
                    "proxy_id": proxy_idx,
                    "candidates": [
                        {"index": idx, "name": self._get_candidate_name(candidate)}
                        for idx, candidate in enumerate(candidate_list)
                    ],
                }
            )

        return (
            "Propose model-index combinations to evaluate for a multi-proxy agent.\n"
            f"Optimization objective: {self.objective}\n"
            "Use candidate indices exactly as provided.\n\n"
            f"Proxy candidates (ordered): {json.dumps(proxies_spec, ensure_ascii=True)}\n\n"
            f"Dataset preview (input + expected): {json.dumps(preview, ensure_ascii=True)}\n\n"
            "Return JSON only with this schema:\n"
            '{"combinations": [[idx_for_proxy0, idx_for_proxy1, ...], ...]}\n'
            "Rules:\n"
            "- Every combination must have exactly one index per proxy.\n"
            "- Indices must be integers within range.\n"
            "- Rank strongest combinations first."
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _parse_proposed_index_combinations(
        self,
        raw_text: str,
        candidate_lists: Sequence[Sequence[Any]],
    ) -> List[Tuple[int, ...]]:
        if not raw_text.strip():
            return []
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("LMProposalModelSelector: proposer returned non-JSON output.")
            return []

        combos = payload.get("combinations", [])
        if not isinstance(combos, list):
            return []

        n_proxies = len(candidate_lists)
        seen = set()
        valid: List[Tuple[int, ...]] = []
        for combo in combos:
            if not isinstance(combo, (list, tuple)) or len(combo) != n_proxies:
                continue
            converted: List[int] = []
            ok = True
            for proxy_idx, raw_idx in enumerate(combo):
                if isinstance(raw_idx, bool):
                    ok = False
                    break
                if not isinstance(raw_idx, int):
                    try:
                        raw_idx = int(raw_idx)
                    except (TypeError, ValueError):
                        ok = False
                        break
                if raw_idx < 0 or raw_idx >= len(candidate_lists[proxy_idx]):
                    ok = False
                    break
                converted.append(raw_idx)
            if not ok:
                continue
            tup = tuple(converted)
            if tup in seen:
                continue
            seen.add(tup)
            valid.append(tup)
        return valid

    def _propose_index_combinations(
        self,
        candidate_lists: Sequence[Sequence[Any]],
        preview: List[Dict[str, Any]],
    ) -> List[Tuple[int, ...]]:
        cache_key = self._proposal_cache_key(candidate_lists, preview)
        cached = self._proposal_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        prompt = self._build_proposer_prompt(candidate_lists, preview)
        try:
            response = self.proposer_client.chat.completions.create(
                model=self.proposer_model,
                temperature=self.proposer_temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful model-selection proposer. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
            raw_text = self._content_to_text(content)
            proposed = self._parse_proposed_index_combinations(raw_text, candidate_lists)
        except Exception as e:
            logger.warning("LM proposer failed, using fallback sampling: %s", e)
            proposed = []

        self._proposal_cache[cache_key] = list(proposed)
        return proposed

    def _get_sampled_combinations(
        self,
    ) -> Tuple[List[ModelProxy], List[tuple], List[tuple]]:
        proxies = list(self._models.keys())
        candidate_lists = list(self._models.values())
        all_combinations = list(itertools.product(*candidate_lists))

        if not all_combinations:
            self.last_proposal_stats = {"total_space": 0, "selected_total": 0}
            self._last_combo_source_by_name = {}
            return proxies, [], []

        if len(all_combinations) <= self.max_combinations:
            self.last_proposal_stats = {
                "total_space": len(all_combinations),
                "selected_total": len(all_combinations),
                "proposed_valid": 0,
                "baseline_selected": 0,
                "exploration_selected": 0,
                "random_fill_selected": 0,
                "full_search": True,
            }
            self._last_combo_source_by_name = {
                " + ".join(self._get_model_name(m) for m in combo): "full_search"
                for combo in all_combinations
            }
            return proxies, all_combinations, all_combinations

        all_index_combinations = list(
            itertools.product(*[range(len(cands)) for cands in candidate_lists])
        )
        index_to_combo: Dict[Tuple[int, ...], tuple] = {
            idx_combo: tuple(
                candidate_lists[i][idx] for i, idx in enumerate(idx_combo)
            )
            for idx_combo in all_index_combinations
        }

        selected_indices: List[Tuple[int, ...]] = []
        selected_set = set()
        source_by_index: Dict[Tuple[int, ...], str] = {}

        def _add(idx_combo: Tuple[int, ...], source: str) -> None:
            if idx_combo in selected_set:
                return
            selected_set.add(idx_combo)
            selected_indices.append(idx_combo)
            source_by_index[idx_combo] = source

        # 1) Baselines.
        for idx_combo in self._build_baseline_index_combos(candidate_lists):
            if len(selected_indices) >= min(
                self.min_include_baselines, self.max_combinations
            ):
                break
            if idx_combo in index_to_combo:
                _add(idx_combo, "baseline")

        # 2) LLM proposals.
        preview = self._dataset_preview()
        proposed_indices = self._propose_index_combinations(candidate_lists, preview)
        for idx_combo in proposed_indices:
            if len(selected_indices) >= self.max_combinations:
                break
            if idx_combo in index_to_combo:
                _add(idx_combo, "proposed")

        remaining = [idx for idx in all_index_combinations if idx not in selected_set]
        rng = random.Random(self.seed)

        # 3) Exploration random picks.
        slots_left = self.max_combinations - len(selected_indices)
        explore_count = min(
            slots_left,
            max(0, math.ceil(self.max_combinations * self.exploration_fraction)),
            len(remaining),
        )
        if explore_count > 0:
            for idx_combo in rng.sample(remaining, explore_count):
                _add(idx_combo, "exploration")
            remaining = [idx for idx in remaining if idx not in selected_set]

        # 4) Fill any leftover slots randomly.
        slots_left = self.max_combinations - len(selected_indices)
        if slots_left > 0 and remaining:
            fill_count = min(slots_left, len(remaining))
            for idx_combo in rng.sample(remaining, fill_count):
                _add(idx_combo, "random_fill")

        sampled = [index_to_combo[idx] for idx in selected_indices]
        self.last_proposal_stats = {
            "total_space": len(all_combinations),
            "selected_total": len(sampled),
            "proposed_valid": len(proposed_indices),
            "baseline_selected": sum(1 for s in source_by_index.values() if s == "baseline"),
            "exploration_selected": sum(
                1 for s in source_by_index.values() if s == "exploration"
            ),
            "random_fill_selected": sum(
                1 for s in source_by_index.values() if s == "random_fill"
            ),
            "full_search": False,
        }
        self._last_combo_source_by_name = {
            " + ".join(self._get_model_name(m) for m in combo): source_by_index[idx]
            for idx, combo in zip(selected_indices, sampled)
        }
        return proxies, all_combinations, sampled

    # ------------------------------------------------------------------
    # Sequential evaluation (proxy swap in-place)
    # ------------------------------------------------------------------

    def _select_sequential(self) -> SelectionResults:
        proxies, all_combinations, sampled_combinations = (
            self._get_sampled_combinations()
        )

        all_results: List[ModelResult] = []
        best_combination = None
        best_accuracy = float("-inf")
        best_latency = float("inf")
        accuracy_tolerance = 1e-9

        print(f"\n{'='*60}")
        print(
            "LLM proposal (sequential): "
            f"{len(sampled_combinations)}/{len(all_combinations)} combinations "
            f"(max={self.max_combinations})"
        )
        print(f"{'='*60}\n")

        for idx, combo in enumerate(sampled_combinations, 1):
            combo_name = " + ".join(self._get_model_name(m) for m in combo)
            for proxy, model_obj in zip(proxies, combo):
                proxy.set_model(model_obj)

            print(f"  [{idx}/{len(sampled_combinations)}] Evaluating: {combo_name}")
            for i, (_, model_obj) in enumerate(zip(proxies, combo)):
                print(f"    proxy[{i}] → {self._get_model_name(model_obj)}")
            try:
                scores, latencies, tokens = self._evaluate_sequential(
                    self.dataset, label=combo_name
                )
                accuracy, _ = self._compute_stats(scores)
                latency = sum(latencies) / len(latencies) if latencies else 0.0
                in_tokens, out_tokens = self._split_tokens(tokens)

                result = self._make_result(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    attribute="combination",
                    is_best=False,
                )
                print(f"  {result}")
                all_results.append(result)

                should_update = False
                if best_combination is None:
                    should_update = True
                elif accuracy > best_accuracy + accuracy_tolerance:
                    should_update = True
                elif (
                    abs(accuracy - best_accuracy) <= accuracy_tolerance
                    and latency < best_latency
                ):
                    should_update = True

                if should_update:
                    best_accuracy = accuracy
                    best_latency = latency
                    best_combination = combo

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

        if best_combination is not None:
            best_name = " + ".join(self._get_model_name(m) for m in best_combination)
            for proxy, model_obj in zip(proxies, best_combination):
                proxy.set_model(model_obj)
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
            src = self._last_combo_source_by_name.get(best_name, "unknown")
            self.last_proposal_stats["best_source"] = src
            self.last_proposal_stats["proposer_hit"] = src == "proposed"
        else:
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results

    # ------------------------------------------------------------------
    # Parallel evaluation (agent clones + thread pool)
    # ------------------------------------------------------------------

    def _select_parallel(
        self,
        max_workers: Optional[int] = None,
    ) -> SelectionResults:
        from ..model_proxy.adapter import get_adapter
        from ..model_proxy.builders import build_llm
        from ..model_proxy.model_copy import clone_model_spec
        from ..model_proxy.token_tracking import TokenAccumulator

        proxies, all_combinations, sampled_combinations = (
            self._get_sampled_combinations()
        )

        if max_workers is None:
            max_workers = len(sampled_combinations)

        adapter = get_adapter(self.agent) if self.agent is not None else self._adapter

        print(f"\n{'='*60}")
        print(
            "LLM proposal (parallel): "
            f"{len(sampled_combinations)}/{len(all_combinations)} combinations, "
            f"{max_workers} workers"
        )
        print(f"{'='*60}\n")

        if self.agent is not None:
            print(f"  Cloning agents for {len(sampled_combinations)} combinations ...")
            tasks: List[Tuple[str, tuple, Any, bool, Any]] = []

            for idx, combo in enumerate(sampled_combinations, 1):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(
                    f"    clone {idx}/{len(sampled_combinations)}: {combo_name}",
                    flush=True,
                )
                for i, (_, model_obj) in enumerate(zip(proxies, combo)):
                    print(f"      proxy[{i}] → {self._get_model_name(model_obj)}")

                try:
                    if adapter is not None:
                        agent_copy = adapter.clone_for_parallel(
                            self.agent, proxies, combo, self._get_model_name
                        )
                    else:
                        import copy

                        agent_copy = copy.deepcopy(self.agent)
                    tracker = (
                        adapter.create_token_tracker(agent_copy)
                        if adapter is not None
                        else None
                    )
                    invoke_fn = (
                        adapter.get_invoke_fn(agent_copy)
                        if adapter is not None
                        else self._make_invoke_fn(
                            agent_copy, self._invoke_method_name, self.is_async
                        )
                    )
                    tasks.append((combo_name, combo, invoke_fn, False, tracker))
                except Exception as e:
                    logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)
                    continue

            if not tasks:
                raise RuntimeError("All clone attempts failed.")

            print(
                f"\n  Evaluating {len(tasks)} combinations across {max_workers} workers ..."
            )
            all_results: List[ModelResult] = []
            future_to_info: Dict[Any, Tuple[str, tuple]] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for combo_name, combo, invoke_fn, is_async_flag, tracker in tasks:
                    future = executor.submit(
                        self._evaluate_thread_safe,
                        invoke_fn,
                        is_async_flag,
                        self.eval_fn,
                        self.dataset,
                        token_tracker=tracker,
                        label=combo_name,
                        cache=self._cache,
                        proxies=proxies,
                    )
                    future_to_info[future] = (combo_name, combo)

                for future in as_completed(future_to_info):
                    combo_name, _ = future_to_info[future]
                    try:
                        scores, latencies, tokens = future.result()
                        accuracy, _ = self._compute_stats(scores)
                        latency = sum(latencies) / len(latencies) if latencies else 0.0
                        in_tokens, out_tokens = self._split_tokens(tokens)
                        result = self._make_result(
                            model_name=combo_name,
                            accuracy=accuracy,
                            latency_seconds=latency,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            attribute="combination",
                            is_best=False,
                        )
                        print(f"  {result}")
                        all_results.append(result)
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
        else:
            print(f"  Preparing {len(sampled_combinations)} combinations ...")
            for idx, combo in enumerate(sampled_combinations, 1):
                combo_name = " + ".join(self._get_model_name(m) for m in combo)
                print(
                    f"    combo {idx}/{len(sampled_combinations)}: {combo_name}",
                    flush=True,
                )
                for i, (_, model_obj) in enumerate(zip(proxies, combo)):
                    print(f"      proxy[{i}] → {self._get_model_name(model_obj)}")

            print(
                f"\n  Evaluating {len(sampled_combinations)} combinations across "
                f"{max_workers} workers ..."
            )
            all_results: List[ModelResult] = []
            future_to_info: Dict[Any, Tuple[str, tuple]] = {}

            def _eval_with_thread_local(
                combo: tuple,
                proxies: List[ModelProxy],
                invoke_fn: Any,
                is_async: bool,
                eval_fn: Any,
                dataset: Any,
                get_model_name: Any,
                label: str,
            ) -> Tuple[float, float, Dict]:
                tracker = TokenAccumulator()
                for proxy, model_spec in zip(proxies, combo):
                    if isinstance(model_spec, str):
                        model_name = get_model_name(model_spec)
                        current_model = object.__getattribute__(proxy, "_optmodel")
                        fresh_model = build_llm(model_name, current_model)
                        if fresh_model is None:
                            raise RuntimeError(
                                f"Cannot build model '{model_name}' for parallel evaluation."
                            )
                    else:
                        fresh_model = clone_model_spec(model_spec)
                    proxy._set_thread_model(fresh_model, tracker)
                try:
                    return self._evaluate_thread_safe(
                        invoke_fn,
                        is_async,
                        eval_fn,
                        dataset,
                        token_tracker=tracker,
                        label=label,
                        cache=self._cache,
                        proxies=proxies,
                    )
                finally:
                    for proxy in proxies:
                        proxy._clear_thread_model()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for combo in sampled_combinations:
                    combo_name = " + ".join(self._get_model_name(m) for m in combo)
                    future = executor.submit(
                        _eval_with_thread_local,
                        combo,
                        proxies,
                        self.invoke_fn,
                        self.is_async,
                        self.eval_fn,
                        self.dataset,
                        self._get_model_name,
                        combo_name,
                    )
                    future_to_info[future] = (combo_name, combo)

                for future in as_completed(future_to_info):
                    combo_name, _ = future_to_info[future]
                    try:
                        scores, latencies, tokens = future.result()
                        accuracy, _ = self._compute_stats(scores)
                        latency = sum(latencies) / len(latencies) if latencies else 0.0
                        in_tokens, out_tokens = self._split_tokens(tokens)
                        result = self._make_result(
                            model_name=combo_name,
                            accuracy=accuracy,
                            latency_seconds=latency,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            attribute="combination",
                            is_best=False,
                        )
                        print(f"  {result}")
                        all_results.append(result)
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

        best_info = self._find_best(all_results)
        if best_info is not None:
            best_name, _ = best_info
            for combo in sampled_combinations:
                if " + ".join(self._get_model_name(m) for m in combo) == best_name:
                    for proxy, model_obj in zip(proxies, combo):
                        proxy.set_model(model_obj)
                    break
            for result in all_results:
                if result.model_name == best_name:
                    result.is_best = True
                    break
            src = self._last_combo_source_by_name.get(best_name, "unknown")
            self.last_proposal_stats["best_source"] = src
            self.last_proposal_stats["proposer_hit"] = src == "proposed"
        else:
            print("\n  No sampled combinations succeeded")

        results = SelectionResults(results=all_results)
        print(results)
        return results
