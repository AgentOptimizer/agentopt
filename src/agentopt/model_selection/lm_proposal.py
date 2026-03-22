"""
LLM-proposal model selector.

This selector asks a proposer LLM to suggest the single best model combination
for a multi-node agent, using node descriptions, model prices, and a dataset
preview to inform its recommendation. The proposed combination is returned
directly without evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from ..base_models import Dataset, EvalFn, ModelCandidate
from ..model_price import get_model_price
from .base import BaseModelSelector, SelectionResults

logger = logging.getLogger(__name__)


class ProposalResponse(BaseModel):
    """Expected JSON response from the proposer LLM."""

    combination: Dict[str, str] = Field(
        description="Mapping of node name to selected model name.",
    )
    reasoning: str = Field(
        default="", description="Brief explanation of why this combination was chosen.",
    )


class LMProposalModelSelector(BaseModelSelector):
    """Model selector where an LLM proposes the single best combination."""

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, ModelCandidate]], Any],
        models: Dict[str, List[ModelCandidate]],
        eval_fn: EvalFn,
        dataset: Dataset,
        invoke_fn: Optional[Callable] = None,
        proposer_model: str = "gpt-4.1",
        proposer_client: Any = None,
        objective: str = "maximize accuracy and then minimize latency and cost",
        dataset_preview_size: int = 10,
        model_prices: Optional[Dict[str, Dict[str, float]]] = None,
        node_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            agent_fn=agent_fn,
            models=models,
            eval_fn=eval_fn,
            dataset=dataset,
            invoke_fn=invoke_fn,
            model_prices=model_prices,
            node_descriptions=node_descriptions,
        )
        if dataset_preview_size < 1:
            raise ValueError("dataset_preview_size must be >= 1.")

        self.proposer_model = proposer_model
        self.objective = objective
        self.dataset_preview_size = dataset_preview_size

        # Build label→index lookup per node for parsing LLM responses.
        self._label_to_index: Dict[str, Dict[str, int]] = {}
        for node in self._node_names:
            self._label_to_index[node] = {
                self._candidate_label(c): idx
                for idx, c in enumerate(self._models[node])
            }

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

    def _run_selection(
        self, parallel: bool = False, max_concurrent: int = 20,
    ) -> SelectionResults:
        if parallel:
            logger.warning(
                "LMProposalModelSelector received parallel=True, but only a single "
                "combination is evaluated; proceeding with sequential evaluation."
            )
        combo_idx = self._ask_proposer()
        if combo_idx is None:
            combo_idx = tuple(0 for _ in self._node_names)

        combo = self._index_combo_to_combo(combo_idx)
        combo_name = self._combo_name(combo)

        print(f"\n{'='*60}")
        print(f"LM proposal: evaluating proposed combination")
        print(f"{'='*60}\n")
        print(f"  [1/1] Evaluating: {combo_name}")

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
                is_best=True,
                num_samples=len(scores),
                datapoint_results=dp_results,
            )
            print(f"  {result}")
        except Exception as e:
            print(f"  [{combo_name}] failed: {e}")
            result = self._make_result(
                model_name=combo_name,
                accuracy=0.0,
                latency_seconds=0.0,
                input_tokens={},
                output_tokens={},
                attribute="combination",
                is_best=True,
            )

        return SelectionResults(results=[result])

    # ------------------------------------------------------------------
    # Prompt construction
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

    def _index_combo_to_combo(
        self, idx_combo: Tuple[int, ...],
    ) -> Dict[str, ModelCandidate]:
        return {
            node: self._models[node][idx]
            for node, idx in zip(self._node_names, idx_combo)
        }

    def _build_prompt(self, preview: List[Dict[str, Any]]) -> str:
        # -- Build nodes info ------------------------------------------------
        nodes_info = []
        for node in self._node_names:
            node_entry: Dict[str, Any] = {"node_name": node}
            if self.node_descriptions and node in self.node_descriptions:
                node_entry["description"] = self.node_descriptions[node]

            candidates = []
            for c in self._models[node]:
                label = self._candidate_label(c)
                candidate_entry: Dict[str, Any] = {"name": label}
                price = get_model_price(label, custom_prices=self._custom_prices)
                if price is not None:
                    candidate_entry["input_price_per_mtok"] = price[0]
                    candidate_entry["output_price_per_mtok"] = price[1]
                candidates.append(candidate_entry)
            node_entry["candidates"] = candidates
            nodes_info.append(node_entry)

        # -- Build response example ------------------------------------------
        example = {
            "combination": {
                node: self._candidate_label(self._models[node][0])
                for node in self._node_names
            },
            "reasoning": "Your explanation here.",
        }

        # -- Assemble prompt -------------------------------------------------
        sections = [
            # Role & Task
            "# Task\n"
            "You are an expert AI model selector. You will be given a multi-agent "
            "workflow where each node can use one of several candidate LLMs. "
            "Your job is to select the best combination of models for the nodes.\n",
            # Objective
            "# The objective to target when selecting the model combination:\n"
            f"{self.objective}\n",
            # Agent Pipeline
            "# Agent Pipeline\n"
            "The agent has the following nodes and each can be assigned one of its candidate models.\n"
            f"```json\n{json.dumps(nodes_info, indent=2, ensure_ascii=True)}\n```\n",
            # Dataset Preview
            "# Dataset Preview\n"
            "Below are sample inputs and their expected outputs. Use these to understand "
            "the task complexity and choose models accordingly.\n"
            f"```json\n{json.dumps(preview, indent=2, ensure_ascii=True)}\n```\n",
            # Response Format
            "# Response Format\n"
            "Respond with a JSON object like this example:\n"
            f"```json\n{json.dumps(example, ensure_ascii=True)}\n```\n",
            # Constraints
            "# Constraints\n"
            "- Each key in `combination` must be a node name from the pipeline above.\n"
            "- Each value must be a candidate model name from that node's candidates list.\n"
            "- All nodes must be included.\n"
            "- Return exactly one combination.\n",
        ]

        prompt = "\n".join(sections)

        return prompt

    # ------------------------------------------------------------------
    # Parsing & proposer
    # ------------------------------------------------------------------

    def _parse_proposed_combination(self, text: str,) -> Optional[Tuple[int, ...]]:
        if not text.strip():
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "LMProposalModelSelector: proposer returned non-JSON output."
            )
            return None

        try:
            response = ProposalResponse.model_validate(payload)
        except ValidationError as e:
            logger.warning("LMProposalModelSelector: invalid response structure: %s", e)
            return None

        if set(response.combination.keys()) != set(self._node_names):
            logger.warning(
                "LMProposalModelSelector: response nodes don't match pipeline nodes."
            )
            return None

        indices: List[int] = []
        for node in self._node_names:
            model_name = response.combination[node]
            lookup = self._label_to_index.get(node, {})
            if model_name not in lookup:
                logger.warning(
                    "LMProposalModelSelector: unknown model '%s' for node '%s'.",
                    model_name,
                    node,
                )
                return None
            indices.append(lookup[model_name])

        return tuple(indices)

    def _ask_proposer(self, max_retries: int = 3) -> Optional[Tuple[int, ...]]:
        preview = self._dataset_preview()
        prompt = self._build_prompt(preview)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert model-selection assistant. "
                    "Analyze the agent pipeline, candidate models, and dataset, "
                    "then return a single JSON object with your recommended "
                    "model combination."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = self.proposer_client.chat.completions.create(
                    model=self.proposer_model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                raw = response.choices[0].message.content or ""
                proposed = self._parse_proposed_combination(raw)
                if proposed is not None:
                    return proposed
                logger.warning(
                    "LM proposer attempt %d/%d: invalid response, retrying...",
                    attempt,
                    max_retries,
                )
            except Exception as e:
                logger.warning(
                    "LM proposer attempt %d/%d failed: %s", attempt, max_retries, e,
                )

        logger.warning(
            "LM proposer exhausted all %d retries; falling back to defaults.",
            max_retries,
        )
        return None
