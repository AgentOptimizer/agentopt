"""
High-level optimization API for finding the best model for an agent.

Provides two convenience functions that delegate to model selection algorithms:
- optimize(): Simple API — no ModelProxy needed, parallel by default
- parallel_select(): Advanced API — works with ModelProxy, parallel evaluation

Both functions internally create a BruteForceModelSelector and call
select_best(parallel=True).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from .base_models import EvalFn
from .model_proxy import ModelProxy
from .model_selection.base import SelectionResults, _AGENT_LLM_ATTRS, _MODEL_FIELDS
from .model_selection.brute_force import BruteForceModelSelector


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize(
    agent: Any,
    models: List[Union[str, Any]],
    eval_fn: EvalFn,
    dataset: List[Tuple[Any, str]],
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> SelectionResults:
    """Find the best model for an agent by evaluating candidates on a dataset.

    This is the simplest API — no ModelProxy or invoke_fn needed. Just pass
    your agent, candidate model names, evaluation function, and dataset.

    Internally creates a ModelProxy and delegates to BruteForceModelSelector.

    Args:
        agent: Agent object with .kickoff() (CrewAI), .invoke() (LangChain),
               or .run() (LlamaIndex) method, and an .llm attribute.
        models: List of model name strings (e.g., ["gpt-4o-mini", "gpt-4o"])
                or pre-built LLM objects.
        eval_fn: Evaluation function (expected: str, actual: Any) -> bool | float.
        dataset: List of (input_data, expected_answer) tuples.
        parallel: If True (default), evaluate models concurrently using threads.
                  Falls back to sequential if agent cannot be copied.
        max_workers: Max threads for parallel mode. Defaults to len(models).

    Returns:
        SelectionResults with accuracy, latency, and best model identified.
        After returning, the agent's LLM is set to the best model.

    Example::

        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.llms.openai import OpenAI
        from agentopt import optimize

        agent = FunctionAgent(
            tools=[...],
            llm=OpenAI(model="gpt-4o-mini"),
            system_prompt="You are a helpful assistant.",
        )

        results = optimize(
            agent=agent,
            models=["gpt-4o-mini", "gpt-4o"],
            eval_fn=lambda expected, actual: expected.lower() in str(actual).lower(),
            dataset=[({"user_msg": "What is 2+2?"}, "4"), ...],
        )
        print(results.get_best())
    """
    # Find the LLM attribute on the agent
    original_llm, llm_attr = _find_llm(agent)
    if original_llm is None:
        raise ValueError(
            f"Could not find an LLM attribute on {type(agent).__name__}. "
            f"Checked: {_AGENT_LLM_ATTRS}. "
            "Use ModelProxy + ModelSelector for custom agent structures."
        )

    # Wrap the LLM in a proxy so we can use the standard selector API
    proxy = ModelProxy(original_llm)
    setattr(agent, llm_attr, proxy)

    # Build candidate list — convert strings to model specs
    candidate_models: List[Any] = list(models)

    selector = BruteForceModelSelector(
        models={proxy: candidate_models},
        eval_fn=eval_fn,
        dataset=dataset,
        agent=agent,
    )

    results = selector.select_best(
        parallel=parallel,
        max_workers=max_workers if max_workers is not None else len(models),
    )

    # Unwrap: set the agent's LLM to the best model directly (not the proxy)
    best = results.get_best()
    if best is not None:
        setattr(agent, llm_attr, proxy.get_model())

    return results


def parallel_select(
    models: Dict[ModelProxy, List[Any]],
    agent: Any,
    eval_fn: EvalFn,
    dataset: List[Tuple[Any, str]],
    max_workers: Optional[int] = None,
) -> SelectionResults:
    """Parallel model selection using ModelProxy.

    Same interface as ModelSelector but evaluates all model combinations
    concurrently using threads. Delegates to BruteForceModelSelector
    with parallel=True.

    Args:
        models: Dict mapping ModelProxy to list of candidate models (strings or objects).
        agent: Agent object with .kickoff(), .invoke(), or .run() method.
        eval_fn: Evaluation function (expected: str, actual: Any) -> bool | float.
        dataset: List of (input_data, expected_answer) tuples.
        max_workers: Max threads. Defaults to number of combinations.

    Returns:
        SelectionResults with accuracy, latency, and best model identified.
        After returning, proxies are set to the best combination.

    Example::

        from agentopt import ModelProxy, parallel_select

        proxy = ModelProxy(LLM(model="openai/gpt-4o-mini"))
        agent = Agent(llm=proxy, ...)
        crew = Crew(agents=[agent], ...)

        results = parallel_select(
            models={proxy: ["openai/gpt-4o-mini", "openai/gpt-4o"]},
            agent=crew,
            eval_fn=eval_fn,
            dataset=dataset,
        )
    """
    selector = BruteForceModelSelector(
        models=models,
        eval_fn=eval_fn,
        dataset=dataset,
        agent=agent,
    )

    return selector.select_best(parallel=True, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_llm(agent: Any) -> Tuple[Any, Optional[str]]:
    """Find the LLM object and its attribute name on the agent."""
    for attr in _AGENT_LLM_ATTRS:
        if hasattr(agent, attr):
            llm = getattr(agent, attr)
            if llm is not None:
                return llm, attr
    return None, None