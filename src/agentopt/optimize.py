"""
High-level optimization API for finding the best model for an agent.

Provides two public functions:
- optimize(): Simple API — no ModelProxy needed, parallel by default
- parallel_select(): Advanced API — works with ModelProxy, parallel evaluation

Agent copies are created via Pydantic model_copy(deep=False) with an LLM update,
then evaluated concurrently via ThreadPoolExecutor.
"""

import asyncio
import copy
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from .base_models import EvalFn
from .model_proxy import ModelProxy
from .model_selection.base import ModelResult, SelectionResults

logger = logging.getLogger(__name__)

# Model name fields to check on LLM objects, in priority order
_MODEL_FIELDS = ("model", "model_name", "model_id")

# Attribute names to check for the LLM on an agent, in priority order
_AGENT_LLM_ATTRS = ("llm",)


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
    original_llm, llm_attr = _find_llm(agent)
    if original_llm is None:
        raise ValueError(
            f"Could not find an LLM attribute on {type(agent).__name__}. "
            f"Checked: {_AGENT_LLM_ATTRS}. "
            "Use ModelProxy + ModelSelector for custom agent structures."
        )

    invoke_method_name, is_async = _resolve_invoke(agent)

    if max_workers is None:
        max_workers = len(models)

    mode = "parallel" if parallel else "sequential"
    print(f"\n{'='*60}")
    print(f"optimize ({mode}): {len(models)} models")
    print(f"{'='*60}\n")

    if parallel:
        try:
            return _optimize_parallel(
                agent, original_llm, llm_attr, models,
                invoke_method_name, is_async, eval_fn, dataset, max_workers,
            )
        except RuntimeError as e:
            logger.warning("Parallel failed, falling back to sequential: %s", e)

    return _optimize_sequential(
        agent, original_llm, llm_attr, models,
        invoke_method_name, is_async, eval_fn, dataset,
    )


def parallel_select(
    models: Dict[ModelProxy, List[Any]],
    agent: Any,
    eval_fn: EvalFn,
    dataset: List[Tuple[Any, str]],
    max_workers: Optional[int] = None,
) -> SelectionResults:
    """Parallel model selection using ModelProxy — like ModelSelector but concurrent.

    Same interface as ModelSelector but evaluates all model combinations
    concurrently using threads.

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
    proxies = list(models.keys())
    candidate_lists = list(models.values())
    all_combinations = list(itertools.product(*candidate_lists))

    if max_workers is None:
        max_workers = len(all_combinations)

    # Detect which agent attribute each proxy corresponds to
    proxy_to_attr = _detect_proxy_attrs(agent, proxies)

    invoke_method_name, is_async = _resolve_invoke(agent)

    print(f"\n{'='*60}")
    print(f"parallel_select: {len(all_combinations)} combinations, {max_workers} workers")
    print(f"{'='*60}\n")

    # Phase 1: Create agent copies (sequential)
    tasks = []
    for combo in all_combinations:
        combo_name = " + ".join(_get_model_name(m) for m in combo)

        # Build LLM update dict for this combination
        llm_updates = {}
        for proxy, model_spec in zip(proxies, combo):
            attr = proxy_to_attr.get(id(proxy))
            if attr is None:
                continue
            if isinstance(model_spec, str):
                fresh = _create_variant(proxy.get_model(), model_spec)
            else:
                fresh = model_spec
            llm_updates[attr] = fresh

        try:
            agent_copy = _clone_agent(agent, llm_updates)
        except Exception as e:
            logger.warning("Clone failed for [%s], skipping: %s", combo_name, e)
            continue

        invoke_fn = _make_invoke_fn(agent_copy, invoke_method_name, is_async)
        tasks.append((combo_name, combo, invoke_fn, is_async))

    if not tasks:
        raise RuntimeError("All clone attempts failed. Cannot evaluate any models.")

    # Phase 2: Evaluate in parallel
    all_results: List[ModelResult] = []
    future_to_info: Dict[Any, Tuple[str, tuple]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for combo_name, combo, invoke_fn, is_async_flag in tasks:
            future = executor.submit(
                _evaluate_single, invoke_fn, is_async_flag, eval_fn, dataset,
            )
            future_to_info[future] = (combo_name, combo)

        for future in as_completed(future_to_info):
            combo_name, combo = future_to_info[future]
            try:
                accuracy, latency = future.result()
                print(f"  [{combo_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s")
                all_results.append(ModelResult(
                    model_name=combo_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    attribute="combination",
                    is_best=False,
                ))
            except Exception as e:
                print(f"  [{combo_name}] failed: {e}")
                all_results.append(ModelResult(
                    model_name=combo_name,
                    accuracy=0.0,
                    latency_seconds=0.0,
                    attribute="combination",
                    is_best=False,
                ))

    # Phase 3: Determine best and set proxies
    best_info = _find_best(all_results)
    if best_info is not None:
        best_name, _ = best_info
        for combo_name, combo, _, _ in tasks:
            if combo_name == best_name:
                for proxy, model_obj in zip(proxies, combo):
                    proxy.set_model(model_obj)
                break
        for r in all_results:
            if r.model_name == best_name:
                r.is_best = True
                break
        print(f"\n  Best: {best_name}")
    else:
        print("\n  No combinations succeeded")

    return SelectionResults(results=all_results)


# ---------------------------------------------------------------------------
# Private: Agent cloning
# ---------------------------------------------------------------------------


def _clone_agent(agent: Any, llm_updates: Dict[str, Any]) -> Any:
    """Create an independent copy of the agent with LLMs replaced.

    Strategy:
    - Pydantic agents: model_copy(deep=False) with update — fast, avoids
      deepcopy of HTTP clients with thread locks.
    - Other agents: copy.deepcopy fallback.

    Args:
        agent: The agent to clone.
        llm_updates: Dict of {attr_name: new_llm} to set on the copy.

    Returns:
        A shallow copy with the specified LLMs replaced.
    """
    if isinstance(agent, BaseModel):
        return agent.model_copy(update=llm_updates, deep=False)
    else:
        # Non-Pydantic fallback: deepcopy then set attributes
        agent_copy = copy.deepcopy(agent)
        for attr, llm in llm_updates.items():
            setattr(agent_copy, attr, llm)
        return agent_copy


def _detect_proxy_attrs(
    agent: Any,
    proxies: List[ModelProxy],
) -> Dict[int, str]:
    """Detect which agent attribute each proxy corresponds to.

    Checks common LLM attribute names on the agent and matches them
    to the provided proxies by identity (is) comparison.

    Returns:
        Dict mapping id(proxy) -> attribute name on agent.
    """
    mapping: Dict[int, str] = {}
    for proxy in proxies:
        proxy_model = proxy.get_model()
        for attr in _AGENT_LLM_ATTRS:
            if hasattr(agent, attr):
                val = getattr(agent, attr)
                if val is proxy or val is proxy_model:
                    mapping[id(proxy)] = attr
                    break
    return mapping


# ---------------------------------------------------------------------------
# Private: LLM handling
# ---------------------------------------------------------------------------


def _find_llm(agent: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Find the LLM object and its attribute name on the agent."""
    for attr in _AGENT_LLM_ATTRS:
        if hasattr(agent, attr):
            llm = getattr(agent, attr)
            if llm is not None:
                return llm, attr
    return None, None


def _resolve_invoke(agent: Any) -> Tuple[str, bool]:
    """Determine which method to call on the agent and whether it's async."""
    if hasattr(agent, "kickoff"):
        return "kickoff", False
    elif hasattr(agent, "invoke"):
        return "invoke", False
    elif hasattr(agent, "run"):
        # Always False: _make_invoke_fn wraps .run() in asyncio.run() internally.
        # LlamaIndex's Workflow.run() needs a running event loop but
        # inspect.iscoroutinefunction may return False due to instrumentation
        # decorators, so we always provide our own event loop wrapper.
        return "run", False
    else:
        raise TypeError(
            f"Unsupported agent type: {type(agent).__name__}. "
            "Agent must have .kickoff(), .invoke(), or .run() method."
        )


def _create_variant(original_llm: Any, model_name: str) -> Any:
    """Create a fresh LLM with a different model name, preserving all other settings.

    Uses model_copy for Pydantic LLMs (avoids HTTP client issues),
    copy.deepcopy for non-Pydantic LLMs.
    """
    if isinstance(original_llm, BaseModel):
        target_field = next(
            (f for f in _MODEL_FIELDS if f in type(original_llm).model_fields),
            None,
        )
        if target_field:
            return original_llm.model_copy(update={target_field: model_name})
    else:
        target_field = next(
            (f for f in _MODEL_FIELDS if hasattr(original_llm, f)),
            None,
        )
        if target_field:
            new_llm = copy.deepcopy(original_llm)
            setattr(new_llm, target_field, model_name)
            return new_llm

    raise TypeError(
        f"Cannot create variant of {type(original_llm).__name__}: "
        f"no supported model field found (checked: {_MODEL_FIELDS})"
    )


# ---------------------------------------------------------------------------
# Private: Invocation
# ---------------------------------------------------------------------------


def _make_invoke_fn(agent_copy: Any, method_name: str, is_async: bool):
    """Create an invocation callable for the copied agent.

    For .run() (LlamaIndex), wraps call in asyncio.run() with an async wrapper.
    LlamaIndex's Workflow.run() internally uses asyncio.create_task() which
    requires a running event loop, even though iscoroutinefunction returns False
    due to instrumentation decorators.

    For .kickoff() and .invoke(), passes input as-is.
    """
    method = getattr(agent_copy, method_name)

    if method_name == "run":
        def _invoke(input_data):
            async def _async_run():
                if isinstance(input_data, dict):
                    result = method(**input_data)
                else:
                    result = method(input_data)
                # Handle both truly async and sync-that-returns-awaitable
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    return await result
                return result
            return asyncio.run(_async_run())
        return _invoke
    else:
        return method


# ---------------------------------------------------------------------------
# Private: Evaluation
# ---------------------------------------------------------------------------


def _evaluate_single(
    invoke_fn,
    is_async: bool,
    eval_fn: EvalFn,
    dataset: List[Tuple[Any, str]],
) -> Tuple[float, float]:
    """Evaluate an agent against the dataset. Thread-safe.

    Each thread creates its own event loop via asyncio.run() for async agents.
    """
    total_score = 0.0
    total = len(dataset)
    total_latency = 0.0

    for input_data, expected_answer in dataset:
        try:
            start_time = time.time()
            if is_async:
                actual_result = asyncio.run(invoke_fn(input_data))
            else:
                actual_result = invoke_fn(input_data)
            latency = time.time() - start_time
            total_latency += latency
            score = eval_fn(expected_answer, actual_result)
            total_score += float(score)
        except Exception as e:
            logger.debug("Evaluation error: %s", e)

    avg_score = total_score / total if total > 0 else 0.0
    avg_latency = total_latency / total if total > 0 else 0.0
    return avg_score, avg_latency


def _get_model_name(model_obj: Any) -> str:
    """Extract a display name from a model object or string."""
    if isinstance(model_obj, str):
        return model_obj
    elif hasattr(model_obj, "model_name"):
        return str(model_obj.model_name)
    elif hasattr(model_obj, "model"):
        return str(model_obj.model)
    else:
        return model_obj.__class__.__name__


def _find_best(results: List[ModelResult]) -> Optional[Tuple[str, float]]:
    """Find the best result by accuracy (ties broken by latency)."""
    best = None
    best_accuracy = float("-inf")
    best_latency = float("inf")
    tol = 1e-9

    for r in results:
        if r.accuracy > best_accuracy + tol:
            best = r
            best_accuracy = r.accuracy
            best_latency = r.latency_seconds
        elif abs(r.accuracy - best_accuracy) <= tol and r.latency_seconds < best_latency:
            best = r
            best_accuracy = r.accuracy
            best_latency = r.latency_seconds

    return (best.model_name, best.accuracy) if best else None


# ---------------------------------------------------------------------------
# Private: Optimization paths
# ---------------------------------------------------------------------------


def _optimize_parallel(
    agent, original_llm, llm_attr, models,
    invoke_method_name, is_async, eval_fn, dataset, max_workers,
) -> SelectionResults:
    """Parallel: create agent copies via model_copy, evaluate with ThreadPoolExecutor."""
    tasks = []

    for model_spec in models:
        model_name = model_spec if isinstance(model_spec, str) else _get_model_name(model_spec)

        if isinstance(model_spec, str):
            fresh_llm = _create_variant(original_llm, model_spec)
        else:
            fresh_llm = model_spec

        try:
            agent_copy = _clone_agent(agent, {llm_attr: fresh_llm})
        except Exception as e:
            logger.warning("Clone failed for %s, skipping: %s", model_name, e)
            continue

        invoke_fn = _make_invoke_fn(agent_copy, invoke_method_name, is_async)
        tasks.append((model_name, invoke_fn, is_async))

    if not tasks:
        raise RuntimeError("All clone attempts failed. Cannot evaluate any models.")

    all_results: List[ModelResult] = []
    future_to_name: Dict[Any, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_name, invoke_fn, is_async_flag in tasks:
            future = executor.submit(
                _evaluate_single, invoke_fn, is_async_flag, eval_fn, dataset,
            )
            future_to_name[future] = model_name

        for future in as_completed(future_to_name):
            model_name = future_to_name[future]
            try:
                accuracy, latency = future.result()
                print(f"  [{model_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s")
                all_results.append(ModelResult(
                    model_name=model_name,
                    accuracy=accuracy,
                    latency_seconds=latency,
                    attribute="model",
                    is_best=False,
                ))
            except Exception as e:
                print(f"  [{model_name}] failed: {e}")
                all_results.append(ModelResult(
                    model_name=model_name,
                    accuracy=0.0,
                    latency_seconds=0.0,
                    attribute="model",
                    is_best=False,
                ))

    return _finalize_optimize_results(all_results, agent, original_llm, llm_attr)


def _optimize_sequential(
    agent, original_llm, llm_attr, models,
    invoke_method_name, is_async, eval_fn, dataset,
) -> SelectionResults:
    """Sequential: swap LLM in place, evaluate, repeat."""
    all_results: List[ModelResult] = []

    for model_spec in models:
        model_name = model_spec if isinstance(model_spec, str) else _get_model_name(model_spec)

        if isinstance(model_spec, str):
            fresh_llm = _create_variant(original_llm, model_spec)
        else:
            fresh_llm = model_spec

        setattr(agent, llm_attr, fresh_llm)
        invoke_fn = _make_invoke_fn(agent, invoke_method_name, is_async)

        try:
            accuracy, latency = _evaluate_single(
                invoke_fn, is_async, eval_fn, dataset,
            )
            print(f"  [{model_name}] Accuracy: {accuracy:.2%}, Latency: {latency:.2f}s")
            all_results.append(ModelResult(
                model_name=model_name,
                accuracy=accuracy,
                latency_seconds=latency,
                attribute="model",
                is_best=False,
            ))
        except Exception as e:
            print(f"  [{model_name}] failed: {e}")
            all_results.append(ModelResult(
                model_name=model_name,
                accuracy=0.0,
                latency_seconds=0.0,
                attribute="model",
                is_best=False,
            ))

    return _finalize_optimize_results(all_results, agent, original_llm, llm_attr)


def _finalize_optimize_results(
    all_results: List[ModelResult],
    agent: Any,
    original_llm: Any,
    llm_attr: str,
) -> SelectionResults:
    """Mark the best result and set the agent's LLM to the winner."""
    best_info = _find_best(all_results)

    if best_info is not None:
        best_name, _ = best_info
        for r in all_results:
            if r.model_name == best_name:
                r.is_best = True
                try:
                    best_llm = _create_variant(original_llm, best_name)
                    setattr(agent, llm_attr, best_llm)
                except Exception:
                    pass
                print(
                    f"\n  Best: {best_name} "
                    f"(accuracy: {r.accuracy:.2%}, latency: {r.latency_seconds:.2f}s)"
                )
                break
    else:
        print("\n  No models succeeded")

    return SelectionResults(results=all_results)