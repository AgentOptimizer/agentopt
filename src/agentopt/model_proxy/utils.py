"""
Proxy utilities: field operations, agent cloning, proxy discovery, variant creation.
"""

import asyncio
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .base import ModelProxy

logger = logging.getLogger(__name__)

# Model name fields to check on LLM objects, in priority order
MODEL_FIELDS = ("model", "model_name", "model_id")


def get_model_field_value(obj: Any) -> str:
    """Extract the current model name string from an LLM object."""
    for field in MODEL_FIELDS:
        val = getattr(obj, field, None)
        if val is not None:
            return str(val)
    return ""


def resolve_model_name(model_spec: Any, reference_obj: Any) -> str:
    """Resolve a model spec to the name format the stored object expects.

    Frameworks like CrewAI strip provider prefixes (``"openai/gpt-4o-mini"``
    -> ``"gpt-4o-mini"``).  This function detects that and applies the same
    stripping to new candidate strings.
    """
    if not isinstance(model_spec, str):
        return get_model_field_value(model_spec)
    stored_name = get_model_field_value(reference_obj)
    if "/" not in stored_name and "/" in model_spec:
        return model_spec.split("/", 1)[1]
    return model_spec


def set_model_field(obj: Any, model_name: str) -> None:
    """Set the model name field on an LLM object, bypassing Pydantic frozen fields."""
    for field in MODEL_FIELDS:
        if isinstance(obj, BaseModel):
            if field in type(obj).model_fields:
                object.__setattr__(obj, field, model_name)
                return
        elif hasattr(obj, field):
            setattr(obj, field, model_name)
            return


def get_model_name(model_obj: Any) -> str:
    """Extract model name from a model object for display purposes."""
    if isinstance(model_obj, str):
        return model_obj
    elif hasattr(model_obj, "model_name"):
        return str(model_obj.model_name)
    elif hasattr(model_obj, "model"):
        return str(model_obj.model)
    else:
        return model_obj.__class__.__name__


# ------------------------------------------------------------------
# Agent cloning utilities
# ------------------------------------------------------------------


def clone_agent(agent: Any, proxy_replacements: Dict[int, Any]) -> Any:
    """Create an independent copy of the agent with proxies replaced.

    Recursively walks the Pydantic object tree to find and replace
    ModelProxy instances (matched by ``id``) at any nesting level.
    Uses ``model_copy(deep=False)`` to avoid deep-copying HTTP clients.

    Args:
        agent: The agent (e.g. Crew) to clone.
        proxy_replacements: ``{id(proxy): fresh_llm, ...}`` mapping
            each proxy (and its underlying model) to the replacement.
    """
    if not proxy_replacements:
        if isinstance(agent, BaseModel):
            return agent.model_copy(deep=False)
        return copy.deepcopy(agent)

    if isinstance(agent, BaseModel):
        return copy_with_replacements(agent, proxy_replacements, {})
    else:
        # Non-Pydantic fallback: deepcopy, then replace by scanning
        # the *original* attrs (whose ids still match the replacements).
        agent_copy = copy.deepcopy(agent)
        for attr_name in list(vars(agent)):
            val = getattr(agent, attr_name, None)
            if id(val) in proxy_replacements:
                setattr(agent_copy, attr_name, proxy_replacements[id(val)])
        return agent_copy


def copy_with_replacements(
    obj: Any,
    replacements: Dict[int, Any],
    memo: Dict[int, Any],
) -> Any:
    """Recursively copy *obj*, swapping any value whose ``id`` is in
    *replacements*.  Only recurses into Pydantic ``BaseModel``
    instances, ``list``s, and ``dict``s -- everything else is returned
    as-is (shared, not deep-copied).

    *memo* caches already-processed objects so that shared references
    (e.g. the same Agent in ``crew.agents`` and ``task.agent``) map to
    the same copy.
    """
    obj_id = id(obj)

    # Direct replacement (proxy or underlying model)
    if obj_id in replacements:
        return replacements[obj_id]

    # Already processed -- return cached result
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, BaseModel):
        memo[obj_id] = obj  # placeholder to break cycles
        updates: Dict[str, Any] = {}
        for field_name in type(obj).model_fields:
            val = getattr(obj, field_name, None)
            if val is None:
                continue
            new_val = copy_with_replacements(val, replacements, memo)
            if new_val is not val:
                updates[field_name] = new_val
        if updates:
            result = obj.model_copy(update=updates, deep=False)
            memo[obj_id] = result
            return result
        return obj

    if isinstance(obj, list):
        new_list = []
        changed = False
        for item in obj:
            new_item = copy_with_replacements(item, replacements, memo)
            new_list.append(new_item)
            if new_item is not item:
                changed = True
        if changed:
            memo[obj_id] = new_list
            return new_list
        return obj

    if isinstance(obj, dict):
        new_dict = {}
        changed = False
        for k, v in obj.items():
            new_v = copy_with_replacements(v, replacements, memo)
            new_dict[k] = new_v
            if new_v is not v:
                changed = True
        if changed:
            memo[obj_id] = new_dict
            return new_dict
        return obj

    return obj


def create_variant(original_llm: Any, model_name: str) -> Any:
    """Create a fresh LLM with a different model name, preserving settings.

    Strategy chain (first success wins):
    1. Framework-specific factory (e.g. CrewAI's ``LLM(model=...)``)
    2. Full reconstruction via ``type(original)(**fields)``
    3. Pydantic ``model_copy`` / shallow-copy fallback
    """
    # Strategy 1: Framework-specific factory
    module = getattr(type(original_llm), "__module__", "") or ""
    if "crewai" in module:
        try:
            from .crewai import create_variant as _crewai_create

            return _crewai_create(original_llm, model_name)
        except (ImportError, Exception):
            pass

    # Strategy 2+3: Generic Pydantic path
    if isinstance(original_llm, BaseModel):
        target_field = next(
            (f for f in MODEL_FIELDS if f in type(original_llm).model_fields),
            None,
        )
        if target_field:
            # Try full construction first (runs __init__ + validators)
            try:
                kwargs = {}
                for field in type(original_llm).model_fields:
                    try:
                        kwargs[field] = getattr(original_llm, field)
                    except AttributeError:
                        pass
                kwargs[target_field] = model_name
                return type(original_llm)(**kwargs)
            except Exception:
                pass
            # Fallback: model_copy (no __init__, may break some objects)
            return original_llm.model_copy(update={target_field: model_name})
    else:
        # Non-Pydantic fallback
        target_field = next(
            (f for f in MODEL_FIELDS if hasattr(original_llm, f)),
            None,
        )
        if target_field:
            new_llm = copy.copy(original_llm)
            setattr(new_llm, target_field, model_name)
            return new_llm

    raise TypeError(
        f"Cannot create variant of {type(original_llm).__name__}: "
        f"no supported model field found (checked: {MODEL_FIELDS})"
    )


def discover_proxy_targets(
    agent: Any,
    proxies: list,
) -> Dict[int, Tuple[Any, Any]]:
    """Walk the agent tree to find stored objects that correspond to proxies.

    Frameworks like CrewAI transform ``ModelProxy`` during Pydantic
    validation, creating new LLM objects with different ``id``s.  This
    method discovers those objects so their ``id``s can be added to the
    replacement map used by ``copy_with_replacements``.

    Returns:
        ``{id(stored_obj): (stored_obj, proxy), ...}`` for each
        discovered object.  The *stored_obj* reference is kept so
        callers can create variants from the properly-initialised
        object rather than from ``proxy.get_model()``.
    """
    # ids we already know about -- skip during scanning
    known_ids: set = set()
    for proxy in proxies:
        known_ids.add(id(proxy))
        known_ids.add(id(proxy.get_model()))

    proxy_model_types = {id(p): type(p.get_model()) for p in proxies}
    unmatched = list(proxies)  # consumed in tree-walk order
    result: Dict[int, Tuple[Any, Any]] = {}

    def _scan(obj: Any, visited: set) -> None:
        obj_id = id(obj)
        if obj_id in visited or obj_id in known_ids:
            return
        visited.add(obj_id)

        # Match by type against unmatched proxies (order-preserving)
        for i, proxy in enumerate(unmatched):
            if type(obj) is proxy_model_types[id(proxy)]:
                result[obj_id] = (obj, proxy)
                unmatched.pop(i)
                return  # don't recurse into matched objects

        if isinstance(obj, BaseModel):
            for field_name in type(obj).model_fields:
                val = getattr(obj, field_name, None)
                if val is not None:
                    _scan(val, visited)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item, visited)
        elif isinstance(obj, dict):
            for v in obj.values():
                _scan(v, visited)

    _scan(agent, set())
    return result


def make_invoke_fn(agent_copy: Any, method_name: str) -> Callable:
    """Create a **synchronous** invocation callable for a copied agent.

    For .run() (LlamaIndex -- typically async), wraps the call in
    ``asyncio.run()`` so the caller can treat it as synchronous.
    For .kickoff() and .invoke(), returns the method directly.
    """
    method = getattr(agent_copy, method_name)

    if method_name == "run":

        def _sync_invoke(input_data):
            async def _run_async():
                if isinstance(input_data, dict):
                    result = method(**input_data)
                else:
                    result = method(input_data)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    return await result
                return result

            return asyncio.run(_run_async())

        return _sync_invoke
    else:
        return method
