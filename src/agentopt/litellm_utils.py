"""
LiteLLM integration utilities.

Provides helpers for reading token usage logs, aggregating per-model
token usage, exporting optimized routing configs, and a proxy callback
that logs per-request token usage to a JSONL file.

Register the callback in your ``litellm_config.yaml``::

    litellm_settings:
      callbacks: token_logger.proxy_handler_instance

The log file path defaults to ``.cache/token_usage.jsonl`` and can be
overridden with the ``TOKEN_LOG_FILE`` environment variable.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litellm.integrations.custom_logger import CustomLogger

_DEFAULT_LOG_FILE = ".cache/token_usage.jsonl"


# ---------------------------------------------------------------------------
# Proxy callback: writes token usage to JSONL
# ---------------------------------------------------------------------------


class TokenFileLogger(CustomLogger):
    """Writes per-request token usage to a JSONL file."""

    def __init__(self) -> None:
        super().__init__()
        self._log_file = os.environ.get("TOKEN_LOG_FILE", _DEFAULT_LOG_FILE)
        self._lock = threading.Lock()
        Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            usage = getattr(response_obj, "usage", None)
            if usage is None:
                return
            entry = {
                "model": kwargs.get("model", "unknown"),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "timestamp": start_time.isoformat() if start_time else "",
            }
            with self._lock:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Never break the proxy for logging failures

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)


# Module-level instance for litellm callback registration.
proxy_handler_instance = TokenFileLogger()


# ---------------------------------------------------------------------------
# Client-side helpers
# ---------------------------------------------------------------------------


def read_token_log(
    log_file: str = _DEFAULT_LOG_FILE, since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read token usage entries from the JSONL log file.

    Args:
        log_file: Path to the JSONL token log file.
        since: Optional ISO-format timestamp. Only entries after this
            time are returned.

    Returns:
        List of log entries (dicts with model, prompt_tokens,
        completion_tokens, timestamp).
    """
    path = Path(log_file)
    if not path.is_file():
        return []

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            pass

    entries: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_dt and entry.get("timestamp"):
                try:
                    entry_dt = datetime.fromisoformat(entry["timestamp"])
                    if entry_dt < since_dt:
                        continue
                except ValueError:
                    pass
            entries.append(entry)
    return entries


def aggregate_tokens(logs: List[Dict[str, Any]],) -> Dict[str, Tuple[int, int]]:
    """Aggregate spend logs into per-model token totals.

    Args:
        logs: List of spend log entries from LiteLLM.

    Returns:
        Dict mapping model name to (total_input_tokens, total_output_tokens).
    """
    totals: Dict[str, List[int]] = {}
    for entry in logs:
        model = entry.get("model", "unknown")
        prompt_tokens = entry.get("prompt_tokens", 0) or 0
        completion_tokens = entry.get("completion_tokens", 0) or 0
        if model not in totals:
            totals[model] = [0, 0]
        totals[model][0] += prompt_tokens
        totals[model][1] += completion_tokens
    return {k: (v[0], v[1]) for k, v in totals.items()}


def export_litellm_config(
    best_combo: Dict[str, str],
    output_path: str,
    api_key_env_vars: Optional[Dict[str, str]] = None,
) -> None:
    """Export an optimized LiteLLM config YAML.

    Args:
        best_combo: Best model mapping, e.g. ``{"planner": "gpt-4o", "solver": "gpt-4o-mini"}``.
        output_path: Path to write the YAML config file.
        api_key_env_vars: Optional mapping of provider prefix to env var name.
            Defaults to common providers.
    """
    if api_key_env_vars is None:
        api_key_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

    def _detect_provider(model_name: str) -> str:
        if "/" in model_name:
            return model_name.split("/")[0]
        # Guess provider from model name
        lower = model_name.lower()
        if "gpt" in lower or lower.startswith("o3") or lower.startswith("o4"):
            return "openai"
        elif "claude" in lower:
            return "anthropic"
        elif "gemini" in lower:
            return "google"
        return "openai"

    def _full_model_name(model_name: str) -> str:
        if "/" in model_name:
            return model_name
        provider = _detect_provider(model_name)
        return f"{provider}/{model_name}"

    # Deduplicate models
    unique_models = set(best_combo.values())

    lines = ["model_list:"]
    for model in sorted(unique_models):
        full_name = _full_model_name(model)
        provider = _detect_provider(model)
        env_var = api_key_env_vars.get(provider, "API_KEY")
        lines.append(f"  - model_name: {model}")
        lines.append(f"    litellm_params:")
        lines.append(f"      model: {full_name}")
        lines.append(f"      api_key: os.environ/{env_var}")
        lines.append("")

    lines.append("# Optimized model mapping (from agentopt):")
    for node, model in best_combo.items():
        lines.append(f"#   {node}: {model}")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
