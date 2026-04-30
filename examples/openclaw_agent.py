"""OpenClaw adapter for agentopt model selection — example, not part of the public API.

Provides :class:`OpenClawAgent` — an agent wrapper that runs
``openclaw infer model run`` as a subprocess and routes all LLM calls
through agentopt's tracking proxy via config-file patching.

OpenClaw's gateway daemon ignores ``HTTPS_PROXY`` env vars, so this
adapter patches ``~/.openclaw/openclaw.json`` with an explicit proxy URL
and inline CA certificate before each inference call, then restores the
original config afterward.

Usage::

    from agentopt import ModelSelector
    from openclaw_agent import OpenClawAgent

    selector = ModelSelector(
        agent=OpenClawAgent,
        models={"agent": ["anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"]},
        eval_fn=my_eval_fn,
        dataset=my_dataset,
        method="brute_force",
    )
    results = selector.select_best(parallel=False)

Note: Use ``parallel=False`` — config patching is not safe for
concurrent evaluation (single shared config file).
"""

import json
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

from agentopt import get_current_session_proxy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENCLAW_CONFIG = os.path.expanduser("~/.openclaw/openclaw.json")
OPENCLAW_BACKUP = OPENCLAW_CONFIG + ".bak.agentopt"
OPENCLAW_TIMEOUT = 120  # seconds per inference call

# Known provider base URLs. Used only when adding a provider that doesn't
# already exist in the user's OpenClaw config.
PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "anthropic": {"baseUrl": "https://api.anthropic.com"},
    "openai": {"baseUrl": "https://api.openai.com/v1"},
    "google": {"baseUrl": "https://generativelanguage.googleapis.com/v1beta"},
    "mistral": {"baseUrl": "https://api.mistral.ai/v1"},
}


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------


def patch_openclaw_config(proxy_url: str, ca_pem: str, model: str) -> dict:
    """Patch OpenClaw config to route LLM calls through the agentopt proxy.

    Sets ``explicit-proxy`` mode with an inline CA cert so OpenClaw trusts
    the proxy's TLS certificates for CONNECT-mode interception.  Also adds
    the model to the provider's model list and the agent allowlist so
    ``openclaw infer`` accepts ``--model`` overrides.

    Args:
        proxy_url: The proxy URL, e.g. ``"http://127.0.0.1:54321"``.
        ca_pem: PEM-encoded CA certificate string for inline trust.
        model: Full model string, e.g. ``"anthropic/claude-sonnet-4-6"``.

    Returns:
        The original config dict (deep copy) for restoration.
    """
    if "/" in model:
        provider, model_id = model.split("/", 1)
    else:
        provider, model_id = "anthropic", model

    if not os.path.exists(OPENCLAW_CONFIG):
        raise FileNotFoundError(
            f"OpenClaw config not found at {OPENCLAW_CONFIG}. "
            "Run `openclaw onboard` to set up at least one provider first."
        )

    with open(OPENCLAW_CONFIG) as f:
        config = json.load(f)

    shutil.copy2(OPENCLAW_CONFIG, OPENCLAW_BACKUP)
    original = json.loads(json.dumps(config))  # deep copy

    config.setdefault("models", {})
    config["models"].setdefault("providers", {})
    config["models"]["providers"].setdefault(provider, {})

    prov = config["models"]["providers"][provider]

    if "baseUrl" not in prov:
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        prov["baseUrl"] = defaults.get("baseUrl", f"https://api.{provider}.com/v1")
    if "models" not in prov:
        prov["models"] = []

    model_ids = [m.get("id") for m in prov["models"]]
    if model_id not in model_ids:
        prov["models"].append({"id": model_id, "name": model_id, "compat": {}})

    prov["request"] = {
        "proxy": {"mode": "explicit-proxy", "url": proxy_url, "tls": {"ca": ca_pem},},
        "allowPrivateNetwork": True,
    }

    config.setdefault("agents", {}).setdefault("defaults", {})
    config["agents"]["defaults"].setdefault("models", {})
    config["agents"]["defaults"]["models"][model] = {}
    config["agents"]["defaults"]["model"] = {"primary": model}

    with open(OPENCLAW_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    return original


def restore_openclaw_config(original: dict) -> None:
    """Restore the original OpenClaw config from a deep copy."""
    with open(OPENCLAW_CONFIG, "w") as f:
        json.dump(original, f, indent=2)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class OpenClawAgent:
    """Agent wrapper for ``openclaw infer model run``.

    Compatible with :func:`agentopt.ModelSelector` — implements the
    ``__init__(models)`` / ``run(input_data)`` interface.

    Each ``run()`` call:

    1. Reads the current session proxy from agentopt
    2. Patches ``~/.openclaw/openclaw.json`` to route through the proxy
    3. Runs ``openclaw infer model run`` as a subprocess
    4. Restores the original config
    5. Returns the model's text output

    Args:
        models: Dict with an ``"agent"`` key mapping to the model string,
            e.g. ``{"agent": "anthropic/claude-sonnet-4-6"}``.
        timeout: Seconds before the subprocess is killed. Default 120.
    """

    def __init__(self, models: Dict[str, str], timeout: int = OPENCLAW_TIMEOUT):
        self.model: str = models["agent"]
        self.timeout: int = timeout
        self._original_config: Optional[dict] = None

    def run(self, input_data: Any) -> str:
        """Run a single inference call through OpenClaw.

        Args:
            input_data: Either a string prompt or a dict with a ``"prompt"`` key.

        Returns:
            The model's text output, or a ``"FAILED: ..."`` string on error.
        """
        prompt = input_data if isinstance(input_data, str) else input_data["prompt"]

        proxy = get_current_session_proxy()
        if proxy is not None:
            self._original_config = patch_openclaw_config(
                proxy.url, proxy.ca_pem, self.model
            )

        cmd = [
            "openclaw",
            "infer",
            "model",
            "run",
            "--prompt",
            prompt,
            "--model",
            self.model,
            "--json",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    **os.environ,
                    "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}",
                },
            )

            if result.returncode != 0:
                return f"FAILED: openclaw exit code {result.returncode}: {result.stderr[:200]}"

            output = json.loads(result.stdout)
            if not output.get("ok"):
                return f"FAILED: {output.get('error', 'unknown error')}"

            texts = [o.get("text", "") for o in output.get("outputs", [])]
            return "\n".join(texts)

        except subprocess.TimeoutExpired:
            return f"FAILED: timeout after {self.timeout}s"
        except FileNotFoundError:
            return "FAILED: openclaw not found — run: npm install -g openclaw"
        except json.JSONDecodeError:
            return f"FAILED: invalid JSON output: {result.stdout[:200]}"
        finally:
            if self._original_config is not None:
                restore_openclaw_config(self._original_config)
                self._original_config = None
