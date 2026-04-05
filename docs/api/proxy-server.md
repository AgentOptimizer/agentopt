# Plan: HTTP Proxy Server for agentopt

## Context

agentopt intercepts LLM calls via in-process httpx monkey-patching. This works only for Python agents using httpx in the same process. We need to support subprocess-based agents (Gemini CLI, CUA, terminal agents, OpenClaw) while keeping the user-facing API unchanged.

**Core idea**: A real HTTP proxy server on localhost. The httpx monkey-patch becomes the "base_url insertion" mechanism --- it rewrites URLs at request time to route through the proxy. For subprocess agents, env vars achieve the same thing. Both produce the same URL format at the proxy. No separate modes.

---

## Architecture

```
  agentopt orchestrator
  |-- tracker.start()  -> starts proxy on localhost:PORT
  |-- tracker.track()  -> creates session, sets ContextVar
  |
  |-- In-process agent                    |-- Subprocess agent
  |   SDK calls api.openai.com           |   env: OPENAI_BASE_URL=
  |   httpx patch rewrites to:           |   http://localhost:PORT/{sid}
  |   http://localhost:PORT/{sid}/...     |   SDK sends directly to proxy
  |   + X-AgentOpt-Target header         |
  +----------------+----------------------+
                   v
  +----------------------------------------------+
  |  Proxy Server (aiohttp, background thread)   |
  |  http://127.0.0.1:PORT                       |
  |                                              |
  |  /{session_id}/{path} ->                     |
  |    1. Look up session                        |
  |    2. Resolve target:                        |
  |       - X-AgentOpt-Target header (exact)     |
  |       - OR auto-detect from path pattern     |
  |    3. Cache check                            |
  |    4. Forward to real LLM API                |
  |    5. Parse usage, record CallRecord         |
  |    6. Return response                        |
  +----------------------------------------------+
```

### How provider routing works

**In-process agents (fully general)**: The httpx monkey-patch sees the original URL (e.g., `https://api.openai.com/v1/chat/completions`) and adds it as `X-AgentOpt-Target: https://api.openai.com` header. The proxy forwards to that exact base URL. Works with ANY provider --- no registration needed.

**Subprocess agents (registered providers)**: The proxy auto-detects the provider from the API path pattern using a configurable registry:

- `/v1/chat/completions`, `/chat/completions` -> `https://api.openai.com`
- `/v1/messages` -> `https://api.anthropic.com`
- `/v1beta/models/`, `/v1/models:generateContent` -> `https://generativelanguage.googleapis.com`
- Custom providers via `tracker.register_provider(name, base_url, path_patterns)`

### Session attribution

Same mechanism for both --- session_id in URL path:

- **In-process**: httpx patch reads `_session_id_var` ContextVar, inserts into URL path
- **Subprocess**: session_id baked into env var URL at subprocess spawn time

### API key passthrough

All headers (Authorization, x-api-key, x-goog-api-key) are forwarded to the upstream provider as-is. The proxy stores no keys. Bound to `127.0.0.1` only.

---

## New dependency

```toml
dependencies = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
    "aiohttp>=3.9.0",
]
```

aiohttp provides async HTTP server (programmable start/stop in background thread) + `aiohttp.ClientSession` for efficient async request forwarding with connection pooling and streaming support.

---

## Files to create

### 1. `src/agentopt/proxy/providers.py`

Provider registry and auto-detection for subprocess agents.

```python
@dataclass
class ProviderConfig:
    name: str
    base_url: str                   # e.g. "https://api.openai.com"
    path_patterns: tuple[str, ...]  # e.g. ("/v1/chat/completions", "/chat/completions")

# Built-in providers
DEFAULT_PROVIDERS: dict[str, ProviderConfig]  # openai, anthropic, google

def detect_provider(path: str) -> Optional[ProviderConfig]:
    """Match a request path against registered provider patterns."""

def resolve_target(
    path: str,
    headers: dict,
    providers: dict[str, ProviderConfig],
) -> tuple[str, str]:
    """Return (target_base_url, upstream_path).
    
    Priority:
    1. X-AgentOpt-Target header -> use directly (in-process path)
    2. Path pattern matching -> registered provider (subprocess path)
    3. Raise error if no match
    """
```

### 2. `src/agentopt/proxy/session.py`

Thread-safe in-memory session store. Lives inside the proxy server process.

```python
@dataclass
class SessionInfo:
    session_id: str
    data_id: Optional[str]
    combo_id: Optional[str]
    agent_id: Optional[str]
    records: list[CallRecord]       # per-session records
    _lock: threading.Lock           # protects records list
    created_at: float

class SessionManager:
    def create_session(self, data_id, combo_id, agent_id=None) -> SessionInfo
    def end_session(self, session_id) -> SessionInfo  # removes from active, keeps records
    def get_session(self, session_id) -> Optional[SessionInfo]
    def add_record(self, session_id, record) -> None
    def get_all_records(self) -> list[CallRecord]     # all ended sessions' records
    def force_end_all(self) -> None                   # cleanup on stop()
```

### 3. `src/agentopt/proxy/server.py`

The aiohttp proxy server. Runs in a background daemon thread.

```python
class ProxyServer:
    def __init__(self, host="127.0.0.1", port=0, cache=None, providers=None):
        self.session_manager = SessionManager()
        self._cache = cache                # existing ResponseCache
        self._providers = providers or DEFAULT_PROVIDERS
        # port=0 -> OS assigns free port

    def start(self) -> None:
        """Start server in background daemon thread. Blocks until ready."""
        # threading.Thread(target=self._run, daemon=True)
        # self._started_event.wait()

    def stop(self) -> None:
        """Graceful shutdown. Force-end remaining sessions."""

    @property
    def base_url(self) -> str:  # "http://127.0.0.1:{port}"

    def session_url(self, session_id: str) -> str:  # "{base_url}/{session_id}"

    def register_provider(self, name, base_url, path_patterns) -> None

    # --- aiohttp handlers ---

    async def _handle_proxy(self, request):
        """Catch-all: ANY /{session_id}/{path:.*}"""
        # 1. Parse session_id from first path segment
        # 2. Look up session in session_manager
        # 3. Resolve target URL via resolve_target(path, headers, providers)
        # 4. Cache check (non-streaming): reuse existing _make_cache_key + ResponseCache
        # 5. Forward via aiohttp.ClientSession:
        #    - Copy all headers (auth, content-type, etc.)
        #    - For non-streaming: read response, parse usage, cache, record
        #    - For streaming: passthrough chunks, accumulate for usage parsing at end
        # 6. Return response

    async def _handle_management(self, request):
        """POST /_agentopt/{start_session|end_session}"""
        # HTTP API for external tools (not used by agentopt internally)
```

**Streaming**: Stream SSE chunks through in real-time via `aiohttp.StreamResponse`. After stream completes, parse accumulated data for token usage. Cache skipped for streaming (same as current).

**Server lifecycle**: Background daemon thread with `asyncio.new_event_loop()`. Start signaled via `threading.Event`. Stop via `loop.call_soon_threadsafe(shutdown_event.set)`.

---

## Files to modify

### 4. `src/agentopt/proxy/interceptor.py`

Replace the current token-parsing interceptor with a simpler URL-rewriting one. All token parsing, caching, and recording moves to the proxy server.

```python
# NEW ContextVar (replaces 3 separate ones for proxy mode)
_session_id_var: ContextVar[Optional[str]] = ContextVar("agentopt_session_id", default=None)

_proxy_base_url: Optional[str] = None

def install_redirect(proxy_base_url: str) -> None:
    """Monkey-patch httpx to redirect LLM requests through the proxy.
    
    This is the 'base_url insertion' for in-process agents.
    """
    # Patched send():
    # 1. Check _is_llm_request(request) -- skip non-LLM traffic
    # 2. session_id = _session_id_var.get() -- skip if None
    # 3. original_base_url = extract base from request.url (scheme + host)
    # 4. remaining_path = request.url.raw_path
    # 5. new_url = f"{proxy_base_url}/{session_id}{remaining_path}"
    # 6. Add header: X-AgentOpt-Target: {original_base_url}
    # 7. Create new httpx.Request with rewritten URL + target header
    # 8. Send via original_send (goes to localhost proxy over plain HTTP)

def uninstall_redirect() -> None:
    """Restore original httpx methods."""
```

Remove the old `install()` / `uninstall()` functions entirely. They are replaced by `install_redirect()` / `uninstall_redirect()`. Delete `_data_id_var`, `_combo_id_var`, `_agent_id_var` --- session attribution is now handled by the proxy server via `_session_id_var` + `SessionManager`, not per-field ContextVars. Update any tests that reference the old functions.

### 5. `src/agentopt/proxy/tracker.py`

`LLMTracker` now manages the proxy server lifecycle. Same public API.

```python
class LLMTracker:
    def __init__(self, cache=True, cache_dir=".agentopt_cache"):
        self._server: Optional[ProxyServer] = None
        self._response_cache = ResponseCache(cache_dir=cache_dir) if cache else None
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def start(self):
        self._server = ProxyServer(cache=self._response_cache)
        self._server.start()
        install_redirect(proxy_base_url=self._server.base_url)

    def stop(self):
        # Collect records from all sessions
        with self._lock:
            self._records.extend(self._server.session_manager.get_all_records())
        uninstall_redirect()
        self._server.stop()
        self._server = None
        if self._response_cache:
            self._response_cache.close()

    @contextmanager
    def track(self, data_id, combo_id, agent_id=None):
        """Create a proxy session for this evaluation scope."""
        session = self._server.session_manager.create_session(
            data_id=data_id, combo_id=combo_id, agent_id=agent_id
        )
        token = _session_id_var.set(session.session_id)
        try:
            yield session  # caller can access session.session_id
        finally:
            _session_id_var.reset(token)
            self._server.session_manager.end_session(session.session_id)

    @contextmanager
    def track_agent(self, agent_id: str):
        """Unchanged -- still sets agent_id ContextVar."""
        # Need to update the current session's agent_id too
        ...

    def get_session_env(self, session_id: str) -> dict[str, str]:
        """Env vars for subprocess agents to route through proxy."""
        url = self._server.session_url(session_id)
        return {
            "OPENAI_BASE_URL": url,
            "ANTHROPIC_BASE_URL": url,
            "GOOGLE_API_BASE": url,
            "AGENTOPT_SESSION_ID": session_id,
            "AGENTOPT_PROXY_URL": self._server.base_url,
        }

    # get_records(), get_usage(), get_cached_latency() -- same API
    # Internally query from server.session_manager + self._records
```

**Backward compat**: `track()` currently yields nothing. New version yields `SessionInfo`. Existing code uses `with tracker.track(...):` (no capture) so this is backward-compatible.

### 6. `src/agentopt/proxy/__init__.py`

Add exports: `SessionInfo`, `ProxyServer`, `get_current_session_env`.

### 7. `src/agentopt/model_selection/base.py`

Minimal changes --- capture session from `track()`:

```python
# _evaluate_agent():
with self._tracker.track(data_id=dp_id, combo_id=label) as session:
    start_time = time.time()
    actual_result = self._invoke_agent(agent, input_data)
    ...

# _evaluate_agent_async(): same change
```

No other changes needed. In-process agents work transparently via the httpx patch.

### 8. `src/agentopt/__init__.py`

Expose `get_current_session_env()` at package level:

```python
def get_current_session_env() -> dict[str, str]:
    """Get proxy env vars for the current session.
    
    Use inside agent.run() when spawning subprocesses:
        env = {**os.environ, **agentopt.get_current_session_env()}
        subprocess.run(["gemini", "cli", ...], env=env)
    """
```

### 9. `pyproject.toml`

Add `aiohttp>=3.9.0` to dependencies.

---

## Implementation sequence

### Phase 1: Foundation

1. Create `providers.py` --- provider registry, `resolve_target()`
2. Create `session.py` --- `SessionManager` + `SessionInfo`
3. Unit tests for both

### Phase 2: Proxy server

4. Create `server.py` --- aiohttp app, background thread lifecycle, catch-all proxy handler
5. Non-streaming forwarding first (with `aiohttp.ClientSession`)
6. Streaming passthrough (SSE chunks forwarded in real-time)
7. Cache integration (reuse existing `ResponseCache` + `_make_cache_key`)
8. Token parsing + `CallRecord` creation (move `_parse_usage` logic to server)

### Phase 3: Interceptor + tracker

9. Add `install_redirect()` / `uninstall_redirect()` to `interceptor.py`
10. Update `tracker.py` --- proxy lifecycle, session-based `track()`
11. Add `get_current_session_env()` to `__init__.py`

### Phase 4: Orchestrator

12. Update `base.py` --- `as session` in `track()` calls

### Phase 5: Tests

13. Integration tests: mock upstream server + proxy + in-process agent
14. Cache-through-proxy tests
15. Concurrent session isolation
16. Existing tests pass

---

## Verification

1. `pytest tests/` --- all existing tests pass
2. Smoke test with `examples/custom_agent_example.py` --- verify proxy starts, calls tracked, cache works
3. Verify `get_current_session_env()` returns correct URLs for subprocess use
4. Run with `parallel=True, max_concurrent=20` --- verify session isolation across concurrent combos

---

## Key files

| File | Action |
|------|--------|
| `src/agentopt/proxy/providers.py` | Create |
| `src/agentopt/proxy/session.py` | Create |
| `src/agentopt/proxy/server.py` | Create |
| `src/agentopt/proxy/interceptor.py` | Modify (simplify to URL rewriting) |
| `src/agentopt/proxy/tracker.py` | Modify (proxy lifecycle) |
| `src/agentopt/proxy/cache.py` | Unchanged (reused server-side) |
| `src/agentopt/proxy/models.py` | Unchanged |
| `src/agentopt/proxy/__init__.py` | Modify (new exports) |
| `src/agentopt/model_selection/base.py` | Modify (`as session` in track) |
| `src/agentopt/__init__.py` | Modify (expose helper) |
| `pyproject.toml` | Modify (add aiohttp) |
