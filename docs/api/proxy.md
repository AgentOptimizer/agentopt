# AgentOpt interception architecture

## What are we trying to do?

You have an agent that makes LLM API calls. You want to try different models (GPT-4o, Claude Sonnet, Haiku, etc.) and find which combination works best. To compare them, you need to measure three things for every LLM call: which model was used, how many tokens it consumed, and how long it took.

The challenge: you want to do this without modifying the agent's code. The agent just calls the LLM API however it wants — through OpenAI SDK, Anthropic SDK, LangChain, whatever. You want to observe those calls invisibly, from the outside.

## Where can you observe LLM calls?

Every LLM API call, regardless of which SDK or framework makes it, eventually becomes an HTTP request. The agent's code calls some SDK function, that SDK uses an HTTP library (almost always `httpx` in Python), and `httpx` sends an HTTPS request over the network to the API server.

So there are only two places you can intercept:

**Inside the process** — patch the HTTP library before it sends the request. You modify `httpx.Client.send()` in memory so your code runs every time the agent makes an HTTP call.

**On the network** — run a proxy server that sits between the agent and the API server. All traffic flows through you.

## In-process agents: the easy case

When the agent runs in the same Python process as your optimization code (LangChain, CrewAI, etc.), you can patch `httpx` directly. You replace `httpx.Client.send` with your own function that:

1. Checks if this looks like an LLM API call (is it a POST to `/v1/chat/completions` or `/v1/messages`?)
2. If yes, looks up the active session (a `ContextVar` set by `tracker.track()`)
3. Checks the response cache; on hit, returns the cached response directly without a network round-trip
4. Otherwise calls the real upstream, times it, extracts token usage from the response, builds a `CallRecord`, and returns the response

This works because you're modifying the HTTP library's function pointer in the process's own memory. You're intercepting before encryption, so you never need to deal with TLS at all. There's no localhost listener — the wrapper records directly and the call goes straight to the real upstream.

## Subprocess agents: why the easy approach breaks

When the agent is a separate process — Claude Code, Gemini CLI, a `tb run` command that runs in Docker — you can't patch their `httpx`. When you do `subprocess.run(["tb", "run", ...])`, that child process loads its own Python runtime, imports its own `httpx`, and calls `httpx.Client.send()` with the original, unpatched code. Your patch only exists in your process's memory. The OS keeps processes isolated from each other — you cannot reach into another process's memory and modify its functions.

So you can't intercept inside the client. The only option left is to intercept on the network. But on the network, the traffic is HTTPS — encrypted. You can see that the subprocess is connecting to `api.openai.com`, but you can't read the request body (which model? what messages?) or the response (how many tokens?).

These are the agents worth optimizing — complex multi-step agents with many LLM calls where the combinatorial space explodes:

| Category | Examples | Topology |
|---|---|---|
| Coding agents | Claude Code, Codex CLI, Aider, OpenHands, SWE-Agent | subprocess, docker |
| CUA agents | Anthropic CUA, OpenAI Operator | VM, docker |
| Terminal agents | TerminalBench agents | docker |
| CLI agents | Gemini CLI | subprocess |
| Autonomous agents | Devin, Manus | docker, remote |
| In-process frameworks | LangChain, CrewAI, LlamaIndex | in-process |

In-process frameworks are increasingly the tutorial-tier agents. The production agents that people actually deploy and want to optimize are almost all out-of-process.

## How HTTPS_PROXY works

Most HTTP libraries check an environment variable called `HTTPS_PROXY` before making connections. If it's set, instead of connecting directly to the API server, the client connects to the proxy address and asks the proxy to relay the connection.

You set this in the subprocess's environment before spawning it:

```python
env["HTTPS_PROXY"] = "http://127.0.0.1:59198"
subprocess.run(["tb", "run", ...], env=env)
```

Now the subprocess's HTTP library sees `HTTPS_PROXY` and does something different. Instead of connecting to `api.openai.com:443`, it connects to `127.0.0.1:59198` (your proxy) and sends:

```
CONNECT api.openai.com:443 HTTP/1.1
```

This is a plain-text message saying: "Please open a connection to `api.openai.com` port 443 for me." This message itself is not encrypted — it's just HTTP. Your proxy can read it and knows what the client wants to connect to.

## The CONNECT tunnel problem

A normal HTTPS proxy would respond `200 Connection Established` and then become a dumb pipe — just forwarding bytes between the client and the real server. The client would then do a TLS handshake with the real `api.openai.com` through this pipe, and all subsequent traffic would be encrypted end-to-end. The proxy would be in the middle but unable to read anything.

This is useless for us. We need to read the traffic.

## The MITM solution: two TLS sessions

Instead of being a dumb pipe, our proxy impersonates the API server. When the client starts its TLS handshake after CONNECT, the proxy responds as if it is `api.openai.com`. This creates two separate encrypted connections:

**Left side**: the agent subprocess ↔ your proxy. The agent thinks it's talking to OpenAI. It sends its API key, prompts, and model name through this encrypted channel. But the proxy holds the encryption key, so it can decrypt and read everything.

**Right side**: your proxy ↔ real OpenAI. The proxy opens its own normal HTTPS connection to the actual API server. It forwards the request (possibly after recording it), gets the response, records the token counts and latency, and sends the response back through the left side to the agent.

The agent has no idea this happened. From its perspective, it made a normal API call and got a normal response.

## Certificates: why the agent accepts the fake connection

For the left-side TLS handshake to work, the proxy needs to present a certificate that says "I am api.openai.com." But the agent's HTTP library will check this certificate — specifically, it checks who signed it. Legitimate certificates are signed by well-known Certificate Authorities (CAs) like DigiCert or Let's Encrypt. The HTTP library has a built-in list of these trusted CAs.

Our proxy can't get DigiCert to sign a fake certificate for `api.openai.com`. So instead, we use [mitmproxy](https://mitmproxy.org)'s CA. mitmproxy ships a battle-tested implementation: on first run it generates a root CA at `~/.mitmproxy/`, and on demand it mints per-hostname leaf certificates signed by that root.

By default, the agent's HTTP library doesn't trust the mitmproxy CA, so it would reject the connection with an SSL error (`SSLV3_ALERT_CERTIFICATE_UNKNOWN`). We fix this by setting `SSL_CERT_FILE` in the subprocess's environment, pointing at a bundle that contains both the normal system CAs (from `certifi`) and the mitmproxy CA. AgentOpt builds and maintains that bundle at `~/.mitmproxy/agentopt-bundle.pem`.

The bundle is important: if we only included the mitmproxy CA, the subprocess couldn't make any other HTTPS connections (like downloading packages or talking to GitHub). By bundling it with the system CAs, everything else works normally. Only connections to LLM API hostnames get intercepted; all other HTTPS traffic passes through the proxy as a raw tunnel (the addon's `tls_clienthello` hook sets `ignore_connection=True` for non-LLM SNIs, so mitmproxy doesn't even attempt TLS termination on them).

## Attribution: knowing which call belongs to which evaluation

When you're evaluating multiple model combinations, you need to know which LLM calls belong to which combo. If you run combo A (GPT-4o) and then combo B (Claude Sonnet), you need to assign the recorded calls to the right combo.

The design uses two attribution mechanisms, one per interception path:

- **In-process** — a `ContextVar[ActiveSession]` holds the current session. Python's `ContextVar` propagates per-task / per-thread automatically, so concurrent `tracker.track()` blocks each see their own active session without mutating any shared state.
- **Subprocess** — **one TCP port per session**. Each `tracker.track()` eagerly spins up a dedicated mitmproxy `DumpMaster` on its own ephemeral port. The subprocess gets `HTTPS_PROXY=http://127.0.0.1:{port}` via a `subprocess.Popen.__init__` monkey-patch that reads the active session from the same `ContextVar`, so the kernel routes the child's traffic to that master, which holds an addon bound to that session.

### ContextVar propagation limits

Both interception paths consult `_active_session_var`. ContextVar propagation matches the standard Python semantics, which means there are three corners where the patches don't see an active session even though the user is "inside" a `track()` block:

- **`threading.Thread`** — a new thread starts with a fresh context, so calls inside the thread don't see the active session. Workaround: wrap the target with `contextvars.copy_context().run(...)` (or use `concurrent.futures.ThreadPoolExecutor` configured to copy the parent's context).
- **`multiprocessing.Process` (spawn mode)** — the child is a fresh Python interpreter; it never called `tracker.start()`, so neither patch is installed there. Tracked subprocesses must be spawned via `subprocess.Popen` from the tracker's own process, not from a fresh interpreter.
- **`multiprocessing.Process` (fork mode)** — the child inherits everything, including the patches and the ContextVar value at fork time. Works, but both parent and child will route to the *same* session ID, which can cause double-counting if both make LLM calls.

These are accepted limitations of the ContextVar design and apply identically to the httpx patch and the subprocess patch.

### Subprocess env merge policy

The subprocess patch injects `HTTPS_PROXY` + the merged CA bundle paths (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`) into the child's environment. *Explicit beats implicit*:

- `Popen(cmd)` / `Popen(cmd, env=None)` (inherit `os.environ`): `{**os.environ, **session_env}` — the session wins over any `HTTPS_PROXY` the parent shell happened to set.
- `Popen(cmd, env={...})` (caller wrote an explicit env): `{**session_env, **user_env}` — caller's keys win on conflicts. Setting `env={"HTTPS_PROXY": "http://custom:8080"}` inside `track()` is respected; setting `env={"PATH": ...}` still gets tracking because the user didn't write the session keys.

This is a design choice, not a forced constraint. Alternatives we rejected:

- **Single shared mitmproxy with dynamic multi-port mode list**: lower per-session overhead but depends on mitmproxy's runtime mode-update behavior, which isn't part of the documented public API.
- **Header-based attribution**: the in-process path could add an `X-AgentOpt-Session` header, but subprocesses don't know to do that — they're opaque clients we can only configure via env vars. Port-as-identity is the natural answer for the subprocess case.
- **Source-port tracking on a single shared proxy**: requires a fragile PID/port mapping that breaks under fork/exec.

Per-session masters cost ~100-300ms startup and ~30MB RSS each. Acceptable for research workloads where session count is low and parallel safety matters more than absolute throughput.

## Local vs daemon mode: where the proxy lives

Everything above describes the *mechanism* of interception. Orthogonal to that mechanism is the question of *where* the proxy runs — in the same Python process as your selector code, or in a long-lived daemon.

`LLMTracker` picks the mode automatically from a single environment variable:

```bash
# Default — in-process: spins per-session mitmproxy masters in this process.
python my_script.py

# Daemon mode — talk to a long-lived gateway.
AGENTOPT_GATEWAY_URL=http://127.0.0.1:9000 python my_script.py
```

The user-facing API is byte-identical between modes. `ModelSelector(...).select_best()`, `tracker.track()`, `tracker.get_records()` — none of it changes. Switching modes is a deployment decision, not an API decision.

### What's identical between modes

Same proxy mechanism end-to-end. Per-session mitmproxy masters, the same `AgentoptAddon`, same CA, same path-pattern detection, same `CallRecord` schema. The only thing that varies is *which process* owns the master.

### What differs

| | Local mode | Daemon mode |
|---|---|---|
| Where the master runs | The user's Python process | The `agentopt serve` daemon |
| Where the cache + records live | In-process | On the daemon |
| Where the in-process httpx patch sends traffic | Directly to the upstream LLM API | Through the daemon's per-session proxy port |
| Multi-process / multi-language clients | Subprocess agents only | First-class: any client that respects `HTTPS_PROXY` |
| State outlives a single experiment | No (process-bound) | Yes (daemon-bound) |
| Setup | None | Run `agentopt serve` separately |

### The `agentopt serve` daemon

A small `aiohttp.web` app that owns one `LocalBackend` and exposes its surface over HTTP. Localhost-only in v1 (no auth).

```bash
agentopt serve --port 9000 --cache-dir .agentopt_cache

# With a daemon-wide default router (per-session overrides still allowed):
agentopt serve --routing-policy random \
    --candidate-models gpt-4o,gpt-4o-mini --seed 42

# With user-defined Router subclasses preloaded:
agentopt serve --policy-module ./my_policies.py
```

Control plane:

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/health`                          | liveness probe |
| `POST`   | `/sessions`                        | open a session — body may include `data_id`, `combo_id`, `agent_id`, and an optional `router={"policy": "...", "kwargs": {...}}` override; returns `{session_id, proxy_port, ca_pem_b64}` |
| `DELETE` | `/sessions/{session_id}`           | close a session |
| `GET`    | `/records?data_id=&combo_id=…`     | filtered `CallRecord` list |
| `GET`    | `/usage?…`                         | aggregated token usage |
| `GET`    | `/cached_latency?…`                | total cached-response latency |
| `POST`   | `/cache/flush`                     | force-flush dirty cache rows |
| `POST`   | `/cache/clear`                     | drop all cached responses |
| `POST`   | `/providers`                       | register a custom LLM provider |
| `GET`    | `/ca`                              | mitmproxy CA cert (also returned in `POST /sessions`) |

The daemon refuses to bind a non-loopback host without `--allow-remote`, which is reserved for a future revision that ships authentication. Until then, refusing fast is safer than accidentally exposing an unauthenticated proxy on the network.

### How the in-process httpx patch routes in daemon mode

In local mode the patched `httpx.Client.send` does the work itself (cache lookup, forward to upstream, record). In daemon mode it would be duplicate machinery — the daemon's `AgentoptAddon` already does all of that. So the patched `send` instead forwards the original request through an `httpx.Client(proxy=daemon_session_url, verify=daemon_ca_bundle)`, and the daemon records + caches.

The seam is a small `CallHandler` ABC inside `interceptor.py`. Two implementations:

- **`LocalHandler`** — today's behaviour (cache, forward to real upstream, record).
- **`RemoteHandler`** — forwards through the daemon's per-session proxy port; the daemon does cache + record.

Both are bound to the `ActiveSession` ContextVar at `track()` entry. The patched `send` is a one-line dispatcher: `return active.handler.handle_sync(...)`. Nothing about the activation, path-pattern filter, or `_active_session_var` plumbing changes between modes.

### Why daemon mode at all

Two motivations:

1. **Multi-language / multi-process clients.** Subprocess agents work today via `HTTPS_PROXY`, but each Python process spins its own proxy. With a daemon, one gateway serves any number of clients in any language — they all just point `HTTPS_PROXY` at the same per-session port returned by `POST /sessions`.
2. **State that outlives a process.** Cache survives across runs. Records can be queried later. A foundation for future features (concurrency caps, request coalescing, cross-call observability) that need a global view a single-process library can't supply.

Routing also works over the wire: the daemon accepts a per-session `router` field on `POST /sessions` (built-in policies like `random`, plus custom `Router` subclasses preloaded via `--policy-module`), and supports a daemon-wide default via `--routing-policy`. See [router.md](router.md#daemon-mode) for details.

## The complete flow

Here's everything that happens end-to-end when you run an evaluation:

**Startup**: `tracker.start()` installs the httpx monkey-patch. No mitmproxy masters are running yet — they're per-session.

**Session creation**: `tracker.track(data_id="dp_1", combo_id="gpt4o")` creates a session, eagerly spins up a `SessionMaster` (mitmproxy `DumpMaster` in a dedicated thread on its own asyncio loop, listening on an ephemeral port), sets the `ContextVar`, and returns session env vars for subprocess use.

**In-process path**: The agent calls the OpenAI SDK → SDK calls `httpx.Client.send()` → the monkey-patch intercepts, reads the active session from `ContextVar`, looks up the cache, calls the real upstream directly, extracts token usage from the response, records a `CallRecord`, returns the response to the agent.

**Subprocess path**: The agent runs as a child process with `HTTPS_PROXY` and `SSL_CERT_FILE` set → the agent's HTTP library connects to the session's mitmproxy port and sends CONNECT → mitmproxy's `tls_clienthello` hook checks the SNI; if it's not in our intercept set, the connection is tunnelled raw and we never see the bytes → otherwise mitmproxy TLS-terminates with a per-hostname cert from its CA → the addon's `request` hook checks the cache and short-circuits on hit → on miss, mitmproxy forwards to the real upstream, the addon's `response` hook records token counts and latency → the response goes back through the encrypted tunnel to the agent.

**Session teardown**: `track()` scope exits → ContextVar is reset, the SessionMaster is shut down (drains in-flight requests, joins the thread), and the session is archived.

**Shutdown**: `tracker.stop()` restores the original `httpx.Client.send`, stops any remaining masters, and flushes the cache. Record queries remain valid after `stop()`; `tracker.close()` does the final teardown (releases the remote backend's long-lived HTTP client). `LLMTracker` is a context manager — `with LLMTracker() as t:` calls `stop()` (not `close()`) on exit so `tracker.print_summary()` immediately after the block still works.

In daemon mode the same calls dispatch over HTTP: `start()` health-checks the daemon, `track()` POSTs `/sessions`, `stop()` closes any lingering sessions, `close()` releases the control-plane client.

## Session lifecycle

```
tracker.start()
  └── httpx monkey-patch installed; no mitmproxy masters running

tracker.track(data_id="dp_1", combo_id="gpt4o+haiku")
  └── Creates a SessionInfo
  └── Spins up a SessionMaster (mitmproxy DumpMaster) on a fresh ephemeral port
  └── Sets ContextVar: _active_session_var = ActiveSession(session, recorder, cache, port)
  └── Returns session env vars for subprocess use
  └── In-process LLM calls: httpx wrapper records into this session
  └── Subprocess traffic on this port: addon records into this session

track() exit:
  └── Reset ContextVar
  └── Shut down SessionMaster (signals mitmproxy.Master.shutdown — thread-safe)
  └── In-flight upstream calls finish and write to the archived session
  └── End session (move from active → ended in SessionManager)
```

### In-process path detail

```
agent.run(input)
  └── OpenAI SDK → httpx.Client.send()
      └── wrapper reads _active_session_var (ContextVar) → ActiveSession
      └── cache.get(hash(request_body)):
          ├── HIT → build httpx.Response from cached bytes; record cached=True; return
          └── MISS → call original_send to the real upstream
              └── extract usage; record CallRecord; cache 200 responses; return
```

### Subprocess path detail

```
subprocess.run(["tb", "run", ...], env=session_env)
  └── inherits HTTPS_PROXY=http://127.0.0.1:{master.port} from env
  └── HTTP library sees HTTPS_PROXY → sends CONNECT api.openai.com:443
      └── mitmproxy's tls_clienthello hook fires:
          ├── SNI not intercepted → ignore_connection=True; tunnel raw bytes
          └── SNI is an LLM host → terminate TLS with per-host cert from mitmproxy CA
              └── addon.request: cache lookup; short-circuit on hit
              └── otherwise: mitmproxy forwards to real upstream
              └── addon.response: extract usage; record CallRecord; cache 200s
```

### Parallel evaluation

```
# Each concurrent evaluation gets its own session, master, and port:
track(combo_id="gpt4o+gpt4o")     → SessionMaster on port 59198
track(combo_id="gpt4o+gpt4o-mini") → SessionMaster on port 59205

# In-process: ContextVar is async/thread-safe; each task sees its own ActiveSession.
# Subprocess: each gets its own env dict passed explicitly, not os.environ.
```

## What lives where

### `agentopt.proxy`

The interception machinery, split between the in-process httpx wrapper, the per-session mitmproxy addon, and (optionally) a long-lived `agentopt serve` daemon.  All paths call into the same recording / cache / token-extraction code, so a record is a record regardless of how the call arrived.

- **`tracker.py`** — `LLMTracker`.  Public surface.  Thin delegator that picks a backend in `__init__` based on `AGENTOPT_GATEWAY_URL`.  Context-manager-friendly: `with LLMTracker() as t:`.
- **`_backend.py`** — `_Backend` ABC and the shared CA-bundle helpers (`_MITMPROXY_CA_CERT`, `_ensure_ca_bundle`).  Both backends import from here.
- **`_local_backend.py`** — `LocalBackend`.  Per-session mitmproxy in this process; holds the shared `SessionManager`, `ResponseCache`, `ProviderRegistry`, `Recorder`; manages `SessionMaster` lifecycles per `track()`.
- **`_remote_backend.py`** — `RemoteBackend` and `RemoteHandler`.  Talks to the daemon's HTTP control plane; the in-process httpx patch's slow path forwards through the daemon's per-session port instead of recording locally.
- **`daemon.py`** — `aiohttp.web` app wrapping a singleton `LocalBackend`; CLI entrypoint registers the `agentopt serve` subparser.  Warms up mitmproxy's CA on startup so the first `/sessions` POST doesn't have to.
- **`cli.py`** — top-level `agentopt` argparse dispatcher; subcommands register themselves via `set_defaults(func=...)`.
- **`interceptor.py`** — the httpx monkey-patch.  On an LLM request inside an active session, the patched `send` is a one-line dispatcher to `active.handler.handle_{sync,async}`.  Two handlers: `LocalHandler` (today's cache+forward+record body) and `RemoteHandler` (forwards through the daemon).  Path-pattern set is a frozenset rebuilt-on-write so `register_provider` can extend it without racing the wrapper hot path.
- **`mitm_addon.py`** — `AgentoptAddon` for mitmproxy.  Hooks: `tls_clienthello` decides intercept-vs-passthrough at the SNI level; `request` does cache lookup and short-circuit; `response` records and caches; `error` records transport failures.
- **`mitm_runner.py`** — `SessionMaster`: hosts one `DumpMaster` per session in a background thread with its own asyncio loop.  Captures the bound port via the `running` addon hook.  Documents the embedded mitmproxy API surface we depend on so a major-version bump is traceable.
- **`recording.py`** — `Recorder`.  The single function that turns (session, request body, response body, latency, status) into a `CallRecord` and dispatches to `SessionManager`.  Owns the warn-once-per-host set for token-extraction failures.  Both the httpx wrapper and the addon use the same instance.
- **`providers.py`** — `Provider` dataclass and `ProviderRegistry`.  Per-`LocalBackend` catalog of LLM hostnames and path patterns.
- **`usage.py`** — pure token-extraction for OpenAI / Anthropic / Gemini response shapes (JSON object, JSON array, SSE).  Raises `UsageExtractionError` with a structured diagnostic on miss; never reports zero tokens silently.
- **`cache.py`** — `ResponseCache`.  In-memory dict, optionally persisted to SQLite via a daemon flush thread.  Keyed by a hash of the request body (excluding `stream`).
- **`session.py`** — `SessionManager`.  Active and archived sessions; `add_record` checks both so a slow upstream that finishes after `end_session` doesn't drop its record.

### `agentopt.proxy.LLMTracker` (the public surface)

Identical between local and daemon modes. Reads `AGENTOPT_GATEWAY_URL` in `__init__` to pick the backend. See [tracker.md](tracker.md) for the full surface; the highlights:

- **Lifecycle.** `tracker.start()` / `tracker.stop()` / `tracker.close()`. Record queries remain valid after `stop()`; `close()` is the final-teardown hook that releases the remote backend's long-lived HTTP client.
- **Context-manager sugar.** `with LLMTracker() as tracker:` calls `stop()` on exit (not `close()`, so `tracker.print_summary()` after the block still works). Passing any of `data_id` / `combo_id` / `agent_id` to the constructor also auto-opens a single tracking session for the lifetime of the `with`.
- **Sessions.** `tracker.track(data_id, combo_id, agent_id, router)` context manager — creates a session, eagerly spins up a `SessionMaster` (local) or POSTs `/sessions` (remote), sets the `ContextVar`. All four args are optional; `router=` falls back to the one passed to `LLMTracker(...)`.
- **Subprocess env.** `tracker.get_session_env(session)` returns `HTTPS_PROXY` + the merged CA bundle path. The URL points at the local `SessionMaster` or the daemon's per-session port, transparently. `agentopt.get_current_session_proxy()` is the module-level convenience reading the active session out of the `ContextVar`.
- **Providers.** `tracker.register_provider(name, base_url, path_patterns)` extends both the shared `ProviderRegistry` (subprocess intercept hosts) and the httpx wrapper's path-pattern set; in remote mode also POSTs `/providers`.
- **Queries.** `tracker.records`, `tracker.get_records(...)`, `tracker.get_usage(...)`, `tracker.get_cached_latency(...)`, `tracker.print_summary(...)`.
- **Cache.** `tracker.flush_cache()`, `tracker.clear_cache()`, `tracker.clear()`.

### The httpx wrapper (no business logic — delegates to `Recorder` + `ResponseCache`)

- `_is_llm_request(request)`: POST + path matches a known LLM endpoint
- Read active session from `ContextVar`
- Cache lookup; short-circuit on hit
- Call original `httpx.Client.send`; time + record + cache on miss

### The subprocess redirect (env vars, no logic)

- `HTTPS_PROXY=http://127.0.0.1:{master.port}`
- `SSL_CERT_FILE=~/.mitmproxy/agentopt-bundle.pem` (mitmproxy CA + certifi system CAs)
- `REQUESTS_CA_BUNDLE=...`, `NODE_EXTRA_CA_CERTS=...` for non-stdlib clients

## Known LLM API hostnames

The proxy only intercepts CONNECT requests to these hostnames (everything else is tunnelled raw, no MITM):

- `api.openai.com`
- `api.anthropic.com`
- `generativelanguage.googleapis.com`
- `cloudcode-pa.googleapis.com` (Gemini CLI OAuth)
- `bedrock-runtime.*.amazonaws.com`
- `*.openai.azure.com`
- `api.mistral.ai`
- `api.groq.com`
- `api.together.xyz`
- `api.deepseek.com`

Extend at runtime with:

```python
tracker.register_provider(
    name="openrouter",
    base_url="https://openrouter.ai",
    path_patterns=("/api/v1/chat/completions",),
)
```

This updates two places at once: the shared `ProviderRegistry` (subprocess intercept hosts — addons see new hosts via shared reference), and the in-process httpx-patch path set (so `_is_llm_request` returns true for the new path).

## Why each piece exists

Every layer of complexity is forced by a real constraint, not a design choice:

1. We want to observe LLM calls → patch httpx in-process
2. But subprocess agents have their own httpx → need a network proxy
3. But network traffic is encrypted → need MITM with fake certificates
4. But the agent rejects fake certificates → need a custom CA in the trust store
5. But we need to know which calls belong to which combo → use a separate port per session
6. But `os.environ` isn't safe for parallel subprocesses → pass env explicitly to each subprocess instead of mutating globals
7. But state dies with the Python process and other languages can't share it → run the same proxy code as a long-lived `agentopt serve` daemon; clients pick local vs daemon via the `AGENTOPT_GATEWAY_URL` env var, with no API change

## Scoping constraints

1. **Python only** — the proxy is Python.  CA trust config targets Python and Node HTTP clients (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
2. **HTTP only** — no gRPC, no WebSocket.  HTTP/HTTPS covers 99%+ of current LLM API traffic.
3. **mitmproxy for CONNECT/TLS** — pinned `mitmproxy>=12,<13`.  The from-scratch HTTP CONNECT handler we used to maintain (~1100 lines of socket / ssl / http.client) is gone; mitmproxy handles framing, TLS termination, HTTP/2, and certificate generation correctly by construction.
4. **One DumpMaster per session** — keeps the lifecycle simple and avoids depending on mitmproxy's undocumented runtime mode-update behavior.  Tradeoff: ~100-300ms startup and ~30MB RSS per concurrent session.
5. **Docker uses sidecar proxy** — the proxy runs inside the container.  Self-contained, no host networking dependency.

## Implementation notes

The one non-obvious correctness invariant we still own:

- **Token-usage extraction must not silently report zero.**  A successful (HTTP 200) call whose usage we can't parse is a real proxy gap, not a 0-token call.  `usage.py` raises `UsageExtractionError` with a diagnostic naming exactly which keys it searched and which were present; `Recorder` attaches that to `CallRecord.error` and uses a `<parse-failed>` sentinel for the model name so the failure surfaces in result summaries.
- **Late records must not be dropped.**  A blocking upstream request can finish *after* the session's `track()` scope exits.  `SessionManager.add_record` looks up both active and ended sessions so the late record lands in the archive instead of being lost.

(The previous CONNECT-prelude-buffering and strict-Content-Length invariants are now mitmproxy's problem, not ours.)

## Open questions

1. **CA certificate compatibility**: do any Python LLM SDKs override `SSL_CERT_FILE` or pin certificates?  Need to test against `openai`, `anthropic`, `google-generativeai`, `boto3`.

2. **Agents that bypass `HTTPS_PROXY`**: some SDK versions may hardcode direct connections.  Mitigation: maintain a compatibility matrix per SDK version.

3. **Real-time streaming forwarding**: SSE / `Transfer-Encoding: chunked` responses are *parsed* correctly today (the SSE token-extractor handles Anthropic's split `message_start` / `message_delta` usage events and Gemini's `usageMetadata`), but the proxy currently buffers the full upstream response before sending it back to the client — so the client doesn't see chunks land in real time.  For evaluation workflows this is fine (you read the final result anyway).  For interactive use you'd want chunk-by-chunk forwarding while still accumulating SSE frames for token extraction.
