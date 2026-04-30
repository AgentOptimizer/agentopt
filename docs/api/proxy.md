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
2. If yes, rewrites the URL from `https://api.openai.com/...` to `http://localhost:{port}/...`
3. Adds a header `X-AgentOpt-Target: https://api.openai.com` so the proxy knows where to forward it
4. Sends it

The critical detail: the URL scheme changes from `https` to `http`. This means the HTTP library doesn't encrypt the request. It sends plaintext to your proxy on localhost. Your proxy reads it trivially — model name, messages, everything is right there in the JSON body. The proxy forwards the request (over real HTTPS) to the actual API server, gets the response, records the token counts and latency, and returns the response to the agent. The agent never knows anything happened.

This works because you're modifying the HTTP library's function pointer in the process's own memory. You're intercepting before encryption, so you never need to deal with TLS at all.

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

Our proxy can't get DigiCert to sign a fake certificate for `api.openai.com`. So instead, we create our own CA:

1. On first run, generate a root CA key and self-signed certificate. Store them in `~/.agentopt/ca/`.
2. When the proxy needs to impersonate `api.openai.com`, generate a leaf certificate for that hostname, signed by our CA.

By default, the agent's HTTP library doesn't trust our CA, so it would reject the connection with an SSL error (`SSLV3_ALERT_CERTIFICATE_UNKNOWN`). We fix this by setting `SSL_CERT_FILE` in the subprocess's environment, pointing at a bundle that contains both the normal system CAs and our AgentOpt CA. Now the HTTP library trusts our CA, accepts the proxy's certificate, and the handshake completes.

The bundle is important: if we only included our CA, the subprocess couldn't make any other HTTPS connections (like downloading packages or talking to GitHub). By bundling our CA with the system CAs, everything else works normally. Only connections to LLM API hostnames get intercepted; all other HTTPS traffic passes through the proxy as a normal tunnel.

## Attribution: knowing which call belongs to which evaluation

When you're evaluating multiple model combinations, you need to know which LLM calls belong to which combo. If you run combo A (GPT-4o) and then combo B (Claude Sonnet), you need to assign the recorded calls to the right combo.

The design uses **one TCP port per session**. When `tracker.track(combo_id="gpt4o")` is called, the proxy opens a new listening socket on a random port (say 59198). All traffic arriving on that port belongs to that session.

For in-process agents, the httpx patch reads the port from a `ContextVar` and rewrites the URL to `http://localhost:59198/...`. For subprocess agents, `HTTPS_PROXY=http://127.0.0.1:59198` is set in the subprocess's environment. Either way, traffic arrives on port 59198, and the proxy knows it belongs to the GPT-4o session.

This solves the parallel evaluation problem cleanly. If you're evaluating two combos concurrently:

- Combo A gets port 59198, its subprocess gets `HTTPS_PROXY=http://127.0.0.1:59198`
- Combo B gets port 59205, its subprocess gets `HTTPS_PROXY=http://127.0.0.1:59205`

Traffic on 59198 → combo A. Traffic on 59205 → combo B. No ambiguity, no heuristics. The OS handles the demultiplexing through its normal socket infrastructure.

## The complete flow

Here's everything that happens end-to-end when you run an evaluation:

**Startup**: `tracker.start()` constructs the proxy server and installs the httpx monkey-patch. No ports are bound yet — listeners are per-session.

**Session creation**: `tracker.track(data_id="dp_1", combo_id="gpt4o")` creates a session, binds a new TCP port for it (each port gets its own listener thread), sets the ContextVar (for in-process), and returns session env vars (for subprocess).

**In-process path**: The agent calls the OpenAI SDK → SDK calls `httpx.Client.send()` → the monkey-patch intercepts, reads the ContextVar to get the port, rewrites the URL to `http://localhost:{port}/path`, adds the `X-AgentOpt-Target` header → sends plaintext HTTP to the proxy → proxy reads the request, forwards to real OpenAI over HTTPS, records the response → returns response to the agent.

**Subprocess path**: The agent runs as a child process with `HTTPS_PROXY` and `SSL_CERT_FILE` set → the agent's HTTP library connects to the proxy port and sends CONNECT → proxy responds 200 → agent starts TLS handshake → proxy impersonates the API server using a fake certificate signed by our CA → agent accepts (our CA is in its trust store via `SSL_CERT_FILE`) → agent sends the API request through the encrypted channel → proxy decrypts, reads, forwards to real API server over a separate HTTPS connection → gets response, records token counts and latency → re-encrypts and returns to agent.

**Session teardown**: `track()` scope exits → ContextVar is reset, the session's listener socket is closed (in-flight connection threads finish on their own and write into the archived session), and all recorded calls are archived.

**Shutdown**: `tracker.stop()` restores the original `httpx.Client.send`, closes any remaining listener sockets, and flushes the cache.

## Session lifecycle

```
tracker.start()
  └── ProxyServer constructed; httpx monkey-patch installed
      └── No listener threads, no ports bound — per-session only

tracker.track(data_id="dp_1", combo_id="gpt4o+haiku")
  └── Creates a SessionInfo
  └── Binds a NEW TCP port (e.g. 59198) for this session
  └── Spawns a listener thread that accepts on that port
  └── Sets ContextVar: _session_port_var = 59198
  └── Returns session env vars for subprocess use
  └── ALL traffic on port 59198 → attributed to this session

track() exit:
  └── Restore ContextVar
  └── Close session listener socket; listener thread exits on next accept()
  └── In-flight connection threads finish and write to the archived session
  └── End session (move from active → ended in SessionManager)
```

### In-process path detail

```
agent.run(input)
  └── OpenAI SDK → httpx.Client.send()
      └── monkey-patch reads _session_port_var (ContextVar) = 59198
      └── rewrites URL: https://api.openai.com/v1/chat/completions
                      → http://127.0.0.1:59198/v1/chat/completions
      └── adds header: X-AgentOpt-Target: https://api.openai.com
      └── sends as plaintext HTTP (not HTTPS)
          └── proxy receives on port 59198 → Direct mode
              └── reads X-AgentOpt-Target → forwards to real OpenAI
              └── records CallRecord → session on port 59198
```

### Subprocess path detail

```
subprocess.run(["tb", "run", ...], env=session_env)
  └── inherits HTTPS_PROXY=http://127.0.0.1:59198 from env
  └── LiteLLM inside tb → httpx/urllib3 → sees HTTPS_PROXY
      └── sends: CONNECT api.openai.com:443 HTTP/1.1 → port 59198
          └── proxy receives on port 59198 → CONNECT mode
              └── TLS termination with local CA cert
              └── reads decrypted request → forwards to real OpenAI
              └── records CallRecord → session on port 59198
```

### Parallel evaluation

```
# Each concurrent evaluation gets its own port and env:
track(combo_id="gpt4o+gpt4o")     → port 59198
track(combo_id="gpt4o+gpt4o-mini") → port 59205

# In-process: ContextVar is async-safe, each task sees its own port.
# Subprocess: each gets its own env dict passed explicitly, not os.environ.
```

## What lives where

### `agentopt.proxy` (the proxy server)

Sync, blocking I/O, thread-per-connection.  No asyncio.  Modules:

- **`server.py`** — `ProxyServer`.  Per-session listener threads, one connection thread per accept.  Single `_process_and_respond` lifecycle (cache → forward → record → respond) shared by both arrival modes.  `http.client` for upstream forwarding; `ssl.SSLContext.wrap_socket` for CONNECT-mode TLS termination.
- **`providers.py`** — `Provider` dataclass and `ProviderRegistry`.  One per `ProxyServer` instance, so registrations don't leak between trackers.  Defines path patterns (direct-mode auto-routing) and the hostname intercept set (CONNECT-mode MITM filter).
- **`usage.py`** — pure token-extraction functions for OpenAI / Anthropic / Gemini response shapes (JSON object, JSON array, SSE).  Raises `UsageExtractionError` with a structured diagnostic on miss; never reports zero tokens silently.
- **`certs.py`** — local CA at `~/.agentopt/ca/`.  Generates a self-signed root on first run; mints per-hostname leaf certs on demand (cached in memory).  Builds a merged `bundle.pem` (system CAs + ours) so subprocesses can still reach non-LLM HTTPS hosts.
- **`cache.py`** — `ResponseCache`.  In-memory dict, optionally persisted to SQLite via a daemon flush thread.  Keyed by a hash of the request body (excluding `stream`).
- **`session.py`** — `SessionManager`.  Active and archived sessions; `add_record` checks both so a slow upstream that finishes after `end_session` doesn't drop its record.
- **`interceptor.py`** — the httpx monkey-patch.  Path-pattern set is mutable so `LLMTracker.register_provider` can extend it for runtime-registered providers.

### `agentopt.proxy.LLMTracker` (the public surface)

- `tracker.start()` / `tracker.stop()` lifecycle
- `tracker.track(data_id, combo_id, agent_id)` context manager — opens a session port and sets the ContextVar
- `tracker.get_session_env(session)` — env-var dict for subprocess agents
- `tracker.register_provider(name, base_url, path_patterns)` — extends the per-server registry, the CONNECT-mode hostname set, and the in-process httpx-patch path set, all at once
- `tracker.get_records(...)`, `tracker.get_usage(...)` — query recorded calls

### The httpx patch (one function, no business logic)

- `_is_llm_request(request)`: POST + path matches a known LLM endpoint
- Read session port from `ContextVar`
- Rewrite URL to `http://localhost:{port}/path`
- Add `X-AgentOpt-Target` header with original base URL

### The subprocess redirect (env vars, no logic)

- `HTTPS_PROXY=http://127.0.0.1:{port}`
- `SSL_CERT_FILE=~/.agentopt/ca/bundle.pem`
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

This updates three places at once: the per-server provider registry (direct-mode routing), the CONNECT-mode hostname set (subprocess MITM filter), and the in-process httpx-patch path set (so `_is_llm_request` returns true for the new path).

## Why each piece exists

Every layer of complexity is forced by a real constraint, not a design choice:

1. We want to observe LLM calls → patch httpx in-process
2. But subprocess agents have their own httpx → need a network proxy
3. But network traffic is encrypted → need MITM with fake certificates
4. But the agent rejects fake certificates → need a custom CA in the trust store
5. But we need to know which calls belong to which combo → use a separate port per session
6. But `os.environ` isn't safe for parallel subprocesses → pass env explicitly to each subprocess instead of mutating globals

## Scoping constraints

1. **Python only** — the proxy is Python.  CA trust config targets Python and Node HTTP clients (`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`).
2. **HTTP only** — no gRPC, no WebSocket.  HTTP/HTTPS covers 99%+ of current LLM API traffic.
3. **Build CONNECT tunneling from scratch** — no mitmproxy dependency.  A minimal CONNECT handler for known LLM API hostnames is a few hundred lines of stdlib (`socket`, `ssl`, `http.client`).
4. **Docker uses sidecar proxy** — the proxy runs inside the container.  Self-contained, no host networking dependency.

## Implementation notes

A few non-obvious correctness invariants the implementation has to honour:

- **CONNECT prelude reads must not be buffered.**  If the proxy's CONNECT-line / header parser uses `socket.makefile()`, the `BufferedReader` can pull bytes beyond the header terminator into user space — and on a fast or pipelined client, those over-read bytes include the start of the TLS ClientHello.  Those bytes would never reach `ssl_ctx.wrap_socket` and the handshake fails intermittently.  The implementation reads the prelude with one-byte `recv` calls, only switching to `makefile` after `wrap_socket` is up.
- **`Content-Length` must be parsed strictly.**  Silently defaulting a missing or malformed header to 0 either truncates the body upstream *or* desynchronizes the keep-alive parser (the unread body bytes get misread as the next request line, corrupting every subsequent request on the connection).  The implementation raises on missing/invalid Content-Length for body-bearing methods.
- **Token-usage extraction must not silently report zero.**  A successful (HTTP 200) call whose usage we can't parse is a real proxy gap, not a 0-token call.  `usage.py` raises `UsageExtractionError` with a diagnostic naming exactly which keys it searched and which were present; the server attaches that to `CallRecord.error` and uses a `<parse-failed>` sentinel for the model name so the failure surfaces in result summaries.
- **Late records must not be dropped.**  A blocking upstream request can finish *after* the session's `track()` scope exits.  `SessionManager.add_record` looks up both active and ended sessions so the late record lands in the archive instead of being lost.

## Open questions

1. **CA certificate compatibility**: do any Python LLM SDKs override `SSL_CERT_FILE` or pin certificates?  Need to test against `openai`, `anthropic`, `google-generativeai`, `boto3`.

2. **Agents that bypass `HTTPS_PROXY`**: some SDK versions may hardcode direct connections.  Mitigation: maintain a compatibility matrix per SDK version.

3. **Real-time streaming forwarding**: SSE / `Transfer-Encoding: chunked` responses are *parsed* correctly today (the SSE token-extractor handles Anthropic's split `message_start` / `message_delta` usage events and Gemini's `usageMetadata`), but the proxy currently buffers the full upstream response before sending it back to the client — so the client doesn't see chunks land in real time.  For evaluation workflows this is fine (you read the final result anyway).  For interactive use you'd want chunk-by-chunk forwarding while still accumulating SSE frames for token extraction.
