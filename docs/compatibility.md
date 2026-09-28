# Provider & MCP compatibility matrix

Issue #15: which integrations are actually verified against the real, live service, versus only
proven correct against a mock transport this test suite wrote itself. Both kinds of testing are
real and both are in the suite, but they answer different questions — a mock proves the code
matches the *documented* wire format; a live run proves the *real service* still behaves that way
today. Presenting them identically hid that difference. Update this table whenever a live check is
re-run, and record the date and what changed.

## How to (re-)verify

```bash
python scripts/live_check.py                          # the default provider from config.yml/.env
python scripts/live_check.py claude/claude-haiku-4-5   # a specific provider/model
python scripts/live_check.py --all                     # every provider that currently has a key
```

Makes real, billed requests — a handful of tiny ones per provider (see the script's own docstring).
Not run in CI (real cost; see Non-goals below). Re-run it and update this table whenever a
provider's SDK, your `.env`/`config.yml`, or `core/providers.py`'s adapter for it changes.

## Providers

| Provider | Streaming | Tool-call round trip | Cancellation | Usage reporting | Failover classification |
| --- | --- | --- | --- | --- | --- |
| `ollama-cloud` | ✅ verified live (2026-09-28, `deepseek-v4.1-flash`) | ✅ verified live (2026-09-28) | ✅ verified live (2026-09-28) | ✅ verified live (2026-09-28) | ✅ verified live (2026-09-28, real 401) |
| `claude` (Anthropic) | mock-tested only | mock-tested only | mock-tested only | mock-tested only | mock-tested only (`tests/test_failover.py`) |
| `openai` | mock-tested only | mock-tested only | mock-tested only | mock-tested only | mock-tested only (`tests/test_failover.py`) |
| `local-llama` | not applicable — needs a locally running Ollama; not attempted in this pass | | | | |
| `claude-cli` (delegate) | not verified in this environment — `claude` binary not on `PATH` here | | | | |
| `codex-cli` (delegate) | not verified in this environment — `codex` binary not on `PATH` here | | | | |

All five checks above come from `scripts/live_check.py` (`streams text`, `tool call round trip`,
`cancellation mid-stream`, `usage reporting accuracy`, `failover error classification` — the last
two and cancellation were added by this issue; the first three already existed).

**`ollama-cloud` verification (2026-09-28):** ran live, real network calls, real cost (~$0.005 for
the four original checks; the three new ones added a few more tiny requests). All seven checks
passed, including a genuine 401 from a deliberately invalid key, correctly classified as
failover-worthy by `core/agent_loop.py`'s `TurnRunner._failover_worthy`.

**`claude` / `openai`: cannot be completed in this environment.** No Anthropic or OpenAI API key
exists anywhere reachable here — checked the auth store (`~/.config/motion-harness/auth.json`,
only holds an `ollama-cloud` key) and the environment (`.env`'s `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`
lines are present but empty). This matches issue #15's own stated blocker exactly. **Running
`scripts/live_check.py claude` and `scripts/live_check.py openai` with a real key, and updating
this table with the result, is the one piece of this issue that needs a maintainer to do.**

**A real bug found while investigating this blocker, fixed alongside this table:**
`ConfigManager.has_api_key("claude")` derived its expected env var name from the provider id
(`CLAUDE_API_KEY`) instead of the vendor's documented name for that endpoint's host
(`ANTHROPIC_API_KEY` for `api.anthropic.com` — `core/providers.py`'s `_HOST_KEY_ENV`, which the
*request-time* key lookup already used correctly). `openai`'s id happens to match its own env var
name, so only `claude` was affected. Net effect: a user who set `ANTHROPIC_API_KEY` in `.env`
exactly as `.env.example`, `README.md` and `docs/setup.md` all document would still see the model
picker show `claude` as locked, get told to set `CLAUDE_API_KEY` (which nothing else ever mentions),
and have `scripts/live_check.py` and any configured `fallback_providers` silently skip it. Fixed in
`core/config.py` — `has_api_key`, `get_provider_config`'s key resolution, and `unavailable_reason`'s
suggested env var name now all consult the same host-based mapping the request path already used.
This does **not** by itself supply a real key — the blocker above is unchanged — but it means a
maintainer who *does* have one and follows the documented setup will actually get to use it.

## MCP

| Server | Transport | Status |
| --- | --- | --- |
| `tests/mcp_echo_server.py` (house-built) | stdio | mock/house-built — proves protocol mechanics, not real-world interop |
| HTTP transport | HTTP | mock-tested only (`httpx.MockTransport`, `tests/test_mcp.py`) |
| `@modelcontextprotocol/server-filesystem` (official reference server, [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)) | stdio, via `npx` | ✅ **verified live** (2026-09-28) — `tests/test_mcp_real_server.py` |

**Why the filesystem server:** it's the official reference implementation, small, stable, and
requires no account/API key — just Node (`npx`) on `PATH`. Run via stdio, the same transport
`core/mcp.py` already supports and the only one the house-built echo server exercises, so this adds
real-world confidence for the actually-most-common case without needing a second code path.

**Network/CI note:** `tests/test_mcp_real_server.py` is skipped automatically wherever `npx` isn't
on `PATH` (`.github/workflows/ci.yml` doesn't install Node, so it's skipped there — this is an
on-demand, maintainer/local check, not a CI gate, matching issue #15's own non-goals). The first run
in a fresh environment needs network access to fetch the package from the npm registry; `npx`
caches it after that.

Verified live: tool discovery returns the server's own real tool set (`read_text_file`,
`write_file`, `list_directory`, and 11 others — not anything this test suite invented), a real
file written to disk by the agent's own `write_file` MCP call, a real file read back byte-for-byte,
and the server's own error (a missing file) correctly surfacing as `MCPError`.

## Non-goals (per issue #15)

- No new provider adapters — this verifies what exists (Anthropic, OpenAI, Ollama-compatible, the
  two CLI delegates), not adding new ones.
- No CI job runs live checks on every PR — real API cost, and the CLI delegates need a human login
  that can't safely live in CI secrets. This is on-demand, maintainer-run verification, re-run
  periodically and recorded here, not a merge gate.
