# CLI & Auth

Motion Harness ships a small CLI (`motion`) plus an opencode-style auth store
for managing provider API keys.

## The `motion` command

| Command | Action |
| :-- | :-- |
| `motion` | Launch the TUI (default) |
| `motion --chat` | Launch the legacy REPL chat |
| `motion --list` | List available providers/models |
| `motion --provider <id>` | Launch with a specific provider/model |
| `motion --test` | Run the Caveman compression test (no model needed) |
| `motion auth login <provider>` | Store an API key (prompts, hidden input) |
| `motion auth logout <provider>` | Remove a stored API key |
| `motion auth list` | List which providers have keys |

Provider ids: `ollama-cloud`, `claude`, `openai`, `local-llama`, or any provider
you add to `config.yml`. Use `provider/model` to pick a specific model, e.g.
`motion --provider ollama-cloud/glm-5.2`.

## Managing API keys

Keys are stored in `~/.config/motion-harness/auth.json` with `0600` permissions
— never in `config.yml` or the shell environment.

```bash
motion auth login ollama-cloud   # prompts for your key, stored locally
motion auth login openai
motion auth login claude
motion auth list                 # see which providers have keys (masked)
motion auth logout ollama-cloud  # remove a key
```

### Lookup order

When resolving a key for a provider, the harness checks, in order:

1. **Auth store** — `~/.config/motion-harness/auth.json`
2. **Environment variable** — e.g. `OLLAMA_API_KEY`, `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`
3. **`config.yml`** — the `api_key` field

### In the TUI

- Press `Ctrl+K` → **Manage API keys** to set a key from the command palette.
- Or type `/auth list`, `/auth login <provider>`, `/auth logout <provider>` in
  the chat composer.

## Adding a provider

Add a block under `providers:` in `config.yml`:

```yaml
providers:
  my-provider:
    name: "My Provider"
    endpoint: "https://api.myprovider.com/v1"
    api_key: null          # set via `motion auth login my-provider`
    provider_type: "cloud" # "cloud" | "local" | "proxy"
    default_model: "my-model-1"
    models:
      my-model-1:
        temperature: 0.7
        max_tokens: 4096
```

Then `motion auth login my-provider`. The built-in catalog (`core/catalog.py`)
is merged with your `config.yml`, so your provider and models are picked up
automatically.

## Headless mode

```bash
motion -p "summarize README.md"                       # answer on stdout
motion -p "..." --output-format json                  # one JSON object
motion -p "..." --output-format stream-json           # NDJSON events, then the result
motion -p "..." --max-steps 10 --max-cost 0.25 --effort low   # bound a scripted run
motion -p "..." --plan --workspace ~/proj --verbose   # read-only, other dir, tool activity on stderr
echo "explain this" | motion -p -                     # prompt from stdin
git diff | motion -p "review this diff" --stdin       # stdin appended as context
```

The JSON result has `ok`, `result`, `error`, `provider`, `model`, `mode`, `steps`, `tool_calls`, `elapsed_s`, `ttft_s`, `usage`, `cost_usd` and `trajectory` (one record per model step: duration, first-token latency, prompt/completion tokens, and each tool call's arguments preview, success and result size). `stream-json` emits each record as a `{"type": "step", ...}` event as it happens. Exit codes: `0` success, `1` the turn failed (provider or tool-loop error), `2` usage/config error, `130` interrupted. `MOTION_CONFIG=/path/config.yml` selects a config file. There is no UI to approve anything, so risky commands, private-network fetches and `ask_user` are refused unless pre-approved with `permissions.commands.allow`. Headless runs don't write to long-term memory unless `remember_turns_headless: true`.

## Slash commands (inside the TUI)

| Command | Action |
| :-- | :-- |
| `/attach [path]` | Attach a file to the **next** message (no path opens a file browser) |
| `/clear` | Drop pending attachments |
| `/compact` | Summarize the conversation to free context |
| `/undo` | Revert the file changes made in the last turn |
| `/diff [on\|off]` | Show the last turn's edits in full / toggle inline diffs |
| `/jobs [stop <id\|all>]` | List background processes; stop one or all |
| `/trajectory [copy\|save [full]\|all]` | Steps of the last turn (or `all` the session): time, tokens, tool calls and result sizes, plus where the cost went. `copy` → clipboard, `save` → `.motion/trajectories/*.json` (`full` adds every message sent) |
| `/budget [steps N\|tokens N\|cost X\|seconds N\|off]` | Show or set per-turn limits (session only; persist them under `budget:` in `config.yml`) |
| `/effort [low\|medium\|high\|off]` | Set `reasoning_effort` for the current model this session |
| `/tracking [on\|off]` | Whether session transcripts are saved to `.motion/sessions/`. No argument shows the state; use `on` if you declined the first-launch prompt |
| `/new` | Start a fresh conversation (approvals and settings are kept) |
| `/resume [id]` | List saved sessions, or reload one (needs interaction tracking) |
| `/todos` | Show the agent's current task list |
| `/skill list\|show <n>\|save <n>\|delete <n>` | Manage reusable skills |
| `/mcp` | Connected MCP servers and their tools |
| `/parallel a ; b ; c` | Run sub-tasks on background workers |
| `/synthesize on\|off` | Toggle auto-crystallization into skills |
| `/auth list\|login\|logout` | Manage provider API keys |
| `/tools`, `/help` | Show tools and commands |

## config.yml reference (optional keys)

```yaml
fallback_providers:          # tried in order if the current provider is down / rate-limited / rejects the key
  - ollama-cloud/deepseek-v4-flash
  - claude
hooks:                       # your commands around tool calls (JSON on stdin; see README)
  pre_tool:                  # non-zero exit blocks the call and tells the model why
    - match: "write_file|replace_in_file|edit_files"   # regex on the tool name
      command: "./scripts/guard.sh"
      timeout: 10
  post_tool:                 # output is appended to the tool result
    - match: "write_file|replace_in_file|edit_files"
      command: "ruff format . >/dev/null; echo formatted"
budget:                      # per-turn limits, all optional
  max_steps: 12              # model calls
  max_tokens: 150000         # prompt + completion
  max_cost_usd: 0.50         # needs input_mtok/output_mtok pricing on the model
  max_seconds: 180
sandbox: auto                # auto = OS write sandbox for shell/Python where available; off disables it
sandbox_allow_read: []       # credential folders to make readable again, e.g. ["~/.aws"] (default: hidden)
sandbox_deny_read: []        # extra folders to hide, e.g. ["~/.ssh"]
sandbox_network: allow       # deny = commands get no network at all
show_diffs: true             # show edits inline as colored diffs (/diff off toggles per session)
remember_turns_headless: false  # let `motion -p` runs write to long-term memory
default_agent_mode: plan     # plan (discuss first, read-only) | build (can write/run; risky commands still ask)
remember_turns: true         # store substantive turns in memory for later recall
recall_timeout: 2.0          # seconds a memory lookup may delay a turn

permissions:
  commands:                  # shell-style globs matched against the whole command
    allow: ["git push origin feature/*"]
    ask:   ["make deploy*"]
    deny:  ["curl *"]

mcp:
  servers:
    files:  { command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "."] }
    remote: { url: "https://example.com/mcp", headers: { Authorization: "Bearer ..." } }

providers:
  my-provider:
    models:
      my-model:
        max_tokens: 8192
        input_mtok: 0.5      # USD per million input tokens  } enables cost tracking for this model
        output_mtok: 1.5     # USD per million output tokens }
        timeout: 120         # idle-read timeout in seconds (connect is always 10s)
        max_retries: 3       # 429 / 5xx / connection errors, with backoff
        native_tools: true   # false forces the text tool protocol
        vision: true         # override the image-capability heuristic
        context_window: 128000
        thinking_budget: 4096  # Anthropic extended thinking (tokens)
        reasoning_effort: low  # OpenAI-compatible reasoning models
        embed_model: text-embedding-3-small  # enables semantic memory on OpenAI-compatible providers
```
