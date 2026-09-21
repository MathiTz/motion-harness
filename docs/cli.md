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

## Slash commands (inside the TUI)

| Command | Action |
| :-- | :-- |
| `/attach [path]` | Attach a file to the **next** message (no path opens a file browser) |
| `/clear` | Drop pending attachments |
| `/compact` | Summarize the conversation to free context |
| `/undo` | Revert the file changes made in the last turn |
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
        timeout: 120         # idle-read timeout in seconds (connect is always 10s)
        max_retries: 3       # 429 / 5xx / connection errors, with backoff
        native_tools: true   # false forces the text tool protocol
        vision: true         # override the image-capability heuristic
        context_window: 128000
        thinking_budget: 4096  # Anthropic extended thinking (tokens)
        reasoning_effort: low  # OpenAI-compatible reasoning models
        embed_model: text-embedding-3-small  # enables semantic memory on OpenAI-compatible providers
```
