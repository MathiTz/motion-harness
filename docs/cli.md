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
