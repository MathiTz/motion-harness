<p align="center">
  <img src="logo.svg" width="200" alt="Motion Harness Logo">
</p>

<p align="center">
  <strong>The self-evolving AI agent harness for high-precision technical workflows.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · 
  <a href="docs/setup.md">Setup Guide</a> · 
  <a href="docs/architecture.md">Architecture</a> · 
  <a href="docs/skills.md">Skills Engine</a> · 
  <a href="docs/roadmap.md">Roadmap</a>
</p>

---

## 🌌 Overview

**Motion Harness** isn't just another agent wrapper; it is a cognitive infrastructure. While standard agents suffer from "context drift" and token inefficiency, Motion Harness implements a persistent **Cognitive Memory Loop**. 

It treats every successful task trajectory as a learning event, crystallizing experience into reusable skills and compressing communication to the absolute theoretical minimum.

### ⚡ The Core Edge

| Feature | The "Standard" Way | The Motion Way |
| :--- | :--- | :--- |
| **Memory** | Simple RAG / Vector Search | **Hybrid Recall** (Semantic + FTS5 Keyword) |
| **Tokens** | Natural Language Verbosity | **Caveman Compression** (Bidirectional Noise Reduction) |
| **Scaling** | Sequential Execution | **Parallel Orchestration** (CPU-aware concurrency) |
| **Growth** | Static Prompting | **Skill Synthesis** (Automatic procedural crystallization) |

---

## 🚀 Quick Start

Get the harness running in under 60 seconds.

### For complete beginners

**1. Install Python 3.11+** if you don't have it. On macOS: `brew install python@3.11`. On Linux: `sudo apt install python3 python3-venv`.

**2. Clone and install the harness:**
```bash
git clone <your-repo-url> motion-harness
cd motion-harness
chmod +x install.sh && ./install.sh
```

**3. Refresh your shell** so the `motion` command is available:
```bash
source ~/.config/fish/config.fish   # if you use fish
source ~/.zshrc                     # if you use zsh
source ~/.bashrc                    # if you use bash
```

**4. Configure your provider (API key):**
```bash
cp config.example.yml config.yml
```
Then add your API key with the `auth` command (opencode-style):
```bash
motion auth login ollama-cloud   # prompts for your key, stored locally
motion auth login openai
motion auth login claude
motion auth list                 # see which providers have keys
motion auth logout ollama-cloud # remove a key
```
Keys are stored in `~/.config/motion-harness/auth.json` (0600 perms) — never in `config.yml`. You can also use env vars (`OLLAMA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), including a `.env` file in the project root — it is loaded automatically on every launch path (`motion`, `motion --chat`, etc.), not just the legacy REPL. Lookup order: **auth store → env var → config.yml**.
> **No need to add models one by one.** The harness ships with a built-in catalog of Anthropic, OpenAI, and Ollama Cloud models. Just add the API key and they're all available — press `Ctrl+O` to browse/search them, or press `Ctrl+A` from anywhere in the TUI to jump straight to auth management. Providers without a stored key prompt for one inline instead of blocking you.

**5. Launch:**
```bash
motion
```

### CLI Usage

```
motion                                    # Launch TUI (default)
motion --provider ollama-cloud/glm-5.2    # TUI with specific provider/model
motion --chat                             # Legacy REPL mode
motion --list                             # List available providers/models
motion --test                             # Run Caveman compression test
motion auth login <provider>              # Store an API key
motion auth logout <provider>             # Remove a stored API key
motion auth list                          # List stored API keys
```

*For detailed native installation and GPU configuration, see the [Setup Guide](docs/setup.md).*

---

## 🛠️ Deep Capabilities

### 🧠 Hybrid Cognitive Memory
Combines the nuance of vector embeddings with the precision of SQLite FTS5. Whether you need a "concept" or a "specific variable name," Motion finds it instantly.

### 🦴 Caveman Protocol
A bidirectional compression layer that strips conversational fluff. 
- **Input**: Natural language $\rightarrow$ Compressed tokens.
- **Output**: Compressed tokens $\rightarrow$ Natural language.
- **Result**: $\sim 50\%$ reduction in token overhead without loss of intent.

### 🎓 Self-Learning Synthesis
When a complex task is solved, the harness doesn't just forget. It analyzes the trajectory and "crystallizes" the steps into a `.md` skill, allowing the agent to execute the same complex workflow in the future with a single reference.

### 🎨 Pro-Grade TUI
A high-performance terminal interface built with `Textual`, designed for daily-driver clarity rather than an engineering dashboard.

**Focused single-chat view (opencode-style)**: no top tab bar. The screen is a top bar, the conversation canvas, a compact bottom composer, and a right-hand **Context panel**. Skills, settings, and model switching are all reachable via the command palette.

**Workspace regions**:
- **Conversation canvas** — markdown-first message cards with author/time headers and compact metadata; response code blocks carry theme-aware syntax highlighting.
- **Right Context panel** — the rolling session context and most recent turns, so you always see what the model is grounded against (`Ctrl+B` to toggle).
- **Composer** — a compact, opencode-style prompt: auto-growing (soft-wraps instead of scrolling), a thick left accent border colored by agent mode, prompt history (↑/↓), and an inline `agent · model · provider` meta row. Enter sends, Shift+Enter adds a newline.

**Agent modes**: `build` (blue) has write access and creates/edits files; `plan` (orange, opencode-style) is read-only and produces a concrete written plan before you approve implementation. Toggle with `Tab` or via the command palette. Plan mode blocks `write_file`/`replace_in_file` with a soft guidance nudge so the model answers with a plan rather than asking you to repeat yourself.

**Grounded responses**: on every turn the harness passes the prior conversation (user + assistant) as history and the rolling session context as a memory-recall query, so the model references what was actually said instead of hallucinating. Long sessions are capped at the most recent 8 turns to keep the latest user message prominent.

**Theme token model**: semantic tokens (`$background`, `$surface`, `$panel`, `$border`, `$primary`, `$secondary`, `$accent`, `$text`, `$text-muted`, `$success`, `$warning`, `$error`) cascade through every widget via Textual's theme system.

**10 Native Themes**: OpenCode (default), One Dark, Solarized Light, Nord, Dracula, Omni Dark, and the full [Catppuccin](https://catppuccin.com/) family (Mocha, Macchiato, Frappé, Latte) — press `Ctrl+T` to open the theme menu (arrow keys preview live, `Enter` confirms, `Esc` reverts). Your choice is saved to `config.yml` (`default_theme`) and restored on the next launch.

**Interaction trace panel** (`F8`): a right-hand panel that shows the live tool loop as human-readable sentences, e.g. "about to run read_file on `main.py`", "write_file finished on `README.md`", or "read_file failed on `missing.txt`: file does not exist". Trace events include the affected path where applicable, plus diagnostics such as `step_cap_hit` (the tool loop hit its safety cap) and `loop_warning` (the same tool+path/command repeated 3x in a row). Unexpected turn failures are also logged with a full traceback to `motion.log` for debugging.

**Real token usage**: when the active provider reports usage, the session footer and per-turn metadata show real prompt/completion/total token counts instead of a character-based estimate; the estimate is used only as a fallback when a provider doesn't return usage data.

**Agent tools**: in `build` mode the agent can call `read_file`, `write_file`, `replace_in_file`, `list_files`, and `run_command` (arbitrary shell commands in the workspace, with a timeout and truncated output) against the workspace. Tool calls that target a path outside the workspace no longer fail outright — you're prompted to **allow once**, **allow for the session**, or **deny**.

**Model persistence**: switching models via `Ctrl+O` (or the startup provider picker) saves your choice as `last_provider` in `config.yml`, so the next `motion` launch reconnects to the same provider/model instead of resetting to the catalog default. An explicit `motion --provider ...` flag always overrides this for that one run and is not persisted.

**Resilient tool loop**: malformed tool calls and individual tool errors are treated as context rather than immediately stopping the agent. Only `Esc` (or an explicit `tool_error`/`Esc` stop signal) halts the loop, and the UI removes stale "Queued" notices as soon as a queued prompt starts running. Provider HTTP calls time out after 30s so cancellation feels responsive.

**Keyboard shortcuts**:
| Key | Action |
| :-- | :-- |
| `Ctrl+T` | Open theme menu |
| `Ctrl+A` | Open auth management |
| `Ctrl+B` | Toggle context panel |
| `Ctrl+O` | Switch model (browse/search all models) |
| `Ctrl+R` (in model dialog) | Refresh the model list (scrapes latest Ollama Cloud models) |
| `Ctrl+N` (in model dialog) | Add a custom model |
| `Ctrl+K` | Command palette (all commands) |
| `Ctrl+E` | Open external editor for the message (runs in the background, no UI freeze) |
| `F7` | Toggle agent thinking (show intermediate tool-loop text inline) |
| `F8` / `Ctrl+Shift+T` | Toggle interaction trace panel |
| `F9` / `Ctrl+Shift+C` | Copy last assistant response |
| `Ctrl+Shift+K` | Copy last code block from the assistant's response |
| `Tab` | Toggle agent (build / plan) |
| `?` | Show shortcuts overlay (generated from live bindings) |
| `Enter` (chat input) | Send message |
| `Shift+Enter` (chat input) | New line |
| `↑` / `↓` (chat input) | Prompt history |
| `/skill save <name>` | Save last reply as a skill |
| `/auth list` | List stored API keys |
| `/auth login <provider>` | Store an API key for a provider |
| `/auth logout <provider>` | Remove a stored API key |
| `Esc` | Stop the current agent interaction |
| `Ctrl+Q` | Quit (kills the process) |

**CLI commands**:
| Command | Action |
| :-- | :-- |
| `motion` | Launch the TUI |
| `motion --chat` | Launch the REPL chat |
| `motion --list` | List providers/models |
| `motion --provider <id>` | Launch with a specific provider/model |
| `motion --test` | Run Caveman compression test |
| `motion auth login <provider>` | Store an API key (prompts, hidden input) |
| `motion auth logout <provider>` | Remove a stored API key |
| `motion auth list` | List which providers have keys |

### ⚠️ Known Limitations (v2 TUI)
- Trace persistence is per-session (not yet written to disk).
- Theme contrast validation is manual; the bundled themes are tuned for readability but very-low-contrast combinations are not auto-corrected.
- Clipboard copy falls back to inserting the response into the input box when the terminal lacks clipboard support.
- Conversation history sent to the model is capped at the most recent 8 turns; earlier context is summarized, not verbatim.
- File/document ingestion (image / PDF / DOCX / XLSX) is on the roadmap — see [Roadmap](docs/roadmap.md).

---

## 🗺️ Roadmap

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#fab283', 'primaryBorderColor': '#484848', 'primaryTextColor': '#eeeeee', 'lineColor': '#5c9cf5' }}}%%
flowchart LR
    subgraph P1[Phase 1 — Core Chat UX]
        direction TB
        A[opencode-style prompt panel] --> B[Agent mode colors: build=blue, plan=orange]
        B --> C[Ctrl+K palette · Ctrl+O model]
        C --> D[Right-hand Context panel]
        D --> E[Drop top tabs]
        E --> F[Grounded responses via history + context_query]
    end

    subgraph P2[Phase 2 — File Ingestion]
        direction TB
        G[Images → base64 vision parts] --> H[Capability gating]
        H --> I[PDF as own modality]
        I --> J[DOC/DOCX → text extraction → memory]
        J --> K[XLSX → sheets → memory]
        K --> L[Unified /attach pipeline + graceful errors]
    end

    subgraph P3[Phase 3 — Providers & Models]
        direction TB
        M[Multimodal payloads in providers] --> N[Per-model capability manifest]
        N --> O[OCR fallback for scanned pages]
    end

    subgraph P4[Phase 4 — Memory & Orchestration]
        direction TB
        P[Document-level memory namespaces] --> Q[Auto-compact context]
        Q --> R[Persistent multi-session context]
        R --> S[Attachment-aware parallel orchestration]
    end

    P1 --> P2 --> P3 --> P4
```

Detailed tracking lives in [docs/roadmap.md](docs/roadmap.md).

---

## 🏗️ Architecture

The harness operates on a **High-Fidelity Cognitive Loop**:

`User Input` $\rightarrow$ `Hybrid Recall` $\rightarrow$ `Model Execution` $\rightarrow$ `Caveman Compression` $\rightarrow$ `TUI Output`

For a technical breakdown of the provider abstraction and the orchestrator, visit [Architecture Docs](docs/architecture.md).

---

## 🤝 Contributing

Motion Harness is in **Active Beta**. We welcome contributions to help us reach `v1.0.0`.

### 🛠️ Contribution Workflow
1. **Fork** the repository.
2. **Create a Feature Branch** from `beta` (not `main`).
3. **Submit a PR** targeting the `beta` branch.
4. **Wait for Review**: Changes will be merged into `beta` for testing before being curated into `main`.

### 🧪 Testing
Ensure all changes are validated against the integration suite:
```bash
pytest tests/test_integration.py
```

**Visual + TUI smoke checks** (headless, no terminal required):
```bash
# Parse/import sanity
python -c "from ui import tui; print('Import OK')"

# Headless TUI smoke: compose MainScreen, Ctrl+K palette, F8 trace toggle, ? overlay
python - <<'PY'
import asyncio, sys
from textual.app import App
from ui.tui import MainScreen, AppState, CommandPalette
from ui.themes import ThemeRegistry

class SmokeApp(App): pass
async def run():
    app = SmokeApp()
    for tid in ThemeRegistry.theme_ids():
        app.register_theme(ThemeRegistry.get_textual_theme(tid))
    async with app.run_test() as pilot:
        app.push_screen(MainScreen(AppState()))
        await pilot.pause()
        await pilot.press("ctrl+k"); await pilot.pause()      # open command palette
        assert isinstance(app.screen, CommandPalette)
        await pilot.press("escape"); await pilot.pause()      # close palette
        await pilot.press("f8"); await pilot.pause()          # toggle trace
        await pilot.press("f8"); await pilot.pause()
        await pilot.press("question_sign"); await pilot.pause()  # shortcuts overlay
        await pilot.press("escape"); await pilot.pause()
        sys.stderr.write("SMOKE OK\n")
asyncio.run(run())
PY
```

Checks cover: MainScreen compose, the `Ctrl+K` command palette, trace disclosure toggle (`F8`), and the shortcuts overlay (`?`/`Escape`).

For more details on our versioning and changelog, see [RELEASES.md](RELEASES.md).
