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
  <a href="docs/skills.md">Skills Engine</a>
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

```bash
# 1. Run the installer
chmod +x install.sh && ./install.sh

# 2. Refresh your shell
source ~/.config/fish/config.fish  # or .zshrc / .bashrc

# 3. Configure your provider
cp config.example.yml config.yml
# Edit config.yml — set your API keys in .env or directly in config.yml

# 4. Launch
motion
```

### CLI Usage

```
motion                                    # Launch TUI (default)
motion --provider ollama-cloud/glm-5.2    # TUI with specific provider/model
motion --chat                             # Legacy REPL mode
motion --list                             # List available providers/models
motion --test                             # Run Caveman compression test
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

**6-Tab Workspace**: Chat, Tasks, Skills, Knowledge, Memory, Settings — tab headers are the canonical navigation (no duplicate button bars).

**Chat workspace regions**:
- **Conversation canvas** — dominant ~70–75% width, markdown-first message cards with author/time headers and compact metadata.
- **Interaction trace side panel** — progressive disclosure: collapsed by default with a summary chip (`trace · N events · last stage`); expand with `F8` or by clicking the chip.
- **Composer / action bar** — grouped primary actions (Trace, Copy, Stats) and secondary actions (Save Skill) with a single-line session metrics strip.

**Theme token model**: semantic tokens (`$background`, `$surface`, `$panel`, `$border`, `$primary`, `$accent`, `$text`, `$text-muted`, `$success`, `$warning`, `$error`) cascade through every widget via Textual's theme system. One emphasized border per visual region (chat canvas keeps `$primary`; trace, tasks, skills, KB, memory, settings use `$border`).

**4 Native Themes**: One Dark, Solarized Light, Nord, Dracula — cycle with `Ctrl+T`.

**Live task monitoring**: real-time status of parallel workers with ⏳⚙️✅❌ indicators and a global activity rail (`Ctrl+B` to toggle).

**Keyboard shortcuts**:
| Key | Action |
| :-- | :-- |
| `Ctrl+1..6` / `Alt+1..6` / `F1..F6` | Switch tab (Chat, Tasks, Skills, KB, Memory, Settings) |
| `Ctrl+]` / `Ctrl+[` / `Ctrl+←→` | Next / previous tab |
| `Ctrl+T` | Cycle theme |
| `Ctrl+B` | Toggle activity rail |
| `F8` / `Ctrl+Shift+T` | Toggle interaction trace panel |
| `F9` / `Ctrl+Shift+C` | Copy last assistant response |
| `?` | Show shortcuts overlay (generated from live bindings) |
| `Enter` (chat input) | Send message |
| `/skill save <name>` | Save last reply as a skill |
| `Ctrl+C` | Cancel current request (does not quit) |
| `Ctrl+Q` | Quit |

**Dashboard integration**: one-click open to the admin dashboard at `https://localhost:7860/`.

### ⚠️ Known Limitations (v2 TUI)
- Trace persistence is per-session (not yet written to disk).
- Theme contrast validation is manual; the bundled themes are tuned for readability but very-low-contrast combinations are not auto-corrected.
- Shortcut help overlay (`?`) reflects MainScreen + ChatPane bindings; pane-local bindings in Tasks/Skills/KB/Memory/Settings are not yet enumerated.
- Clipboard copy falls back to inserting the response into the input box when the terminal lacks clipboard support.

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

# Headless TUI smoke: compose MainScreen, tab nav, F8 trace toggle, ? overlay
python - <<'PY'
import asyncio, sys
from textual.app import App
from ui.tui import MainScreen, AppState
from ui.themes import ThemeRegistry

class SmokeApp(App): pass
async def run():
    app = SmokeApp()
    for tid in ThemeRegistry.theme_ids():
        app.register_theme(ThemeRegistry.get_textual_theme(tid))
    async with app.run_test() as pilot:
        app.push_screen(MainScreen(AppState()))
        await pilot.pause()
        await pilot.press("ctrl+2"); await pilot.pause()
        await pilot.press("ctrl+1"); await pilot.pause()
        await pilot.press("f8"); await pilot.pause()
        await pilot.press("f8"); await pilot.pause()
        await pilot.press("question_sign"); await pilot.pause()
        await pilot.press("escape"); await pilot.pause()
        sys.stderr.write("SMOKE OK\n")
asyncio.run(run())
PY
```

Checks cover: MainScreen compose, tab navigation (`Ctrl+1`/`Ctrl+2`), trace disclosure toggle (`F8`), and the shortcuts overlay (`?`/`Escape`).

For more details on our versioning and changelog, see [RELEASES.md](RELEASES.md).
