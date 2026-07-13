# Motion Harness Setup Guide

This guide provides detailed instructions for installing and configuring Motion Harness across different environments.

## 🚀 Quick Installation (Recommended)

The fastest way to get started is using the provided installation script, which handles Docker setup and shell aliasing.

```bash
chmod +x install.sh
./install.sh
```

### What the installer does:
1. **Docker Image**: Builds the `motion-harness` image containing all Python dependencies and the SQLite FTS5 environment.
2. **Global Alias**: Adds a `motion` command to your shell configuration (`.zshrc`, `.bashrc`, or `config.fish`).
3. **Volume Mapping**: Configures the alias to mount your current working directory as the agent's workspace.

---

## 💻 Native Installation

If you prefer to run the harness natively without Docker, follow these steps.

### Prerequisites
- **Python 3.11+**
- **SQLite 3.40+** (Must have FTS5 enabled)
- **Terminal Emulator** that supports full TUI rendering (recommended: Warp, iTerm2, Alacritty).

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/MathiTz/motion-harness.git
cd motion-harness

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch
python main.py
```

---

## ⚙️ Configuration

### Provider Setup
Motion Harness supports multiple LLM backends. Edit `config.yaml` or create a `.env` file to define your providers:

```yaml
providers:
  openai:
    api_key: "sk-..."
    model: "gpt-4o"
  anthropic:
    api_key: "sk-ant-..."
    model: "claude-3-5-sonnet"
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3"
```

### Workspace Management
The agent operates within a defined workspace. When using the `motion` alias, the current directory is automatically passed as the workspace. 

---

## 🛠️ Troubleshooting

### SQLite FTS5 Syntax Errors
If you encounter `syntax error near "?"` during memory retrieval, ensure you are using SQLite 3.40+. If running natively on macOS, the system SQLite may be outdated. Install the latest via Homebrew:
```bash
brew install sqlite
```

### TUI Rendering Issues
If the interface looks distorted, ensure your terminal is set to a compatible font (e.g., Nerd Fonts) and that the window size is at least 80x24.