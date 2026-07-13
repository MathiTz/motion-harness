# 🚀 Motion Harness

Motion Harness is a high-performance, self-evolving AI agent framework designed for technical workflows. Unlike static agents, Motion Harness treats every interaction as a learning opportunity, building a local, persistent knowledge base of successful trajectories and environment-specific constraints.

## ✨ Key Features

- **🧠 Hybrid Memory Recall**: Combines `sqlite-vec` (Semantic) and `FTS5` (Keyword) search to ensure flawless context retrieval without hallucinations.
- **🦴 Caveman Token Compression**: A specialized internal protocol that strips conversational fluff from agent-to-agent communication, reducing token costs by up to 50%.
- **🛠️ Model Agnostic**: Native support for Local (Ollama, vLLM), Cloud (Anthropic, OpenAI), and Proxy endpoints.
- **🎓 Self-Learning Loop**: Automatically "crystallizes" successful task completions into reusable `.md` skills.
- **🎨 Themed TUI**: A professional terminal interface with multiple VS Code-inspired themes (One Dark, Nord, Dracula).

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- SQLite 3.40+ (with FTS5 enabled)

### Setup
```bash
git clone git@github.com:MathiTz/motion-harness.git
cd motion-harness
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Configure your provider**:
   Edit `.env` or `providers.yaml` to set your model endpoint and API key.

2. **Launch the Harness**:
   ```bash
   python main.py
   ```

3. **Ingest Documentation**:
   Use the `/ingest <path_to_folder>` command to feed the agent your project documentation.

## 🏗️ Architecture
The harness operates on a **Cognitive Loop**:
`User Input` $\rightarrow$ `Hybrid Recall` $\rightarrow$ `Model Execution` $\rightarrow$ `Caveman Compression` $\rightarrow$ `TUI Output`.
