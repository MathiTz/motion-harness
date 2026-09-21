# Motion Harness Architecture

Motion Harness is designed as a **Cognitive Infrastructure**, moving away from stateless prompting toward a stateful, evolving memory system.

## 🔄 The Cognitive Loop

The core of the system is a unidirectional loop that ensures every single interaction is grounded in memory and optimized for cost.

`User Input` → `Hybrid Recall` (bounded, in parallel with context loading) → `Model step` (streamed) → `Tool calls` (parallel when independent) → … → `Final answer`

A turn is implemented in `core/agent_loop.py`. It talks to the model in one of three ways, chosen per provider: **native** tool calling (streamed; preferred), a streamed **text/XML** tool protocol (automatic fallback when an endpoint rejects `tools`), or plain `complete()` for simple/custom providers. Live output reaches the UI as marker-prefixed chunks (`_delta_`, `_think_`, `_endstep_`, `_tool_`, `_out_`) documented at the top of that module.

### 1. Hybrid Recall (The Memory Layer)
Two retrievers fused with reciprocal-rank fusion:
- **Keyword track**: SQLite `FTS5`. The query is the significant terms of the prompt joined with `OR` (BM25-ranked), not the whole prompt as one exact phrase.
- **Semantic track**: `sqlite-vec` cosine search, used **only when a real embedding model is available** (local Ollama, or `embed_model` on an OpenAI-compatible provider). Without one, the hash-based fallback vectors carry no meaning, so semantic search is skipped rather than returning unrelated memories.
- Recall is bounded by `recall_timeout`, skipped for an empty DB, and never fails a turn. Substantive turns are stored back (`remember_turns`), and `memory_save` notes persist here too.

### 2. Model Execution (The Provider Layer)
The harness uses a provider abstraction that allows for seamless routing:
- **Local**: Ollama/vLLM for privacy-critical tasks.
- **Cloud**: Claude/GPT for high-reasoning tasks.
- **Proxy**: Custom endpoints for specialized model lairs.

### 3. Caveman Filter (optional)
A small reversible filter that strips stock filler phrases from output handed to another agent. It is not applied to model input or to replies shown to the user; see the README for the honest scope.

### 4. Tools, permissions and safety
Tools are declared once in `core/tool_specs.py`; that registry generates both the native tool schemas and the text-protocol prompt. `core/workspace_tools.py` implements them (async, process-tree kill on cancel). `core/permissions.py` classifies shell commands (allow / ask / deny); `core/toolstate.py` holds cross-turn session state (approvals, undo checkpoints, read tracking, todos). MCP servers (`core/mcp.py`) are connected once, their tools discovered with `tools/list` and exposed as `mcp__<server>__<tool>`.

### 5. Parallel Orchestration
`TaskManager` (`core/orchestrator.py`) runs background agents behind an `asyncio` semaphore sized from the CPU count, exposed in the TUI as `/parallel a ; b`. Each task gets a private in-memory store and **no interactive callbacks**, so anything that would need a prompt is refused. Transcripts are saved under `<workspace>/.motion/tasks/`; completion is reported in the chat.
