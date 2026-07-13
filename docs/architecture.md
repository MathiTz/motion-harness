# Motion Harness Architecture

Motion Harness is designed as a **Cognitive Infrastructure**, moving away from stateless prompting toward a stateful, evolving memory system.

## 🔄 The Cognitive Loop

The core of the system is a unidirectional loop that ensures every single interaction is grounded in memory and optimized for cost.

`User Input` $\rightarrow$ `Hybrid Recall` $\rightarrow$ `Model Execution` $\rightarrow$ `Caveman Compression` $\rightarrow$ `TUI Output`

### 1. Hybrid Recall (The Memory Layer)
Instead of relying solely on vector embeddings (which can be imprecise for specific technical terms), Motion uses a dual-track retrieval system:
- **Semantic Track**: Uses `sqlite-vec` for concept-based retrieval (e.g., "How does the auth system work?").
- **Keyword Track**: Uses SQLite `FTS5` for exact match retrieval (e.g., "Find the `AUTH_TOKEN_SESS` variable").
- **Fusion**: The results are merged and ranked, providing the LLM with a high-precision context window.

### 2. Model Execution (The Provider Layer)
The harness uses a provider abstraction that allows for seamless routing:
- **Local**: Ollama/vLLM for privacy-critical tasks.
- **Cloud**: Claude/GPT for high-reasoning tasks.
- **Proxy**: Custom endpoints for specialized model lairs.

### 3. Caveman Compression (The Efficiency Layer)
To combat token bloat in long agent-to-agent trajectories, the harness applies a **Bidirectional Compression Protocol**.

- **Noise Reduction**: Strips common conversational fluff ("I understand," "Based on the context provided," etc.).
- **Semantic Preservation**: Maintains the core logic, constraints, and variables.
- **Symmetry**: The compression is reversible, meaning the TUI can "decompress" the output back into natural language for the user, while the internal agents communicate in a dense, token-efficient format.

### 4. Parallel Orchestration
The orchestrator manages task concurrency using an `asyncio` semaphore gated by the system's CPU core count. This prevents system lockup during massive parallel research tasks while maximizing throughput.