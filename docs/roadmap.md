# Motion Harness Roadmap

This is the living roadmap for Motion Harness. Items are ordered roughly by
priority within each phase, not strict release order. The guiding principle is
the **Cognitive Loop** (Input → Hybrid Recall → Model Execution → Compression →
Output): every feature must keep the agent grounded in real context and cheap to
operate.

---

## Phase 1 — Core Chat UX (done)

- [x] opencode-style compact prompt panel (input + agent · model · provider meta)
- [x] Agent mode coloring — build (blue), plan (orange)
- [x] Ctrl+K command palette, Ctrl+O model dialog
- [x] Right-hand Context panel (rolling session context) replacing the activity rail
- [x] Drop top tabs; single focused chat view
- [x] Soft-wrapping, auto-growing composer (no horizontal scroll on long input)
- [x] Grounded responses — pass prior conversation (history) + session context
      (`context_query`) to the model on every turn to reduce hallucination
- [x] Remove "Thinking…" placeholder from the response area
- [x] Full theme-aware response rendering (code syntax colors follow theme)
- [x] **Auth store (opencode-style)** — `motion auth login/logout/list` + `/auth`
      commands; keys stored in `~/.config/motion-harness/auth.json` (0600), never
      in `config.yml`. Lookup order: auth store → env var → config.yml.
- [x] **Model dialog (Ctrl+O)** — lists every model individually (all cloud
      models appear once a key is set) with live search + `Ctrl+R` refresh
      (scrapes the latest Ollama Cloud model list).
- [x] Remove dead code — unused Tasks/Skills/KB/Memory/Settings panes and the
      hardcoded dashboard admin key.

> **Note:** `TaskManager` (parallel orchestration) and `SkillSynthesizer`
> (auto-crystallization) exist in `core/` but are **not yet exposed in the TUI** —
> there's no UI to spawn parallel tasks or toggle auto-synthesis. Manual skills
> work via `/skill save <name>`. Parallel orchestration is tracked in Phase 4.

## Phase 1.5 — Harness fundamentals (done)

Speed, correctness and safety work that the chat UX sits on.

- [x] Streaming everywhere (answer, reasoning, tool progress), incl. Anthropic SSE
- [x] Native tool calling (OpenAI-compatible / Anthropic / Ollama) with automatic
      fallback to the text protocol; independent tool calls run in parallel
- [x] Tools run off the event loop; `Esc` cancels and kills the process tree
      (the old Esc/Ctrl+C handler silently never worked)
- [x] The original request stays in the conversation across tool steps
- [x] Retries/backoff, split connect/idle timeouts, real usage on streams,
      configurable Anthropic endpoint + prompt caching
- [x] Live status line (phase, elapsed, first-token time), "thought for Ns",
      throttled rendering, buffered trace panel
- [x] Context hygiene: tool-result trimming, windowed `read_file`, compaction
      (`/compact` + automatic), attachments sent once
- [x] New tools: `grep`, `todo_write`, `ask_user`, `use_skill`, persistent
      `memory_save`; `.gitignore`-aware listing; diffs; read-before-overwrite
- [x] Command permission policy (allow/ask/deny), catastrophic-command block,
      secret-free child environment, SSRF guard, untrusted web/MCP content
- [x] `/undo` (per-turn file checkpoints), `/new`, `/resume`, `/todos`, `/mcp`
- [x] MCP: real persistent stdio + HTTP client, discovered tools with schemas
- [x] Memory: natural-language keyword search, rank fusion, embedding cache,
      dimension adaptation, keyword-only mode without a real embedder
- [x] Project instructions (`AGENTS.md`), environment block, saved-skill index
- [x] `/parallel` results reported in chat; tasks isolated and non-interactive
- [x] Harness state consolidated under `<workspace>/.motion/`
- [x] CI on Python 3.11 + 3.14 with lint; provider/loop/MCP/TUI test suites
- [x] OS write sandbox for shell/Python (Seatbelt / bubblewrap) + code-aware policy
- [x] Headless mode (`motion -p`, text / json / stream-json, exit codes)
- [x] Cost tracking from real usage and per-model pricing
- [x] Inline colored diffs and `/diff`
- [x] Background jobs (`job_start` / `job_output` / `job_stop`, `/jobs`)
- [x] Sub-agents (`task` tool): isolated context, parallel read-only, guarded general mode

### Still open
- [ ] Sandbox: Windows backend; hiding *all* of $HOME (only credential stores and harness secrets are hidden today)
- [ ] Sandbox on Linux is implemented but only exercised where bubblewrap works (CI runners often lack user namespaces)
- [ ] User-configurable hooks (pre/post tool)
- [ ] Multi-file patch tool; model failover; recorded-provider evals against real models
- [ ] Live verification of the Anthropic and OpenAI wire formats (mock-tested only)

## Phase 2 — Document & File Ingestion (in progress)

Adopts the opencode model of **base64 data-URL content parts** with **capability
gating**, rather than blind OCR/text extraction. The model either reads the file
natively or the harness says it can't and tells the user why.

**Goal:** drop a file (image / PDF / DOCX / XLSX) and the agent understands it,
without hallucinating or faking content.

- [ ] **Image support (vision)**
  - [x] Convert attached images to base64 image content parts (all providers)
  - [x] Detect a vision-capable model (heuristic; override with `vision:` per model)
  - [x] If unsupported, degrade gracefully and tell the user/model
  - [x] Empty / oversized (>5 MB) image guard before sending
- [ ] **PDF support**
  - [ ] Treat `application/pdf` as its own modality (`pdf`)
  - [x] Text extraction with pypdf (fixed: the old code used a removed API)
  - [ ] Pass natively to models that accept PDFs; capability-gate otherwise
- [ ] **DOC / DOCX support**
  - [x] Text extraction via `python-docx` (paragraphs + tables) when the model
        cannot ingest DOCX natively
  - [ ] Chunk + embed extracted text into `MemoryDB` for Hybrid Recall
- [ ] **XLSX (Excel) support**
  - [x] Sheet → CSV-like text via `openpyxl`
  - [ ] Chunk + embed into `MemoryDB` for Hybrid Recall
- [ ] **Unified attachment pipeline**
  - [x] `/attach <path>` (or browse) in the composer
  - [ ] Route by MIME: image → vision part; pdf → pdf part; doc/docx/xlsx → text
        extraction → memory
  - [ ] Capability gating mirrors opencode (`mimeToModality` + `input[modality]`)
  - [ ] Graceful error messaging that informs the user (never silently drops)

## Phase 3 — Provider & Model Enhancements

- [x] Multimodal payload support in `core/providers.py` (image content parts; pdf pending)
- [ ] Per-model capability manifest (`input.image`, `input.pdf`, …)
- [ ] Model switching preserves attachments (re-attach on provider change)
- [ ] Optional vision-model fallback path for non-vision models (OCR for scanned
      pages via `pytesseract`)

## Phase 4 — Memory & Orchestration

- [ ] Document-level memory (per-file retrieval namespaces)
- [x] Auto-compact conversation context (summarizes at 60% of the model window)
- [x] Persistent multi-session context across restarts (`/resume`, opt-in tracking)
- [ ] Parallel orchestration with attachment-aware task scheduling

---

## Notes

- The **opencode reference** for this approach is `anomalyco/opencode`:
  - `packages/opencode/src/provider/transform.ts` — `mimeToModality()`,
    `unsupportedParts()` (capability gating + empty-image guard)
  - `packages/opencode/src/acp/content.ts` — `filePartToContentChunks()` +
    `decodeDataUrl()` (base64 data-URL decoding)
  - `packages/opencode/src/tool/code-mode.ts` — `dataUrl()` helper
- Principle: **never pretend to read a file.** If the active model can't ingest a
  modality, surface a clear message and inform the user.
