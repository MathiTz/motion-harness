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

## Phase 2 — Document & File Ingestion (next)

Adopts the opencode model of **base64 data-URL content parts** with **capability
gating**, rather than blind OCR/text extraction. The model either reads the file
natively or the harness says it can't and tells the user why.

**Goal:** drop a file (image / PDF / DOCX / XLSX) and the agent understands it,
without hallucinating or faking content.

- [ ] **Image support (vision)**
  - [ ] Convert attached images to base64 `data:<mime>;base64,...` content parts
  - [ ] Detect a vision-capable model (`capabilities.input.image`)
  - [ ] If unsupported, degrade gracefully: replace with
        `ERROR: Cannot read image (this model does not support image input)`
  - [ ] Empty / corrupt image guard before sending
- [ ] **PDF support**
  - [ ] Treat `application/pdf` as its own modality (`pdf`)
  - [ ] Pass natively to models that accept PDFs; capability-gate otherwise
- [ ] **DOC / DOCX support**
  - [ ] Text extraction via `python-docx` (paragraphs + tables) when the model
        cannot ingest DOCX natively
  - [ ] Chunk + embed extracted text into `MemoryDB` for Hybrid Recall
- [ ] **XLSX (Excel) support**
  - [ ] Sheet → CSV-like text via `openpyxl`
  - [ ] Chunk + embed into `MemoryDB` for Hybrid Recall
- [ ] **Unified attachment pipeline**
  - [ ] `/attach <path>` (or drag-in) in the composer
  - [ ] Route by MIME: image → vision part; pdf → pdf part; doc/docx/xlsx → text
        extraction → memory
  - [ ] Capability gating mirrors opencode (`mimeToModality` + `input[modality]`)
  - [ ] Graceful error messaging that informs the user (never silently drops)

## Phase 3 — Provider & Model Enhancements

- [ ] Multimodal payload support in `core/providers.py` (image/pdf content parts)
- [ ] Per-model capability manifest (`input.image`, `input.pdf`, …)
- [ ] Model switching preserves attachments (re-attach on provider change)
- [ ] Optional vision-model fallback path for non-vision models (OCR for scanned
      pages via `pytesseract`)

## Phase 4 — Memory & Orchestration

- [ ] Document-level memory (per-file retrieval namespaces)
- [ ] Auto-compact conversation context under configurable token thresholds
- [ ] Persistent multi-session context across restarts
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
