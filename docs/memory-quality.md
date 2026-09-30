# Memory & compaction recall-quality evaluation

Issue #18: the harness's hybrid-recall memory (`memory/db.py`, `memory/retriever.py`) and context
compaction (`core/context.py`, `MotionAgent.summarize`) had unit/integration tests confirming their
*mechanics* work, but nothing checked whether what recall actually surfaces is correct — a fast
retriever that returns plausible-looking but wrong or stale context can make the agent perform
worse than having no memory at all.

This is about recall quality *within* a correctly-scoped project — cross-project leakage is #17
(fixed, merged), and every scenario here runs against a freshly isolated workspace to make sure
it's actually testing what it claims to.

## Running it

```bash
python scripts/eval_memory_quality.py                                    # all three scenarios
python scripts/eval_memory_quality.py --scenario conflicting_memories
python scripts/eval_memory_quality.py --provider ollama-cloud/deepseek-v4-flash
```

Two of the three scenarios make real, billed provider calls; the third (compaction) always does —
scripting the summarization step would only test the script's own assumptions, not whether a real
model actually preserves anything. Not run in CI, same as `scripts/run_evals.py` and
`scripts/live_check.py`.

## Results

Total cost: **$0.0071** for the original three-scenario run (2026-09-28); the conflicting-memories
scenario was re-run once more (2026-09-29, $0.0048) after the fix described below.

### 1. Conflicting memories — relevance: **fixed**, downstream: OK

Seeded two memories: an old one ("this project uses TAB characters for indentation") and a newer
one that explicitly supersedes it ("this project now uses SPACES for indentation, not tabs").
Queried `HybridRetriever` directly with "what indentation style should I use in this project?"

**Original finding (2026-09-28): the stale (tabs) memory ranked *above* the current (spaces)
one** — scores 0.0164 vs 0.0161, a difference of one rank position in a reciprocal-rank-fusion tie.
Verified in code why: `HybridRetriever.retrieve()` never read the `timestamp` column
`memory/db.py`'s schema already carried on every row — ranking was pure keyword/semantic relevance
with no recency signal at all. Two memories that are topically identical (both clearly "about
indentation") but factually opposed were, at that point, a coin flip.

At the time, the **downstream task still succeeded**: asked to write `add.py` "following this
project's established indentation convention," the agent produced space-indented code. Both
memories fit in context together (`top_k=5`, only 2 existed), and the model itself picked up on
"supersedes the earlier one" in the text and resolved the conflict correctly. The retrieval-quality
gap was real even though it hadn't caused visible harm yet — precisely the "measure both, they can
diverge" argument for why this issue asks for two separate measurements rather than one, and
exactly the kind of gap a slightly different query or a model that doesn't happen to notice
"supersedes" could turn into a real, silent wrong answer.

**Fixed (2026-09-29):** `HybridRetriever` now reads each memory's timestamp and, when the top two
fused scores are within `RECENCY_TIE_MARGIN` (15%) of each other, breaks the tie by recency instead
of leaving the order to incidental fusion arithmetic — this is a targeted fix for near-ties
specifically, not a general time-decay ranking model. Every recalled chunk shown to the model
(`core/agent_loop.py`'s `_format_recalled_chunk`) is now also labeled with its age ("recorded
today" / "recorded 6y ago") and, for a semantic match, its raw cosine confidence, flagged
explicitly as weak below `HybridRetriever.WEAK_SEMANTIC_SCORE` (0.5) — a fix to the adjacent problem
that a 0.4-confidence guess used to be pasted into the prompt looking exactly as authoritative as a
strong match, with no way for the model (or a person reading the trace) to tell the difference.

**Re-verified for real (2026-09-29):** same two seeded memories, same query — the current (spaces)
memory now ranks first (scores 0.0161 / 0.0164, current one on top), and the downstream task still
passes. `tests/test_memory_quality.py`'s `test_near_tied_conflicting_memories_are_broken_by_recency`
locks this in as a permanent regression test, plus two tests guarding the tie-break's edges: a clear
non-tied winner is never displaced by a newer but weaker match, and a missing/unparseable timestamp
never sorts as if it were recent.

**Deliberately not done:** a universal "good" cosine-similarity threshold, or a general recency
decay curve, would need calibration against this harness's own real embedding-score distribution,
which wasn't available to reason from honestly — both `RECENCY_TIE_MARGIN` and `WEAK_SEMANTIC_SCORE`
are explicit, revisitable starting points (see their doc comments in `memory/retriever.py`), not
values derived from real data.

### 2. Stale memory — relevance: surfaced (expected), downstream: OK

Seeded one memory naming a function that doesn't exist under that name in the real file
(`utils.py` actually has `calculate_total()`; the memory said `sum_items()`). Asked the retriever
which function computes the total.

**Finding: the stale memory is surfaced with full confidence** — retrieval has no way to
cross-reference a memory against current file content, so a renamed/removed reference is recalled
exactly as if it were still accurate. Expected, given how retrieval works; not a bug to fix in this
issue (see Non-goals), but now measured and documented rather than assumed.

**Downstream, no harm occurred**: asked to "add a 10% discount to the function in utils.py that
computes the order total," the agent correctly modified `calculate_total()` — it read the real file
rather than blindly trusting the memory's function name. The stale memory didn't mislead the agent
into acting on the wrong (nonexistent) name, because real tool use (reading the actual file) caught
what memory alone couldn't.

### 3. A turn-1 constraint surviving compaction — downstream: OK

Built an 11-turn conversation: turn 1 states "never touch anything under `legacy/`, no matter what,"
followed by 10 unrelated filler turns (list files, explain the config loader, docstrings, TODOs,
etc.). Called `MotionAgent.summarize()` for real (the same method `/compact` uses) on the whole
thing.

**Finding: the real summary preserved the constraint**, explicitly calling it out as a "Constraint"
/ "Hard constraint" separate from the filler, even noting the filler turns were mostly unanswered
placeholders. Two runs, same result. This scenario has no meaningful "relevance" measurement (it's
about summarization, not retrieval), so only downstream success applies here.

## What this does and doesn't mean

- These are three scripted scenarios I wrote from reading the retrieval/compaction code, **not**
  from an existing report of Motion using stale or wrong context in practice. If real usage has hit
  a specific memory-quality failure, that's better evidence than these and should take priority —
  see this issue's own "assumptions" section.
- One real run each, at one provider/model, is not a statistically robust sample — `--repeat`-style
  averaging (as `scripts/skill_ab_test.py` does for skill promotion) would strengthen this if it
  becomes a recurring check; it wasn't done here to keep the first pass cheap and fast.
- The one confirmed gap this issue's measurement surfaced (conflicting memories weren't ranked by
  recency) was real, and was fixed as a follow-up once found — this issue's own point was to make
  that finding possible at all, not to guarantee every finding stays open.
