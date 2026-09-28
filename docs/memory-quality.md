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

## Results (2026-09-28, `ollama-cloud/deepseek-v4.1-flash`, real run)

Total cost: **$0.0071** for all three scenarios.

### 1. Conflicting memories — relevance: **BAD**, downstream: OK

Seeded two memories: an old one ("this project uses TAB characters for indentation") and a newer
one that explicitly supersedes it ("this project now uses SPACES for indentation, not tabs").
Queried `HybridRetriever` directly with "what indentation style should I use in this project?"

**Finding: the stale (tabs) memory ranked *above* the current (spaces) one** — scores 0.0164 vs
0.0161, a difference of one rank position in a reciprocal-rank-fusion tie. Verified in code why:
`HybridRetriever.retrieve()` never reads the `timestamp` column `memory/db.py`'s schema already
carries on every row — ranking is pure keyword/semantic relevance with no recency signal at all.
Two memories that are topically identical (both clearly "about indentation") but factually opposed
are, today, a coin flip.

Despite that, the **downstream task still succeeded**: asked to write `add.py` "following this
project's established indentation convention," the agent produced space-indented code. Both memories
fit in context together (`top_k=5`, only 2 existed), and the model itself picked up on "supersedes
the earlier one" in the text and resolved the conflict correctly. The retrieval-quality gap is real;
it just didn't cause real harm in this instance, precisely the "measure both, they can diverge"
argument for why this issue asks for two separate measurements rather than one.

**Not fixed here** — a recency signal (or explicit conflict detection) in `HybridRetriever` is a
natural follow-up, out of this issue's own scope (see Non-goals). `tests/test_memory_quality.py`'s
`test_conflicting_memories_are_not_ranked_by_recency_today` locks in this finding with a fast, free,
deterministic test so a future fix has something to turn from red to green.

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
- The one confirmed gap (conflicting memories aren't ranked by recency) is real and worth a fix, but
  fixing it is explicitly out of this issue's scope — this issue is the measurement existing and its
  first real results, not a promise to re-architect retrieval.
