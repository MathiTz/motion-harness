"""Recall-quality scenarios (issue #18): does the memory system retrieve the *right* project facts,
ignore *stale* ones, and preserve a constraint stated early in a long session - as opposed to
`tests/test_session_context.py`/`tests/test_agent_loop.py`, which confirm recall's *mechanics* work
(called with a timeout, writes gated by auto_remember, etc.) but never check whether what comes back
is actually correct or helpful.

Two measurements per scenario, per the issue's own scope:
- relevance: did the retriever surface the right chunk for a given query (a classic IR check,
  against core.memory.retriever.HybridRetriever directly, no LLM call, free and deterministic).
- downstream success: did having that memory in context actually change the agent's real behavior
  for the better or worse (a real, billed headless run, reusing core.headless.run_headless - the
  same engine evals/lib.py's task runner uses, per issue #13).

Findings are recorded, not fixed here - per the issue's own non-goal, this measures the existing
retrieval/compaction algorithms, it doesn't replace them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.session import state_dir
from evals.lib import _run_headless_capturing


async def aseed_memory_db(workspace: Path, entries: List[Dict[str, Any]], *, agent=None) -> None:
    """Write ``entries`` (each ``{"content": str, "timestamp": "ISO 8601 str" (optional)}``) into
    ``workspace``'s own memory DB (issue #17's per-workspace path), embedded the same way a real
    agent would (``agent.get_embedding`` if given - the hash fallback when the configured provider
    has no real embedding model, exactly like production) so the seeded data isn't artificially
    more or less "findable" than a real memory would be."""
    from memory.db import EMBEDDING_DIM, MemoryChunk, MemoryDB

    db_path = state_dir(workspace) / "memory.db"
    db = MemoryDB(str(db_path))
    try:
        for entry in entries:
            embedding = await agent.get_embedding(entry["content"]) if agent is not None else None
            if not embedding:
                from core.learning import _fallback_embedding

                embedding = _fallback_embedding(entry["content"], EMBEDDING_DIM)
            mem_id = db.add_memory(MemoryChunk(content=entry["content"], embedding=embedding, metadata={}, mem_type="DOC"))
            if entry.get("timestamp"):
                db.conn.execute("UPDATE memories SET timestamp = ? WHERE id = ?", (entry["timestamp"], mem_id))
        db.conn.commit()
    finally:
        db.close()


@dataclass
class ScenarioResult:
    name: str
    relevance_finding: str
    relevance_ok: Optional[bool]  # None if relevance isn't a meaningful concept for this scenario
    downstream_finding: str
    downstream_ok: Optional[bool]  # None if not run (e.g. no provider available)
    cost_usd: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


def _build_agent(provider_id: Optional[str], workspace: Path):
    """Always pass workspace: without it, build_agent falls back to the harness's own shared,
    real, historically-accumulated motion_memory.db (main.py's _default_memory_path with no
    workspace) - exactly the cross-project leakage issue #17 fixed - which would make every
    scenario here query real, unrelated production memories instead of the isolated ones seeded
    for the test."""
    from core.headless import build_agent

    return build_agent(provider_id, str(workspace))


async def scenario_conflicting_memories(workdir: Path, provider_id: Optional[str]) -> ScenarioResult:
    """Two memories that disagree ("tabs" recorded first, "spaces" recorded later, as a corrected
    decision) - does recall surface the current one, or does the agent get confused by both?"""
    from memory.retriever import HybridRetriever

    agent = _build_agent(provider_id, workdir)
    old = {"content": "Team decision: this project uses TAB characters for indentation.", "timestamp": "2020-01-01T00:00:00"}
    new = {"content": "Team decision (supersedes the earlier one): this project now uses SPACES for indentation, not tabs.", "timestamp": "2030-01-01T00:00:00"}
    await aseed_memory_db(workdir, [old, new], agent=agent)

    retriever = HybridRetriever(agent.memory, agent)
    results = await retriever.retrieve("what indentation style should I use in this project?")
    top = results[0]["content"] if results else ""
    # Both memories mention indentation strongly; report whether the CURRENT one is unambiguously first.
    relevance_ok = bool(results) and "now uses SPACES" in top
    relevance_finding = (
        f"top result: {top[:90]!r}" if results else "no results returned"
    ) + f" ({len(results)} total, scores: {[round(r['score'], 4) for r in results]})"

    summary = await _run_headless_capturing(
        "Write a new file add.py containing exactly one function, add(a, b), that returns a + b, "
        "following this project's established indentation convention.",
        provider_id=provider_id, workspace=str(workdir),
    )
    written = (workdir / "add.py")
    downstream_ok = None
    downstream_finding = "add.py was not created"
    if written.exists():
        text = written.read_text()
        uses_tabs = "\n\t" in text
        uses_spaces_indent = any(line.startswith("    ") for line in text.splitlines())
        downstream_ok = uses_spaces_indent and not uses_tabs
        downstream_finding = f"add.py indentation: {'tabs' if uses_tabs else ('spaces' if uses_spaces_indent else 'unclear')}"

    return ScenarioResult(
        "conflicting_memories", relevance_finding, relevance_ok, downstream_finding, downstream_ok,
        cost_usd=summary.get("cost_usd") or 0.0,
    )


async def scenario_stale_memory(workdir: Path, provider_id: Optional[str]) -> ScenarioResult:
    """A memory about a function that no longer exists under that name after a later rename - is it
    still recalled, and does it mislead the agent into acting on the wrong (nonexistent) name?"""
    from memory.retriever import HybridRetriever

    (workdir / "utils.py").write_text("def calculate_total(items):\n    return sum(items)\n")
    agent = _build_agent(provider_id, workdir)
    stale = {"content": "The function that computes the order total in utils.py is called sum_items().", "timestamp": "2020-01-01T00:00:00"}
    await aseed_memory_db(workdir, [stale], agent=agent)

    retriever = HybridRetriever(agent.memory, agent)
    results = await retriever.retrieve("which function in utils.py computes the total?")
    relevance_ok = bool(results) and "sum_items" in results[0]["content"]  # True = the stale fact IS surfaced (expected - retrieval has no way to know it's stale)
    relevance_finding = (
        f"stale memory surfaced: {results[0]['content'][:80]!r}" if results else "nothing surfaced"
    )

    summary = await _run_headless_capturing(
        "Add a 10% discount to the function in utils.py that computes the order total.",
        provider_id=provider_id, workspace=str(workdir),
    )
    content = (workdir / "utils.py").read_text()
    downstream_ok = "calculate_total" in content and ("0.9" in content or "10%" in content or "discount" in content.lower())
    downstream_finding = (
        "correctly modified calculate_total() despite the stale memory naming sum_items()" if downstream_ok
        else f"did NOT correctly apply the discount to calculate_total(); final utils.py:\n{content[:400]}"
    )

    return ScenarioResult(
        "stale_memory", relevance_finding, relevance_ok, downstream_finding, downstream_ok,
        cost_usd=summary.get("cost_usd") or 0.0,
    )


async def scenario_constraint_survives_compaction(workdir: Path, provider_id: Optional[str]) -> ScenarioResult:
    """A constraint stated in turn 1 of a long session ("never touch legacy/") - does /compact's
    real model-written summary (MotionAgent.summarize, core/agent_config's compact_with_model)
    still carry it once dozens of unrelated turns have piled up around it, or does it get silently
    dropped along with the tool-call noise?"""
    agent = _build_agent(provider_id, workdir)
    turns = [("Important constraint for this whole project: never touch anything under the legacy/ directory, no matter what.",
              "Understood - I will not touch anything under legacy/ for the rest of this session.")]
    filler_topics = [
        "list the files in the src directory", "what does the config loader do",
        "add a docstring to the main function", "check if there are any TODO comments",
        "explain what the test suite covers", "summarize the recent changes",
        "is there a linter configured", "what Python version does this target",
        "are there any circular imports", "describe the directory layout",
    ]
    for i, topic in enumerate(filler_topics):
        turns.append((topic, f"Here is the answer about: {topic}. (filler turn {i})"))

    summary = await agent.summarize(turns)
    downstream_ok = "legacy" in summary.lower()
    downstream_finding = (
        f"summary {'DOES' if downstream_ok else 'does NOT'} mention legacy/ after {len(turns)} turns:\n{summary[:500]}"
    )
    return ScenarioResult(
        "constraint_survives_compaction",
        relevance_finding="not applicable - this scenario is about summarization, not retrieval",
        relevance_ok=None,
        downstream_finding=downstream_finding, downstream_ok=downstream_ok,
    )


SCENARIOS = {
    "conflicting_memories": scenario_conflicting_memories,
    "stale_memory": scenario_stale_memory,
    "constraint_survives_compaction": scenario_constraint_survives_compaction,
}


def build_report(results: List[ScenarioResult]) -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": [
            {
                "name": r.name, "relevance_ok": r.relevance_ok, "relevance_finding": r.relevance_finding,
                "downstream_ok": r.downstream_ok, "downstream_finding": r.downstream_finding,
                "cost_usd": r.cost_usd, "detail": r.detail,
            }
            for r in results
        ],
        "total_cost_usd": sum(r.cost_usd for r in results),
    }
