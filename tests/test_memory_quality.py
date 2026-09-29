"""Recall-quality scenarios (issue #18): deterministic, free checks against real retrieval/
compaction mechanics - no LLM call, no real cost. These lock in what was found by actually running
scripts/eval_memory_quality.py for real (see docs/memory-quality.md for the live-verified results,
which this file's mocked/fake-embedding tests can't reproduce on their own - a real run is the only
way to see whether a real model's summarization or code-writing behavior changes).
"""
from pathlib import Path

import pytest

from core.learning import _fallback_embedding
from core.session import state_dir
from evals.memory_quality import ScenarioResult, aseed_memory_db, build_report
from memory.db import EMBEDDING_DIM, MemoryDB
from memory.retriever import HybridRetriever


class FakeAgent:
    """A get_embedding that behaves like the real hash fallback (semantic_available False) -
    matches what a provider with no real embedding endpoint does in production."""
    semantic_available = False

    async def get_embedding(self, text):
        return _fallback_embedding(text, EMBEDDING_DIM)


# ── seeding lands in the right per-workspace path ───────────────────────────

async def test_aseed_memory_db_writes_into_the_workspaces_own_memory_db(tmp_path: Path):
    await aseed_memory_db(tmp_path, [{"content": "a seeded fact"}], agent=FakeAgent())
    db_path = state_dir(tmp_path, create=False) / "memory.db"
    assert db_path.exists()
    db = MemoryDB(str(db_path))
    assert db.keyword_search("seeded fact")
    db.close()


async def test_aseed_memory_db_can_set_an_explicit_timestamp(tmp_path: Path):
    await aseed_memory_db(tmp_path, [{"content": "an old fact", "timestamp": "2001-01-01T00:00:00"}], agent=FakeAgent())
    db = MemoryDB(str(state_dir(tmp_path, create=False) / "memory.db"))
    row = db.conn.execute("SELECT timestamp FROM memories WHERE content = ?", ("an old fact",)).fetchone()
    db.close()
    assert row[0] == "2001-01-01T00:00:00"


# ── the real, verified finding: conflicting memories have no recency signal ─

async def test_conflicting_memories_are_not_ranked_by_recency_today(tmp_path: Path):
    """Regression/documentation test for the real finding in docs/memory-quality.md: retrieve()
    never reads the timestamp column, so a stale memory can rank ABOVE the current one that
    explicitly supersedes it. This does not assert that's *correct* - it's not - it documents
    today's actual behavior per issue #18's "findings are recorded, even if nothing needs fixing
    yet" instruction (here, something does need fixing - a natural follow-up, not this issue's
    scope, per its own non-goals)."""
    agent = FakeAgent()
    await aseed_memory_db(tmp_path, [
        {"content": "Team decision: this project uses TAB characters for indentation.", "timestamp": "2020-01-01T00:00:00"},
        {"content": "Team decision (supersedes the earlier one): this project now uses SPACES for indentation, not tabs.", "timestamp": "2030-01-01T00:00:00"},
    ], agent=agent)
    db = MemoryDB(str(state_dir(tmp_path, create=False) / "memory.db"))
    retriever = HybridRetriever(db, agent)
    results = await retriever.retrieve("what indentation style should I use in this project?")
    db.close()
    assert len(results) == 2
    # Both come back (context isn't lost); it's the RANKING that has no recency awareness - the
    # two scores are close enough that either can lead, which is itself the finding.
    assert abs(results[0]["score"] - results[1]["score"]) < 0.01


async def test_a_stale_memory_is_surfaced_with_no_awareness_that_it_is_stale(tmp_path: Path):
    """retrieve() has no way to cross-reference a memory against current file content - a renamed
    function is recalled exactly as confidently as a correct one. Documented, not fixed, per this
    issue's non-goals (re-architecting retrieval is a separate, natural follow-up)."""
    agent = FakeAgent()
    await aseed_memory_db(tmp_path, [
        {"content": "The function that computes the order total in utils.py is called sum_items()."},
    ], agent=agent)
    db = MemoryDB(str(state_dir(tmp_path, create=False) / "memory.db"))
    retriever = HybridRetriever(db, agent)
    results = await retriever.retrieve("which function in utils.py computes the total?")
    db.close()
    assert results and "sum_items" in results[0]["content"]


# ── compaction input actually contains the early constraint ────────────────

async def test_summarize_prompt_carries_a_turn_one_constraint_through_many_filler_turns():
    """A precondition for the model to be ABLE to preserve a constraint: it must still be present
    in the text sent to summarize(), not dropped before the model even sees it. Whether the model's
    own summary keeps it is a real-LLM question, verified live and recorded in
    docs/memory-quality.md - not something this offline test can check."""
    from core.providers import ModelConfig
    from main import MotionAgent

    agent = MotionAgent(ModelConfig(name="t", endpoint="http://x", provider_type="local"), memory_path=":memory:")
    captured = {}

    async def fake_complete(prompt, system_prompt=""):
        captured["prompt"] = prompt
        return "a summary"

    agent.provider.complete = fake_complete
    turns = [("never touch anything under legacy/", "understood")] + [(f"filler {i}", f"answer {i}") for i in range(20)]
    await agent.summarize(turns)
    assert "legacy/" in captured["prompt"]


# ── report building ──────────────────────────────────────────────────────────

def test_build_report_aggregates_scenario_results_and_total_cost():
    results = [
        ScenarioResult("a", "rel a", True, "down a", True, cost_usd=0.001),
        ScenarioResult("b", "rel b", False, "down b", None, cost_usd=0.002),
    ]
    report = build_report(results)
    assert [s["name"] for s in report["scenarios"]] == ["a", "b"]
    assert report["scenarios"][1]["downstream_ok"] is None
    assert report["total_cost_usd"] == pytest.approx(0.003)


# ── CLI ──────────────────────────────────────────────────────────────────────

async def test_cli_runs_only_the_requested_scenario_and_writes_a_report(tmp_path, monkeypatch):
    import argparse

    import scripts.eval_memory_quality as cli

    async def fake_scenario(workdir, provider_id):
        return ScenarioResult("fake", "rel", True, "down", True, cost_usd=0.01)

    monkeypatch.setattr(cli, "SCENARIOS", {"fake": fake_scenario})
    out_path = tmp_path / "report.json"
    code = await cli.main_async(argparse.Namespace(scenario=None, provider=None, out=str(out_path)))
    assert code == 0 and out_path.exists()
    import json

    report = json.loads(out_path.read_text())
    assert report["scenarios"][0]["name"] == "fake" and report["total_cost_usd"] == pytest.approx(0.01)
