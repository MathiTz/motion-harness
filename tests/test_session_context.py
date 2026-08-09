"""Tests for the rolling session-context feature.

Covers:
  - AppState.update_session_context / build_context_prompt / context_summary
  - MotionAgent.run() augmenting memory recall with a context_query
  - De-duplication of recalled chunks
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.providers import ModelConfig
from main import MotionAgent
from ui.tui import AppState


# ─── AppState session context ────────────────────────────────────────────────

def test_context_empty_initial():
    state = AppState()
    assert state.build_context_prompt() == ""
    assert state.context_summary() == "No context yet"


def test_context_keeps_recent_turns_verbatim():
    state = AppState()
    for i in range(3):
        state.update_session_context(f"prompt {i}", f"response {i}")

    block = state.build_context_prompt()
    # Recent turns are kept verbatim (not folded).
    assert "Q: prompt 0" in block
    assert "A: response 0" in block
    assert "Q: prompt 2" in block
    # No prior-context folding yet (only 3 turns).
    assert "[Prior context]" not in block


def test_context_folds_older_turns():
    state = AppState()
    # Add more than KEEP (4) turns so older ones get folded into a summary.
    for i in range(7):
        state.update_session_context(f"prompt {i}", f"response {i}")

    block = state.build_context_prompt()
    # Older turns are folded into a [Prior context] summary.
    assert "[Prior context]" in block
    # The most recent 4 turns remain verbatim.
    assert "Q: prompt 3" in block
    assert "Q: prompt 6" in block
    # The very first turn is no longer verbatim (it was folded).
    assert "Q: prompt 0" not in block


def test_context_summary_reports_turn_count_and_last_prompt():
    state = AppState()
    state.update_session_context("hello", "hi there")
    state.update_session_context("how are you", "fine")
    summary = state.context_summary()
    assert "2 turns" in summary
    assert "how are you" in summary


def test_context_summary_singular():
    state = AppState()
    state.update_session_context("only one", "ok")
    assert "1 turn" in state.context_summary()


# ─── MotionAgent context_query recall ────────────────────────────────────────

class _MockProvider:
    class _Config:
        provider_type = "local"

    def __init__(self):
        self.config = self._Config()

    async def complete(self, prompt, system_prompt="", **kwargs):
        return "answer"

    async def close(self):
        pass


class _MockRetriever:
    """Records queries and returns canned results."""

    def __init__(self):
        self.queries = []
        self.results = {
            "user prompt": [{"content": "memory A", "score": 1.0}],
            "context query": [{"content": "memory B", "score": 1.0}],
        }

    async def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return self.results.get(query, [])


async def test_agent_uses_context_query_for_recall():
    agent = MotionAgent(ModelConfig(name="t", endpoint="http://localhost", provider_type="local"))
    agent.provider = _MockProvider()
    retriever = _MockRetriever()
    agent.retriever = retriever

    await agent.run("user prompt", context_query="context query")

    # Both the prompt and the context query were used for recall.
    assert "user prompt" in retriever.queries
    assert "context query" in retriever.queries


async def test_agent_dedupes_recalled_chunks():
    agent = MotionAgent(ModelConfig(name="t", endpoint="http://localhost", provider_type="local"))
    agent.provider = _MockProvider()

    class _DupRetriever:
        async def retrieve(self, query, top_k=5):
            # Both the prompt and context query return the SAME content.
            return [{"content": "same memory", "score": 1.0}]

    agent.retriever = _DupRetriever()

    # Patch the provider to capture the system prompt (which contains the
    # de-duplicated memory context).
    captured = {}

    async def fake_complete(prompt, system_prompt="", **kwargs):
        captured["system"] = system_prompt
        return "answer"

    agent.provider.complete = fake_complete

    await agent.run("user prompt", context_query="context query")

    # The same memory should appear only once in the system prompt.
    assert captured["system"].count("same memory") == 1


async def test_agent_without_context_query_uses_only_prompt():
    agent = MotionAgent(ModelConfig(name="t", endpoint="http://localhost", provider_type="local"))
    agent.provider = _MockProvider()
    retriever = _MockRetriever()
    agent.retriever = retriever

    await agent.run("user prompt")

    # Only the prompt was used; no context query.
    assert retriever.queries == ["user prompt"]


# ─── Auto-compact ────────────────────────────────────────────────────────────

def test_should_compact_below_threshold():
    state = AppState()
    state.session_metrics["total_tokens_est"] = 1000
    # 1000 < 8192 * 0.95 → no compaction.
    assert state.should_compact(8192) is False


def test_should_compact_at_threshold():
    state = AppState()
    state.session_metrics["total_tokens_est"] = 8000
    # 8000 >= 8192 * 0.95 → compact.
    assert state.should_compact(8192) is True


def test_should_compact_ignores_invalid_window():
    state = AppState()
    state.session_metrics["total_tokens_est"] = 100000
    assert state.should_compact(0) is False
    assert state.should_compact(-1) is False


def test_compact_folds_turns_and_resets_metrics():
    state = AppState()
    for i in range(5):
        state.update_session_context(f"prompt {i}", f"response {i}")
    state.session_metrics["total_tokens_est"] = 9000
    state.session_metrics["prompt_tokens_est"] = 5000
    state.session_metrics["output_tokens_est"] = 4000

    state.compact()

    # All remaining recent turns folded into a compacted summary.
    assert "[Compacted context]" in state.session_context
    assert "Q: prompt 1" in state.session_context
    assert "Q: prompt 4" in state.session_context
    # Recent turns cleared.
    assert state._context_turns == []
    assert state.conversation_turns == []
    # Token metrics reset.
    assert state.session_metrics["total_tokens_est"] == 0
    assert state.session_metrics["prompt_tokens_est"] == 0
    assert state.session_metrics["output_tokens_est"] == 0


def test_compact_preserves_context_in_build_prompt():
    state = AppState()
    state.update_session_context("hello", "hi")
    state.compact()
    # The compacted summary is still available as a context query.
    assert "[Compacted context]" in state.build_context_prompt()


# ─── Agent mode (build/plan) ──────────────────────────────────────────────────

def test_agent_mode_defaults_to_build():
    state = AppState()
    assert state.agent_mode == "build"


def test_agent_mode_toggle_cycles():
    state = AppState()
    assert state.agent_mode == "build"
    state.agent_mode = "plan" if state.agent_mode == "build" else "build"
    assert state.agent_mode == "plan"
    state.agent_mode = "plan" if state.agent_mode == "build" else "build"
    assert state.agent_mode == "build"
