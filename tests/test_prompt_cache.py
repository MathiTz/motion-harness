"""Prompt-cache friendliness (issue #35): a provider that caches the prompt prefix must not have
that prefix rewritten under it, and cached tokens must be visible and priced as such.

Measured on a real turn before the change: trimming old tool output on every step rewrote results
the previous request had just sent in full, so only ~77% of prompt tokens were cached (opencode:
~96% on the same task). These tests pin the mechanism, not the percentage, which depends on the model.
"""
from pathlib import Path

import pytest

from core.context import CACHE_FRIENDLY_TRIM_TOKENS, trim_old_tool_results, trim_when_needed
from core.pricing import estimate_cost, turn_cost
from core.providers import StreamEvent
from core.trajectory import insights, totals
from tests.test_agent_loop import Scripted, call, make_agent, run, text


def tool_msg(chars: int) -> dict:
    return {"role": "tool", "name": "read_file", "tool_call_id": "x", "content": "y" * chars}


def history(n_results: int, chars: int) -> list:
    msgs = [{"role": "user", "content": "go"}]
    for _ in range(n_results):
        msgs += [{"role": "assistant", "content": ""}, tool_msg(chars)]
    return msgs


# ── the trimming policy ──────────────────────────────────────────────────────

def test_without_a_caching_provider_old_results_are_trimmed_every_step_as_before():
    msgs = history(6, 4000)
    assert trim_when_needed(msgs, "", 32768, prompt_cache_seen=False) > 0
    sizes = [len(m["content"]) for m in msgs if m["role"] == "tool"]
    assert sizes[-2:] == [4000, 4000] and all(s < 1000 for s in sizes[:-2])  # all but the newest two trimmed


def test_with_a_caching_provider_a_small_conversation_is_left_alone():
    msgs = history(6, 4000)               # ~6k tokens, far below the trigger
    before = [dict(m) for m in msgs]
    assert trim_when_needed(msgs, "", 1_000_000, prompt_cache_seen=True) == 0
    assert msgs == before                  # byte-identical: the cached prefix survives


def test_with_a_caching_provider_a_large_conversation_is_trimmed_in_one_batch():
    msgs = history(12, 8000)              # ~24k tokens, above the trigger
    removed = trim_when_needed(msgs, "", 1_000_000, prompt_cache_seen=True)
    assert removed > 0
    sizes = [len(m["content"]) for m in msgs if m["role"] == "tool"]
    assert sizes[-2:] == [8000, 8000] and all(s < 1000 for s in sizes[:-2])   # newest two kept in full
    # and now it is small again, so the next many steps will not touch the prefix
    assert trim_when_needed(msgs, "", 1_000_000, prompt_cache_seen=True) == 0


def test_the_trigger_never_exceeds_a_third_of_a_small_window():
    msgs = history(5, 4000)               # ~5k tokens
    assert trim_when_needed(msgs, "", 8_000, prompt_cache_seen=True) > 0   # 8k window: trigger is 2.8k, not 20k
    assert CACHE_FRIENDLY_TRIM_TOKENS == 20_000


# ── through the real loop ────────────────────────────────────────────────────

def usage(prompt: int, cached: int) -> list:
    return [StreamEvent("usage", usage={"prompt_tokens": prompt, "completion_tokens": 10, "total_tokens": prompt + 10,
                                         **({"cached_tokens": cached} if cached else {})})]


async def prefix_rewrites(tmp_path: Path, cached: int) -> int:
    """How many already-sent messages differ in a later request, for a run of five big reads."""
    (tmp_path / "big.txt").write_text("".join(f"line {i} padding padding padding\n" for i in range(3000)))
    steps = [call(str(i), "read_file", path="big.txt", offset=1 + i * 300, limit=300) + usage(2000, cached) for i in range(5)]
    steps.append(text("done") + usage(2000, cached))
    p = Scripted(steps, window=1_000_000)
    agent = make_agent(p)
    await run(agent, workspace=str(tmp_path))
    rewrites = 0
    for prev, cur in zip(p.requests, p.requests[1:]):
        for a, b in zip(prev["messages"], cur["messages"]):
            rewrites += a != b
    return rewrites


async def test_a_caching_provider_never_has_sent_messages_rewritten(tmp_path: Path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    assert await prefix_rewrites(tmp_path / "a", cached=1500) == 0          # reports cache hits: prefix stays put
    assert await prefix_rewrites(tmp_path / "b", cached=0) > 0              # no cache reported: trimmed as before


async def test_cache_use_is_remembered_across_turns(tmp_path: Path):
    p = Scripted([text("hi") + usage(1000, 600), text("again") + usage(1000, 900)], window=1_000_000)
    agent = make_agent(p)
    await run(agent, workspace=str(tmp_path))
    assert getattr(agent, "prompt_cache_seen", False) is True


# ── visibility and price ─────────────────────────────────────────────────────

def test_trajectory_totals_and_insight_say_how_much_was_cached():
    recs = [{"step": i, "agent": "lead", "prompt_tokens": 10_000, "cached_tokens": 8_000, "completion_tokens": 50,
             "duration_s": 1.0, "tools": []} for i in range(1, 5)]
    t = totals(recs)
    assert t["prompt_tokens"] == 40_000 and t["cached_tokens"] == 32_000
    line = next(i for i in insights(recs) if "cache" in i)
    assert "80%" in line and "fresh" in line


def test_zero_cached_tokens_over_many_steps_is_called_out_rather_than_hidden():
    recs = [{"step": i, "agent": "lead", "prompt_tokens": 5_000, "completion_tokens": 50, "duration_s": 1.0, "tools": []}
            for i in range(1, 8)]
    assert any("no cached prompt tokens" in i for i in insights(recs))


def test_cached_tokens_are_priced_at_cached_mtok_only_when_that_is_known():
    opts = {"input_mtok": 1.0, "output_mtok": 4.0}
    assert estimate_cost(opts, 1_000_000, 0, cached_tokens=800_000) == pytest.approx(1.0)       # upper bound: full input price
    assert estimate_cost({**opts, "cached_mtok": 0.1}, 1_000_000, 0, 800_000) == pytest.approx(0.2 + 0.08)
    assert estimate_cost({**opts, "cached_mtok": 0.1}, 1000, 0, cached_tokens=5000) == pytest.approx(1000 * 0.1 / 1e6)  # clamped
    assert estimate_cost({"input_mtok": 1.0}, 10, 10, 5) is None                                 # unpriced stays unpriced
    assert turn_cost("local", opts, 10**6, 0, 10**6) == 0.0
