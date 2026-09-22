"""Per-turn budgets: limits, the forced wrap-up answer, sub-agent accounting, /budget and headless flags."""
import io
import json
from pathlib import Path

import pytest

from core.budget import Budget
from core.headless import run_headless
from core.providers import StreamEvent
from tests.test_agent_loop import Scripted, call, make_agent, run, text, tool_msgs
from tests.test_tui_flow import send, system_lines, tui_app, wait_idle


def usage(prompt, completion):
    return [StreamEvent("usage", usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})]


# ── the limits themselves ───────────────────────────────────────────────────

def test_budget_config_parsing_ignores_junk():
    b = Budget.from_config({"budget": {"max_steps": "12", "max_tokens": 0, "max_cost_usd": "abc", "max_seconds": 30}}.get)
    assert (b.max_steps, b.max_tokens, b.max_cost_usd, b.max_seconds) == (12, None, None, 30.0) and b.active
    assert not Budget.from_config({}.get).active and not Budget.from_config({"budget": "x"}.get).active
    assert Budget().describe() == "no limits" and "12 steps" in b.describe()


def test_each_limit_reports_a_readable_reason():
    u = {"prompt_tokens": 90_000, "completion_tokens": 10_000, "total_tokens": 100_000}
    opts = {"input_mtok": 1.0, "output_mtok": 5.0}
    assert "5 model steps" in Budget(max_steps=5).exceeded(steps=5, usage=u, elapsed=1)
    assert Budget(max_steps=5).exceeded(steps=4, usage=u, elapsed=1) is None
    assert "100,000 tokens" in Budget(max_tokens=100_000).exceeded(steps=1, usage=u, elapsed=1)
    assert "$0.1400" in Budget(max_cost_usd=0.10).exceeded(steps=1, usage=u, elapsed=1, options=opts)
    assert "12s" in Budget(max_seconds=10).exceeded(steps=1, usage=u, elapsed=12)


def test_a_cost_limit_never_fires_for_unpriced_or_free_models():
    u = {"prompt_tokens": 10**7, "completion_tokens": 10**6, "total_tokens": 11 * 10**6}
    assert Budget(max_cost_usd=0.01).exceeded(steps=1, usage=u, elapsed=1, provider_type="cloud", options={}) is None   # unpriced
    assert Budget(max_cost_usd=0.01).exceeded(steps=1, usage=u, elapsed=1, provider_type="local") is None              # free


# ── the loop ────────────────────────────────────────────────────────────────

def reads(n, per_step_prompt=1000):
    return [call(str(i), "read_file", path="f.txt") + usage(per_step_prompt, 50) for i in range(n)]


async def budgeted(tmp_path, steps, **limits):
    (tmp_path / "f.txt").write_text("x")
    agent = make_agent(Scripted(steps))
    agent.budget = Budget(**limits)
    resp, chunks, traces = await run(agent, workspace=str(tmp_path))
    return resp, agent.provider, traces


async def test_step_budget_forces_a_final_tool_free_answer(tmp_path: Path):
    resp, provider, traces = await budgeted(tmp_path, reads(3) + [text("Best answer from what I read.") + usage(1200, 30)], max_steps=3)
    assert resp.startswith("Best answer from what I read.") and "Stopped early" in resp and "3 model steps" in resp
    assert len(provider.requests) == 4                                       # 3 working steps + the wrap-up, not 9
    last = provider.requests[-1]
    assert last["tools"]                     # still declared: Anthropic rejects tool_use history without tools; calls are ignored
    assert "Budget reached: 3 model steps" in str(last["messages"][-1]["content"])
    assert [r for s, r in traces if s == "budget_hit"][0]["reason"].startswith("3 model steps")


async def test_token_budget_counts_provider_usage(tmp_path: Path):
    resp, provider, _ = await budgeted(tmp_path, reads(3, per_step_prompt=6000) + [text("wrapped") + usage(7000, 20)], max_tokens=15_000)
    assert resp.startswith("wrapped") and "tokens" in resp and len(provider.requests) == 4     # 3 x 6.05k >= 15k


async def test_time_budget(tmp_path: Path, monkeypatch):
    import core.agent_loop as loop

    clock = iter(x * 10.0 for x in range(1000))
    real = loop.time.monotonic
    monkeypatch.setattr(loop.time, "monotonic", lambda: next(clock))
    resp, provider, _ = await budgeted(tmp_path, reads(3) + [text("in time")], max_seconds=25)
    monkeypatch.setattr(loop.time, "monotonic", real)
    assert "Stopped early" in resp and "limit 25s" in resp


async def test_no_budget_or_unmet_budget_changes_nothing(tmp_path: Path):
    resp, provider, traces = await budgeted(tmp_path, reads(2) + [text("done")], max_steps=50, max_tokens=10**9)
    assert resp == "done" and not any(s == "budget_hit" for s, _ in traces)
    (tmp_path / "f.txt").write_text("x")
    resp2, _, _ = await run(make_agent(Scripted(reads(2) + [text("done")])), workspace=str(tmp_path))
    assert resp2 == "done"


async def test_a_model_that_ignores_the_wrap_up_and_emits_a_tool_call_still_ends_the_turn(tmp_path: Path):
    xml_call = '<motion_tool>{"name":"read_file","arguments":{"path":"f.txt"}}</motion_tool>'
    resp, provider, _ = await budgeted(tmp_path, reads(2) + [text("Partial findings. " + xml_call), text("never used")], max_steps=2)
    assert "Partial findings." in resp and "<motion_tool>" not in resp and "Stopped early" in resp
    assert len(provider.requests) == 3                                       # nothing ran after the forced answer


async def test_a_wrap_up_step_that_only_calls_a_tool_gives_a_clear_placeholder_and_runs_nothing(tmp_path: Path):
    resp, provider, _ = await budgeted(tmp_path, reads(4), max_steps=2)      # the 3rd (forced) step is just another tool call
    assert "no answer could be produced within the budget" in resp and "Stopped early" in resp
    assert len(provider.requests) == 3 and len(tool_msgs(provider.requests[-1])) == 2   # only the 2 real reads ever ran


async def test_subagent_tokens_count_toward_the_leads_budget(tmp_path: Path):
    from tests.test_subagents import Router, task

    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha") + usage(500, 10), text("wrapped up") + usage(600, 10)],
               subs={"alpha": [text("big report") + usage(50_000, 200)]})
    agent = make_agent(p)
    agent.budget = Budget(max_tokens=40_000)
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert "Stopped early" in resp and any(s == "budget_hit" for s, _ in traces)   # the sub-agent's 50k alone tripped it


async def test_a_subagent_is_not_cut_off_by_the_leads_budget(tmp_path: Path):
    from tests.test_subagents import Router, task

    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha"), text("done")], subs={"alpha": [call("s1", "read_file", path="f.txt"), call("s2", "read_file", path="f.txt"), text("report")]})
    agent = make_agent(p)
    agent.budget = Budget(max_steps=1)                                       # the lead's own steps; sub-agents finish their job
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert len(p.sub_requests) == 3 and "report" in str(p.lead_requests[1]["messages"])


# ── headless ────────────────────────────────────────────────────────────────

async def test_headless_flags_set_limits_and_the_result_says_they_were_hit(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    out = io.StringIO()
    await run_headless("go", out=out, err=io.StringIO(), workspace=str(tmp_path), output_format="json",
                       limits={"max_steps": 2, "max_tokens": None},
                       agent_factory=lambda: make_agent(Scripted(reads(2) + [text("summary")])))
    data = json.loads(out.getvalue())
    assert data["ok"] and "Stopped early" in data["result"] and data["budget_hit"].startswith("2 model steps")
    assert data["steps"] == 3 and data["result"].startswith("summary")


# ── TUI ─────────────────────────────────────────────────────────────────────

async def test_budget_command_shows_sets_validates_and_clears(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        agent = app.state.agent
        await send(app, pilot, "/budget")
        await pilot.pause(0.1)
        assert any("no limits" in t and "/budget steps 12" in t for t in system_lines(app))
        for cmd in ("/budget steps 12", "/budget cost $0.5", "/budget tokens 200000", "/budget seconds 90"):
            await send(app, pilot, cmd)
            await pilot.pause(0.1)
        b = agent.budget
        assert (b.max_steps, b.max_cost_usd, b.max_tokens, b.max_seconds) == (12, 0.5, 200_000, 90.0)
        await send(app, pilot, "/budget steps banana")
        await send(app, pilot, "/budget cores 4")
        await pilot.pause(0.1)
        assert agent.budget.max_steps == 12 and sum("Usage: /budget" in t for t in system_lines(app)) == 2
        await send(app, pilot, "/budget off")
        await pilot.pause(0.1)
        assert not agent.budget.active


async def test_a_budgeted_tui_turn_ends_with_the_stopped_early_note(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("x")
    async with tui_app(tmp_path, monkeypatch, reads(2) + [text("Here is what I found.")]) as (app, pilot):
        app.state.agent.budget = Budget(max_steps=2)
        await send(app, pilot, "look around")
        await wait_idle(app, pilot)
        assert "Stopped early" in app.state.last_agent_response and "Here is what I found." in app.state.last_agent_response
