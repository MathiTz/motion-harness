"""Sub-agents (the `task` tool): isolation, parallelism and guardrails."""
import asyncio
import copy
import time
from pathlib import Path

import httpx
import pytest

import core.agent_loop as loop
from core.providers import BaseProvider, ModelConfig, StreamEvent, ToolCall
from core.toolstate import ToolSession
from core.workspace_tools import WorkspaceTools
from tests.test_agent_loop import EmptyRetriever, call, calls, make_agent, run, text, tool_msgs
from tests.test_tui_flow import _text_of, send, system_lines, tui_app, wait_idle

MARK = "You are a sub-agent"


def first_user_text(messages):
    c = messages[0]["content"]
    return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c)


class Router(BaseProvider):
    """Lead agent requests read from `lead`; each sub-agent request reads from
    `subs[keyword]` where keyword appears in its (first) user message."""

    def __init__(self, lead, subs, delay=0.0, native=True):
        super().__init__(ModelConfig(name="r", endpoint="http://x", provider_type="local",
                                     options={"native_tools": native, "model": "m", "context_window": 10_000_000}))
        self.lead, self.subs, self.delay = list(lead), {k: list(v) for k, v in subs.items()}, delay
        self.lead_requests, self.sub_requests = [], []
        self.active = self.max_active = 0

    async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
        record = {"messages": copy.deepcopy(messages), "system": system_prompt, "tools": tools}
        if MARK in system_prompt:
            self.sub_requests.append(record)
            key = next(k for k in self.subs if k in first_user_text(messages))
            step = self.subs[key].pop(0)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if isinstance(step, Exception):
                    raise step
                for ev in step:
                    yield ev
            finally:
                self.active -= 1
        else:
            self.lead_requests.append(record)
            for ev in self.lead.pop(0):
                yield ev


def task(cid, key, mode=None, **extra):
    args = {"description": key, "prompt": f"Investigate {key}", **({"mode": mode} if mode else {}), **extra}
    return call(cid, "task", **args)


def names(request):
    return {t["name"] for t in request["tools"]}


# ── isolation & reports ─────────────────────────────────────────────────────

async def test_subagent_gets_fresh_context_and_returns_only_a_report(tmp_path: Path):
    (tmp_path / "big.txt").write_text("SECRET-BULK-CONTENT " * 500)
    p = Router(
        lead=[text("Let me delegate.") + task("1", "alpha"), text("Thanks, done.")],
        subs={"alpha": [call("s1", "read_file", path="big.txt"), text("Report: the answer is in src/x.py:12")]},
    )
    resp, _, _ = await run(make_agent(p), "Please dig into alpha", workspace=str(tmp_path), agent_mode="build")
    assert resp == "Thanks, done."

    sub = p.sub_requests[0]
    assert sub["messages"] == [{"role": "user", "content": "Investigate alpha"}]   # nothing from the lead's chat
    assert MARK in sub["system"] and "Please dig into alpha" not in sub["system"]
    assert "read_file" in names(sub) and not ({"write_file", "run_command", "task"} & names(sub))   # explore = read-only, no recursion

    # The lead sees the report, not the sub-agent's bulky intermediate output.
    lead_second = p.lead_requests[1]["messages"]
    result = tool_msgs(p.lead_requests[1])[0]["content"]
    assert "src/x.py:12" in result and '"tool_calls": 1' in result
    assert "SECRET-BULK-CONTENT" not in str(lead_second)


async def test_lead_is_offered_the_task_tool_but_plain_toolsets_are_not(tmp_path: Path):
    p = Router(lead=[text("hi")], subs={})
    await run(make_agent(p), workspace=str(tmp_path))
    assert "task" in names(p.lead_requests[0])
    assert "task" not in {s["name"] for s in WorkspaceTools(tmp_path).tool_schemas()}
    assert "- task:" in WorkspaceTools(tmp_path, subagents=True).system_instructions(False)
    assert "- task:" not in WorkspaceTools(tmp_path).system_instructions(False)


async def test_subagent_progress_is_forwarded_but_its_text_is_not(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha"), text("done")],
               subs={"alpha": [call("s1", "read_file", path="f.txt"), text("SUB-ONLY-TEXT report")]})
    _, chunks, traces = await run(make_agent(p), workspace=str(tmp_path))
    await asyncio.sleep(0.05)
    assert any(c.startswith("_tool_ ↳ [alpha] read `f.txt`") for c in chunks)
    assert not any("SUB-ONLY-TEXT" in c for c in chunks)                       # would corrupt the lead's live answer
    stages = [s for s, _ in traces]
    assert "subagent_start" in stages and "subagent_done" in stages


async def test_subagent_token_usage_counts_toward_the_turn(tmp_path: Path):
    usage = [StreamEvent("usage", usage={"prompt_tokens": 700, "completion_tokens": 30, "total_tokens": 730})]
    p = Router(lead=[task("1", "alpha"), text("done")], subs={"alpha": [text("report") + usage]})
    _, _, traces = await run(make_agent(p), workspace=str(tmp_path))
    await asyncio.sleep(0.05)
    seen = [payload for stage, payload in traces if stage == "usage"]
    assert any(u.get("total_tokens") == 730 and "sub-agent" in u["message"] for u in seen)


# ── parallelism ─────────────────────────────────────────────────────────────

async def test_explore_subagents_run_in_parallel_up_to_the_cap(tmp_path: Path):
    keys = ["k1", "k2", "k3", "k4", "k5"]
    p = Router(lead=[calls(*[(str(i), "task", {"description": k, "prompt": f"Investigate {k}"}) for i, k in enumerate(keys)]),
                     text("all done")],
               subs={k: [text(f"report {k}")] for k in keys}, delay=0.3)
    t0 = time.monotonic()
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    elapsed = time.monotonic() - t0
    assert resp == "all done"
    assert p.max_active == loop.SUBAGENT_MAX_PARALLEL == 3       # never more than the cap...
    assert 0.55 < elapsed < 1.3                                  # ...two waves, not five sequential runs (1.5s)
    results = tool_msgs(p.lead_requests[1])
    assert [m["tool_call_id"] for m in results] == ["0", "1", "2", "3", "4"]      # order preserved
    assert all(f"report k{i + 1}" in results[i]["content"] for i in range(5))


async def test_general_subagents_run_one_at_a_time(tmp_path: Path):
    p = Router(lead=[calls(("1", "task", {"description": "a1", "prompt": "Investigate a1", "mode": "general"}),
                           ("2", "task", {"description": "a2", "prompt": "Investigate a2", "mode": "general"})),
                     text("done")],
               subs={"a1": [text("r1")], "a2": [text("r2")]}, delay=0.25)
    t0 = time.monotonic()
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build")
    assert p.max_active == 1 and time.monotonic() - t0 > 0.45   # writers are sequential


# ── general mode & safety ───────────────────────────────────────────────────

async def test_general_subagent_can_edit_and_the_leads_undo_covers_it(tmp_path: Path):
    session = ToolSession()
    p = Router(
        lead=[call("1", "write_file", path="lead.txt", content="from lead"), task("2", "alpha", "general"), text("done")],
        subs={"alpha": [call("s1", "write_file", path="sub.txt", content="from sub"), text("wrote sub.txt")]},
    )
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", session=session)
    assert (tmp_path / "lead.txt").exists() and (tmp_path / "sub.txt").read_text() == "from sub"
    assert "write_file" in names(p.sub_requests[0])
    lines = session.checkpoints.undo_last_turn()          # one /undo reverts BOTH agents' edits
    assert len(lines) == 2 and not (tmp_path / "lead.txt").exists() and not (tmp_path / "sub.txt").exists()


async def test_subagents_cannot_use_the_leads_approval_ui(tmp_path: Path):
    (tmp_path / "d").mkdir()
    asked = []
    p = Router(lead=[task("1", "alpha", "general"), text("done")],
               subs={"alpha": [call("s1", "run_command", command="rm -rf d"), text("could not")]})
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", on_approval=lambda *a: asked.append(a) or "once")
    assert (tmp_path / "d").exists() and asked == []                       # non-interactive: refused, user never prompted
    assert "needs user approval" in tool_msgs(p.sub_requests[1])[0]["content"]


async def test_shared_session_approvals_carry_into_subagents(tmp_path: Path):
    (tmp_path / "d").mkdir()
    session = ToolSession()
    session.approved_commands.add("rm -rf d")                              # the user said "allow for this session" earlier
    p = Router(lead=[task("1", "alpha", "general"), text("done")],
               subs={"alpha": [call("s1", "run_command", command="rm -rf d"), text("removed")]})
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="build", session=session)
    assert not (tmp_path / "d").exists()


async def test_read_only_lead_can_only_delegate_read_only_work(tmp_path: Path):
    p = Router(lead=[task("1", "alpha", "general"), text("plan")],
               subs={"alpha": [call("s1", "write_file", path="x.txt", content="no"), text("blocked")]})
    await run(make_agent(p), workspace=str(tmp_path), agent_mode="plan")
    assert "write_file" not in names(p.sub_requests[0]) and not (tmp_path / "x.txt").exists()
    assert '"mode": "explore"' in tool_msgs(p.lead_requests[1])[0]["content"]


async def test_subagents_cannot_spawn_subagents(tmp_path: Path):
    p = Router(lead=[task("1", "alpha"), text("done")],
               subs={"alpha": [task("s1", "beta"), text("could not recurse")]})   # a hallucinated nested call
    await run(make_agent(p), workspace=str(tmp_path))
    assert "cannot start other sub-agents" in tool_msgs(p.sub_requests[1])[0]["content"]
    assert len(p.sub_requests) == 2                                              # no third agent was ever started


# ── limits & failures ───────────────────────────────────────────────────────

async def test_runaway_subagent_hits_its_own_step_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loop, "SUBAGENT_MAX_STEPS", 3)
    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha"), text("done")],
               subs={"alpha": [call(str(i), "read_file", path="f.txt") for i in range(10)]})
    await run(make_agent(p), workspace=str(tmp_path))
    assert "safety limit (3 tool calls)" in tool_msgs(p.lead_requests[1])[0]["content"]


async def test_hung_subagent_times_out_and_the_lead_continues(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(loop, "SUBAGENT_TIMEOUT", 0.3)
    p = Router(lead=[task("1", "alpha"), text("carried on")], subs={"alpha": [text("never")]}, delay=5)
    t0 = time.monotonic()
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    assert resp == "carried on" and time.monotonic() - t0 < 3
    assert "timed out" in tool_msgs(p.lead_requests[1])[0]["content"]


async def test_subagent_failures_become_tool_errors_not_lead_crashes(tmp_path: Path):
    p = Router(lead=[task("1", "alpha"), task("2", "beta"), text("survived")],
               subs={"alpha": [RuntimeError("kaboom")], "beta": [httpx.ReadTimeout("slow")]})
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    assert resp == "survived"
    assert "kaboom" in tool_msgs(p.lead_requests[1])[0]["content"]
    assert "timed out" in tool_msgs(p.lead_requests[2])[1]["content"]         # provider timeout reported in the report


async def test_bad_task_arguments_are_reported(tmp_path: Path):
    p = Router(lead=[call("1", "task", description="x", prompt="  "), call("2", "task", description="x", prompt="p", mode="root"),
                     text("done")], subs={})
    await run(make_agent(p), workspace=str(tmp_path))
    out = [m["content"] for m in tool_msgs(p.lead_requests[2])]
    assert "non-empty 'prompt'" in out[0] and "must be 'explore' or 'general'" in out[1]


async def test_subagent_turns_are_not_stored_as_memories(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    agent = make_agent(Router(lead=[task("1", "alpha"), text("The lead summarised the investigation in detail.")],
                              subs={"alpha": [call("s1", "read_file", path="f.txt"), text("sub report " * 40)]}))
    agent.auto_remember = True
    await run(agent, "please investigate alpha thoroughly", workspace=str(tmp_path))
    await asyncio.gather(*list(agent._bg_tasks))
    assert agent.memory.count() == 1                                           # only the lead's turn


async def test_xml_text_protocol_models_can_delegate_too(tmp_path: Path):
    p = Router(
        lead=[text('<motion_tool>{"name":"task","arguments":{"description":"alpha","prompt":"Investigate alpha"}}</motion_tool>'),
              text("done")],
        subs={"alpha": [text("xml report")]}, native=False,
    )
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    assert resp == "done" and "xml report" in p.lead_requests[1]["messages"][-1]["content"]


# ── TUI ─────────────────────────────────────────────────────────────────────

async def test_tui_shows_subagent_progress_and_undo_reverts_its_edits(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("unused")]) as (app, pilot):
        router = Router(
            lead=[task("1", "alpha", "general"), text("delegated")],
            subs={"alpha": [call("s1", "write_file", path="made.txt", content="hi"), text("created made.txt")]},
        )
        app.state.agent.provider = router
        await send(app, pilot, "delegate it")
        await wait_idle(app, pilot)
        assert (tmp_path / "made.txt").exists()
        steps = "\n".join(_text_of(w) for w in app.screen.query(tui_steps())).replace("\\[", "[")  # undo markup escaping
        assert "↳ [alpha]" in steps and "sub-agent `alpha` finished" in steps
        await send(app, pilot, "/undo")
        await pilot.pause(0.1)
        assert not (tmp_path / "made.txt").exists() and any("Reverted" in t for t in system_lines(app))


def tui_steps():
    import ui.tui as tui

    return tui.StepsMessage
