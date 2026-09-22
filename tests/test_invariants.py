"""Conversation invariants the model APIs enforce (and that break a whole session when violated):

  every assistant tool call gets exactly one tool result, before the next assistant turn.

A single dangling call makes every LATER request fail, so this is checked on every request sent in a wide
range of scenarios (failures, blocks, denials, parallel calls, sub-agents, budget cut-offs, failover).
"""
from pathlib import Path

import httpx
import pytest

from core.budget import Budget
from core.hooks import Hooks
from core.providers import ProviderError
from tests.test_agent_loop import Scripted, call, calls, make_agent, run, text
from tests.test_failover import arm


def assert_paired(messages, where=""):
    pending: dict = {}
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            assert m["tool_call_id"] in pending, f"{where}: tool result {m['tool_call_id']} at #{i} has no matching call"
            del pending[m["tool_call_id"]]
        else:
            assert not pending, f"{where}: {sorted(pending)} had no result before message #{i} ({m['role']})"
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    assert tc["id"] not in pending
                    pending[tc["id"]] = tc
    # the request that ends the list may legitimately end on the assistant's calls only if the tool results are next
    return pending


def check_all(provider, label):
    for n, request in enumerate(provider.requests):
        pending = assert_paired(request["messages"], f"{label} request {n}")
        assert not pending, f"{label}: request {n} ends with unanswered calls {sorted(pending)}"


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("alpha\n")
    return tmp_path


SCENARIOS = {
    "parallel reads": lambda ws: ([calls(("1", "read_file", {"path": "a.txt"}), ("2", "list_files", {"path": "."}), ("3", "grep", {"pattern": "alpha"})), text("done")], {}),
    "a tool that fails": lambda ws: ([call("1", "read_file", path="missing.txt"), text("done")], {}),
    "unknown tool name": lambda ws: ([call("1", "no_such_tool", x=1), text("done")], {}),
    "bad arguments": lambda ws: ([call("1", "read_file", path=123), text("done")], {}),
    "write blocked in plan mode": lambda ws: ([call("1", "write_file", path="x.txt", content="no"), text("plan")], {"agent_mode": "plan"}),
    "out-of-workspace denied": lambda ws: ([call("1", "read_file", path="/etc/hosts"), text("done")], {}),
    "risky command refused": lambda ws: ([call("1", "run_command", command="rm -rf a.txt"), text("done")], {}),
    "mixed good and bad in one step": lambda ws: ([calls(("1", "read_file", {"path": "a.txt"}), ("2", "read_file", {"path": "nope"}), ("3", "run_command", {"command": "rm -rf ."})), text("done")], {}),
    "many steps then finish": lambda ws: ([call(str(i), "read_file", path="a.txt") for i in range(6)] + [text("done")], {}),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_every_tool_call_is_answered_in_every_request(ws, name):
    steps, kw = SCENARIOS[name](ws)
    p = Scripted(steps)
    await run(make_agent(p), workspace=str(ws), agent_mode=kw.get("agent_mode", "build"))
    assert p.requests, "the scenario sent no requests"
    check_all(p, name)


async def test_hook_blocked_calls_are_still_answered(ws):
    p = Scripted([call("1", "write_file", path="x.txt", content="no"), call("2", "read_file", path="a.txt"), text("done")])
    agent = make_agent(p)
    agent.hooks = Hooks.from_config({"hooks": {"pre_tool": [{"match": "write_file", "command": "exit 1"}]}}.get)
    await run(agent, workspace=str(ws), agent_mode="build")
    check_all(p, "hook blocked")


async def test_a_budget_cut_off_leaves_no_dangling_calls_even_when_the_model_ignores_the_wrap_up(ws):
    p = Scripted([call(str(i), "read_file", path="a.txt") for i in range(6)])
    agent = make_agent(p)
    agent.budget = Budget(max_steps=2)
    await run(agent, workspace=str(ws))
    check_all(p, "budget")


async def test_subagent_conversations_are_well_formed_too(ws):
    from tests.test_subagents import Router, task

    p = Router(lead=[calls(("1", "task", {"description": "a", "prompt": "Investigate a"}), ("2", "read_file", {"path": "a.txt"})), text("done")],
               subs={"a": [call("s1", "read_file", path="a.txt"), call("s2", "read_file", path="nope"), text("report")]})
    await run(make_agent(p), workspace=str(ws))
    for n, r in enumerate(p.lead_requests):
        assert not assert_paired(r["messages"], f"lead {n}")
    for n, r in enumerate(p.sub_requests):
        assert not assert_paired(r["messages"], f"sub {n}")


async def test_a_failover_in_the_middle_of_a_tool_loop_hands_over_a_well_formed_conversation(ws):
    primary = Scripted([calls(("1", "read_file", {"path": "a.txt"}), ("2", "list_files", {"path": "."})), ProviderError("HTTP 503", status_code=503)])
    backup = Scripted([text("finished")])
    await run(arm(make_agent(primary), backup), workspace=str(ws))
    check_all(primary, "primary")
    check_all(backup, "backup")


async def test_a_cancelled_turn_does_not_corrupt_the_next_one(ws):
    """Cancel while a tool is running (the Esc key), then run another turn on the same agent."""
    import asyncio

    (ws / "slow.sh").write_text("sleep 5\n")
    agent = make_agent(Scripted([call("1", "run_command", command="sleep 5"), text("never"), text("second turn ok")]))
    task = asyncio.ensure_future(run(agent, "first", workspace=str(ws), agent_mode="build"))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    agent.provider.steps = [text("second turn ok")]                             # what the model says next
    resp, _, _ = await run(agent, "second", workspace=str(ws), agent_mode="build")
    assert resp == "second turn ok"
    for n, r in enumerate(agent.provider.requests):
        assert not assert_paired(r["messages"], f"after cancel, request {n}")
