"""TurnRunner's CLI-delegate short-circuit (core/agent_loop.py): a delegate provider runs its own
agent in one call instead of our per-step tool loop. Uses a fake delegate double (the real subprocess
machinery is tested separately in test_cli_delegate.py) to test the INTEGRATION: streaming, session
continuity, cost/usage reporting, the one-time caveat, error handling, and that memory-save/turn_done
still fire normally afterward."""
from pathlib import Path

import pytest

from core.providers import ModelConfig, StreamEvent
from tests.test_agent_loop import EmptyRetriever, make_agent, run


class FakeDelegate:
    """Mimics CLIDelegateProvider's public surface without touching a subprocess."""

    is_delegate = True
    native_tools = False

    def __init__(self, display_name, calls, config=None):
        self.display_name = display_name
        self.calls = calls              # list of dicts: {"text":.., "tool_lines":[..], "usage":{..}, "cost_usd":..} or an Exception
        self.session_ref = None
        self.config = config or ModelConfig(name="d", endpoint="", provider_type="cli", options={})
        self.seen_calls = []

    async def run_delegate(self, prompt, *, mode, workspace, on_event=None):
        self.seen_calls.append({"prompt": prompt, "mode": mode, "session_ref": self.session_ref})
        spec = self.calls.pop(0)
        if isinstance(spec, Exception):
            raise spec
        if on_event is not None:
            for ev in spec.get("events", [StreamEvent("text", text=spec.get("text", ""))]):
                r = on_event(ev)
                if hasattr(r, "__await__"):
                    await r
        from core.cli_delegate import DelegateResult

        return DelegateResult(
            text=spec.get("text", ""), session_ref=spec.get("session_ref"), cost_usd=spec.get("cost_usd"),
            usage=spec.get("usage"), tool_lines=spec.get("tool_lines", []),
        )


def delegate_agent(display_name, calls):
    agent = make_agent(FakeDelegate(display_name, calls))
    agent.retriever = EmptyRetriever()
    return agent


async def test_delegate_turn_streams_text_live_and_skips_the_normal_tool_loop(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "The answer is 42.", "session_ref": "s1"}])
    resp, chunks, traces = await run(agent, "what is the answer?", workspace=str(tmp_path))
    assert resp == "The answer is 42."
    assert "_delta_ The answer is 42." in chunks                         # streamed, not just returned at the end
    assert not any(s == "model_step" for s, _ in traces)                 # never entered the per-step loop
    assert not any(s == "tool_start" for s, _ in traces)


async def test_tool_progress_lines_stream_as_tool_chunks(tmp_path: Path):
    events = [StreamEvent("reasoning", text="\n🔧 Read(file_path=a.py)\n"), StreamEvent("text", text="done")]
    agent = delegate_agent("Claude Code", [{"events": events, "text": "done", "tool_lines": ["🔧 Read(file_path=a.py)"]}])
    resp, chunks, traces = await run(agent, workspace=str(tmp_path))
    assert any(c == "_tool_ 🔧 Read(file_path=a.py)" for c in chunks)
    assert resp == "done"


async def test_the_undo_and_sandbox_caveat_shows_once_per_agent_not_every_turn(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "one"}, {"text": "two"}])
    _, chunks1, _ = await run(agent, "first", workspace=str(tmp_path))
    _, chunks2, _ = await run(agent, "second", workspace=str(tmp_path))
    assert any("/undo does not cover its edits" in c for c in chunks1)
    assert not any("/undo does not cover its edits" in c for c in chunks2)


async def test_session_continuity_across_turns_through_the_agent(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "first", "session_ref": "sess-1"}, {"text": "second"}])
    provider = agent.provider
    await run(agent, "one", workspace=str(tmp_path))
    assert provider.session_ref == "sess-1"
    await run(agent, "two", workspace=str(tmp_path))
    assert provider.seen_calls[1]["session_ref"] == "sess-1"             # the second call resumed the first


async def test_cost_is_reported_when_known_and_labelled_as_subscription_otherwise(tmp_path: Path):
    with_cost = delegate_agent("Claude Code", [{"text": "x", "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "cost_usd": 0.0142}])
    _, chunks, traces = await run(with_cost, workspace=str(tmp_path))
    assert any("reported cost $0.0142" in c for c in chunks)
    assert any(s == "usage" for s, _ in traces)

    no_cost = delegate_agent("Codex", [{"text": "x", "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "cost_usd": None}])
    _, chunks2, _ = await run(no_cost, workspace=str(tmp_path))
    assert any("uses your subscription, not a separate API charge" in c for c in chunks2)

    no_usage = delegate_agent("Codex", [{"text": "x"}])
    _, chunks3, _ = await run(no_usage, workspace=str(tmp_path))
    assert not any("via Codex" in c for c in chunks3)                    # nothing invented when there's no usage at all


async def test_plan_vs_build_mode_is_forwarded_to_the_delegate(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "x"}, {"text": "y"}])
    await run(agent, workspace=str(tmp_path), agent_mode="plan")
    assert agent.provider.seen_calls[0]["mode"] == "plan"
    await run(agent, workspace=str(tmp_path), agent_mode="build")
    assert agent.provider.seen_calls[1]["mode"] == "build"


async def test_delegate_failure_is_reported_without_crashing_the_turn(tmp_path: Path):
    from core.cli_delegate import CLIDelegateError

    agent = delegate_agent("Claude Code", [CLIDelegateError("Claude Code exited 1: not logged in")])
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert "⚠️" in resp and "not logged in" in resp
    assert any(s == "provider_error" for s, _ in traces)
    assert not any(s == "turn_done" for s, _ in traces)                  # matches how a normal provider failure behaves


async def test_an_unexpected_exception_from_the_delegate_is_also_reported_not_raised(tmp_path: Path):
    agent = delegate_agent("Codex", [RuntimeError("boom")])
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert "⚠️" in resp and "boom" in resp
    assert any(s == "provider_error" for s, _ in traces)


async def test_empty_final_text_still_produces_a_readable_response(tmp_path: Path):
    agent = delegate_agent("Codex", [{"text": "", "tool_lines": ["🔧 command_execution(ls)"]}])
    resp, _, _ = await run(agent, workspace=str(tmp_path))
    assert "Codex" in resp and "no final text" in resp


async def test_memory_save_and_turn_done_still_fire_normally_after_a_delegate_turn(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "a long enough response to qualify for memory " * 5}])
    agent.auto_remember = True
    resp, _, traces = await run(agent, workspace=str(tmp_path))
    assert any(s == "turn_done" for s, _ in traces)
    import asyncio

    await asyncio.gather(*list(agent._bg_tasks))
    assert agent.memory.count() == 1


async def test_tool_lines_mark_the_turn_as_having_used_a_tool(tmp_path: Path):
    agent = delegate_agent("Claude Code", [{"text": "wrote it", "tool_lines": ["🔧 Write(path=a.py)"]}])
    from core.agent_loop import TurnRunner

    runner = TurnRunner(agent, "make a.py", workspace=str(tmp_path), agent_mode="build")
    resp = await runner.run()
    assert resp == "wrote it" and runner.used_tool is True and runner.tool_operations == ["🔧 Write(path=a.py)"]
