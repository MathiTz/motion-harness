"""Trajectory of a turn (per-step time / tokens / tools), /trajectory, trace copy and /tracking."""
import io
import json
from pathlib import Path

import pytest

import core.trajectory as traj
from core.headless import run_headless
from core.providers import StreamEvent
from tests.test_agent_loop import Scripted, call, calls, make_agent, run, text
from tests.test_tui_flow import _text_of, chat_texts, send, system_lines, tui_app, wait_idle


def usage(prompt, completion):
    return [StreamEvent("usage", usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})]


def records_of(traces):
    return [p["record"] for stage, p in traces if stage == "step_record"]


# ── recording ───────────────────────────────────────────────────────────────

async def test_every_model_step_is_recorded_with_tokens_time_and_tool_results(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello\n" * 100)
    steps = [
        call("1", "read_file", path="a.txt", offset=1, limit=50) + usage(1000, 40),
        calls(("2", "grep", {"pattern": "hello"}), ("3", "list_files", {"path": "."})) + usage(1800, 60),
        text("All done.") + usage(2400, 5),
    ]
    _, _, traces = await run(make_agent(Scripted(steps)), workspace=str(tmp_path))
    recs = records_of(traces)
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert [(r["prompt_tokens"], r["completion_tokens"]) for r in recs] == [(1000, 40), (1800, 60), (2400, 5)]
    assert [t["name"] for t in recs[1]["tools"]] == ["grep", "list_files"]
    first = recs[0]["tools"][0]
    assert first["name"] == "read_file" and "path=a.txt" in first["args"] and first["ok"] and first["result_chars"] > 100
    assert recs[2]["tools"] == [] and recs[2]["text_chars"] == len("All done.")
    assert all(r["agent"] == "lead" and r["duration_s"] >= 0 and r["context_tokens_est"] > 0 for r in recs)
    assert recs[0]["context_tokens_est"] < recs[2]["context_tokens_est"]           # context grows as results accumulate


async def test_failed_tool_calls_are_marked_and_large_arguments_are_elided(tmp_path: Path):
    steps = [call("1", "read_file", path="missing.txt"), call("2", "write_file", path="o.txt", content="x" * 5000), text("done")]
    _, _, traces = await run(make_agent(Scripted(steps)), workspace=str(tmp_path), agent_mode="build")
    recs = records_of(traces)
    assert recs[0]["tools"][0]["ok"] is False
    args = recs[1]["tools"][0]["args"]
    assert "5000 chars" in args and "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in args and len(args) <= 100


async def test_missing_provider_usage_is_reported_as_unknown_not_zero(tmp_path: Path):
    _, _, traces = await run(make_agent(Scripted([text("hi")])), workspace=str(tmp_path))
    rec = records_of(traces)[0]
    assert rec["prompt_tokens"] is None and rec["completion_tokens"] is None
    assert "token usage not reported" in traj.render([rec])


async def test_turn_done_carries_the_full_transcript_for_export(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    _, _, traces = await run(make_agent(Scripted([call("1", "read_file", path="a.txt"), text("ok")])), "the question", workspace=str(tmp_path))
    done = next(p for s, p in traces if s == "turn_done")
    assert "You are Motion Agent" in done["system_prompt"]
    roles = [m["role"] for m in done["transcript"]]
    assert roles[0] == "user" and "tool" in roles and roles[-1] == "assistant"
    assert done["transcript"][-1]["content"] == "ok" and "the question" in str(done["transcript"][0]["content"])


# ── rendering & analysis ────────────────────────────────────────────────────

def rec(step, prompt, tools=(), duration=5.0, completion=100, agent="lead", reasoning=0):
    return {"turn": 1, "agent": agent, "step": step, "duration_s": duration, "ttft_s": 1.0, "prompt_tokens": prompt,
            "completion_tokens": completion, "context_tokens_est": prompt, "text_chars": 0, "reasoning_chars": reasoning,
            "tools": [{"name": n, "args": a, "ok": True, "result_chars": c, "duration_s": 0.01} for n, a, c in tools]}


def test_totals_and_table():
    recs = [rec(1, 3000, [("read_file", "path=ui/tui.py", 12000)]), rec(2, 9000, [("grep", "pattern=x", 800)]), rec(3, 10000)]
    t = traj.totals(recs)
    assert t["steps"] == 3 and t["tool_calls"] == 2 and t["prompt_tokens"] == 22000 and t["completion_tokens"] == 300
    table = traj.render(recs, "Trajectory of turn 1")
    assert "3 steps · 2 tool calls" in table and "22,000 prompt + 300 completion tokens" in table
    assert "read_file(path=ui/tui.py) → 12.0k" in table and "9.0k" in table


def test_insights_explain_where_the_tokens_went():
    recs = [rec(i, 3000 + i * 1800, [("read_file", f"path=f{i % 3}.py", 9000)], reasoning=8000) for i in range(1, 11)]
    found = "\n".join(traj.insights(recs))
    assert "prompt tokens grew" in found and "re-sent on every step" in found
    assert "largest tool results" in found and "read_file" in found
    assert "of the time was the model generating" in found and "/effort low" in found
    assert "10 steps" in found and "several independent tool calls in one step" in found
    assert "repeated call: read_file" in found


def test_subagent_steps_are_labelled_and_a_short_run_stays_quiet():
    assert "sub:explore" in traj.render([rec(1, 500, agent="sub:explore")])
    assert traj.insights([rec(1, 500), rec(2, 600)]) == []


def test_turn_selection_and_json_document():
    recs = [{**rec(1, 100), "turn": 1}, {**rec(1, 200), "turn": 2}, {**rec(2, 300), "turn": 2}]
    assert [r["prompt_tokens"] for r in traj.turn_records(recs)] == [200, 300]
    assert len(traj.turn_records(recs, 1)) == 1 and traj.turn_records([]) == []
    doc = traj.to_json(recs, turn=2, provider="p/m", system_prompt="S", messages=[{"role": "user", "content": "hi"}])
    assert doc["turn"] == 2 and doc["summary"]["steps"] == 2 and doc["provider"] == "p/m" and doc["messages"][0]["content"] == "hi"
    json.dumps(doc)                                                          # serializable as-is


# ── headless ────────────────────────────────────────────────────────────────

async def test_headless_json_and_stream_include_the_trajectory(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    def make():
        return make_agent(Scripted([call("1", "read_file", path="f.txt") + usage(500, 20), text("done") + usage(700, 5)]))
    out = io.StringIO()
    await run_headless("go", out=out, err=io.StringIO(), agent_factory=make, workspace=str(tmp_path), output_format="stream-json")
    events = [json.loads(l) for l in out.getvalue().splitlines()]
    steps = [e for e in events if e["type"] == "step"]
    assert [s["step"] for s in steps] == [1, 2] and steps[0]["tools"][0]["name"] == "read_file"
    result = events[-1]
    assert result["type"] == "result" and [s["step"] for s in result["trajectory"]] == [1, 2]
    assert sum(s["prompt_tokens"] for s in result["trajectory"]) == result["usage"]["prompt_tokens"] == 1200


# ── TUI ─────────────────────────────────────────────────────────────────────

def trajectory_widgets(app):
    return [w for w in app.screen.query("#chat_log .trajectory")]


async def two_step_turn(app, pilot, tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    await send(app, pilot, "look at a.txt")
    await wait_idle(app, pilot)


async def test_trajectory_command_shows_the_last_turn(tmp_path, monkeypatch):
    steps = [call("1", "read_file", path="a.txt") + usage(1000, 30), text("It says hello.") + usage(1400, 8),
             text("Second answer") + usage(1600, 4)]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await two_step_turn(app, pilot, tmp_path)
        assert len(app.state.trajectory) == 2 and {r["turn"] for r in app.state.trajectory} == {1}
        await send(app, pilot, "/trajectory")
        await pilot.pause(0.1)
        shown = str(trajectory_widgets(app)[0].renderable)
        assert "Trajectory of turn 1 · 2 steps · 1 tool calls" in shown and "read_file(path=a.txt)" in shown
        assert "2,400 prompt + 38 completion tokens" in shown
        assert any("/trajectory copy" in t for t in system_lines(app))
        await send(app, pilot, "second question")                                  # a new turn replaces "the last turn"
        await wait_idle(app, pilot)
        await send(app, pilot, "/trajectory")
        await pilot.pause(0.1)
        assert "turn 2 · 1 steps" in str(trajectory_widgets(app)[-1].renderable)
        await send(app, pilot, "/trajectory all")
        await pilot.pause(0.1)
        assert "whole session, 2 turns" in str(trajectory_widgets(app)[-1].renderable) and "3 steps" in str(trajectory_widgets(app)[-1].renderable)


async def test_trajectory_copy_and_the_empty_case(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [call("1", "read_file", path="a.txt"), text("ok")]) as (app, pilot):
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", lambda t: copied.append(t))
        await send(app, pilot, "/trajectory")
        await pilot.pause(0.1)
        assert "No steps recorded yet" in str(trajectory_widgets(app)[0].renderable)
        await send(app, pilot, "/trajectory copy")
        await pilot.pause(0.1)
        assert copied == []                                                       # nothing to copy yet, and it said so
        await two_step_turn(app, pilot, tmp_path)
        await send(app, pilot, "/trajectory copy")
        await pilot.pause(0.1)
        assert len(copied) == 1 and "Trajectory of turn 1" in copied[0] and "read_file(path=a.txt)" in copied[0]
        await send(app, pilot, "/trajectory bogus")
        await pilot.pause(0.1)
        assert any("Usage: /trajectory" in t for t in system_lines(app))


async def test_trajectory_save_writes_json_and_full_adds_every_message(tmp_path, monkeypatch):
    steps = [call("1", "read_file", path="a.txt") + usage(1000, 30), text("It says hello.") + usage(1400, 8)]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await two_step_turn(app, pilot, tmp_path)
        await send(app, pilot, "/trajectory save")
        await pilot.pause(0.1)
        files = sorted((tmp_path / ".motion" / "trajectories").glob("turn-1-*.json"))
        assert len(files) == 1
        doc = json.loads(files[0].read_text())
        assert doc["summary"]["steps"] == 2 and doc["steps"][0]["tools"][0]["name"] == "read_file" and "messages" not in doc
        assert any("Saved" in t and "add 'full'" in t for t in system_lines(app))
        await send(app, pilot, "/trajectory save full")
        await pilot.pause(0.1)
        full = json.loads(max((tmp_path / ".motion" / "trajectories").glob("turn-1-*.json"), key=lambda p: p.stat().st_mtime_ns).read_text())
        assert "You are Motion Agent" in full["system_prompt"]
        assert any(m["role"] == "user" and "look at a.txt" in str(m["content"]) for m in full["messages"])


async def test_new_session_clears_the_trajectory(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        await send(app, pilot, "hello")
        await wait_idle(app, pilot)
        assert app.state.trajectory
        app.state.new_session()
        assert app.state.trajectory == [] and app.state.trajectory_turn == 0 and app.state.last_transcript is None


async def test_the_trace_log_can_be_copied_as_plain_text(tmp_path, monkeypatch):
    """The panel can't be selected with the mouse, so F10 / the palette copies the whole log."""
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", lambda t: copied.append(t))
        chat = app.screen.query_one(tui.ChatPane)
        chat._trace_lines.clear()                                                 # (the log starts with a session.start line)
        chat.action_copy_trace()
        assert copied == []                                                       # empty log: nothing copied
        await send(app, pilot, "hello")
        await wait_idle(app, pilot)
        await pilot.press("f10")
        await pilot.pause(0.1)
        assert len(copied) == 1
        assert "turn.done" in copied[0] and "model.step" in copied[0] and "[dim]" not in copied[0]   # markup stripped
        assert copied[0].count("\n") >= 3


async def test_palette_offers_copy_trace_trajectory_and_tracking(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        await pilot.press("ctrl+k")
        await pilot.pause(0.1)
        labels = [c[0] for c in app.screen._commands]
        assert {"Copy trace log", "Copy turn trajectory", "Toggle interaction tracking"} <= set(labels)
        assert any("Show turn trajectory" in l for l in labels)


# ── tracking: undoing "No thanks" ───────────────────────────────────────────

async def test_tracking_can_be_turned_on_after_declining_and_then_records_turns(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("first"), text("second")], track=False) as (app, pilot):   # declined
        cm = app.state.config_manager
        assert not cm.get("track_interactions")
        await send(app, pilot, "one")
        await wait_idle(app, pilot)
        assert not (tmp_path / ".motion" / "sessions").exists()                   # nothing saved while off
        await send(app, pilot, "/tracking")
        await pilot.pause(0.1)
        assert any("OFF — nothing is saved" in t and "/tracking on" in t for t in system_lines(app))
        await send(app, pilot, "/tracking on")
        await pilot.pause(0.1)
        assert cm.get("track_interactions") is True and any("ON — every turn is saved" in t for t in system_lines(app))
        await send(app, pilot, "two")
        await wait_idle(app, pilot)
        saved = list((tmp_path / ".motion" / "sessions").glob("*.jsonl"))
        assert len(saved) == 1 and "second" in saved[0].read_text() and "first" not in saved[0].read_text()
        await send(app, pilot, "/tracking off")
        await pilot.pause(0.1)
        assert cm.get("track_interactions") is False
        await send(app, pilot, "/tracking maybe")
        await pilot.pause(0.1)
        assert any("Usage: /tracking" in t for t in system_lines(app))


async def test_palette_entry_toggles_tracking(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")], track=False) as (app, pilot):
        chat = app.screen.query_one(tui.ChatPane)
        chat.set_tracking("toggle")
        assert app.state.config_manager.get("track_interactions") is True
        chat.set_tracking("toggle")
        assert app.state.config_manager.get("track_interactions") is False


async def test_declining_the_consent_prompt_says_how_to_change_it_later(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")], track=False) as (app, pilot):
        screen = tui.TrackingConsentScreen()
        await app.push_screen(screen)
        await pilot.pause(0.1)
        assert "/tracking on|off" in str(screen.query_one("#tracking_body").renderable)
        notes = []
        monkeypatch.setattr(screen, "notify", lambda msg, **k: notes.append(msg))
        screen.action_decline()
        assert app.state.config_manager.get("track_interactions") is False
        assert notes and "/tracking on" in notes[0]


import ui.tui as tui  # noqa: E402  (kept at the bottom so the helper imports above stay grouped)


async def test_subagent_steps_appear_in_the_leads_trajectory_labelled(tmp_path: Path):
    from tests.test_subagents import Router, task

    (tmp_path / "f.txt").write_text("x")
    p = Router(lead=[task("1", "alpha") + usage(900, 20), text("done") + usage(1200, 5)],
               subs={"alpha": [call("s1", "read_file", path="f.txt") + usage(300, 10), text("report") + usage(400, 6)]})
    _, _, traces = await run(make_agent(p), workspace=str(tmp_path))
    import asyncio
    await asyncio.sleep(0.05)
    recs = records_of(traces)
    assert [r["agent"] for r in recs].count("lead") == 2
    subs = [r for r in recs if r["agent"] == "sub:alpha"]
    assert len(subs) == 2 and subs[0]["tools"][0]["name"] == "read_file" and subs[1]["prompt_tokens"] == 400
    assert "sub:alpha" in traj.render(recs)
