"""End-to-end TUI tests: a headless Textual app driven by a scripted provider."""
import asyncio
import contextlib
import os
import time
from pathlib import Path

import pytest

import ui.tui as tui
from core.config import ConfigManager
from core.providers import ModelConfig, StreamEvent
from tests.test_agent_loop import Scripted, call, calls, text


class _NoRecall:
    async def retrieve(self, q, top_k=5):
        return []


def reasoning(*chunks):
    return [StreamEvent("reasoning", text=c) for c in chunks]


@contextlib.asynccontextmanager
async def tui_app(tmp_path: Path, monkeypatch, steps, *, mode="build", track=False, vision=False, extra_cfg=""):
    cfg = tmp_path / "config.yml"
    cfg.write_text(f"track_interactions: {'true' if track else 'false'}\n{extra_cfg}")
    monkeypatch.setattr(ConfigManager, "CONFIG_PATHS", [str(cfg)])
    monkeypatch.setattr(tui, "WORKSPACE", str(tmp_path))
    monkeypatch.setattr(tui, "_suppress_logging", lambda: None)
    app = tui.MotionTUI(
        model_config=ModelConfig(name="x", endpoint="http://x", provider_type="local", options={"model": "m"}),
        provider_id="x/m",
        workspace=str(tmp_path),  # the default is evaluated at import time
    )
    async with app.run_test(size=(150, 50)) as pilot:
        await pilot.pause()
        app.initial_mode = app.state.agent_mode
        provider = Scripted(steps, vision=vision)
        app.state.agent.provider = provider
        app.state.agent.retriever = _NoRecall()
        app.state.agent.auto_remember = False
        app.state.agent_mode = mode
        app.provider = provider  # convenience for assertions
        yield app, pilot


def _text_of(widget) -> str:
    r = getattr(widget, "renderable", None) or getattr(widget, "content", None)
    return getattr(r, "markup", None) or str(r)


def chat_texts(app):
    log = app.screen.query_one("#chat_log")
    return [(type(w).__name__, _text_of(w)) for w in log.children]


async def send(app, pilot, message: str):
    box = app.screen.query_one("#chat_input")
    box.value = message
    box.cursor_position = len(message)
    await pilot.press("enter")


async def wait_idle(app, pilot, timeout=8.0):
    deadline = time.monotonic() + timeout
    await pilot.pause(0.05)
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if not app.state.busy and type(app.screen).__name__ == "MainScreen":
            return
    raise AssertionError("app never went idle: " + type(app.screen).__name__)


async def wait_for_screen(app, pilot, cls, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if isinstance(app.screen, cls):
            return app.screen
    raise AssertionError(f"{cls.__name__} never appeared; on {type(app.screen).__name__}")


def system_lines(app):
    return [t for kind, t in chat_texts(app) if kind == "SystemMessage"]


# ── the basic turn ──────────────────────────────────────────────────────────

async def test_streamed_turn_renders_answer_and_reports_timing(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("Hello ", "there!")]) as (app, pilot):
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        kinds = chat_texts(app)
        assert ("AgentMessage", "Hello there!") in kinds
        assert app.state.conversation_turns == [("hi", "Hello there!")]
        m = app.state.last_turn_metrics
        assert m["elapsed_s"] >= 0 and m["ttft_s"] is not None
        status = _text_of(app.screen.query_one("#chat_status_text"))
        assert "last turn" in status and "first token" in status


async def test_trace_is_buffered_and_stream_chunks_are_not_traced(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("a", "b", "c", "d", "e")]) as (app, pilot):
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        pane = app.screen.query_one(tui.ChatPane)
        joined = "\n".join(pane._trace_lines)
        assert "model.step" in joined and "turn.done" in joined and "stream.chunk" not in joined
        # Panel hidden => no per-event widgets were mounted...
        assert len(app.screen.query_one("#trace_log").children) == 0
        # ...and opening it renders the buffered history.
        pane.action_toggle_trace_panel()
        await pilot.pause()
        assert len(app.screen.query_one("#trace_log").children) == len(pane._trace_lines)


async def test_trace_buffer_is_bounded(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        pane = app.screen.query_one(tui.ChatPane)
        for i in range(1000):
            pane._append_trace("tool_progress", f"line {i}")
        assert len(pane._trace_lines) == pane.TRACE_BUFFER_MAX and pane._trace_count >= 1000


async def test_reasoning_streams_then_collapses_to_thought_time(tmp_path, monkeypatch):
    steps = [reasoning("let me ", "think") + text("42")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "meaning of life?")
        await wait_idle(app, pilot)
        rendered = [t for kind, t in chat_texts(app) if kind == "ReasoningMessage"]
        assert rendered and "thought for" in rendered[0]
        assert ("AgentMessage", "42") in chat_texts(app)


async def test_inline_think_tags_are_split_from_the_answer(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("<think>hmm</think>", "The answer")]) as (app, pilot):
        await send(app, pilot, "q")
        await wait_idle(app, pilot)
        assert ("AgentMessage", "The answer") in chat_texts(app)
        assert app.state.last_agent_response == "The answer"


def test_extract_reasoning_handles_unclosed_think_while_streaming():
    assert tui._extract_reasoning_and_answer("hi <think>still going", streaming=True) == ("still going", "hi")
    assert tui._extract_reasoning_and_answer("<think>a</think>b") == ("a", "b")
    assert tui._extract_reasoning_and_answer("no tags") == ("", "no tags")


async def test_live_status_shows_phase_elapsed_and_command_output(tmp_path, monkeypatch):
    steps = [call("1", "run_command", command="echo working; sleep 1.2"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "run it")
        seen = set()
        for _ in range(40):
            await pilot.pause(0.05)
            if app.state.busy:
                seen.add(_text_of(app.screen.query_one("#chat_status_text")))
            if not app.state.busy and app.state.conversation_turns:
                break
        blob = "\n".join(seen)
        assert "running a command" in blob and "Esc cancels" in blob
        assert "run_command" not in blob  # human wording, not the tool's internal name
        assert "working" in blob  # latest line of the command's output
        await wait_idle(app, pilot)


async def test_ui_stays_responsive_while_a_command_runs(tmp_path, monkeypatch):
    steps = [call("1", "run_command", command="sleep 1.5"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "slow")
        ticks = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            await pilot.pause(0.05)
            ticks.append(time.monotonic())
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        assert len(ticks) >= 12 and max(gaps) < 0.4  # the event loop kept turning
        await wait_idle(app, pilot, 10)


async def test_escape_cancels_and_kills_the_running_command(tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    steps = [call("1", "run_command", command=f"echo $$ > {pidfile}; sleep 30"), text("never")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "long")
        for _ in range(60):
            await pilot.pause(0.05)
            if pidfile.exists() and pidfile.read_text().strip():
                break
        pid = int(pidfile.read_text())
        await pilot.press("escape")
        await wait_idle(app, pilot)
        await asyncio.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert any("Cancelled" in t for t in system_lines(app))


# ── approvals & questions ───────────────────────────────────────────────────

async def test_risky_command_shows_approval_modal_and_allow_once_runs_it(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    steps = [call("1", "run_command", command="rm -rf d"), text("removed")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "clean up")
        screen = await wait_for_screen(app, pilot, tui.PermissionScreen)
        assert "rm -rf d" in _text_of(screen.query_one("#permission_path"))
        assert "risky command" in _text_of(screen.query_one("#permission_title"))
        await pilot.press("enter")  # "Allow once"
        await wait_idle(app, pilot)
        assert not (tmp_path / "d").exists()


async def test_denying_the_approval_leaves_files_alone(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    steps = [call("1", "run_command", command="rm -rf d"), text("ok, skipped")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "clean up")
        await wait_for_screen(app, pilot, tui.PermissionScreen)
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert (tmp_path / "d").exists()
        assert "user denied" in str(app.provider.requests[1]["messages"][-1]["content"])


async def test_ask_user_modal_returns_the_typed_answer(tmp_path, monkeypatch):
    steps = [call("1", "ask_user", question="Project name?"), text("ok")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "start a project")
        screen = await wait_for_screen(app, pilot, tui.AskUserScreen)
        screen.query_one("#ask_input").value = "apollo"
        await pilot.press("enter")
        await wait_idle(app, pilot)
        assert '"answer": "apollo"' in str(app.provider.requests[1]["messages"][-1]["content"])


async def test_ask_user_options_can_be_picked_from_a_list(tmp_path, monkeypatch):
    steps = [call("1", "ask_user", question="DB?", options=["sqlite", "postgres"]), text("ok")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "pick")
        await wait_for_screen(app, pilot, tui.AskUserScreen)
        await pilot.press("down", "enter")
        await wait_idle(app, pilot)
        assert '"answer": "postgres"' in str(app.provider.requests[1]["messages"][-1]["content"])


async def test_todo_list_appears_in_the_context_panel_and_via_command(tmp_path, monkeypatch):
    steps = [call("1", "todo_write", todos=[{"content": "write code", "status": "in_progress"}]), text("ok")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "plan it")
        await wait_idle(app, pilot)
        assert app.state.todos == [{"content": "write code", "status": "in_progress"}]
        await send(app, pilot, "/todos")
        await pilot.pause(0.1)
        assert any("▶ write code" in t for t in system_lines(app))


# ── conversation commands ───────────────────────────────────────────────────

async def test_undo_reverts_the_last_turns_file_changes(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="new.txt", content="hello"), text("wrote it")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "make a file")
        await wait_idle(app, pilot)
        assert (tmp_path / "new.txt").exists()
        await send(app, pilot, "/undo")
        await pilot.pause(0.1)
        assert not (tmp_path / "new.txt").exists()
        assert any("Reverted" in t for t in system_lines(app))
        await send(app, pilot, "/undo")
        await pilot.pause(0.1)
        assert any("Nothing to undo" in t for t in system_lines(app))


async def test_new_clears_the_conversation(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("one")]) as (app, pilot):
        await send(app, pilot, "first")
        await wait_idle(app, pilot)
        assert app.state.conversation_turns
        await send(app, pilot, "/new")
        await pilot.pause(0.1)
        assert app.state.conversation_turns == [] and app.state.session_context == ""
        kinds = [k for k, _ in chat_texts(app)]
        assert "UserMessage" not in kinds and "AgentMessage" not in kinds


async def test_compact_replaces_history_with_a_model_summary(tmp_path, monkeypatch):
    steps = [text("answer one"), text("answer two"), text("- user asked two things; answered both")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        for q in ("first question", "second question"):
            await send(app, pilot, q)
            await wait_idle(app, pilot)
        await send(app, pilot, "/compact")
        for _ in range(60):
            await pilot.pause(0.05)
            if any("Compacted" in t for t in system_lines(app)):
                break
        assert app.state.conversation_turns == [
            ("[Summary of the conversation so far]", "- user asked two things; answered both")
        ]


async def test_auto_compact_triggers_when_history_nears_the_window(tmp_path, monkeypatch):
    steps = [text("- summary of everything"), text("fresh answer")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        app.provider.config.options["context_window"] = 1000
        app.state.conversation_turns = [("q" * 2000, "a" * 2000)]  # ~1000 tokens > 60% of 1000
        await send(app, pilot, "next")
        await wait_idle(app, pilot)
        assert any("Context is filling up" in t for t in system_lines(app))
        assert app.state.conversation_turns[0][0] == "[Summary of the conversation so far]"
        assert app.state.conversation_turns[-1] == ("next", "fresh answer")


async def test_sessions_are_saved_and_can_be_resumed(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("A1"), text("A2")], track=True) as (app, pilot):
        await send(app, pilot, "Q1")
        await wait_idle(app, pilot)
        await send(app, pilot, "Q2")
        await wait_idle(app, pilot)
        sessions = list((tmp_path / ".motion" / "sessions").glob("*.jsonl"))
        assert len(sessions) == 1
        sid = sessions[0].stem
        await send(app, pilot, "/new")
        await pilot.pause(0.1)
        await send(app, pilot, "/resume")
        await pilot.pause(0.1)
        assert any(sid in t and "2 turn(s)" in t for t in system_lines(app))
        await send(app, pilot, f"/resume {sid}")
        await pilot.pause(0.2)
        assert app.state.conversation_turns == [("Q1", "A1"), ("Q2", "A2")]
        assert [t for k, t in chat_texts(app) if k == "AgentMessage"] == ["A1", "A2"]


async def test_resume_explains_when_history_is_off(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], track=False) as (app, pilot):
        await send(app, pilot, "/resume")
        await pilot.pause(0.1)
        assert any("history is off" in t for t in system_lines(app))


async def test_resuming_a_session_does_not_re_run_a_side_effecting_command(tmp_path, monkeypatch):
    """/resume only replays the saved prompt/response text into conversation history for a NEW
    turn - it never re-invokes a tool call from a past turn. Proven here, not assumed: a command
    that leaves a countable side effect is run once, the session is resumed, and the side effect
    must still show exactly one occurrence."""
    counter = tmp_path / "count.txt"
    steps = [call("1", "run_command", command=f"echo x >> {counter}"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps, track=True) as (app, pilot):
        await send(app, pilot, "run it")
        await wait_idle(app, pilot)
        assert counter.read_text().count("x") == 1
        sid = list((tmp_path / ".motion" / "sessions").glob("*.jsonl"))[0].stem
        await send(app, pilot, "/new")
        await pilot.pause(0.1)
        await send(app, pilot, f"/resume {sid}")
        await pilot.pause(0.2)
        assert app.state.conversation_turns == [("run it", "done")]
        assert counter.read_text().count("x") == 1  # unchanged - resume did not re-run the command


async def test_cancelling_a_turn_still_leaves_a_visible_record_not_a_silent_gap(tmp_path, monkeypatch):
    """Before this, cancelling or erroring out of a turn wrote nothing to the transcript at all -
    the turn just vanished, same as a hard crash would. Every exit path now finalizes a turn_end,
    so /resume shows what happened instead of silently dropping the turn."""
    pidfile = tmp_path / "pid"
    steps = [call("1", "run_command", command=f"echo $$ > {pidfile}; sleep 30"), text("never")]
    async with tui_app(tmp_path, monkeypatch, steps, track=True) as (app, pilot):
        await send(app, pilot, "long")
        for _ in range(60):
            await pilot.pause(0.05)
            if pidfile.exists() and pidfile.read_text().strip():
                break
        await pilot.press("escape")
        await wait_idle(app, pilot)
        sid = list((tmp_path / ".motion" / "sessions").glob("*.jsonl"))[0].stem
        turns = tui.SessionStore.load(tmp_path, sid)
        assert len(turns) == 1
        assert turns[0]["prompt"] == "long"
        assert "cancelled" in turns[0]["response"].lower()
        assert not turns[0].get("interrupted")  # a clean cancel is recorded, not a bare "vanished"


async def test_a_turn_killed_before_it_can_finish_shows_up_as_interrupted_on_resume(tmp_path, monkeypatch):
    """The actual crash case: the process dies between start_turn (written before any tool runs)
    and the code that would write its turn_end - nothing further executes, so this simulates that
    boundary directly (a real kill -9 of the whole test process can't be scripted from inside it;
    core/command_watchdog.py's own tests cover that failure mode for run_command's child process
    at the OS level). /resume must show an explicit interrupted turn, not silently skip it."""
    async with tui_app(tmp_path, monkeypatch, [text("unused")], track=True) as (app, pilot):
        app.state.start_turn("started but the harness died right here")
        sid = list((tmp_path / ".motion" / "sessions").glob("*.jsonl"))[0].stem
        await send(app, pilot, "/resume")
        await pilot.pause(0.1)
        assert any(sid in t and "1 turn(s)" in t for t in system_lines(app))
        await send(app, pilot, f"/resume {sid}")
        await pilot.pause(0.2)
        assert app.state.conversation_turns == [
            ("started but the harness died right here", tui.INTERRUPTED_MARKER)
        ]


async def test_help_lists_new_commands_and_autocomplete_knows_them(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        await send(app, pilot, "/help")
        await pilot.pause(0.1)
        blob = "\n".join(system_lines(app))
        for cmd in ("/compact", "/undo", "/resume", "/todos", "/mcp", "grep"):
            assert cmd in blob
        names = {c for c, _ in tui.ChatComposer.SLASH_COMMANDS}
        assert {"/compact", "/undo", "/new", "/resume", "/todos", "/mcp"} <= names


async def test_default_agent_mode_can_be_configured(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], extra_cfg="default_agent_mode: build\n") as (app, _):
        assert app.initial_mode == "build"
    plain = tmp_path / "plain"
    plain.mkdir()
    async with tui_app(plain, monkeypatch, [text("x")]) as (app, _):
        assert app.initial_mode == "plan"  # unset => discuss-first, as before


# ── attachments ─────────────────────────────────────────────────────────────

async def test_text_attachment_is_sent_once_not_with_every_message(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("r1"), text("r2")]) as (app, pilot):
        app.state.attachments.append({"path": str(tmp_path / "notes.txt"), "type": "text", "content": "SECRET-PAYLOAD"})
        await send(app, pilot, "summarize")
        await wait_idle(app, pilot)
        await send(app, pilot, "and again")
        await wait_idle(app, pilot)
        first, second = app.provider.requests
        assert "SECRET-PAYLOAD" in first["messages"][-1]["content"]
        assert all("SECRET-PAYLOAD" not in str(m["content"]) for m in second["messages"])
        assert app.state.attachments == []
        # history keeps a short note, not the file contents
        assert app.state.conversation_turns[0][0] == "summarize\n[attached: notes.txt]"


async def test_image_attachment_goes_as_a_real_image_part_to_vision_models(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("i see it")], vision=True) as (app, pilot):
        app.state.attachments.append({"path": str(tmp_path / "p.png"), "type": "image", "mime": "image/png", "data": "QUJD"})
        await send(app, pilot, "what is this?")
        await wait_idle(app, pilot)
        content = app.provider.requests[0]["messages"][0]["content"]
        assert content[1] == {"type": "image", "mime": "image/png", "data": "QUJD"}


async def test_clear_reports_the_real_count(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        app.state.attachments.extend([{"path": "a"}, {"path": "b"}])
        await send(app, pilot, "/clear")
        await pilot.pause(0.1)
        assert any("Cleared 2 attached" in t for t in system_lines(app))


def test_pdf_attachments_use_the_current_pypdf_api(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = tmp_path / "blank.pdf"
    with open(pdf, "wb") as fh:
        writer.write(fh)
    pane = tui.ChatPane.__new__(tui.ChatPane)
    assert pane._extract_pdf(pdf) == ""  # blank page: no text, but no crash (used to raise on PdfFileReader)


def test_extract_attachment_rejects_empty_and_huge_images(tmp_path):
    pane = tui.ChatPane.__new__(tui.ChatPane)
    (tmp_path / "e.png").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        pane._extract_attachment(tmp_path / "e.png")
    (tmp_path / "big.png").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="MB"):
        pane._extract_attachment(tmp_path / "big.png")
    (tmp_path / "ok.png").write_bytes(b"\x89PNG" + b"x" * 400_000)
    att = pane._extract_attachment(tmp_path / "ok.png")
    import base64
    assert len(base64.b64decode(att["data"])) == 400_004  # complete, not truncated


# ── background tasks ────────────────────────────────────────────────────────

async def test_parallel_tasks_report_results_in_the_chat(tmp_path, monkeypatch):
    import core.orchestrator as orch
    from main import MotionAgent

    def fake_agent(config, memory_path=None, mcp_manager=None):
        agent = MotionAgent(config, memory_path=":memory:")
        agent.provider = Scripted([call("1", "write_file", path="out.txt", content="x"), text("task answer")])
        agent.retriever = _NoRecall()
        return agent

    monkeypatch.setattr(orch, "MotionAgent", fake_agent)
    async with tui_app(tmp_path, monkeypatch, [text("unused")]) as (app, pilot):
        await send(app, pilot, "/parallel first job ; second job")
        for _ in range(100):
            await pilot.pause(0.05)
            if sum("finished in" in t for t in system_lines(app)) + sum("failed" in t for t in system_lines(app)) >= 1:
                break
        lines = system_lines(app)
        assert any("finished in" in t and "task answer" in t and "_tool_" not in t for t in lines), lines
        assert any(".motion/tasks/" in t or ".motion" in t for t in lines)
        assert (tmp_path / ".motion" / "tasks").is_dir() and not (tmp_path / "tasks").exists()


async def test_mcp_command_reports_when_nothing_is_configured(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        await send(app, pilot, "/mcp")
        await pilot.pause(0.1)
        assert any("No MCP servers configured" in t for t in system_lines(app))


async def test_permission_rules_from_config_yml_are_honoured(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    cfg = 'permissions:\n  commands:\n    allow: ["rm -rf d"]\n'
    steps = [call("1", "run_command", command="rm -rf d"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps, extra_cfg=cfg) as (app, pilot):
        await send(app, pilot, "clean")
        await wait_idle(app, pilot)  # never blocks on an approval modal
        assert not (tmp_path / "d").exists()


# ── diffs ───────────────────────────────────────────────────────────────────

def diff_widgets(app):
    return list(app.screen.query(tui.DiffMessage))


async def test_edits_are_shown_as_diffs_but_new_files_are_not(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    steps = [
        call("1", "read_file", path="a.py"),
        call("2", "replace_in_file", path="a.py", old="x = 1", new="x = 100"),
        call("3", "write_file", path="new.py", content="print('hi')\n"),
        call("4", "write_file", path="a.py", content="x = 100\ny = 3\nz = 4\n"),
        text("done"),
    ]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "edit things")
        await wait_idle(app, pilot)
        widgets = diff_widgets(app)
        assert [w.path for w in widgets] == ["a.py", "a.py"]        # two edits; the new file has no diff
        assert "-x = 1" in widgets[0].diff_text and "+x = 100" in widgets[0].diff_text
        assert "-y = 2" in widgets[1].diff_text and "+z = 4" in widgets[1].diff_text
        assert len(app.state.turn_diffs) == 2 and app.state.turn_diffs[0][2:] == (1, 1)


async def test_diff_command_replays_the_last_turns_edits_and_toggles_inline(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("one\ntwo\n")
    steps = [call("1", "read_file", path="a.py"), call("2", "replace_in_file", path="a.py", old="one", new="uno"), text("done"),
             call("3", "read_file", path="a.py"), call("4", "replace_in_file", path="a.py", old="uno", new="eins"), text("again")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "first")
        await wait_idle(app, pilot)
        assert len(diff_widgets(app)) == 1
        await send(app, pilot, "/diff")
        await pilot.pause(0.1)
        assert len(diff_widgets(app)) == 2                              # replayed
        await send(app, pilot, "/diff off")
        await pilot.pause(0.1)
        assert app.state.show_diffs is False
        await send(app, pilot, "second")
        await wait_idle(app, pilot)
        assert len(diff_widgets(app)) == 2                              # nothing new inline...
        assert app.state.turn_diffs and "+eins" in app.state.turn_diffs[0][1]   # ...but still recorded for /diff
        await send(app, pilot, "/diff")
        await pilot.pause(0.1)
        assert len(diff_widgets(app)) == 3


async def test_diff_command_with_no_edits_says_so_and_config_can_disable_inline(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")], extra_cfg="show_diffs: false\n") as (app, pilot):
        assert app.state.show_diffs is False
        await send(app, pilot, "/diff")
        await pilot.pause(0.1)
        assert any("No file edits" in t for t in system_lines(app))


async def test_long_diffs_are_truncated_inline_but_complete_in_the_command(tmp_path, monkeypatch):
    (tmp_path / "big.txt").write_text("".join(f"line {i}\n" for i in range(60)))
    new = "".join(f"LINE {i}\n" for i in range(60))
    steps = [call("1", "read_file", path="big.txt"), call("2", "write_file", path="big.txt", content=new), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "rewrite")
        await wait_idle(app, pilot)
        inline = diff_widgets(app)[0]
        assert "more line(s)" in _text_of_group(inline)
        assert inline.diff_text.count("\n") > 24                        # the full text is kept


def _text_of_group(widget) -> str:
    from rich.console import Console

    console = Console(width=120, record=True, file=open(os.devnull, "w"))
    console.print(widget.renderable if hasattr(widget, "renderable") else widget.content)
    return console.export_text()


# ── readability: thinking, steps, scrolling, token perception ───────────────

def test_human_tool_labels_and_one_line_truncation():
    assert tui._tool_label("run_command") == "running a command"
    assert tui._tool_label("mcp__github__x") == "calling an MCP tool"
    assert tui._tool_label("some_new_tool") == "some new tool"  # unknown names still read as words
    long = "echo start\n" + "x" * 300
    assert tui._one_line(long) == "echo start"
    assert len(tui._one_line("y" * 300)) == 110 and tui._one_line("y" * 300).endswith("…")
    assert (tui._fmt_tok(950), tui._fmt_tok(18_250), tui._fmt_tok(2_400_000)) == ("950", "18.2k", "2.4M")


async def test_command_steps_are_one_short_line_and_gone_after_the_turn(tmp_path, monkeypatch):
    heredoc = "cat <<'EOF' > /dev/null\n" + "line\n" * 40 + "EOF"
    steps = [call("1", "run_command", command=heredoc), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "go")
        await wait_idle(app, pilot)
        assert not list(app.screen.query(tui.StepsMessage))  # nothing left cluttering the chat
        pane = app.screen.query_one(tui.ChatPane)
        progress = [ln for ln in pane._trace_lines if "tool.progress" in ln]
        assert progress and "line" not in progress[0]  # only the first line of the heredoc reached the UI


async def test_status_reports_context_size_separately_from_summed_input(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        await send(app, pilot, "q")
        await wait_idle(app, pilot)
        app.state.last_turn_metrics.update(
            context_tokens=18_250, prompt_tokens_est=220_000, cached_tokens=190_000,
            output_tokens_est=3_000, tokens_are_real=True, tool_calls=6,
        )
        app.state.session_metrics.update(prompt_tokens_est=220_000, cached_tokens_est=190_000, output_tokens_est=3_000)
        app.screen.query_one(tui.ChatPane)._refresh_status()
        status = _text_of(app.screen.query_one("#chat_status_text"))
        assert "ctx 18.2k" in status and "in 220.0k (190.0k cached)" in status and "out 3.0k" in status
        assert "6 tool calls" in status


async def test_streaming_does_not_yank_the_view_back_while_you_read_above(tmp_path, monkeypatch):
    steps = [call("1", "run_command", command="sleep 1"), text(*[f"chunk {i}\n\n" for i in range(60)])]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        log = app.screen.query_one("#chat_log")
        for i in range(80):
            await log.mount(tui.SystemMessage(f"filler {i}"))
        await send(app, pilot, "go")
        for _ in range(40):
            await pilot.pause(0.05)
            if app.state.busy:
                break
        log.scroll_to(y=0, animate=False)  # the user scrolls up to read
        await pilot.pause(0.1)
        await wait_idle(app, pilot)
        assert log.scroll_y == 0
