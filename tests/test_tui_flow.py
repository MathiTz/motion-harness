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
        assert "running run_command" in blob and "Esc cancels" in blob
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
