"""The turn card and the "Done" strip (ui/turncard.py): pure TurnLog logic, then the real TUI turn."""
import time

from rich.console import Console

import ui.tui as tui
from tests.test_agent_loop import call, text
from tests.test_tui_flow import _text_of, send, tui_app, wait_idle
from ui.turncard import ActionChip, TurnLog, TurnSummary, headline_parts, line_role, render_card

COLORS = {"ok": "green", "fail": "red", "warn": "yellow", "run": "cyan", "plain": ""}


def plain(renderable) -> str:
    console = Console(width=120, color_system=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def done(log: TurnLog, tool: str, path: str = "", **kw) -> None:
    log.start(tool, path)
    log.finish(tool, path, **kw)


# ── TurnLog ──────────────────────────────────────────────────────────────────

def test_summary_says_what_was_done_from_successful_calls():
    log = TurnLog()
    for p in ("a.py", "b.py", "a.py"):
        done(log, "read_file", p)
    done(log, "replace_in_file", "cart.py")
    done(log, "run_command")
    done(log, "run_command")
    assert log.summary_parts() == ["read 2 files", "edited cart.py", "ran 2 commands"]


def test_created_and_edited_files_are_told_apart_and_long_lists_collapse():
    log = TurnLog()
    done(log, "write_file", "new.py", created=True)
    for p in ("a.py", "b.py", "c.py"):
        done(log, "replace_in_file", p)
    assert log.summary_parts() == ["created new.py", "edited 3 files"]


def test_a_refused_write_is_never_reported_as_an_edit():
    log = TurnLog()
    done(log, "write_file", "x.py", blocked=True)     # Plan mode refused it
    done(log, "replace_in_file", "y.py", ok=False)    # and this one failed
    assert log.summary_parts() == ["1 failed", "1 blocked"]
    assert "edited" not in " ".join(log.summary_parts())
    assert log.failed() == 1 and log.blocked() == 1


def test_finish_matches_the_right_call_when_parallel_calls_end_out_of_order():
    log = TurnLog()
    a, b = log.start("read_file", "a.py"), log.start("read_file", "b.py")
    log.finish("read_file", "b.py")
    assert b.ok is True and a.ok is None and log.running() == [a]


def test_an_event_for_a_call_that_never_started_closes_nothing():
    log = TurnLog()
    running = log.start("run_command")
    assert log.finish("invalid", ok=False) is None
    assert running.ok is None


def test_finish_records_the_path_a_tool_only_reported_at_the_end():
    log = TurnLog()
    call_ = log.start("replace_in_file")
    log.finish("replace_in_file", "late.py")
    assert call_.path == "late.py" and log.summary_parts() == ["edited late.py"]


# ── the card ─────────────────────────────────────────────────────────────────

def test_line_roles():
    assert line_role("wrote `a.py` (3 bytes)") == "ok"
    assert line_role("`run_command` failed: boom") == "fail"
    assert line_role("⚠️ glm failed (rate limit); switching to claude") == "fail"
    assert line_role("ℹ️ Using Claude Code with your login") == "info"


def test_card_shows_finished_work_then_what_is_running_now():
    log = TurnLog()
    log.add_line("read `cart.py`")
    log.add_line("`run_command` failed: exit 1")
    log.start("replace_in_file", "cart.py")
    out = plain(render_card(log, COLORS))
    assert out.splitlines() == ["✓ read `cart.py`", "✗ `run_command` failed: exit 1", "● editing `cart.py`…"]


def test_card_folds_old_lines_and_never_ends_with_a_blank_row():
    log = TurnLog()
    for i in range(10):
        log.add_line(f"read `f{i}.py`")
    out = plain(render_card(log, COLORS, rows=4))
    assert out.splitlines()[0].strip() == "… 6 earlier" and "f9.py" in out and not out.endswith("\n\n")


# ── the headline ─────────────────────────────────────────────────────────────

def joined(parts) -> str:
    return "".join(t for t, _ in parts)


def test_headline_prefers_files_changed_and_adds_time_and_cost():
    log = TurnLog()
    done(log, "replace_in_file", "cart.py")
    assert joined(headline_parts(log, 1, 3, 4, 4.94, "$0.005")) == "✓ Done · 1 file changed (+3 −4) · 4.9s · $0.005"
    assert joined(headline_parts(log, 2, 0, 0, 1.0, None)) == "✓ Done · 2 files changed · 1.0s"


def test_headline_falls_back_to_the_summary_and_flags_failures():
    log = TurnLog()
    done(log, "read_file", "a.py")
    done(log, "run_command")
    assert joined(headline_parts(log, 0, 0, 0, 2.0, None)) == "✓ Done · read 1 file · ran 1 command · 2.0s"
    done(log, "run_command", ok=False)
    out = headline_parts(log, 0, 0, 0, 2.0, None)
    assert out[0] == ("⚠ Done with 1 failed tool call", "warn")


# ── in the real TUI ──────────────────────────────────────────────────────────

def strips(app):
    return list(app.screen.query(TurnSummary))


def chip_labels(strip) -> list:
    return [str(c.renderable.plain if hasattr(c.renderable, "plain") else c.renderable) for c in strip.query(ActionChip)]


async def test_a_turn_that_edits_a_file_ends_with_a_done_strip_and_actions(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="new.txt", content="hello"), text("wrote it")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "make a file")
        await wait_idle(app, pilot)
        (strip,) = strips(app)
        headline = _text_of(strip.query_one("#summary_text"))
        assert "✓ Done" in headline and "1 file changed" in headline
        labels = " | ".join(chip_labels(strip))
        assert "Undo edit /undo" in labels and "Trace F8" in labels and "Copy answer F9" in labels
        # the strip sits directly under the answer
        log = app.screen.query_one("#chat_log")
        kinds = [type(w).__name__ for w in log.children]
        assert kinds.index("TurnSummary") == kinds.index("AgentMessage") + 1
        reasoning = [t for k, t in [(type(w).__name__, _text_of(w)) for w in log.children] if k == "ReasoningMessage"]
        assert reasoning and "created new.txt" in reasoning[0]


async def test_clicking_undo_reverts_the_edit_and_retires_the_chip(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="new.txt", content="hello"), text("wrote it")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "make a file")
        await wait_idle(app, pilot)
        assert (tmp_path / "new.txt").exists()
        await pilot.click("#chip_undo_edit")
        for _ in range(40):
            await pilot.pause(0.05)
            if not (tmp_path / "new.txt").exists():
                break
        assert not (tmp_path / "new.txt").exists()
        (strip,) = strips(app)
        await pilot.pause(0.1)
        assert not strip.active and not list(strip.query(ActionChip))  # one undo per strip


async def test_a_plain_answer_with_no_tool_calls_gets_no_strip(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("just talking")]) as (app, pilot):
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        assert strips(app) == []


async def test_an_old_strips_actions_are_gone_once_the_next_turn_starts(tmp_path, monkeypatch):
    """undo/diff/copy act on the last turn, so a stale chip must not be able to reach a newer one."""
    steps = [call("1", "write_file", path="a.txt", content="1"), text("first"),
             call("2", "write_file", path="b.txt", content="2"), text("second")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "one")
        await wait_idle(app, pilot)
        await send(app, pilot, "two")
        await wait_idle(app, pilot)
        await pilot.pause(0.1)
        first, second = strips(app)
        assert not first.active and not list(first.query(ActionChip))
        assert second.active and list(second.query(ActionChip))
        assert "1 file changed" in _text_of(first.query_one("#summary_text"))  # the record stays


async def test_plan_mode_refusal_is_reported_as_blocked_not_edited(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="x.py", content="x"), text("here is my plan")]
    async with tui_app(tmp_path, monkeypatch, steps, mode="plan") as (app, pilot):
        await send(app, pilot, "change x")
        await wait_idle(app, pilot)
        log = app.screen.query_one("#chat_log")
        process = [_text_of(w) for w in log.children if type(w).__name__ == "ReasoningMessage"]
        assert process and "blocked" in process[0] and "edited" not in process[0] and "created" not in process[0]
        assert not (tmp_path / "x.py").exists()
