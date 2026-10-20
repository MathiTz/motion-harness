"""The session rail (ui/rail.py): pure renderers first, then the real TUI turn."""
from types import SimpleNamespace

from rich.console import Console

from core.budget import Budget
from tests.test_agent_loop import call, text
from tests.test_tui_flow import _text_of, send, tui_app, wait_idle
from ui.rail import RAIL_MIN_WIDTH, RailData, bar, budget_text, compact, files_text, plan_text, recall_text, safety_text

COLORS = {"ok": "green", "fail": "red", "warn": "yellow", "run": "cyan", "plain": "", "learned": "magenta", "info": "blue"}


def plain(t) -> str:
    return t.plain


# ── data ─────────────────────────────────────────────────────────────────────

def test_note_edit_merges_per_file_and_lists_the_most_recent_first():
    d = RailData()
    d.note_edit("a.py", 3, 1)
    d.note_edit("b.py", 0, 0, created=True)
    d.note_edit("a.py", 2, 0)
    assert list(d.files) == ["b.py", "a.py"]  # a.py was touched last
    assert d.files["a.py"] == {"added": 5, "removed": 1, "created": False}
    assert d.files["b.py"]["created"] is True


def test_new_session_clears_the_conversation_state_but_keeps_the_sandbox_fact():
    d = RailData(sandbox={"active": True})
    d.note_edit("a.py", 1, 1)
    d.recalled, d.use = [{"content": "x"}], {"steps": 3}
    d.new_session()
    assert d.files == {} and d.recalled is None and d.use == {} and d.sandbox == {"active": True}


def test_number_helpers():
    assert (compact(950), compact(18_250), compact(150_000), compact(2_400_000)) == ("950", "18.2k", "150k", "2.4M")
    assert bar(0.5) == "█████░░░░░" and bar(0) == "░" * 10 and bar(2) == "█" * 10 and bar(-1) == "░" * 10


# ── renderers ────────────────────────────────────────────────────────────────

def test_plan_shows_status_and_hides_when_there_is_no_plan():
    todos = [{"content": "find the test", "status": "completed"}, {"content": "fix it", "status": "in_progress"},
             {"content": "run the suite", "status": "pending"}]
    assert plain(plan_text(todos, COLORS)).splitlines() == ["PLAN", "✓ find the test", "● fix it", "○ run the suite"]
    assert plain(plan_text([], COLORS)) == ""


def test_files_list_newest_first_with_created_marker_and_a_cap():
    files = {f"f{i}.py": {"added": i, "removed": 0, "created": i == 0} for i in range(11)}
    out = plain(files_text(files, COLORS)).splitlines()
    assert out[0] == "FILES TOUCHED (11)" and out[1].startswith("f10.py") and "+10" in out[1]
    assert out[-1] == "… 3 more" and len(out) == 1 + 8 + 1
    assert plain(files_text({"n.py": {"added": 0, "removed": 0, "created": True}}, COLORS)).splitlines()[1] == "+ n.py"
    assert plain(files_text({}, COLORS)) == ""


def test_recall_shows_age_and_confidence_and_flags_a_weak_match():
    items = [
        {"content": "apply_discount lives in cart.py", "raw_score": 0.82, "age": "3d ago"},
        {"content": "team prefers tabs", "raw_score": 0.41, "age": "6mo ago"},
        {"content": "deploy script is scripts/ship.sh", "raw_score": None, "age": "today"},
    ]
    out = plain(recall_text(items, COLORS))
    assert "◆ 0.82 · 3d ago" in out and "◆ 0.41 · 6mo ago · weak" in out and "◆ keyword · today" in out
    assert out.count("weak") == 1  # the strong match and the keyword hit are not flagged


def test_recall_distinguishes_nothing_recalled_from_not_yet_run():
    assert plain(recall_text(None, COLORS)) == ""
    assert "nothing this turn" in plain(recall_text([], COLORS))


def test_recall_clips_long_content():
    out = plain(recall_text([{"content": "word " * 100, "raw_score": 0.9}], COLORS))
    assert out.splitlines()[-1].endswith("…") and len(out.splitlines()[-1]) <= 69


def test_budget_with_no_limits_says_how_to_set_one():
    out = plain(budget_text(Budget(), {}, COLORS))
    assert "no limit set" in out and "/budget" in out
    assert "no limit set" in plain(budget_text(None, {}, COLORS))


def test_budget_bars_warn_near_the_limit_and_fail_at_it():
    b = Budget(max_steps=10, max_cost_usd=0.5)
    out = plain(budget_text(b, {"steps": 4, "cost": 0.012}, COLORS)).splitlines()
    assert out[1].startswith("steps") and out[1].endswith("4/10") and "tokens" not in "".join(out)
    assert out[2].endswith("$0.012/$0.500")

    def style_of(used):
        t = budget_text(Budget(max_steps=10), {"steps": used}, COLORS)
        return next(sp.style for sp in t.spans if "█" in t.plain[sp.start:sp.end] or "░" in t.plain[sp.start:sp.end])

    assert (style_of(4), style_of(8), style_of(10)) == ("blue", "yellow", "red")


def test_budget_before_any_use_shows_dashes_not_zero():
    assert "–/12" in plain(budget_text(Budget(max_steps=12), {}, COLORS))


def test_safety_states():
    assert "first turn" in plain(safety_text(None, COLORS))
    assert "● sandbox on" in plain(safety_text({"active": True}, COLORS))
    off = plain(safety_text({"active": False}, COLORS))
    assert "○ sandbox off" in off and "policy checks only" in off


# ── in the real TUI ──────────────────────────────────────────────────────────

class Recalling:
    semantic_available = True

    async def retrieve(self, q, top_k=5):
        return [
            {"content": "apply_discount lives in cart.py", "score": 0.03, "type": "hybrid", "raw_score": 0.82, "age": "3d ago"},
            {"content": "team prefers tabs", "score": 0.02, "type": "semantic", "raw_score": 0.41, "age": "6mo ago"},
        ]


def block(app, name: str) -> str:
    return _text_of(app.screen.query_one(f"#rail_{name}"))


def shown(app, name: str) -> bool:
    return app.screen.query_one(f"#rail_{name}").display


async def test_the_rail_replaces_the_context_panel_and_does_not_repeat_the_chat(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        assert "Session" in _text_of(app.screen.query_one("#context_header"))
        await send(app, pilot, "hello there")
        await wait_idle(app, pilot)
        body = " ".join(_text_of(w) for w in app.screen.query("#context_body Static"))
        assert "Recent turns" not in body and "BUDGET" in body and "SAFETY" in body
        assert tui_alias_is_the_rail()


def tui_alias_is_the_rail() -> bool:
    import ui.tui as tui
    from ui.rail import SessionRail

    return tui.ContextPanel is SessionRail


async def test_files_touched_accumulate_across_turns_and_clear_on_new(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="a.txt", content="1\n"), text("one"),
             call("2", "write_file", path="b.txt", content="2\n"), text("two")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "first")
        await wait_idle(app, pilot)
        assert "FILES TOUCHED (1)" in block(app, "files")
        await send(app, pilot, "second")
        await wait_idle(app, pilot)
        files = block(app, "files")
        assert "FILES TOUCHED (2)" in files and "b.txt" in files and "a.txt" in files
        await send(app, pilot, "/new")
        await pilot.pause(0.2)
        assert not shown(app, "files")


async def test_recalled_memory_shows_with_age_and_a_weak_flag(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("answer")]) as (app, pilot):
        app.state.agent.retriever = Recalling()
        assert not shown(app, "recall")  # nothing before a turn has run
        await send(app, pilot, "where is the discount logic?")
        await wait_idle(app, pilot)
        recall = block(app, "recall")
        assert "◆ 0.82" in recall and "3d ago" in recall and "◆ 0.41" in recall and "weak" in recall


async def test_a_turn_with_nothing_recalled_says_so_rather_than_keeping_the_last_turns(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("one"), text("two")]) as (app, pilot):
        app.state.agent.retriever = Recalling()
        await send(app, pilot, "first")
        await wait_idle(app, pilot)
        assert "weak" in block(app, "recall")
        app.state.agent.retriever = SimpleNamespace(semantic_available=True, retrieve=_none)
        await send(app, pilot, "second")
        await wait_idle(app, pilot)
        assert "nothing this turn" in block(app, "recall") and "weak" not in block(app, "recall")


async def _none(q, top_k=5):
    return []


async def test_budget_block_shows_the_limit_and_what_the_turn_used(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="a.txt", content="x"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        assert "no limit set" in block(app, "budget")
        app.state.agent.budget = Budget(max_steps=5)
        await send(app, pilot, "go")
        await wait_idle(app, pilot)
        out = block(app, "budget")
        assert "steps" in out and "2/5" in out


async def test_safety_block_reports_the_sandbox_after_the_first_turn(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hi")]) as (app, pilot):
        assert "first turn" in block(app, "safety")
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        assert app.state.rail.sandbox is not None
        assert "first turn" not in block(app, "safety") and "sandbox" in block(app, "safety")


async def test_the_rail_hides_on_a_narrow_terminal_and_ctrl_b_still_brings_it_back(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        rail = app.screen.query_one("#context_panel")
        assert rail.styles.display == "block"  # the harness runs at 150 columns
        await pilot.resize_terminal(RAIL_MIN_WIDTH - 10, 40)
        await pilot.pause(0.2)
        assert rail.styles.display == "none"   # too narrow: the chat gets the room
        await pilot.press("ctrl+b")
        await pilot.pause(0.1)
        assert rail.styles.display == "block"  # an explicit request wins over the auto-hide
        await pilot.press("ctrl+b")
        await pilot.pause(0.1)
        assert rail.styles.display == "none"
        await pilot.resize_terminal(150, 40)
        await pilot.pause(0.2)
        assert rail.styles.display == "none"   # the user turned it off: wide terminals don't override that
