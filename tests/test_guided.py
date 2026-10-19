"""Guided dialogs (redesign step 5): the approval explainer, the grouped palette, next-step chips.

Nothing here runs a destructive command: approval tests only read the dialog and then deny, or use
a command that fails harmlessly in an empty temp directory.
"""
import asyncio
import time

import ui.tui as tui
from tests.test_agent_loop import call, text
from tests.test_tui_flow import _text_of, send, tui_app, wait_for_screen, wait_idle
from ui.commands import GROUPS, catalog
from ui.explain import NO_UNDO, explain_code, explain_command, explain_network, where
from ui.turncard import REVIEW_PROMPT, ActionChip, TurnSummary, next_steps

WS = "/work/proj"


def rendered(widget) -> str:
    """The text a Static (holding a Rich Table or Text) actually draws, wrapped at 80 columns."""
    from rich.console import Console

    console = Console(width=80, color_system=None)
    with console.capture() as cap:
        console.print(widget.renderable)
    return " ".join(cap.get().split())


def rows(explanation):
    return {label or "note": (body, role) for label, body, role in explanation.rows}


# ── the explainer ────────────────────────────────────────────────────────────

def test_rm_says_what_goes_and_that_it_is_inside_the_project():
    r = rows(explain_command("rm -rf build/", "deletes files recursively/forcefully", WS))
    assert r["What it does"][0] == "deletes `build/` and everything inside"
    assert r["Where"] == ("inside this project", "ok")
    assert r["Why asked"][0] == "deletes files recursively/forcefully" and r["note"][0] == NO_UNDO


def test_a_path_that_escapes_the_project_is_flagged_even_when_it_does_not_exist():
    for cmd in ("rm -rf ../other", "rm -rf /tmp/x", "rm -rf ~/stuff", "rm -rf $HOME/stuff"):
        body, role = rows(explain_command(cmd, "", WS, home="/home/me"))["Where"]
        assert role == "warn" and body.startswith("outside this project"), cmd


def test_a_sibling_directory_with_a_shared_prefix_is_not_inside():
    assert where(["/work/proj-other/x"], WS)[2] == "warn"   # the classic startswith trap
    assert where(["/work/proj/sub/x", "sub/y", "./z"], WS)[2] == "ok"
    assert where(["sub/../../escape"], WS)[2] == "warn"


def test_sudo_is_called_out_and_its_targets_still_checked():
    r = rows(explain_command("sudo rm -rf /var/log/x", "runs with elevated privileges", WS))
    assert r["What it does"][0].startswith("as an administrator, deletes `/var/log/x`")
    assert r["Where"][1] == "warn"


def test_git_commands_are_described_by_their_effect():
    assert "publishes your commits" in rows(explain_command("git push origin main", "", WS))["What it does"][0]
    assert "force push" in rows(explain_command("git push --force origin main", "", WS))["What it does"][0]
    assert "discards all uncommitted changes" in rows(explain_command("cd app && git reset --hard HEAD~1", "", WS))["What it does"][0]
    assert "files git doesn't track" in rows(explain_command("git clean -fd", "", WS))["What it does"][0]


def test_pipe_to_shell_is_described_plainly():
    assert rows(explain_command("curl https://x.sh | sh", "", WS))["What it does"][0] == "downloads a script and runs it immediately"


def test_an_unrecognised_command_gets_the_policy_reason_not_a_guess():
    r = rows(explain_command("weirdtool --nuke", "matches permissions.commands.ask (weirdtool*)", WS))
    assert r["What it does"][0] == "runs a command that matched a safety rule"
    assert r["Why asked"][0] == "matches permissions.commands.ask (weirdtool*)"
    assert "Where" not in r  # no file operands to place


def test_network_and_code_explanations():
    n = rows(explain_network("http://10.0.0.5/admin", "10.0.0.5 is a local/private network address"))
    assert "private-network address: http://10.0.0.5/admin" in n["What it does"][0] and "Why asked" in n
    c = rows(explain_code("run_python: os.remove('x')", "deletes files/directories"))
    assert c["Why asked"][0] == "deletes files/directories" and c["note"][0] == NO_UNDO and "What it does" not in c


# ── the palette catalogue ────────────────────────────────────────────────────

def test_catalogue_is_grouped_in_the_documented_order_and_actions_are_unique():
    cmds = catalog()
    order = [c.group for c in cmds]
    assert sorted(set(order), key=order.index) == [g for g in GROUPS if g in order] == list(GROUPS)
    assert order == sorted(order, key=GROUPS.index)  # contiguous: one heading per group
    actions = [c.action for c in cmds]
    assert len(actions) == len(set(actions))


def test_every_slash_row_prints_the_command_it_runs():
    for c in catalog():
        if c.action.startswith("slash:"):
            assert c.hint == c.action[len("slash:"):], c


def test_catalogue_shows_live_state():
    by = {c.action: c for c in catalog(skills_waiting=2, tracking_on=True)}
    assert by["slash:/skill candidates"].note == "2 waiting" and by["tracking"].note == "on"
    by = {c.action: c for c in catalog(skills_waiting=0, tracking_on=False)}
    assert by["slash:/skill candidates"].note == "none waiting" and by["tracking"].note == "off"


# ── next steps ───────────────────────────────────────────────────────────────

def test_next_steps_follow_what_happened():
    edited = next_steps(True, 0, "build")
    assert [s.label for s in edited] == ["Add a test for this change", "Run the full test suite", "Review my changes"]
    assert edited[2].prompt == REVIEW_PROMPT and edited[2].mode == "plan"   # review is read-only
    assert next_steps(False, 1, "build")[0].label == "Diagnose the failure"
    plan = next_steps(False, 0, "plan")
    assert [(s.label, s.mode) for s in plan] == [("Switch to Build and do it", "build")]
    assert next_steps(False, 0, "build") == []   # nothing happened that suggests a next move


def test_next_steps_never_exceed_three():
    assert len(next_steps(True, 2, "build")) == 3


# ── in the real TUI ──────────────────────────────────────────────────────────

async def test_the_approval_dialog_explains_the_command_in_plain_words(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    steps = [call("1", "run_command", command="rm -rf d"), text("ok")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "clean up")
        screen = await wait_for_screen(app, pilot, tui.PermissionScreen)
        body = rendered(screen.query_one("#permission_explain"))
        assert "deletes `d` and everything inside" in body and "inside this project" in body
        assert "Undo can't restore this" in body
        assert "rm -rf d" in _text_of(screen.query_one("#permission_path"))
        await pilot.press("escape")  # deny: nothing is deleted
        await wait_idle(app, pilot)
        assert (tmp_path / "d").exists()


async def test_letter_shortcuts_are_ignored_right_after_the_dialog_appears(tmp_path, monkeypatch):
    """A prompt popping up while you type a chat message must not be answered by your own keystroke."""
    (tmp_path / "d").mkdir()
    steps = [call("1", "run_command", command="rm -rf d"), text("ok")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "clean up")
        screen = await wait_for_screen(app, pilot, tui.PermissionScreen)
        await pilot.press("y")                       # inside the grace period: ignored
        await pilot.pause(0.1)
        assert app.screen is screen and (tmp_path / "d").exists()
        await asyncio.sleep(tui.PermissionScreen.LETTER_GRACE_S)
        await pilot.press("n")                       # after it: a real answer
        await wait_idle(app, pilot)
        assert (tmp_path / "d").exists()
        assert "user denied" in str(app.provider.requests[1]["messages"][-1]["content"])


async def test_y_allows_once_after_the_grace_period(tmp_path, monkeypatch):
    # `git stash clear` is on the risky list and fails harmlessly in an empty, non-repo directory
    steps = [call("1", "run_command", command="git stash clear"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "tidy")
        await wait_for_screen(app, pilot, tui.PermissionScreen)
        await asyncio.sleep(tui.PermissionScreen.LETTER_GRACE_S)
        await pilot.press("y")
        await wait_idle(app, pilot)
        assert "user denied" not in str(app.provider.requests[1]["messages"][-1]["content"])


async def test_out_of_workspace_prompt_still_works_without_an_explanation(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        screen = tui.PermissionScreen("/etc/hosts")
        await app.push_screen(screen)
        await pilot.pause(0.1)
        assert not list(screen.query("#permission_explain"))
        assert "y once" in _text_of(screen.query_one("#permission_hint"))


async def open_palette(app, pilot):
    await pilot.press("ctrl+k")
    await pilot.pause(0.2)
    return app.screen


def row_text(screen, label: str) -> str:
    idx = [lbl for lbl, _ in screen._commands].index(label)
    return _text_of(screen._row_labels[idx])


async def test_palette_is_grouped_with_headings_hints_and_live_notes(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], track=True) as (app, pilot):
        screen = await open_palette(app, pilot)
        assert row_text(screen, "New conversation").startswith("[dim bold]WORK")
        assert "/undo" in row_text(screen, "Undo the last turn's edits")
        assert "on" in row_text(screen, "Toggle interaction tracking")
        assert "Ctrl+O" in row_text(screen, "Switch model…")
        headings = [t for t in (_text_of(r) for r in screen._row_labels) if "[dim bold]" in t]
        assert len(headings) == len(GROUPS)


async def test_filtering_the_palette_keeps_one_heading_on_the_first_match_only(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        screen = await open_palette(app, pilot)
        screen.query_one("#palette_input").value = "copy"
        await pilot.pause(0.2)
        visible = [i for i, e in enumerate(screen._entries) if screen.query_one("#palette_list").children[i].display]
        assert [screen._entries[i].group for i in visible] == ["REVIEW"] * len(visible) and len(visible) >= 3
        headed = [i for i in visible if "[dim bold]REVIEW" in _text_of(screen._row_labels[i])]
        assert headed == [visible[0]]


async def test_palette_can_filter_by_the_key_it_teaches(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        screen = await open_palette(app, pilot)
        screen.query_one("#palette_input").value = "/undo"
        await pilot.pause(0.2)
        shown = [e.label for i, e in enumerate(screen._entries) if screen.query_one("#palette_list").children[i].display]
        assert shown == ["Undo the last turn's edits"]


async def test_a_slash_row_runs_the_command_without_touching_the_draft(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="a.txt", content="x"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps) as (app, pilot):
        await send(app, pilot, "make a file")
        await wait_idle(app, pilot)
        assert (tmp_path / "a.txt").exists()
        box = app.screen.query_one("#chat_input")
        box.value, box.cursor_position = "a half-written message", 22
        screen = await open_palette(app, pilot)
        screen.query_one("#palette_input").value = "undo the last"
        await pilot.pause(0.2)
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if not (tmp_path / "a.txt").exists():
                break
        assert not (tmp_path / "a.txt").exists()                     # /undo ran
        assert app.screen.query_one("#chat_input").value == "a half-written message"  # the draft survived


async def test_next_step_chips_fill_the_box_and_never_send(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="a.txt", content="x"), text("done")]
    async with tui_app(tmp_path, monkeypatch, steps, mode="build") as (app, pilot):
        await send(app, pilot, "make a file")
        await wait_idle(app, pilot)
        (strip,) = list(app.screen.query(TurnSummary))
        labels = [str(c.renderable.plain) for c in strip.query("#summary_next ActionChip")]
        assert labels == ["Add a test for this change", "Run the full test suite", "Review my changes"]
        requests_before = len(app.provider.requests)
        await pilot.click("#next_review_my_changes")
        await pilot.pause(0.2)
        assert app.screen.query_one("#chat_input").value == REVIEW_PROMPT
        assert app.state.agent_mode == "plan"                        # review is read-only
        assert len(app.provider.requests) == requests_before and not app.state.busy


async def test_plan_mode_answer_offers_the_handoff_to_build(tmp_path, monkeypatch):
    steps = [call("1", "read_file", path="a.txt"), text("here is the plan")]
    (tmp_path / "a.txt").write_text("x")
    async with tui_app(tmp_path, monkeypatch, steps, mode="plan") as (app, pilot):
        await send(app, pilot, "plan it")
        await wait_idle(app, pilot)
        await pilot.click("#next_switch_to_build_and_do_it")
        await pilot.pause(0.2)
        assert app.state.agent_mode == "build"
        assert app.screen.query_one("#chat_input").value == "Go ahead and implement the plan."


async def test_next_step_chips_retire_with_the_strip(tmp_path, monkeypatch):
    steps = [call("1", "write_file", path="a.txt", content="x"), text("one"), text("two")]
    async with tui_app(tmp_path, monkeypatch, steps, mode="build") as (app, pilot):
        await send(app, pilot, "first")
        await wait_idle(app, pilot)
        await send(app, pilot, "second")
        await wait_idle(app, pilot)
        await pilot.pause(0.1)
        (strip,) = list(app.screen.query(TurnSummary))   # the second turn used no tools, so no new strip
        assert not strip.active and not list(strip.query(ActionChip)) and not list(strip.query("#summary_next"))
