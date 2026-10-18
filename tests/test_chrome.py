"""Cobalt theme and the window chrome (ui/chrome.py): the status bar, the hint bar and the mode colours.

The theme tests pin the contrast numbers the palette was adjusted for, so a later colour tweak that
quietly makes muted text unreadable fails here instead of in someone's terminal.
"""
import shutil
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

import ui.tui as tui
from tests.test_agent_loop import text
from tests.test_tui_flow import tui_app
from ui.chrome import HintBar, StatusBar, git_branch, short_path
from ui.themes import ThemeRegistry


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    chans = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def plain(renderable) -> str:
    console = Console(width=100, color_system=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


# ── theme ────────────────────────────────────────────────────────────────────

def test_cobalt_is_registered_and_the_default_for_a_fresh_state():
    assert "cobalt" in ThemeRegistry.theme_ids()
    assert tui.AppState().current_theme == "cobalt"
    assert "opencode" in ThemeRegistry.theme_ids()  # still selectable, just no longer the default


def test_cobalt_gives_each_colour_one_meaning():
    t = ThemeRegistry.get_textual_theme("cobalt")
    assert t.primary.upper() == "#FFC600"    # you / Build
    assert t.secondary.upper() == "#0088FF"  # the agent / Plan
    assert t.accent.upper() == "#9EFFFF"     # learned: memory, skills
    assert len({t.primary, t.secondary, t.accent, t.warning, t.error, t.success}) == 6


def test_every_theme_defines_the_custom_variables_our_css_can_use():
    """An undefined $variable is a CSS error, so a variable only Cobalt had would break the rest."""
    for tid in ThemeRegistry.theme_ids():
        variables = ThemeRegistry.get_textual_theme(tid).variables
        assert variables.get("blue-text") and variables.get("error-text"), tid


def test_cobalt_text_colours_stay_readable():
    t = ThemeRegistry.get_textual_theme("cobalt")
    v = t.variables
    assert contrast(v["text-muted"], t.background) >= 4.5
    assert contrast(v["text-muted"], t.surface) >= 4.5
    assert contrast(v["blue-text"], t.background) >= 4.5   # the raw #0088FF is only 3.6:1 as text
    assert contrast(v["error-text"], t.background) >= 4.5
    assert contrast(t.foreground, t.background) >= 7
    assert contrast(t.primary, t.background) >= 7


def test_cobalt_borders_are_visible_not_cobalt2s_invisible_line_colour():
    t = ThemeRegistry.get_textual_theme("cobalt")
    assert t.variables["border"].upper() == "#2A5A7A"
    assert contrast(t.variables["border"], t.background) >= 1.5
    assert contrast("#0D3A58", t.background) < 1.2  # the colour we deliberately did not use


def test_cobalt_has_a_code_highlight_style():
    assert tui.ChatPane._theme_code_style("cobalt") != "default"


# ── helpers ──────────────────────────────────────────────────────────────────

def test_short_path_abbreviates_home_and_keeps_the_tail_of_a_long_path():
    assert short_path("/Users/me/proj/shop", home="/Users/me") == "~/proj/shop"
    assert short_path("/Users/me", home="/Users/me") == "~"
    assert short_path("/Users/mellow/x", home="/Users/me") == "/Users/mellow/x"  # not a prefix match on a name
    long = short_path("/a/" + "deep/" * 20 + "leaf", home="/nowhere", limit=30)
    assert len(long) == 30 and long.startswith("…") and long.endswith("leaf")


def test_git_branch_is_empty_outside_a_repository(tmp_path):
    assert git_branch(str(tmp_path)) == ""
    assert git_branch(str(tmp_path / "does-not-exist")) == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_branch_reads_the_current_branch_even_before_the_first_commit(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "feat/x", str(tmp_path)], check=True)
    assert git_branch(str(tmp_path)) == "feat/x"


# ── the main screen ──────────────────────────────────────────────────────────

async def test_main_screen_has_our_chrome_and_no_stock_header_or_footer(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        assert app.screen.query_one(StatusBar) and app.screen.query_one(HintBar)
        assert not list(app.screen.query("Header")) and not list(app.screen.query("Footer"))


async def test_status_bar_shows_where_you_are_and_what_the_agent_may_do(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], mode="build") as (app, pilot):
        pane = app.screen.query_one("ChatPane")
        pane._refresh_meta()
        bar = plain(app.screen.query_one(StatusBar).build())
        assert "motion" in bar and "Build" in bar and "can edit files" in bar

        await pilot.press("tab")  # Build -> Plan
        await pilot.pause()
        bar = plain(app.screen.query_one(StatusBar).build())
        assert "Plan" in bar and "read-only" in bar and "Build" not in bar


async def test_mode_colours_are_primary_for_build_and_secondary_for_plan(tmp_path, monkeypatch):
    """Build (can change files) is primary - gold in Cobalt; Plan (read-only) is secondary - blue.
    Orange (warning) is no longer a mode colour, so it only ever means "needs a look"."""
    async with tui_app(tmp_path, monkeypatch, [text("x")], mode="build") as (app, pilot):
        pane = app.screen.query_one("ChatPane")
        assert pane._agent_color() == "$primary"
        app.state.agent_mode = "plan"
        assert pane._agent_color() == "$secondary"


async def test_hint_bar_lists_only_the_keys_that_matter_in_the_current_state(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        hints = app.screen.query_one(HintBar)
        idle = plain(hints.build())
        assert "Enter" in idle and "send" in idle and "Esc" not in idle

        app.state.busy = True
        app.screen.query_one("ChatPane")._refresh_status()
        busy = plain(hints.build())
        assert "Esc" in busy and "stop" in busy and "Enter" not in busy

        app.state.busy = False
        app.screen.query_one("ChatPane")._refresh_status()
        assert plain(hints.build()) == idle


async def test_status_bar_branch_is_filled_in_after_mount(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q", "-b", "calm-pass", str(tmp_path)], check=True)
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        for _ in range(40):  # the branch lookup runs off the event loop
            await pilot.pause(0.05)
            if "calm-pass" in plain(app.screen.query_one(StatusBar).build()):
                break
        assert "calm-pass" in plain(app.screen.query_one(StatusBar).build())
