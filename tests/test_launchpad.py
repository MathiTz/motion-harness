"""The Launchpad (ui/launchpad.py): the card/chip logic, then the real empty-session screen."""
import shutil
import subprocess
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from core.session import SessionStore
from tests.test_agent_loop import text
from tests.test_tui_flow import _text_of, send, tui_app, wait_idle
from ui.launchpad import EXPLORE_PROMPT, REVIEW_PROMPT, LaunchCard, Launchpad, ago, build_cards, collect, git_changes, state_chips

NOW = datetime(2026, 10, 18, 12, 0)


# ── pure logic ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("modified,expected", [
    ("2026-10-18 11:59", "1m ago"),
    ("2026-10-18 11:00", "1h ago"),
    ("2026-10-18 06:30", "5h ago"),
    ("2026-10-17 11:00", "yesterday"),
    ("2026-10-17 12:30", "23h ago"),
    ("2026-10-10 12:00", "8d ago"),
    ("garbage", "garbage"),
])
def test_ago(modified, expected):
    assert ago(modified, NOW) == expected


def test_ago_for_the_same_minute_is_just_now():
    assert ago("2026-10-18 12:00", NOW) == "just now"


def titles(cards):
    return [c.title for c in cards]


def test_cards_for_a_repo_with_changes_and_a_saved_session():
    sessions = [{"id": "20261018-090000", "turns": 3, "first_prompt": "fix cart rounding", "modified": "2026-10-18 10:00"}]
    one, two, three, four = build_cards(3, sessions, tracking_on=True, now=NOW)
    assert (one.mode, one.prompt, one.enabled) == ("plan", EXPLORE_PROMPT, True)
    assert (two.mode, two.prompt) == ("build", "Fix this: ")
    assert (three.mode, three.prompt, three.enabled) == ("plan", REVIEW_PROMPT, True) and "3 changed files" in three.subtitle
    assert four.kind == "resume" and four.session_id == "20261018-090000" and four.enabled
    assert "fix cart rounding" in four.subtitle and "3 turns" in four.subtitle and "2h ago" in four.subtitle


def test_every_starting_move_that_edits_says_so_and_every_read_only_one_says_so():
    cards = build_cards(1, [], True)
    assert "can edit files" in cards[1].tag and all("read-only" in c.tag for c in (cards[0], cards[2]))


def test_review_card_explains_why_it_is_unavailable():
    not_a_repo = build_cards(None, [], True)[2]
    assert not not_a_repo.enabled and "not a git repository" in not_a_repo.subtitle
    clean = build_cards(0, [], True)[2]
    assert not clean.enabled and "No uncommitted changes" in clean.subtitle
    assert "1 changed file " in build_cards(1, [], True)[2].subtitle  # singular


def test_resume_card_explains_why_it_is_unavailable_and_how_to_fix_it():
    none_yet = build_cards(1, [], tracking_on=True)[3]
    assert not none_yet.enabled and "No saved sessions" in none_yet.subtitle
    off = build_cards(1, [], tracking_on=False)[3]
    assert not off.enabled and "History is off" in off.subtitle and "/tracking on" in off.tag


def test_resume_card_truncates_a_long_prompt():
    s = [{"id": "x", "turns": 1, "first_prompt": "a very long first prompt " * 5, "modified": "2026-10-18 11:59"}]
    sub = build_cards(1, s, True, now=NOW)[3].subtitle
    assert len(sub) < 70 and "1 turn ·" in sub


def test_state_chips_report_live_facts():
    chips = dict(state_chips(128, 2, True, True))
    assert chips["◆ memory · 128 notes in this project"] == "learned"
    assert "◆ skills · 2 awaiting your review" in chips
    assert "● history on · sessions are saved" in chips and "● per-turn budget set" in chips


def test_state_chips_when_nothing_is_set_up_point_at_the_command():
    texts = [t for t, _ in state_chips(0, 0, False, False)]
    assert "◆ memory · nothing saved in this project yet" in texts
    assert "○ history off · /tracking on" in texts and "○ budget · no limit set · /budget" in texts
    assert not any("skills" in t for t in texts)  # only listed when something waits for review


def test_git_changes_is_none_outside_a_repo(tmp_path):
    assert git_changes(str(tmp_path)) is None
    assert git_changes(str(tmp_path / "missing")) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_changes_counts_files_with_uncommitted_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert git_changes(str(tmp_path)) == 0
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "b.py").write_text("2")
    assert git_changes(str(tmp_path)) == 2


def test_collect_survives_every_source_failing(tmp_path):
    class Broken:
        @property
        def memory(self):
            raise RuntimeError("db is gone")

    cards, chips = collect(str(tmp_path), Broken(), tracking_on=False)
    assert len(cards) == 4 and chips  # blank facts, never an exception


def test_collect_reads_memory_and_budget_from_the_agent(tmp_path):
    agent = SimpleNamespace(memory=SimpleNamespace(count=lambda: 7), budget=SimpleNamespace(active=True))
    _, chips = collect(str(tmp_path), agent, tracking_on=True)
    texts = [t for t, _ in chips]
    assert "◆ memory · 7 notes in this project" in texts and "● per-turn budget set" in texts


# ── in the real TUI ──────────────────────────────────────────────────────────

async def wait_cards(app, pilot, n=4, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if len(list(app.screen.query(LaunchCard))) >= n:
            return
    raise AssertionError("the Launchpad cards never appeared")


def composer_value(app) -> str:
    return app.screen.query_one("#chat_input").value


async def test_an_empty_session_opens_on_the_launchpad(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):
        await wait_cards(app, pilot)
        assert len(list(app.screen.query(Launchpad))) == 1
        assert "What are we working on?" in _text_of(app.screen.query_one("#lp_title"))
        assert "memory" in _text_of(app.screen.query_one("#lp_chips"))


async def test_a_card_sets_the_mode_and_fills_the_composer_without_sending(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], mode="build") as (app, pilot):
        await wait_cards(app, pilot)
        await pilot.click("#launch_card_1")
        await pilot.pause(0.2)
        assert composer_value(app) == EXPLORE_PROMPT
        assert app.state.agent_mode == "plan"            # the card is read-only, so the mode followed it
        assert not app.state.busy and app.provider.requests == []  # nothing was sent
        assert not list(app.screen.query(Launchpad))     # the empty state is over


async def test_the_fix_card_switches_to_build_and_waits_for_you_to_finish_the_sentence(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")], mode="plan") as (app, pilot):
        await wait_cards(app, pilot)
        await pilot.click("#launch_card_2")
        await pilot.pause(0.2)
        assert composer_value(app) == "Fix this: " and app.state.agent_mode == "build"
        assert app.screen.query_one("#chat_input").cursor_position == len("Fix this: ")


async def test_a_disabled_card_does_nothing_when_clicked(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("x")]) as (app, pilot):  # tmp_path is not a git repo
        await wait_cards(app, pilot)
        card = app.screen.query_one("#launch_card_3")
        assert not card.card.enabled and card.has_class("-disabled")
        await pilot.click("#launch_card_3")
        await pilot.pause(0.2)
        assert composer_value(app) == "" and list(app.screen.query(Launchpad))


async def test_the_resume_card_reloads_the_last_saved_session(tmp_path, monkeypatch):
    store = SessionStore(tmp_path, "20261018-090000")
    store.append({"prompt": "fix cart rounding", "response": "done"})
    async with tui_app(tmp_path, monkeypatch, [text("x")], track=True) as (app, pilot):
        await wait_cards(app, pilot)
        card = app.screen.query_one("#launch_card_4")
        assert card.card.enabled and "fix cart rounding" in card.card.subtitle
        await pilot.click("#launch_card_4")
        for _ in range(40):
            await pilot.pause(0.05)
            if app.state.conversation_turns:
                break
        assert [p for p, _ in app.state.conversation_turns] == ["fix cart rounding"]
        assert not list(app.screen.query(Launchpad))


async def test_the_first_message_ends_the_empty_state(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hello")]) as (app, pilot):
        await wait_cards(app, pilot)
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        assert not list(app.screen.query(Launchpad))


async def test_a_message_that_starts_with_a_digit_is_just_a_message(tmp_path, monkeypatch):
    """Digits are not card shortcuts: "3 tests are failing" must reach the model as typed."""
    async with tui_app(tmp_path, monkeypatch, [text("ok")]) as (app, pilot):
        await wait_cards(app, pilot)
        await send(app, pilot, "3 tests are failing")
        await wait_idle(app, pilot)
        assert "3 tests are failing" in str(app.provider.requests[0]["messages"][-1]["content"])


async def test_new_conversation_brings_the_launchpad_back(tmp_path, monkeypatch):
    async with tui_app(tmp_path, monkeypatch, [text("hello")]) as (app, pilot):
        await wait_cards(app, pilot)
        await send(app, pilot, "hi")
        await wait_idle(app, pilot)
        await send(app, pilot, "/new")
        await wait_cards(app, pilot)
        assert len(list(app.screen.query(Launchpad))) == 1
