"""The Launchpad (redesign step 4): what an empty session shows instead of one line and a tip.

Four starting moves built from things the harness really has (a read-only exploration, a fix, a
review of your uncommitted diff, resuming the last saved session), and a row of live facts (how
much memory this project has, skills waiting for review, whether history is on, whether a budget is
set). It appears only on an empty session and is gone at the first message.

A card never sends anything: it sets the mode and puts a prompt in the composer for you to edit.
Cards are clickable; digits are deliberately not shortcuts, because a first message that starts
with "3 tests are failing" must not be eaten as "pick card 3".
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rich.text import Text
from textual.containers import Grid, Vertical
from textual.message import Message
from textual.widgets import Static

EXPLORE_PROMPT = (
    "Explore this repo and summarize how it's organized: the main modules, the entry points, "
    "and how to run the tests."
)
from ui.turncard import REVIEW_PROMPT  # noqa: E402  (one wording for both "review my changes" entry points)


@dataclass
class Card:
    number: int
    title: str
    subtitle: str
    tag: str
    tag_role: str            # a key of ui.chrome.role_colors: info | run | learned | warn
    kind: str = "prompt"     # prompt | resume
    prompt: str = ""
    mode: Optional[str] = None   # "plan" | "build": the mode the card switches to
    session_id: str = ""
    enabled: bool = True


def git_changes(workspace: str, timeout: float = 1.5) -> Optional[int]:
    """How many files have uncommitted changes; None when this is not a git repository (or git is
    missing or slow). Read-only, and never raises."""
    try:
        done = subprocess.run(
            ["git", "-C", workspace, "status", "--porcelain"], capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return sum(1 for line in done.stdout.splitlines() if line.strip())


def ago(modified: str, now: Optional[datetime] = None) -> str:
    """`"2026-10-18 20:11"` -> `"2h ago"`. Unparseable input comes back unchanged."""
    try:
        then = datetime.strptime(modified, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return modified or ""
    seconds = ((now or datetime.now()) - then).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 172800:
        return "yesterday"
    return f"{int(seconds // 86400)}d ago"


def build_cards(changes: Optional[int], sessions: List[Dict[str, Any]], tracking_on: bool,
                now: Optional[datetime] = None) -> List[Card]:
    cards = [
        Card(1, "Explore this repo", "Learn the layout and where things live.",
             "Plan · read-only", "info", prompt=EXPLORE_PROMPT, mode="plan"),
        Card(2, "Fix something", "Describe a bug or paste a failing test.",
             "Build · can edit files", "run", prompt="Fix this: ", mode="build"),
    ]
    if changes is None:
        cards.append(Card(3, "Review my changes", "This folder is not a git repository.", "nothing to diff", "warn",
                          enabled=False))
    elif changes == 0:
        cards.append(Card(3, "Review my changes", "No uncommitted changes to review.", "git working tree is clean",
                          "warn", enabled=False))
    else:
        cards.append(Card(3, "Review my changes", f"Check your {changes} changed file{'s' if changes != 1 else ''} for problems.", "Plan · read-only", "info", prompt=REVIEW_PROMPT, mode="plan"))
    if sessions:
        last = sessions[0]
        prompt = " ".join(str(last.get("first_prompt", "")).split()) or "(no prompt)"
        what = prompt if len(prompt) <= 18 else prompt[:17] + "…"  # short enough that the card stays 3 lines
        turns = int(last.get("turns", 0))
        cards.append(Card(4, "Pick up where I left off",
                          f"“{what}” · {turns} turn{'s' if turns != 1 else ''} · {ago(last.get('modified', ''), now)}",
                          "Resumes the saved session", "learned", kind="resume", session_id=str(last.get("id", ""))))
    elif tracking_on:
        cards.append(Card(4, "Pick up where I left off", "No saved sessions yet.", "they appear after your first turn",
                          "warn", kind="resume", enabled=False))
    else:
        cards.append(Card(4, "Pick up where I left off", "History is off, so nothing is saved.",
                          "/tracking on turns it on", "warn", kind="resume", enabled=False))
    return cards


def state_chips(memory_count: Optional[int], pending_skills: int, budget_active: bool,
                tracking_on: bool) -> List[Tuple[str, str]]:
    """Live facts as (text, role). Skills are listed only when something waits for review."""
    chips: List[Tuple[str, str]] = []
    # Short on purpose: the row must fit on one line at the Launchpad's width, because a chip that
    # wraps mid-phrase reads as two facts.
    if memory_count:
        chips.append((f"◆ memory · {memory_count} note{'s' if memory_count != 1 else ''}", "learned"))
    else:
        chips.append(("◆ memory · empty", "dim"))
    if pending_skills:
        chips.append((f"◆ {pending_skills} skill{'s' if pending_skills != 1 else ''} to review", "learned"))
    chips.append(("● history on", "ok") if tracking_on else ("○ history off · /tracking on", "dim"))
    chips.append(("● budget set", "ok") if budget_active else ("○ no budget · /budget", "dim"))
    return chips


def collect(workspace: str, agent: Any, tracking_on: bool) -> Tuple[List[Card], List[Tuple[str, str]]]:
    """Gather the facts the cards and chips are built from. Every source is optional: a missing
    agent, an unreadable memory DB or a non-repo folder just makes that fact blank, never an error."""
    from core.session import SessionStore
    from core.skills import SkillLibrary

    sessions: List[Dict[str, Any]] = []
    try:
        sessions = SessionStore.list_sessions(workspace, limit=1)
    except Exception:
        pass
    memory_count: Optional[int] = None
    try:
        memory_count = agent.memory.count()
    except Exception:
        pass
    pending = 0
    try:
        pending = len(SkillLibrary.for_workspace(workspace).pending())
    except Exception:
        pass
    budget = getattr(agent, "budget", None)
    cards = build_cards(git_changes(workspace), sessions, tracking_on)
    return cards, state_chips(memory_count, pending, bool(getattr(budget, "active", False)), tracking_on)


# ── widgets ───────────────────────────────────────────────────────────────────

class LaunchCard(Static, can_focus=True):
    """One starting move. Clicking (or Enter when focused) posts `Chosen`; a disabled card says why."""

    DEFAULT_CSS = """
    LaunchCard {
        height: auto;
        padding: 0 2;
        background: $background;
        border: round $border;
    }
    LaunchCard:hover, LaunchCard:focus { border: round $primary; background: $surface; }
    LaunchCard.-disabled:hover { border: round $border; background: $background; }
    """

    class Chosen(Message):
        def __init__(self, card: Card) -> None:
            self.card = card
            super().__init__()

    def __init__(self, card: Card, colors: Dict[str, str], **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.card = card
        text = Text()
        text.append(card.title, style="dim bold" if not card.enabled else "bold")
        text.append("\n" + card.subtitle, style="dim")
        text.append("\n" + card.tag, style=colors.get(card.tag_role, "") or "dim")
        self.update(text)
        self.set_class(not card.enabled, "-disabled")

    def _choose(self) -> None:
        if self.card.enabled:
            self.post_message(self.Chosen(self.card))

    def on_click(self) -> None:
        self._choose()

    def key_enter(self) -> None:
        self._choose()


class Launchpad(Vertical):
    DEFAULT_CSS = """
    Launchpad {
        height: auto;
        width: 100%;
        margin: 2 0 0 0;
        align-horizontal: center;
    }
    Launchpad #lp_title, Launchpad #lp_sub, Launchpad #lp_cards, Launchpad #lp_chips { width: 96; max-width: 100%; }
    Launchpad #lp_title { text-style: bold; text-align: center; margin-bottom: 0; }
    Launchpad #lp_sub { color: $text-muted; text-align: center; margin-bottom: 1; }
    Launchpad #lp_cards { layout: grid; grid-size: 2; grid-gutter: 0 1; height: auto; }
    Launchpad #lp_chips { color: $text-muted; margin-top: 1; height: auto; }
    """

    def __init__(self, state: Any, workspace: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._workspace = workspace

    def compose(self):
        yield Static("What are we working on?", id="lp_title")
        yield Static("Pick a starting move, or just type below.", id="lp_sub")
        yield Grid(id="lp_cards")
        yield Static("", id="lp_chips")

    def on_mount(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        import asyncio

        tracking = bool(self._state.config_manager.get("track_interactions"))
        cards, chips = await asyncio.to_thread(collect, self._workspace, self._state.agent, tracking)
        from ui.chrome import role_colors

        colors = role_colors(self.app)
        grid = self.query_one("#lp_cards", Grid)
        for card in cards:
            await grid.mount(LaunchCard(card, colors, id=f"launch_card_{card.number}"))
        text = Text()
        for i, (chip, role) in enumerate(chips):
            if i:
                text.append("   ")
            text.append(chip, style=colors.get(role, "") if role != "dim" else "dim")
        self.query_one("#lp_chips", Static).update(text)
