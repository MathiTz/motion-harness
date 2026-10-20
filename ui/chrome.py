"""Window chrome for the main screen: a status bar on top and a hint bar at the bottom.

These replace Textual's stock Header (a clock) and Footer (the same key list in every state). The
top bar says where you are and what the agent is allowed to do; the bottom bar lists only the keys
that matter right now. They live here, not in ui/tui.py, so that file does not keep growing.

Colour meaning (see ui/themes.py): primary = you / Build mode, which can change files; the blue
text tint = the agent / Plan mode, which is read-only.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

# mode -> (label, what it means). Shown in the status bar so the mode is never a guess.
MODES = {
    "build": ("Build", "can edit files"),
    "plan": ("Plan", "read-only"),
}

IDLE_HINTS: List[Tuple[str, str]] = [
    ("Enter", "send"), ("/", "commands"), ("Tab", "plan↔build"), ("Ctrl+K", "everything"), ("?", "keys"),
]
BUSY_HINTS: List[Tuple[str, str]] = [("Esc", "stop"), ("F8", "trace"), ("Ctrl+B", "panel")]


def short_path(path: str, home: Optional[str] = None, limit: int = 40) -> str:
    """`/Users/me/proj/shop` -> `~/proj/shop`; a long path keeps its tail, behind an ellipsis."""
    home = home if home is not None else str(Path.home())
    text = str(path)
    if home and (text == home or text.startswith(home.rstrip(os.sep) + os.sep)):
        text = "~" + text[len(home.rstrip(os.sep)):]
    if len(text) > limit:
        text = "…" + text[-(limit - 1):]
    return text


def git_branch(workspace: str, timeout: float = 1.0) -> str:
    """The current branch, `@<sha>` for a detached HEAD, or "" when this is not a git repository
    (or git is missing or slow). Read-only, and never raises: the status bar must not be able to
    break the screen."""
    def run(*args: str) -> str:
        done = subprocess.run(["git", "-C", workspace, *args], capture_output=True, text=True, timeout=timeout)
        return done.stdout.strip() if done.returncode == 0 else ""

    try:
        return run("symbolic-ref", "--short", "-q", "HEAD") or (f"@{sha}" if (sha := run("rev-parse", "--short", "HEAD")) else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _hex(app: Any, which: str, fallback: str) -> str:
    """A theme colour as a hex string Rich can use (`primary`, `secondary`, `blue-text`, ...)."""
    try:
        if which in ("blue-text", "error-text"):
            return app.get_css_variables().get(which) or fallback
        return getattr(app.get_theme(app.theme), which, None) or fallback
    except Exception:
        return fallback


def role_colors(app: Any) -> dict:
    """Theme colours by meaning, as hex strings for Rich: ok (success), fail (readable error tint),
    warn, run (what is happening now: the primary colour), learned (accent: memory, skills) and info
    (the readable blue tint)."""
    return {
        "ok": _hex(app, "success", "#7fd88f"), "fail": _hex(app, "error-text", "#e06c75"),
        "warn": _hex(app, "warning", "#f5a742"), "run": _hex(app, "primary", "#fab283"), "plain": "",
        "learned": _hex(app, "accent", "#9d7cd8"), "info": _hex(app, "blue-text", "#5c9cf5"),
    }


class StatusBar(Static):
    """`motion · ~/path · branch` on the left, the agent mode and what it allows on the right."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, state: Any, workspace: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._state = state
        self._workspace = workspace
        self._branch = ""

    def on_mount(self) -> None:
        self.refresh_bar()
        try:  # repaint with the new colours when the theme changes
            self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_bar())
        except Exception:
            pass
        self.run_worker(self._load_branch(), exclusive=True)

    async def _load_branch(self) -> None:
        self._branch = await asyncio.to_thread(git_branch, self._workspace)
        self.refresh_bar()

    def refresh_bar(self) -> None:
        self.update(self.build())

    def build(self) -> Table:
        primary = _hex(self.app, "primary", "#fab283")
        mode = getattr(self._state, "agent_mode", "build")
        label, meaning = MODES.get(mode, MODES["build"])
        mode_color = primary if mode == "build" else _hex(self.app, "blue-text", "#5c9cf5")

        left = Text()  # no base style: it would tint every appended segment
        left.append("motion", style=f"bold {primary}")
        left.append(f" · {short_path(self._workspace)}", style="dim")
        if self._branch:
            left.append(f" · {self._branch}", style="dim")
        right = Text("● ", style=mode_color)
        right.append(label, style=f"bold {mode_color}")
        right.append(f" · {meaning}", style="dim")

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(justify="right", no_wrap=True)
        grid.add_row(left, right)
        return grid


class HintBar(Static):
    """Only the keys that matter in the current state: idle, or a turn running."""

    DEFAULT_CSS = """
    HintBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._busy = False

    def on_mount(self) -> None:
        self.refresh_bar()
        try:
            self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_bar())
        except Exception:
            pass

    def set_busy(self, busy: bool) -> None:
        if busy != self._busy:
            self._busy = busy
            self.refresh_bar()

    def refresh_bar(self) -> None:
        self.update(self.build())

    def build(self) -> Text:
        primary = _hex(self.app, "primary", "#fab283")
        text = Text()
        for i, (key, label) in enumerate(BUSY_HINTS if self._busy else IDLE_HINTS):
            if i:
                text.append("   ")
            text.append(key, style=f"bold {primary}")
            text.append(f" {label}", style="dim")
        return text
