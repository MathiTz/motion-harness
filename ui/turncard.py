"""The turn card and the "Done" strip (redesign step 2).

While a turn runs, the chat shows a card of what the agent has done so far; when it ends, the
process collapses to one readable line ("read 3 files · edited cart.py · ran 1 command") and a
strip under the answer says what changed and offers the next actions as clickable chips.

`TurnLog` is plain data fed from the same tool events the trace panel already receives, so it has
no widget or theme dependency and is tested on its own. The widgets below only draw it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

READ_TOOLS = {"read_file", "read_image"}
SEARCH_TOOLS = {"grep", "list_files"}
EDIT_TOOLS = {"write_file", "replace_in_file", "edit_files"}
RUN_TOOLS = {"run_command", "run_script", "run_python"}
WEB_TOOLS = {"web_fetch", "web_search"}


@dataclass
class Call:
    """One tool call, from `tool_start` to `tool_done` / `tool_error`."""
    tool: str
    path: str = ""
    started: float = 0.0
    ended: Optional[float] = None
    ok: Optional[bool] = None   # None while it is still running
    blocked: bool = False       # refused by policy (e.g. a write attempted in read-only Plan mode)
    created: bool = False


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _name(path: str) -> str:
    return PurePath(path).name or path


@dataclass
class TurnLog:
    """What a turn did so far: tool calls (for the summary) and the human lines the tool loop
    streamed about them (for the card)."""
    clock: Callable[[], float] = time.monotonic
    calls: List[Call] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    def start(self, tool: str, path: str = "") -> Call:
        call = Call(tool=tool, path=path, started=self.clock())
        self.calls.append(call)
        return call

    def finish(self, tool: str, path: str = "", *, ok: bool = True, blocked: bool = False, created: bool = False) -> Optional[Call]:
        """Close the best-matching running call: same tool and path, else same tool. Parallel calls
        can finish out of order, so it is not simply the oldest or newest; an event for a tool that
        never started (a malformed call) matches nothing rather than closing someone else's."""
        running = [c for c in self.calls if c.ok is None]
        match = (
            next((c for c in running if c.tool == tool and c.path == path), None)
            or next((c for c in running if c.tool == tool), None)
        )
        if match is None:
            return None
        match.ended, match.ok, match.blocked, match.created = self.clock(), ok and not blocked, blocked, created
        if path and not match.path:
            match.path = path
        return match

    def add_line(self, line: str) -> None:
        if line:
            self.lines.append(line)

    # ── what the card draws ──
    def running(self) -> List[Call]:
        return [c for c in self.calls if c.ok is None]

    def failed(self) -> int:
        return sum(1 for c in self.calls if c.ok is False and not c.blocked)

    def blocked(self) -> int:
        return sum(1 for c in self.calls if c.blocked)

    def has_activity(self) -> bool:
        return bool(self.calls or self.lines)

    # ── the one-line summary ──
    def summary_parts(self) -> List[str]:
        """["read 3 files", "edited cart.py", "ran 2 commands", ...], from successful calls only: a
        refused or failed write is reported as such, never as "edited"."""
        done = [c for c in self.calls if c.ok]
        parts: List[str] = []

        reads = {c.path or f"#{i}" for i, c in enumerate(done) if c.tool in READ_TOOLS}
        if reads:
            parts.append(f"read {plural(len(reads), 'file')}")
        searches = sum(1 for c in done if c.tool in SEARCH_TOOLS)
        if searches:
            parts.append(plural(searches, "search") if searches > 1 else "searched the code")

        def files(created: bool) -> List[Call]:
            return [c for c in done if c.tool in EDIT_TOOLS and c.created == created]

        for verb, group in (("created", files(True)), ("edited", files(False))):
            names = sorted({_name(c.path) for c in group if c.path})
            if names:
                parts.append(f"{verb} {', '.join(names)}" if len(names) <= 2 else f"{verb} {plural(len(names), 'file')}")
            elif group:
                parts.append(f"{verb} files")

        runs = sum(1 for c in done if c.tool in RUN_TOOLS)
        if runs:
            parts.append(f"ran {plural(runs, 'command')}")
        web = sum(1 for c in done if c.tool in WEB_TOOLS)
        if web:
            parts.append(plural(web, "web lookup"))
        subagents = sum(1 for c in done if c.tool == "task")
        if subagents:
            parts.append(plural(subagents, "sub-agent"))
        if self.failed():
            parts.append(f"{self.failed()} failed")
        if self.blocked():
            parts.append(f"{self.blocked()} blocked")
        return parts


def line_role(line: str) -> str:
    """How to draw one streamed tool line: done, failed or informational."""
    stripped = line.lstrip()
    if stripped.startswith("⚠") or " failed" in stripped[:120]:
        return "fail"
    if stripped.startswith(("ℹ", "via ", "↳")):
        return "info"
    return "ok"


def render_card(log: TurnLog, colors: Dict[str, str], rows: int = 6, width: int = 110) -> Text:
    """The live card: the last few finished operations, then what is running right now."""
    text = Text(no_wrap=True, overflow="ellipsis")
    shown = log.lines[-rows:]
    hidden = len(log.lines) - len(shown)
    if hidden > 0:
        text.append(f"  … {hidden} earlier\n", style="dim")
    for line in shown:
        role = line_role(line)
        icon, color = {"ok": ("✓", colors["ok"]), "fail": ("✗", colors["fail"]), "info": ("·", "")}[role]
        text.append(f"{icon} ", style=color or "dim")
        flat = " ".join(line.split())
        text.append((flat[: width - 4] + "…") if len(flat) > width - 3 else flat, style="" if role != "info" else "dim")
        text.append("\n")
    running = log.running()
    if running:
        first = running[0]
        label = RUNNING_LABELS.get(first.tool, first.tool.replace("_", " "))
        extra = f" `{_name(first.path)}`" if first.path else ""
        more = f" (+{len(running) - 1} more)" if len(running) > 1 else ""
        text.append("● ", style=colors["run"])
        text.append(f"{label}{extra}{more}…", style=colors["run"])
    elif text.plain.endswith("\n"):
        text.right_crop(1)
    return text


RUNNING_LABELS = {
    "read_file": "reading", "read_image": "looking at", "list_files": "listing files", "grep": "searching",
    "write_file": "writing", "replace_in_file": "editing", "edit_files": "editing files",
    "run_command": "running a command", "run_script": "running a script", "run_python": "running Python",
    "web_fetch": "fetching", "web_search": "searching the web", "task": "a sub-agent is working",
}


# ── the "Done" strip ──────────────────────────────────────────────────────────

@dataclass
class Chip:
    label: str
    hint: str            # the key or command that does the same thing, so the chip teaches it
    action: Callable[[], None]


class ActionChip(Static):
    """A clickable action. The hint shows the equivalent key or command: click now, type it later."""

    DEFAULT_CSS = """
    ActionChip {
        width: auto;
        height: 1;
        padding: 0 1;
        margin-right: 1;
        background: $panel;
        color: $text;
    }
    ActionChip:hover { background: $boost; }
    """

    def __init__(self, chip: Chip, **kwargs: Any) -> None:
        label = Text(chip.label)
        if chip.hint:
            label.append(f" {chip.hint}", style="dim")
        super().__init__(label, **kwargs)
        self._chip = chip

    def on_click(self) -> None:
        self._chip.action()


class TurnSummary(Vertical):
    """`✓ Done · 2 files changed (+7 −5) · 4.9s · $0.005` and the actions that make sense now.

    The chips act on "the last turn" (undo, diff, copy), so they are retired the moment the next
    turn starts: an old strip must never be able to undo a newer turn's edits."""

    DEFAULT_CSS = """
    TurnSummary {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 2;
        background: $surface;
        border-left: solid $border;
    }
    TurnSummary > #summary_chips { height: 1; }
    """

    def __init__(self, headline: Sequence[Tuple[str, str]], chips: Sequence[Chip], colors: Dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._headline = list(headline)
        self._chips = list(chips)
        self._colors = colors
        self.active = bool(chips)

    def compose(self):
        text = Text()
        for part, role in self._headline:
            text.append(part, style=self._colors.get(role, "") if role != "dim" else "dim")
        yield Static(text, id="summary_text")
        if self._chips:
            with Horizontal(id="summary_chips"):
                for chip in self._chips:
                    yield ActionChip(chip, id="chip_" + "".join(c if c.isalnum() else "_" for c in chip.label.lower()))

    def retire(self) -> None:
        """Keep the record of what happened; drop the actions."""
        if not self.active:
            return
        self.active = False
        try:
            self.query_one("#summary_chips").remove()
        except Exception:
            pass


def headline_parts(log: TurnLog, files_changed: int, added: int, removed: int, elapsed: float,
                   cost: Optional[str]) -> List[Tuple[str, str]]:
    """The strip's first line as (text, role) parts; roles are resolved to theme colours when drawn."""
    failed = log.failed()
    parts: List[Tuple[str, str]] = []
    if failed:
        parts.append((f"⚠ Done with {plural(failed, 'failed tool call')}", "warn"))
    else:
        parts.append(("✓ Done", "ok"))
    if files_changed:
        delta = f" (+{added} −{removed})" if added or removed else ""
        parts.append((f" · {plural(files_changed, 'file')} changed{delta}", "plain"))
    else:
        summary = [p for p in log.summary_parts() if not p.endswith(" failed")]
        if summary:
            parts.append((" · " + " · ".join(summary), "plain"))
    parts.append((f" · {elapsed:.1f}s", "dim"))
    if cost:
        parts.append((f" · {cost}", "dim"))
    return parts
