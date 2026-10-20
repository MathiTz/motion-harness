"""The session rail (redesign step 3): the harness's own state, beside the chat.

It replaces the old Context panel, which mostly repeated the conversation (a summary and the last
turns). The rail shows what the chat does not: the plan, the files touched this session, what
memory was recalled for the current turn (with age and confidence), the per-turn budget against its
limits, and whether the command sandbox is confining anything.

The renderers are plain functions over plain data so they can be tested without a terminal; the
widget below only places them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Label, Static

from memory.retriever import HybridRetriever

# Below this many columns the rail would squeeze the chat, so it hides itself (Ctrl+B still shows it).
RAIL_MIN_WIDTH = 110
MAX_FILES = 8
MAX_RECALLED = 5


@dataclass
class RailData:
    """What the rail shows that nothing else keeps: files touched, recall, sandbox, live budget use."""
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # path -> {added, removed, created}
    recalled: Optional[List[Dict[str, Any]]] = None  # None until this turn's recall has run
    sandbox: Optional[Dict[str, Any]] = None         # {"active": bool, "detail": str}, from the loop's trace
    use: Dict[str, Any] = field(default_factory=dict)  # steps / tokens / cost / seconds of the current or last turn

    def note_edit(self, path: str, added: int = 0, removed: int = 0, created: bool = False) -> None:
        entry = self.files.pop(path, None) or {"added": 0, "removed": 0, "created": False}
        entry["added"] += int(added or 0)
        entry["removed"] += int(removed or 0)
        entry["created"] = entry["created"] or bool(created)
        self.files[path] = entry  # re-inserted last: most recently touched files list first

    def new_session(self) -> None:
        self.files.clear()
        self.recalled = None
        self.use = {}  # the sandbox state is config, not conversation: it stays


def compact(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k" if n < 100_000 else f"{n / 1000:.0f}k"
    return f"{n:.0f}"


def bar(fraction: float, width: int = 10) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "░" * (width - filled)


def _header(text: Text, title: str, style: str = "dim bold") -> None:
    if text.plain:
        text.append("\n")
    text.append(title, style=style)


def plan_text(todos: List[Dict[str, Any]], colors: Dict[str, str]) -> Text:
    text = Text()
    if not todos:
        return text
    _header(text, "PLAN")
    marks = {"completed": ("✓", colors["ok"]), "in_progress": ("●", colors["run"]), "pending": ("○", "")}
    for t in todos[:12]:
        icon, color = marks.get(t.get("status"), ("○", ""))
        text.append("\n")
        text.append(f"{icon} ", style=color or "dim")
        done = t.get("status") == "completed"
        text.append(str(t.get("content", ""))[:60], style="dim" if done else "")
    return text


def files_text(files: Dict[str, Dict[str, Any]], colors: Dict[str, str]) -> Text:
    text = Text()
    if not files:
        return text
    _header(text, f"FILES TOUCHED ({len(files)})")
    items = list(files.items())[::-1]  # newest first
    for path, e in items[:MAX_FILES]:
        text.append("\n")
        text.append(("+ " if e["created"] else "") + (PurePath(path).name or path))
        if e["added"] or e["removed"]:
            text.append("  ")
            text.append(f"+{e['added']}", style=colors["ok"])
            text.append(" ")
            text.append(f"−{e['removed']}", style=colors["fail"])
    if len(items) > MAX_FILES:
        text.append(f"\n… {len(items) - MAX_FILES} more", style="dim")
    return text


def recall_text(items: Optional[List[Dict[str, Any]]], colors: Dict[str, str], width: int = 34) -> Text:
    """What memory handed the model this turn. Age and raw similarity are shown, and a weak match
    is flagged, because a 0.41 guess must not look as authoritative as a 0.82 match (PR 37)."""
    text = Text()
    if items is None:
        return text
    _header(text, "RECALLED", style=f"bold {colors['learned']}")
    if not items:
        text.append("\nnothing this turn", style="dim")
        return text
    for item in items[:MAX_RECALLED]:
        raw = item.get("raw_score")
        weak = raw is not None and raw < HybridRetriever.WEAK_SEMANTIC_SCORE
        kind = f"{raw:.2f}" if raw is not None else "keyword"
        text.append("\n")
        text.append(f"◆ {kind}", style=colors["warn"] if weak else colors["learned"])
        if item.get("age"):
            text.append(f" · {item['age']}", style="dim")
        if weak:
            text.append(" · weak", style=colors["warn"])
        snippet = " ".join(str(item.get("content", "")).split())
        text.append("\n" + (snippet[: 2 * width - 1] + "…" if len(snippet) > 2 * width else snippet), style="dim")
    return text


def budget_text(budget: Any, use: Dict[str, Any], colors: Dict[str, str]) -> Text:
    """Each limit that is set, against what the current (or last) turn has used. With none set it
    says how to set one: a guardrail nobody knows exists protects nobody."""
    text = Text()
    _header(text, "BUDGET")
    rows = []
    if budget is not None and getattr(budget, "active", False):
        spec = (
            ("steps", use.get("steps"), budget.max_steps, lambda v: f"{v:.0f}"),
            ("tokens", use.get("tokens"), budget.max_tokens, compact),
            ("cost", use.get("cost"), budget.max_cost_usd, lambda v: f"${v:.3f}"),
            ("time", use.get("seconds"), budget.max_seconds, lambda v: f"{v:.0f}s"),
        )
        rows = [s for s in spec if s[2] is not None]
    if not rows:
        text.append("\nno limit set · /budget", style="dim")
        return text
    for name, used, limit, fmt in rows:
        text.append("\n")
        text.append(f"{name:<7}", style="dim")
        if used is None:
            text.append(bar(0) + f" –/{fmt(limit)}", style="dim")
            continue
        frac = used / limit if limit else 0
        color = colors["fail"] if frac >= 1 else colors["warn"] if frac >= 0.8 else colors["info"]
        text.append(bar(frac), style=color)
        text.append(f" {fmt(used)}/{fmt(limit)}")
    return text


def safety_text(sandbox: Optional[Dict[str, Any]], colors: Dict[str, str]) -> Text:
    text = Text()
    _header(text, "SAFETY")
    text.append("\n")
    if sandbox is None:
        text.append("sandbox: checked on your first turn", style="dim")
    elif sandbox.get("active"):
        text.append("● ", style=colors["ok"])
        text.append("sandbox on · writes stay in this folder")
    else:
        text.append("○ ", style=colors["warn"])
        text.append("sandbox off · policy checks only", style=colors["warn"])
    return text


# ── the widget ────────────────────────────────────────────────────────────────

class SessionRail(Vertical):
    """Plan, files, recall, budget and safety. Each block is its own Static that hides when it has
    nothing to say, and is repainted on its own: nothing is torn down and re-mounted."""

    DEFAULT_CSS = """
    SessionRail {
        width: 38;
        min-width: 30;
        max-width: 44;
        border-left: solid $border;
        margin-left: 1;
        padding: 1 1;
        background: $panel;
    }
    #context_header {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 1;
        padding: 0 1;
        background: transparent;
        border: blank;
    }
    #context_body {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    .rail_block {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text;
        background: transparent;
    }
    """

    BLOCKS = ("plan", "files", "recall", "budget", "safety", "session")

    def __init__(self, state: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Session", id="context_header")
        with VerticalScroll(id="context_body"):
            for name in self.BLOCKS:
                yield Static("", id=f"rail_{name}", classes="rail_block")

    def on_mount(self) -> None:
        self.refresh_all()
        try:  # new colours when the theme changes
            self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_all())
        except Exception:
            pass

    def colors(self) -> Dict[str, str]:
        from ui.chrome import role_colors

        return role_colors(self.app)

    def _set(self, name: str, text: Text) -> None:
        try:
            block = self.query_one(f"#rail_{name}", Static)
        except Exception:
            return
        block.update(text)
        block.display = bool(text.plain)

    def refresh_all(self) -> None:
        colors, st = self.colors(), self.state
        data: RailData = st.rail
        self.refresh_plan(colors)
        self._set("files", files_text(data.files, colors))
        self._set("recall", recall_text(data.recalled, colors))
        self.refresh_budget(colors)
        self._set("safety", safety_text(data.sandbox, colors))
        summary = Text()
        if getattr(st, "_context_turns", None):
            summary.append(f"session · {st.context_summary()}", style="dim")
        else:
            summary.append("No turns yet — start a conversation.", style="dim")
        self._set("session", summary)

    def refresh_plan(self, colors: Optional[Dict[str, str]] = None) -> None:
        self._set("plan", plan_text(getattr(self.state, "todos", None) or [], colors or self.colors()))

    def refresh_budget(self, colors: Optional[Dict[str, str]] = None) -> None:
        agent = getattr(self.state, "agent", None)
        self._set("budget", budget_text(getattr(agent, "budget", None), self.state.rail.use, colors or self.colors()))

    # kept for the callers that fed the old Context panel
    def update_todos(self, todos: list) -> None:
        self.refresh_plan()

    def refresh_context(self) -> None:
        self.refresh_all()
