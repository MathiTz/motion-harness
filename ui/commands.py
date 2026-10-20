"""The command palette's catalogue (redesign step 5): every action grouped by what you want to do.

The palette used to be one flat list of 15 labels. Grouping by intent (work, review, memory and
skills, session, setup) makes it browsable by someone who doesn't know the command names, and every
row prints the key or slash command that does the same thing, so the palette teaches the shortcuts
instead of replacing them.

An `action` is either an id the palette already knew ("agent", "trace", "model", ...) or
`slash:<command>`, which runs that slash command exactly as if it had been typed (without touching
the draft in the composer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

GROUPS = ("WORK", "REVIEW", "MEMORY AND SKILLS", "SESSION", "SETUP")


@dataclass(frozen=True)
class Command:
    group: str
    label: str
    action: str
    hint: str = ""       # the key or slash command that does the same thing
    note: str = ""       # live state: "2 waiting", "on"


def catalog(skills_waiting: int = 0, tracking_on: bool = False) -> List[Command]:
    c = Command
    return [
        c("WORK", "New conversation", "slash:/new", "/new"),
        c("WORK", "Toggle agent (build/plan)", "agent", "Tab"),
        c("WORK", "Undo the last turn's edits", "slash:/undo", "/undo"),
        c("WORK", "Compact the conversation", "slash:/compact", "/compact"),
        c("WORK", "Show the agent's task list", "slash:/todos", "/todos"),
        c("REVIEW", "View the last diff", "slash:/diff", "/diff"),
        c("REVIEW", "Toggle trace panel", "trace", "F8"),
        c("REVIEW", "Show turn trajectory (/trajectory)", "trajectory", "/trajectory"),
        c("REVIEW", "Toggle agent thinking", "thinking", "F7"),
        c("REVIEW", "Copy last response", "copy", "F9"),
        c("REVIEW", "Copy last code block", "copy_code", "Ctrl+Shift+K"),
        c("REVIEW", "Copy trace log", "copy_trace", "F10"),
        c("REVIEW", "Copy turn trajectory", "copy_trajectory"),
        c("MEMORY AND SKILLS", "Review skill candidates", "slash:/skill candidates", "/skill candidates",
          f"{skills_waiting} waiting" if skills_waiting else "none waiting"),
        c("MEMORY AND SKILLS", "List saved skills", "slash:/skill list", "/skill list"),
        c("SESSION", "Resume a saved session", "slash:/resume", "/resume"),
        c("SESSION", "Toggle interaction tracking", "tracking", "/tracking", "on" if tracking_on else "off"),
        c("SESSION", "Show background jobs", "slash:/jobs", "/jobs"),
        c("SESSION", "Show connected MCP servers", "slash:/mcp", "/mcp"),
        c("SETUP", "Switch model…", "model", "Ctrl+O"),
        c("SETUP", "Set a budget for each turn", "slash:/budget", "/budget"),
        c("SETUP", "Theme menu", "theme", "Ctrl+T"),
        c("SETUP", "Manage API keys (/auth)", "auth", "Ctrl+A"),
        c("SETUP", "Toggle session rail", "context", "Ctrl+B"),
        c("SETUP", "Show shortcuts", "shortcuts", "?"),
        c("SETUP", "Close menu (Esc)", "close", "Esc"),
    ]
