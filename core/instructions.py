"""Project instructions and environment context for the system prompt."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

INSTRUCTION_FILES = ("AGENTS.md", "MOTION.md", "CLAUDE.md")
MAX_INSTRUCTION_CHARS = 12_000
GLOBAL_DIR = Path(os.getenv("MOTION_AUTH_DIR", str(Path.home() / ".config" / "motion-harness")))

_git_cache: Dict[str, Tuple[float, str]] = {}


def load_project_instructions(workspace: "str | Path") -> str:
    """Return the first project instruction file found (AGENTS.md, MOTION.md,
    CLAUDE.md) plus the user's global one, formatted for the system prompt."""
    blocks = []
    for label, base in (("User instructions", GLOBAL_DIR), ("Project instructions", Path(workspace))):
        for fname in INSTRUCTION_FILES:
            path = base / fname
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        blocks.append(f"{label} ({fname}):\n{text[:MAX_INSTRUCTION_CHARS]}")
                    break
            except OSError:
                continue
    return "\n\n".join(blocks)


def _git(workspace: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(workspace), capture_output=True, text=True, timeout=1.5,
            stdin=subprocess.DEVNULL,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_summary(workspace: "str | Path") -> str:
    ws = Path(workspace)
    key = str(ws)
    cached = _git_cache.get(key)
    if cached and time.monotonic() - cached[0] < 30:
        return cached[1]
    branch = _git(ws, "rev-parse", "--abbrev-ref", "HEAD")
    summary = ""
    if branch:
        changed = len([l for l in _git(ws, "status", "--porcelain").splitlines() if l.strip()])
        summary = f"git branch: {branch} ({changed} changed file{'s' if changed != 1 else ''})"
    _git_cache[key] = (time.monotonic(), summary)
    return summary


def build_context_blocks(workspace: "str | Path") -> str:
    """Environment facts + project instructions. Blocking (runs git); call it
    from a worker thread."""
    env_lines = [
        f"Today's date: {date.today().isoformat()}",
        f"Platform: {platform.system()} {platform.machine()}",
        f"Working directory: {Path(workspace).resolve()}",
    ]
    git = git_summary(workspace)
    if git:
        env_lines.append(git)
    parts = ["Environment:\n" + "\n".join(f"- {l}" for l in env_lines)]
    instructions = load_project_instructions(workspace)
    if instructions:
        parts.append(instructions)
    return "\n\n".join(parts)
