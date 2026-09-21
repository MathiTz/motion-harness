"""Per-session tool state shared across turns: approvals, undo checkpoints,
files the model has read, and its todo list."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.jobs import JobManager

# Don't snapshot huge files for undo (they'd bloat memory); the edit is still
# allowed, it just isn't restorable.
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
MAX_CHECKPOINTS = 300


@dataclass
class Checkpoint:
    turn: int
    path: Path
    previous: Optional[bytes]  # None => the file did not exist before
    restorable: bool
    label: str
    ts: float = field(default_factory=time.time)


class CheckpointStore:
    """Records file states before the agent modifies them so a whole turn can
    be rolled back with ``undo_last_turn()`` (the ``/undo`` command)."""

    def __init__(self) -> None:
        self.entries: List[Checkpoint] = []
        self.turn = 0

    def begin_turn(self) -> None:
        self.turn += 1

    def record(self, path: Path, label: str = "") -> None:
        # One snapshot per file per turn is enough to restore the pre-turn state.
        if any(e.turn == self.turn and e.path == path for e in self.entries):
            return
        previous: Optional[bytes] = None
        restorable = True
        try:
            if path.exists():
                if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                    restorable = False
                else:
                    previous = path.read_bytes()
        except OSError:
            restorable = False
        self.entries.append(Checkpoint(self.turn, path, previous, restorable, label))
        if len(self.entries) > MAX_CHECKPOINTS:
            del self.entries[: len(self.entries) - MAX_CHECKPOINTS]

    def __len__(self) -> int:
        return len(self.entries)

    def last_turn_entries(self) -> List[Checkpoint]:
        if not self.entries:
            return []
        turn = self.entries[-1].turn
        return [e for e in self.entries if e.turn == turn]

    def undo_last_turn(self) -> List[str]:
        """Restore every file changed in the most recent turn that changed any.
        Returns human-readable lines describing what was done."""
        batch = self.last_turn_entries()
        if not batch:
            return []
        lines: List[str] = []
        for entry in reversed(batch):
            try:
                if not entry.restorable:
                    lines.append(f"skipped {entry.path} (too large to snapshot)")
                elif entry.previous is None:
                    if entry.path.exists():
                        entry.path.unlink()
                    lines.append(f"removed {entry.path} (was created)")
                else:
                    entry.path.parent.mkdir(parents=True, exist_ok=True)
                    entry.path.write_bytes(entry.previous)
                    lines.append(f"restored {entry.path}")
            except OSError as exc:
                lines.append(f"could not restore {entry.path}: {exc}")
        self.entries = [e for e in self.entries if e not in batch]
        return lines


@dataclass
class ToolSession:
    """State that must outlive a single ``MotionAgent.run()`` call."""

    allowed_paths: Set[Path] = field(default_factory=set)
    checkpoints: CheckpointStore = field(default_factory=CheckpointStore)
    read_files: Set[Path] = field(default_factory=set)
    approved_commands: Set[str] = field(default_factory=set)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    # Background processes started with job_start; stopped when the session ends.
    jobs: JobManager = field(default_factory=JobManager)
