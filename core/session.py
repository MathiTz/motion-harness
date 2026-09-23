"""Per-workspace harness state under ``<workspace>/.motion/`` and session
transcripts (JSONL) that can be resumed.

Everything the harness writes into the user's project lives under one
directory (``.motion/``) that ignores itself, instead of scattering
``tasks/``, ``skills/``, ``sessions/`` and logs across the project root.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_DIRNAME = ".motion"

INTERRUPTED_MARKER = (
    "[turn interrupted — the harness process ended before this turn finished; any side "
    "effects a tool call made before that point are not recorded here]"
)


def state_dir(workspace: "str | Path", *parts: str, create: bool = True) -> Path:
    """``<workspace>/.motion[/parts...]``; created on demand with a
    self-ignoring ``.gitignore`` so it never ends up in the user's commits."""
    base = Path(workspace) / STATE_DIRNAME
    path = base.joinpath(*parts) if parts else base
    if create:
        path.mkdir(parents=True, exist_ok=True)
        gi = base / ".gitignore"
        if not gi.exists():
            try:
                gi.write_text("*\n", encoding="utf-8")
            except OSError:
                pass
    return path


class SessionStore:
    """Append-only JSONL transcript for one session."""

    def __init__(self, workspace: "str | Path", session_id: Optional[str] = None) -> None:
        self.workspace = Path(workspace)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._path: Optional[Path] = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = state_dir(self.workspace, "sessions") / f"{self.session_id}.jsonl"
        return self._path

    def append(self, record: Dict[str, Any]) -> None:
        record = {"timestamp": datetime.now().isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start_turn(self, prompt: str) -> str:
        """Record that a turn began, before any tool call runs, and return a ``turn_id`` to pass
        to ``append`` (as ``{"type": "turn_end", "turn_id": ...}``) once it finishes - however it
        finishes (a normal answer, a cancellation, or a caught error all get their own turn_end).

        If the process is killed uncatchably (crash, OOM, `kill -9`) between this call and the
        matching turn_end, no code runs to write that turn_end at all - this start record is the
        only trace left. `_read` turns an unmatched one into a visible "interrupted" placeholder
        on `/resume` instead of the turn silently vanishing from the transcript, which is what
        happened before this existed."""
        turn_id = uuid.uuid4().hex
        self.append({"type": "turn_start", "turn_id": turn_id, "prompt": prompt})
        return turn_id

    @staticmethod
    def list_sessions(workspace: "str | Path", limit: int = 20) -> List[Dict[str, Any]]:
        directory = Path(workspace) / STATE_DIRNAME / "sessions"
        if not directory.is_dir():
            return []
        out: List[Dict[str, Any]] = []
        for path in sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            turns = SessionStore._read(path)
            out.append({
                "id": path.stem,
                "turns": len(turns),
                "first_prompt": (turns[0].get("prompt", "") if turns else "")[:80],
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        return out

    @staticmethod
    def _read(path: Path) -> List[Dict[str, Any]]:
        """Reconciles turn_start/turn_end pairs (see `start_turn`) into one entry per turn, in the
        order they happened. A turn_start with no matching turn_end - the process died before the
        turn could finish - becomes a placeholder with `interrupted: True` and no response, at the
        position its turn_start occupied, rather than being silently omitted. Records from before
        this reconciliation existed (plain {"prompt", "response", ...}, no "type") are treated as
        already-complete turns, so old transcripts keep working unchanged."""
        turns: List[Dict[str, Any]] = []
        pending_index: Dict[str, int] = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    rtype = rec.get("type")
                    if rtype == "turn_start":
                        turn_id = rec.get("turn_id")
                        if not turn_id:
                            continue
                        pending_index[turn_id] = len(turns)
                        turns.append({
                            "timestamp": rec.get("timestamp", ""),
                            "prompt": rec.get("prompt", ""),
                            "response": INTERRUPTED_MARKER,
                            "interrupted": True,
                        })
                    elif rtype == "turn_end":
                        clean = {k: v for k, v in rec.items() if k not in ("type", "turn_id")}
                        idx = pending_index.pop(rec.get("turn_id"), None)
                        if idx is not None:
                            turns[idx] = clean
                        else:
                            turns.append(clean)
                    elif "prompt" in rec:
                        turns.append(rec)
        except OSError:
            pass
        return turns

    @staticmethod
    def load(workspace: "str | Path", session_id: str) -> List[Dict[str, Any]]:
        path = Path(workspace) / STATE_DIRNAME / "sessions" / f"{Path(session_id).stem}.jsonl"
        return SessionStore._read(path)
