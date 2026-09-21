"""Per-workspace harness state under ``<workspace>/.motion/`` and session
transcripts (JSONL) that can be resumed.

Everything the harness writes into the user's project lives under one
directory (``.motion/``) that ignores itself, instead of scattering
``tasks/``, ``skills/``, ``sessions/`` and logs across the project root.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_DIRNAME = ".motion"


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
        turns: List[Dict[str, Any]] = []
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
                    if isinstance(rec, dict) and "prompt" in rec:
                        turns.append(rec)
        except OSError:
            pass
        return turns

    @staticmethod
    def load(workspace: "str | Path", session_id: str) -> List[Dict[str, Any]]:
        path = Path(workspace) / STATE_DIRNAME / "sessions" / f"{Path(session_id).stem}.jsonl"
        return SessionStore._read(path)
