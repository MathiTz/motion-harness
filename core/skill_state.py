"""Lifecycle for synthesized skills (issue #16): candidate -> evaluated -> active, with provenance
and rollback.

Before this, `SkillSynthesizer.synthesize()` wrote a skill straight into the active skills
directory and indexed it for recall in the same call - there was no gate between "the synthesizer
produced this" and "a live turn can use it." Every synthesized skill now gets a metadata sidecar
file (`<name>.meta.json`, next to `<name>.md`) recording its status, version history and
provenance. `SkillLibrary` (core/skills.py) only ever surfaces a skill whose status is ACTIVE (or
one with no sidecar at all - a skill saved by hand with `/skill save`, which this issue's scope
does not touch: it was already user-authored and trusted, not synthesizer output).

Storage choice: a JSON sidecar rather than frontmatter inside the `.md` file, so every existing
reader of a skill's content (the system prompt listing, `/skill show`, `use_skill`) keeps reading
plain markdown with no parsing changes; only the gatekeeping (`SkillLibrary._files`) needs to know
sidecars exist at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CANDIDATE = "candidate"
EVALUATED = "evaluated"
ACTIVE = "active"
REJECTED = "rejected"
STATUSES = (CANDIDATE, EVALUATED, ACTIVE, REJECTED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Provenance:
    """Where a skill version came from, so a bad one can be traced back."""
    task_id: str = ""
    prompt: str = ""
    model: str = ""
    verified: bool = False  # True only once a real outcome check (issue #13's eval baseline) confirmed it
    created_at: str = field(default_factory=_now)


def meta_path_for(skill_path: "str | Path") -> Path:
    skill_path = Path(skill_path)
    return skill_path.with_suffix("").with_name(skill_path.stem + ".meta.json")


def load_meta(skill_path: "str | Path") -> Optional[Dict[str, Any]]:
    """None means "no sidecar" - a hand-saved skill, or one from before this existed - which
    SkillLibrary treats as active by default (see its own docstring)."""
    path = meta_path_for(skill_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_meta(skill_path: "str | Path", meta: Dict[str, Any]) -> None:
    meta_path_for(skill_path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def is_active(skill_path: "str | Path") -> bool:
    meta = load_meta(skill_path)
    return meta is None or meta.get("status") == ACTIVE


def write_candidate(skill_path: "str | Path", content: str, provenance: Provenance) -> Dict[str, Any]:
    """Write a new synthesized skill as a fresh CANDIDATE, version 1. If a skill with this name
    already exists (any status), its current content+meta are pushed onto history first, so
    regenerating a skill never silently loses the version that was previously active."""
    skill_path = Path(skill_path)
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    version = 1
    if skill_path.exists():
        prior_meta = load_meta(skill_path) or {"status": ACTIVE, "version": 1, "provenance": {}, "history": []}
        history = list(prior_meta.get("history", []))
        history.append({
            "version": prior_meta.get("version", 1), "status": prior_meta.get("status", ACTIVE),
            "provenance": prior_meta.get("provenance", {}), "content": skill_path.read_text(encoding="utf-8"),
        })
        version = prior_meta.get("version", 1) + 1

    skill_path.write_text(content, encoding="utf-8")
    meta = {
        "status": CANDIDATE, "version": version, "provenance": asdict(provenance),
        "created_at": _now(), "updated_at": _now(), "history": history,
    }
    save_meta(skill_path, meta)
    return meta


def set_status(skill_path: "str | Path", status: str) -> Optional[Dict[str, Any]]:
    """Move a skill to EVALUATED/ACTIVE/REJECTED. Returns None if there's no sidecar to update (a
    hand-saved skill - there's no candidate lifecycle to move it through)."""
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status!r}")
    meta = load_meta(skill_path)
    if meta is None:
        return None
    meta["status"] = status
    meta["updated_at"] = _now()
    save_meta(skill_path, meta)
    return meta


def rollback(skill_path: "str | Path") -> Optional[Dict[str, Any]]:
    """Restore the previous version from history (content + meta), demoting/discarding the current
    one. Returns the restored meta, or None if there is no prior version to roll back to (nothing
    is changed in that case - never silently delete the only version)."""
    skill_path = Path(skill_path)
    meta = load_meta(skill_path)
    if not meta or not meta.get("history"):
        return None
    previous = meta["history"][-1]
    remaining_history = meta["history"][:-1]
    skill_path.write_text(previous["content"], encoding="utf-8")
    restored = {
        "status": previous.get("status", ACTIVE), "version": previous.get("version", 1),
        "provenance": previous.get("provenance", {}), "created_at": meta.get("created_at", _now()),
        "updated_at": _now(), "history": remaining_history,
    }
    save_meta(skill_path, restored)
    return restored


def delete_skill(skill_path: "str | Path") -> None:
    skill_path = Path(skill_path)
    skill_path.unlink(missing_ok=True)
    meta_path_for(skill_path).unlink(missing_ok=True)
