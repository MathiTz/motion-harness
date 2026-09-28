"""Saved skills: reusable procedures stored as markdown files.

Skills come from three places, project-local first:
  <workspace>/.motion/skills/*.md   (saved with /skill save)
  <workspace>/skills/*.md           (legacy location)
  <harness>/skills/*.md             (auto-synthesized, shared across projects)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from core.skill_state import is_active, load_meta

REPO_DIR = Path(__file__).resolve().parent.parent
MAX_SKILL_CHARS = 20_000


def slugify(name: str) -> str:
    value = re.sub(r"\s+", "_", name.strip().lower())
    return re.sub(r"[^a-z0-9_-]+", "", value)[:60].strip("_-")


class SkillLibrary:
    def __init__(self, dirs: Iterable[Path]) -> None:
        self.dirs = [Path(d) for d in dirs]

    @classmethod
    def for_workspace(cls, workspace: "str | Path") -> "SkillLibrary":
        ws = Path(workspace)
        return cls([ws / ".motion" / "skills", ws / "skills", REPO_DIR / "skills"])

    def _files(self, include_inactive: bool = False) -> dict[str, Path]:
        """Only ACTIVE skills by default (issue #16) - a synthesized skill starts as a candidate
        and must not be surfaced to a live turn until something explicitly promotes it. A skill
        with no metadata sidecar (hand-saved via /skill save, or from before this lifecycle
        existed) has no candidate stage to gate on, so it's treated as active."""
        found: dict[str, Path] = {}
        for d in self.dirs:
            try:
                for p in sorted(d.glob("*.md")):
                    if not include_inactive and not is_active(p):
                        continue
                    found.setdefault(p.stem, p)  # earlier dirs win
            except OSError:
                continue
        return found

    @staticmethod
    def _describe(path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        for line in text.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and not line.lower().startswith(("skill:", "trigger")):
                return line[:100]
        return ""

    def index(self, include_inactive: bool = False) -> List[Tuple[str, str]]:
        return [(name, self._describe(p)) for name, p in self._files(include_inactive).items()]

    def pending(self) -> List[Tuple[str, str, dict]]:
        """Candidates and evaluated-but-not-yet-active skills, for a human to review - active and
        hand-saved skills are never "pending" (nothing to review)."""
        out = []
        for name, p in self._files(include_inactive=True).items():
            meta = load_meta(p)
            if meta is not None and meta.get("status") != "active":
                out.append((name, self._describe(p), meta))
        return out

    def get(self, name: str, include_inactive: bool = False) -> Optional[str]:
        path = self.resolve_path(name, include_inactive)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:MAX_SKILL_CHARS]
        except OSError:
            return None

    def resolve_path(self, name: str, include_inactive: bool = False) -> Optional[Path]:
        """The on-disk .md path for a skill by name, or None - the public way to find it (e.g. to
        promote/reject/roll it back), rather than reaching into `_files()` directly."""
        files = self._files(include_inactive)
        return files.get(name) or files.get(slugify(name))
