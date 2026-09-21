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

    def _files(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for d in self.dirs:
            try:
                for p in sorted(d.glob("*.md")):
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

    def index(self) -> List[Tuple[str, str]]:
        return [(name, self._describe(p)) for name, p in self._files().items()]

    def get(self, name: str) -> Optional[str]:
        files = self._files()
        path = files.get(name) or files.get(slugify(name))
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:MAX_SKILL_CHARS]
        except OSError:
            return None
