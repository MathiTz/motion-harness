"""OS-level write confinement for commands the agent runs.

The command *policy* (core/permissions.py) is a UX layer built on patterns and
can always be sidestepped (`python -c "shutil.rmtree(...)"`, a script, an
unusual spelling). This module is the enforcement layer: shell, script and
Python execution run inside an OS sandbox where

  * writes are only possible inside the workspace, paths the user approved,
    temp directories and common tool caches (pip/npm/cargo/...);
  * the harness's own secrets (auth store, its .env) cannot be read.

Reads, network and process spawning are left alone: confining those breaks
ordinary development workflows (git over ssh, package installs, ...).

Backends: macOS Seatbelt (`sandbox-exec`) and Linux bubblewrap (`bwrap`). Each
is probed once by actually running it, so environments where it can't work
(nested sandboxes, unprivileged user namespaces disabled) fall back to no
sandbox instead of failing every command. Windows has no backend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

HOME = Path.home()

# Tool caches that builds and package managers legitimately write to.
_HOME_CACHE_DIRS = (
    ".cache", "Library/Caches", ".npm", ".cargo", ".rustup", ".gradle", ".m2", ".local/share",
)
_TEMP_DIRS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/private/var/folders", "/dev/shm")

BLOCKED_HINT = (
    "If this failed because the write sandbox blocked a write: shell/Python commands can only write inside the "
    "workspace, user-approved paths, and temp/cache directories. To change a file elsewhere use "
    "write_file (it asks the user for approval) or ask the user."
)


def _sbpl(path: Union[str, Path]) -> str:
    """Quote a path for a Seatbelt profile string literal."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def seatbelt_profile(writable: Iterable[Path], deny_read: Iterable[Path]) -> str:
    """Build a Seatbelt profile: everything allowed, all writes denied, then
    writes re-allowed for ``writable``; ``deny_read`` paths are unreadable.
    (Later rules win in SBPL.)"""
    subpaths = " ".join(f'(subpath "{_sbpl(p)}")' for p in writable)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f'(allow file-write* {subpaths} (literal "/dev/null") (literal "/dev/tty") '
        f'(regex #"^/dev/(fd/|ttys)"))',
    ]
    for p in deny_read:
        kind = "subpath" if p.is_dir() else "literal"
        lines.append(f'(deny file-read* ({kind} "{_sbpl(p)}"))')
    return "\n".join(lines)


def bwrap_argv(writable: Iterable[Path], deny_read: Iterable[Path]) -> List[str]:
    """bubblewrap prefix: read-only root, writable binds, secrets masked."""
    argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    for p in writable:
        if p.exists():
            argv += ["--bind", str(p), str(p)]
    for p in deny_read:
        if p.is_dir():
            argv += ["--tmpfs", str(p)]
        elif p.exists():
            argv += ["--ro-bind", "/dev/null", str(p)]
    return argv + ["--die-with-parent", "--"]


@lru_cache(maxsize=1)
def _detect_backend() -> Optional[str]:
    """Return 'seatbelt' / 'bwrap' if that sandbox really works here, else None."""
    try:
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            probe = subprocess.run(
                ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
                capture_output=True, timeout=5,
            )
            return "seatbelt" if probe.returncode == 0 else None
        if sys.platform.startswith("linux") and shutil.which("bwrap"):
            probe = subprocess.run(
                ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "true"],
                capture_output=True, timeout=5,
            )
            return "bwrap" if probe.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


class Sandbox:
    def __init__(
        self,
        workspace: Union[str, Path],
        mode: str = "auto",
        protected: Sequence[Path] = (),
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.mode = (mode or "auto").lower()
        self.protected = [Path(p) for p in protected]

    @property
    def backend(self) -> Optional[str]:
        if self.mode in ("off", "false", "no", "disabled"):
            return None
        return _detect_backend()

    @property
    def active(self) -> bool:
        return self.backend is not None

    def describe(self) -> str:
        if self.mode in ("off", "false", "no", "disabled"):
            return "off (disabled in config)"
        backend = self.backend
        if backend is None:
            return "unavailable on this system: commands are NOT write-confined (policy checks only)"
        return f"{'Seatbelt' if backend == 'seatbelt' else 'bubblewrap'}: writes limited to workspace, approved paths, temp/cache"

    def writable_paths(self, extra: Iterable[Path] = ()) -> List[Path]:
        paths: List[Path] = [self.workspace]
        paths.extend(Path(p) for p in extra)  # user-approved out-of-workspace paths
        for t in _TEMP_DIRS:
            paths.append(Path(t))
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            paths.append(Path(tmpdir))
        for rel in _HOME_CACHE_DIRS:
            paths.append(HOME / rel)
        # Both the given and the symlink-resolved form (/tmp -> /private/tmp).
        seen: List[Path] = []
        for p in paths:
            for candidate in (p, p.resolve()):
                if candidate not in seen:
                    seen.append(candidate)
        return seen

    def wrap(
        self,
        command: Union[str, Sequence[str]],
        *,
        shell: bool,
        extra_writable: Iterable[Path] = (),
    ) -> List[str]:
        """Return the argv that runs ``command`` inside the sandbox."""
        inner: List[str] = ["/bin/sh", "-c", str(command)] if shell else [str(c) for c in command]
        writable = self.writable_paths(extra_writable)
        protected = [p for p in self.protected if p.exists()]
        backend = self.backend
        if backend == "seatbelt":
            return ["sandbox-exec", "-p", seatbelt_profile(writable, protected), *inner]
        if backend == "bwrap":
            return [*bwrap_argv(writable, protected), *inner]
        return inner


def default_protected_paths() -> List[Path]:
    """Files/dirs that hold the harness's own secrets."""
    from core import auth

    repo = Path(__file__).resolve().parent.parent
    return [auth.AUTH_DIR, repo / ".env"]
