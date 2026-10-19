"""Plain-language explanations for the approval dialog (redesign step 5).

When the agent wants to run a risky command, "matches rm -r/-f" is not much to decide on. These
functions turn the command into what it does, where it acts, why the harness is asking, and what
undo can and cannot do. Display only: nothing here runs, resolves symlinks or touches the disk.

The templates cover the risky patterns core/permissions.py already knows. A command none of them
recognise gets the policy's own reason instead of a guess, because a wrong explanation of a risky
command is worse than a blunt one.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from core.permissions import _variants, split_segments

NO_UNDO = "Undo can't restore this: /undo only covers file edits, not commands."

# (label, text, role): role is one of plain | ok | warn | fail | dim
Row = Tuple[str, str, str]


@dataclass
class Explanation:
    rows: List[Row]


def _tokens(segment: str) -> List[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _targets(args: List[str]) -> List[str]:
    """The non-option arguments of a command: its file operands."""
    return [a for a in args if not a.startswith("-") and a not in ("&&", "||", ";", "|")]


def _show(paths: List[str], limit: int = 3) -> str:
    shown = ", ".join(f"`{p}`" for p in paths[:limit])
    return shown + (f" and {len(paths) - limit} more" if len(paths) > limit else "")


def where(paths: List[str], workspace: str, home: Optional[str] = None) -> Optional[Row]:
    """Whether each path is inside the project. `~`, `$HOME` and `..` are resolved lexically, never
    against the disk, so this cannot be fooled into a wrong "inside" by a path that doesn't exist."""
    if not paths:
        return None
    home = home if home is not None else os.path.expanduser("~")
    root = os.path.normpath(workspace)
    outside = []
    for raw in paths:
        p = raw.replace("$HOME", home).replace("${HOME}", home)
        if p == "~" or p.startswith("~/"):
            p = home + p[1:]
        full = os.path.normpath(p if os.path.isabs(p) else os.path.join(root, p))
        if full != root and not full.startswith(root + os.sep):
            outside.append(raw)
    if outside:
        return ("Where", f"outside this project: {_show(outside)}", "warn")
    return ("Where", "inside this project", "ok")


def _rm(args: List[str], _rest: str) -> Tuple[str, List[str]]:
    flags = "".join(a for a in args if a.startswith("-"))
    targets = _targets(args)
    recursive = "r" in flags.lower()
    what = f"deletes {_show(targets) or 'files'}"
    if recursive:
        what += " and everything inside"
    return what, targets


def _git(args: List[str], _rest: str) -> Tuple[str, List[str]]:
    words = [a for a in args if not a.startswith("-")]
    sub = words[0] if words else ""
    forced = any(a in ("-f", "--force") or a.startswith("--force") for a in args)
    if sub == "push":
        return ("overwrites history on the remote (force push)" if forced else "publishes your commits to a remote"), []
    if sub == "reset" and "--hard" in args:
        return "discards all uncommitted changes and moves the branch", []
    if sub == "clean":
        return "deletes files git doesn't track", []
    if sub == "checkout":
        return "throws away local edits to files", []
    if sub == "branch":
        return "deletes a branch even if it is not merged", []
    if sub == "rebase":
        return "rewrites commit history", []
    if sub == "stash":
        return "deletes stashed work", []
    return "rewrites git history", []


_TEMPLATES: List[Tuple[re.Pattern, Callable[[List[str], str], Tuple[str, List[str]]]]] = [
    (re.compile(r"^rm$"), _rm),
    (re.compile(r"^git$"), _git),
    (re.compile(r"^(chmod|chown)$"), lambda a, r: ("changes permissions or ownership recursively", _targets(a)[1:])),
    (re.compile(r"^(pkill|killall|kill)$"), lambda a, r: ("kills running processes", [])),
    (re.compile(r"^(truncate|shred)$"), lambda a, r: ("destroys the contents of files", _targets(a))),
    (re.compile(r"^find$"), lambda a, r: ("deletes every file it matches" if "-delete" in a else "searches files", _targets(a)[:1])),
    (re.compile(r"^docker$"), lambda a, r: ("removes containers, images or volumes", [])),
    (re.compile(r"^(npm|yarn|pnpm|pip|twine)$"), lambda a, r: ("publishes or removes packages", [])),
    (re.compile(r"^(printenv|env)$"), lambda a, r: ("prints environment variables, which may contain secrets", [])),
    (re.compile(r"^(shutdown|reboot|halt|poweroff)$"), lambda a, r: ("powers off this machine", [])),
]


def _first_match(command: str) -> Tuple[str, List[str]]:
    """(effect, file operands) for the first segment a template recognises, else ("", [])."""
    if re.search(r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?)\b", command):
        return "downloads a script and runs it immediately", []
    for segment in split_segments(command):
        variants = _variants(segment)
        elevated = len(variants) > 1 and segment.lstrip().startswith(("sudo", "doas"))
        toks = _tokens(variants[-1])
        if not toks:
            continue
        for pattern, build in _TEMPLATES:
            if pattern.match(os.path.basename(toks[0])):
                effect, targets = build(toks[1:], variants[-1])
                if effect:
                    return (f"as an administrator, {effect}" if elevated else effect), targets
        if elevated:
            return f"runs `{variants[-1][:60]}` as an administrator", _targets(toks[1:])
    if re.search(r"(\.ssh/|\.aws/|\.gnupg|auth\.json|(^|[\s/])\.env\b|id_rsa)", command):
        return "reads or changes credential files", []
    return "", []


def explain_command(command: str, reason: str, workspace: str, home: Optional[str] = None) -> Explanation:
    """What a risky shell command does, where, why we're asking, and what undo can't do."""
    effect, targets = _first_match(command)
    rows: List[Row] = []
    rows.append(("What it does", effect or "runs a command that matched a safety rule", "plain"))
    place = where(targets, workspace, home)
    if place:
        rows.append(place)
    if reason:
        rows.append(("Why asked", reason, "warn"))
    rows.append(("", NO_UNDO, "fail"))
    return Explanation(rows)


def explain_network(subject: str, reason: str) -> Explanation:
    rows: List[Row] = [("What it does", f"fetches a page from a private-network address: {subject}", "plain")]
    if reason:
        rows.append(("Why asked", reason, "warn"))
    rows.append(("", "Nothing is changed on disk, but the response is shown to the model.", "dim"))
    return Explanation(rows)


def explain_code(code: str, reason: str) -> Explanation:
    """Python the agent wants to run: the policy's reason is the explanation; the code itself is shown."""
    rows: List[Row] = []
    if reason:
        rows.append(("Why asked", reason, "warn"))
    rows.append(("", NO_UNDO, "fail"))
    return Explanation(rows)
