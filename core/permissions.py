"""Command permission policy for ``run_command``.

Decision order (first match wins):

1. ``permissions.commands.deny`` patterns from config.yml     -> deny
2. built-in catastrophic patterns (rm -rf /, mkfs, fork bomb) -> deny
3. commands the user approved earlier this session            -> allow
4. ``permissions.commands.allow`` patterns from config.yml    -> allow
5. ``permissions.commands.ask`` patterns from config.yml      -> ask
6. built-in risky patterns (rm -r, sudo, git push, curl|sh...) -> ask
7. everything else                                            -> allow

Patterns in config use shell-style globs matched against the whole command,
e.g. ``pytest*`` or ``git push*``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from typing import Iterable, Optional, Set, Tuple

_CATASTROPHIC = [
    # Target may be quoted, end in "/" or "/*", and be followed by a shell or
    # code delimiter (so it also matches inside os.system('rm -rf ~') and
    # "$HOME"). Was previously only matched before whitespace/end-of-string.
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*[\"']?(/|~/?|\$HOME/?|\$\{HOME\}/?|/\*|~/\*|\$HOME/\*|\*)[\"']*(\s|$|[;&|)\]},])", "recursive delete of / , ~ or *"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b", "formats a filesystem"),
    (r"\bdd\b[^|;&]*\bof=/dev/", "writes directly to a device"),
    (r">\s*/dev/(sd|nvme|disk|hd)\w*", "writes directly to a device"),
    (r"\bchmod\s+-R\s+[0-7]{3,4}\s+/(\s|$)", "recursive chmod on /"),
]

_RISKY = [
    (r"\brm\b[^|;&]*\s-[a-zA-Z]*[rRfF]", "deletes files recursively/forcefully"),
    (r"\bsudo\b|\bdoas\b", "runs with elevated privileges"),
    (r"\bgit\s+(push|reset\s+--hard|clean|checkout\s+--|branch\s+-D|rebase|stash\s+(drop|clear)|filter-branch)\b", "rewrites/publishes git state"),
    (r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?)\b", "pipes a download into a shell"),
    (r"\b(chmod|chown)\s+-R\b", "recursive permission change"),
    (r"\b(pkill|killall)\b|\bkill\s+-9\b", "kills processes"),
    (r"\bfind\b[^|;&]*\s-delete\b|\btruncate\b|\bshred\b", "destroys file contents"),
    (r"\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm|kill)\b", "removes containers/images"),
    (r"\b(npm|yarn|pnpm)\s+(publish|unpublish)\b|\bpip\s+uninstall\b|\btwine\s+upload\b", "publishes or removes packages"),
    (r"\b(printenv|env)\s*($|[|;&>])", "dumps environment (may contain secrets)"),
    (r"(\.ssh/|\.aws/|\.gnupg|auth\.json|(^|[\s/])\.env\b|id_rsa)", "touches credentials/secrets"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "powers off the machine"),
]

# Python code the agent wants to run. This is a UX prompt, not a security
# boundary (code can always be obfuscated): the OS sandbox is what enforces
# confinement. It exists so common destructive/spawning idioms ask first, the
# way their shell equivalents do.
_CODE_RISKY = [
    (r"\b(shutil\.rmtree|rmtree)\s*\(|\bos\.(remove|unlink|rmdir|removedirs)\s*\(|\.(unlink|rmdir)\s*\(", "deletes files/directories"),
    (r"\bos\.(system|popen|exec\w*|spawn\w*)\s*\(|\bsubprocess\b|\bpty\.spawn\b", "spawns a subprocess, bypassing the shell command rules"),
]

_CATASTROPHIC_RE = [(re.compile(p), why) for p, why in _CATASTROPHIC]
_CODE_RISKY_RE = [(re.compile(p), why) for p, why in _CODE_RISKY]
_RISKY_RE = [(re.compile(p), why) for p, why in _RISKY]


class CommandPolicy:
    def __init__(
        self,
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        ask: Iterable[str] = (),
        approved: Optional[Set[str]] = None,
    ) -> None:
        self.allow = list(allow)
        self.deny = list(deny)
        self.ask = list(ask)
        # Commands approved "for this session" (shared with the caller so the
        # approval survives across turns).
        self.approved: Set[str] = approved if approved is not None else set()

    @classmethod
    def from_config(cls, config: Optional[dict], approved: Optional[Set[str]] = None) -> "CommandPolicy":
        cmds = ((config or {}).get("permissions") or {}).get("commands") or {}
        return cls(cmds.get("allow") or (), cmds.get("deny") or (), cmds.get("ask") or (), approved)

    @staticmethod
    def _matches(command: str, patterns: Iterable[str]) -> Optional[str]:
        for pat in patterns:
            if fnmatch.fnmatchcase(command, pat):
                return pat
        return None

    def decide(self, command: str) -> Tuple[str, str]:
        """Return ``(decision, reason)`` where decision is allow|ask|deny."""
        cmd = (command or "").strip()
        hit = self._matches(cmd, self.deny)
        if hit:
            return "deny", f"blocked by permissions.commands.deny ({hit})"
        for rx, why in _CATASTROPHIC_RE:
            if rx.search(cmd):
                return "deny", f"blocked: {why}"
        if cmd in self.approved:
            return "allow", "approved earlier this session"
        if self._matches(cmd, self.allow):
            return "allow", "allowed by permissions.commands.allow"
        hit = self._matches(cmd, self.ask)
        if hit:
            return "ask", f"matches permissions.commands.ask ({hit})"
        for rx, why in _RISKY_RE:
            if rx.search(cmd):
                return "ask", why
        return "allow", ""

    @staticmethod
    def code_key(code: str) -> str:
        return "code:" + hashlib.sha1((code or "").encode("utf-8", "replace")).hexdigest()

    def decide_code(self, code: str) -> Tuple[str, str]:
        """Classify Python source (from run_python / run_script).

        Catastrophic shell strings embedded in the code are refused; deleting
        or process-spawning idioms ask, like their shell equivalents.
        """
        text = code or ""
        for rx, why in _CATASTROPHIC_RE:
            if rx.search(text):
                return "deny", f"blocked: {why}"
        if self.code_key(text) in self.approved:
            return "allow", "approved earlier this session"
        for rx, why in _CODE_RISKY_RE:
            if rx.search(text):
                return "ask", why
        return "allow", ""

    def remember(self, command: str) -> None:
        self.approved.add((command or "").strip())
