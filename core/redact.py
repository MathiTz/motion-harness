"""Best-effort redaction of secret-shaped values from text that is about to be written to disk or
shared - currently only `/trajectory save full` (issue #19), which is the one place the harness
writes a raw system prompt and every raw message it sent to the model.

This is pattern-based redaction of obviously secret-shaped strings, not a general DLP system: it
will not catch every possible sensitive value (a secret with no recognizable shape - a plain
password typed in prose, say - passes through untouched). It exists to stop the common, clearly-
identifiable cases (an env var's value, a bearer token, a vendor API key) from landing in a shared
debug file by default, not to guarantee the file is safe to publish.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "[REDACTED]"

# NAME=value / NAME: "value" where NAME looks like a secret (same family of names as
# core/workspace_tools.py's _SECRET_ENV, which strips these from child-process environments -
# reused here as the starting point for values appearing in plain text, e.g. a .env file's
# contents read back by `read_file` and now sitting in a tool result in the transcript).
_SECRET_NAME = r"[A-Za-z0-9][A-Za-z0-9_]*(?:_API_KEY|_SECRET|_SECRET_KEY|_PASSWORD|_TOKEN|_ACCESS_KEY)|API_KEY|SECRET|PASSWORD|TOKEN"
_KEY_VALUE = re.compile(
    rf'(?im)\b({_SECRET_NAME})(\s*[:=]\s*)(")?([^\s"\']+)(")?'
)

# Authorization: Bearer <token> / a bare "Bearer <token>" - the value only, keep the scheme visible
# so a redacted transcript still shows THAT auth was attempted, just not with what.
_BEARER = re.compile(r"(?i)\b(Bearer\s+)([A-Za-z0-9._~+/=-]{8,})")

# Vendor key prefixes distinctive enough to redact even with no nearby "KEY=" label - a bare key
# pasted into a file or error message. Each pattern is (label shown in place of the value, regex).
_PREFIXED_KEYS = [
    (r"sk-ant-[A-Za-z0-9_-]{20,}", "anthropic"),
    (r"sk-[A-Za-z0-9]{20,}", "openai-style"),
    (r"ghp_[A-Za-z0-9]{30,}", "github-pat"),
    (r"gho_[A-Za-z0-9]{30,}", "github-oauth"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack"),
    (r"AKIA[A-Z0-9]{16}", "aws-access-key-id"),
    (r"AIza[A-Za-z0-9_-]{35}", "google-api-key"),
]
_PREFIXED = [(re.compile(pat), tag) for pat, tag in _PREFIXED_KEYS]


def redact_text(text: str) -> str:
    """Return ``text`` with secret-shaped values replaced by ``[REDACTED]``. Safe on any string,
    including one with no secrets (returned unchanged)."""
    if not text:
        return text

    def _kv(m: "re.Match[str]") -> str:
        name, sep, q1, _value, q2 = m.groups()
        return f"{name}{sep}{q1 or ''}{_REDACTED}{q2 or ''}"

    out = _KEY_VALUE.sub(_kv, text)
    out = _BEARER.sub(lambda m: f"{m.group(1)}{_REDACTED}", out)
    for pattern, tag in _PREFIXED:
        out = pattern.sub(f"{_REDACTED}:{tag}", out)
    return out


def redact_value(value: Any) -> Any:
    """Recursively apply ``redact_text`` to every string inside ``value`` - a message list is a
    mix of dicts, lists and strings (content, tool_calls, tool results); this walks all of it
    without needing to know the exact shape in advance."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value
