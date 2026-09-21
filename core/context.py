"""Context-window hygiene for the agent loop.

The tool loop resends the whole conversation on every step, so unbounded tool
output makes each step slower and pricier than the last. These helpers keep
the working set small: old tool output is trimmed, and if the conversation
still approaches the model's window the oldest tool exchanges are dropped in
favour of a short summary of what was done.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

CHARS_PER_TOKEN = 4
RESULT_PREFIX = "<motion_tool_result>"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def message_tokens(msg: Dict[str, Any]) -> int:
    total = estimate_tokens(_content_text(msg.get("content")))
    for call in msg.get("tool_calls") or []:
        total += estimate_tokens(str(call.get("arguments", "")))
    if isinstance(msg.get("content"), list):
        # Images: rough flat cost rather than counting base64 characters.
        total += 800 * sum(1 for p in msg["content"] if isinstance(p, dict) and p.get("type") == "image")
    return total


def messages_tokens(messages: List[Dict[str, Any]], system_prompt: str = "") -> int:
    return estimate_tokens(system_prompt) + sum(message_tokens(m) for m in messages)


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    return msg.get("role") == "user" and isinstance(content, str) and content.startswith(RESULT_PREFIX)


def trim_old_tool_results(messages: List[Dict[str, Any]], keep_recent: int = 6, max_chars: int = 1500) -> int:
    """Shorten all but the newest ``keep_recent`` tool results in place.
    Returns the number of characters removed."""
    idxs = [i for i, m in enumerate(messages) if _is_tool_result(m)]
    removed = 0
    for i in idxs[: max(0, len(idxs) - keep_recent)]:
        content = messages[i].get("content")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        cut = len(content) - max_chars
        keep_tail = ""
        if content.startswith(RESULT_PREFIX):
            keep_tail = "</motion_tool_result>"
        messages[i] = {
            **messages[i],
            "content": f"{content[:max_chars]}…[{cut} chars of older tool output trimmed; re-run the tool if you need it]{keep_tail}",
        }
        removed += cut
    return removed


def _describe_group(group: List[Dict[str, Any]]) -> List[str]:
    ops: List[str] = []
    for m in group:
        for call in m.get("tool_calls") or []:
            args = call.get("arguments") or {}
            target = args.get("path") or args.get("command") or args.get("pattern") or args.get("url") or ""
            ops.append(f"{call.get('name')}({str(target)[:60]})")
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            for name in re.findall(r'"name"\s*:\s*"([a-z_]+)"', _content_text(m.get("content"))):
                ops.append(name)
    return ops


def compact_messages(
    messages: List[Dict[str, Any]],
    system_prompt: str,
    window_tokens: int,
    start_index: int,
    high: float = 0.75,
    target: float = 0.5,
) -> bool:
    """If the conversation exceeds ``high`` of the window, drop the oldest
    assistant/tool exchanges (after ``start_index``, the current user prompt)
    until it is under ``target``. Whole exchanges are removed so tool-call /
    tool-result pairing stays valid. Returns True if anything was dropped."""
    if window_tokens <= 0 or messages_tokens(messages, system_prompt) <= window_tokens * high:
        return False
    # Exchange boundaries = each assistant message after the user prompt.
    starts = [i for i in range(start_index + 1, len(messages)) if messages[i].get("role") == "assistant"]
    if len(starts) <= 2:
        return False
    groups = [messages[s:e] for s, e in zip(starts, starts[1:] + [len(messages)])]
    dropped: List[List[Dict[str, Any]]] = []
    while len(groups) > 2 and messages_tokens(
        messages[: start_index + 1] + [m for g in groups for m in g], system_prompt
    ) > window_tokens * target:
        dropped.append(groups.pop(0))
    if not dropped:
        return False
    ops = [op for g in dropped for op in _describe_group(g)]
    summary = (
        f"[{len(dropped)} earlier step(s) were trimmed to save context. "
        f"Actions taken so far: {', '.join(ops[:40])}{'…' if len(ops) > 40 else ''}. "
        "Re-read files if you need their current contents.]"
    )
    kept = messages[: start_index + 1] + [{"role": "user", "content": summary}] + [m for g in groups for m in g]
    messages[:] = kept
    return True
