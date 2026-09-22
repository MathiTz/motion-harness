"""User hooks: run your own commands before/after tool calls.

    hooks:
      pre_tool:                              # can BLOCK the call (non-zero exit = block)
        - match: "write_file|replace_in_file|edit_files"
          command: "./scripts/guard.sh"
          timeout: 10
      post_tool:                             # annotate the result (never blocks)
        - match: "write_file|replace_in_file|edit_files"
          command: "ruff format ."

The hook gets the call as JSON on stdin ({event, tool, arguments, workspace, result?})
and MOTION_HOOK_EVENT / MOTION_TOOL / MOTION_WORKSPACE in its environment, runs in the
workspace root, and its output goes back to the model. Hooks are commands YOU configured,
so they run outside the write sandbox with your normal permissions.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

OUTPUT_LIMIT = 2000
DEFAULT_TIMEOUT = 15.0
EVENTS = ("pre_tool", "post_tool")


@dataclass
class Hook:
    event: str
    pattern: "re.Pattern[str]"
    command: str
    timeout: float = DEFAULT_TIMEOUT


@dataclass
class HookResult:
    blocked: bool = False
    output: str = ""


@dataclass
class Hooks:
    hooks: List[Hook] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)  # config entries that were ignored, and why

    @classmethod
    def from_config(cls, get: Callable[..., Any]) -> "Hooks":
        raw = get("hooks", None) or {}
        out = cls()
        if not isinstance(raw, dict):
            out.problems.append("hooks must be a mapping of pre_tool/post_tool lists")
            return out
        for event in EVENTS:
            for i, entry in enumerate(raw.get(event) or []):
                if not isinstance(entry, dict) or not str(entry.get("command") or "").strip():
                    out.problems.append(f"hooks.{event}[{i}]: needs a 'command'")
                    continue
                try:
                    pattern = re.compile(str(entry.get("match") or ".*"))
                    timeout = float(entry.get("timeout") or DEFAULT_TIMEOUT)
                except (re.error, TypeError, ValueError) as exc:
                    out.problems.append(f"hooks.{event}[{i}]: {exc}")
                    continue
                out.hooks.append(Hook(event, pattern, str(entry["command"]), max(0.5, timeout)))
        return out

    def __bool__(self) -> bool:
        return bool(self.hooks)

    def matching(self, event: str, tool: str) -> List[Hook]:
        return [h for h in self.hooks if h.event == event and h.pattern.fullmatch(tool)]

    async def run(self, event: str, tool: str, arguments: Dict[str, Any], workspace: str,
                  result: Optional[Dict[str, Any]] = None) -> HookResult:
        """Run every matching hook in order. For ``pre_tool`` the first failure blocks the call."""
        outputs: List[str] = []
        payload = {"event": event, "tool": tool, "arguments": arguments, "workspace": workspace}
        if result is not None:
            payload["result"] = {k: v for k, v in result.items() if not k.startswith("_")}
        stdin = json.dumps(payload, default=str)[:200_000].encode()
        env = {**os.environ, "MOTION_HOOK_EVENT": event, "MOTION_TOOL": tool, "MOTION_WORKSPACE": workspace}
        for hook in self.matching(event, tool):
            code, text = await _run_one(hook, stdin, env, workspace)
            if text:
                outputs.append(text)
            if code != 0 and event == "pre_tool":
                reason = text or f"hook `{hook.command}` exited {code}"
                return HookResult(blocked=True, output=reason)
            if code != 0:
                outputs.append(f"(hook `{hook.command}` exited {code})")
        return HookResult(blocked=False, output="\n".join(outputs)[:OUTPUT_LIMIT])


async def _run_one(hook: Hook, stdin: bytes, env: Dict[str, str], cwd: str) -> "tuple[int, str]":
    try:
        proc = await asyncio.create_subprocess_shell(
            hook.command, cwd=cwd, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        return 127, f"could not start hook: {exc}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin), hook.timeout)
    except asyncio.TimeoutError:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, 9)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        await proc.wait()
        return 124, f"hook `{hook.command}` timed out after {hook.timeout:.0f}s"
    text = out.decode("utf-8", "replace").strip()
    return (proc.returncode if proc.returncode is not None else 1), text[:OUTPUT_LIMIT]
