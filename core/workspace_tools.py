"""Workspace-scoped filesystem tools used by MotionAgent."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

# Default ceiling on how long a run_command invocation may take. Kept
# generous since build tasks may run tests/installs, but bounded so a
# hanging command can't stall the tool loop forever.
DEFAULT_COMMAND_TIMEOUT = 120
# Truncate captured output so a chatty command doesn't blow up the context
# window the same way read_file's `limit` protects against huge files.
COMMAND_OUTPUT_LIMIT = 20_000


# `\s*` right after `<` (and after `<` before `/`) tolerates stray spaces
# models occasionally insert, e.g. "< motion_tool>" or "< /motion_tool>".
# Without this, such near-misses fall through every pattern below AND the
# malformed-tag leak guard in parse_tool_call(), so the raw tool-call JSON
# gets shown to the user as if it were the model's final answer.
TOOL_CALL_PATTERN = re.compile(
    r"<\s*motion_tool\b[^>]*>\s*(\{.*?\})\s*<\s*/\s*motion_tool\b[^>]*>",
    re.DOTALL,
)
MOTION_ENVELOPE_PATTERN = re.compile(
    r"<\s*motion_[^>]*>\s*(?P<payload>.*?)\s*<\s*/\s*motion_tool\b[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
DIRECT_TOOL_PATTERN = re.compile(
    r"<\s*(?P<name>list_files|read_file|write_file|replace_in_file|run_command)\b[^>]*>"
    r"\s*(?P<arguments>\{.*?\})\s*<\s*/\s*(?P=name)\b[^>]*>",
    re.DOTALL,
)
# Some providers (observed on Ollama Cloud / deepseek-v4-flash) leak their own
# internal tool-channel markup verbatim instead of using <motion_tool>, e.g.
# "<| DSML | tool:write_file>{...}</ | DSML | tool>". Recognize this shape
# directly so it executes instead of leaking as the final answer.
DSML_TOOL_PATTERN = re.compile(
    r"<\s*\|\s*DSML\s*\|\s*tool\s*:\s*(?P<name>[a-zA-Z_]+)\s*>"
    r"\s*(?P<arguments>\{.*?\})\s*<\s*/\s*\|\s*DSML\s*\|\s*tool\s*>",
    re.DOTALL | re.IGNORECASE,
)
TOOL_MARKERS = (
    "motion_tool",
    "list_files",
    "read_file",
    "write_file",
    "replace_in_file",
    "run_command",
)


def _has_unknown_tool_envelope(raw: str) -> bool:
    """Catch unknown XML-like envelopes wrapping tool-shaped JSON.

    The old heuristic ("<" anywhere plus JSON-looking keys anywhere) falsely
    flagged natural-language responses that mentioned "arguments", "path", or
    "content" near an HTML tag. Only fire when there is a real <tag>...</tag>
    pair around a JSON object with tool-shaped keys.
    """
    for match in re.finditer(
        r"<([a-zA-Z_][\w:-]*)[^>]*>\s*(\{.*?\})\s*</\1\s*>",
        raw,
        re.DOTALL,
    ):
        payload = match.group(2)
        if ('"name"' in payload and '"arguments"' in payload) or (
            '"path"' in payload and '"content"' in payload
        ):
            return True
    return False


class WorkspaceToolError(ValueError):
    """Raised when a tool request is invalid or escapes the workspace."""


class OutOfWorkspaceError(WorkspaceToolError):
    """Raised when a tool request resolves to a path outside the workspace
    and hasn't been explicitly approved. Carries the resolved path so
    callers can offer the user a permission prompt instead of a hard
    failure (FR in the issue report)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"path escapes the workspace: {path}")


class WorkspaceTools:
    """Small, deterministic filesystem toolset restricted to one workspace."""

    def __init__(
        self,
        workspace: str | Path,
        read_only: bool = False,
        allowed_paths: "set[Path] | None" = None,
    ) -> None:
        self.root = Path(workspace).expanduser().resolve()
        self.read_only = read_only
        # Paths outside self.root that the user has explicitly approved this
        # session (see OutOfWorkspaceError / MotionAgent.run's
        # on_permission_request). Passed in by the caller so approvals
        # persist across turns, not just within one run() call.
        self.allowed_paths: set[Path] = allowed_paths if allowed_paths is not None else set()

    @property
    def instructions(self) -> str:
        mode = "READ-ONLY plan mode" if self.read_only else "BUILD mode with write access"
        write_tools = "" if self.read_only else """
- write_file: {"path": "relative/path", "content": "complete file contents"}
- replace_in_file: {"path": "relative/path", "old": "exact text", "new": "replacement text"}
- run_command: {"command": "pytest -q"} — runs a shell command in the workspace root (build mode only), returns exit_code/stdout/stderr"""
        if self.read_only:
            # Plan mode must never be told to write files - write_file/replace_in_file
            # are unavailable and calling them always fails. Instead of leaving the
            # model to punt back to the user ("say exactly what to build"), instruct
            # it to produce a concrete plan as its final text answer.
            goal_block = """
CRITICAL: You are in Plan mode. write_file and replace_in_file are DISABLED here - do
not attempt them. When the user describes something to build, do not ask them to repeat
or restate it. Instead, explore the workspace only as needed (list_files/read_file),
then respond with a concrete, structured PLAN as your final plain-text answer: a
proposed directory/file layout, the approach for each major piece, key libraries or
APIs to use, and any open questions. This plan is what the user will review and then
ask you to implement after switching you to Build mode.
""".strip()
        else:
            goal_block = """
CRITICAL: When the user asks you to create or generate something (a project, a script,
a scraper, a component, etc.), you MUST actually write files to the workspace using
write_file. If an earlier message in this conversation already proposed a plan (from
Plan mode), follow it instead of re-deriving one. Listing files or describing the plan
again is not enough. Produce concrete, complete files with sensible relative paths and
then summarize what you created.
""".strip()
        return f"""
You are an agent running on the user's machine in {mode}.
Workspace root: {self.root}

You have real filesystem tools. When the user asks you to create, modify, or inspect
project files, use these tools directly. Never say you cannot access the filesystem,
and never give the user a shell script merely to create files you can create yourself.

Available tools:
- list_files: {{"path": ".", "pattern": "*.py"}}
- read_file: {{"path": "relative/path"}}{write_tools}

To call a tool, respond with exactly one call and no surrounding prose:
<motion_tool>{{"name":"read_file","arguments":{{"path":"README.md"}}}}</motion_tool>

After each call you will receive a <motion_tool_result> message. Continue calling tools
until the requested work is complete, then give a concise final summary. Use relative
paths. Do not invent tool results. Do not place tool calls in Markdown fences.

CRITICAL: Always respond to the most recent user message above, not to earlier
messages in the conversation. If the user changes topic or asks a follow-up,
answer that follow-up directly.

{goal_block}
""".strip()

    def _resolve(self, raw_path: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkspaceToolError("path must be a non-empty string")
        candidate = (self.root / raw_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            if candidate in self.allowed_paths:
                return candidate
            raise OutOfWorkspaceError(candidate)
        return candidate

    def _display_path(self, path: Path) -> str:
        """Format a resolved path for tool results: relative to the workspace
        when inside it (the common case), or absolute for an
        explicitly-approved out-of-workspace path."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise WorkspaceToolError("arguments must be an object")
        if name == "list_files":
            return self._list_files(arguments)
        if name == "read_file":
            return self._read_file(arguments)
        if name == "write_file":
            self._require_write_access()
            return self._write_file(arguments)
        if name == "replace_in_file":
            self._require_write_access()
            return self._replace_in_file(arguments)
        if name == "run_command":
            self._require_write_access()
            return self._run_command(arguments)
        raise WorkspaceToolError(f"unknown tool: {name}")

    def _require_write_access(self) -> None:
        if self.read_only:
            raise WorkspaceToolError("write tools are disabled in plan mode")

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        directory = self._resolve(arguments.get("path", "."))
        pattern = arguments.get("pattern", "*")
        if not directory.exists():
            raise WorkspaceToolError(f"path does not exist: {arguments.get('path', '.')}")
        if not directory.is_dir():
            raise WorkspaceToolError("list_files path must be a directory")
        files = [
            self._display_path(path)
            for path in directory.rglob(pattern)
            if path.is_file()
        ]
        return {"files": files[:500], "truncated": len(files) > 500}

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path", ""))
        if not path.is_file():
            raise WorkspaceToolError(f"file does not exist: {arguments.get('path', '')}")
        content = path.read_text(encoding="utf-8", errors="replace")
        limit = 200_000
        return {
            "path": self._display_path(path),
            "content": content[:limit],
            "truncated": len(content) > limit,
        }

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path", ""))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise WorkspaceToolError("content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": self._display_path(path),
            "bytes_written": len(content.encode("utf-8")),
        }

    def _replace_in_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path", ""))
        old = arguments.get("old")
        new = arguments.get("new")
        if not isinstance(old, str) or not old:
            raise WorkspaceToolError("old must be a non-empty string")
        if not isinstance(new, str):
            raise WorkspaceToolError("new must be a string")
        if not path.is_file():
            raise WorkspaceToolError(f"file does not exist: {arguments.get('path', '')}")
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old)
        if occurrences != 1:
            raise WorkspaceToolError(
                f"old text must occur exactly once; found {occurrences} occurrences"
            )
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return {"path": self._display_path(path), "replacements": 1}

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a shell command in the workspace root (build mode only).

        Uses the same access boundary as write_file/replace_in_file - no
        extra approval prompt or allowlist beyond that, per the agreed
        safety model. Output is captured (not streamed) and truncated to
        keep the tool result within a reasonable size for the model.
        """
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise WorkspaceToolError("command must be a non-empty string")
        timeout = arguments.get("timeout") or DEFAULT_COMMAND_TIMEOUT
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = DEFAULT_COMMAND_TIMEOUT
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceToolError(
                f"command timed out after {timeout:.0f}s: {command}"
            ) from exc
        except Exception as exc:
            raise WorkspaceToolError(f"failed to run command: {exc}") from exc
        stdout = (proc.stdout or "")
        stderr = (proc.stderr or "")
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout[:COMMAND_OUTPUT_LIMIT],
            "stderr": stderr[:COMMAND_OUTPUT_LIMIT],
            "truncated": len(stdout) > COMMAND_OUTPUT_LIMIT or len(stderr) > COMMAND_OUTPUT_LIMIT,
        }


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one Motion tool call from a model response."""
    raw = text or ""
    match = TOOL_CALL_PATTERN.search(raw)
    if not match:
        envelope = MOTION_ENVELOPE_PATTERN.search(raw)
        if envelope:
            return _parse_motion_payload(envelope.group("payload"))
    if not match:
        direct_match = DIRECT_TOOL_PATTERN.search(raw)
        if direct_match:
            arguments = json.loads(direct_match.group("arguments"))
            if not isinstance(arguments, dict):
                raise WorkspaceToolError("tool arguments must be an object")
            return direct_match.group("name"), arguments
    if not match:
        dsml_match = DSML_TOOL_PATTERN.search(raw)
        if dsml_match:
            arguments = json.loads(dsml_match.group("arguments"))
            if not isinstance(arguments, dict):
                raise WorkspaceToolError("tool arguments must be an object")
            return dsml_match.group("name"), arguments
    if not match:
        # Never leak a malformed tool request into the user-facing response.
        # Models occasionally produce variants such as <motion_tool"> or
        # provider-style direct tags such as <list_files>, or insert a stray
        # space right after "<" (e.g. "< motion_tool>"). `\s*` after `<`
        # catches that last case so it's treated as a recoverable parse
        # error (which prompts the model to retry) instead of leaking the
        # raw tag/JSON to the user as if it were the final answer.
        lowered = raw.lower()
        looks_like_leaked_tool_call = (
            re.search(r"<\s*motion_", lowered)
            or any(
                re.search(rf"<\s*{re.escape(marker)}\b", lowered) for marker in TOOL_MARKERS
            )
            or "dsml" in lowered
            # Generic last-resort net: whatever exotic envelope name a model
            # invents next, a tag wrapping something shaped like our tool
            # JSON (write_file's path+content, or a name/arguments envelope)
            # is never a legitimate final answer.
            or _has_unknown_tool_envelope(raw)
        )
        if looks_like_leaked_tool_call:
            raise WorkspaceToolError("malformed filesystem tool envelope")
        return None
    return _parse_motion_payload(match.group(1))


def _parse_motion_payload(payload_text: str) -> tuple[str, dict[str, Any]]:
    """Parse JSON, with a narrow recovery path for malformed write_file content.

    Some models emit a valid outer shape but fail to JSON-escape source-code
    quotes inside the content value. For write_file only, recover the path and
    treat everything after the content delimiter as raw source text.
    """
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return _recover_write_file_payload(payload_text)

    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str):
        raise WorkspaceToolError("tool name must be a string")
    if not isinstance(arguments, dict):
        raise WorkspaceToolError("tool arguments must be an object")
    return name, arguments


def _recover_write_file_payload(payload_text: str) -> tuple[str, dict[str, Any]]:
    """Recover the specific malformed JSON shape produced for source files."""
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', payload_text)
    if not name_match or name_match.group(1) != "write_file":
        raise WorkspaceToolError("malformed motion tool JSON")

    path_match = re.search(
        r'"path"\s*:\s*"((?:\\.|[^"\\])*)"',
        payload_text,
        re.DOTALL,
    )
    content_match = re.search(r'"content"\s*:\s*', payload_text)
    if not path_match or not content_match:
        raise WorkspaceToolError("malformed write_file arguments")

    try:
        path = json.loads(f'"{path_match.group(1)}"')
    except json.JSONDecodeError as exc:
        raise WorkspaceToolError("malformed write_file path") from exc

    content = payload_text[content_match.end():].strip()
    # Remove outer object terminators, then the JSON delimiter quote. Internal
    # source-code quotes are intentionally preserved as raw file content.
    content = re.sub(r"\s*}\s*}\s*$", "", content, count=1).strip()
    if content.startswith('"'):
        content = content[1:]
    if content.endswith('"'):
        content = content[:-1]
    content = (
        content.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )
    return "write_file", {"path": path, "content": content}


def format_tool_result(
    name: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    payload = {"name": name, "ok": error is None}
    if error is None:
        payload["result"] = result or {}
    else:
        payload["error"] = error
    return f"<motion_tool_result>{json.dumps(payload)}</motion_tool_result>"
