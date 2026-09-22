"""Delegate a whole turn to an already-installed, already-logged-in coding CLI (Claude Code, Codex),
so someone with a Pro/Max or ChatGPT subscription - not a separate API key - can still use this
harness.

This is legitimate, sanctioned automation, not a workaround: each CLI documents running itself
headlessly this exact way, using whatever the SAME binary is already authenticated with when you use
it interactively (subscription or API key - not our concern, that is the CLI's own resolution). We
never touch OAuth, store a token, or reuse either vendor's own client credentials:

  Claude Code: `claude -p "<prompt>" --output-format stream-json` (without --bare, so it uses the
  session you already have) is the pattern Anthropic's own docs show for scripting, including from an
  npm script (https://code.claude.com/docs/en/headless). `--bare` explicitly does NOT use the
  subscription login; we deliberately never pass it.

  Codex CLI: `codex exec` reuses the credentials `codex login` cached at ~/.codex/auth.json
  (https://learn.chatgpt.com/codex/auth, /codex/non-interactive-mode). OpenAI's own docs recommend an
  API key as the default for automation, which this harness respects: the Codex delegate is opt-in
  (config `enable_codex_cli_delegate: true`), never auto-selected just because `codex` is on PATH.

A delegate is a fundamentally different shape than a normal provider: it runs its OWN agent internally
(its own tools, its own edits, directly in the workspace) rather than being a bare completion endpoint
our tool-calling loop drives - see core/agent_loop.py's DELEGATE short-circuit for how a turn differs.
Consequences worth stating plainly, and that HANDLED IN THIS MODULE state to the user at the start of
a delegate turn: our own sandbox, budget and hooks do not gate what happens inside the call (the CLI's
own permission flags do, set here to mirror plan/build mode), and our /undo checkpoints - recorded
only around OUR OWN tool calls - do not cover its edits.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from core.providers import BaseProvider, ModelConfig, ProviderError, StreamEvent

UNDO_CAVEAT = (
    "edits made through this session are not covered by /undo (its checkpoints only record the "
    "harness's own file-tool calls)"
)


@lru_cache(maxsize=4)
def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def clear_detection_cache() -> None:
    """Tests, and a config/session that installs one of these mid-run, need this - PATH lookups are
    cached so a session doesn't re-stat the filesystem on every keystroke of the model picker."""
    _which.cache_clear()


def claude_cli_available() -> bool:
    return _which("claude") is not None


def codex_cli_available() -> bool:
    return _which("codex") is not None


@dataclass
class DelegateResult:
    text: str
    session_ref: Optional[str] = None       # opaque id/path this CLI uses to resume the conversation
    cost_usd: Optional[float] = None        # None: not reported, or not meaningful (subscription usage)
    usage: Optional[Dict[str, int]] = None
    tool_lines: List[str] = field(default_factory=list)   # short, human-readable progress ("wrote x.py")


class CLIDelegateError(ProviderError):
    """The delegate CLI could not be run, or reported it could not complete the turn."""


class CLIDelegateProvider(BaseProvider):
    """Base for a provider that shells out to an installed CLI instead of calling an HTTP API.

    Subclasses supply ``binary``, ``display_name``, ``build_argv`` and ``parse_line``; this class owns
    the subprocess lifecycle (spawn, stream stdout as NDJSON, cancellation, timeout) and turns whatever
    ``parse_line`` reports into :class:`StreamEvent`s. Tool activity the delegate reports is surfaced as
    ``kind="reasoning"`` events (the existing live-narration channel: TurnRunner already forwards these
    as visible progress) - never as ``kind="tool_call"``, which would make TurnRunner try to execute it
    as one of OUR tools. See core/agent_loop.py's delegate short-circuit for the other half of this.
    """

    binary = ""
    display_name = ""
    #: seconds of no output at all before giving up (the CLI's own tools can legitimately run long -
    #: a build, a test suite - so this is generous; per-provider timeout config can override it)
    default_timeout = 1800.0

    def __init__(self, config: ModelConfig) -> None:
        # No httpx client needed for a subprocess-based provider.
        self.config = config
        self.last_usage: Optional[Dict[str, int]] = None
        self._native_tools_disabled = True  # never meaningful here; see the class docstring
        self.last_result: Optional[DelegateResult] = None
        self.session_ref: Optional[str] = config.options.get("_session_ref")
        self._proc: Optional[asyncio.subprocess.Process] = None

    @property
    def native_tools(self) -> bool:
        return False

    @property
    def is_delegate(self) -> bool:  # a cheap, name-stable marker other modules can isinstance-check or duck-type
        return True

    def close(self) -> None:  # matches BaseProvider's interface; nothing to close
        return None

    # ── the pieces a concrete CLI fills in ──────────────────────────────────
    def build_argv(self, prompt: str, *, mode: str, session_ref: Optional[str]) -> List[str]:
        raise NotImplementedError

    def parse_line(self, data: Dict[str, Any], state: Dict[str, Any]) -> Optional[StreamEvent]:
        """One parsed NDJSON line -> a StreamEvent to emit, or None. ``state`` is a plain dict this
        call may freely read/write to accumulate things (session id, cost, tool lines) across lines of
        one call; read back via ``finalize(state)`` once the process exits."""
        raise NotImplementedError

    def finalize(self, state: Dict[str, Any], final_text: str) -> DelegateResult:
        raise NotImplementedError

    # ── shared subprocess/streaming machinery ───────────────────────────────
    async def run_delegate(
        self, prompt: str, *, mode: str, workspace: str, on_event: Optional[Callable[[StreamEvent], Any]] = None,
    ) -> DelegateResult:
        """Run one turn, calling ``on_event`` (if given) with each StreamEvent as it arrives for live
        display. Raises CLIDelegateError if the binary is missing or the run fails outright; a non-zero
        exit with SOME text already produced is treated as a completed (if possibly truncated) answer,
        not an error - matching how a real interactive run would show whatever it got that far."""
        if not shutil.which(self.binary):
            raise CLIDelegateError(f"{self.display_name} ('{self.binary}') is no longer on PATH")
        argv = self.build_argv(prompt, mode=mode, session_ref=self.session_ref)
        state: Dict[str, Any] = {}
        text_parts: List[str] = []
        timeout = float(self.config.options.get("timeout", self.default_timeout))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=workspace, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )
        except OSError as exc:
            raise CLIDelegateError(f"could not start {self.display_name}: {exc}") from exc
        self._proc = proc
        assert proc.stdout is not None and proc.stderr is not None
        stderr_task = asyncio.create_task(proc.stderr.read())  # must be drained concurrently, or a
        # chatty process can fill its pipe buffer and deadlock waiting for us to read it
        last_output = time.monotonic()
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=max(1.0, timeout))
                except asyncio.TimeoutError:
                    if time.monotonic() - last_output >= timeout:
                        raise CLIDelegateError(
                            f"{self.display_name} produced no output for {timeout:.0f}s and was stopped"
                        ) from None
                    continue
                if not raw:
                    break
                last_output = time.monotonic()
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                ev = self.parse_line(data, state)
                if ev is not None:
                    if ev.kind == "text":
                        text_parts.append(ev.text)
                    if on_event is not None:
                        result = on_event(ev)
                        if asyncio.iscoroutine(result):
                            await result
            rc = await proc.wait()
        except asyncio.CancelledError:
            await self._kill()
            raise
        finally:
            self._proc = None
        stderr = b""
        try:
            stderr = await asyncio.wait_for(stderr_task, timeout=2)
        except Exception:
            pass
        if rc != 0 and not text_parts and not state.get("session_id"):
            detail = stderr.decode("utf-8", "replace").strip()[:800]
            raise CLIDelegateError(
                f"{self.display_name} exited {rc} without producing a result" + (f": {detail}" if detail else "")
            )
        return self.finalize(state, "".join(text_parts))

    async def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=3)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    # ── BaseProvider interface: not really used (see the TurnRunner short-circuit in
    # core/agent_loop.py, which calls run_delegate directly), kept so a delegate is at least inert
    # rather than broken if something ever calls it as an ordinary provider ──
    async def chat_stream(self, messages, system_prompt="", tools=None, **kwargs) -> AsyncIterator[StreamEvent]:
        prompt = messages[-1]["content"] if messages else ""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        pending: List[StreamEvent] = []
        result = await self.run_delegate(prompt, mode="build", workspace=os.getcwd(), on_event=pending.append)
        for ev in pending:
            yield ev
        self.last_result = result
        self.session_ref = result.session_ref or self.session_ref
        if result.usage:
            self.last_usage = result.usage
            yield StreamEvent("usage", usage=result.usage)


# ── Claude Code ──────────────────────────────────────────────────────────────

class ClaudeCLIProvider(CLIDelegateProvider):
    binary = "claude"
    display_name = "Claude Code"

    def build_argv(self, prompt, *, mode, session_ref):
        argv = [
            self.binary, "-p", prompt,
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--permission-prompts", "none",  # nothing can answer a prompt in this run; deny rather than hang
        ]
        if mode == "plan":
            argv += ["--allowedTools", "Read,Grep,Glob,WebSearch,WebFetch"]
        else:
            argv += ["--permission-mode", "acceptEdits",
                     "--allowedTools", "Bash,Read,Edit,Write,Grep,Glob,WebSearch,WebFetch"]
        if session_ref:
            argv += ["--resume", session_ref]
        return argv

    def parse_line(self, data, state):
        etype = data.get("type")
        if etype == "system" and data.get("subtype") == "init":
            state["session_id"] = data.get("session_id")
            return None
        if etype == "stream_event":
            event = data.get("event") or {}
            if event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    state["saw_text"] = True
                    return StreamEvent("text", text=delta["text"])
            return None
        if etype in ("assistant", "user"):
            for block in ((data.get("message") or {}).get("content") or []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    line = f"🔧 {block.get('name', 'tool')}"
                    inp = block.get("input")
                    if isinstance(inp, dict) and inp:
                        first = next(iter(inp.items()))
                        line += f"({first[0]}={str(first[1])[:60]})"
                    state.setdefault("tool_lines", []).append(line)
                    return StreamEvent("reasoning", text=f"\n{line}\n")
            return None
        if etype == "result":
            state["session_id"] = data.get("session_id") or state.get("session_id")
            state["cost_usd"] = data.get("total_cost_usd")
            usage = data.get("usage") or {}
            if usage:
                state["usage"] = {
                    "prompt_tokens": int(usage.get("input_tokens") or 0),
                    "completion_tokens": int(usage.get("output_tokens") or 0),
                    "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
                }
            if data.get("is_error"):
                state["error"] = data.get("result") or "the run reported an error"
            elif not state.get("saw_text") and data.get("result"):
                # --include-partial-messages sometimes omits deltas for a very short reply; the result
                # event always carries the final text, so use it if nothing streamed.
                return StreamEvent("text", text=str(data["result"]))
            return None
        return None

    def finalize(self, state, final_text):
        if state.get("error") and not final_text.strip():
            raise CLIDelegateError(f"{self.display_name}: {state['error']}")
        return DelegateResult(
            text=final_text.strip(), session_ref=state.get("session_id"), cost_usd=state.get("cost_usd"),
            usage=state.get("usage"), tool_lines=state.get("tool_lines", []),
        )


# ── Codex CLI ────────────────────────────────────────────────────────────────

class CodexCLIProvider(CLIDelegateProvider):
    binary = "codex"
    display_name = "Codex"

    def build_argv(self, prompt, *, mode, session_ref):
        sandbox = "read-only" if mode == "plan" else "workspace-write"
        base = [self.binary, "exec", "--json", "--sandbox", sandbox, "--skip-git-repo-check"]
        if session_ref:
            # Best-effort: resuming a previous thread is not as firmly documented for Codex as
            # Claude's --resume. If this exact form is wrong for the installed version, run_delegate's
            # caller (see core/agent_loop.py) retries once as a fresh thread rather than failing the turn.
            return [*base, "resume", session_ref, prompt]
        return [*base, prompt]

    def parse_line(self, data, state):
        etype = data.get("type")
        if etype == "thread.started":
            state["session_id"] = data.get("thread_id")
            return None
        if etype == "item.completed":
            item = data.get("item") or {}
            kind = item.get("type")
            if kind == "agent_message" and item.get("text"):
                return StreamEvent("text", text=str(item["text"]))
            label = kind or "item"
            detail = item.get("command") or item.get("path") or item.get("summary") or ""
            line = f"🔧 {label}" + (f"({str(detail)[:60]})" if detail else "")
            state.setdefault("tool_lines", []).append(line)
            return StreamEvent("reasoning", text=f"\n{line}\n")
        if etype == "turn.completed":
            usage = data.get("usage") or {}
            if usage:
                state["usage"] = {
                    "prompt_tokens": int(usage.get("input_tokens") or 0),
                    "completion_tokens": int(usage.get("output_tokens") or 0),
                    "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
                }
            return None
        if etype in ("turn.failed", "error"):
            state["error"] = data.get("message") or data.get("error") or "the run reported an error"
            return None
        return None

    def finalize(self, state, final_text):
        if state.get("error") and not final_text.strip():
            raise CLIDelegateError(f"{self.display_name}: {state['error']}")
        return DelegateResult(
            text=final_text.strip(), session_ref=state.get("session_id"),
            cost_usd=None,  # Codex reports tokens, not a dollar figure - usage is under a subscription, not metered here
            usage=state.get("usage"), tool_lines=state.get("tool_lines", []),
        )


DELEGATE_CATALOG: Dict[str, Callable[[], bool]] = {
    "claude-cli": claude_cli_available,
    "codex-cli": codex_cli_available,
}
DELEGATE_CLASSES: Dict[str, type] = {"claude-cli": ClaudeCLIProvider, "codex-cli": CodexCLIProvider}
DELEGATE_NAMES: Dict[str, str] = {"claude-cli": "Claude Code (your login)", "codex-cli": "Codex (your login)"}
