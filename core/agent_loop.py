"""One agent turn: recall -> (model step -> tool calls)* -> final answer.

Three interchangeable ways of talking to the model, chosen per provider:

  native  - provider tool-calling API (streamed; several independent calls
            per step run in parallel)
  xml     - streamed text with <motion_tool> envelopes, for models without a
            tool API (or endpoints that reject one)
  legacy  - plain ``complete()`` providers (test doubles, custom providers)

Live output reaches the UI through ``on_stream_chunk`` as strings with a
marker prefix:

  _delta_ <text>   answer text as it streams in
  _endstep_        the text just streamed was narration before tool calls
  _think_ <text>   model reasoning as it streams in
  _tool_ <text>    a finished tool operation ("wrote `x.py` ...")
  _out_ <text>     latest line of output from a running command
  _step_ <text>    (legacy only) a step's text, sent after the step finished

An unprefixed chunk is final-answer text (legacy mode only).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from core.context import compact_messages, trim_old_tool_results
from core.instructions import build_context_blocks
from core.permissions import CommandPolicy
from core.providers import BaseProvider, NativeToolsUnsupported, ToolCall
from core.sandbox import Sandbox, default_protected_paths
from core.skills import SkillLibrary
from core.tool_specs import ALL_TOOL_NAMES, MUTATING_TOOLS
from core.toolstate import ToolSession
from core.workspace_tools import (
    OutOfWorkspaceError,
    WorkspaceToolError,
    WorkspaceTools,
    format_tool_result,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

# Per-turn cap on model steps. This is a last-resort safety valve against a
# truly stuck/looping model, not a task-size limit - large multi-file builds
# are expected to run for many steps. If it's ever hit, whatever progress was
# made is still reported instead of being silently discarded.
MAX_TOOL_STEPS = 150

# A single tool result larger than this is cut before it enters the context.
MAX_RESULT_CHARS = 40_000

CONTINUE_PROMPT = "Continue the task using the tool result above."

# Sub-agents (the `task` tool)
SUBAGENT_MAX_STEPS = 40
SUBAGENT_TIMEOUT = 600.0
SUBAGENT_MAX_PARALLEL = 3
SUBAGENT_REPORT_CHARS = 12_000
SUBAGENT_PROMPT = (
    "You are a sub-agent working for a lead agent, not talking to the user. Complete the task below "
    "independently and finish with a concise, self-contained report: what you found or did, exact file "
    "paths and line numbers, and anything the lead should double-check. You cannot ask questions and "
    "cannot start other sub-agents. Do not repeat the task back."
)


@dataclass
class StepResult:
    text: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    streamed: bool = False
    ttft: Optional[float] = None  # seconds to first text/reasoning token


@dataclass
class Outcome:
    text: str
    failed: bool = False
    nudge: Optional[str] = None


class _ToolTagFilter:
    """Withholds streamed text that may be the start of a tool-call tag so the
    user never sees raw ``<motion_tool>`` markup flash by."""

    _TOOL_START = re.compile(
        r"^<\s*(motion_|\|\s*DSML|(?:%s)\b)" % "|".join(re.escape(n) for n in ALL_TOOL_NAMES), re.IGNORECASE
    )
    _IN_PROGRESS = re.compile(r"^<\s*[A-Za-z_|:]*$")

    def __init__(self) -> None:
        self.buf = ""
        self.suppress = False

    def feed(self, text: str) -> str:
        if self.suppress:
            return ""
        self.buf += text
        out = []
        while self.buf:
            i = self.buf.find("<")
            if i == -1:
                out.append(self.buf)
                self.buf = ""
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i:]
            if self._TOOL_START.match(self.buf):
                self.suppress = True
                self.buf = ""
                break
            if len(self.buf) < 24 and self._IN_PROGRESS.match(self.buf):
                break  # could still become a tool tag; wait for more text
            out.append("<")
            self.buf = self.buf[1:]
        return "".join(out)

    def flush(self) -> str:
        if self.suppress:
            return ""
        out, self.buf = self.buf, ""
        return out


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _empty() -> str:
    return ""


class TurnRunner:
    def __init__(
        self,
        agent: Any,
        prompt: str,
        *,
        target: str = "user",
        on_stream_chunk: Any = None,
        on_trace_event: Any = None,
        history: Optional[list] = None,
        context_query: Optional[str] = None,
        workspace: Optional[str] = None,
        agent_mode: str = "build",
        on_permission_request: Any = None,
        allowed_paths: Optional[set] = None,
        session: Optional[ToolSession] = None,
        on_ask_user: Any = None,
        on_approval: Any = None,
        on_todo: Any = None,
        images: Optional[List[Dict[str, str]]] = None,
        depth: int = 0,
        max_steps: int = MAX_TOOL_STEPS,
        extra_system: str = "",
    ) -> None:
        self.agent = agent
        self.depth = depth  # 0 = the lead agent, 1 = a sub-agent
        self.max_steps = max_steps
        self.extra_system = extra_system
        self.prompt = prompt
        self.target = target
        self.on_stream_chunk = on_stream_chunk
        self.on_trace_event = on_trace_event
        self.history = list(history or [])
        self.context_query = context_query
        self.workspace = workspace or os.getcwd()
        self.agent_mode = agent_mode
        self.on_permission_request = on_permission_request
        self.on_ask_user = on_ask_user
        self.on_approval = on_approval
        self.on_todo = on_todo
        self.images = images or []
        self.session = session or ToolSession()
        self.allowed_paths = allowed_paths
        if allowed_paths is not None:
            self.session.allowed_paths = allowed_paths
        # Only one modal prompt at a time, even when tools run in parallel.
        self.prompt_lock = asyncio.Lock()

        self.mode = "legacy"
        self.system_prompt = ""
        self.messages: List[Dict[str, Any]] = []
        self.tools: Optional[WorkspaceTools] = None
        self.used_tool = False
        self.tool_operations: List[str] = []
        self.inspection_only_loops = 0
        self.last_call_signature: Optional[tuple] = None
        self.repeat_count = 0
        self.turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._pending_images: List[Dict[str, str]] = []
        self.tool_call_count = 0
        self.first_ttft: Optional[float] = None

    # ── plumbing ─────────────────────────────────────────────────────────
    @property
    def provider(self) -> Any:
        return self.agent.provider

    @property
    def provider_type(self) -> str:
        return getattr(getattr(self.provider, "config", None), "provider_type", "unknown")

    @property
    def endpoint(self) -> str:
        return getattr(getattr(self.provider, "config", None), "endpoint", "unknown")

    async def trace(self, stage: str, message: str, **extra: Any) -> None:
        cb = self.on_trace_event
        if not cb:
            return
        payload = {"stage": stage, "message": message, **extra}
        try:
            try:
                maybe = cb(stage, payload)
            except TypeError:
                maybe = cb(payload)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            pass

    async def emit(self, text: str) -> None:
        if not self.on_stream_chunk or not text:
            return
        try:
            await _maybe_await(self.on_stream_chunk(text))
        except Exception:
            pass

    async def _emit_usage(self, label: str, usage: Optional[Dict[str, int]]) -> None:
        if usage:
            for k in self.turn_usage:
                self.turn_usage[k] += int(usage.get(k) or 0)
            await self.trace("usage", f"{label} usage", **usage)

    # ── memory / context ─────────────────────────────────────────────────
    async def _recall(self) -> str:
        agent = self.agent
        await self.trace("memory_recall_start", "Running retriever.retrieve")
        queries = [self.prompt] + ([self.context_query] if self.context_query else [])
        timeout = getattr(agent, "recall_timeout", 2.0)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(agent.retriever.retrieve(q) for q in queries), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            results = []
            await self.trace("memory_recall_timeout", f"Memory recall exceeded {timeout:.1f}s; continuing without it")
        chunks: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("memory recall failed: %s", r)
                continue
            chunks += r
        # De-duplicate by content, keep order (one retrieve() can itself
        # surface duplicate rows).
        seen = set()
        deduped = []
        for c in chunks:
            if c["content"] not in seen:
                seen.add(c["content"])
                deduped.append(c)
        deduped = deduped[:5]
        await self.trace("memory_recall_done", "Memory recall complete", chunks=len(deduped))
        return "\n".join(c["content"] for c in deduped)

    def _build_tools(self) -> WorkspaceTools:
        agent = self.agent
        policy = CommandPolicy.from_config(
            getattr(agent, "permissions_config", None), approved=self.session.approved_commands
        )

        async def approve(kind: str, subject: str, reason: str) -> str:
            if self.on_approval is None:
                return "deny"
            async with self.prompt_lock:
                try:
                    choice = await _maybe_await(self.on_approval(kind, subject, reason))
                except Exception:
                    choice = "deny"
            await self.trace(
                "permission_request",
                f"{'Approved (' + choice + ')' if choice in ('once', 'session') else 'Denied'} {kind}: {subject[:120]}",
                tool=kind,
            )
            return choice if choice in ("once", "session") else "deny"

        async def ask_user(question: str, options: List[str]) -> Any:
            if self.on_ask_user is None:
                return None
            async with self.prompt_lock:
                return await _maybe_await(self.on_ask_user(question, options))

        def on_todo(todos: List[Dict[str, Any]]) -> None:
            if self.on_todo:
                try:
                    self.on_todo(todos)
                except Exception:
                    pass

        return WorkspaceTools(
            self.workspace,
            read_only=self.agent_mode == "plan",
            allowed_paths=self.allowed_paths,
            mcp_manager=getattr(agent, "mcp_manager", None),
            session=self.session,
            policy=policy,
            ask_user=ask_user if self.on_ask_user is not None else None,
            approve=approve if self.on_approval is not None else None,
            on_todo=on_todo,
            skills=SkillLibrary.for_workspace(self.workspace),
            notes=getattr(agent, "notes", None),
            enforce_read_before_write=True,
            sandbox=Sandbox(self.workspace, getattr(agent, "sandbox_mode", "auto"), default_protected_paths()),
            subagents=self.depth == 0 and isinstance(agent.provider, BaseProvider),
        )

    def _pick_mode(self) -> str:
        provider = self.provider
        if not isinstance(provider, BaseProvider):
            return "legacy"
        return "native" if provider.native_tools else "xml"

    def _compose_system_prompt(self, memory_text: str, context_blocks: str) -> str:
        instructions = self.tools.system_instructions(native=self.mode == "native")  # type: ignore[union-attr]
        extra = f"\n\n{context_blocks}" if context_blocks else ""
        sub = f"\n\n{self.extra_system}" if self.extra_system else ""
        return f"You are Motion Agent.{sub}\n\n{instructions}{extra}\n\nMemory Context:\n{memory_text}"

    def _user_message(self) -> Dict[str, Any]:
        if not self.images:
            return {"role": "user", "content": self.prompt}
        if isinstance(self.provider, BaseProvider) and self.provider.supports_vision:
            parts: List[Dict[str, Any]] = [{"type": "text", "text": self.prompt}]
            parts += [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in self.images]
            return {"role": "user", "content": parts}
        names = ", ".join(i.get("name", "image") for i in self.images)
        return {
            "role": "user",
            "content": f"{self.prompt}\n\n[Attached image(s) {names} could not be shown: the current model "
                       "does not accept image input. Tell the user, or use OCR via run_command if available.]",
        }

    # ── the model step ───────────────────────────────────────────────────
    async def _model_step(self, step: int) -> StepResult:
        provider = self.provider
        if self.mode == "legacy":
            if step == 0:
                text = await provider.complete(
                    self.prompt, system_prompt=self.system_prompt, history=self.messages[:-1] or None
                )
            else:
                text = await provider.complete(
                    CONTINUE_PROMPT, system_prompt=self.system_prompt, history=self.messages or None
                )
            usage = getattr(provider, "last_usage", None)
            try:
                provider.last_usage = None
            except Exception:
                pass
            return StepResult(text=text or "", usage=usage)

        res = StepResult()
        tools_arg = self.tools.tool_schemas() if self.mode == "native" else None  # type: ignore[union-attr]
        tag_filter = _ToolTagFilter() if self.mode == "xml" else None
        started = time.monotonic()
        async for ev in provider.chat_stream(self.messages, system_prompt=self.system_prompt, tools=tools_arg):
            if ev.kind == "text":
                if res.ttft is None:
                    res.ttft = time.monotonic() - started
                res.text += ev.text
                shown = tag_filter.feed(ev.text) if tag_filter else ev.text
                if shown:
                    res.streamed = True
                    await self.emit("_delta_ " + shown)
            elif ev.kind == "reasoning":
                if res.ttft is None:
                    res.ttft = time.monotonic() - started
                res.reasoning += ev.text
                await self.emit("_think_ " + ev.text)
            elif ev.kind == "tool_call" and ev.tool_call:
                if res.ttft is None:
                    res.ttft = time.monotonic() - started
                res.tool_calls.append(ev.tool_call)
            elif ev.kind == "usage":
                res.usage = ev.usage
        if tag_filter:
            tail = tag_filter.flush()
            if tail:
                res.streamed = True
                await self.emit("_delta_ " + tail)
        return res

    # ── tool execution ───────────────────────────────────────────────────
    def _on_output(self, name: str):
        last = 0.0

        async def cb(_tag: str, text: str) -> None:
            nonlocal last
            now = time.monotonic()
            if now - last < 0.5:
                return
            line = next((l for l in reversed(text.strip().splitlines()) if l.strip()), "")
            if line:
                last = now
                await self.emit(f"_out_ {name}: {line.strip()[:120]}")

        return cb

    async def _ask_permission(self, path: str) -> str:
        cb = self.on_permission_request
        if not cb:
            return "deny"
        async with self.prompt_lock:
            try:
                decision = await _maybe_await(cb(path))
            except Exception:
                decision = "deny"
        return decision if decision in ("once", "session") else "deny"

    def _wrap(self, name: str, result: Optional[dict] = None, error: Optional[str] = None) -> str:
        text = format_tool_result(name, result=result, error=error, wrap=self.mode != "native")
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "…[result truncated]" + ("</motion_tool_result>" if self.mode != "native" else "")
        return text

    async def _fail(self, name: str, exc: Exception, failed_path: str) -> str:
        await self.emit(f"_tool_ `{name}` failed: {exc}")
        await self.trace(
            "tool_error", f"{name} failed on `{failed_path or '(no path)'}`: {exc}",
            tool=name, path=failed_path, error=str(exc),
        )
        return self._wrap(name, error=str(exc))

    async def _run_call(self, call: ToolCall) -> Outcome:
        name, arguments = call.name, call.arguments
        await self.trace("tool_start", f"Running {name}", tool=name, path=str(arguments.get("path", "")))
        if call.parse_error:
            text = await self._fail(name, WorkspaceToolError(call.parse_error), "")
            return Outcome(text, failed=True)

        # Plan mode rejects writes by design. Treat this as a soft policy
        # nudge rather than a tool_error, so the model can recover with a
        # real plan instead of the loop stopping on the first write attempt.
        if self.agent_mode == "plan" and name in MUTATING_TOOLS:
            await self.trace("tool_done", f"{name} blocked in plan mode", tool=name)
            return Outcome(
                self._wrap(name, error=f"{name} is disabled in plan mode"),
                nudge=(
                    "You are in read-only Plan mode - mutations/runs are disabled here. "
                    "Do not retry write_file/replace_in_file/run_command/run_script/run_python. "
                    "Respond now with a concrete written plan: proposed files/directories, the "
                    "approach for each major piece, and any libraries you'd use. The user will "
                    "review this and switch you to Build mode (Tab) to implement/execute it."
                ),
            )

        path = str(arguments.get("path", "") or "").strip()
        tools = self.tools
        assert tools is not None
        permission_retry_used = False
        while True:
            try:
                if name == "task":
                    result = await self._run_subagent(arguments)
                else:
                    result = await tools.aexecute(name, arguments, on_output=self._on_output(name))
            except OutOfWorkspaceError as exc:
                if permission_retry_used or not self.on_permission_request:
                    return Outcome(await self._fail(name, exc, str(exc.path)), failed=True)
                decision = await self._ask_permission(str(exc.path))
                approved = decision in ("once", "session")
                await self.trace(
                    "permission_request",
                    f"{'Approved (' + decision + ')' if approved else 'Denied'} out-of-workspace access to `{exc.path}`",
                    tool=name,
                    path=str(exc.path),
                )
                if not approved:
                    return Outcome(await self._fail(name, exc, str(exc.path)), failed=True)
                # "once" only affects this turn's remaining calls; "session"
                # also persists into the caller's shared set.
                tools.allowed_paths.add(exc.path)
                if decision == "session" and self.allowed_paths is not None:
                    self.allowed_paths.add(exc.path)
                permission_retry_used = True
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return Outcome(await self._fail(name, exc, path), failed=True)
            break

        path = str(result.get("path") or path or "").strip()
        operation, stream_text = self._describe(name, arguments, result, path)
        self.tool_operations.append(operation)
        # Stream every tool op (not just writes) so the UI can show live
        # step-by-step progress for the whole loop, however long it runs.
        await self.emit(f"_tool_ {stream_text}")
        await self.trace(
            "tool_done", f"Completed {name}", tool=name, path=path,
            diff=result.get("_diff", ""), created=bool(result.get("created")),
            added=result.get("lines_added", 0), removed=result.get("lines_removed", 0),
        )
        if name == "todo_write":
            await self.trace("todo_update", f"{result.get('completed', 0)}/{result.get('todos', 0)} done", tool=name)

        model_result = dict(result)
        image = model_result.pop("_image", None) if name == "read_image" else None
        if name == "read_image":
            model_result.pop("data_url", None)  # tens of thousands of useless tokens as text
            model_result["note"] = (
                "image attached below" if image and self._can_show_images()
                else "this model cannot view images; use OCR through run_command if available"
            )
        outcome = Outcome(self._wrap(name, result=model_result))
        if image and self._can_show_images():
            outcome.nudge = None
            self._pending_images.append({"name": path, "mime": image["mime"], "data": image["data"]})
        return outcome

    def _can_show_images(self) -> bool:
        return self.mode == "native" and isinstance(self.provider, BaseProvider) and self.provider.supports_vision

    @staticmethod
    def _describe(name: str, arguments: dict, result: dict, path: str) -> tuple[str, str]:
        if name == "write_file":
            delta = f", +{result.get('lines_added', 0)} −{result.get('lines_removed', 0)}" if "lines_added" in result else ""
            return f"wrote `{path}`", f"wrote `{path}` ({result.get('bytes_written', 0)} bytes{delta})"
        if name == "replace_in_file":
            op = f"updated `{path}`"
            delta = f" (+{result.get('lines_added', 0)} −{result.get('lines_removed', 0)})" if "lines_added" in result else ""
            return op, op + delta
        if name == "read_file":
            op = f"read `{path}`"
            return op, op
        if name == "list_files":
            op = f"listed `{path or '.'}`"
            return op, op
        if name == "grep":
            op = f"searched `{str(arguments.get('pattern', ''))[:40]}` ({result.get('count', 0)} matches)"
            return op, op
        if name == "run_command":
            op = f"ran `{str(arguments.get('command', '')).strip()}` (exit {result.get('exit_code')})"
            return op, op
        if name in ("run_script", "run_python"):
            return f"ran `{name}` (exit {result.get('exit_code')})", f"{name} finished (exit {result.get('exit_code')})"
        if name in ("web_fetch", "web_search"):
            op = f"`{name}` -> {result.get('status', result.get('count', ''))}"
            return op, op
        if name == "task":
            op = f"sub-agent `{str(arguments.get('description', ''))[:50]}` finished ({result.get('tool_calls', 0)} tool calls)"
            return op, op
        if name == "job_start":
            op = f"started background job `{result.get('job_id', '')}` ({str(arguments.get('command', ''))[:60]})"
            return op, op
        if name == "job_output":
            op = f"read output of `{result.get('job_id', '')}` ({result.get('lines_returned', 0)} lines, {result.get('status', '')})"
            return op, op
        if name == "job_stop":
            op = f"stopped job `{result.get('job_id', '')}`"
            return op, op
        if name == "todo_write":
            op = f"updated todo list ({result.get('completed', 0)}/{result.get('todos', 0)} done)"
            return op, op
        if name == "ask_user":
            return "asked the user", "asked the user a question"
        if name == "use_skill":
            op = f"loaded skill `{result.get('name', '')}`"
            return op, op
        if name == "mcp_call" or name.startswith("mcp__"):
            op = f"called MCP `{result.get('server', '')}/{result.get('tool', '')}`"
            return op, op
        op = f"ran `{name}`"
        return op, op

    def _parallel_ok(self, call: ToolCall) -> bool:
        """Read-only tools and read-only (explore) sub-agents may run concurrently."""
        if call.name == "task":
            return self.agent_mode == "plan" or (call.arguments.get("mode") or "explore") == "explore"
        return self.tools.is_parallel_safe(call.name)  # type: ignore[union-attr]

    async def _run_subagent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run a `task` call: a fresh, isolated turn whose final text is the report."""
        if self.depth >= 1:
            raise WorkspaceToolError("sub-agents cannot start other sub-agents")
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkspaceToolError("task needs a non-empty 'prompt'")
        label = str(arguments.get("description") or "sub-agent")[:60]
        mode = arguments.get("mode") or "explore"
        if mode not in ("explore", "general"):
            raise WorkspaceToolError("task mode must be 'explore' or 'general'")
        if self.agent_mode == "plan":
            mode = "explore"  # a read-only lead can only delegate read-only work

        sem = getattr(self.agent, "_subagent_sem", None)
        if sem is None:
            sem = self.agent._subagent_sem = asyncio.Semaphore(SUBAGENT_MAX_PARALLEL)

        # Share what the user already decided/what /undo must cover; nothing else.
        sub_session = ToolSession(
            allowed_paths=self.session.allowed_paths,
            checkpoints=self.session.checkpoints,
            read_files=self.session.read_files,
            approved_commands=self.session.approved_commands,
        )
        forwarded = {"tool_calls": 0, "steps": 0}

        def on_chunk(chunk: str) -> None:
            # Only tool progress reaches the UI (marked as belonging to the
            # sub-agent); its streamed text would corrupt the lead's live answer.
            if chunk.startswith("_tool_ "):
                forwarded["tool_calls"] += 1
                asyncio.ensure_future(self.emit(f"_tool_ ↳ [{label}] {chunk[7:]}"))

        def on_trace(stage: str, payload: Dict[str, Any]) -> None:
            if stage == "usage":  # sub-agent tokens count toward the turn's usage and cost
                asyncio.ensure_future(self.trace("usage", f"sub-agent usage ({label})", **{
                    k: payload[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in payload
                }))
            elif stage == "model_step":
                forwarded["steps"] = payload.get("step", forwarded["steps"])

        runner = TurnRunner(
            self.agent,
            prompt.strip(),
            target="user",
            on_stream_chunk=on_chunk,
            on_trace_event=on_trace,
            workspace=self.workspace,
            agent_mode="plan" if mode == "explore" else "build",
            allowed_paths=self.allowed_paths,
            session=sub_session,
            depth=1,
            max_steps=SUBAGENT_MAX_STEPS,
            extra_system=SUBAGENT_PROMPT,
        )
        async with sem:
            await self.trace("subagent_start", f"Sub-agent started: {label} ({mode})")
            try:
                report = await asyncio.wait_for(runner.run(), timeout=SUBAGENT_TIMEOUT)
            except asyncio.TimeoutError:
                raise WorkspaceToolError(f"sub-agent '{label}' timed out after {SUBAGENT_TIMEOUT:.0f}s") from None
            finally:
                await sub_session.jobs.stop_all()
            await self.trace("subagent_done", f"Sub-agent finished: {label}")
        report = (report or "").strip() or "(the sub-agent returned no report)"
        if len(report) > SUBAGENT_REPORT_CHARS:
            report = report[:SUBAGENT_REPORT_CHARS] + "\n…[report truncated]"
        return {
            "description": label, "mode": mode, "report": report,
            "tool_calls": runner.tool_call_count, "steps": forwarded["steps"],
        }

    def _note_call(self, call: ToolCall) -> None:
        """Bookkeeping for stuck-loop detection and the build-mode nudge."""
        self.used_tool = True
        self.tool_call_count += 1
        # Track loops that only inspect. If the model keeps listing/reading
        # without writing on a build-mode task, nudge it to create files.
        if call.name in {"list_files", "read_file", "grep", "glob_files"}:
            self.inspection_only_loops += 1
        else:
            self.inspection_only_loops = 0
        # Surface repeated identical calls so the trace panel makes a stuck
        # loop obvious.
        signature = (call.name, str(call.arguments.get("path", "") or call.arguments.get("command", "")))
        if signature == self.last_call_signature:
            self.repeat_count += 1
        else:
            self.repeat_count = 1
            self.last_call_signature = signature

    async def _warn_if_looping(self, call: ToolCall) -> None:
        if self.repeat_count == 3 and self.last_call_signature:
            await self.trace(
                "loop_warning",
                f"{call.name} on `{self.last_call_signature[1]}` has repeated {self.repeat_count}x in a row",
                tool=call.name,
                path=self.last_call_signature[1],
            )

    # ── the turn ─────────────────────────────────────────────────────────
    async def run(self) -> str:
        agent = self.agent
        t_turn = time.monotonic()
        if self.depth == 0:  # a sub-agent's edits join the lead's turn so /undo covers them
            self.session.checkpoints.begin_turn()
            await self.trace("turn_start", "Turn started")

        recall = asyncio.create_task(self._recall() if self.depth == 0 else _empty())
        ctx = asyncio.create_task(asyncio.to_thread(build_context_blocks, self.workspace))
        try:
            memory_text = await recall
            try:
                context_blocks = await ctx
            except Exception:
                context_blocks = ""
        finally:
            for t in (recall, ctx):
                if not t.done():
                    t.cancel()

        self.tools = self._build_tools()
        if self.tools.sandbox is not None and not self.tools.read_only:
            await self.trace("sandbox", f"Command sandbox: {self.tools.sandbox.describe()}")
        self.mode = self._pick_mode()
        self.system_prompt = self._compose_system_prompt(memory_text, context_blocks)
        self.messages = self.history + [self._user_message()]
        prompt_index = len(self.messages) - 1
        window = int(getattr(self.provider, "context_window", 32768) or 32768)

        tool_response: Optional[str] = None
        hit_cap = False
        final_streamed = False
        empty_retries = 0

        for tool_step in range(self.max_steps):
            trim_old_tool_results(self.messages)
            if compact_messages(self.messages, self.system_prompt, window, prompt_index):
                await self.trace("context_compacted", "Trimmed older tool steps to stay within the context window")

            step_started = time.monotonic()
            try:
                step = await self._model_step(tool_step)
            except NativeToolsUnsupported as exc:
                # This endpoint/model has no tool API: fall back to the text
                # protocol for the rest of the session.
                self.provider.disable_native_tools()
                await self.trace("native_tools_disabled", f"Native tool calling rejected; using text tools ({exc})")
                self.mode = "xml"
                self.system_prompt = self._compose_system_prompt(memory_text, context_blocks)
                continue
            except httpx.TimeoutException as exc:
                error_msg = (
                    f"Provider {self.provider_type} timed out ({self.endpoint}). "
                    "The model did not respond within the configured timeout."
                )
                await self.trace("provider_error", error_msg, provider=self.provider_type, error=str(exc))
                return (
                    f"⚠️ {error_msg}\n\n"
                    "I couldn't reach the model provider in time. Try again, "
                    "check your connection, or select a different provider."
                )
            except httpx.HTTPError as exc:
                error_msg = f"Provider {self.provider_type} request failed ({self.endpoint}): {exc}"
                await self.trace("provider_error", error_msg, provider=self.provider_type, error=str(exc))
                return (
                    f"⚠️ {error_msg}\n\n"
                    "I couldn't reach the model provider. Check your connection or provider status."
                )

            step_secs = time.monotonic() - step_started
            if step.ttft is not None and self.first_ttft is None:
                self.first_ttft = step.ttft
            await self._emit_usage(f"tool_step_{tool_step}", step.usage)
            await self.trace(
                "model_step",
                f"step {tool_step + 1}: {step_secs:.1f}s",
                step=tool_step + 1,
                duration_ms=int(step_secs * 1000),
                ttft_ms=int(step.ttft * 1000) if step.ttft is not None else None,
            )

            candidate = step.text
            calls: List[ToolCall]
            if self.mode == "native":
                calls = list(step.tool_calls)
            else:
                # Stream visible progress for legacy steps so the UI doesn't stay
                # blank while tools run. Strip any tool markup from previews.
                if self.mode == "legacy":
                    visible = re.sub(r"<[^>]+>", "", candidate or "").strip()
                    if visible:
                        await self.emit(f"_step_ {visible[:500]}")
                try:
                    parsed = parse_tool_call(candidate)
                except Exception as exc:
                    # Treat a malformed tool call as context, not a fatal stop.
                    # The model sees the error and can self-correct on the next
                    # turn, while the user stays in control via Esc.
                    raw_preview = (candidate or "")[:400].replace("\n", " ")
                    error_msg = f"Invalid tool call: {exc}"
                    await self.trace(
                        "tool_error", error_msg, tool="invalid", path="", error=str(exc), raw_preview=raw_preview,
                    )
                    if step.streamed:
                        await self.emit("_endstep_ ")
                    self.messages.extend([
                        {"role": "assistant", "content": candidate},
                        {"role": "user", "content": format_tool_result("invalid", error=error_msg)},
                    ])
                    continue
                calls = [ToolCall(id=f"call_{tool_step}", name=parsed[0], arguments=parsed[1])] if parsed else []

            if not calls and not (candidate or "").strip() and self.used_tool:
                empty_retries += 1
                if empty_retries <= 2:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. Continue the task: use another "
                            "tool if work remains, otherwise provide a concise completion summary."
                        ),
                    })
                    continue
                break

            if not calls:
                tool_response = candidate
                final_streamed = step.streamed
                break

            # ── tool calls ───────────────────────────────────────────────
            if step.streamed:
                await self.emit("_endstep_ ")
            if self.mode == "native":
                self.messages.append({
                    "role": "assistant",
                    "content": candidate,
                    "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                })
            else:
                self.messages.append({"role": "assistant", "content": candidate})

            for c in calls:
                self._note_call(c)
                await self._warn_if_looping(c)

            parallel = len(calls) > 1 and all(self._parallel_ok(c) for c in calls)
            if parallel:
                outcomes = list(await asyncio.gather(*(self._run_call(c) for c in calls)))
            else:
                outcomes = [await self._run_call(c) for c in calls]

            nudges: List[str] = []
            for c, out in zip(calls, outcomes):
                if self.mode == "native":
                    self.messages.append({
                        "role": "tool", "tool_call_id": c.id, "name": c.name, "content": out.text,
                    })
                else:
                    self.messages.append({"role": "user", "content": out.text})
                if out.nudge and out.nudge not in nudges:
                    nudges.append(out.nudge)
            if self._pending_images:
                imgs, self._pending_images = self._pending_images, []
                self.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Image(s) from read_image: " + ", ".join(i["name"] for i in imgs)}]
                    + [{"type": "image", "mime": i["mime"], "data": i["data"]} for i in imgs],
                })
            for n in nudges:
                self.messages.append({"role": "user", "content": n})

            if (
                self.agent_mode == "build"
                and self.inspection_only_loops >= 2
                and not any(op.startswith(("wrote ", "updated ")) for op in self.tool_operations)
                and all(not o.failed for o in outcomes)
            ):
                self.messages.append({
                    "role": "user",
                    "content": (
                        "You have inspected the workspace enough. The user asked you to create "
                        "something. Now use write_file to create the requested files with concrete, "
                        "complete content. Do not ask for clarification and do not return a script."
                    ),
                })
        else:
            hit_cap = True

        if hit_cap:
            # Never discard real progress: if tools actually ran before the cap
            # was hit, tell the user what was done and how to resume. Hitting
            # this almost always means the model is stuck looping rather than
            # that the task was too big.
            await self.trace(
                "step_cap_hit",
                f"Hit the {self.max_steps}-step safety cap",
                steps=self.max_steps,
                operations=len(self.tool_operations),
            )
            if self.tool_operations:
                completed = "\n".join(f"- {op}" for op in self.tool_operations)
                tool_response = (
                    f"Hit the internal safety limit ({self.max_steps} tool calls) before "
                    f"finishing - this usually means something got stuck. Progress so far:\n"
                    f"{completed}\n\nSay \"continue\" and I'll pick up from here."
                )
            else:
                tool_response = (
                    f"Hit the internal safety limit ({self.max_steps} tool calls) without "
                    "making any progress. Please narrow the task and try again."
                )

        raw_response = (tool_response or "").strip()
        if not raw_response and self.used_tool:
            write_operations = [op for op in self.tool_operations if op.startswith(("wrote ", "updated "))]
            if write_operations:
                raw_response = "Completed filesystem changes:\n" + "\n".join(f"- {op}" for op in write_operations)
            elif self.agent_mode == "plan":
                raw_response = (
                    "I inspected the workspace but couldn't finish a plan in the space "
                    "available. Ask me again, or narrow the scope, and I'll lay out the "
                    "file/directory approach here in Plan mode before you switch to Build."
                )
            else:
                raw_response = (
                    "I inspected the workspace but did not make any filesystem changes. "
                    "If you want me to create files, say exactly what to build and I will "
                    "use write_file to create it."
                )

        # Ensure the user always sees the final text. Streamed modes already
        # delivered it token by token.
        if raw_response and not final_streamed:
            await self.emit(raw_response)
        await self.trace(
            "model_done",
            "Agent tool loop finished" if self.used_tool else "Completion finished",
            chars=len(raw_response),
        )

        # Empty answer with no tools used: ask once more via a plain completion.
        provider = self.provider
        if not raw_response:
            await self.trace(
                "model_start", "Calling provider for completion",
                mode="stream" if self.on_stream_chunk else "oneshot", provider=self.provider_type,
            )
            history = self.history or None
            if self.on_stream_chunk and hasattr(provider, "stream_complete"):
                chunk_count = 0
                parts = []
                async for chunk in provider.stream_complete(self.prompt, system_prompt=self.system_prompt, history=history):
                    chunk_count += 1
                    parts.append(chunk)
                    await self.emit("_delta_ " + chunk if isinstance(provider, BaseProvider) else chunk)
                raw_response = "".join(parts)
                await self._emit_usage("final_stream", getattr(provider, "last_usage", None))
                await self.trace("model_done", "Streaming completion finished", stream_chunks=chunk_count, chars=len(raw_response))
            else:
                raw_response = await provider.complete(self.prompt, system_prompt=self.system_prompt, history=history) or ""
                await self._emit_usage("final_oneshot", getattr(provider, "last_usage", None))
                await self.trace("model_done", "One-shot completion finished", chars=len(raw_response))

        final_response = agent.caveman.process_outgoing(raw_response, target=self.target)
        await self.trace("finalize", "Post-processing completed", chars=len(final_response or ""))
        if self.depth == 0:
            await self.trace(
                "turn_done",
                f"Turn finished in {time.monotonic() - t_turn:.1f}s",
                elapsed_ms=int((time.monotonic() - t_turn) * 1000),
                tool_calls=self.tool_call_count,
                ttft_ms=int(self.first_ttft * 1000) if self.first_ttft is not None else None,
            )

        # Remember substantive turns so later sessions can recall them.
        if self.depth == 0 and getattr(agent, "auto_remember", False) and raw_response and (self.used_tool or len(raw_response) > 200):
            agent.schedule_remember(self.prompt, raw_response, self.tool_operations)

        # Skill crystallization (manual-first: disabled by default)
        if self.depth == 0 and agent.auto_skill_synthesis:
            from core.learning import Trajectory

            try:
                await self.trace("skill_synthesis_start", "Running skill synthesizer")
                trajectory = Trajectory(
                    task_id="single",
                    prompt=self.prompt,
                    steps=[{"tool": "model", "input": self.prompt, "output": raw_response}],
                    final_result=raw_response,
                    success=True,
                )
                skill_path = await agent.synthesizer.synthesize(trajectory)
                if skill_path:
                    logger.info(f"Skill crystallized: {skill_path}")
                    await self.trace("skill_synthesis_done", "Skill synthesized", path=skill_path)
                else:
                    await self.trace("skill_synthesis_done", "Skill synthesis skipped")
            except Exception as e:
                logger.debug(f"Skill synthesis skipped: {e}")
                await self.trace("skill_synthesis_error", f"Skill synthesis error: {e}")
        return final_response
