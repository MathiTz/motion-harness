"""Non-interactive one-shot mode: ``motion -p "prompt"``.

For scripts and CI. Runs a single turn and prints the result; there is no UI,
so nothing can be approved interactively: risky commands, private-network
fetches and ``ask_user`` are refused (allow specific commands ahead of time
with ``permissions.commands.allow`` in config.yml).

Output formats
  text         final answer on stdout (progress with --verbose goes to stderr)
  json         one JSON object on stdout when the turn ends
  stream-json  newline-delimited JSON events as they happen, ending with the
               same result object

Exit codes: 0 success, 1 the turn failed (provider/tool-loop error),
2 usage/config error, 130 interrupted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, TextIO

from core.pricing import turn_cost
from core.toolstate import ToolSession

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
FORMATS = ("text", "json", "stream-json")


class HeadlessUsageError(Exception):
    """Bad arguments or configuration (exit code 2)."""


def build_agent(provider_id: Optional[str]):
    """Create a MotionAgent from config.yml the same way the TUI does."""
    from core.config import ConfigManager
    from core.providers import ModelConfig
    from main import MotionAgent

    cm = ConfigManager()
    pid = provider_id or cm.get_default_provider()
    try:
        cfg = cm.get_provider_config(pid)
    except ValueError as exc:
        raise HeadlessUsageError(str(exc)) from exc
    model_config = ModelConfig(
        name=cfg.get("name", pid),
        endpoint=cfg["endpoint"],
        api_key=cfg.get("api_key"),
        provider_type=cfg.get("provider_type", "cloud"),
        options=cfg.get("options", {}),
    )
    servers = (cm.get("mcp") or {}).get("servers") or {}
    mcp = None
    if servers:
        from core.mcp import MCPManager

        mcp = MCPManager(servers)
    agent = MotionAgent(model_config, mcp_manager=mcp)
    agent.permissions_config = cm.data
    agent.sandbox_mode = str(cm.get("sandbox", "auto"))
    # One-shot runs (often in CI) shouldn't write to the long-term memory DB
    # unless the user opts in explicitly.
    agent.auto_remember = bool(cm.get("remember_turns_headless", False))
    try:
        agent.recall_timeout = float(cm.get("recall_timeout", 2.0))
    except (TypeError, ValueError):
        pass
    agent.provider_id = pid  # type: ignore[attr-defined]
    return agent


async def run_headless(
    prompt: str,
    *,
    provider_id: Optional[str] = None,
    workspace: Optional[str] = None,
    plan: bool = False,
    output_format: str = "text",
    verbose: bool = False,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
    agent_factory: Optional[Callable[[], Any]] = None,
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    if output_format not in FORMATS:
        raise HeadlessUsageError(f"--output-format must be one of {', '.join(FORMATS)}")
    if not (prompt or "").strip():
        raise HeadlessUsageError("empty prompt")
    workspace = os.path.abspath(workspace or os.getenv("MOTION_WORKSPACE") or os.getcwd())
    if not os.path.isdir(workspace):
        raise HeadlessUsageError(f"workspace is not a directory: {workspace}")

    agent = agent_factory() if agent_factory else build_agent(provider_id)
    provider_cfg = getattr(agent.provider, "config", None)
    stats: Dict[str, Any] = {
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "steps": 0, "tool_calls": 0, "ttft_s": None, "elapsed_s": None, "error": None,
    }

    def event(obj: Dict[str, Any]) -> None:
        if output_format == "stream-json":
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()

    def on_chunk(chunk: str) -> None:
        if chunk.startswith("_delta_ "):
            event({"type": "text", "text": chunk[8:]})
        elif chunk.startswith("_think_ "):
            event({"type": "reasoning", "text": chunk[8:]})
        elif chunk.startswith("_tool_ "):
            text = chunk[7:].strip()
            event({"type": "tool", "text": text})
            if verbose:
                err.write(f"• {text}\n")
                err.flush()
        elif chunk.startswith("_endstep_"):
            event({"type": "step_end"})

    def on_trace(stage: str, payload: Dict[str, Any]) -> None:
        if stage == "usage":
            for k in stats["usage"]:
                stats["usage"][k] += int(payload.get(k) or 0)
        elif stage == "model_step":
            stats["steps"] = payload.get("step", stats["steps"])
            if stats["ttft_s"] is None and payload.get("ttft_ms") is not None:
                stats["ttft_s"] = payload["ttft_ms"] / 1000
        elif stage == "turn_done":
            stats["elapsed_s"] = payload.get("elapsed_ms", 0) / 1000
            stats["tool_calls"] = payload.get("tool_calls", 0)
        elif stage == "provider_error":
            stats["error"] = payload.get("message") or "provider error"
        elif stage in ("sandbox", "permission_request") and verbose:
            err.write(f"[{stage}] {payload.get('message', '')}\n")
            err.flush()

    session = ToolSession()
    ok, result = True, ""
    try:
        if agent.mcp_manager is not None:
            try:
                await asyncio.wait_for(agent.mcp_manager.initialize_all(), timeout=20)
            except Exception as exc:  # MCP is optional; never block the run on it
                if verbose:
                    err.write(f"[mcp] setup failed: {exc}\n")
        result = await agent.run(
            prompt,
            target="user",
            on_stream_chunk=on_chunk,
            on_trace_event=on_trace,
            workspace=workspace,
            agent_mode="plan" if plan else "build",
            session=session,
        )
        if stats["error"]:
            ok = False
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        ok, stats["error"] = False, f"{type(exc).__name__}: {exc}"
    finally:
        try:  # background jobs must not outlive the run
            await asyncio.wait_for(session.jobs.stop_all(), timeout=5)
        except Exception:
            pass
        for closer in (
            lambda: agent.provider.close(),
            lambda: agent.mcp_manager.close_all() if agent.mcp_manager is not None else None,
        ):
            try:
                r = closer()
                if r is not None:
                    await asyncio.wait_for(r, timeout=3)
            except Exception:
                pass
        try:
            agent.memory.close()
        except Exception:
            pass

    usage = stats["usage"]
    cost = turn_cost(
        getattr(provider_cfg, "provider_type", "cloud"),
        getattr(provider_cfg, "options", {}) or {},
        usage["prompt_tokens"], usage["completion_tokens"],
    )
    summary = {
        "type": "result",
        "ok": ok,
        "result": result if ok else (result or ""),
        "error": stats["error"],
        "provider": getattr(agent, "provider_id", getattr(provider_cfg, "name", "")),
        "model": (getattr(provider_cfg, "options", {}) or {}).get("model"),
        "workspace": workspace,
        "mode": "plan" if plan else "build",
        "steps": stats["steps"],
        "tool_calls": stats["tool_calls"],
        "elapsed_s": stats["elapsed_s"],
        "ttft_s": stats["ttft_s"],
        "usage": usage,
        "cost_usd": cost,
    }
    if output_format in ("json", "stream-json"):
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")
        out.flush()
    else:
        if ok:
            out.write((result or "").rstrip("\n") + "\n")
            out.flush()
        else:
            err.write((stats["error"] or result or "the turn failed") + "\n")
            err.flush()
    return EXIT_OK if ok else EXIT_FAILED


def main_headless(args: Any, stdin: Optional[TextIO] = None) -> int:
    """Entry point used by main.py; turns exceptions into exit codes."""
    stdin = stdin or sys.stdin
    prompt = args.prompt
    try:
        if prompt == "-":
            prompt = stdin.read()
        elif getattr(args, "stdin", False):
            extra = stdin.read()
            if extra.strip():
                prompt = f"{prompt}\n\n<stdin>\n{extra}\n</stdin>"
        return asyncio.run(run_headless(
            prompt or "",
            provider_id=args.provider,
            workspace=args.workspace,
            plan=args.plan,
            output_format=args.output_format,
            verbose=args.verbose,
        ))
    except HeadlessUsageError as exc:
        print(f"motion: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
