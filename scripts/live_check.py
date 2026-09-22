#!/usr/bin/env python
"""Live check of a provider's wire format: does it stream, report usage, call tools and take tool results back?

The Anthropic and OpenAI adapters are covered by mock-transport tests, which prove the code matches the
documented formats but not that the real services still behave that way. Run this with a real key:

    python scripts/live_check.py                       # the default provider from config.yml / .env
    python scripts/live_check.py claude                # a configured provider id (ollama-cloud/glm-5.2, openai, ...)
    python scripts/live_check.py --all                 # every provider that has a key

It sends four tiny requests (a few hundred tokens in total) and prints PASS/FAIL per check with the reason.
Exit code 0 only if every check passed. API keys are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ECHO_TOOL = [{
    "name": "echo",
    "description": "Echo back the given text.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}]


async def collect(provider: Any, messages: List[Dict[str, Any]], system: str = "", tools: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"text": "", "calls": [], "usage": None, "events": 0}
    async for ev in provider.chat_stream(messages, system_prompt=system, tools=tools):
        out["events"] += 1
        if ev.kind == "text":
            out["text"] += ev.text
        elif ev.kind == "tool_call" and ev.tool_call:
            out["calls"].append(ev.tool_call)
        elif ev.kind == "usage":
            out["usage"] = ev.usage
    return out


async def check_streaming(p: Any) -> str:
    r = await collect(p, [{"role": "user", "content": "Reply with exactly the single word: PONG"}])
    assert "PONG" in r["text"].upper(), f"unexpected reply: {r['text'][:80]!r}"
    assert r["events"] >= 2, f"expected a stream of several events, got {r['events']}"
    return f"{r['events']} stream events"


async def check_usage(p: Any) -> str:
    r = await collect(p, [{"role": "user", "content": "Say hi."}])
    u = r["usage"]
    assert u and u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens", 0) > 0, f"no usable usage: {u}"
    return f"prompt={u['prompt_tokens']} completion={u['completion_tokens']}"


async def check_system_prompt(p: Any) -> str:
    r = await collect(p, [{"role": "user", "content": "What is the secret word?"}], system="The secret word is TANGERINE. Answer in one word.")
    assert "TANGERINE" in r["text"].upper(), f"system prompt not honoured: {r['text'][:80]!r}"
    return "system prompt honoured"


async def check_tool_round_trip(p: Any) -> str:
    if not getattr(p, "supports_tools", True) or not getattr(p, "native_tools", True):
        return "skipped: this provider uses the text tool protocol"
    messages: List[Dict[str, Any]] = [{"role": "user", "content": "Use the echo tool with text 'ping', then tell me what it returned."}]
    first = await collect(p, messages, tools=ECHO_TOOL)
    assert first["calls"], f"model did not call the tool (said {first['text'][:80]!r})"
    call = first["calls"][0]
    assert call.name == "echo" and not call.parse_error, f"bad call: {call.name} {call.arguments} {call.parse_error}"
    messages.append({"role": "assistant", "content": first["text"], "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}]})
    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": '{"echoed": "ping"}'})
    second = await collect(p, messages, tools=ECHO_TOOL)
    assert second["text"].strip(), "no final answer after the tool result was returned"
    return f"called echo({call.arguments}) then answered"


CHECKS: List[Tuple[str, Callable[[Any], Awaitable[str]]]] = [
    ("streams text", check_streaming),
    ("reports token usage", check_usage),
    ("honours the system prompt", check_system_prompt),
    ("tool call round trip", check_tool_round_trip),
]


async def run_provider(provider_id: str, cm: Any) -> bool:
    from core.agent_config import make_provider_builder

    provider = make_provider_builder(cm)(provider_id)
    if provider is None:
        print(f"\n{provider_id}: SKIPPED (unknown id, or no API key: `motion auth login` / set its env var)")
        return True
    opts = provider.config.options
    print(f"\n{provider_id}  ({provider.config.provider_type}, model {opts.get('model')}, {provider.config.endpoint})")
    ok = True
    try:
        for name, fn in CHECKS:
            t0 = time.monotonic()
            try:
                detail = await asyncio.wait_for(fn(provider), 90)
                print(f"  PASS  {name:<28} {detail}  ({time.monotonic() - t0:.1f}s)")
            except AssertionError as exc:
                ok = False
                print(f"  FAIL  {name:<28} {exc}")
            except Exception as exc:  # network / HTTP errors: show the (key-free) message
                ok = False
                print(f"  FAIL  {name:<28} {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        close = getattr(provider, "close", None)
        if close:
            try:
                await close()
            except Exception:
                pass
    return ok


async def main_async(args: argparse.Namespace) -> int:
    from main import _load_dotenv
    from core.config import ConfigManager

    _load_dotenv(REPO)
    cm = ConfigManager()
    if args.all:
        ids = [pid for pid in cm.list_providers() if pid != "default" and cm.has_api_key(pid)]
    else:
        ids = [args.provider or cm.get_default_provider()]
    if not ids:
        print("No providers with an API key found. Run `motion auth login`.")
        return 2
    results = [await run_provider(pid, cm) for pid in ids]
    print("\nAll checks passed." if all(results) else "\nSome checks FAILED: the adapter and the live service disagree.")
    return 0 if all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Live wire-format check for a provider")
    parser.add_argument("provider", nargs="?", help="provider id (default: the configured default)")
    parser.add_argument("--all", action="store_true", help="check every provider that has an API key")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
