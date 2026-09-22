"""scripts/live_check.py: PASS/FAIL logic, exercised against stub providers (no network)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from core.providers import ModelConfig, StreamEvent, ToolCall

SPEC = importlib.util.spec_from_file_location("live_check", Path(__file__).resolve().parents[1] / "scripts" / "live_check.py")
live = importlib.util.module_from_spec(SPEC)
sys.modules["live_check"] = live
SPEC.loader.exec_module(live)


class Stub:
    supports_tools = True
    native_tools = True

    def __init__(self, *, usage=True, tools=True, system=True, secret_in_error=False):
        self.config = ModelConfig(name="stub", endpoint="http://stub", api_key="sk-SECRET", provider_type="cloud", options={"model": "m"})
        self.flags = dict(usage=usage, tools=tools, system=system, secret_in_error=secret_in_error)

    async def chat_stream(self, messages, system_prompt="", tools=None):
        if self.flags["secret_in_error"]:
            raise RuntimeError("HTTP 500 from stub (key sk-SECRET)")
        last = messages[-1]
        if tools and last["role"] == "user":
            if self.flags["tools"]:
                yield StreamEvent("tool_call", tool_call=ToolCall("c1", "echo", {"text": "ping"}))
            else:
                yield StreamEvent("text", text="I will not use tools.")
        elif tools:
            yield StreamEvent("text", text="It returned ping.")
        elif "secret word" in str(last["content"]):
            yield StreamEvent("text", text="TANGERINE" if self.flags["system"] else "I don't know")
        else:
            yield StreamEvent("text", text="PO")
            yield StreamEvent("text", text="NG hi")
        if self.flags["usage"]:
            yield StreamEvent("usage", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


def run(monkeypatch, provider, capsys):
    import asyncio

    monkeypatch.setattr("core.agent_config.make_provider_builder", lambda cm: (lambda pid: provider))
    ok = asyncio.run(live.run_provider("stub", object()))
    return ok, capsys.readouterr().out


def test_a_well_behaved_provider_passes_every_check(monkeypatch, capsys):
    ok, out = run(monkeypatch, Stub(), capsys)
    assert ok and out.count("PASS") == 4 and "FAIL" not in out and "called echo" in out


@pytest.mark.parametrize("flag,check,reason", [
    ("usage", "reports token usage", "no usable usage"),
    ("tools", "tool call round trip", "did not call the tool"),
    ("system", "honours the system prompt", "system prompt not honoured"),
])
def test_each_kind_of_misbehaviour_fails_its_own_check_with_a_reason(monkeypatch, capsys, flag, check, reason):
    ok, out = run(monkeypatch, Stub(**{flag: False}), capsys)
    line = next(l for l in out.splitlines() if check in l)
    assert not ok and "FAIL" in line and reason in line
    assert out.count("PASS") == 3                                            # the other checks still ran


def test_errors_are_reported_without_ever_printing_the_api_key(monkeypatch, capsys):
    ok, out = run(monkeypatch, Stub(secret_in_error=True), capsys)
    assert not ok and out.count("FAIL") == 4 and "RuntimeError" in out
    assert "api_key" not in out.lower() and "Authorization" not in out       # the script itself never prints config or headers


def test_an_unusable_provider_is_skipped_not_failed(monkeypatch, capsys):
    import asyncio

    monkeypatch.setattr("core.agent_config.make_provider_builder", lambda cm: (lambda pid: None))
    assert asyncio.run(live.run_provider("claude", object())) is True
    assert "SKIPPED" in capsys.readouterr().out
