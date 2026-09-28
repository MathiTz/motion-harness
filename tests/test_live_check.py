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

    def __init__(self, *, usage=True, tools=True, system=True, secret_in_error=False, implausible_usage=False):
        self.config = ModelConfig(name="stub", endpoint="http://stub", api_key="sk-SECRET", provider_type="cloud", options={"model": "m"})
        self.flags = dict(usage=usage, tools=tools, system=system, secret_in_error=secret_in_error, implausible_usage=implausible_usage)

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
            # Scales with the real request size so check_usage_plausibility (which sends ~1800
            # known chars) passes the same way a real provider would, unless implausible_usage
            # deliberately reports a fixed, too-small count regardless of input size.
            prompt_tokens = 10 if self.flags["implausible_usage"] else max(10, len(str(messages)) // 4)
            yield StreamEvent("usage", usage={"prompt_tokens": prompt_tokens, "completion_tokens": 5, "total_tokens": prompt_tokens + 5})


def run(monkeypatch, provider, capsys):
    """Runs every check except check_failover_classification: that one deliberately makes its own
    real network connection (it exists to classify a REAL vendor error - see the manual live run
    recorded in docs/compatibility.md), which this Stub-based "no network" file must not trigger
    just by exercising a DIFFERENT check's pass/fail wiring. It has its own dedicated,
    fully-mocked tests below."""
    import asyncio

    monkeypatch.setattr(live, "CHECKS", [c for c in live.CHECKS if c[1] is not live.check_failover_classification])
    monkeypatch.setattr("core.agent_config.make_provider_builder", lambda cm: (lambda pid: provider))
    ok = asyncio.run(live.run_provider("stub", object()))
    return ok, capsys.readouterr().out


def test_a_well_behaved_provider_passes_every_check(monkeypatch, capsys):
    ok, out = run(monkeypatch, Stub(), capsys)
    assert ok and out.count("PASS") == len(live.CHECKS) and "FAIL" not in out and "called echo" in out
    assert "closed cleanly" in out and "plausible" in out


@pytest.mark.parametrize("flag,bad_value,check,reason", [
    ("usage", False, "reports token usage", "no usable usage"),
    ("tools", False, "tool call round trip", "did not call the tool"),
    ("system", False, "honours the system prompt", "system prompt not honoured"),
    ("implausible_usage", True, "usage reporting accuracy", "implausible"),
])
def test_each_kind_of_misbehaviour_fails_its_own_check_with_a_reason(monkeypatch, capsys, flag, bad_value, check, reason):
    ok, out = run(monkeypatch, Stub(**{flag: bad_value}), capsys)
    line = next(l for l in out.splitlines() if check in l)
    assert not ok and "FAIL" in line and reason in line
    assert "PASS" in out                                                     # unrelated checks still ran


def test_errors_are_reported_without_ever_printing_the_api_key(monkeypatch, capsys):
    ok, out = run(monkeypatch, Stub(secret_in_error=True), capsys)
    assert not ok and out.count("FAIL") == len(live.CHECKS) and "RuntimeError" in out
    assert "api_key" not in out.lower() and "Authorization" not in out       # the script itself never prints config or headers


def test_an_unusable_provider_is_skipped_not_failed(monkeypatch, capsys):
    import asyncio

    monkeypatch.setattr("core.agent_config.make_provider_builder", lambda cm: (lambda pid: None))
    assert asyncio.run(live.run_provider("claude", object())) is True
    assert "SKIPPED" in capsys.readouterr().out


# ── check_failover_classification: logic only, no real network ─────────────
# (the check's whole point is classifying a REAL vendor error - see the manual live run recorded
# in docs/compatibility.md; this only proves the classification/assertion logic itself is correct)

class _RaisingProvider:
    def __init__(self, config, exc):
        self.config = config
        self._exc = exc

    async def chat_stream(self, messages, system_prompt="", tools=None):
        raise self._exc
        yield  # pragma: no cover - never reached, makes this an async generator

    async def close(self):
        pass


def test_failover_check_passes_when_the_real_error_is_failover_worthy(monkeypatch):
    import asyncio

    from core.providers import ProviderError

    exc = ProviderError("unauthorized", status_code=401)
    monkeypatch.setattr(
        "core.providers.ProviderFactory.get_provider",
        staticmethod(lambda cfg: _RaisingProvider(cfg, exc)),
    )
    detail = asyncio.run(live.check_failover_classification(Stub()))
    assert "ProviderError" in detail and "401" in detail and "correctly classified" in detail


def test_failover_check_fails_loudly_when_a_real_error_would_be_missed(monkeypatch):
    """If _failover_worthy ever regressed to miss a real 401, this must FAIL, not silently pass -
    the whole point of this check is catching exactly that gap."""
    import asyncio

    from core.providers import ProviderError

    exc = ProviderError("bad request", status_code=400)  # never failover-worthy, by design
    monkeypatch.setattr(
        "core.providers.ProviderFactory.get_provider",
        staticmethod(lambda cfg: _RaisingProvider(cfg, exc)),
    )
    with pytest.raises(AssertionError, match="NOT classified as failover-worthy"):
        asyncio.run(live.check_failover_classification(Stub()))
