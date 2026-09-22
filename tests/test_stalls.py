"""A provider that keeps the connection alive without producing output must not hang a turn."""
import asyncio
import time
from pathlib import Path

import httpx
import pytest

from core.providers import CloudProvider, ModelConfig
from tests.test_agent_loop import Scripted, make_agent, run, text
from tests.test_failover import arm


class KeepaliveOnly(httpx.AsyncByteStream):
    """An SSE response that sends only comment/ping lines, forever."""

    def __init__(self, first: bytes = b""):
        self.first = first

    async def __aiter__(self):
        if self.first:
            yield self.first
        while True:
            await asyncio.sleep(0.03)
            yield b": keepalive\n\n"

    async def aclose(self):
        pass


def stalled_provider(first: bytes = b""):
    p = CloudProvider(ModelConfig(name="t", endpoint="https://api.example.com/v1", api_key="k", provider_type="cloud",
                                  options={"model": "m"}))
    p._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=KeepaliveOnly(first))),
        timeout=httpx.Timeout(30.0),                                   # a normal read timeout: pings keep resetting it
    )
    return p


async def test_keepalive_pings_do_not_keep_a_stalled_turn_alive(tmp_path: Path):
    agent = make_agent(stalled_provider())
    agent.stall_timeout = 0.4
    t0 = time.monotonic()
    resp, _, _ = await run(agent, "hi", workspace=str(tmp_path))
    assert time.monotonic() - t0 < 4                                   # without the guard this never returns
    assert "timed out" in resp


async def test_a_stall_after_the_first_words_also_fails_and_a_fallback_takes_over(tmp_path: Path):
    first = b'data: {"choices": [{"delta": {"content": "Let me th"}}]}\n\n'
    backup = Scripted([text("finished on the backup")])
    agent = arm(make_agent(stalled_provider(first)), backup)
    agent.stall_timeout = 0.4
    resp, _, traces = await run(agent, "hi", workspace=str(tmp_path))
    assert resp == "finished on the backup" and any(s == "failover" for s, _ in traces)


async def test_slow_but_steady_output_is_not_a_stall(tmp_path: Path):
    class Dribble(Scripted):
        async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
            from core.providers import StreamEvent
            for word in ["a ", "b ", "c ", "d"]:
                await asyncio.sleep(0.15)                              # 0.6s in total, each gap well under the limit
                yield StreamEvent("text", text=word)

    agent = make_agent(Dribble([]))
    agent.stall_timeout = 0.4
    resp, _, _ = await run(agent, "hi", workspace=str(tmp_path))
    assert resp == "a b c d"


async def test_zero_disables_the_guard_and_the_config_key_is_read(tmp_path: Path):
    from core.agent_config import configure_agent

    class A: pass
    assert configure_agent(A(), {}.get).stall_timeout == 180.0
    assert configure_agent(A(), {"stall_timeout": 45}.get).stall_timeout == 45.0
    assert configure_agent(A(), {"stall_timeout": 0}.get).stall_timeout == 0.0
    assert configure_agent(A(), {"stall_timeout": "junk"}.get).stall_timeout == 180.0
    agent = make_agent(Scripted([text("fine")]))
    agent.stall_timeout = 0
    assert (await run(agent, "hi", workspace=str(tmp_path)))[0] == "fine"
