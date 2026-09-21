"""Latency guards: streaming must be visible before a step finishes, and the
loop's own overhead must stay small (catches accidental quadratic work)."""
import asyncio
import time
from pathlib import Path

from core.providers import BaseProvider, ModelConfig, StreamEvent, ToolCall
from tests.test_agent_loop import EmptyRetriever, Scripted, call, make_agent, run, text


class SlowStream(BaseProvider):
    """Reasons for 0.1s, then goes quiet for 0.6s before answering."""

    def __init__(self):
        super().__init__(ModelConfig(name="s", endpoint="http://x", provider_type="local", options={"model": "m"}))

    async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
        await asyncio.sleep(0.1)
        yield StreamEvent("reasoning", text="thinking...")
        await asyncio.sleep(0.6)
        yield StreamEvent("text", text="answer")


async def test_first_output_is_visible_long_before_the_step_completes(tmp_path: Path):
    agent = make_agent(SlowStream())
    t0 = time.monotonic()
    first = {}

    def on_chunk(c):
        first.setdefault(c.split(" ")[0], time.monotonic() - t0)

    await agent.run("q", on_stream_chunk=on_chunk, workspace=str(tmp_path))
    total = time.monotonic() - t0
    assert first["_think_"] < 0.35  # user sees reasoning almost immediately...
    assert total > 0.65              # ...instead of waiting for the whole step
    assert first["_delta_"] > first["_think_"]


async def test_loop_overhead_stays_small_over_many_steps(tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello\n" * 50)
    steps = [call(str(i), "read_file", path="f.txt") for i in range(120)] + [text("done")]
    p = Scripted(steps, window=10_000_000)
    t0 = time.monotonic()
    resp, _, _ = await run(make_agent(p), workspace=str(tmp_path))
    elapsed = time.monotonic() - t0
    assert resp == "done"
    assert elapsed < 6.0, f"120 no-op steps took {elapsed:.1f}s - the loop itself got slow"


async def test_batched_reads_cost_one_round_trip_not_n(tmp_path: Path):
    for i in range(6):
        (tmp_path / f"{i}.txt").write_text(str(i))

    class OneShot(Scripted):
        async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
            await asyncio.sleep(0.2)  # per-request model latency
            async for ev in super().chat_stream(messages, system_prompt, tools):
                yield ev

    reads = [("%d" % i, "read_file", {"path": f"{i}.txt"}) for i in range(6)]
    batched = OneShot([
        [StreamEvent("tool_call", tool_call=ToolCall(cid, n, a)) for cid, n, a in reads],
        text("done"),
    ])
    t0 = time.monotonic()
    await run(make_agent(batched), workspace=str(tmp_path))
    batched_time = time.monotonic() - t0
    assert len(batched.requests) == 2 and batched_time < 0.7  # 2 model round trips total, not 7
