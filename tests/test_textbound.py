"""Head-plus-tail bounding of model-bound text (core/textbound.py and its call sites).

A plain `text[:limit]` cut keeps the start of a command's output and drops the end, which is where
the error, the summary line and (in a result's JSON) `stderr`, the exit status and the `truncated`
flag live. Three separate head-only cuts existed: per output stream, on the whole serialized tool
result, and on older results kept in history.
"""
import asyncio
import json
import tracemalloc
from pathlib import Path

import pytest

from core.agent_loop import MAX_RESULT_CHARS, TurnRunner
from core.context import trim_old_tool_results
from core.textbound import OutputCapture, bound_text
from core.workspace_tools import COMMAND_OUTPUT_LIMIT, WorkspaceTools


def test_short_text_is_untouched():
    assert bound_text("hello", 100) == "hello"
    assert bound_text("x" * 100, 100) == "x" * 100


def test_long_text_keeps_both_ends_and_never_exceeds_the_limit():
    text = "HEAD" + "m" * 50_000 + "TAIL: error at the end"
    out = bound_text(text, 2_000)
    assert len(out) <= 2_000
    assert out.startswith("HEAD") and out.endswith("TAIL: error at the end")
    assert "characters omitted" in out


def test_omitted_count_is_accurate():
    text = "".join(chr(97 + i % 26) for i in range(10_000))
    out = bound_text(text, 1_000)
    head, rest = out.split("\n…[", 1)
    omitted = int(rest.split(" ", 1)[0])
    tail = rest.split("]…\n", 1)[1]
    assert len(head) + omitted + len(tail) == len(text)
    assert text.startswith(head) and text.endswith(tail)


@pytest.mark.parametrize("limit", [10, 50, 200])
def test_tiny_limits_still_respect_the_limit(limit):
    assert len(bound_text("z" * 5_000, limit)) <= limit


def test_streaming_capture_matches_the_one_shot_result():
    text = "".join(f"line {i}\n" for i in range(5_000))
    cap = OutputCapture(3_000)
    for i in range(0, len(text), 777):  # arbitrary chunking must not change the result
        cap.feed(text[i:i + 777])
    assert cap.truncated and cap.render() == bound_text(text, 3_000)


def test_streaming_capture_below_the_limit_returns_everything():
    cap = OutputCapture(1_000)
    for piece in ("a" * 300, "b" * 300, "c" * 300):
        cap.feed(piece)
    assert not cap.truncated and cap.render() == "a" * 300 + "b" * 300 + "c" * 300


def test_streaming_capture_memory_stays_bounded():
    cap = OutputCapture(20_000)
    tracemalloc.start()
    for _ in range(2_000):
        cap.feed("y" * 10_000)  # 20 MB in total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert cap.total == 20_000_000 and peak < 1_000_000


async def test_command_error_at_the_end_of_long_output_reaches_the_model(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    result = await tools.aexecute(
        "run_command", {"command": "seq 1 40000; echo 'FATAL: build failed at the very end' 1>&2; exit 2"}
    )
    assert result["exit_code"] == 2 and result["truncated"]
    assert result["stdout"].startswith("1\n") and result["stdout"].rstrip().endswith("40000")
    assert "FATAL: build failed" in result["stderr"]
    assert len(result["stdout"]) <= COMMAND_OUTPUT_LIMIT


async def test_python_snippet_output_is_bounded_from_both_ends_too(tmp_path: Path):
    result = await WorkspaceTools(tmp_path).aexecute(
        "run_python", {"code": "print('START'); print('x' * 60000); print('END')"}
    )
    assert "START" in result["stdout"] and "END" in result["stdout"] and result["truncated"]


def test_the_whole_result_cap_keeps_stderr_and_the_truncated_flag():
    """stderr and `truncated` come after stdout in the serialized result, so the old head-only cut on the
    whole text dropped them first exactly when stdout was large."""
    runner = TurnRunner.__new__(TurnRunner)
    runner.mode = "native"
    result = {"exit_code": 1, "stdout": "o" * 30_000, "stderr": "Traceback ... ValueError: boom", "truncated": True}
    text = runner._wrap("run_command", result=result)
    assert len(text) <= MAX_RESULT_CHARS
    assert "ValueError: boom" in text and '"truncated": true' in text.lower().replace(" ", " ")
    assert text.startswith("{")


def test_the_whole_result_cap_keeps_the_xml_wrapper_closed_for_text_protocol_models():
    runner = TurnRunner.__new__(TurnRunner)
    runner.mode = "legacy"
    text = runner._wrap("run_command", result={"exit_code": 0, "stdout": "o" * 30_000, "stderr": "", "truncated": True})
    assert text.startswith("<motion_tool_result>") and text.endswith("</motion_tool_result>")
    assert len(text) <= MAX_RESULT_CHARS + len("</motion_tool_result>")


def test_older_tool_results_keep_their_tail_when_trimmed():
    body = json.dumps({"name": "run_command", "ok": True, "result": {"stdout": "S" * 5_000, "stderr": "FINAL ERROR"}})
    msgs = [{"role": "tool", "content": body}, {"role": "tool", "content": "recent 1"}, {"role": "tool", "content": "recent 2"}]
    trim_old_tool_results(msgs, keep_recent=2, max_chars=600)
    trimmed = msgs[0]["content"]
    assert trimmed.startswith('{"name": "run_command"') and "FINAL ERROR" in trimmed and "trimmed" in trimmed
    assert len(trimmed) < len(body) // 4
