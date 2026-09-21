"""Provider wire-protocol tests using httpx.MockTransport (no network)."""
import json

import httpx
import pytest

from core.providers import (
    CloudProvider,
    LocalProvider,
    ModelConfig,
    NativeToolsUnsupported,
    ProviderError,
    ProxyProvider,
    _to_anthropic_messages,
    _to_openai_messages,
)

TOOLS = [{
    "name": "read_file",
    "description": "read",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}]


def _sse(*events: dict, done: bool = True) -> str:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return body + ("data: [DONE]\n\n" if done else "")


def _provider(cls, handler, endpoint="https://api.example.com/v1", ptype="cloud", **opts):
    p = cls(ModelConfig(name="t", endpoint=endpoint, api_key="k", provider_type=ptype, options={"model": "m", **opts}))
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p.max_retries = 2
    return p


async def _collect(provider, messages, tools=None):
    return await provider.chat(messages, system_prompt="sys", tools=tools)


# ── OpenAI-compatible ───────────────────────────────────────────────────────

async def test_openai_streams_text_reasoning_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        body = _sse(
            {"choices": [{"delta": {"reasoning_content": "hmm "}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3}},
        )
        return httpx.Response(200, text=body)

    p = _provider(CloudProvider, handler)
    res = await _collect(p, [{"role": "user", "content": "hi"}])
    assert res.text == "Hello"
    assert res.reasoning == "hmm "
    assert res.usage == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    assert p.last_usage["total_tokens"] == 14
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["stream_options"] == {"include_usage": True}
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["auth"] == "Bearer k"


async def test_openai_assembles_incremental_tool_calls():
    def handler(request):
        body = _sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"pa'}},
                {"index": 1, "id": "c2", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
        )
        return httpx.Response(200, text=body)

    res = await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "go"}], TOOLS)
    assert [(c.id, c.name, c.arguments) for c in res.tool_calls] == [
        ("c1", "read_file", {"path": "a.py"}),
        ("c2", "read_file", {"path": "b.py"}),
    ]


async def test_openai_bad_tool_json_is_reported_not_crashed():
    def handler(request):
        return httpx.Response(200, text=_sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "read_file", "arguments": "{oops"}}]}}]}
        ))

    res = await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "x"}], TOOLS)
    assert res.tool_calls[0].parse_error


async def test_openai_retries_transient_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, headers={"retry-after": "0"}, text="overloaded")
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "ok"}}]}))

    res = await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "x"}])
    assert res.text == "ok"
    assert calls["n"] == 3


async def test_openai_gives_up_after_max_retries():
    def handler(request):
        return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")

    with pytest.raises(ProviderError) as exc:
        await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "x"}])
    assert exc.value.status_code == 429
    assert isinstance(exc.value, httpx.HTTPError)  # existing handlers still catch it


async def test_openai_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError):
        await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "x"}])
    assert calls["n"] == 1


async def test_openai_adapts_to_max_completion_tokens_and_stream_options():
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if "max_tokens" in payload:
            return httpx.Response(400, text="Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'")
        if "stream_options" in payload:
            return httpx.Response(400, text="unknown field stream_options")
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "fine"}}]}))

    p = _provider(CloudProvider, handler, max_tokens=100)
    res = await _collect(p, [{"role": "user", "content": "x"}])
    assert res.text == "fine"
    assert "max_completion_tokens" in payloads[-1] and "max_tokens" not in payloads[-1]
    assert "stream_options" not in payloads[-1]


async def test_openai_tool_rejection_raises_native_tools_unsupported():
    def handler(request):
        return httpx.Response(400, text='{"error":"model does not support tools"}')

    with pytest.raises(NativeToolsUnsupported):
        await _collect(_provider(CloudProvider, handler), [{"role": "user", "content": "x"}], TOOLS)


async def test_openai_message_conversion_tool_roundtrip_and_images():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "mime": "image/png", "data": "QUJD"}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "{}"},
    ]
    out = _to_openai_messages(msgs, "")
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert out[1]["content"] is None
    assert out[1]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a"}'}
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


async def test_cloud_key_only_comes_from_matching_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = CloudProvider(ModelConfig(name="c", endpoint="https://api.anthropic.com", provider_type="cloud"))
    # The old fallback would have sent the Ollama key to Anthropic.
    assert p._api_key() == ""
    q = CloudProvider(ModelConfig(name="o", endpoint="https://ollama.com/v1", provider_type="cloud"))
    assert q._api_key() == "ollama-secret"


async def test_proxy_provider_uses_openai_wire_format():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "proxied"}}]}))

    p = _provider(ProxyProvider, handler, endpoint="https://gw.local/v1", ptype="proxy")
    assert (await _collect(p, [{"role": "user", "content": "x"}])).text == "proxied"
    assert seen["url"] == "https://gw.local/v1/chat/completions"


# ── Anthropic ───────────────────────────────────────────────────────────────

def _anthropic_events():
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 20, "cache_read_input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan..."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Reading."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tu1", "name": "read_file"}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"path":'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ' "a.py"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "usage": {"output_tokens": 42}},
        {"type": "message_stop"},
    ]


async def test_anthropic_streams_thinking_text_tools_and_usage():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, text="".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in _anthropic_events()))

    p = _provider(CloudProvider, handler, endpoint="https://api.anthropic.com", max_tokens=128000)
    res = await p.chat([{"role": "user", "content": "hi"}], system_prompt="sys", tools=TOOLS)
    assert res.text == "Reading."
    assert res.reasoning == "plan..."
    assert res.tool_calls[0].arguments == {"path": "a.py"} and res.tool_calls[0].id == "tu1"
    assert res.usage == {"prompt_tokens": 25, "completion_tokens": 42, "total_tokens": 67}
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["payload"]["stream"] is True  # required for very large max_tokens
    assert seen["payload"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen["payload"]["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert seen["payload"]["tools"][0]["input_schema"]["required"] == ["path"]
    assert seen["headers"]["x-api-key"] == "k"


async def test_anthropic_extended_thinking_drops_temperature():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, text='data: {"type":"message_stop"}\n\n')

    p = _provider(CloudProvider, handler, endpoint="https://api.anthropic.com", thinking_budget=2048, temperature=0.7)
    await p.chat([{"role": "user", "content": "x"}])
    assert seen["payload"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in seen["payload"]


async def test_anthropic_error_event_raises():
    def handler(request):
        return httpx.Response(200, text='data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n')

    p = _provider(CloudProvider, handler, endpoint="https://api.anthropic.com")
    with pytest.raises(ProviderError, match="overloaded_error"):
        await p.chat([{"role": "user", "content": "x"}])


def test_anthropic_message_conversion_merges_tool_results_and_alternates():
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "a", "name": "read_file", "arguments": {"path": "1"}},
            {"id": "b", "name": "read_file", "arguments": {"path": "2"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "one"},
        {"role": "tool", "tool_call_id": "b", "content": "two"},
        {"role": "user", "content": "nudge"},
    ]
    out = _to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    # both results + the trailing nudge share one user turn
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "text"]


def test_anthropic_conversion_never_starts_with_assistant():
    out = _to_anthropic_messages([{"role": "assistant", "content": "hi"}])
    assert out[0]["role"] == "user"


# ── Ollama (local) ──────────────────────────────────────────────────────────

async def test_ollama_streams_thinking_content_tools_usage():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        lines = [
            {"message": {"thinking": "t"}},
            {"message": {"content": "Hi"}},
            {"message": {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "x"}}}]}},
            {"done": True, "prompt_eval_count": 9, "eval_count": 4},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(l) for l in lines))

    p = _provider(LocalProvider, handler, endpoint="http://localhost:11434", ptype="local", num_ctx=4096)
    res = await p.chat([{"role": "user", "content": "x"}], system_prompt="s", tools=TOOLS)
    assert (res.text, res.reasoning) == ("Hi", "t")
    assert res.tool_calls[0].arguments == {"path": "x"}
    assert res.usage["total_tokens"] == 13
    assert seen["payload"]["options"]["num_ctx"] == 4096
    assert seen["payload"]["tools"][0]["function"]["name"] == "read_file"


# ── shared behaviours ───────────────────────────────────────────────────────

async def test_complete_and_stream_complete_wrappers_still_work():
    def handler(request):
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {"content": "b"}}]}))

    p = _provider(CloudProvider, handler)
    assert await p.complete("q", system_prompt="s", history=[{"role": "user", "content": "prev"}]) == "ab"
    assert [c async for c in p.stream_complete("q")] == ["a", "b"]


async def test_native_tools_flag_and_downgrade():
    p = CloudProvider(ModelConfig(name="t", endpoint="https://x/v1", provider_type="cloud"))
    assert p.native_tools is True
    p.disable_native_tools()
    assert p.native_tools is False
    q = CloudProvider(ModelConfig(name="t", endpoint="https://x/v1", provider_type="cloud", options={"native_tools": False}))
    assert q.native_tools is False


def test_vision_and_context_window_capabilities():
    def mk(**o):
        return CloudProvider(ModelConfig(name="t", endpoint="https://x/v1", provider_type="cloud", options=o))

    assert mk(model="claude-sonnet-5").supports_vision
    assert not mk(model="deepseek-v4-flash").supports_vision
    assert mk(model="deepseek-v4-flash", vision=True).supports_vision
    assert mk(context_window=200000).context_window == 200000
    assert mk().context_window == 32768
