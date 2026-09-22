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


# ── missing / rejected API keys ─────────────────────────────────────────────

def _keyless(endpoint, handler, monkeypatch):
    for var in ("OLLAMA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MOTION_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    p = CloudProvider(ModelConfig(name="t", endpoint=endpoint, api_key=None, provider_type="cloud", options={"model": "m"}))
    p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return p


@pytest.mark.parametrize("endpoint,env", [
    ("https://ollama.com/v1", "OLLAMA_API_KEY"),
    ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
])
async def test_a_missing_key_fails_fast_with_instructions_and_sends_nothing(endpoint, env, monkeypatch):
    sent = []
    p = _keyless(endpoint, lambda request: sent.append(request) or httpx.Response(200), monkeypatch)
    with pytest.raises(ProviderError) as exc:
        await _collect(p, [{"role": "user", "content": "x"}])
    assert sent == []                                            # no doomed request went out
    msg = str(exc.value)
    assert "No API key configured" in msg and "motion auth login" in msg and env in msg
    assert exc.value.status_code == 401


async def test_env_var_or_config_key_still_works(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "ok"}}]}))

    p = _keyless("https://ollama.com/v1", handler, monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "from-env")
    assert (await _collect(p, [{"role": "user", "content": "x"}])).text == "ok" and seen == ["Bearer from-env"]


async def test_keyless_custom_endpoints_are_not_blocked(monkeypatch):
    """A self-hosted gateway may need no key: only the well-known hosts are refused up front."""
    p = _keyless("https://gateway.internal/v1", lambda r: httpx.Response(200, text=_sse({"choices": [{"delta": {"content": "hi"}}]})), monkeypatch)
    assert (await _collect(p, [{"role": "user", "content": "x"}])).text == "hi"


async def test_a_rejected_key_says_so_and_how_to_fix_it():
    def handler(request):
        return httpx.Response(401, text='{"error":"Unauthorized"}')

    with pytest.raises(ProviderError) as exc:
        await _collect(_provider(CloudProvider, handler, endpoint="https://ollama.com/v1"), [{"role": "user", "content": "x"}])
    msg = str(exc.value)
    assert "HTTP 401 from ollama.com" in msg and "rejected the API key" in msg and "OLLAMA_API_KEY" in msg


async def test_the_agent_does_not_blame_the_network_for_a_credentials_problem(tmp_path, monkeypatch):
    from tests.test_agent_loop import make_agent, run

    for var in ("OLLAMA_API_KEY", "MOTION_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    provider = CloudProvider(ModelConfig(name="t", endpoint="https://ollama.com/v1", api_key=None, provider_type="cloud", options={"model": "m"}))
    resp, _, _ = await run(make_agent(provider), "hi", workspace=str(tmp_path))
    assert "No API key configured for ollama.com" in resp and "motion auth login" in resp
    assert "check your connection" not in resp.lower()


# ── Anthropic conversation caching ──────────────────────────────────────────

def test_conversation_cache_breakpoint_goes_on_the_last_block_only():
    from core.providers import _with_conversation_cache_breakpoint as mark

    msgs = [{"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "1", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "res"}]}]
    out = mark(msgs)
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert not any("cache_control" in str(m) for m in out[:-1])
    assert "cache_control" not in str(msgs)                                     # the input was not mutated
    as_text = mark([{"role": "user", "content": "a"}, {"role": "user", "content": "the question"}])
    assert as_text[-1]["content"] == [{"type": "text", "text": "the question", "cache_control": {"type": "ephemeral"}}]


@pytest.mark.parametrize("msgs", [
    [{"role": "user", "content": "only one message: nothing to reuse yet"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}]}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "   "}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": []}],
])
def test_no_breakpoint_where_anthropic_would_reject_or_it_cannot_pay_off(msgs):
    from core.providers import _with_conversation_cache_breakpoint as mark

    assert mark(msgs) == msgs and "cache_control" not in str(mark(msgs))


async def test_the_anthropic_request_carries_three_breakpoints_and_counts_cache_tokens():
    seen = {}

    def handler(request):
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, text=_sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 50, "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 100}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "message_delta", "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
            done=False,
        ))

    p = _provider(CloudProvider, handler, endpoint="https://api.anthropic.com")
    res = await p.chat([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}],
                       system_prompt="sys", tools=TOOLS)
    payload = seen["payload"]
    assert payload["system"][0]["cache_control"] and payload["tools"][-1]["cache_control"]
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert str(payload).count("cache_control") == 3                              # Anthropic allows 4
    assert res.usage["prompt_tokens"] == 50 + 4000 + 100                         # cached tokens still count toward prompt size


# ── a connection that closes early is not a finished answer ─────────────────

from core.providers import StreamCutError  # noqa: E402


def _raw(body: str):
    return lambda request: httpx.Response(200, text=body)


def _oai(*chunks, done=True):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + ("data: [DONE]\n\n" if done else "")


TEXT = {"choices": [{"delta": {"content": "The answer is fo"}}]}
TOOL_START = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"path": "sr'}}]}}]}
FINISH = {"choices": [{"delta": {}, "finish_reason": "stop"}]}


async def test_openai_stream_cut_mid_text_is_an_error_not_a_short_answer():
    with pytest.raises(StreamCutError) as exc:
        await _collect(_provider(CloudProvider, _raw(_oai(TEXT, done=False))), [{"role": "user", "content": "x"}])
    assert "closed before the model finished" in str(exc.value) and "incomplete" in str(exc.value)
    assert isinstance(exc.value, ProviderError) and isinstance(exc.value, httpx.HTTPError)


async def test_openai_stream_cut_mid_tool_call_never_yields_the_partial_call():
    p = _provider(CloudProvider, _raw(_oai(TOOL_START, done=False)))
    got = []
    with pytest.raises(StreamCutError):
        async for ev in p.chat_stream([{"role": "user", "content": "x"}], tools=TOOLS):
            got.append(ev)
    assert not [e for e in got if e.kind == "tool_call"]                        # nothing half-formed reaches the loop


async def test_an_empty_response_is_an_error_too():
    with pytest.raises(StreamCutError):
        await _collect(_provider(CloudProvider, _raw("")), [{"role": "user", "content": "x"}])


@pytest.mark.parametrize("body", [
    _oai({"choices": [{"delta": {"content": "hi"}}]}),                            # [DONE] only
    _oai({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}, done=False),   # finish_reason only
    _oai({"choices": [{"delta": {"content": "hi"}}]}, FINISH),                   # both
])
async def test_either_terminator_marks_an_openai_stream_complete(body):
    assert (await _collect(_provider(CloudProvider, _raw(body)), [{"role": "user", "content": "x"}])).text == "hi"


async def test_lenient_streams_accepts_gateways_that_never_send_a_terminator():
    p = _provider(CloudProvider, _raw(_oai({"choices": [{"delta": {"content": "hi"}}]}, done=False)), lenient_streams=True)
    assert (await _collect(p, [{"role": "user", "content": "x"}])).text == "hi"


async def test_local_ollama_stream_needs_its_done_marker():
    def ndjson(*items):
        return "".join(json.dumps(i) + "\n" for i in items)

    cut = ndjson({"message": {"content": "par"}, "done": False})
    ok = ndjson({"message": {"content": "full"}, "done": False}, {"message": {}, "done": True, "prompt_eval_count": 5, "eval_count": 2})
    p = _provider(LocalProvider, _raw(cut), endpoint="http://localhost:11434", ptype="local")
    with pytest.raises(StreamCutError):
        await _collect(p, [{"role": "user", "content": "x"}])
    good = _provider(LocalProvider, _raw(ok), endpoint="http://localhost:11434", ptype="local")
    assert (await _collect(good, [{"role": "user", "content": "x"}])).text == "full"


async def test_anthropic_stream_needs_message_stop_and_a_cut_tool_use_is_not_dropped_silently():
    start = {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}
    text = [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me look"}},
            {"type": "content_block_stop", "index": 0}]
    tool_open = [{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "read_file"}},
                 {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": "a'}}]
    anth = lambda events: _provider(CloudProvider, _raw(_sse(*events, done=False)), endpoint="https://api.anthropic.com")
    with pytest.raises(StreamCutError):                                            # cut mid tool_use: used to succeed with just the text
        await _collect(anth([start, *text, *tool_open]), [{"role": "user", "content": "x"}])
    with pytest.raises(StreamCutError):
        await _collect(anth([start, *text]), [{"role": "user", "content": "x"}])
    done = await _collect(anth([start, *text, {"type": "message_delta", "usage": {"output_tokens": 3}}, {"type": "message_stop"}]),
                          [{"role": "user", "content": "x"}])
    assert done.text == "Let me look"


async def test_the_agent_reports_a_cut_stream_and_a_configured_fallback_takes_over(tmp_path):
    from tests.test_agent_loop import Scripted, make_agent, run, text
    from tests.test_failover import arm

    resp, _, _ = await run(make_agent(Scripted([StreamCutError("the connection closed before the model finished (x)")])), "hi", workspace=str(tmp_path))
    assert "closed before the model finished" in resp
    backup = Scripted([text("recovered on the backup")])
    agent = arm(make_agent(Scripted([StreamCutError("the connection closed before the model finished (x)")])), backup)
    resp2, _, _ = await run(agent, "hi", workspace=str(tmp_path))
    assert resp2 == "recovered on the backup"
