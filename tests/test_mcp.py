"""MCP client tests against a tiny stdio echo server."""
import sys
from pathlib import Path

import pytest

from core.mcp import MCPError, MCPManager
from core.providers import BaseProvider, ModelConfig, StreamEvent, ToolCall
from main import MotionAgent

SERVER = str(Path(__file__).parent / "mcp_echo_server.py")
CFG = {"echo": {"command": sys.executable, "args": [SERVER]}}


async def test_discovers_tools_and_calls_them_over_a_persistent_connection():
    mgr = MCPManager(CFG)
    try:
        await mgr.initialize_all()
        assert mgr.errors == {}
        assert {t[1] for t in mgr.tool_index()} == {"echo", "add", "boom"}
        assert (await mgr.call_tool("echo", "echo", {"text": "hi"}))["text"] == "hi"
        assert (await mgr.call_tool("echo", "add", {"a": 2, "b": 3}))["text"] == "5"
        conn = mgr._conns["echo"]
        await mgr.call_tool("echo", "echo", {"text": "again"})
        assert mgr._conns["echo"] is conn  # same process reused, not respawned per call
    finally:
        await mgr.close_all()


async def test_tool_errors_and_unknown_servers_raise_mcp_error():
    mgr = MCPManager(CFG)
    try:
        with pytest.raises(MCPError, match="kaboom"):
            await mgr.call_tool("echo", "boom", {})
        with pytest.raises(MCPError, match="unknown"):
            await mgr.call_tool("nope", "x", {})
    finally:
        await mgr.close_all()


async def test_failing_server_is_recorded_not_fatal():
    mgr = MCPManager({"bad": {"command": "definitely-not-a-binary-xyz"}, **CFG})
    try:
        await mgr.initialize_all()
        assert "bad" in mgr.errors and "echo" not in mgr.errors
        assert {t[0] for t in mgr.tool_index()} == {"echo"}
    finally:
        await mgr.close_all()


async def test_native_schemas_are_namespaced_with_the_servers_own_schema():
    mgr = MCPManager(CFG)
    try:
        await mgr.initialize_all()
        schemas = {s["name"]: s for s in mgr.native_tool_schemas()}
        assert set(schemas) == {"mcp__echo__echo", "mcp__echo__add", "mcp__echo__boom"}
        assert schemas["mcp__echo__add"]["parameters"]["required"] == ["a", "b"]
        assert schemas["mcp__echo__echo"]["description"].startswith("[MCP:echo]")
        assert mgr.resolve_native_name("mcp__echo__add") == ("echo", "add")
        assert mgr.resolve_native_name("mcp__echo__zzz") is None
    finally:
        await mgr.close_all()


async def test_server_that_dies_is_restarted_on_next_call():
    mgr = MCPManager(CFG)
    try:
        await mgr.initialize_all()
        mgr._conns["echo"].proc.kill()
        await mgr._conns["echo"].proc.wait()
        assert (await mgr.call_tool("echo", "echo", {"text": "back"}))["text"] == "back"
    finally:
        await mgr.close_all()


class _Scripted(BaseProvider):
    def __init__(self, steps):
        super().__init__(ModelConfig(name="s", endpoint="http://x", provider_type="local", options={"model": "m"}))
        self.steps, self.requests = list(steps), []

    async def chat_stream(self, messages, system_prompt="", tools=None, **kw):
        self.requests.append({"messages": list(messages), "tools": tools, "system": system_prompt})
        for ev in self.steps.pop(0):
            yield ev


class _NoRecall:
    async def retrieve(self, q, top_k=5):
        return []


async def test_agent_can_call_mcp_tools_natively_and_via_mcp_call(tmp_path):
    mgr = MCPManager(CFG)
    try:
        await mgr.initialize_all()
        agent = MotionAgent(ModelConfig(name="t", endpoint="http://x", provider_type="local"), memory_path=":memory:", mcp_manager=mgr)
        agent.retriever = _NoRecall()
        provider = _Scripted([
            [StreamEvent("tool_call", tool_call=ToolCall("1", "mcp__echo__add", {"a": 40, "b": 2}))],
            [StreamEvent("tool_call", tool_call=ToolCall("2", "mcp_call", {"server": "echo", "tool": "echo", "arguments": {"text": "x"}}))],
            [StreamEvent("text", text="done")],
        ])
        agent.provider = provider
        resp = await agent.run("add", workspace=str(tmp_path), agent_mode="build")
        assert resp == "done"
        assert "mcp__echo__add" in {t["name"] for t in provider.requests[0]["tools"]}
        results = [m for m in provider.requests[2]["messages"] if m["role"] == "tool"]
        assert '"text": "42"' in results[0]["content"] and '"untrusted": true' in results[0]["content"]
        assert '"text": "x"' in results[1]["content"]
    finally:
        await mgr.close_all()


# ── Streamable-HTTP transport ───────────────────────────────────────────────

async def test_http_transport_handles_json_and_sse_replies_and_session_ids():
    import json

    import httpx

    seen = {"methods": [], "sessions": [], "auth": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["methods"].append(body["method"])
        seen["sessions"].append(request.headers.get("mcp-session-id"))
        seen["auth"].append(request.headers.get("authorization"))
        mid = body.get("id")
        if body["method"] == "initialize":
            return httpx.Response(
                200, headers={"mcp-session-id": "sess-1"},
                json={"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2024-11-05", "capabilities": {}}},
            )
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        if body["method"] == "tools/list":  # plain JSON
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "ping", "description": "p", "inputSchema": {"type": "object", "properties": {}}}]}})
        # tools/call answers as an SSE stream with an unrelated event first
        events = [
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}},
            {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": "pong"}]}},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="".join(f"event: message\ndata: {json.dumps(e)}\n\n" for e in events),
        )

    mgr = MCPManager({"remote": {"url": "https://mcp.example/rpc", "headers": {"Authorization": "Bearer t"}}})
    from core.mcp import _HttpConnection

    real_init = _HttpConnection.__init__

    def patched_init(self, server):
        real_init(self, server)
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    _HttpConnection.__init__ = patched_init
    try:
        await mgr.initialize_all()
        assert mgr.errors == {} and [t[1] for t in mgr.tool_index()] == ["ping"]
        assert (await mgr.call_tool("remote", "ping", {}))["text"] == "pong"
    finally:
        _HttpConnection.__init__ = real_init
        await mgr.close_all()
    assert seen["methods"] == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert seen["sessions"][0] is None and set(seen["sessions"][1:]) == {"sess-1"}  # session id echoed back
    assert set(seen["auth"]) == {"Bearer t"}
