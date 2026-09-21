"""MCP (Model Context Protocol) client support.

Servers are declared in config.yml under an ``mcp:`` block:

    mcp:
      servers:
        github:
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-github"]
          env: { GITHUB_PERSONAL_ACCESS_TOKEN: "..." }
        docs:
          url: "https://example.com/mcp"        # Streamable HTTP transport
          headers: { Authorization: "Bearer ..." }

Each server is connected once (stdio subprocess or HTTP) and kept alive. Its
tools are discovered with ``tools/list`` and exposed to the model as native
tools named ``mcp__<server>__<tool>`` with the server's own JSON schema (and,
for models without tool calling, through the generic ``mcp_call`` envelope).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
MAX_RESULT_CHARS = 30_000
_SECRET_ENV = re.compile(r"(?i)(_API_KEY|_SECRET|_PASSWORD|_SECRET_KEY)$|^(API_KEY|SECRET|PASSWORD)$")


class MCPError(ValueError):
    """Raised when an MCP operation fails."""


class MCPServer:
    """A single configured MCP server."""

    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.command = cfg.get("command")
        self.args = cfg.get("args", [])
        self.env = dict(cfg.get("env", {}))
        self.url = cfg.get("url")
        self.headers = dict(cfg.get("headers", {}))

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"

    def build_cmd(self) -> List[str]:
        if not self.command:
            raise MCPError(f"server {self.name} is missing 'command' for stdio transport")
        parts = shlex.split(self.command) if isinstance(self.command, str) else [str(self.command)]
        return parts + [str(a) for a in self.args]


class _StdioConnection:
    """Newline-delimited JSON-RPC over a persistent subprocess."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._next_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader: Optional[asyncio.Task] = None

    async def start(self) -> None:
        cmd = self.server.build_cmd()
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
        env.update({k: str(v) for k, v in self.server.env.items()})
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
        except FileNotFoundError as e:
            raise MCPError(f"binary not found for {self.server.name}: {cmd[0]}") from e
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg and ("result" in msg or "error" in msg):
                    fut = self._pending.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                elif "id" in msg and "method" in msg:
                    # Server->client request (ping, roots/list, ...): answer politely.
                    reply: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"]}
                    if msg["method"] == "ping":
                        reply["result"] = {}
                    else:
                        reply["error"] = {"code": -32601, "message": "method not supported"}
                    await self._send(reply)
        finally:
            err = MCPError(f"MCP server {self.server.name} closed its connection")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None or self.proc.returncode is not None:
            raise MCPError(f"MCP server {self.server.name} is not running")
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        await self.proc.stdin.drain()

    async def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> Any:
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as e:
            self._pending.pop(rid, None)
            raise MCPError(f"MCP {self.server.name}/{method} timed out after {timeout:.0f}s") from e
        if "error" in msg:
            err = msg["error"] or {}
            raise MCPError(f"MCP error {err.get('code')}: {err.get('message')}")
        return msg.get("result")

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), 3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class _HttpConnection:
    """Streamable-HTTP transport: each JSON-RPC message is a POST; the reply is
    JSON or an SSE stream carrying the response."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self._client = httpx.AsyncClient(timeout=30.0)
        self._next_id = 0
        self._session_id: Optional[str] = None
        self.alive = True

    async def start(self) -> None:
        return None

    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
        h.update({k: str(v) for k, v in self.server.headers.items()})
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    async def _post(self, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
        try:
            resp = await self._client.post(self.server.url, json=payload, headers=self._headers(), timeout=timeout)
        except httpx.HTTPError as e:
            raise MCPError(f"MCP {self.server.name} HTTP error: {e}") from e
        if resp.status_code >= 400:
            raise MCPError(f"MCP {self.server.name} HTTP {resp.status_code}: {resp.text[:200]}")
        self._session_id = resp.headers.get("mcp-session-id") or self._session_id
        if "id" not in payload or resp.status_code == 202 or not resp.content:
            return None
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    try:
                        msg = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    if isinstance(msg, dict) and msg.get("id") == payload["id"]:
                        return msg
            raise MCPError(f"MCP {self.server.name}: no response in event stream")
        return resp.json()

    async def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> Any:
        self._next_id += 1
        msg = await self._post(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}, timeout
        )
        if msg is None:
            return None
        if "error" in msg:
            err = msg["error"] or {}
            raise MCPError(f"MCP error {err.get('code')}: {err.get('message')}")
        return msg.get("result")

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        await self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, 30.0)

    async def close(self) -> None:
        self.alive = False
        await self._client.aclose()


def _sanitize(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", part)


class MCPManager:
    """Owns the configured MCP servers, their live connections and tool lists."""

    def __init__(self, servers_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.servers: Dict[str, MCPServer] = {}
        self._conns: Dict[str, Any] = {}
        self._tools: Dict[str, List[Dict[str, Any]]] = {}
        self._native_names: Dict[str, Tuple[str, str]] = {}
        self._lock: Optional[asyncio.Lock] = None
        self.errors: Dict[str, str] = {}
        if servers_cfg:
            for name, cfg in servers_cfg.items():
                self.servers[name] = MCPServer(name, cfg)

    def has_servers(self) -> bool:
        return bool(self.servers)

    # ── connection management ────────────────────────────────────────────
    async def _connect(self, name: str) -> Any:
        server = self.servers.get(name)
        if not server:
            raise MCPError(f"unknown MCP server: {name}")
        conn = self._conns.get(name)
        if conn is not None and conn.alive:
            return conn
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            conn = self._conns.get(name)
            if conn is not None and conn.alive:
                return conn
            conn = _HttpConnection(server) if server.transport == "http" else _StdioConnection(server)
            await conn.start()
            try:
                await conn.request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "motion-harness", "version": "1"},
                    },
                    timeout=30,
                )
                await conn.notify("notifications/initialized")
            except Exception:
                await conn.close()
                raise
            self._conns[name] = conn
            return conn

    async def initialize_all(self) -> None:
        """Connect to every configured server concurrently and cache its tools.
        A failing server is recorded in ``self.errors`` and skipped."""

        async def one(name: str) -> None:
            try:
                self._tools[name] = await self.list_tools(name)
                self.errors.pop(name, None)
            except Exception as e:
                self.errors[name] = str(e)
                logger.warning("MCP server %s init failed: %s", name, e)

        await asyncio.gather(*(one(n) for n in list(self.servers)))
        self._rebuild_native_names()

    async def list_tools(self, name: str) -> List[Dict[str, Any]]:
        conn = await self._connect(name)
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):  # pagination guard
            result = await conn.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += (result or {}).get("tools", [])
            cursor = (result or {}).get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, name: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        conn = await self._connect(name)
        result = await conn.request("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=60)
        result = result or {}
        parts: List[str] = []
        for block in result.get("content") or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                parts.append(f"[image {block.get('mimeType', '')} omitted]")
            elif block.get("type") == "resource":
                res = block.get("resource") or {}
                parts.append(res.get("text") or f"[resource {res.get('uri', '')}]")
        text = "\n".join(parts)
        if not text and result.get("structuredContent") is not None:
            text = json.dumps(result["structuredContent"])
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "…[truncated]"
        if result.get("isError"):
            raise MCPError(text or "tool reported an error")
        return {"text": text}

    def run_call(self, name: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise MCPError("MCP tools run through the async agent loop; use call_tool()")

    async def close_all(self) -> None:
        for conn in list(self._conns.values()):
            try:
                await conn.close()
            except Exception:
                pass
        self._conns.clear()

    # ── exposure to the model ────────────────────────────────────────────
    def _rebuild_native_names(self) -> None:
        self._native_names = {}
        for server, tools in self._tools.items():
            for tool in tools:
                base = f"mcp__{_sanitize(server)}__{_sanitize(tool.get('name', ''))}"[:64]
                name, n = base, 2
                while name in self._native_names:  # disambiguate truncation collisions
                    name = f"{base[:60]}_{n}"
                    n += 1
                self._native_names[name] = (server, tool.get("name", ""))

    def native_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        lookup = {v: k for k, v in self._native_names.items()}
        for server, tools in self._tools.items():
            for tool in tools:
                native = lookup.get((server, tool.get("name", "")))
                if not native:
                    continue
                schemas.append({
                    "name": native,
                    "description": f"[MCP:{server}] {tool.get('description', '')}"[:1000],
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                })
        return schemas

    def resolve_native_name(self, name: str) -> Optional[Tuple[str, str]]:
        return self._native_names.get(name)

    def tool_index(self) -> List[Tuple[str, str, str]]:
        return [
            (server, t.get("name", ""), (t.get("description") or "").replace("\n", " "))
            for server, tools in self._tools.items()
            for t in tools
        ]

    def mcp_call_instruction(self) -> str:
        if not self.servers:
            return ""
        names = ", ".join(sorted(self.servers))
        return (
            f"\nAdditional tools are available from MCP servers ({names}). Call them with:\n"
            f'- mcp_call: {{"server": "<name>", "tool": "<tool>", "arguments": {{...}}}} — invokes a '
            f"tool exposed by an external MCP server"
        )
