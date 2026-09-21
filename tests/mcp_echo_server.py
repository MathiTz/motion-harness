"""A minimal MCP stdio server used by the test-suite (not a test itself)."""
import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo text back",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {"name": "boom", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
]


def reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(mid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "1"}})
    elif method == "tools/list":
        reply(mid, {"tools": TOOLS})
    elif method == "tools/call":
        params = msg.get("params", {})
        name, args = params.get("name"), params.get("arguments", {})
        if name == "echo":
            reply(mid, {"content": [{"type": "text", "text": args.get("text", "")}]})
        elif name == "add":
            reply(mid, {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]})
        elif name == "boom":
            reply(mid, {"content": [{"type": "text", "text": "kaboom"}], "isError": True})
        else:
            reply(mid, error={"code": -32602, "message": "unknown tool"})
    elif mid is not None:
        reply(mid, error={"code": -32601, "message": "method not found"})
