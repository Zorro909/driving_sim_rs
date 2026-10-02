"""Minimal client for the game's MCP server (tools/AltdMcp)."""
import json, sys, urllib.request

URL = "http://127.0.0.1:47800/mcp"
_id = 0

def call(name, args=None, timeout=330):
    global _id
    _id += 1
    body = {"jsonrpc": "2.0", "id": _id, "method": "tools/call",
            "params": {"name": name, "arguments": args or {}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    if "error" in res:
        raise RuntimeError(f"{name}: {res['error']}")
    result = res["result"]
    if result.get("isError"):
        raise RuntimeError(f"{name}: {result['content'][0]['text']}")
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])

if __name__ == "__main__":
    print(json.dumps(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else None), indent=1))
