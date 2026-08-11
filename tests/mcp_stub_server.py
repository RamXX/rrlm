"""A real, offline MCP server for rrlm integration tests (every transport).

Like ``stub_server.py``, this is not a mock: it is the official ``mcp`` SDK
serving in a genuine subprocess. Tests mount it through ``rrlm.mcptools`` and
the agent's REPL awaits its tools for real; determinism comes from the tools'
trivial semantics, not from patching anything.

Transports (argv): no arguments serves stdio; ``streamable-http PORT``
serves streamable HTTP at ``http://127.0.0.1:PORT/mcp`` (append
``--stateless`` for the current stateless-server style, omit for the older
stateful style); ``sse PORT`` serves the legacy HTTP+SSE transport at
``http://127.0.0.1:PORT/sse``. One tool set either way, so the same
assertions cover every transport and both protocol styles.

Tools: ``add(a, b)`` (exact arithmetic the test can verify end-to-end) and
``fail(reason)`` (always errors, to exercise the tool-error path).
"""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("rrlm-test-tools")


@server.tool(description="Add two integers and return the sum.")
def add(a: int, b: int) -> int:
    return a + b


@server.tool(description="Always fails; used to test error propagation.")
def fail(reason: str = "requested") -> str:
    raise RuntimeError(f"deliberate failure: {reason}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        server.run(transport="stdio")
    elif sys.argv[1] == "streamable-http":
        server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=int(sys.argv[2]),
            stateless_http="--stateless" in sys.argv,
        )
    elif sys.argv[1] == "sse":
        server.run(transport="sse", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        raise SystemExit(f"unknown transport argv: {sys.argv[1:]}")
