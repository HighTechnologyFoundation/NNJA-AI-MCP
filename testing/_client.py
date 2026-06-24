"""Shared MCP client setup for the testing/ scripts

Each test calls `run_tool(...)`
The server must be in HTTP mode:
    uv run testing/test_linear_trend.py
"""

import asyncio
from time import perf_counter
from typing import Any

from fastmcp import Client

DEFAULT_URL = "http://localhost:8000/mcp"


async def _call(tool: str, args: dict[str, Any], url: str) -> Any:
    client = Client(url)
    try:
        async with client:
            print(f"Connected: {client.is_connected()}")
            return await client.call_tool(tool, args)
    except RuntimeError as e:
        if "failed to connect" in str(e):
            raise SystemExit(
                f"Could not reach the MCP server at {url}. Start it in HTTP mode first:\n"
                "- PowerShell: $env:MCP_TRANSPORT='http'; uv run server.py\n"
                "- bash/zsh:   MCP_TRANSPORT=http uv run server.py\n"
                "- Docker:     docker build -t nnja-ai-mcp .\n"
                "              docker run -p 8000:8000 nnja-ai-mcp"
            ) from e


def _check_result(tool: str, payload: Any, duration: float) -> None:
    """Raise SystemExit if the tool failed or returned nothing."""
    if isinstance(payload, str) and payload.startswith("Error"):
        raise SystemExit(f"{tool} errored after {duration:.2f}s: {payload}")
    if payload is None:
        raise SystemExit(f"{tool} returned no result after {duration:.2f}s")


def run_tool(tool: str, args: dict[str, Any], *, url: str = DEFAULT_URL) -> Any:
    """Connect, call one tool, print timing + result, and return it."""
    start = perf_counter()
    result = asyncio.run(_call(tool, args, url))
    duration = perf_counter() - start
    _check_result(tool, result.data, duration)
    print(f"Time taken: {duration:.2f} seconds")
    print(result)
    return result
