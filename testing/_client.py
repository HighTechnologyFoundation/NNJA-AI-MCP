"""Shared MCP client setup for the testing/ scripts

Each test calls `run_tool(...)`
The server must be in HTTP mode:
    uv run testing/test_linear_trend.py"""

import asyncio
from time import perf_counter
from typing import Any

from fastmcp import Client

DEFAULT_URL = "http://localhost:8000/mcp"


async def _call(tool: str, args: dict[str, Any], url: str) -> Any:
    client = Client(url)
    async with client:
        print(f"Connected: {client.is_connected()}")
        return await client.call_tool(tool, args)


def run_tool(tool: str, args: dict[str, Any], *, url: str = DEFAULT_URL) -> Any:
    """Connect, call one tool, print timing + result, and return it."""
    start = perf_counter()
    result = asyncio.run(_call(tool, args, url))
    duration = perf_counter() - start

    payload = result.data
    if isinstance(payload, str) and payload.startswith("Error"):
        raise SystemExit(f"{tool} errored after {duration:.2f}s: {payload}")

    print(f"Time taken: {duration:.2f} seconds")
    print(result)
    return result
