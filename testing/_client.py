"""Shared MCP client setup for the testing/ scripts

Each test calls `run_tool(...)`
The server must be in HTTP mode:
    uv run testing/test_linear_trend.py"""

import asyncio
from time import perf_counter
from typing import Any

from fastmcp import Client

DEFAULT_URL = "http://localhost:8000/mcp"

async def _call(tool: str, args: dict[str, Any], url:str) -> Any:
    client = Client(DEFAULT_URL)
    async with client:
        print(f"Connected: {client.is_connected()}")
        return await client.call_tool(tool, args)

def run_tool(tool: str, args: dict[str, Any], *, url: str = DEFAULT_URL) -> Any:
    """Connect, call one tool, print timing + result, and return it."""
    start = perf_counter()
    result = asyncio.run(_call(tool, args, url))
    print(f"Time taken: {perf_counter() - start:.2f} seconds")
    print(result)
    return result
