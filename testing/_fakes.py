"""Lightweight fakes for unit-testing the MCP client without a live server or Gemini.

Imported by `test_gateway.py` / `test_handlers.py` the same way `test_client_guard.py`
imports `_client` (pytest puts the `testing/` dir on `sys.path`).
"""

import asyncio
import functools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import mcp.types as types
from pydantic import AnyUrl


def sync(fn):
    """Run an `async def test_...` body via asyncio.run (no pytest-asyncio needed).

    Without this, a bare `async def test_...` is collected, returns a coroutine that
    is never awaited, and passes trivially without running its assertions.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


# Result builders (shape mirrors what mcp.ClientSession returns)


def text_contents(text, mime="application/json", uri="data://x"):
    """A read_resource result whose single content block is text."""
    return SimpleNamespace(
        contents=[types.TextResourceContents(uri=AnyUrl(uri), mimeType=mime, text=text)]
    )


def blob_contents(blob="aGk=", uri="data://x"):
    """A read_resource result whose single content block is non-text (a blob)."""
    return SimpleNamespace(
        contents=[
            types.BlobResourceContents(
                uri=AnyUrl(uri), mimeType="application/octet-stream", blob=blob
            )
        ]
    )


def resource(name, uri):
    return types.Resource(uri=AnyUrl(uri), name=name)


def prompt(name, args=()):
    """A prompt; `args` is an iterable of (name, required) tuples."""
    return types.Prompt(
        name=name,
        arguments=[types.PromptArgument(name=a, required=req) for a, req in args],
    )


def text_message(role, text):
    return types.PromptMessage(
        role=role, content=types.TextContent(type="text", text=text)
    )


# Dependency stand-ins


def fake_session(
    *,
    prompts=None,
    resources=None,
    messages=None,
    read_return=None,
    read_side_effect=None,
):
    """An AsyncMock stand-in for mcp.ClientSession (the MCPGateway dependency)."""
    s = SimpleNamespace()
    s.list_prompts = AsyncMock(return_value=SimpleNamespace(prompts=prompts or []))
    s.list_resources = AsyncMock(
        return_value=SimpleNamespace(resources=resources or [])
    )
    s.get_prompt = AsyncMock(return_value=SimpleNamespace(messages=messages or []))
    if read_side_effect is not None:
        s.read_resource = AsyncMock(side_effect=read_side_effect)
    else:
        s.read_resource = AsyncMock(return_value=read_return)
    s.call_tool = AsyncMock()
    return s


def stub_gateway(
    *, resources=None, read=None, prompts=None, messages=None, tool_result=None
):
    """A stand-in for MCPGateway (the GeminiQueryHandler dependency)."""
    g = SimpleNamespace()
    g.list_resources = AsyncMock(return_value=resources or [])
    g.read_resource = AsyncMock(return_value=read)
    g.list_prompts = AsyncMock(return_value=prompts or [])
    g.get_prompt = AsyncMock(return_value=messages or [])
    g.client_session = SimpleNamespace(call_tool=AsyncMock(return_value=tool_result))
    return g


def bare_handler(gateway):
    """A GeminiQueryHandler with only `.mcp` set, skipping Gemini client init.

    The query-pipeline helpers (`_extract_resources`, `_process_command`) depend only
    on the gateway, so bypassing __init__ isolates them from the Gemini SDK / API key.
    """
    from mcp_client.handlers import GeminiQueryHandler

    h = GeminiQueryHandler.__new__(GeminiQueryHandler)
    h.mcp = gateway
    return h
