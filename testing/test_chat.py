"""Unit test for ChatSession's Ctrl-C handling.

A cancelled in-flight query must return to the prompt gracefully — `_respond`
catches CancelledError, tells the user how to exit, and restores the SIGINT
handler — instead of crashing the REPL.

It fires SIGINT with `signal.raise_signal` — the portable, in-process way to drive
a signal handler in a test, without an interactive Ctrl-C or `os.kill` (whose SIGINT
handling is awkward on Windows). It checks that the REPL recovers when SIGINT fires;
it does not exercise an interactive terminal Ctrl-C.
"""

import asyncio
import io
import signal
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from _fakes import sync

import mcp_client.chat as chat
from mcp_client.chat import ChatSession


@sync
async def test_ctrl_c_cancels_query_and_restores_handler():
    started = asyncio.Event()

    async def blocking_query(_query):
        started.set()
        await asyncio.Event().wait()  # never completes -> query stays "in flight"

    async def quiet_spinner(*_args, **_kwargs):
        await asyncio.sleep(3600)  # silent, cancellable stand-in for the spinner

    # Build a real session so every attribute __init__ sets is present, stubbing only
    # _build_session — the lone piece that needs a terminal. Avoids re-breaking when
    # __init__ gains new attributes (e.g. pause_spinner, console).
    handler: Any = SimpleNamespace(process_query=blocking_query, mcp=None)
    with patch.object(ChatSession, "_build_session", lambda _self: None):
        session = ChatSession(handler)

    original = signal.getsignal(signal.SIGINT)
    captured = io.StringIO()
    handler_after = None
    try:
        with patch.object(chat, "_show_thinking", quiet_spinner):
            respond = asyncio.create_task(session._respond("hello"))

            # query is running, so _respond has installed its handler
            await started.wait()
            # _respond swapped in its own SIGINT handler
            assert signal.getsignal(signal.SIGINT) is not original

            with redirect_stdout(captured):
                # fire SIGINT in-process: runs _respond's handler -> cancels the query
                signal.raise_signal(signal.SIGINT)

                # must return, not raise/hang
                await asyncio.wait_for(respond, timeout=5)

            # read before the defensive restore
            handler_after = signal.getsignal(signal.SIGINT)
    finally:
        # never leak handler state to other tests
        signal.signal(signal.SIGINT, original)

    assert "cancel" in captured.getvalue().lower()  # the user saw the exit hint
    assert handler_after is original  # _respond restored the SIGINT handler itself


@sync
async def test_dispatch_local_handles_bare_slash():
    """A lone '/' names no command -> return False (fall through), never IndexError.

    Regression for CLI B1: `query[1:].split()[0]` raised IndexError on a bare '/',
    which surfaced as a confusing "list index out of range" instead of the input
    simply being treated as "not a local command".
    """
    handler: Any = SimpleNamespace(mcp=None)
    with patch.object(ChatSession, "_build_session", lambda _self: None):
        session = ChatSession(handler)

    called = False

    async def fake_handler():
        nonlocal called
        called = True

    session.local_commands = {"refresh": {"description": "x", "handler": fake_handler}}

    # The bug: bare "/" (and "/" + whitespace) must not raise; they name no command.
    assert await session._dispatch_local("/") is False
    assert await session._dispatch_local("/   ") is False
    # A non-slash line and an unknown command both fall through without dispatching.
    assert await session._dispatch_local("hello") is False
    assert await session._dispatch_local("/nope") is False
    assert called is False
    # A real local command still dispatches -- the guard didn't break the happy path.
    assert await session._dispatch_local("/refresh") is True
    assert called is True
