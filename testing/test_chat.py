"""Unit test for ChatSession's Ctrl-C handling.

A cancelled in-flight query must return to the prompt gracefully — `_respond`
catches CancelledError, tells the user how to exit, and restores the SIGINT
handler — instead of crashing the REPL.

This drives the cancellation *path* directly: it invokes the installed SIGINT
handler rather than delivering a real OS signal (faithful signal delivery is
OS-specific and flaky to test, especially on Windows). It does not assert that
the OS delivers SIGINT — only that, once a cancel is triggered, the REPL recovers.
"""

import asyncio
import io
import signal
from contextlib import redirect_stdout
from types import SimpleNamespace
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

    session = ChatSession.__new__(ChatSession)  # skip prompt_toolkit setup
    session.handler = SimpleNamespace(process_query=blocking_query)

    original = signal.getsignal(signal.SIGINT)
    captured = io.StringIO()
    handler_after = None
    try:
        with patch.object(chat, "_show_thinking", quiet_spinner):
            respond = asyncio.create_task(session._respond("hello"))

            # query is running, so _respond has installed its handler
            await started.wait()
            installed = signal.getsignal(signal.SIGINT)
            assert installed is not original  # _respond swapped SIGINT in

            with redirect_stdout(captured):
                # simulate Ctrl-C
                installed(signal.SIGINT, None)

                # must return, not raise/hang
                await asyncio.wait_for(respond, timeout=5)

            # read before the defensive restore
            handler_after = signal.getsignal(signal.SIGINT)
    finally:
        # never leak handler state to other tests
        signal.signal(signal.SIGINT, original)

    assert "cancel" in captured.getvalue().lower()  # the user saw the exit hint
    assert handler_after is original  # _respond restored the SIGINT handler itself
