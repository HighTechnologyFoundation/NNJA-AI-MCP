"""Unit tests for CLI argument parsing, logging, and elicitation handling."""

import contextlib
import logging
import signal
import sys
from types import SimpleNamespace

from _fakes import sync

from mcp_client.__main__ import configure_logging
from mcp_client.cli import parse_args
from mcp_client.mcp_client import MCPClient


def test_verbose_flag_sets_true(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mcp-client", "--members", "-v"])
    assert parse_args().verbose is True


def test_verbose_defaults_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mcp-client", "--members"])
    assert parse_args().verbose is False


def test_configure_logging_toggles_mcp_client_debug():
    pkg = logging.getLogger("mcp_client")
    original = pkg.level
    try:
        configure_logging(verbose=True)
        assert logging.getLogger("mcp_client.handlers").isEnabledFor(logging.DEBUG)

        configure_logging(verbose=False)
        assert not logging.getLogger("mcp_client.handlers").isEnabledFor(logging.DEBUG)
    finally:
        pkg.setLevel(original)  # don't leak global logger state to other tests


def _client_answering(monkeypatch, answer):
    """An MCPClient whose elicitation prompt returns `answer` (or raises it, if it is an
    exception, to simulate Ctrl-C/Ctrl-D), with no real terminal."""

    class FakePromptSession:
        async def prompt_async(self, _message):
            if isinstance(answer, BaseException):
                raise answer
            return answer

    monkeypatch.setattr("mcp_client.mcp_client.PromptSession", FakePromptSession)
    monkeypatch.setattr("mcp_client.mcp_client.patch_stdout", contextlib.nullcontext)
    return MCPClient("server.py")


@sync
async def test_elicitation_accepts_on_yes(monkeypatch):
    # Whitespace/case are normalized before the yes-check.
    client = _client_answering(monkeypatch, "  YES ")
    params = SimpleNamespace(message="This will load ~3 GB. Proceed?")

    result = await client._handle_elicitation(None, params)

    assert result.action == "accept"


@sync
async def test_elicitation_declines_on_anything_else(monkeypatch):
    # Anything that isn't an explicit yes (here, an empty answer) declines.
    client = _client_answering(monkeypatch, "")
    params = SimpleNamespace(message="This will load ~3 GB. Proceed?")

    result = await client._handle_elicitation(None, params)

    assert result.action == "decline"


@sync
async def test_elicitation_cancels_on_interrupt(monkeypatch):
    # Ctrl-C at the prompt must return "cancel" (not crash the receive loop), clear the
    # spinner-pause flag, and re-raise SIGINT so the query's handler cancels the turn.
    # raise_signal is stubbed: with no query SIGINT handler installed here, a real one
    # would hit the default handler and abort the test.
    raised = []
    monkeypatch.setattr(signal, "raise_signal", raised.append)
    client = _client_answering(monkeypatch, KeyboardInterrupt())
    params = SimpleNamespace(message="This will load ~3 GB. Proceed?")

    result = await client._handle_elicitation(None, params)

    assert result.action == "cancel"
    assert not client._elicitation_active.is_set()
    assert raised == [signal.SIGINT]
