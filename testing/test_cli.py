"""Unit tests for CLI argument parsing and logging configuration."""

import logging
import sys

from mcp_client.__main__ import configure_logging
from mcp_client.cli import parse_args


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
