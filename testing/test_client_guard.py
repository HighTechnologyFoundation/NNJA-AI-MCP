import _client
import pytest
from _client import _check_result
from _fakes import sync


def test_none_payload_raises():
    with pytest.raises(SystemExit):
        _check_result("some_tool", None, 0.1)


def test_error_string_raises():
    with pytest.raises(SystemExit):
        _check_result("some_tool", "Error: no dataset found", 0.1)


def test_valid_payload_passes():
    _check_result("some_tool", {"trend": 1.2}, 0.1)


def test_zero_is_valid():
    _check_result("some_tool", 0.0, 0.1)


def _fake_client_raising(message):
    """A fastmcp.Client stand-in whose call_tool raises RuntimeError(message)."""

    class _C:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def is_connected(self):
            return True

        async def call_tool(self, tool, args):
            raise RuntimeError(message)

    return _C


@sync
async def test_call_reraises_unrelated_runtime_error(monkeypatch):
    # A RuntimeError that isn't a connection failure must propagate, not be swallowed.
    monkeypatch.setattr(_client, "Client", _fake_client_raising("unrelated boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await _client._call("any_tool", {}, "http://x/mcp")


@sync
async def test_call_converts_connection_failure_to_system_exit(monkeypatch):
    # A connection failure is turned into the friendly "start the server" SystemExit.
    msg = "Client failed to connect: All connection attempts failed"
    monkeypatch.setattr(_client, "Client", _fake_client_raising(msg))
    with pytest.raises(SystemExit, match="Could not reach"):
        await _client._call("any_tool", {}, "http://x/mcp")
