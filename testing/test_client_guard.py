import pytest
from _client import _check_result


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
