from _client import _check_result


def raises(payload):
    try:
        _check_result("t", payload, 0.0)
        return False
    except SystemExit:
        return True


assert raises(None)
assert raises("Error: x")
assert not raises(0.0)  # falsy but valid
assert not raises({"k": 1})
print("guard tests passed")
