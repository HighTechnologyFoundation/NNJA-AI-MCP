"""Unit tests for server.py variable resolution.

VARIABLE_ALIASES resolution short-circuits before `_fuzzy_variable_search` touches
the DataCatalog (it returns early once every variable is resolved), so these run
with no server, no catalog, and no network — only a fake dataset with a `.name`.
"""

from types import SimpleNamespace
from typing import Any

import server


def test_brightness_temperature_resolves_via_alias():
    # "brightness temperature" (spaced) must hit the underscore-keyed alias via the
    # space->underscore normalization, not fall through to fuzzy matching (B4).
    dataset: Any = SimpleNamespace(name="amsua-1bamua-NC021023")

    result = server._fuzzy_variable_search(dataset, ["brightness temperature"])

    assert result == {"brightness temperature": "BRITCSTC.TMBR_00001"}


def test_alias_lookup_normalizes_spaces():
    # An existing underscore alias ("wind_speed") should also match spaced input.
    dataset: Any = SimpleNamespace(name="conv-adpsfc-NC000001")

    result = server._fuzzy_variable_search(dataset, ["wind speed"])

    assert result == {"wind speed": "WNDSQ1.WSPD"}
