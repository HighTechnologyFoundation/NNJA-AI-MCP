"""Unit tests for server.py variable resolution.

VARIABLE_ALIASES resolution short-circuits before `_fuzzy_variable_search` touches
the DataCatalog (it returns early once every variable is resolved), so these run
with no server, no catalog, and no network — only a fake dataset with a `.name`.
"""

from types import SimpleNamespace
from typing import Any

from nnja_ai.exceptions import EmptyTimeSubsetError

import server


def test_empty_time_subset_returns_empty_frame(monkeypatch):
    # An empty time subset is recoverable ("no data for this date, try another"), so
    # _access_dataset must swallow EmptyTimeSubsetError into an empty DataFrame and let
    # callers return the friendly "No data found" string, not leak a hard tool error.
    # Aliased variables ("temperature") short-circuit before any catalog/network access;
    # only _resolve_dataset needs the catalog, so a tiny fake covers the whole path.
    class FakeDataset:
        name = "conv-adpsfc-NC000001"

        def sel(self, **kwargs):
            raise EmptyTimeSubsetError()

    fake = FakeDataset()

    class FakeCatalog:
        def search(self, query):
            return [SimpleNamespace(name=fake.name)]

        def __getitem__(self, name):
            return fake

    # _catalog is only bound when the lifespan runs, so it has no value at import time.
    monkeypatch.setattr(server, "_catalog", FakeCatalog(), raising=False)

    result = server._access_dataset(fake.name, "2021-01-01", ["temperature"])

    assert result.data.empty
    assert result.var_mapping == {"temperature": "TMPSQ1.TMDB"}


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
