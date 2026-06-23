"""Unit tests for server.py variable resolution.

VARIABLE_ALIASES resolution short-circuits before `_fuzzy_variable_search` touches
the DataCatalog (it returns early once every variable is resolved), so these run
with no server, no catalog, and no network — only a fake dataset with a `.name`.
"""

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from _fakes import sync
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


def _fake_ctx(*, supports: bool, action: str = "accept") -> Any:
    """A minimal Context stand-in: a capability check plus an async elicit."""

    async def elicit(_message, **_kwargs):
        return SimpleNamespace(action=action)

    session = SimpleNamespace(check_client_capability=lambda _capability: supports)
    return SimpleNamespace(session=session, elicit=elicit)


@sync
async def test_large_load_gate_skips_without_elicitation_support(monkeypatch):
    # No elicitation capability -> short-circuit before estimating or prompting.
    calls = []
    monkeypatch.setattr(
        server, "_estimate_query_mb", lambda *a: calls.append(a) or (10_000.0, 50)
    )
    ctx = _fake_ctx(supports=False, action="decline")  # would raise if it prompted

    await server._confirm_large_load(ctx, ["ds"], "2021-01-01", None)

    assert calls == []


@sync
async def test_large_load_gate_allows_small_query(monkeypatch):
    # Below the threshold, the gate never prompts (so a "decline" ctx wouldn't matter).
    monkeypatch.setattr(server, "_estimate_query_mb", lambda *a: (10.0, 1))
    ctx = _fake_ctx(supports=True, action="decline")

    await server._confirm_large_load(ctx, ["ds"], "2021-01-01", None)  # no raise


@sync
async def test_large_load_gate_raises_on_decline(monkeypatch):
    monkeypatch.setattr(server, "_estimate_query_mb", lambda *a: (10_000.0, 50))
    ctx = _fake_ctx(supports=True, action="decline")

    with pytest.raises(ValueError, match="cancelled"):
        await server._confirm_large_load(ctx, ["ds"], "2021-01-01", None)


@sync
async def test_large_load_gate_proceeds_on_accept(monkeypatch):
    monkeypatch.setattr(server, "_estimate_query_mb", lambda *a: (10_000.0, 50))
    ctx = _fake_ctx(supports=True, action="accept")

    await server._confirm_large_load(ctx, ["ds"], "2021-01-01", None)  # no raise


@sync
async def test_load_data_sample_awaits_gated_access(monkeypatch):
    # The async tool must thread ctx through _gated_access and return its data as JSON.
    df = pd.DataFrame({"LAT": [1.5], "LON": [2.5]})

    async def fake_gated(_ctx, *args, **kwargs):
        return server.DatasetResult(data=df, var_mapping={})

    monkeypatch.setattr(server, "_gated_access", fake_gated)
    ctx = _fake_ctx(supports=False)

    result = await server.load_data_sample("ds", "2021-01-01", ["latitude"], ctx=ctx)

    assert '"LAT":1.5' in result
