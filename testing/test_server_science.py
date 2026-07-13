"""Unit tests for server.py's scientific categorization layer.

These cover the pure, deterministic functions that turn raw brightness-temperature /
lapse-rate numbers into climatological category labels:
  - `_data_category` — the vectorized `np.select` classifier for lapse_rate,
    cloud_cooling, and wildfire_risk (every threshold + boundary operator + guard).
  - `_calculate_wildfire_risk_index` — specifically its solar-time `local_hour`/
    `is_night` derivation, which selects the day-vs-night threshold set.
  - `_calculate_cloud_cooling_index` — result-dict shape.

All of these take a hand-built DataFrame and touch neither the catalog nor the
network, so (unlike the `integration_*.py` scripts) they run under plain `uv run
pytest`. They guard the exact place a flipped threshold or boundary operator would
silently mislabel data while every integration check still passed.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import server

# _data_category: lapse_rate
# Boundaries: <0 inversion | [0,6) stable | [6,9.8) cond. unstable | >=9.8 unstable


@pytest.mark.parametrize(
    "value, expected",
    [
        (-5.0, "Extremely Stable (Inversion)"),
        (-0.01, "Extremely Stable (Inversion)"),
        (0.0, "Stable"),  # boundary: 0 is Stable, not inversion
        (3.0, "Stable"),
        (5.99, "Stable"),
        (6.0, "Conditionally Unstable"),  # boundary: moist adiabatic (6 K/km)
        (9.79, "Conditionally Unstable"),
        (9.8, "Unstable"),  # boundary: dry adiabatic (9.8 K/km)
        (12.0, "Unstable"),
    ],
)
def test_data_category_lapse_rate_boundaries(value, expected):
    df = pd.DataFrame({"lapse_rate": [value]})
    result = server._data_category("lapse_rate", df, "lapse_rate")
    assert result[0] == expected


def test_data_category_lapse_rate_nan_is_unclassified():
    # No condition matches NaN, so np.select's default must surface it.
    df = pd.DataFrame({"lapse_rate": [np.nan]})
    result = server._data_category("lapse_rate", df, "lapse_rate")
    assert result[0] == "Unclassified"


# _data_category: cloud_cooling
# var = BT_10.8um column; df["index_value"] = BTD (BT_108 - BT_120).
# Freezing=273.15, convective<240, BTD bands: warm<=1.0, ice_min>1.5, supercooled<-0.5


@pytest.mark.parametrize(
    "bt108, btd, expected",
    [
        (280.0, 2.0, "Clear Sky (Warm/Humid Surface)"),  # warm & btd>1.0
        (280.0, 1.0, "Warm Water Clouds / Low Fog"),  # btd==1.0 -> warm-cloud branch
        (280.0, 0.5, "Warm Water Clouds / Low Fog"),
        (273.15, 2.0, "Thin Ice Clouds (Cirrus)"),  # exactly freezing -> cold side
        (250.0, 2.0, "Thin Ice Clouds (Cirrus)"),  # cold & btd>1.5
        (250.0, 1.5, "Mixed Phase / Opaque Clouds"),  # btd==1.5 not cirrus; bt>=240
        (235.0, 0.0, "Thick Ice / Deep Convective Clouds"),  # bt<240 convective
        (
            240.0,
            0.0,
            "Mixed Phase / Opaque Clouds",
        ),  # bt==240 not convective (strict <)
        (250.0, -0.5, "Mixed Phase / Opaque Clouds"),  # btd==-0.5 not supercooled
        (235.0, -0.5, "Thick Ice / Deep Convective Clouds"),
        (250.0, -1.0, "Supercooled Water Clouds"),  # btd<-0.5
    ],
)
def test_data_category_cloud_cooling_boundaries(bt108, btd, expected):
    df = pd.DataFrame({"bt108": [bt108], "index_value": [btd]})
    result = server._data_category("cloud_cooling", df, "bt108")
    assert result[0] == expected


# _data_category: wildfire_risk (per-row night/day thresholds)
# var = BT_3.9um column; df["index_value"] = BTD (BT_39 - BT_108); is_night Series.
# NIGHT: High btd>=20 & bt39>310 | Med btd>=10 | Low btd>=2
# DAY:   High btd>=25 & bt39>320 | Med btd>=15 | Low btd>=6


@pytest.mark.parametrize(
    "night, btd, bt39, expected",
    [
        # --- night ---
        (True, 20.0, 311.0, "High Risk (Active Wildfire)"),  # btd>=20 & bt39>310
        (True, 20.0, 310.0, "Medium Risk (Probable Fire)"),  # bt39==310 not > 310
        (True, 20.0, 305.0, "Medium Risk (Probable Fire)"),  # hot-enough btd, cool bt39
        (True, 10.0, 300.0, "Medium Risk (Probable Fire)"),  # btd==10 med boundary
        (True, 9.9, 300.0, "Low Risk (Thermal Anomaly)"),
        (True, 2.0, 300.0, "Low Risk (Thermal Anomaly)"),  # btd==2 low boundary
        (True, 1.9, 300.0, "No Risk (Clear / Cool Surface)"),
        # --- day (higher thresholds) ---
        (False, 25.0, 321.0, "High Risk (Active Wildfire)"),  # btd>=25 & bt39>320
        (False, 25.0, 320.0, "Medium Risk (Probable Fire)"),  # bt39==320 not > 320
        (
            False,
            20.0,
            350.0,
            "Medium Risk (Probable Fire)",
        ),  # btd<25 -> not High by day
        (False, 15.0, 300.0, "Medium Risk (Probable Fire)"),  # btd==15 med boundary
        (False, 14.9, 300.0, "Low Risk (Thermal Anomaly)"),
        (False, 6.0, 300.0, "Low Risk (Thermal Anomaly)"),  # btd==6 low boundary
        (False, 5.9, 300.0, "No Risk (Clear / Cool Surface)"),
    ],
)
def test_data_category_wildfire_boundaries(night, btd, bt39, expected):
    df = pd.DataFrame({"bt39": [bt39], "index_value": [btd]})
    is_night = pd.Series([night])
    result = server._data_category("wildfire_risk", df, "bt39", is_night)
    assert result[0] == expected


def test_data_category_wildfire_same_input_flips_with_day_night():
    # The core of why day/night matters: identical (btd=20, bt39=311) is High Risk at
    # night (lower thresholds) but only Medium Risk by day. Both rows in one call also
    # exercises the per-row np.where threshold vectorization.
    df = pd.DataFrame({"bt39": [311.0, 311.0], "index_value": [20.0, 20.0]})
    is_night = pd.Series([True, False])
    result = server._data_category("wildfire_risk", df, "bt39", is_night)
    assert result[0] == "High Risk (Active Wildfire)"  # night
    assert result[1] == "Medium Risk (Probable Fire)"  # day


# _data_category: guards


def test_data_category_unknown_analysis_raises():
    bad_analysis: Any = "not_a_real_type"
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="Unknown categorization type"):
        server._data_category(bad_analysis, df, "x")


def test_data_category_wildfire_requires_is_night():
    # Without the per-row mask the day/night thresholds are undefined -> hard error.
    df = pd.DataFrame({"bt39": [311.0], "index_value": [20.0]})
    with pytest.raises(RuntimeError, match="is_night"):
        server._data_category("wildfire_risk", df, "bt39")  # is_night defaults to None


# _calculate_wildfire_risk_index solar-time day/night derivation
# local_hour = (utc_hour + int(LON / 15)) % 24 ; is_night = local<6 or local>18
# LON is kept a multiple of 15 so the night calc is independent of the known
# truncation-toward-zero edge for sub-15-degree negative longitudes (server B3).


def _wildfire_df(hours, lons, btd, bt39):
    """Build the DataFrame `_calculate_wildfire_risk_index` expects.

    Needs MSG_DATE (datetime, for the UTC hour), LON, index_value (BTD), and the
    named BT_3.9um column ("bt39"). One row per (hour, lon).
    """
    n = len(hours)
    return pd.DataFrame(
        {
            "MSG_DATE": pd.to_datetime([f"2023-07-01 {h:02d}:00:00" for h in hours]),
            "LON": lons,
            "index_value": [btd] * n,
            "bt39": [bt39] * n,
        }
    )


def test_wildfire_index_night_vs_day_flips_classification():
    # btd=20/bt39=311 -> High at night, Medium by day. UTC hour drives is_night here
    # (LON=0 so local_hour == utc_hour): 00:00 is night, 12:00 is day.
    night = server._calculate_wildfire_risk_index(
        _wildfire_df([0], [0.0], btd=20.0, bt39=311.0), "bt39"
    )
    day = server._calculate_wildfire_risk_index(
        _wildfire_df([12], [0.0], btd=20.0, bt39=311.0), "bt39"
    )
    assert night["summary"]["dominant_category"] == "High Risk (Active Wildfire)"
    assert night["summary"]["active_wildfire_pixels"] == 1
    assert day["summary"]["dominant_category"] == "Medium Risk (Probable Fire)"
    assert day["summary"]["active_wildfire_pixels"] == 0


def test_wildfire_index_is_night_boundaries():
    # is_night = (local < 6) | (local > 18). With LON=0, local_hour == utc_hour.
    # Rows at 5 (night), 6 (day), 18 (day), 19 (night), all btd=20/bt39=311 -> High
    # only when night, so exactly 2 of the 4 are active-wildfire pixels.
    df = _wildfire_df([5, 6, 18, 19], [0.0, 0.0, 0.0, 0.0], btd=20.0, bt39=311.0)
    result = server._calculate_wildfire_risk_index(df, "bt39")
    assert result["summary"]["active_wildfire_pixels"] == 2
    assert result["units"]["index_value"] == "K"


@pytest.mark.parametrize(
    "hour, lon, expect_night",
    [
        (12, 0.0, False),  # local 12 -> day
        (6, 0.0, False),  # local 6 -> day (boundary: 6 is NOT night)
        (12, 180.0, True),  # +12h -> local 0 -> night
        (12, -180.0, True),  # -12h -> local 0 -> night
        (20, 90.0, True),  # +6h -> local 26 % 24 = 2 -> night (wrap-around)
        (23, 15.0, True),  # +1h -> local 24 % 24 = 0 -> night (wrap-around)
    ],
)
def test_wildfire_index_local_hour_offset_and_wrap(hour, lon, expect_night):
    # btd=20/bt39=311 is High only at night, so an active pixel means night was
    # detected. This pins the longitude offset and the % 24 wrap.
    df = _wildfire_df([hour], [lon], btd=20.0, bt39=311.0)
    result = server._calculate_wildfire_risk_index(df, "bt39")
    is_night_detected = result["summary"]["active_wildfire_pixels"] == 1
    assert is_night_detected is expect_night


# _calculate_cloud_cooling_index result shape


def test_cloud_cooling_index_result_structure():
    # Two warm-clear rows + one cirrus row -> a clear dominant and a 2-bucket split.
    df = pd.DataFrame(
        {
            "bt108": [280.0, 280.0, 250.0],
            "index_value": [2.0, 2.0, 2.0],
        }
    )
    result = server._calculate_cloud_cooling_index(df, "bt108")

    assert set(result) == {"summary", "index_distribution", "raw_stats", "units"}
    assert result["units"] == {
        "index_value": "K",
        "index_distribution": "percent of observations",
    }
    assert set(result["summary"]) == {
        "dominant_category",
        "mean_index_value",
        "sample_size",
    }
    assert result["summary"]["dominant_category"] == "Clear Sky (Warm/Humid Surface)"
    assert result["summary"]["sample_size"] == 3
    assert result["summary"]["mean_index_value"] == 2.0
    # Distribution is a percentage split over the present categories.
    assert round(sum(result["index_distribution"].values())) == 100
