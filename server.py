import asyncio
import contextlib
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from fuzzywuzzy import process
from mcp.types import ClientCapabilities, ElicitationCapability
from nnja_ai import DataCatalog, NNJADataset
from nnja_ai.exceptions import EmptyTimeSubsetError
from scipy import stats

# Set up logging to communicate startup time
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,  # Using stderr to avoid MCP communication issues with stdout
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nnja-ai-mcp")
logger.setLevel(logging.INFO)

_catalog: DataCatalog  # set during lifespan


@lifespan
async def catalog_lifespan(server: FastMCP):
    """Lifespan function to initialize the NNJA_AI dataset catalog when the server starts.

    Args:
        server (FastMCP): The FastMCP server instance.

    Yields:
        dict[str, Any]: An empty context dict. Tools use the module-level _catalog variable.

    Raises:
        RuntimeError: If the catalog fails to initialize (e.g., GCS unreachable or authentication failure).
    """
    global _catalog
    t = perf_counter()
    try:
        _catalog = await asyncio.to_thread(DataCatalog)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize NNJA_AI dataset catalog: {e}") from e
    logger.info("DataCatalog ready in %.2f seconds.", perf_counter() - t)
    yield {}


mcp = FastMCP("NNJA-AI-MCP", lifespan=catalog_lifespan)

# Above this size (MB, estimated from the partition manifest before any data is
# fetched), a data-loading tool asks the user to confirm. Tunable for the environment.
LARGE_QUERY_MB = 500.0


# Common variable aliases for semantic mapping across datasets
VARIABLE_ALIASES = {
    "brightness_temperature": {
        "amsua-1bamua-NC021023": "BRITCSTC.TMBR_00001",
        "atms-atms-NC021203": "BRITCSTC.TMBR_00001",
        "mhs-1bmhs-NC021027": "BRITCSTC.TMBR_00001",
    },
    "temperature": {
        "conv-adpsfc-NC000001": "TMPSQ1.TMDB",
        "conv-adpsfc-NC000002": "TMPSQ1.TMDB",
        "conv-adpsfc-NC000007": "TMPSQ1.TMDB",
        "conv-adpsfc-NC000101": "TMPSQ1.TMDB",
        "conv-adpupa-NC002001": "TMDB_PRLC100000",
        "amsua-1bamua-NC021023": "BRITCSTC.TMBR_00001",
        "atms-atms-NC021203": "BRITCSTC.TMBR_00001",
        "mhs-1bmhs-NC021027": "BRITCSTC.TMBR_00001",
    },
    "dewpoint": {
        "conv-adpsfc-NC000001": "TMPSQ1.TMDP",
        "conv-adpsfc-NC000002": "TMPSQ1.TMDP",
        "conv-adpsfc-NC000007": "TMPSQ1.TMDP",
        "conv-adpsfc-NC000101": "TMPSQ1.TMDP",
        "conv-adpupa-NC002001": "TMDP_PRLC100000",
    },
    "wind_speed": {
        "conv-adpsfc-NC000001": "WNDSQ1.WSPD",
        "conv-adpsfc-NC000002": "WNDSQ1.WSPD",
        "conv-adpsfc-NC000007": "WNDSQ1.WSPD",
        "conv-adpsfc-NC000101": "WNDSQ1.WSPD",
        "conv-adpupa-NC002001": "WSPD_PRLC100000",
    },
    "wind_direction": {
        "conv-adpsfc-NC000001": "WNDSQ1.WDIR",
        "conv-adpsfc-NC000002": "WNDSQ1.WDIR",
        "conv-adpsfc-NC000007": "WNDSQ1.WDIR",
        "conv-adpsfc-NC000101": "WNDSQ1.WDIR",
        "conv-adpupa-NC002001": "WDIR_PRLC100000",
    },
    "pressure": {
        "conv-adpsfc-NC000001": "PRSSQ1.PRES",
        "conv-adpsfc-NC000002": "PRSSQ1.PRES",
        "conv-adpsfc-NC000007": "PRSSQ1.PRES",
        "conv-adpsfc-NC000101": "PRSSQ1.PRES",
    },
    "latitude": {"DEFAULT": "LAT"},
    "longitude": {"DEFAULT": "LON"},
    "obs_date": {"DEFAULT": "OBS_DATE"},
}


# Named IR channels each dataset provides, used by the spectral indices below. The
# role->ID map is per dataset because channel numbering follows each instrument's own
# scheme. Each dataset carries an SCCF (satellite channel centre frequency) variable per
# channel, so a channel's wavelength is c / SCCF -- that is how the mappings below were
# verified (read from a live sample). Channel layouts:
#   SEVIRI (MSG):            ch4 = 3.9um, ch9 = 10.8um, ch10 = 12.0um
#   GOES ABI / Himawari AHI: channels 1-10 are bands 7-16, so ch1 = 3.9um,
#                            ch8 = 11.2um, ch9 = 12.3um (TMBRST channel N aligns with SCCF N)
# Caveat: ABI/AHI have no 10.8um channel -- "ir_108" uses their 11.2um clean window and
# "ir_120" their 12.3um band. The _data_category thresholds were tuned for SEVIRI's exact
# 3.9/10.8/12.0um bands, so on ABI/AHI the index is an approximation, not band-matched.
SPECTRAL_CHANNELS: dict[str, dict[str, str]] = {
    "seviri-sevasr-NC021042": {
        "ir_039": "RPSEQ10.TMBRST_allsky_00004",  # 3.9um shortwave
        "ir_108": "RPSEQ10.TMBRST_allsky_00009",  # 10.8um longwave
        "ir_120": "RPSEQ10.TMBRST_allsky_00010",  # 12.0um
    },
    "geo-gsrasr-NC021045": {  # GOES ABI all-sky -- 11.2/12.3um windows (see caveat)
        "ir_039": "ALLSKYRC.TMBRST_allsky_00001",  # 3.9um
        "ir_108": "ALLSKYRC.TMBRST_allsky_00008",  # 11.2um
        "ir_120": "ALLSKYRC.TMBRST_allsky_00009",  # 12.3um
    },
    "geo-gsrcsr-NC021046": {  # GOES ABI clear-sky -- 11.2/12.3um windows (see caveat)
        "ir_039": "CSRADSEQ.TMBRST_00001",  # 3.9um
        "ir_108": "CSRADSEQ.TMBRST_00008",  # 11.2um
        "ir_120": "CSRADSEQ.TMBRST_00009",  # 12.3um
    },
    "geo-ahicsr-NC021044": {  # Himawari AHI clear-sky -- 11.2/12.3um windows (see caveat)
        "ir_039": "RPSEQ11.TMBRST_00001",  # 3.9um
        "ir_108": "RPSEQ11.TMBRST_00008",  # 11.2um
        "ir_120": "RPSEQ11.TMBRST_00009",  # 12.3um
    },
}

# Each spectral index defined once, by channel role -- independent of dataset.
SPECTRAL_INDICES: dict[str, tuple[str, str]] = {
    "wildfire_risk": ("ir_039", "ir_108"),  # shortwave vs longwave
    "cloud_cooling": ("ir_108", "ir_120"),  # split-window
}


class DatasetResult(NamedTuple):
    """A named tuple to hold the dataset result along with metadata for tools that need it.

    Attributes:
        data (pd.DataFrame): The resulting dataset as a pandas DataFrame.
        var_mapping (dict[str, str | None]): A mapping from requested variable names to actual dataset variable IDs.
            A value of None indicates that the variable could not be resolved.
    """

    data: pd.DataFrame
    var_mapping: dict[str, str | None]


@mcp.tool()
def cite_data() -> str:
    """Get the appropriate citation for the data accessed.

    Returns:
        str: A string containing a data citation.
    """
    return f"NOAA NASA Joint Archive (NNJA) was accessed on {date.today()} from https://psl.noaa.gov/data/nnja_obs/"


@mcp.prompt()
def cite() -> str:
    """Cite the NNJA-AI dataset."""
    return "Cite the NNJA-AI dataset as having been accessed today."


@mcp.tool()
def available_datasets() -> str:
    """Get a list of available NNJA-AI datasets.

    Returns:
        str: A string listing the available NNJA-AI datasets.
    """
    return str(_catalog.list_datasets())


@mcp.resource("data://datasets", mime_type="application/json")
def list_datasets() -> list[str]:
    """Get a list of available NNJA-AI datasets. Used for auto-completion in the CLI client.

    Returns:
        list[str]: A list of available NNJA-AI dataset names.
    """
    return _catalog.list_datasets()


@mcp.tool()
def dataset_info(dataset: str) -> str:
    """Get a summary of the requested dataset.

    Args:
        dataset (str): The name of the dataset to describe, which will be used to search for the most similar valid dataset name.

    Returns:
        str: A string containing a summary of the requested dataset.
    """
    try:
        chosen_dataset = _resolve_dataset(dataset)
        return chosen_dataset.info()
    except ValueError as e:
        return f"Error: {e}"


@mcp.resource("data://variable-aliases", mime_type="application/json")
def variable_aliases() -> dict[str, dict[str, str]]:
    """Get the mapping of common variable names to dataset-specific IDs.

    Lets a caller translate a plain-English variable name (e.g. "temperature")
    into the ID a given dataset uses, so the data tools get valid variable names.

    Returns:
        dict[str, dict[str, str]]: {variable_name: {dataset_name: variable_id}};
            a "DEFAULT" dataset key holds an ID used when a dataset has no entry.
    """
    return VARIABLE_ALIASES


@mcp.tool()
def variables_info(dataset: str) -> str:
    """Get a list of variables and their descriptions from the requested dataset.

    Args:
        dataset (str): The name of the dataset to describe, which will be used to search for the most similar valid dataset name.

    Returns:
        str: A string containing a list of the variables in the requested dataset and their descriptions.
    """
    try:
        chosen_dataset = _resolve_dataset(dataset)
        vars_str = str(chosen_dataset.list_variables())
    except ValueError as e:
        return f"Error: {e}"

    # Add info about variable aliases
    var_aliases = []
    for v, mapping in VARIABLE_ALIASES.items():
        if chosen_dataset.name in mapping or "DEFAULT" in mapping:
            var_aliases.append(v)

    if var_aliases:
        vars_str += f"\n\nNote: You can also use the following common variable names: {', '.join(var_aliases)}"

    return vars_str


@mcp.tool()
async def load_data_sample(
    dataset: str,
    time: str,
    variables: list[str],
    rows: int = 100,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Load a sample of the requested dataset into a JSON string, sliced down to the subset of interest.

    Note: This tool can take a very long time to run, based on spatial bounds, time range, and rows requested.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        variables (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to include. Defaults to 100.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = (
            await _gated_access(
                ctx, dataset, time, variables, rows, lat_bounds, lon_bounds, end_time
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    result = df.to_json(orient="records")
    assert result is not None
    return result


@mcp.tool()
async def descriptive_stats_dataset(
    dataset: str,
    time: str,
    variables: list[str],
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Analyze the columns wanted from the requested dataset and return the descriptive statistics as a JSON string, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        variables (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the descriptive statistics of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = (
            await _gated_access(
                ctx, dataset, time, variables, rows, lat_bounds, lon_bounds, end_time
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    result = df.describe().to_json()
    assert result is not None
    return result


@mcp.tool()
async def correlation_matrix_dataset(
    dataset: str,
    time: str,
    variables: list[str],
    corr_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Analyze the columns wanted from the requested dataset and return the correlation matrix as a JSON string, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        variables (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        corr_method (Literal["pearson", "kendall", "spearman"], optional): The correlation method to use. Must be "pearson", "kendall", or "spearman". Defaults to "pearson".
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the correlation matrix of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = (
            await _gated_access(
                ctx, dataset, time, variables, rows, lat_bounds, lon_bounds, end_time
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "Error: No numeric columns available for correlation."

    result = numeric_df.corr(method=corr_method).to_json()
    assert result is not None
    return result


@mcp.tool()
async def calculate_trend(
    dataset: str,
    start_time: str,
    end_time: str,
    variable: str,
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    *,
    ctx: Context,
) -> str:
    """Calculate the linear trend for a specific variable over a time range.

    Note: Requires that the dataset contains the OBS_DATE variable/column.

    Args:
        dataset (str): The name of the dataset.
        start_time (str): Start date (YYYY-MM-DD).
        end_time (str): End date (YYYY-MM-DD).
        variable (str): The variable to calculate the trend for.
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].

    Returns:
        str: A JSON string with slope, intercept, r-squared, p-value, and supporting metadata.
    """

    try:
        df = (
            await _gated_access(
                ctx,
                dataset,
                start_time,
                [variable, "OBS_DATE"],
                rows,
                lat_bounds,
                lon_bounds,
                end_time,
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    unwanted_cols = {"OBS_DATE", "LAT", "LON"}
    data_cols = [col for col in df.columns if col not in unwanted_cols]
    if not data_cols:
        return "Error: No data variable found in result."
    actual_var = data_cols[0]

    df_mean = df.groupby("OBS_DATE")[actual_var].mean().reset_index()

    if len(df_mean) < 2:
        return "Error: Not enough time points to calculate a trend (need at least 2 dates)."

    df_mean["time_numeric"] = pd.to_numeric(pd.to_datetime(df_mean["OBS_DATE"]))

    res = stats.linregress(df_mean["time_numeric"].values, df_mean[actual_var].values)

    result = {
        "variable": variable,
        "actual_id": actual_var,
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue) ** 2,
        "p_value": float(res.pvalue),
        "mean_value": float(df_mean[actual_var].mean()),
        "start_date": str(df_mean["OBS_DATE"].min()),
        "end_date": str(df_mean["OBS_DATE"].max()),
    }

    return json.dumps(result)


@mcp.tool()
async def calculate_spectral_index(
    dataset: str,
    time: str,
    index_name: Literal["wildfire_risk", "cloud_cooling"],
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Calculate a domain-specific spectral index for satellite infrared data.

    Supported on the SEVIRI (MSG), GOES ABI, and Himawari AHI infrared datasets;
    other datasets return an error.

    Note: the index thresholds are tuned for SEVIRI's exact 3.9/10.8/12.0um bands.
    GOES ABI and Himawari AHI lack the 10.8um and 12.0um channels, so their nearest
    windows (~11.2um and ~12.3um respectively) are substituted -- results on those
    datasets are an approximation, not band-matched, and should be reported with that
    caveat. (The 3.9um shortwave band matches across all three.)

    Args:
        dataset (str): The name of the dataset (e.g., seviri-sevasr-NC021042, or a
            GOES ABI / Himawari AHI dataset).
        time (str): The time of interest (YYYY-MM-DD).
        index_name (str): The index to calculate.
            - "wildfire_risk": Based on difference between shortwave (3.9um) and longwave (10.8um) IR.
            - "cloud_cooling": Based on brightness temperature differences between 10.8um and 12.0um.
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string with the calculated index statistics.
    """
    # Resolve the entered dataset for use in guard conditions
    resolved_dataset = _resolve_dataset(dataset).name

    channels = SPECTRAL_CHANNELS.get(resolved_dataset)
    if channels is None:
        return f"Error: Dataset '{dataset}' ({resolved_dataset}) not supported for index calculation."

    role1, role2 = SPECTRAL_INDICES[index_name]
    if role1 not in channels or role2 not in channels:
        return f"Error: Index '{index_name}' not implemented for dataset '{dataset}' ({resolved_dataset})."

    var1, var2 = channels[role1], channels[role2]

    load_vars = [var1, var2]
    if index_name == "wildfire_risk":
        # per-row time & lon for day/night classification
        load_vars += ["MSG_DATE", "longitude"]

    try:
        df = (
            await _gated_access(
                ctx, dataset, time, load_vars, rows, lat_bounds, lon_bounds, end_time
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    missing_cols = [v for v in [var1, var2] if v not in df.columns]
    if missing_cols:
        return f"Error: Expected columns not found in data: {missing_cols}"

    # Calculate index (brightness temperature difference)
    df["index_value"] = df[var1] - df[var2]

    match index_name:
        case "cloud_cooling":
            return json.dumps(_calculate_cloud_cooling_index(df, var1))
        case "wildfire_risk":
            return json.dumps(_calculate_wildfire_risk_index(df, var1))


@mcp.tool()
async def calculate_lapse_rate(
    time: str,
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    level1_hpa: int = 1000,
    level2_hpa: int = 500,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Calculate the lapse rate between two pressure levels using ADPUPA (Upper-Air) data.
    Lapse rate is calculated as - (T2 - T1) / (Z2 - Z1) in K/km.

    Note: This tool is currently implemented only for the conv-adpupa-NC002001 dataset.

    Args:
        time (str): The time of interest (YYYY-MM-DD).
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries.
        lon_bounds (list[float], optional): Longitude boundaries.
        level1_hpa (int): First pressure level in hPa (e.g., 1000).
        level2_hpa (int): Second pressure level in hPa (e.g., 500).
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string with lapse rate statistics.
    """
    dataset = "conv-adpupa-NC002001"

    # Map hPa to Pa variable suffixes
    level1_pa = level1_hpa * 100
    level2_pa = level2_hpa * 100

    # t variables are temperature at the pressure levels
    t1_var = f"TMDB_PRLC{level1_pa}"
    t2_var = f"TMDB_PRLC{level2_pa}"

    # z variables are geopotential height at the pressure levels
    z1_var = f"GP10_PRLC{level1_pa}"
    z2_var = f"GP10_PRLC{level2_pa}"

    required_vars = [t1_var, t2_var, z1_var, z2_var]

    try:
        df = (
            await _gated_access(
                ctx,
                dataset,
                time,
                required_vars,
                rows,
                lat_bounds,
                lon_bounds,
                end_time,
            )
        ).data
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    # Ensure all required columns are present
    missing = [v for v in required_vars if v not in df.columns]
    if missing:
        return f"Error: Missing variables in dataset: {missing}"

    # Calculate lapse rate: - (T2 - T1) / ((Z2 - Z1) / 10000)  -> K/km
    # Note: GP10 is geopotential height in geopotential decimeters
    df["lapse_rate"] = -(df[t2_var] - df[t1_var]) / (
        (df[z2_var] - df[z1_var]) / 10000.0
    )

    # Filter out infinities or NaNs
    df["lapse_rate"] = df["lapse_rate"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["lapse_rate"])

    if df.empty:
        return (
            "Error: Could not calculate lapse rate (division by zero or missing data)."
        )

    # Apply vectorized lapse rate categorization to the data
    df["stability_category"] = _data_category("lapse_rate", df, "lapse_rate")

    # Get descriptive statistics of the lapse rate as a dictionary
    desc_stats = df["lapse_rate"].describe().to_dict()

    # Get the relative frequency distribution of stability categories
    distribution = df["stability_category"].value_counts(normalize=True).to_dict()
    distribution = {k: round(v * 100, 2) for k, v in distribution.items()}

    # Combine the results into a structured response to return
    result = {
        "summary": {
            "dominant_condition": df["stability_category"].mode()[0],
            "mean_lapse_rate": round(desc_stats["mean"], 2),
            "sample_size": int(desc_stats["count"]),
        },
        "stability_distribution": distribution,
        "raw_stats": {
            k: round(v, 2) if isinstance(v, (int, float)) else v
            for k, v in desc_stats.items()
        },
    }

    return json.dumps(result)


@mcp.tool()
async def compare_datasets(
    datasets: list[str],
    time: str,
    variables: list[str],
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Compare multiple datasets by aligning them spatially for a given day.
    Calculates the regional mean for the requested variables across all specified datasets.

    Args:
        datasets (list[str]): List of dataset names to compare.
        time (str): The time of interest (YYYY-MM-DD).
        variables (list[str]): List of variables to compare.
        rows (int, optional): The number of rows of data to use for analysis. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string with the compared statistics.
    """

    # Helper function to load and calculate mean & give var mapping for one dataset
    def _load_one(ds_name: str) -> dict[str, Any]:
        try:
            # Access data for this dataset
            dataset = _access_dataset(
                ds_name, time, variables, rows, lat_bounds, lon_bounds, end_time
            )

            df = dataset.data

            if df.empty:
                return {"error": "No data found in this region."}

            # Filter only numeric columns for mean calculation
            numeric_df = df.select_dtypes(include=[np.number])

            # Ensure means is a dictionary
            means = numeric_df.mean().to_dict()

            # Map requested variable names to actual IDs using the mapping from _access_dataset
            mapped_means = {}
            actual_ids = {}
            for v in variables:
                actual_v = dataset.var_mapping.get(v)
                if actual_v and actual_v in means:
                    mapped_means[v] = means[actual_v]
                    actual_ids[v] = actual_v

            return {
                "means": mapped_means,
                "variable_ids": actual_ids,
                "observation_count": len(df),
            }
        except ValueError as e:
            # ValueErrors are recorded per-dataset so the LLM sees partial results
            return {"error": str(e)}

    # Confirm once on the combined estimate before fetching any of the datasets.
    try:
        await _confirm_large_load(ctx, datasets, time, end_time)
    except ValueError as e:
        return f"Error: {e}"

    def _run_pool() -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=min(8, len(datasets) or 1)) as pool:
            results = list(pool.map(_load_one, datasets))
        return dict(zip(datasets, results))

    # Run the blocking thread-pool loads off the event loop.
    return json.dumps(await asyncio.to_thread(_run_pool))


def _client_supports_elicitation(ctx: Context) -> bool:
    """Return whether the connected client advertised support for elicitation."""
    return ctx.session.check_client_capability(
        ClientCapabilities(elicitation=ElicitationCapability())
    )


def _estimate_query_mb(
    datasets: list[str], time: str, end_time: str | None
) -> tuple[float, int]:
    """Estimate the on-disk size (MB) and file count a query would read.

    Reads only the in-memory partition manifest (no data is fetched), summed across
    the given datasets for the requested time range.

    Args:
        datasets (list[str]): Dataset names to estimate.
        time (str): Start date (YYYY-MM-DD).
        end_time (str | None): End date for a range, or None for a single day.

    Returns:
        tuple[float, int]: Total size in MB and total number of partition files.
    """
    total_mb = 0.0
    total_files = 0
    time_sel = slice(time, end_time) if end_time else time
    for ds in datasets:
        manifest = _resolve_dataset(ds).sel(time=time_sel).manifest
        total_mb += float(manifest["size_in_mb"].sum())
        total_files += len(manifest)
    return total_mb, total_files


async def _confirm_large_load(
    ctx: Context, datasets: list[str], time: str, end_time: str | None
) -> None:
    """Ask the user to confirm before a query that would load a lot of data.

    Best-effort and purely additive: it no-ops when the client cannot be asked
    (tests, HTTP, headless) or when the size cannot be estimated, letting
    `_access_dataset` validate inputs and raise its own errors. Only an explicit
    decline raises, which the calling tool turns into a friendly "Error: ..." string.

    Args:
        ctx (Context): The tool call context, used to reach the client session.
        datasets (list[str]): Dataset names the query will load.
        time (str): Start date (YYYY-MM-DD).
        end_time (str | None): End date for a range, or None for a single day.

    Raises:
        ValueError: If the user declines (or cancels) the confirmation.
    """
    if not _client_supports_elicitation(ctx):
        return
    try:
        total_mb, n_files = _estimate_query_mb(datasets, time, end_time)
    except Exception:
        # Never block on an estimation failure; let _access_dataset validate the input.
        return
    if total_mb < LARGE_QUERY_MB:
        return

    label = datasets[0] if len(datasets) == 1 else f"{len(datasets)} datasets"
    # response_type=None is the content-less "confirm" form: the accept/decline action
    # carries the answer, which is all the client's y/N handler returns.
    result = await ctx.elicit(
        f"Loading {label} for this time range will read about {total_mb:.0f} MB "
        f"across {n_files} file(s) and may be slow.",
        response_type=None,
    )
    if result.action != "accept":
        raise ValueError(
            "Large query cancelled. Narrow the time range to reduce how much data is loaded."
        )


async def _gated_access(
    ctx: Context,
    dataset: str,
    time: str,
    variables: list[str],
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
) -> DatasetResult:
    """Confirm an oversized load if needed, then run the blocking access off-thread.

    Wraps `_access_dataset` so tools get the elicitation gate and keep the event loop
    free during the (blocking) network read.
    """
    await _confirm_large_load(ctx, [dataset], time, end_time)
    return await asyncio.to_thread(
        _access_dataset,
        dataset,
        time,
        variables,
        rows,
        lat_bounds,
        lon_bounds,
        end_time,
    )


# Internal function for accessing a dataset
def _access_dataset(
    dataset: str,
    time: str,
    variables: list[str],
    rows: int | None = None,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
) -> DatasetResult:
    """Access the requested dataset as a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format, used as start time if end_time is specified.
        variables (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows to sample from the dataset. Defaults to None (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        DatasetResult: A named tuple containing the dataset result and metadata.

    Raises:
        ValueError: If there are issues with the input parameters (e.g., invalid dataset name, variable resolution failure, invalid bounds, or time range issues).
    """
    # Validate rows input, if entered
    if rows is not None and rows <= 0:
        raise ValueError("rows must be a positive integer.")

    # Validate latitude and longitude bound lengths, if entered
    if (lat_bounds and len(lat_bounds) != 2) or (lon_bounds and len(lon_bounds) != 2):
        raise ValueError(
            "Latitude and longitude bounds must be lists of two floats: [min, max]."
        )

    # Validate latitude bound values, if entered
    if lat_bounds and not (-90 <= lat_bounds[0] <= 90 and -90 <= lat_bounds[1] <= 90):
        raise ValueError("Latitude bound values must be between -90 to 90.")

    # Validate longitude bound values, if entered
    if lon_bounds and not (
        -180 <= lon_bounds[0] <= 180 and -180 <= lon_bounds[1] <= 180
    ):
        raise ValueError("Longitude bound values must be between -180 to 180.")

    # Validate time input(s) formatting
    try:
        parsed_time = date.fromisoformat(time)
        parsed_end_time = date.fromisoformat(end_time) if end_time else None
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD (e.g., '2021-06-15').")

    # Validate end_time input, if provided
    if parsed_end_time and parsed_end_time < parsed_time:
        raise ValueError("end_time must be greater than or equal to time (start_time).")

    # Search for the most similar valid dataset available
    chosen_dataset = _resolve_dataset(dataset)

    # Search for valid variable names using the input variable list
    # Always include LAT and LON for spatial subsetting if requested
    search_vars = list(variables)
    if lat_bounds and "LAT" not in search_vars and "latitude" not in search_vars:
        search_vars.append("latitude")
    if lon_bounds and "LON" not in search_vars and "longitude" not in search_vars:
        search_vars.append("longitude")

    var_mapping = _fuzzy_variable_search(chosen_dataset, search_vars)

    # Error handling for any unresolved variables
    unresolved = [v for v, resolved in var_mapping.items() if resolved is None]
    if unresolved:
        raise ValueError(f"Could not resolve variable(s): {unresolved}")

    valid_vars = list(set(var_mapping.values()))

    # Logging of dropped duplicate variables
    dupes = len(var_mapping) - len(valid_vars)
    if dupes:
        logger.warning(
            "%d variable(s) resolved to duplicate column IDs and were deduplicated.",
            dupes,
        )

    time_sel = slice(time, end_time) if end_time else time

    # Redirect any NNJA-AI stdout output to stderr (prevent MCP protocol corruption)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            # Filter the valid dataset down to only the subset of interest
            filtered_dataset = chosen_dataset.sel(time=time_sel, variables=valid_vars)
        except (EmptyTimeSubsetError, KeyError):
            # No partitions matched the requested time(s) (EmptyTimeSubsetError)
            # OR a time outside the dataset's range was entered (KeyError).
            # Treat this as an empty result so callers return the friendly,
            # recoverable "No data found" message (letting the LLM retry a different time)
            # rather than leaking a hard tool error to the client.
            return DatasetResult(data=pd.DataFrame(), var_mapping=var_mapping)

        # Load the chosen dataset into a pandas DataFrame
        df = filtered_dataset.load_dataset(backend="pandas")

    # Spatial filtering
    if lat_bounds:
        df = df[(df["LAT"] >= lat_bounds[0]) & (df["LAT"] <= lat_bounds[1])]
    if lon_bounds:
        df = df[(df["LON"] >= lon_bounds[0]) & (df["LON"] <= lon_bounds[1])]

    if rows is not None:
        df = df.head(rows)

    # Return the dataset (and var mapping metadata for tools that need it)
    return DatasetResult(data=df, var_mapping=var_mapping)


# Internal function for resolving a dataset from a dataset name
def _resolve_dataset(dataset: str) -> NNJADataset:
    """Searches the NNJA-AI dataset catalog for the most similar dataset matching the input name.

    Args:
        dataset (str): The name of the dataset to search for.

    Returns:
        NNJADataset: The most similar valid dataset.

    Raises:
        ValueError: If no valid dataset is found matching the input name.
    """
    # Search for valid dataset names using the input dataset name
    valid_datasets = _catalog.search(dataset)

    # If no valid datasets are found, raise an error
    if not valid_datasets:
        raise ValueError(f"No dataset matching '{dataset}' found.")

    # Get and return a valid dataset
    return _catalog[valid_datasets[0].name]


# Internal function for fuzzy searching of dataset variables
def _fuzzy_variable_search(
    dataset: NNJADataset, var_list: list[str]
) -> dict[str, str | None]:
    """Uses fuzzy matching to get valid variables to filter a dataset down to.

    Args:
        dataset (NNJADataset): The dataset of interest.
        var_list (list[str]): A list of variables to search for actual valid column names.

    Returns:
        dict[str, str | None]: A mapping of each input variable name to its resolved actual variable ID.
    """
    result: dict[str, str | None] = {}
    remaining_vars = []

    for var in var_list:
        var_lower = var.lower().replace(" ", "_")
        if var_lower in VARIABLE_ALIASES:
            if dataset.name in VARIABLE_ALIASES[var_lower]:
                result[var] = VARIABLE_ALIASES[var_lower][dataset.name]
            elif "DEFAULT" in VARIABLE_ALIASES[var_lower]:
                result[var] = VARIABLE_ALIASES[var_lower]["DEFAULT"]
            else:
                remaining_vars.append(var)
        else:
            remaining_vars.append(var)

    if not remaining_vars:
        return result

    # Use the cached variable index when possible for efficiency
    all_valid_ids, dataset_vars = _build_variable_index(dataset.name)

    choices = list(dataset_vars.keys()) + list(all_valid_ids)

    for var in remaining_vars:
        if var in all_valid_ids:
            result[var] = var
        elif var in dataset_vars:
            result[var] = dataset_vars[var]
        else:
            # fuzzy_var is a tuple of form: (best_match, match_score)
            fuzzy_var = process.extractOne(var, choices, score_cutoff=60)
            if fuzzy_var:
                match_val = fuzzy_var[0]
                result[var] = (
                    match_val if match_val in all_valid_ids else dataset_vars[match_val]
                )
            else:
                result[var] = None

    return result


def _calculate_cloud_cooling_index(
    df: pd.DataFrame,
    bt_108: str,
) -> dict[str, Any]:
    """Calculate a cloud cooling index for satellite data based on difference between 10.8um and 12.0um IR.

    Args:
        df (pd.DataFrame): The DataFrame containing the satellite data.
        bt_108 (str): The name of the variable representing the brightness temperature at 10.8um.

    Returns:
        dict[str, Any]: A dictionary with the calculated index statistics.
    """
    desc_stats = df["index_value"].describe().to_dict()

    # Apply vectorized cloud cooling categorization to the data, providing bt_108
    df["index_category"] = _data_category("cloud_cooling", df, bt_108)

    # Get the relative frequency distribution of index categories
    distribution = df["index_category"].value_counts(normalize=True).to_dict()
    distribution = {k: round(v * 100, 2) for k, v in distribution.items()}

    # Combine the results into a structured response to return
    result = {
        "summary": {
            "dominant_category": df["index_category"].mode()[0],
            "mean_index_value": round(desc_stats["mean"], 2),
            "sample_size": int(desc_stats["count"]),
        },
        "index_distribution": distribution,
        "raw_stats": {
            k: round(v, 2) if isinstance(v, (int, float)) else v
            for k, v in desc_stats.items()
        },
    }

    return result


def _calculate_wildfire_risk_index(
    df: pd.DataFrame,
    bt_39: str,
) -> dict[str, Any]:
    """Calculate a wildfire risk index for satellite data based on difference between shortwave (3.9um) and longwave (10.8um) IR.

    Args:
        df (pd.DataFrame): The DataFrame containing the satellite data.
        bt_39 (str): The name of the variable representing the brightness temperature at 3.9um.

    Returns:
        dict[str, Any]: A dictionary with the calculated index statistics.
    """
    desc_stats = df["index_value"].describe().to_dict()

    # Parse UTC hour from NNJA-AI time configuration
    utc_hour = df["MSG_DATE"].dt.hour

    # Solar time adjustment (15 degrees longitude = 1 hour difference from UTC)
    local_hour = (utc_hour + np.floor(df["LON"] / 15.0).astype(int)) % 24
    is_night = (local_hour < 6) | (local_hour > 18)  # per-row boolean Series

    # Apply vectorized wildfire risk categorization to the data, providing bt_39 and the is_night flag
    df["index_category"] = _data_category("wildfire_risk", df, bt_39, is_night)

    # Get the relative frequency distribution of index categories
    distribution = df["index_category"].value_counts(normalize=True).to_dict()
    distribution = {k: round(v * 100, 2) for k, v in distribution.items()}

    # Combine the results into a structured response to return
    result = {
        "summary": {
            "dominant_category": df["index_category"].mode()[0],
            "mean_index_value": round(desc_stats["mean"], 2),
            "active_wildfire_pixels": int(
                (df["index_category"] == "High Risk (Active Wildfire)").sum()
            ),
            "sample_size": int(desc_stats["count"]),
        },
        "index_distribution": distribution,
        "raw_stats": {
            k: round(v, 2) if isinstance(v, (int, float)) else v
            for k, v in desc_stats.items()
        },
    }
    return result


# Internal function to categorize data analysis values, vectorized using np.select for performance
def _data_category(
    analysis: Literal["lapse_rate", "cloud_cooling", "wildfire_risk"],
    df: pd.DataFrame,
    var: str,
    is_night: pd.Series | None = None,
) -> np.ndarray:
    """Categorize data analysis values based on typical conditions and any other provided factors.

    Args:
        analysis (Literal["lapse_rate", "cloud_cooling", "wildfire_risk"]): The type of analysis results to categorize.
        df (pd.DataFrame): The DataFrame containing the relevant variables.
        var (str): The name of the variable needed to make specific classifications.
        is_night (pd.Series, optional): Boolean mask of whether each observation is during nighttime, which affects the interpretation of wildfire risk.
            Only used for "wildfire_risk" analysis type, ignored otherwise.

    Returns:
        np.ndarray: An array of category labels for the variable values analyzed.

    Raises:
        ValueError: If an unknown analysis type is provided.
    """
    match analysis:
        case "lapse_rate":
            DRY_ADIABATIC_LAPSE_RATE = 9.8  # K/km
            MOIST_ADIABATIC_LAPSE_RATE = (
                6  # K/km (approximate average, varies with temperature and pressure)
            )

            # The variable passed in should be the calculated lapse rate in K/km
            lapse_rate = df[var]

            conditions = [
                (lapse_rate < 0),  # Temperature increases with height
                (lapse_rate >= 0) & (lapse_rate < MOIST_ADIABATIC_LAPSE_RATE),
                (lapse_rate >= MOIST_ADIABATIC_LAPSE_RATE)
                & (lapse_rate < DRY_ADIABATIC_LAPSE_RATE),
                (lapse_rate >= DRY_ADIABATIC_LAPSE_RATE),
            ]

            categories = [
                "Extremely Stable (Inversion)",
                "Stable",
                "Conditionally Unstable",  # Unstable only if air is saturated
                "Unstable",
            ]

        case "cloud_cooling":
            FREEZING_POINT_K = 273.15  # 0 degrees C in Kelvin
            CONVECTIVE_THRESHOLD_K = (
                240  # BT in K below which clouds are likely convective/deep
            )
            BTD_ICE_CLOUD_MIN = 1.5
            BTD_WARM_CLOUD_MAX = 1.0
            BTD_SUPERCOOLED_MAX = -0.5

            # The variable passed in should be the brightness temperature at 10.8um, which closely approximates physical temperature
            bt_108 = df[var]

            # This should be pre-calculated as the difference between the two channels (BT_108 - BT_120)
            btd = df["index_value"]

            conditions = [
                (bt_108 > FREEZING_POINT_K) & (btd > BTD_WARM_CLOUD_MAX),
                (bt_108 > FREEZING_POINT_K) & (btd <= BTD_WARM_CLOUD_MAX),
                (bt_108 <= FREEZING_POINT_K) & (btd > BTD_ICE_CLOUD_MIN),
                (bt_108 <= FREEZING_POINT_K)
                & (btd >= BTD_SUPERCOOLED_MAX)
                & (btd <= BTD_ICE_CLOUD_MIN)
                & (bt_108 < CONVECTIVE_THRESHOLD_K),
                (bt_108 <= FREEZING_POINT_K)
                & (btd >= BTD_SUPERCOOLED_MAX)
                & (btd <= BTD_ICE_CLOUD_MIN)
                & (bt_108 >= CONVECTIVE_THRESHOLD_K),
                (bt_108 <= FREEZING_POINT_K) & (btd < BTD_SUPERCOOLED_MAX),
            ]

            categories = [
                "Clear Sky (Warm/Humid Surface)",
                "Warm Water Clouds / Low Fog",
                "Thin Ice Clouds (Cirrus)",
                "Thick Ice / Deep Convective Clouds",
                "Mixed Phase / Opaque Clouds",
                "Supercooled Water Clouds",
            ]

        case "wildfire_risk":
            NIGHT_HIGH_RISK_BTD_MIN = 20.0
            NIGHT_HIGH_RISK_BT39_MIN = 310.0
            NIGHT_MEDIUM_RISK_BTD_MIN = 10.0
            NIGHT_LOW_RISK_BTD_MIN = 2.0

            DAY_HIGH_RISK_BTD_MIN = 25.0
            DAY_HIGH_RISK_BT39_MIN = 320.0
            DAY_MEDIUM_RISK_BTD_MIN = 15.0
            DAY_LOW_RISK_BTD_MIN = 6.0

            # The variable passed in should be the brightness temperature at 3.9um, which is more sensitive to high temperatures from fires
            bt_39 = df[var]

            # This should be pre-calculated as the difference between the two channels (BT_39 - BT_108)
            btd = df["index_value"]

            # Per-row day/night classification requires the is_night mask (loaded from MSG_DATE)
            if is_night is None:
                raise RuntimeError(
                    "wildfire_risk categorization requires a per-row is_night mask"
                )

            # Per-row thresholds: night vs day values based on is_night
            high_btd_min = np.where(
                is_night, NIGHT_HIGH_RISK_BTD_MIN, DAY_HIGH_RISK_BTD_MIN
            )
            high_bt_39_min = np.where(
                is_night, NIGHT_HIGH_RISK_BT39_MIN, DAY_HIGH_RISK_BT39_MIN
            )
            med_btd_min = np.where(
                is_night, NIGHT_MEDIUM_RISK_BTD_MIN, DAY_MEDIUM_RISK_BTD_MIN
            )
            low_btd_min = np.where(
                is_night, NIGHT_LOW_RISK_BTD_MIN, DAY_LOW_RISK_BTD_MIN
            )

            conditions = [
                (btd >= high_btd_min) & (bt_39 > high_bt_39_min),
                (btd >= med_btd_min),
                (btd >= low_btd_min),
                (btd < low_btd_min),
            ]

            categories = [
                "High Risk (Active Wildfire)",
                "Medium Risk (Probable Fire)",
                "Low Risk (Thermal Anomaly)",
                "No Risk (Clear / Cool Surface)",
            ]

        case _:
            raise ValueError(f"Unknown categorization type: {analysis}")

    return np.select(conditions, categories, default="Unclassified")


# Internal function to cache variable metadata for quicker repeated access
@lru_cache(maxsize=None)
def _build_variable_index(dataset_name: str) -> tuple[set[str], dict[str, str]]:
    """Build a searchable index of variable IDs and descriptions for a dataset.

    Args:
        dataset_name (str): The name of the dataset to build the variable index for.

    Returns:
        tuple[set[str], dict[str, str]]: A tuple of (all_valid_ids, dataset_vars) where
            all_valid_ids is a set of all variable IDs and dataset_vars maps descriptions to IDs.
    """
    dataset = _catalog[dataset_name]

    all_valid_ids = set()
    dataset_vars = {}

    for var_category in dataset.list_variables().values():
        for var in var_category:
            all_valid_ids.add(var.id)
            matches = re.findall(r"\d+", var.id)
            # If the variable name has numbers, use the LAST one for the description mapping
            # (as it usually indicates the channel or pressure level)
            key = var.description + (" " + str(int(matches[-1])) if matches else "")
            dataset_vars[key] = var.id

    return all_valid_ids, dataset_vars


# Run the server when this Python file runs
if __name__ == "__main__":
    # Check if the server should run as an HTTP server (for Docker)
    # or as a stdio server for the other clients
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

    if transport == "http":
        # Run the MCP server at http://0.0.0.0:8000/mcp
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")
