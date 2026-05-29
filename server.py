from fastmcp import FastMCP
from nnja_ai import DataCatalog, NNJADataset
from datetime import date
from fuzzywuzzy import process
import re
from typing import Literal
import pandas as pd
import numpy as np
from scipy import stats
import os
import json

mcp = FastMCP("NNJA-AI-MCP")

# Virtual Variable Registry for semantic mapping across datasets
VIRTUAL_VARIABLE_REGISTRY = {
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
}


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
    return f"{DataCatalog().list_datasets()}"


@mcp.resource("data://datasets", mime_type="application/json")
def list_datasets() -> list[str]:
    """Get a list of available NNJA-AI datasets. Used for auto-completion in the CLI client.

    Returns:
        list[str]: A list of available NNJA-AI dataset names.
    """
    return DataCatalog().list_datasets()


@mcp.tool()
def dataset_info(dataset: str) -> str:
    """Get a summary of the requested dataset.

    Args:
        dataset (str): The name of the dataset to describe, which will be used to search for the most similar valid dataset name.

    Returns:
        str: A string containing a summary of the requested dataset.
    """
    try:
        chosen_dataset = _fuzzy_dataset_search(dataset)
        return chosen_dataset.info()
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def variables_info(dataset: str) -> str:
    """Get a list of variables and their descriptions from the requested dataset.

    Args:
        dataset (str): The name of the dataset to describe, which will be used to search for the most similar valid dataset name.

    Returns:
        str: A string containing a list of the variables in the requested dataset and their descriptions.
    """
    try:
        chosen_dataset = _fuzzy_dataset_search(dataset)
        vars_str = str(chosen_dataset.list_variables())
    except ValueError as e:
        return f"Error: {e}"

    # Add info about virtual variables
    virtual_vars = []
    for v, mapping in VIRTUAL_VARIABLE_REGISTRY.items():
        if chosen_dataset.name in mapping or "DEFAULT" in mapping:
            virtual_vars.append(v)

    if virtual_vars:
        vars_str += f"\n\nNote: You can also use the following common variable names: {', '.join(virtual_vars)}"

    return vars_str


@mcp.tool()
def load_data_sample(
    dataset: str,
    time: str,
    vars: list[str],
    rows: int = 100,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
) -> str:
    """Load a sample of the requested dataset into a JSON string, sliced down to the subset of interest.

    Note: This tool can take a very long time to run, based on spatial bounds, time range, and rows requested.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to include. Defaults to 100.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = _access_dataset(
            dataset, time, vars, rows, lat_bounds, lon_bounds, end_time
        )
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    return df.to_json(orient="records")


@mcp.tool()
def descriptive_stats_dataset(
    dataset: str,
    time: str,
    vars: list[str],
    rows: int = -1,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
) -> str:
    """Analyze the columns wanted from the requested dataset and return the descriptive statistics as a JSON string, sliced down to the subset of interest.

    Note: This tool can take a very long time to run,based on spatial bounds, time range, and rows requested.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to use for analysis. Defaults to -1 (all rows).
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the descriptive statistics of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = _access_dataset(
            dataset, time, vars, rows, lat_bounds, lon_bounds, end_time
        )
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    return df.describe().to_json()


@mcp.tool()
def correlation_matrix_dataset(
    dataset: str,
    time: str,
    vars: list[str],
    corr_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str:
    """Analyze the columns wanted from the requested dataset and return the correlation matrix as a JSON string, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        corr_method (Literal["pearson", "kendall", "spearman"], optional): The correlation method to use. Must be "pearson", "kendall", or "spearman". Defaults to "pearson".
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the correlation matrix of the loaded dataset, filtered down to the subset of interest.
    """
    try:
        df = _access_dataset(
            dataset, time, vars, lat_bounds=lat_bounds, lon_bounds=lon_bounds
        )
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "Error: No numeric columns available for correlation."

    return numeric_df.corr(method=corr_method).to_json()


@mcp.tool()
def calculate_trend(
    dataset: str,
    start_time: str,
    end_time: str,
    variable: str,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str:
    """Calculate the linear trend for a specific variable over a time range.

    Args:
        dataset (str): The name of the dataset.
        start_time (str): Start date (YYYY-MM-DD).
        end_time (str): End date (YYYY-MM-DD).
        variable (str): The variable to calculate the trend for.
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].

    Returns:
        str: A JSON string with trend coefficient, p-value, and intercept.
    """

    try:
        df = _access_dataset(
            dataset,
            start_time,
            [variable, "OBS_DATE"],
            rows=-1,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            end_time=end_time,
        )
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
def calculate_spectral_index(
    dataset: str,
    time: str,
    index_name: Literal["wildfire_risk", "cloud_cooling"],
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str:
    """Calculate a domain-specific spectral index for satellite data.

    Note: This tool is currently implemented only for seviri-sevasr-NC021042.
    Do **NOT** use other datasets with this tool for now.

    Args:
        dataset (str): The name of the dataset (e.g., seviri-sevasr-NC021042).
        time (str): The time of interest (YYYY-MM-DD).
        index_name (str): The index to calculate.
            - "wildfire_risk": Based on difference between shortwave (3.9um) and longwave (10.8um) IR.
            - "cloud_cooling": Based on brightness temperature differences between 10.8um and 12.0um.
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].

    Returns:
        str: A JSON string with the calculated index statistics.
    """
    # Input validation for dataset, since this tool only works for one dataset for now
    if dataset != "seviri-sevasr-NC021042":
        raise ValueError(
            "This tool is currently only implemented for the seviri-sevasr-NC021042 dataset."
        )

    # Mapping for SEVIRI channels
    # Channel 4: 3.9um, Channel 9: 10.8um, Channel 10: 12.0um
    mapping = {
        "wildfire_risk": ("RPSEQ10.TMBRST_allsky_00004", "RPSEQ10.TMBRST_allsky_00009"),
        "cloud_cooling": ("RPSEQ10.TMBRST_allsky_00009", "RPSEQ10.TMBRST_allsky_00010"),
    }

    if index_name not in mapping:
        raise ValueError(f"Index '{index_name}' not implemented.")

    var1, var2 = mapping[index_name]

    try:
        df = _access_dataset(
            dataset,
            time,
            [var1, var2],
            rows=5000,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    missing_cols = [v for v in [var1, var2] if v not in df.columns]
    if missing_cols:
        return f"Error: Expected columns not found in data: {missing_cols}"

    # Calculate index (brightness temperature difference)
    df["index_value"] = df[var1] - df[var2]
    desc_stats = df["index_value"].describe().to_dict()

    # Stats and additional categorization for cloud cooling index
    if index_name == "cloud_cooling":
        # Apply vectorized cloud cooling categorization to the data, providing bt_108
        df["index_category"] = _data_category("cloud_cooling", df, var1)

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

        return json.dumps(result)

    if index_name == "wildfire_risk":
        # Parse UTC hour from NNJA-AI time configuration
        if "T" in time:
            utc_hour = int(time.split("T")[1].split(":")[0])
        else:
            utc_hour = 0  # Default to 0 UTC if no time provided

        # Extract spatial layout to deduce Local Solar Time over SEVIRI disk
        if lon_bounds and len(lon_bounds) == 2:
            avg_lon = sum(lon_bounds) / 2
        else:
            avg_lon = 0.0  # Default to Prime Meridian

        # Solar time adjustment (15 degrees longitude = 1 hour difference from UTC)
        local_hour = (utc_hour + int(avg_lon / 15.0)) % 24
        is_night = local_hour < 6 or local_hour > 18

        # Apply vectorized wildfire risk categorization to the data, providing bt_39 and the is_night flag
        df["index_category"] = _data_category("wildfire_risk", df, var1, is_night)

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
                "thresholds_used": "Nighttime" if is_night else "Daytime",
                "calculated_local_hour": round(local_hour, 1),
                "sample_size": int(desc_stats["count"]),
            },
            "index_distribution": distribution,
            "raw_stats": {
                k: round(v, 2) if isinstance(v, (int, float)) else v
                for k, v in desc_stats.items()
            },
        }
        return json.dumps(result)

    # Default return of descriptive stats for any spectral index without specific categorization logic
    return json.dumps(desc_stats)


@mcp.tool()
def calculate_lapse_rate(
    time: str,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    level1_hpa: int = 1000,
    level2_hpa: int = 500,
) -> str:
    """Calculate the lapse rate between two pressure levels using ADPUPA (Upper-Air) data.
    Lapse rate is calculated as - (T2 - T1) / (Z2 - Z1) in K/km.

    Note: This tool is currently implemented only for the conv-adpupa-NC002001 dataset.

    Args:
        time (str): The time of interest (YYYY-MM-DD).
        lat_bounds (list[float], optional): Latitude boundaries.
        lon_bounds (list[float], optional): Longitude boundaries.
        level1_hpa (int): First pressure level in hPa (e.g., 1000).
        level2_hpa (int): Second pressure level in hPa (e.g., 500).

    Returns:
        str: A JSON string with lapse rate statistics.
    """
    dataset = "conv-adpupa-NC002001"

    # Map hPa to Pa variable suffixes
    l1 = level1_hpa * 100
    l2 = level2_hpa * 100

    t1_var = f"TMDB_PRLC{l1}"
    t2_var = f"TMDB_PRLC{l2}"
    z1_var = f"GP10_PRLC{l1}"
    z2_var = f"GP10_PRLC{l2}"

    required_vars = [t1_var, t2_var, z1_var, z2_var]

    try:
        df = _access_dataset(
            dataset,
            time,
            required_vars,
            rows=1000,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
        )
    except ValueError as e:
        return f"Error: {e}"

    if df.empty:
        return "Error: No data found for the given criteria."

    # Ensure all required columns are present
    missing = [v for v in required_vars if v not in df.columns]
    if missing:
        return f"Error: Missing variables in dataset: {missing}"

    # Calculate lapse rate: - (T2 - T1) / ((Z2 - Z1) / 1000)  -> K/km
    # Note: GP10 is geopotential height in geopotential decimeters
    df["lapse_rate"] = -(df[t2_var] - df[t1_var]) / (
        (df[z2_var] - df[z1_var]) / 10000.0
    )

    # Filter out infinities or NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lapse_rate"])

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
def compare_datasets(
    datasets: list[str],
    time: str,
    vars: list[str],
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str:
    """Compare multiple datasets by aligning them spatially for a given day.
    Calculates the regional mean for the requested variables across all specified datasets.

    Args:
        datasets (list[str]): List of dataset names to compare.
        time (str): The time of interest (YYYY-MM-DD).
        vars (list[str]): Variables to compare (uses virtual mapping).
        lat_bounds (list[float], optional): Latitude boundaries [min, max].
        lon_bounds (list[float], optional): Longitude boundaries [min, max].

    Returns:
        str: A JSON string with the compared statistics.
    """
    results = {}

    for ds_name in datasets:
        try:
            # Access data for this dataset
            # We use rows=-1 to load all data in the region for better averaging
            df = _access_dataset(
                ds_name,
                time,
                vars,
                rows=-1,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
            )

            if df.empty:
                results[ds_name] = "No data found in this region."
                continue

            # Filter only numeric columns for mean calculation
            numeric_df = df.select_dtypes(include=[np.number])

            # Ensure means is a dictionary
            means_result = numeric_df.mean()
            if isinstance(means_result, (int, float, np.number)):
                means = {numeric_df.columns[0]: float(means_result)}
            else:
                means = means_result.to_dict()

            # Map requested variable names to actual IDs using the mapping from _access_dataset
            mapped_means = {}
            for v in vars:
                actual_v = df._var_mapping.get(v)
                if actual_v and actual_v in means:
                    mapped_means[v] = means[actual_v]

            results[ds_name] = {
                "means": mapped_means,
                "observation_count": len(df),
            }
        except Exception as e:
            # Errors are recorded per-dataset so the LLM sees partial results
            results[ds_name] = f"Error: {str(e)}"

    return json.dumps(results)


# Internal function for accessing a dataset
def _access_dataset(
    dataset: str,
    time: str,
    vars: list[str],
    rows: int = 100,
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Access the requested dataset as a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format, used as start time if end_time is specified.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows to sample from the dataset. Defaults to 100.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.
        end_time (str, optional): The end time for a time range to keep from the dataset in YYYY-MM-DD format, use if a range is wanted.

    Returns:
        pd.DataFrame: A pandas DataFrame of the requested dataset, sliced down to the subset of interest.
    """
    # Validate latitude and longitude bounds, if entered
    if (lat_bounds and len(lat_bounds) != 2) or (lon_bounds and len(lon_bounds) != 2):
        raise ValueError(
            "Latitude and longitude bounds must be lists of two floats: [min, max]."
        )

    # Validate end_time input, if provided
    if end_time and date.fromisoformat(end_time) < date.fromisoformat(time):
        raise ValueError("end_time must be greater than or equal to time (start_time).")

    # Search for the most similar valid dataset available
    chosen_dataset = _fuzzy_dataset_search(dataset)

    # Search for valid variable names using the input variable list
    # Always include LAT and LON for spatial subsetting if requested
    search_vars = list(vars)
    if lat_bounds and "LAT" not in search_vars and "latitude" not in search_vars:
        search_vars.append("latitude")
    if lon_bounds and "LON" not in search_vars and "longitude" not in search_vars:
        search_vars.append("longitude")

    var_mapping = _fuzzy_variable_search(chosen_dataset, search_vars)
    valid_vars = list(set(var_mapping.values()))

    if end_time:
        time = slice(time, end_time)

    # Filter the valid dataset down to only the subset of interest
    filtered_dataset = chosen_dataset.sel(time=time, variables=valid_vars)

    # Load the chosen dataset into a pandas DataFrame
    df = filtered_dataset.load_dataset(backend="pandas")

    # Spatial filtering
    if lat_bounds:
        df = df[(df["LAT"] >= lat_bounds[0]) & (df["LAT"] <= lat_bounds[1])]
    if lon_bounds:
        df = df[(df["LON"] >= lon_bounds[0]) & (df["LON"] <= lon_bounds[1])]

    # NOTE: DataFrame size must be reduced to fully fit into AI free-tier input and output token limits
    if rows > 0:
        df = df[:rows]

    # Add the dataset name and var mapping as an attribute to the DataFrame for tools that need it
    df._name = chosen_dataset.name
    df._var_mapping = var_mapping

    # Return the DataFrame
    return df


# Internal function for fuzzy searching of dataset names
def _fuzzy_dataset_search(dataset: str) -> NNJADataset:
    """Uses fuzzy matching to get valid dataset names.

    Args:
        dataset (str): The name of the dataset to search for.

    Returns:
        str: The most similar valid dataset name.
    """
    # Initialize the NNJA_AI dataset catalog
    catalog = DataCatalog()

    # Search for valid dataset names using the input dataset name
    valid_datasets = catalog.search(dataset)

    # If no valid datasets are found, raise an error
    if not valid_datasets:
        raise ValueError(f"No dataset matching '{dataset}' found.")

    # Get and return a valid dataset
    return catalog[valid_datasets[0].name]


# Internal function for fuzzy searching of dataset variables
def _fuzzy_variable_search(dataset: NNJADataset, var_list: list[str]) -> dict[str, str]:
    """Uses fuzzy matching to get valid variables to filter a dataset down to.

    Args:
        dataset (NNJADataset): The dataset of interest.
        var_list (list[str]): A list of variables to search for actual valid column names.

    Returns:
        dict[str, str]: A mapping of each input variable name to its resolved actual variable ID.
    """
    result: dict[str, str] = {}
    remaining_vars = []

    for var in var_list:
        var_lower = var.lower()
        if var_lower in VIRTUAL_VARIABLE_REGISTRY:
            if dataset.name in VIRTUAL_VARIABLE_REGISTRY[var_lower]:
                result[var] = VIRTUAL_VARIABLE_REGISTRY[var_lower][dataset.name]
            elif "DEFAULT" in VIRTUAL_VARIABLE_REGISTRY[var_lower]:
                result[var] = VIRTUAL_VARIABLE_REGISTRY[var_lower]["DEFAULT"]
            else:
                remaining_vars.append(var)
        else:
            remaining_vars.append(var)

    if not remaining_vars:
        return result

    # Initialize a dictionary to hold the valid variables and a set of all valid IDs
    all_valid_ids = set()
    dataset_vars = {}

    for var_category in dataset.list_variables().values():
        for var in var_category:
            all_valid_ids.add(var.id)
            matches = re.findall(r"\d+", var.id)

            # If the variable name has numbers, use the LAST one for the description mapping
            # (as it usually indicates the channel or pressure level)
            if matches:
                # Append the number to the end of the description (without leading 0s)
                dataset_vars[var.description + " " + str(int(matches[-1]))] = var.id
            else:
                dataset_vars[var.description] = var.id

    for var in remaining_vars:
        if var in all_valid_ids:
            result[var] = var
        elif var in dataset_vars:
            result[var] = dataset_vars[var]
        else:
            # fuzzy_var is a tuple of form: (best_match, match_score)
            choices = list(dataset_vars.keys()) + list(all_valid_ids)
            fuzzy_var = process.extractOne(var, choices)
            if fuzzy_var:
                match_val = fuzzy_var[0]
                result[var] = (
                    match_val if match_val in all_valid_ids else dataset_vars[match_val]
                )

    return result


# Internal function to categorize data analysis values, vectorized using np.select for performance
def _data_category(
    analysis: Literal["lapse_rate", "cloud_cooling", "wildfire_risk"],
    df: pd.DataFrame,
    var: str,
    is_night: bool | None = None,
) -> np.ndarray:
    """Categorize data analysis values based on typical conditions and any other provided factors.

    Args:
        analysis (Literal["lapse_rate", "cloud_cooling", "wildfire_risk"]): The type of analysis results to categorize.
        df (pd.DataFrame): The DataFrame containing the relevant variables.
        var (str): The name of the variable needed to make specific classifications.
        is_night (bool, optional): Whether the observation is during nighttime, which affects the interpretation of wildfire risk.

    Returns:
        np.ndarray: An array of category labels for the lapse rates.
    """
    match analysis:
        case "lapse_rate":
            # The variable passed in should be the calculated lapse rate in K/km
            lapse_rate = df[var]
            conditions = [
                # Inversion (temperature increases with height, extremely stable)
                (lapse_rate < 0),
                # Stable (less than moist adiabatic lapse rate)
                (lapse_rate >= 0) & (lapse_rate < 6),
                # Conditionally Unstable (unstable if saturated, stable if unsaturated)
                (lapse_rate >= 6) & (lapse_rate < 9.8),
                # Unstable (greater than dry adiabatic lapse rate)
                (lapse_rate >= 9.8),
            ]
            categories = [
                "Extremely Stable (Inversion)",
                "Stable",
                "Conditionally Unstable",
                "Unstable",
            ]

        case "cloud_cooling":
            # The variable passed in should be the brightness temperature at 10.8um, which closely approximates physical temperature
            bt_108 = df[var]

            # This should be pre-calculated as the difference between the two channels (BT_108 - BT_120)
            btd = df["index_value"]

            # Define conditions for categorization of cloud phases
            conditions = [
                # Warm Surface Conditions (bt_108 > 273.15)
                (bt_108 > 273.15) & (btd > 1.0),
                (bt_108 > 273.15) & (btd <= 1.0),
                # Cold Conditions (bt_108 <= 273.15)
                (bt_108 <= 273.15) & (btd > 1.5),
                (bt_108 <= 273.15) & (btd >= -0.5) & (btd <= 1.5) & (bt_108 < 240),
                (bt_108 <= 273.15) & (btd >= -0.5) & (btd <= 1.5) & (bt_108 >= 240),
                (bt_108 <= 273.15) & (btd < -0.5),
            ]

            # Match each condition to its respective category label
            categories = [
                "Clear Sky (Warm/Humid Surface)",
                "Warm Water Clouds / Low Fog",
                "Thin Ice Clouds (Cirrus)",
                "Thick Ice / Deep Convective Clouds",
                "Mixed Phase / Opaque Clouds",
                "Supercooled Water Clouds",
            ]

        case "wildfire_risk":
            # The variable passed in should be the brightness temperature at 3.9um, which is more sensitive to high temperatures from fires
            bt_39 = df[var]
            # This should be pre-calculated as the difference between the two channels (BT_39 - BT_108)
            btd = df["index_value"]

            if is_night:
                conditions = [
                    (btd >= 20.0) & (bt_39 > 310.0),
                    (btd >= 10.0),
                    (btd >= 2.0),
                    (btd < 2.0),
                ]
            else:
                conditions = [
                    (btd >= 25.0) & (bt_39 > 320.0),
                    (btd >= 15.0),
                    (btd >= 6.0),
                    (btd < 6.0),
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
