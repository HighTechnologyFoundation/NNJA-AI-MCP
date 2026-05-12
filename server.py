from fastmcp import FastMCP
from nnja_ai import DataCatalog, NNJADataset
from datetime import date
from fuzzywuzzy import process
import re
from typing import Literal, Any
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
    # Search for the most similar valid dataset available
    chosen_dataset = _fuzzy_dataset_search(dataset)

    # Return a summary of the dataset
    return chosen_dataset.info()


@mcp.tool()
def variables_info(dataset: str) -> str:
    """Get a list of variables and their descriptions from the requested dataset.

    Args:
        dataset (str): The name of the dataset to describe, which will be used to search for the most similar valid dataset name.

    Returns:
        str: A string containing a list of the variables in the requested dataset and their descriptions.
    """
    # Search for the most similar valid dataset available
    chosen_dataset = _fuzzy_dataset_search(dataset)

    # Return a list of variables and their descriptions from the dataset
    vars_str = str(chosen_dataset.list_variables())

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
) -> str | None:
    """Load the requested dataset into a JSON-format list of dictionaries that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to include. Defaults to 100.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset (`rows` must be reasonably small if used by AI)
    df = _access_dataset(dataset, time, vars, rows, lat_bounds, lon_bounds)

    # Convert the DataFrame into a list of dictionaries, which can be returned from the MCP tool
    dicts = df.to_json(orient="records")

    # Return the JSON formatted data
    return dicts


@mcp.tool()
def descriptive_stats_dataset(
    dataset: str,
    time: str,
    vars: list[str],
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str | None:
    """Analyze the columns wanted from the requested dataset and return the descriptive statistics as a JSON string that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the descriptive statistics of the loaded dataset, filtered down to the subset of interest.
    """
    print("vars:", vars)
    # Access the requested dataset
    df = _access_dataset(
        dataset, time, vars, lat_bounds=lat_bounds, lon_bounds=lon_bounds
    )

    # Create a DataFrame of descriptive stats about the data
    stats = df.describe()

    # Convert the stats DataFrame into a JSON string, which can be returned from the MCP tool
    dicts = stats.to_json()

    # Return the JSON string of stats
    return dicts


@mcp.tool()
def correlation_matrix_dataset(
    dataset: str,
    time: str,
    vars: list[str],
    corr_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    lat_bounds: list[float] | None = None,
    lon_bounds: list[float] | None = None,
) -> str | None:
    """Analyze the columns wanted from the requested dataset and return the correlation matrix as a JSON string that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        corr_method (Literal["pearson", "kendall", "spearman"], optional): The correlation method to use. Must be one of "pearson", "kendall", or "spearman". Defaults to "pearson".
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the correlation matrix of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset
    df = _access_dataset(
        dataset, time, vars, lat_bounds=lat_bounds, lon_bounds=lon_bounds
    )

    # Create a DataFrame of the correlation matrix of the data
    correlation_matrix = df.corr(method=corr_method)

    # Convert the correlation matrix DataFrame into a JSON string, which can be returned from the MCP tool
    dicts = correlation_matrix.to_json()

    # Return the JSON string representation of the correlation matrix
    return dicts


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

    # Search for the most similar valid dataset available
    chosen_dataset = _fuzzy_dataset_search(dataset)

    # Map virtual variable if applicable
    actual_var_list = _fuzzy_variable_search(chosen_dataset, [variable])
    if not actual_var_list:
        return f"Error: Variable '{variable}' not found in dataset."
    actual_var = actual_var_list[0]

    # Subsetting
    filtered_dataset = chosen_dataset.sel(
        time=slice(start_time, end_time), variables=[actual_var, "LAT", "LON"]
    )

    # Load dataset
    df = filtered_dataset.load_dataset(backend="pandas")

    # Spatial filter
    if lat_bounds:
        df = df[(df["LAT"] >= lat_bounds[0]) & (df["LAT"] <= lat_bounds[1])]
    if lon_bounds:
        df = df[(df["LON"] >= lon_bounds[0]) & (df["LON"] <= lon_bounds[1])]

    df = pd.DataFrame(df)

    if df.empty:
        return "Error: No data found for the given criteria."

    # Group by time and calculate mean if there are multiple observations per timestamp
    df_mean = df.groupby("OBS_DATE")[actual_var].mean().reset_index()

    # Convert dates to numbers for regression
    df_mean["time_numeric"] = pd.to_numeric(pd.to_datetime(df_mean["OBS_DATE"]))

    # Linear regression
    # Cast to Any to satisfy Pyright as it sometimes misinterprets the LinregressResult return type
    res: Any = stats.linregress(
        df_mean["time_numeric"].values, df_mean[actual_var].values
    )

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
    # Mapping for SEVIRI channels
    # Channel 4: 3.9um, Channel 9: 10.8um, Channel 10: 12.0um
    mapping = {
        "wildfire_risk": ("RPSEQ10.TMBRST_allsky_00004", "RPSEQ10.TMBRST_allsky_00009"),
        "cloud_cooling": ("RPSEQ10.TMBRST_allsky_00009", "RPSEQ10.TMBRST_allsky_00010"),
    }

    if index_name not in mapping:
        return f"Error: Index '{index_name}' not implemented."

    var1, var2 = mapping[index_name]

    # Access data
    df = _access_dataset(
        dataset,
        time,
        [var1, var2],
        rows=5000,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    if df.empty:
        return "Error: No data found for the given criteria."

    # Calculate index
    df["index_value"] = df[var1] - df[var2]

    # Descriptive stats
    stats = df["index_value"].describe()

    return str(stats.to_json())


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

    # Access data
    df = _access_dataset(
        dataset,
        time,
        required_vars,
        rows=1000,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    if df.empty:
        return "Error: No data found for the given criteria."

    # Ensure all required columns are present
    missing = [v for v in required_vars if v not in df.columns]
    if missing:
        return f"Error: Missing variables in dataset: {missing}"

    # Calculate lapse rate: - (T2 - T1) / ((Z2 - Z1) / 1000)  -> K/km
    # Note: GP10 is geopotential height in geopotential decimeters
    df["lapse_rate"] = -(df[t2_var] - df[t1_var]) / ((df[z2_var] - df[z1_var]) / 10000.0)

    # Filter out infinities or NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lapse_rate"])

    if df.empty:
        return (
            "Error: Could not calculate lapse rate (division by zero or missing data)."
        )

    stats = df["lapse_rate"].describe()

    return str(stats.to_json())


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

            # Calculate means for requested variables
            # Map requested vars to actual IDs for this dataset
            chosen_dataset = _fuzzy_dataset_search(ds_name)
            actual_vars = _fuzzy_variable_search(chosen_dataset, vars)

            # Filter only numeric columns for mean calculation
            numeric_df = df[actual_vars].select_dtypes(include=[np.number])

            # Ensure means is a dictionary
            means_result = numeric_df.mean()
            if isinstance(means_result, (int, float, np.number)):
                # Handle single column result returning a scalar
                means = {actual_vars[0]: float(means_result)}
            else:
                means = means_result.to_dict()

            # Map back to requested variable names for easy comparison
            mapped_means = {}
            for v in vars:
                actual_v_list = _fuzzy_variable_search(chosen_dataset, [v])
                if actual_v_list:
                    actual_v = actual_v_list[0]
                    if actual_v in means:
                        mapped_means[v] = means[actual_v]

            results[ds_name] = {
                "means": mapped_means,
                "observation_count": len(df),
            }
        except Exception as e:
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
) -> pd.DataFrame:
    """Access the requested dataset as a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows to sample from the dataset. Defaults to 100.
        lat_bounds (list[float], optional): Latitude boundaries [min, max] for spatial subsetting.
        lon_bounds (list[float], optional): Longitude boundaries [min, max] for spatial subsetting.

    Returns:
        pd.DataFrame: A pandas DataFrame of the requested dataset, sliced down to the subset of interest.
    """
    # Search for the most similar valid dataset available
    chosen_dataset = _fuzzy_dataset_search(dataset)

    # Search for valid variable names using the input variable list
    # Always include LAT and LON for spatial subsetting if requested
    search_vars = list(vars)
    if "LAT" not in search_vars and "latitude" not in search_vars:
        search_vars.append("latitude")
    if "LON" not in search_vars and "longitude" not in search_vars:
        search_vars.append("longitude")

    valid_vars = _fuzzy_variable_search(chosen_dataset, search_vars)

    # Filter the valid dataset down to only the subset of interest
    filtered_dataset = chosen_dataset.sel(time=f"{time}", variables=valid_vars)

    # Load the chosen dataset into a pandas DataFrame
    df = filtered_dataset.load_dataset(backend="pandas")

    # Spatial filtering
    if lat_bounds:
        df = df[(df["LAT"] >= lat_bounds[0]) & (df["LAT"] <= lat_bounds[1])]
    if lon_bounds:
        df = df[(df["LON"] >= lon_bounds[0]) & (df["LON"] <= lon_bounds[1])]

    # Print original data rows x columns amounts
    print("Original data shape (rows, columns):", df.shape)

    # NOTE: DataFrame size must be reduced to fully fit into AI free-tier input and output token limits
    if rows > 0:
        df = df[:rows]

    # Print new rows x columns amounts
    print("Sliced data shape (rows, columns):", df.shape)

    # Return the DataFrame
    return pd.DataFrame(df)


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

    # Get and return a valid dataset
    return catalog[valid_datasets[0].name]


# Internal function for fuzzy searching of dataset variables
def _fuzzy_variable_search(dataset: NNJADataset, var_list: list[str]) -> list[str]:
    """Uses fuzzy matching to get valid variables to filter a dataset down to.

    Args:
        dataset (NNJADataset): The dataset of interest.
        var_list (list[str]): A list of variables to search for actual valid column names.

    Returns:
        list[str]: A list of valid variable names to filter dataset columns down to.
    """
    # First, try to map variables using the VIRTUAL_VARIABLE_REGISTRY
    mapped_vars = []
    remaining_vars = []

    for var in var_list:
        var_lower = var.lower()
        if var_lower in VIRTUAL_VARIABLE_REGISTRY:
            # Check if there is a specific mapping for this dataset
            if dataset.name in VIRTUAL_VARIABLE_REGISTRY[var_lower]:
                mapped_vars.append(VIRTUAL_VARIABLE_REGISTRY[var_lower][dataset.name])
            # Check if there is a default mapping
            elif "DEFAULT" in VIRTUAL_VARIABLE_REGISTRY[var_lower]:
                mapped_vars.append(VIRTUAL_VARIABLE_REGISTRY[var_lower]["DEFAULT"])
            else:
                remaining_vars.append(var)
        else:
            remaining_vars.append(var)

    if not remaining_vars and mapped_vars:
        return list(set(mapped_vars))

    # Initialize a dictionary to hold the valid variables and a set of all valid IDs
    all_valid_ids = set()
    dataset_vars = {}

    # Iterate through each variable category in the dataset
    for var_category in dataset.list_variables().values():
        # Iterate through each variable in each category
        for var in var_category:
            all_valid_ids.add(var.id)
            # Find all numbers in the variable ID
            matches = re.findall(r"\d+", var.id)

            # If the variable name has numbers, use the LAST one for the description mapping
            # (as it usually indicates the channel or pressure level)
            if matches:
                # Append the number to the end of the description (without leading 0s)
                dataset_vars[var.description + " " + str(int(matches[-1]))] = var.id
            else:
                # Store variables directly
                dataset_vars[var.description] = var.id

    # Search through the valid variables for those wanted
    for var in remaining_vars:
        # If the current var is already a valid ID, keep it
        if var in all_valid_ids:
            mapped_vars.append(var)

        # Else, check if it's in our descriptive mapping
        elif var in dataset_vars:
            mapped_vars.append(dataset_vars[var])

        # Else, fuzzy match to find a valid variable among descriptions and IDs
        else:
            # fuzzy_var is a tuple of form: (best_match, match_score)
            choices = list(dataset_vars.keys()) + list(all_valid_ids)
            fuzzy_var = process.extractOne(var, choices)

            # If fuzzy_var is not None (if there is any fuzzy match), ...
            if fuzzy_var:
                match_val = fuzzy_var[0]
                if match_val in all_valid_ids:
                    mapped_vars.append(match_val)
                else:
                    mapped_vars.append(dataset_vars[match_val])

    # Return valid, fuzzy-matched variables
    return list(set(mapped_vars))


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
