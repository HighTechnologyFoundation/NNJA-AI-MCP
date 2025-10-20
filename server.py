from fastmcp import FastMCP
from nnja_ai import DataCatalog, NNJADataset
from datetime import date
from fuzzywuzzy import process
import re

from pandas import DataFrame

mcp = FastMCP("NNJA-AI-MCP")


@mcp.tool()
def available_datasets() -> str:
    """Get a list of available NNJA-AI datasets with necessary data citation.

    Returns:
        str: A string containing a data citation and listing the available datasets."""
    return f"Citation: NOAA NASA Joint Archive (NNJA) was accessed on {date.today()} from https://psl.noaa.gov/data/nnja_obs/\nDatasets in catalog: {DataCatalog().list_datasets()}"


# TODO: Find way to get more data accessed within AI input and output limits, may not be possible
# Problem 1: AI models have strict output token limits that massively limit the data able to be transferred
# Problem 2: AI models have somewhat restrictive input token limits that reduce amount of data able to be read
@mcp.tool()
def load_dataset(dataset: str, time: str, vars: list[str]) -> str | None:
    """Load the requested dataset into a JSON-format list of dictionaries that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset (subsample must be very large if used by AI)
    df = _access_dataset(dataset, time, vars, 100)

    # Convert the DataFrame into a list of dictionaries, which can be returned from the MCP tool
    dicts = df.to_json(orient="records")

    # Return the JSON formatted data
    return dicts


@mcp.tool()
def analyze_dataset(dataset: str, time: str, vars: list[str]) -> str | None:
    """Analyze the columns wanted from the requested dataset and return the descriptive statistics as a JSON string that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset
    df = _access_dataset(dataset, time, vars)

    # Create a DataFrame of descriptive stats about the data
    stats = df.describe()

    # Convert the stats DataFrame into a JSON string, which can be returned from the MCP tool
    dicts = stats.to_json()

    # Return the JSON string of stats
    return dicts


# Internal function for accessing a dataset
def _access_dataset(
    dataset: str, time: str, vars: list[str], subsample: int | None = None
) -> DataFrame:
    """Access the requested dataset as a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        subsample (int, optional): The amount to step by in order to reduce the size of the dataset. Defaults to None.

    Returns:
        DataFrame: A pandas DataFrame of the requested dataset, sliced down to the subset of interest.
    """
    # Initialize the NNJA_AI dataset catalog
    catalog = DataCatalog()

    # Search for valid dataset names using the input dataset name
    valid_datasets = catalog.search(dataset)

    # Get valid dataset
    chosen_dataset = catalog[valid_datasets[0].name]

    # Search for valid variable names using the input variable list
    valid_vars = _fuzzy_variable_search(chosen_dataset, vars)

    # Filter the valid dataset down to only the subset of interest
    filtered_dataset = chosen_dataset.sel(time=f"{time} 00Z", variables=valid_vars)

    # Load the chosen dataset into a pandas DataFrame
    df = filtered_dataset.load_dataset(backend="pandas")

    # NOTE: DataFrame size must be reduced to fully fit into AI free-tier token limits, not sure how else to handle this yet
    if subsample is not None:
        df = df[::subsample]

    # Print new rows x columns amounts
    print("Sliced data shape (rows, columns):", df.shape)

    # Return the DataFrame
    return DataFrame(df)


# Internal function for fuzzy searching of dataset variables
def _fuzzy_variable_search(dataset: NNJADataset, var_list: list[str]) -> list[str]:
    """Uses fuzzy matching to get valid variables to filter a dataset down to.

    Args:
        dataset (NNJADataset): The dataset of interest.
        var_list (list[str]): A list of variables to search for actual valid column names.

    Returns:
        list[str]: A list of valid variable names to filter dataset columns down to.
    """
    # Initialize a dictionary to hold the valid variables
    dataset_vars = {}

    # Iterate through each variable category in the dataset
    for var_category in dataset.list_variables().values():
        # Iterate through each variable in each category
        for var in var_category:
            # Check if the variable name has a number
            match = re.search(r"\d+", var.id)

            # If the variable name has a number, include the number in the description
            if match:
                # Append the number to the end of the description (without leading 0s)
                dataset_vars[var.description + " " + str(int(match.group(0)))] = var.id
            else:
                # Store variables directly
                dataset_vars[var.description] = var.id

    # Initialize empty list to store valid variables
    variables = []

    # Search through the valid variables for those wanted
    for var in var_list:
        # If the current var is valid, keep it
        if var in dataset_vars.values():
            variables.append(var)

        # Else, fuzzy match to find a valid variable
        else:
            # fuzzy_var is a tuple of form: (best_match, match_score)
            fuzzy_var = process.extractOne(var, dataset_vars.keys())

            # If fuzzy_var is not None (if there is any fuzzy match), ...
            if fuzzy_var:
                # Add the fuzzy-matched variable name
                variables.append(dataset_vars[fuzzy_var[0]])

    # Return valid, fuzzy-matched variables
    return variables


# Run the server when this Python file runs
if __name__ == "__main__":
    # Run the MCP server at http://0.0.0.0:8000/mcp
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
