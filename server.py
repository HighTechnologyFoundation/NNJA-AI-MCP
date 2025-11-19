'''
pip install fastmcp 
pip install git+https://github.com/brightbandtech/nnja-ai.git
pip install fuzzywuzzy
pip install pandas
'''

from fastmcp import FastMCP
from nnja_ai import DataCatalog, NNJADataset
from datetime import date
from fuzzywuzzy import process
import re

from pandas import DataFrame

mcp = FastMCP("NNJA-AI-MCP")


@mcp.tool()
def cite_data() -> str:
    """Get the appropriate citation for the data accessed.

    Returns:
        str: A string containing a data citation."""
    return ("NOAA NASA Joint Archive (NNJA) was accessed on "
            f"{date.today()} from https://psl.noaa.gov/data/nnja_obs/")


@mcp.tool()
def available_datasets() -> str:
    """Get a list of available NNJA-AI datasets.

    Returns:
        str: A string listing the available NNJA-AI datasets."""
    return f"{DataCatalog().list_datasets()}"


@mcp.tool()
def load_data_sample(
    dataset: str, time: str, vars: list[str], rows: int | None = None
) -> str | None:
    """Load the requested dataset into a JSON-format list of dictionaries that 
    can be easily converted to a pandas DataFrame, sliced down to the subset 
    of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows of data to include. Defaults to None.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset (subsample must be very large if used by AI)
    df = _access_dataset(dataset, time, vars, rows)

    # Convert the DataFrame into a list of dictionaries, which can be returned
    return df.to_json(orient="records")


@mcp.tool()
def summarize_dataset(dataset: str, time: str, vars: list[str]) -> str | None:
    """Analyze the columns wanted from the requested dataset and return the 
    descriptive statistics as a JSON string that can be easily converted to 
    a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of 
        the descriptive statistics of the loaded dataset, filtered down to the 
        subset of interest.
    """
    # KJW: Should not output to console on server
    # print("vars:", vars)

    # Access the requested dataset
    df = _access_dataset(dataset, time, vars)

    # Create a DataFrame of descriptive stats about the data
    stats = df.describe()

    # Convert the stats DataFrame into a JSON string, which can be returned from the MCP tool
    dicts = stats.to_json()

    # Return the JSON string of stats
    return dicts


@mcp.tool()
def correlation_matrix_dataset(dataset: str, time: str, vars: list[str]) -> str | None:
    """Analyze the columns wanted from the requested dataset and return the 
    correlation matrix as a JSON string that can be easily converted to a 
    pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.

    Returns:
        str: A JSON string that can be easily converted to a pandas DataFrame of the correlation matrix of the loaded dataset, filtered down to the subset of interest.
    """
    # Access the requested dataset
    df = _access_dataset(dataset, time, vars)

    # Create a DataFrame of the correlation matrix of the data
    correlation_matrix = df.corr()

    # Convert the correlation matrix DataFrame into a JSON string, which can be returned
    return correlation_matrix.to_json()


# Internal function for accessing a dataset
def _access_dataset(
    dataset: str, time: str, vars: list[str], rows: int | None = None
) -> DataFrame:
    """Access the requested dataset as a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to 
        search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset, which will be fuzzy matched to get valid columns names.
        rows (int, optional): The number of rows to sample from the dataset. Defaults to None.

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
    filtered_dataset = chosen_dataset.sel(time=f"{time}", variables=valid_vars)

    # Load the chosen dataset into a pandas DataFrame
    df:DataFrame = filtered_dataset.load_dataset(backend="pandas")

    # Print original data rows x columns amounts
    # KJW: Should not output to console on server
    # print("Original data shape (rows, columns):", df.shape)

    # NOTE: DataFrame size must be reduced to fully fit into AI free-tier input and output token limits
    return df[:rows] if rows is not None else df


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
                dataset_vars[f"{var.description} {str(int(match.group(0)))}"] = var.id
            else:
                # Store variables directly
                dataset_vars[var.description] = var.id

    # Initialize empty list to store valid variables
    variables:list[str] = []

    # Search through the valid variables for those wanted
    for var in var_list:
        # If the current var is valid, keep it
        if var in dataset_vars.values():
            variables.append(var)

        else: # fuzzy match to find a valid variable
            # fuzzy_var is a tuple of form: (best_match, match_score)
            fuzzy_var = process.extractOne(var, dataset_vars.keys())

            # if there is any fuzzy match
            if fuzzy_var:
                # Add the fuzzy-matched variable name
                variables.append(dataset_vars[fuzzy_var[0]])

    return variables


# Run the server when this Python file runs
if __name__ == "__main__":
    # Run the MCP server at http://0.0.0.0:8000/mcp
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
