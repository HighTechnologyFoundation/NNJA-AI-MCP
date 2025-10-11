from fastmcp import FastMCP
from nnja_ai import DataCatalog
from datetime import date
from typing import Any

mcp = FastMCP("NNJA-AI-MCP")


@mcp.tool()
def available_datasets() -> str:
    """Get a list of available NNJA-AI datasets with necessary data citation."""
    return f"Citation: NOAA NASA Joint Archive (NNJA) was accessed on {date.today()} from https://psl.noaa.gov/data/nnja_obs/\nDatasets in catalog: {DataCatalog().list_datasets()}"


@mcp.tool()
def load_dataset(dataset: str, time: str, vars: list[str]) -> list[dict[str, Any]]:
    """Load the requested dataset into a list of dictionaries that can be easily converted to a pandas DataFrame, sliced down to the subset of interest.

    Args:
        dataset (str): The name of the dataset to load, which will be used to search for the most similar valid dataset name.
        time (str): The time of interest to keep from the dataset in YYYY-MM-DD format.
        vars (list[str]): A list of columns of interest to keep from the dataset.

    Returns:
        list[dict[str, Any]]: A list of dictionaries that can be easily converted to a pandas DataFrame of the loaded dataset, filtered down to the subset of interest.
    """
    # Initialize the NNJA_AI dataset catalog
    catalog = DataCatalog()

    # Search for valid dataset names using the input dataset name
    valid_datasets = catalog.search(dataset)

    # Filter the dataset down to only the subset of interest
    filtered_dataset = catalog[valid_datasets[0].name].sel(
        time=f"{time} 00Z", variables=vars
    )

    # Load the chosen dataset into a pandas DataFrame
    df = filtered_dataset.load_dataset(backend="pandas")

    # Convert the DataFrame into a list of dictionaries, which can be returned from the MCP tool
    dicts = df.to_dict(orient="records")

    # Return the list of dictionaries
    return dicts


# Run the server when this Python file runs
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
