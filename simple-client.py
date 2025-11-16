from fastmcp import Client
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        # List the tools available from the MCP server
        tools = await mcp_client.list_tools()
        print("Available tools:")
        [print(f"- {tool.name}: {tool.description}\n") for tool in tools]

        # List the datasets available from the MCP server
        datasets = await mcp_client.call_tool("available_datasets")
        print(datasets.data)

        # ---------- load_data_sample ----------
        # Call the `load_data_sample` tool, specifying the subset of interest
        loaded = await mcp_client.call_tool(
            "load_data_sample",
            {
                "dataset": "amsu",  # Doesn't need to be an exact dataset name
                "time": "2021-01-01",
                "vars": [  # Variable names don't need to be exact
                    "Latitude",
                    "LON",
                    "brightness temp",
                ],  # No `rows` argument so the sample will be the full data
            },
        )

        # Plot the data obtained from the server
        plot_json_data(loaded.data)

        # ---------- summarize_dataset ----------
        # Call the `descriptive_stats_dataset` tool, specifying the subset of interest
        stats = await mcp_client.call_tool(
            "descriptive_stats_dataset",
            {
                "dataset": "amsu",  # Doesn't need to be an exact dataset name
                "time": "2021-01-01",
                "vars": [  # Variable names don't need to be exact
                    "Latitude",
                    "LON",
                    "brightness temp",
                    "brightness temp 2",
                    "brightness temp 3",
                    "brightness temp 4",
                    "brightness temp 5",
                ],
            },
        )

        # Read the returned data as a literal JSON string
        json_stats = StringIO(stats.data)

        # Convert the list of dictionaries to a DataFrame
        stats_df = pd.read_json(json_stats)

        # Print the accessed statistical data
        print("Descriptive Statistics:")
        print(stats_df)

        # ---------- correlation_matrix_dataset ----------
        # Call the `correlation_matrix_dataset` tool, specifying the subset of interest
        stats = await mcp_client.call_tool(
            "correlation_matrix_dataset",
            {
                "dataset": "amsu",  # Doesn't need to be an exact dataset name
                "time": "2021-01-01",
                "vars": [  # Variable names don't need to be exact
                    "Latitude",
                    "LON",
                    "brightness temp",
                    "brightness temp 2",
                    "brightness temp 3",
                    "brightness temp 4",
                    "brightness temp 5",
                ],
            },
        )

        # Read the returned data as a literal JSON string
        json_correlation_matrix = StringIO(stats.data)

        # Convert the list of dictionaries to a DataFrame
        correlation_matrix_df = pd.read_json(json_correlation_matrix)

        # Print the accessed correlation matrix DataFrame
        print("Correlation Matrix:")
        print(correlation_matrix_df)


# Function to plot data from JSON format
def plot_json_data(json_data: str):
    """Plot JSON data representing a pandas DataFrame using Matplotlib.

    Args:
        json_data (str): A string of JSON data in records orientation representing data to plot.
    """
    # Convert Gemini's JSON output into a DataFrame
    df = pd.read_json(StringIO(json_data), orient="records")

    # Plot the data obtained from the query
    plot_col = df.columns[2]
    plt.figure(figsize=(12, 8))
    plt.scatter(df["LON"], df["LAT"], s=2, c=df[plot_col])
    plt.title(f"AMSU Brightness Temperature for {plot_col}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
