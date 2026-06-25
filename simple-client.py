import asyncio
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
from fastmcp import Client

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    try:
        async with mcp_client:
            print(f"Connected: {mcp_client.is_connected()}")

            # List the tools available from the MCP server
            tools = await mcp_client.list_tools()
            print("Available tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}\n")

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
                    "variables": [  # Variable names don't need to be exact
                        "Latitude",
                        "LON",
                        "brightness temperature",
                    ],
                },
            )

            # Plot the data obtained from the server
            plot_json_data(_require_data(loaded))

            # ---------- descriptive_stats_dataset ----------
            # Call the `descriptive_stats_dataset` tool, specifying the subset of interest
            stats = await mcp_client.call_tool(
                "descriptive_stats_dataset",
                {
                    "dataset": "amsu",  # Doesn't need to be an exact dataset name
                    "time": "2021-01-01",
                    "variables": [  # Variable names don't need to be exact
                        "Latitude",
                        "LON",
                        "brightness temperature",
                        "brightness temperature 2",
                        "brightness temperature 3",
                        "brightness temperature 4",
                        "brightness temperature 5",
                    ],
                },
            )

            # Read the returned data as a literal JSON string
            json_stats = StringIO(_require_data(stats))

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
                    "variables": [  # Variable names don't need to be exact
                        "Latitude",
                        "LON",
                        "brightness temperature",
                        "brightness temperature 2",
                        "brightness temperature 3",
                        "brightness temperature 4",
                        "brightness temperature 5",
                    ],
                },
            )

            # Read the returned data as a literal JSON string
            json_correlation_matrix = StringIO(_require_data(stats))

            # Convert the list of dictionaries to a DataFrame
            correlation_matrix_df = pd.read_json(json_correlation_matrix)

            # Print the accessed correlation matrix DataFrame
            print("Correlation Matrix:")
            print(correlation_matrix_df)
    except RuntimeError as e:
        if "failed to connect" not in str(e):
            raise  # a real error, not a down server
        raise SystemExit(
            "Could not reach the MCP server at http://localhost:8000/mcp.\n"
            "Start it in HTTP mode first:\n"
            "- PowerShell: $env:MCP_TRANSPORT='http'; uv run server.py\n"
            "- bash/zsh:   MCP_TRANSPORT=http uv run server.py\n"
            "- Docker:     docker build -t nnja-ai-mcp .\n"
            "              docker run -p 8000:8000 nnja-ai-mcp\n"
            "then re-run this script."
        ) from e


def _require_data(result):
    """Return the tool's data payload, exiting with its message if the tool failed.

    Tools report failures as an "Error: ..." string rather than raising, so check for
    that before feeding the payload to pd.read_json (which would otherwise choke on a
    non-JSON value with an opaque traceback).
    """
    if isinstance(result.data, str) and result.data.startswith("Error:"):
        raise SystemExit(result.data)
    return result.data


# Function to plot data from JSON format
def plot_json_data(json_data: str):
    """Plot JSON data representing a pandas DataFrame using Matplotlib.

    Args:
        json_data (str): A string of JSON data in records orientation representing data to plot.
    """
    # Convert Gemini's JSON output into a DataFrame
    df = pd.read_json(StringIO(json_data), orient="records")

    # The server returns columns in an unspecified order, so find the one we want
    value_cols = [col for col in df.columns if col not in ("LAT", "LON")]
    plot_col = value_cols[0]

    # Plot the data obtained from the query
    plt.figure(figsize=(12, 8))
    plt.scatter(df["LON"], df["LAT"], s=2, c=df[plot_col])
    plt.title(f"AMSU Brightness Temperature for {plot_col}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
