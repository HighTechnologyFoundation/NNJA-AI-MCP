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
        [print(f"   - {tool.name}: {tool.description}") for tool in tools]

        # List the datasets available from the MCP server
        datasets = await mcp_client.call_tool("available_datasets")
        print(datasets.data)

        # Call the `load_dataset` tool, specifying the subset of interest
        loaded = await mcp_client.call_tool(
            "load_dataset",
            {
                "dataset": "amsu",  # Doesn't need to be an exact dataset name
                "time": "2021-01-01",
                "vars": ["LAT", "LON", "BRITCSTC.TMBR_00001"],
            },
        )
        # Read the returned data as a literal JSON string
        json_loaded = StringIO(loaded.data)

        # Make sure there is data to handle
        if json_loaded:
            # Convert the list of dictionaries to a DataFrame
            df = pd.read_json(json_loaded, orient="records")
            # Print the first few rows to verify the data has been accessed
            print(df.head())

            # Plot the data obtained from the server
            plot_col = "BRITCSTC.TMBR_00001"
            plt.figure(figsize=(12, 8))
            plt.scatter(df["LON"], df["LAT"], s=15, c=df[plot_col])
            plt.title(f"AMSU Brightness Temperature for {plot_col}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
