from fastmcp import Client
import asyncio
import pandas as pd

client = Client("http://localhost:8000/mcp")


async def main():
    async with client:
        print(f"Connected: {client.is_connected()}")

        # List the tools available from the MCP server
        tools = await client.list_tools()
        print("Available tools:", tools)
        [print(f"   - {tool.name}: {tool.description}") for tool in tools]

        # List the datasets available from the MCP server
        datasets = await client.call_tool("available_datasets")
        print(datasets.data)

        # Call the `load_dataset` tool, specifying the subset of interest
        loaded = await client.call_tool(
            "load_dataset",
            {
                "dataset": "amsu",  # Doesn't need to be an exact dataset name
                "time": "2021-01-01",
                "vars": ["LAT", "LON", "BRITCSTC.TMBR_00001"],
            },
        )
        # Read the returned data as JSON
        json_loaded = loaded.structured_content

        # Make sure there is data to handle
        if json_loaded is not None:
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(json_loaded["result"])
            # Print the first few rows to verify the data has been accessed
            print(df.head())


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
