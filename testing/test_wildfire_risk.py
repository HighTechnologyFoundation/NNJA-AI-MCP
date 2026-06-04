import asyncio

from fastmcp import Client

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        res = await mcp_client.call_tool(
            "calculate_spectral_index",
            {
                "dataset": "seviri-sevasr-NC021042",
                "time": "2024-08-15",
                "index_name": "wildfire_risk",
                "lat_bounds": [15, 30],
                "lon_bounds": [0, 15],
                "end_time": "2024-08-16",
            },
        )

        print(res)


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
