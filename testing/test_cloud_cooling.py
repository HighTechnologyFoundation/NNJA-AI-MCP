from fastmcp import Client
import asyncio

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        res = await mcp_client.call_tool(
            "calculate_spectral_index",
            {
                "dataset": "seviri-sevasr-NC021042",
                "time": "2024-11-20",
                "index_name": "cloud_cooling",
                "lat_bounds": [35, 50],
                "lon_bounds": [-5, 15],
                "end_time": "2024-11-21",
            },
        )

        print(res)


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
