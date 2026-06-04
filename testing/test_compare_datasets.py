import asyncio
import time

from fastmcp import Client

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        start = time.time()

        res = await mcp_client.call_tool(
            "compare_datasets",
            {
                "datasets": [
                    "amsua-1bamua-NC021023",
                    "atms-atms-NC021203",
                    "mhs-1bmhs-NC021027",
                ],
                "time": "2023-01-01",
                "end_time": "2023-01-02",
                "variables": ["brightness temperature"],
                "lat_bounds": [30, 40],
                "lon_bounds": [-120, -110],
            },
        )

        end = time.time()
        print(f"Time taken: {end - start:.2f} seconds")

        print(res)


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
