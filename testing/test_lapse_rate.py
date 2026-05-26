from fastmcp import Client
import asyncio

mcp_client = Client("http://localhost:8000/mcp")


async def main():
    async with mcp_client:
        print(f"Connected: {mcp_client.is_connected()}")

        res = await mcp_client.call_tool(
            "calculate_lapse_rate",
            {
                "time": "2023-07-16",
                "lat_bounds": [30, 40],  # desert: 30, 40 | tropical: 10, 30
                "lon_bounds": [-120, -110],  # desert: -120, -110 | tropical: -85, -50
                "level1_hpa": 925,  # 1000 | 1000 | 925 | 850
                "level2_hpa": 850,  #  850 |  500 | 850 | 500
            },
        )

        print(res)


# Run the client when this Python file runs
if __name__ == "__main__":
    asyncio.run(main())
