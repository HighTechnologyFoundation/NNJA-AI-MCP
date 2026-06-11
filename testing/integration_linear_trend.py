from _client import run_tool

run_tool(
    "calculate_trend",
    {
        "dataset": "amsua-1bamua-NC021023",
        "start_time": "2023-07-01",
        "end_time": "2023-07-31",
        "variable": "brightness temperature",
        "lat_bounds": [30, 40],
        "lon_bounds": [-120, -110],
    },
)
