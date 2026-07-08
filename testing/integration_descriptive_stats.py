from _client import run_tool

run_tool(
    "descriptive_stats_dataset",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "2023-07-01",
        "variables": ["brightness temperature"],
        "lat_bounds": [30, 40],  # US Southwest (desert)
        "lon_bounds": [-120, -110],
    },
)
