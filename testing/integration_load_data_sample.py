from _client import run_tool

run_tool(
    "load_data_sample",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "2023-07-01",
        "variables": ["brightness temperature"],
        "lat_bounds": [30, 40],  # US Southwest (desert)
        "lon_bounds": [-120, -110],
        "rows": 10,
    },
)
