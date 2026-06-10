from _client import run_tool

run_tool(
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
