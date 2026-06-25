from _client import run_tool

run_tool(
    "calculate_spectral_index",
    {
        "dataset": "seviri-sevasr-NC021042",
        "time": "2024-11-20",
        "index_name": "cloud_cooling",
        "lat_bounds": [35, 50],  # Western Europe
        "lon_bounds": [-5, 15],
        "end_time": "2024-11-21",
    },
)
