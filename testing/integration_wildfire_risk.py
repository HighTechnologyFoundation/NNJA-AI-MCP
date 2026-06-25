from _client import run_tool

run_tool(
    "calculate_spectral_index",
    {
        "dataset": "seviri-sevasr-NC021042",
        "time": "2024-08-15",
        "index_name": "wildfire_risk",
        "lat_bounds": [15, 30],  # North Africa (Sahel)
        "lon_bounds": [0, 15],
        "end_time": "2024-08-16",
    },
)
