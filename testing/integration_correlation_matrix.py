from _client import run_tool

# Three numeric columns (brightness temperature plus the auto-added LAT/LON from the
# spatial bounds) so the correlation matrix is non-trivial.
run_tool(
    "correlation_matrix_dataset",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "2023-07-01",
        "variables": ["brightness temperature", "latitude", "longitude"],
        "lat_bounds": [30, 40],  # US Southwest (desert)
        "lon_bounds": [-120, -110],
    },
)
