from _client import expect_error

# Negative smoke tests: lock in the "Error: ..." contract for common bad inputs.
# Each asserts the tool returns a friendly error string containing the expected
# phrase; expect_error fails loudly on an unexpected success or a mismatched message.
# All four are cheap -- they fail before any heavy partition load.

# Unknown dataset name -> _resolve_dataset reports no match.
expect_error(
    "load_data_sample",
    {
        "dataset": "nonexistent-dataset-xyz",
        "time": "2023-07-01",
        "variables": ["latitude"],
    },
    "No dataset matching",
)

# Out-of-range latitude bounds -> _access_dataset bound validation.
expect_error(
    "load_data_sample",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "2023-07-01",
        "variables": ["latitude"],
        "lat_bounds": [200, 300],
    },
    "Latitude bound values must be between",
)

# A date with no partitions -> EmptyTimeSubsetError handled as a recoverable "no data".
expect_error(
    "load_data_sample",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "1900-01-01",
        "variables": ["latitude"],
    },
    "No data found",
)

# A dataset the spectral index doesn't support.
expect_error(
    "calculate_spectral_index",
    {
        "dataset": "amsua-1bamua-NC021023",
        "time": "2023-07-01",
        "index_name": "wildfire_risk",
    },
    "not supported for index calculation",
)

# Unknown dataset name on the spectral index -> the guarded resolve returns the
# friendly "No dataset matching" string, not a hard tool error. Regression guard:
# calculate_spectral_index used to resolve the dataset outside its try/except, so a
# bad name escaped as an isError result instead of the recoverable string every other
# tool returns for the same typo.
expect_error(
    "calculate_spectral_index",
    {
        "dataset": "nonexistent-dataset-xyz",
        "time": "2023-07-01",
        "index_name": "wildfire_risk",
    },
    "No dataset matching",
)
