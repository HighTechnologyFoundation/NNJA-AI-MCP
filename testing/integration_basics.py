from _client import run_tool

# Tests for the lightweight tools the rest of the integration suite skips:
# citation and the metadata lookups (no heavy GCS data pull).
run_tool("cite_data", {})
run_tool("available_datasets", {})
run_tool("dataset_info", {"dataset": "amsua"})
run_tool("variables_info", {"dataset": "amsua-1bamua-NC021023"})
