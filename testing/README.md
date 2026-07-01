# `testing/`

Two kinds of tests live here, told apart by filename.

## Unit tests — `test_*.py`

Fast, hermetic `pytest` tests. **No server and no network** - they use fakes
(`_fakes.py`) or hand-built DataFrames and monkeypatch the catalog / `_gated_access`.

```bash
uv run pytest
```

They cover the `mcp_client/` package (gateway caching, the Gemini query pipeline, CLI
parsing, Ctrl-C handling, elicitation) and `server.py`'s catalog-free logic (variable
resolution, the large-load gate, the async data tools, and the scientific
categorization / trend functions). This is what CI runs.

## Integration scripts — `integration_*.py`

Standalone scripts that drive the **real** server end to end. Each one needs the MCP
server already running in **HTTP mode** at `http://localhost:8000/mcp`.

```bash
# 1. Start the server first, in another terminal:
MCP_TRANSPORT=http uv run server.py
#    PowerShell:  $env:MCP_TRANSPORT="http"; uv run server.py
#    or Docker:   docker build -t nnja-ai-mcp . && docker run -p 8000:8000 nnja-ai-mcp

# 2. Run any script (list them with: ls testing/integration_*.py):
uv run testing/integration_lapse_rate.py
```

They pull real data from GCS, so some are slow. A server that isn't running is the #1
reason a script "fails" - each prints a friendly "Could not reach the MCP server ..."
hint when it can't connect.

## Helpers — `_*.py`

`_client.py` (`run_tool` / `expect_error`, used by the integration scripts) and
`_fakes.py` (fakes plus the `sync` decorator for the unit tests). The leading
underscore keeps `pytest` from collecting them as test modules.

## Why the naming matters

`uv run pytest` collects only `test_*.py` (pytest's default `python_files`). The
`integration_*.py` scripts call `run_tool(...)` at import time and would fail without a
live server, so it's important that they are **not** collected - the `integration_`
prefix is what keeps them out of the pytest run (and therefore out of CI). Don't rename
an integration script to `test_*` unless you want CI to try to execute it against a
server that isn't there.
