"""Integration check: every VARIABLE_ALIASES entry points at a variable its dataset has.

Guards against alias-vs-reality drift -- e.g. server B5, where the ATMS
`brightness_temperature` / `temperature` alias pointed at `BRITCSTC.TMBR_00001`, an ID
ATMS doesn't have (it uses `ATMSCH.TMBR_00001`). Reads the running server's
`data://variable-aliases` resource and, for each per-dataset `alias -> variable id`,
confirms that ID appears in the dataset's `variables_info`. The `DEFAULT` ids
(LAT/LON/OBS_DATE) are universal columns, not per-dataset mappings, so they're skipped.

Unlike a catalog-free unit test (which would only assert the table equals a hand-copied
value), this validates the table against the *real* datasets on the running server -- the
one property that would have caught B5. It requires the server in HTTP mode (README.md).

    uv run testing/integration_aliases.py
"""

import asyncio
import json

from _client import DEFAULT_URL
from fastmcp import Client
from mcp.types import TextResourceContents


async def _find_bad_aliases(url: str) -> list[str]:
    """Return a list of `alias[dataset] -> id` strings whose id is missing from the dataset."""
    client = Client(url)
    async with client:
        print(f"Connected: {client.is_connected()}")

        # The alias table, as the running server actually has it.
        contents = await client.read_resource("data://variable-aliases")
        content = contents[0]
        if not isinstance(content, TextResourceContents):
            raise SystemExit(
                "data://variable-aliases returned unexpected non-text content"
            )
        table = json.loads(content.text)

        info_by_dataset: dict[str, str] = {}  # cache variables_info per dataset
        failures: list[str] = []
        checked = 0
        for alias, per_dataset in table.items():
            for dataset, var_id in per_dataset.items():
                if dataset == "DEFAULT":
                    continue
                if dataset not in info_by_dataset:
                    result = await client.call_tool(
                        "variables_info", {"dataset": dataset}
                    )
                    info_by_dataset[dataset] = str(result.data)
                checked += 1
                # variables_info renders ids quoted, e.g. NNJAVariable("BRITCSTC.TMBR_00001" ...)
                if f'"{var_id}"' not in info_by_dataset[dataset]:
                    failures.append(f"{alias}[{dataset}] -> {var_id}")

        print(
            f"Checked {checked} alias->variable mapping(s) across "
            f"{len(info_by_dataset)} dataset(s)."
        )
        return failures


def main() -> None:
    try:
        failures = asyncio.run(_find_bad_aliases(DEFAULT_URL))
    except RuntimeError as e:
        if "failed to connect" not in str(e):
            raise  # a real error, not a down server
        raise SystemExit(
            f"Could not reach the MCP server at {DEFAULT_URL}. Start it in HTTP mode first:\n"
            "- PowerShell: $env:MCP_TRANSPORT='http'; uv run server.py\n"
            "- bash/zsh:   MCP_TRANSPORT=http uv run server.py\n"
            "- Docker:     docker build -t nnja-ai-mcp .\n"
            "              docker run -p 8000:8000 nnja-ai-mcp"
        ) from e

    if failures:
        listed = "\n".join(
            f"  - {f}  (not a variable in this dataset)" for f in failures
        )
        raise SystemExit(
            "Alias-integrity check FAILED -- these VARIABLE_ALIASES entries point at "
            f"variables their dataset doesn't have:\n{listed}"
        )
    print("OK: every alias maps to a variable its dataset actually has.")


main()
