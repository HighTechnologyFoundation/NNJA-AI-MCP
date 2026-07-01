"""Run every integration_*.py script once against a live server and tally the results.

A one-command confidence check: executes each `testing/integration_*.py` in sequence
(each drives the real MCP server end to end), catches the `SystemExit` a script raises
on failure, and prints a PASS/FAIL summary. Exits non-zero if any script fails.

Requires the server running in HTTP mode at http://localhost:8000/mcp (see README.md).
This is a manual/local check -- it is NOT part of the pytest suite or CI, which run
without a server. (Named `integration_*` so pytest doesn't collect it either.)

    uv run testing/integration_all.py
"""

import pathlib
import runpy
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
SELF = pathlib.Path(__file__).name

# Discover the integration scripts. The glob matches this runner too, so exclude self
# (otherwise it would recurse into itself).
scripts = sorted(p for p in HERE.glob("integration_*.py") if p.name != SELF)

results: list[tuple[str, bool, str]] = []  # (script name, passed, failure detail)

print(f"Running {len(scripts)} integration script(s) against the live server...\n")

for script in scripts:
    print("=" * 72)
    print(f"RUN  {script.stem}")
    print("=" * 72)
    try:
        # runpy executes the file exactly like `python testing/<script>.py`: its top-level
        # run_tool(...) / expect_error(...) calls fire here. `testing/` is already on
        # sys.path (this runner was launched from it), so the scripts' `from _client
        # import ...` resolves.
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        # Scripts report failures (and an unreachable server) by raising SystemExit(msg).
        detail = str(e)
        results.append((script.stem, False, detail))
        print(f"[FAIL] {script.stem}: {detail}\n")
        # A down server fails every remaining script identically -- stop and say so.
        if "Could not reach the MCP server" in detail:
            print(
                "Server unreachable -- aborting the rest. "
                "Start it first (see README.md).\n"
            )
            break
    except Exception as e:
        # An error outside the scripts' SystemExit contract -- isolate it and continue.
        results.append((script.stem, False, f"unexpected {type(e).__name__}: {e}"))
        traceback.print_exc()
        print(f"[FAIL] {script.stem}: unexpected error\n")
    else:
        results.append((script.stem, True, ""))
        print(f"[PASS] {script.stem}\n")

failed = [(name, detail) for name, ok, detail in results if not ok]
print("=" * 72)
print(
    f"Summary: {sum(ok for _, ok, _ in results)} passed, {len(failed)} failed, "
    f"{len(scripts) - len(results)} not run (of {len(scripts)})"
)
for name, detail in failed:
    print(f"  [FAIL] {name}: {detail}")

sys.exit(1 if failed else 0)
