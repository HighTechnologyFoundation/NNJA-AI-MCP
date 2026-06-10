<!-- The PR title becomes the squash commit message — make it a Conventional Commit,
     e.g. `feat(cli): support connecting to an HTTP MCP server`. -->

## What & why

<!-- What does this change do, and why? Link related context (e.g. a temp.md review item) if any. -->

## How tested

<!-- Commands run / manual verification — e.g. `uvx ty check`, ran the CLI against an HTTP server. -->

## Checklist

- [ ] PR title is a Conventional Commit (it becomes the squash commit message)
- [ ] Scope is one logical change (unrelated changes split into separate PRs)
- [ ] `uvx ruff format --check` and `uvx ruff check` pass
- [ ] `uvx ty check` passes
- [ ] Noted above if the change is **not** behavior-preserving
