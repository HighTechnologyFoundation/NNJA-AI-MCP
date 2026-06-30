"""The NNJA-AI server contract this CLI depends on.

The CLI is a generic MCP client at the transport layer -- `MCPClient` connects to any
server by path or URL, and the gateway/completer/elicitation plumbing is server-agnostic.
On top of that it adds a thin layer of NNJA-aware conveniences: expanding an `@mentioned`
dataset name into a summary, and recognizing/labelling the dataset-list resource for
auto-completion. Those conveniences assume the specific tool and resource names the
NNJA-AI server (`server.py`) exposes.

Those assumptions are collected here so they live in one place: this is the file to
re-point (or copy and edit) if the client is ever forked for a different MCP server.
"""

# The dataset-list resource. The NNJA server exposes it at `data://datasets` with the
# resource name `list_datasets`, so the client recognizes it two ways: by a substring of
# the resource *name* (the @mention fallback and completion) and by its URI *prefix* (the
# completion-menu label).
DATASET_LIST_RESOURCE_HINT = "datasets"
DATASET_LIST_RESOURCE_URI_PREFIX = "data://datasets"
DATASET_META_LABEL = "Dataset"

# The per-dataset detail tool, used to expand an @mentioned dataset name into a summary.
DATASET_INFO_TOOL = "dataset_info"
DATASET_INFO_ARG = "dataset"
