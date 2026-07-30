# NNJA-AI-MCP — natural-language access to NOAA science data, for AI agents

**What it is.** A working proof of concept of a reusable pattern: an **MCP server** that wraps
NOAA/NASA Joint Archive (NNJA) observational data, driven by an **AI agent** in plain English.
Ask a question — the agent finds the dataset, subsets it in space and time, resolves variables
by everyday name, runs domain analyses, and explains the result. No code, no memorizing
variable IDs.

**Why it matters here.** The *architecture* is the transferable part, not the obs data. Swap
the observation tools for forecast tools and the same pattern is an agent driving a
**nested-EAGLE MCP server**.

## The pattern

```
  You (plain English)
        │
        ▼
   AI agent (LLM)  ──calls──▶   MCP server   ──▶  data library  ──▶  cloud storage
   (Gemini today)    tools       (FastMCP)        (nnja-ai)          (GCS Parquet)
        ▲                            │
        └─────────── result ─────────┘
```

- **MCP (Model Context Protocol)** is an open, model-agnostic standard for exposing tools and
  resources to LLMs. The server declares *what* it can do; any MCP-capable agent can drive it.
- The agent and server are **decoupled**: change the model, change the data domain, or point
  one agent at several servers — the contract stays the same.

## What NNJA-AI-MCP actually provides

**Server** (`server.py`, FastMCP) — 14 NNJA datasets (satellite radiances: AMSU-A, ATMS, MHS,
CrIS, IASI, SEVIRI, GOES ABI, Himawari AHI; conventional: surface synoptic + radiosonde),
exposed as:

- **11 tools** — discovery (`available_datasets`, `dataset_info`, `variables_info`), data
  (`load_data_sample`, `descriptive_stats_dataset`, `correlation_matrix_dataset`,
  `calculate_trend`, `compare_datasets`), and domain science (`calculate_spectral_index` →
  wildfire risk / cloud phase, `calculate_lapse_rate`, `cite_data`).
- **2 resources** (`data://datasets`, `data://variable-aliases`) and **1 prompt**.
- Data streamed from **GCS Parquet** via the `nnja-ai` library. Runs over **stdio** or **HTTP**
  (containerized).

**Agent** (`mcp_client/`) — a terminal client bridging an LLM to the server: it runs the
tool-calling loop, resolves plain-English variable names (aliases + fuzzy match), handles
spatial/temporal subsetting, and confirms very large loads before fetching. It surfaces each
tool call live, so you can watch the agent drive the server.

## A real interaction

1. *"Assess wildfire risk from SEVIRI for 2023-07-01."*
2. Agent → `calculate_spectral_index(dataset=SEVIRI, index_name=wildfire_risk, time=2023-07-01)`
3. Server subsets the radiances, computes the shortwave–longwave index, classifies day/night
   risk, returns JSON.
4. Agent explains the distribution in plain language.

## How this maps to nested-EAGLE

| NNJA-AI-MCP (today) | A nested-EAGLE MCP server |
| :--- | :--- |
| obs / radiance datasets | forecast fields, ensemble members |
| `calculate_spectral_index`, `calculate_lapse_rate` | `run_forecast`, `fetch_field`, `ensemble_spread`, `verify_vs_analysis` |
| subset observations by space/time | subset forecasts by domain / lead time |
| **the same agent code drives it** | **the same agent code drives it** |

- **The agent doesn't change.** It discovers whatever tools a server declares — point it at a
  nested-EAGLE server and it drives forecast tools the same way.
- **One agent, many servers.** An agent can connect to several MCP servers at once — e.g. pull
  *observations* from an NNJA server and *forecasts* from a nested-EAGLE server in one
  conversation, and reason across both.
- **Building one is mostly domain design.** The MCP plumbing is small (the server is essentially
  a file of tool definitions over a data library); the real work is choosing the right tools and
  parameters for your model.

## Positioning (honest)

- **Cloud-neutral** — data on GCS Parquet today; the server is a container that deploys
  anywhere, Azure included.
- **Domain-specific** — obs/atmospheric, tabular data, plain-English variable names.
  Complementary to raster/gridded platforms (including Microsoft Planetary Computer's MCP
  tooling), which serve point/atmospheric observations poorly.
- **A proof of concept, not production** — the pattern is proven and runs today; authentication
  and scale come with a real deployment.
