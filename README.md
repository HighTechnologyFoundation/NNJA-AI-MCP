# NNJA-AI-MCP

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Model--Context--Protocol-orange)](https://modelcontextprotocol.io)

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server
implementation for accessing **NOAA NASA Joint Archive (NNJA)** datasets.
This server enables LLMs to programatically explore, load, and analyze
Brightband reprocessed NNJA data (NNJA-AI).

## Available Tools

The MCP server exposes the following tools to connected LLMs:

| Tool | Description |
| :--- | :--- |
| `available_datasets` | Lists all available NNJA-AI datasets. |
| `dataset_info` | Provides a summary/metadata for a specific dataset. |
| `variables_info` | Lists variables and descriptions within a dataset. |
| `load_data_sample` | Loads a subset of data as a JSON/DataFrame-ready format. |
| `descriptive_stats_dataset` | Returns statistical summaries (mean, std, etc.). |
| `correlation_matrix_dataset`| Computes correlation matrices for dataset variables. |
| `cite_data` | Generates the correct citation for the data accessed. |

> Pro Tip: The server uses Fuzzy Matching for dataset and variable names. If you
(or the AI) make a typo, the server will automatically find the closest valid match!

## Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Highly recommended for dependency management)
- A Google Gemini API Key (for the AI-powered clients)

### Installation & Setup

1. Clone the repository

1. Configure environment variables:

    ```bash
    cp .env.template .env
    # Open .env and add your GEMINI_API_KEY:
    # GEMINI_API_KEY=your_key_here
    ```

1. Install dependencies (and the CLI client):

    ```bash
    uv sync
    ```

1. Run the AI chat

    The fastest way to test the server is with the built-in interactive CLI:

    ```bash
    uv run mcp-client --chat

    # Alternatively, you can activate the virtual environment and then run the CLI:
    source .venv/bin/activate      # Linux/macOS
    source .venv\Scripts\activate  # Windows
    mcp-client --chat
    ```

## Example Clients

This repository includes three different ways to interact with the MCP server:

1. `mcp-client/` (CLI): An interactive, AI-powered chat interface. Best for
natural language exploration of the data.
1. `client.py`: A scripted AI agent that performs a specific query and exits.
Useful for seeing how to integrate MCP into your own Python automation.
1. `simple-client.py`: A "manual" client that calls MCP tools directly without
an LLM. Use this to see how to programmatically convert tool outputs into
Pandas DataFrames for traditional data science.

## Running with Docker

The server can be containerized for easier deployment or integration
with other MCP clients.

```bash
docker build -t mcp-server .        # Build the Docker image
docker run -p 8000:8000 mcp-server  # Run the Docker container
```

## Project Structure

```text
.
├── mcp_client/       # Source code for the CLI chat interface
├── server.py         # The core MCP Server (Tools, Resources, Prompts)
├── client.py         # Scripted AI agent example
├── simple-client.py  # Manual tool-use & Pandas integration example
├── Dockerfile        # Container configuration
└── pyproject.toml    # Project dependencies and CLI entry points
```

## NNJA-AI Data Citation

Reprocessed NNJA data from:
[![DOI](https://zenodo.org/badge/899259654.svg)](https://doi.org/10.5281/zenodo.14633508)
