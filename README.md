# NNJA-AI-MCP

This repository contains code to create an Model-Context Protocol (MCP) server,
allowing Large Language Models (LLMs) to access data from the NOAA NASA
Joint Archive (NNJA) datasets. Specifically, the server provides access to the
Brightband reprocessed NNJA data (NNJA-AI).

This repository also includes 2 example clients that connect to the
MCP server and test out the available tools.

Make sure to install all dependencies before running the server or clients.

## Running the Server

The server file, `server.py`, can be run through Docker (preferred)
or directly by running the file.

### Running the Server with Docker

Note that in order to run the server with Docker, you will need to have installed:

- docker
- docker-buildx (replaces Docker's legacy builder)

```bash
# Step 1: Build the Docker image
docker build -t mcp-server .

# Step 2: Run the Docker container
docker run -p 8000:8000 mcp-server
```

### Running the Server File Directly

- Using uv:

  ```bash
  uv run server.py
  ```

- Using python:

  ```bash
  python3 server.py
  ```

## Running the Clients

There are 3 example client files provided:

- `mcp_client/` directory contains a CLI-based client, allowing
a user to chat with Gemini and giving it access to the MCP tools.
- `client.py` runs a query through Gemini, which accesses some of
the tools itself and provides the requested output. Note that
the AI model's input and output token limits restrict how
much data can be transferred through them.
- `simple-client.py` runs all of the MCP tools manually and shows how the
results of the `load_dataset`, `analyze_dataset`, and
`correlation_matrix_dataset` tools can be converted into a pandas DataFrame.

The client files can be run directly in a separate terminal, as shown below:

### Running `mcp_client/`

- Using uv:

```bash
# Setting up the client
uv sync
source .venv/bin/activate

# Using the client
mcp-client --help              # Shows help menu
mcp-client --members server.py # Lists the available tools, prompts, and resources
mcp-client --chat server.py    # Runs the AI-powered chat
```

### Running `client.py`

- Using uv:

  ```bash
  uv run client.py
  ```

- Using python:

  ```bash
  python3 client.py
  ```

### Running `simple-client.py`

- Using uv:

  ```bash
  uv run simple-client.py
  ```

- Using python:

  ```bash
  python3 simple-client.py
  ```

## NNJA-AI Data Citation

Reprocessed NNJA data from:
[![DOI](https://zenodo.org/badge/899259654.svg)](https://doi.org/10.5281/zenodo.14633508)
