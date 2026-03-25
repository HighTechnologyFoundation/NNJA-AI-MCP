FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* .

# Install dependencies using uv
RUN uv venv
RUN uv sync --locked

# Copy application code
COPY server.py .

# Expose the port the server runs on
EXPOSE 8000

# Set the environment variable so the server knows it's in a Docker container
ENV MCP_TRANSPORT=http

# Command to run the server
CMD ["uv", "run", "server.py"]
