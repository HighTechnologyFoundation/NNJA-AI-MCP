FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Create non-root user
RUN useradd --create-home --uid 1000 appuser

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install server dependencies using uv
RUN uv sync --locked --no-default-groups

# Copy application code
COPY server.py .

# Run as a non-root user (least privilege)
USER appuser

# Expose the port the server runs on
EXPOSE 8000

# Set the environment variable so the server knows it's in a Docker container
ENV MCP_TRANSPORT=http

# Command to run the server
CMD ["/app/.venv/bin/python", "server.py"]
