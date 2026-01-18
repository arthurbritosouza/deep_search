FROM python:3.12-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --no-dev excludes development dependencies if any
RUN uv sync --no-dev

# Copy the rest of the application
COPY . .

# Set the path to include the virtual environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the application
CMD ["python", "main.py"]
