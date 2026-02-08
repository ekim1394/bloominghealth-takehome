# Multi-stage build for the Blooming Health FastAPI service
# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Put venv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY app/ ./app/

# Copy data directory (calls.json for prediction training)
COPY data/ ./data/

# Create mlruns directory for MLflow tracking
RUN mkdir -p /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
