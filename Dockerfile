FROM python:3.11-slim AS builder

# Install git (needed for setuptools-scm) and uv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy entire git repo for version detection
COPY .git/ ./.git/

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Copy source code
COPY src/ ./src/

# Install dependencies and build
RUN uv pip install --system --no-cache .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create output directories
RUN mkdir -p predictions metrics plots

# Set Python path
ENV PYTHONUNBUFFERED=1

# Default command: run all models with 2 forecast dates
CMD ["election-forecast", "--dates", "2"]
