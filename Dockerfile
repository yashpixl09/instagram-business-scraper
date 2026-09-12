# Multi-stage Dockerfile for Lead Engine on Railway
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into /install
COPY pyproject.toml README.md ./
COPY lead_engine ./lead_engine

RUN pip install --no-cache-dir --prefix=/install .

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code, migrations, and entrypoint
COPY pyproject.toml ./
COPY lead_engine ./lead_engine
COPY migrations ./migrations
COPY env.example ./
COPY entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
