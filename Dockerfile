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

# Copy application code and migrations
COPY pyproject.toml ./
COPY lead_engine ./lead_engine
COPY migrations ./migrations
COPY env.example ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-c", "import os, subprocess, sys; subprocess.run([sys.executable, '-m', 'lead_engine.db.migrate'], check=False); subprocess.run([sys.executable, '-m', 'uvicorn', '--factory', 'lead_engine.api.app:create_app', '--host', '0.0.0.0', '--port', os.environ.get('PORT', '8000')])"]
