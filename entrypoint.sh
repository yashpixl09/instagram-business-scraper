#!/bin/sh
set -e

echo "Applying database migrations..."
python -m lead_engine.db.migrate

echo "Starting Uvicorn web server..."
exec uvicorn --factory lead_engine.api.app:create_app --host 0.0.0.0 --port "${PORT:-8000}"
