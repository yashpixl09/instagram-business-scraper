web: sh -c "python -m lead_engine.db.migrate && uvicorn --factory lead_engine.api.app:create_app --host 0.0.0.0 --port ${PORT:-8000}"
