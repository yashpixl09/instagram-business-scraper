"""`POST /api/search/run` -- the synchronous run a frontend's "run now" button needs.

Unlike `POST /api/search` (proven elsewhere to enqueue and spend nothing), this endpoint
is supposed to actually search -- against fixtures here, never a live provider. See
`Engine.execute_search_now`'s docstring for why this exists and the tradeoff it accepts.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from lead_engine import cli
from lead_engine.api.app import create_app
from lead_engine.api.deps import Engine
from lead_engine.config import Settings
from lead_engine.db.migrate import apply_migrations
from lead_engine.geo.resolver import SeedResolver

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")
CORPUS = Path(__file__).parent / "fixtures" / "searchapi"
CITY, AREA = "Bangalore", "Indiranagar"

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN, reason="needs psycopg and LEAD_ENGINE_TEST_DSN"
)


def search_body(**overrides) -> dict:
    body = {
        "location": {"city": CITY, "areas": [AREA]},
        "niches": ["salon"],
        "limit": 10,
        "use_ai": False,
    }
    body.update(overrides)
    return body


@pytest.fixture
def pool():
    schema = "test_" + uuid.uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f'CREATE SCHEMA "{schema}"')
    with psycopg.connect(DSN) as connection:
        connection.execute(f"SET search_path = {schema}, public")
        apply_migrations(connection)
        connection.commit()

    made = ConnectionPool(
        DSN, min_size=1, max_size=4, open=True,
        kwargs={"options": f"-c search_path={schema},public"},
    )
    try:
        yield made
    finally:
        made.close()
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def fixture_client(pool, *, budget=None) -> TestClient:
    settings = Settings(lead_engine_dsn=DSN, searchapi_key="sk-live-run-now-test")
    engine = Engine(
        settings,
        pool=pool,
        maps_provider=cli.fixture_provider(CORPUS),
        resolver=SeedResolver(),
    )
    if budget is None:
        engine.ensure_budget()
    return TestClient(create_app(settings, engine=engine), raise_server_exceptions=False)


@needs_postgres
def test_run_now_actually_searches_and_returns_a_completed_run(pool):
    with fixture_client(pool) as client:
        response = client.post("/api/search/run", json=search_body())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["stats"]["searches_spent"] == 1
    assert body["stats"]["new_businesses"] > 0


@needs_postgres
def test_run_now_actually_wrote_businesses_to_the_database(pool):
    with fixture_client(pool) as client:
        client.post("/api/search/run", json=search_body())

    with pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM businesses").fetchone()[0]
    assert count > 0


@needs_postgres
def test_run_now_refuses_a_plan_that_exceeds_the_remaining_budget(pool):
    settings = Settings(lead_engine_dsn=DSN, searchapi_key="sk-live-run-now-test-2")
    engine = Engine(
        settings,
        pool=pool,
        maps_provider=cli.fixture_provider(CORPUS),
        resolver=SeedResolver(),
    )
    # Seed a ledger with nothing left, rather than the default 50.
    engine.ensure_budget()
    with pool.connection() as conn:
        conn.execute("UPDATE search_budget SET limit_total = used")

    with TestClient(create_app(settings, engine=engine), raise_server_exceptions=False) as client:
        response = client.post("/api/search/run", json=search_body())

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "budget_exhausted"
    with pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM businesses").fetchone()[0]
    assert count == 0  # refused before anything was searched
