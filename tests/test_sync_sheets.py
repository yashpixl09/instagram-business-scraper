"""`Engine.sync_sheets`, the CLI flag, and the API route: the piece that lets an operator
run the whole loop -- discover, enrich, sync -- without anyone hand-running a script.

Real Postgres is required (this method issues raw SQL through `Engine._select` and
`require_pool`, which a pure fake repository cannot answer); Google Sheets is not --
`FakeSheetsClient` from `test_sheets.py` is injected via `Engine(..., sheets_client=...)`,
the same seam a real caller with its own auth would use.

NO TEST HERE REACHES GOOGLE OR SPENDS A SEARCHAPI CREDIT.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg_pool import ConnectionPool

from lead_engine import cli
from lead_engine.api.deps import Engine
from lead_engine.api.schemas import ApiProblem
from lead_engine.config import Settings
from lead_engine.db.migrate import apply_migrations
from lead_engine.geo.resolver import SeedResolver
from tests.test_sheets import FakeSheetsClient, bare_row, seeded

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")
CORPUS = Path(__file__).parent / "fixtures" / "searchapi"

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)

KEY = "sk-live-sync-sheets-test-key"


def make_settings(**overrides) -> Settings:
    base = {"lead_engine_dsn": DSN, "searchapi_key": KEY}
    base.update(overrides)
    return Settings(**base)


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
        DSN,
        min_size=1,
        max_size=4,
        open=True,
        kwargs={"options": f"-c search_path={schema},public"},
    )
    try:
        yield made
    finally:
        made.close()
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def seed_business(pool, *, name: str, place_id: str, total: int = 80) -> uuid.UUID:
    business_id = uuid.uuid4()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO businesses (id, place_id, name, niche_id, city, country, state,"
            " address, lat, lng)"
            " VALUES (%s, %s, %s, 'salon', 'Bangalore', 'India', 'Karnataka', 'x', 1, 1)",
            (business_id, place_id, name),
        )
        conn.execute(
            "INSERT INTO scores (business_id, scorer_version, signals, evidence, total)"
            " VALUES (%s, 'google-only', '{}', '{}', %s)",
            (business_id, total),
        )
    return business_id


def engine_for(pool, sheets_client) -> Engine:
    return Engine(make_settings(), pool=pool, sheets_client=sheets_client)


@needs_postgres
def test_a_fresh_business_is_appended_to_master(pool):
    seed_business(pool, name="Cake Bee", place_id="place-1")
    client = FakeSheetsClient()
    engine = engine_for(pool, client)

    result = engine.sync_sheets()

    assert result.rows_appended == 1
    assert result.verdicts_read == 0


@needs_postgres
def test_a_verdict_already_on_the_sheet_is_written_into_verdicts(pool):
    business_id = seed_business(pool, name="Cake Bee", place_id="place-1")
    client = FakeSheetsClient()
    seeded(client, [bare_row("place-1", "Cake Bee", my_verdict="high", notes="called")])
    engine = engine_for(pool, client)

    result = engine.sync_sheets()

    assert result.verdicts_read == 1
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT my_verdict, notes FROM verdicts WHERE business_id = %s", (business_id,)
        ).fetchone()
    assert row == ("high", "called")


@needs_postgres
def test_a_blank_row_on_the_sheet_writes_no_verdict(pool):
    seed_business(pool, name="Cake Bee", place_id="place-1")
    client = FakeSheetsClient()
    seeded(client, [bare_row("place-1", "Cake Bee")])  # no verdict typed yet
    engine = engine_for(pool, client)

    result = engine.sync_sheets()

    assert result.verdicts_read == 0
    with pool.connection() as conn:
        count = conn.execute("SELECT count(*) FROM verdicts").fetchone()[0]
    assert count == 0


@needs_postgres
def test_sheets_not_configured_raises_a_clear_problem(pool):
    engine = Engine(make_settings(), pool=pool)  # no sheets_client, none configured
    with pytest.raises(ApiProblem) as caught:
        engine.sync_sheets()
    assert caught.value.code == "sheets_not_configured"


@needs_postgres
def test_the_cli_flag_reports_the_sync(pool, capsys):
    seed_business(pool, name="Cake Bee", place_id="place-1")
    client = FakeSheetsClient()
    # Passing engine= to cli.main bypasses --fixtures entirely (that flag is
    # build_cli_engine's concern, never consulted when an engine is pre-built) -- so the
    # fixture provider and offline resolver are wired here explicitly, the same way
    # test_cli.py's fixture_engine() does for exactly this reason.
    engine = Engine(
        make_settings(),
        pool=pool,
        maps_provider=cli.fixture_provider(CORPUS),
        resolver=SeedResolver(),
        sheets_client=client,
    )
    engine.ensure_budget()

    code = cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--niche", "salon",
            "--fixtures", str(CORPUS),
            "--sync-sheets",
        ],
        settings=make_settings(),
        engine=engine,
    )

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert "Google Sheets sync:" in captured.out
