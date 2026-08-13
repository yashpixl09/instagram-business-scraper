"""Migrations and the banding view, against a real Postgres.

Skipped unless `LEAD_ENGINE_TEST_DSN` is set. That is not politeness: roughly nine tenths
of this suite is pure and must keep running on a laptop with no Docker, on a train, in a
CI job that never provisions a database. A test file that turns the whole run red when
Postgres is absent gets deleted or `-k`-excluded within a week, and then it never runs at
all.

Isolation is a throwaway schema per test rather than TRUNCATE between tests. It is faster
(DDL on an empty schema beats scanning tables), it is parallel-safe, and it means a test
that leaves the transaction aborted cannot leak into the next one -- the schema it dirtied
no longer exists.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

try:
    import psycopg
    from psycopg.types.json import Jsonb

    from lead_engine.db.migrate import apply_migrations, migration_files
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        psycopg is None or not DSN,
        reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
    ),
]

# Explicit, so deleting a migration file fails here loudly instead of surfacing months
# later as a missing table in a worker nobody has written yet.
EXPECTED_TABLES = {
    # 0002 control
    "goals",
    "runs",
    "tasks",
    "events",
    # 0003 data
    "businesses",
    "contacts",
    "enrichments",
    "scores",
    "verdicts",
    "outreach",
    # 0004 discovery
    "search_cells",
    "geo_cache",
    "negative_cache",
    # 0005 automation
    "automation_opportunities",
    # 0006 memory
    "memories",
    "run_summaries",
    "agent_notes",
    # 0007 llm
    "llm_calls",
    "llm_rate_buckets",
    # 0010 budget
    "search_budget",
    # 0012 niche verification
    "niche_status",
    # the runner's own ledger
    "schema_migrations",
}

NOW = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def vector_extension():
    """Install pgvector once for the whole session.

    `CREATE EXTENSION` is database-scoped, not schema-scoped. Installing it per test schema
    would create and drop it dozens of times over, and each test schema's copy would vanish
    with `DROP SCHEMA CASCADE`. Installed once into `public` and kept on the search_path,
    `vector(1536)` resolves from every scratch schema and migration 0001 is a no-op there.
    """
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


@pytest.fixture
def conn(vector_extension):
    """A connection scoped to a schema that exists only for this test."""
    schema = "test_" + uuid.uuid4().hex
    connection = psycopg.connect(DSN)
    connection.execute(f'CREATE SCHEMA "{schema}"')
    # `public` stays on the path for the vector type only; every table created below lands
    # in the scratch schema, which is first.
    connection.execute(f'SET search_path = "{schema}", public')
    connection.commit()
    try:
        yield connection
    finally:
        connection.rollback()
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.commit()
        connection.close()


@pytest.fixture
def db(conn):
    """The scratch schema, fully migrated."""
    apply_migrations(conn)
    return conn


def current_schema(connection) -> str:
    return connection.execute("SELECT current_schema()").fetchone()[0]


def add_business(connection, *, name="Cafe Noir", niche="cafe", city="Bangalore", key=None):
    business_id = uuid.uuid4()
    connection.execute(
        "INSERT INTO businesses (id, name, niche_id, city, dedupe_key) VALUES (%s, %s, %s, %s, %s)",
        (business_id, name, niche, city, key),
    )
    return business_id


def add_score(connection, business_id, audience_index, *, at=NOW, version="google-only"):
    connection.execute(
        "INSERT INTO scores (business_id, scorer_version, signals, evidence,"
        " audience_index, scored_at) VALUES (%s, %s, '{}', '{}', %s, %s)",
        (business_id, version, audience_index, at),
    )


def add_reviews(connection, business_id, reviews, *, at=NOW, status="ok"):
    connection.execute(
        "INSERT INTO enrichments (business_id, source, status, data, fetched_at)"
        " VALUES (%s, 'google_maps', %s, %s, %s)",
        (business_id, status, Jsonb({"reviews": reviews}), at),
    )


def bands(connection) -> dict[uuid.UUID, tuple[str, str, int]]:
    rows = connection.execute(
        "SELECT business_id, audience_band, banding_method, cohort_size FROM lead_bands"
    ).fetchall()
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


# --- the runner -------------------------------------------------------------------------


def test_migrating_from_empty_creates_every_expected_table(db):
    found = {
        row[0]
        for row in db.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = %s AND table_type = 'BASE TABLE'",
            (current_schema(db),),
        )
    }
    assert found == EXPECTED_TABLES


def test_migrating_from_empty_creates_the_banding_view(db):
    views = {
        row[0]
        for row in db.execute(
            "SELECT table_name FROM information_schema.views WHERE table_schema = %s",
            (current_schema(db),),
        )
    }
    assert views == {"lead_bands"}


def test_a_second_run_applies_nothing(conn):
    first = apply_migrations(conn)
    assert first == [path.stem for path in migration_files()]
    assert apply_migrations(conn) == []


def test_the_ledger_matches_the_files_on_disk(db):
    recorded = [row[0] for row in db.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    assert recorded == [path.stem for path in migration_files()]


def test_two_runners_starting_together_do_not_race(conn):
    """Two workers booting at once must not both apply the schema.

    The assertion is on what each call *returned*, not on the state it left behind. The
    ledger is the wrong thing to check: ten rows and every table present is equally
    consistent with the lock working and with one runner having lost a race it should never
    have been in. Only the return values distinguish "one applied, one correctly did
    nothing" from "both tried" -- the same reason a budget ledger reading 100 says nothing
    about how many billed requests went out.

    Without the advisory lock this fails loudly rather than subtly: both runners read an
    empty ledger, both start applying, and the loser raises DuplicateTable partway through
    with half a schema behind it.
    """
    schema = current_schema(conn)

    def run() -> list[str]:
        with psycopg.connect(DSN) as other:
            other.execute(f'SET search_path = "{schema}", public')
            other.commit()
            return apply_migrations(other)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run), pool.submit(run)]
        results = sorted((future.result() for future in futures), key=len)

    assert results[0] == []
    assert results[1] == [path.stem for path in migration_files()]
    # And exactly one ledger row per file: no version recorded twice.
    recorded = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    assert recorded == len(migration_files())


def test_every_migration_is_numbered_and_unique():
    stems = [path.stem for path in migration_files()]
    assert stems == sorted(stems)
    assert len(set(stems)) == len(stems)
    assert all(stem[:4].isdigit() for stem in stems)


# --- dedupe key -------------------------------------------------------------------------


def test_dedupe_key_rejects_a_duplicate(db):
    add_business(db, name="Cafe Noir", key="provider|gplace123")
    with pytest.raises(psycopg.errors.UniqueViolation):
        # Same place, different spelling, arriving from a second overlapping search tile.
        add_business(db, name="Cafe-Noir", key="provider|gplace123")


def test_dedupe_key_allows_many_rows_without_one(db):
    # NULLs are distinct in Postgres, so a row inserted before its key is derived does not
    # collide with every other such row.
    add_business(db, name="One", key=None)
    add_business(db, name="Two", key=None)
    assert db.execute("SELECT count(*) FROM businesses").fetchone()[0] == 2


def test_audience_index_must_be_a_unit_interval(db):
    business_id = add_business(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        add_score(db, business_id, 1.5)


# --- lead_bands: cohort size decides the rule --------------------------------------------


def test_a_small_cohort_is_banded_on_absolute_thresholds(db):
    # Five businesses is not a distribution. Thresholds, not percentiles.
    expected = {}
    for index, (reviews, band) in enumerate(
        [(10, "small"), (99, "small"), (100, "medium"), (500, "medium"), (501, "large")]
    ):
        business_id = add_business(db, name=f"b{index}")
        add_score(db, business_id, 0.1 * index)
        add_reviews(db, business_id, reviews)
        expected[business_id] = band

    result = bands(db)
    assert len(result) == 5
    for business_id, band in expected.items():
        assert result[business_id] == (band, "absolute", 5)


def test_a_small_cohort_does_not_promote_its_biggest_member(db):
    # The failure the 30 threshold exists to prevent: NTILE(3) over three rows hands one of
    # them 'large' and the operator reads it as "one of the bigger salons around here".
    for index, reviews in enumerate([12, 30, 44]):
        business_id = add_business(db, name=f"tiny{index}")
        add_score(db, business_id, 0.2 * index)
        add_reviews(db, business_id, reviews)

    assert {row[0] for row in db.execute("SELECT audience_band FROM lead_bands")} == {"small"}


def test_a_full_cohort_is_banded_on_relative_tiles(db):
    for index in range(40):
        business_id = add_business(db, name=f"b{index}")
        add_score(db, business_id, index / 40)
        # Review counts that would put every one of them in the 'small' absolute bucket,
        # so a passing relative assertion cannot be the absolute rule in disguise.
        add_reviews(db, business_id, 10 + index)

    rows = db.execute(
        "SELECT audience_band, banding_method, cohort_size, audience_index"
        " FROM lead_bands ORDER BY audience_index"
    ).fetchall()

    assert len(rows) == 40
    assert {row[1] for row in rows} == {"relative"}
    assert {row[2] for row in rows} == {40}
    # NTILE(3) over 40 rows: 14/13/13, remainder to the first tile.
    tiles = ("small", "medium", "large")
    counts = {band: sum(1 for row in rows if row[0] == band) for band in tiles}
    assert counts == {"small": 14, "medium": 13, "large": 13}
    # Monotonic: the band never goes backwards as the index rises.
    order = {"small": 0, "medium": 1, "large": 2}
    ranks = [order[row[0]] for row in rows]
    assert ranks == sorted(ranks)


def test_the_threshold_sits_between_29_and_30(db):
    for index in range(29):
        business_id = add_business(db, name=f"b{index}")
        add_score(db, business_id, index / 29)
    assert {value[1] for value in bands(db).values()} == {"absolute"}

    business_id = add_business(db, name="b29")
    add_score(db, business_id, 1.0)
    assert {value[1] for value in bands(db).values()} == {"relative"}


def test_cohorts_are_independent(db):
    # A salon in Bangalore is not competing with a salon in Delhi, and a cafe next door is
    # not its cohort either.
    for index in range(30):
        add_score(db, add_business(db, name=f"blr{index}", niche="salon"), index / 30)
    for index in range(4):
        add_score(db, add_business(db, city="Delhi", name=f"del{index}", niche="salon"), 0.9)
    for index in range(4):
        add_score(db, add_business(db, name=f"cafe{index}", niche="cafe"), 0.9)

    rows = db.execute(
        "SELECT niche_id, city, banding_method, cohort_size FROM lead_bands"
    ).fetchall()
    methods = {(row[0], row[1]): (row[2], row[3]) for row in rows}
    assert methods == {
        ("salon", "Bangalore"): ("relative", 30),
        ("salon", "Delhi"): ("absolute", 4),
        ("cafe", "Bangalore"): ("absolute", 4),
    }


# --- lead_bands: missing evidence ---------------------------------------------------------


def test_a_business_with_no_audience_index_bands_unknown(db):
    scored = add_business(db, name="scored")
    add_score(db, scored, 0.5)
    add_reviews(db, scored, 42)
    unscored = add_business(db, name="index-less")
    add_score(db, unscored, None)

    result = bands(db)
    assert result[unscored][0] == "unknown"
    assert result[scored][0] == "small"


def test_a_business_with_no_score_row_at_all_still_appears(db):
    # The leads the pipeline failed on are exactly the ones worth looking at. Dropping them
    # from the sheet hides the failure.
    orphan = add_business(db, name="never scored")
    result = bands(db)
    assert set(result) == {orphan}
    assert result[orphan][0] == "unknown"


def test_unknown_rows_do_not_consume_tiles(db):
    # Left in the ranking, NULL indexes sort last, take the top tile, and push genuinely
    # large businesses down into 'medium' -- the bands would be counting missing data.
    for index in range(40):
        add_score(db, add_business(db, name=f"b{index}"), index / 40)
    for index in range(5):
        add_score(db, add_business(db, name=f"blank{index}"), None)

    rows = db.execute("SELECT audience_band, cohort_size FROM lead_bands").fetchall()
    counts = {band: sum(1 for row in rows if row[0] == band) for band, _ in rows}
    assert len(rows) == 45
    assert counts == {"small": 14, "medium": 13, "large": 13, "unknown": 5}
    assert {row[1] for row in rows} == {40}


def test_an_unparseable_review_count_reads_as_unknown_not_zero(db):
    business_id = add_business(db)
    add_score(db, business_id, 0.4)
    db.execute(
        "INSERT INTO enrichments (business_id, source, status, data)"
        " VALUES (%s, 'google_maps', 'ok', %s)",
        (business_id, Jsonb({"reviews": "unavailable"})),
    )
    assert bands(db)[business_id][0] == "unknown"


# --- lead_bands: score versioning ----------------------------------------------------------


def test_only_the_latest_score_row_is_used(db):
    business_id = add_business(db)
    add_reviews(db, business_id, 42)
    add_score(db, business_id, 0.10, at=NOW - timedelta(days=1), version="google-only")
    add_score(db, business_id, 0.90, at=NOW, version="instagram")

    row = db.execute(
        "SELECT audience_index, scorer_version, cohort_size FROM lead_bands"
    ).fetchall()
    assert len(row) == 1
    assert float(row[0][0]) == pytest.approx(0.90)
    assert row[0][1] == "instagram"
    # Two score rows for one business is still one business in the cohort.
    assert row[0][2] == 1


def test_a_superseded_score_cannot_change_the_band(db):
    # 39 ranked businesses plus one whose enrichment pass demoted it. If the stale row won,
    # this business would sit in the top tile on evidence it no longer has.
    for index in range(39):
        add_score(db, add_business(db, name=f"b{index}"), 0.5 + index / 100)
    demoted = add_business(db, name="demoted")
    add_score(db, demoted, 0.99, at=NOW - timedelta(days=2), version="google-only")
    add_score(db, demoted, 0.01, at=NOW, version="instagram")

    assert bands(db)[demoted][0] == "small"
