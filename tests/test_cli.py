"""The command line: arguments in, exit codes out, and never a surprise bill.

The CLI holds no behaviour of its own -- it parses, prints and exits, and everything else is
`lead_engine.api.deps.Engine`, which the HTTP routes call too. So the tests here are about
the three things that are genuinely this file's:

*Arguments become the same request the API validates.* `--niche` and `--area` repeat, an
alias resolves, and a bad `--limit` is refused with the code an HTTP caller would get. If
`build_spec` drifted from `parse_search_request`, the CLI would accept requests the API
rejects and the frontend would inherit a surface the terminal does not have.

*Nothing is spent without being announced.* `--dry-run` runs against a ledger that records
every write and a provider that raises on every call, and asserts both stayed silent. That
is the difference between a preview and an expensive mistake, out of fifty non-renewing
searches.

*Exit codes mean what a script assumes they mean.* 2 is "this cannot start", 1 is "nothing
found", 0 is "there is a sheet". A wrapper script that retried on 2 would burn the allowance
on a run that was never going to work.

The integration section drives the whole pipeline -- discovery, scoring, the workbook --
against a real Postgres and the recorded fixtures, which is exactly how the pipeline is
meant to be developed: for free. NO TEST HERE REACHES THE NETWORK: the provider is a
fixture or raises, and the resolver is the offline seed.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from lead_engine import cli
from lead_engine.api.deps import Engine
from lead_engine.config import Settings
from lead_engine.geo.resolver import SeedResolver
from lead_engine.providers.errors import ProviderError
from lead_engine.providers.fixtures import FixtureMapsProvider
from lead_engine.providers.searchapi import SearchApiClient

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

integration = pytest.mark.integration

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)

#: The committed corpus, five niches deep. `tests/test_fixtures.py` owns the claim that a
#: fixture run and a billed run parse identically; this file relies on it.
CORPUS = Path(__file__).parent / "fixtures" / "searchapi"

KEY = "sk-live-cli-4b7e2f9a1c3d5e60"


def make_settings(**overrides) -> Settings:
    base = dict(
        lead_engine_dsn=None,
        searchapi_key=None,
        lead_engine_search_budget=50,
        tinyfish_api_key=None,
        firecrawl_api_key=None,
        groq_api_key=None,
        gemini_api_key=None,
        gemini_model=None,
        nvidia_api_key=None,
    )
    base.update(overrides)
    # `_env_file=None`, because this repository has a real `.env` and the machine running
    # the suite may have real keys exported. A test whose result depends on that is noise.
    return Settings(_env_file=None, **base)


class RefusingProvider:
    """Raises rather than searching. A dry run that touches this fails loudly."""

    api_key = "configured-but-must-not-be-called"

    def search_places(self, *args, **kwargs):
        raise AssertionError("a dry run issued a billed search")


class RecordingBudget:
    """Answers reads, records writes. `spends` staying empty is the assertion."""

    def __init__(self, remaining: int = 50) -> None:
        self.remaining = remaining
        self.spends: list[tuple] = []

    def remaining_total(self, provider: str, *, key_fingerprint: str = "") -> int:
        return self.remaining

    def spend(self, provider: str, n: int = 1, **kwargs) -> int:
        self.spends.append((provider, n))
        return self.remaining

    def ensure_allowance(self, *args, **kwargs) -> dict[str, int]:
        return {"discover": self.remaining}


def args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


# --- arguments become a request -------------------------------------------------------


def test_repeated_niche_and_area_flags_build_one_spec():
    spec = cli.build_spec(
        args(
            "--city", "Bangalore",
            "--state", "Karnataka",
            "--country", "India",
            "--area", "Indiranagar",
            "--area", "Koramangala",
            "--niche", "salon",
            "--niche", "cafe",
            "--limit", "40",
        )
    )

    assert spec.scope.city == "Bangalore"
    assert spec.scope.state == "Karnataka"
    assert spec.scope.areas == ("Indiranagar", "Koramangala")
    assert spec.niche_ids == ("salon", "cafe")
    assert spec.limit == 40
    assert spec.use_ai is True


def test_niches_resolve_through_the_registry_exactly_as_the_api_does():
    """An alias is a niche, and the same niche twice is one niche.

    Two spellings of one area or one niche buy two identical searches at full price, so the
    collapsing happens in the shared parser rather than in whichever surface remembered.
    """
    spec = cli.build_spec(
        args(
            "--city", "Pune",
            "--niche", "hair salon",
            "--niche", "Salon",
            "--area", "Koregaon Park",
            "--area", "koregaon  park",
        )
    )

    assert spec.niche_ids == ("salon",)
    assert spec.scope.areas == ("Koregaon Park",)


def test_no_ai_pins_the_run_to_deterministic_copy():
    assert cli.build_spec(args("--city", "Pune", "--niche", "salon", "--no-ai")).use_ai is False


def test_a_city_alone_is_a_whole_metro_target():
    spec = cli.build_spec(args("--city", "Bangalore", "--niche", "salon"))

    assert spec.scope.areas == ()
    assert spec.scope.targets() == (("Bangalore", "city"),)


# --- exit codes -----------------------------------------------------------------------


def test_an_unsupported_niche_exits_2_and_names_the_supported_ones(capsys):
    code = cli.main(
        ["--city", "Bangalore", "--niche", "nail bar"], settings=make_settings(searchapi_key=KEY)
    )

    assert code == cli.EXIT_CANNOT_START
    stderr = capsys.readouterr().err
    assert "unsupported_niche" in stderr
    assert "salon" in stderr


def test_a_limit_outside_the_range_exits_2(capsys):
    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--limit", "0"],
        settings=make_settings(searchapi_key=KEY),
    )

    assert code == cli.EXIT_CANNOT_START
    assert "invalid_limit" in capsys.readouterr().err


def test_no_key_and_no_fixtures_exits_2_naming_both_ways_out(capsys):
    code = cli.main(["--city", "Bangalore", "--niche", "salon"], settings=make_settings())

    assert code == cli.EXIT_CANNOT_START
    stderr = capsys.readouterr().err
    assert "provider_not_configured" in stderr
    assert "SEARCHAPI_KEY" in stderr
    assert "--fixtures" in stderr


def test_a_real_run_without_a_database_exits_2(capsys):
    """A run records goals, cells, businesses and scores. There is nowhere to put them."""
    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--fixtures", str(CORPUS)],
        settings=make_settings(),
    )

    assert code == cli.EXIT_CANNOT_START
    assert "database_not_configured" in capsys.readouterr().err


def test_a_fixtures_directory_that_does_not_exist_exits_2(capsys):
    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--fixtures", "no/such/dir", "--dry-run"],
        settings=make_settings(),
    )

    assert code == cli.EXIT_CANNOT_START
    assert "not a directory" in capsys.readouterr().err


def test_an_unwritable_output_dir_exits_2_before_anything_is_billed(capsys, tmp_path):
    """The sheet is the product, and its directory is checked before the first search.

    Discovering the typo after the credits are gone leaves an operator who paid for a run
    and got no file -- recoverable from Postgres, but they paid for it.
    """
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("", encoding="utf-8")

    code = cli.main(
        [
            "--city", "Bangalore",
            "--niche", "salon",
            "--fixtures", str(CORPUS),
            "--output-dir", str(occupied),
        ],
        settings=make_settings(lead_engine_dsn="postgresql://localhost/x"),
    )

    assert code == cli.EXIT_CANNOT_START
    assert "--output-dir" in capsys.readouterr().err


def test_a_missing_city_is_argparses_own_usage_error():
    # argparse exits 2 on its own, which is the same code this CLI uses for "cannot start".
    with pytest.raises(SystemExit) as raised:
        cli.main(["--niche", "salon"])

    assert raised.value.code == cli.EXIT_CANNOT_START


# --- the dry run ----------------------------------------------------------------------


def dry_run_engine(budget: RecordingBudget, provider: RefusingProvider) -> Engine:
    return Engine(
        make_settings(searchapi_key=KEY),
        maps_provider=provider,
        resolver=SeedResolver(),
        budget=budget,
    )


def test_a_dry_run_prints_the_plan_and_spends_nothing(capsys):
    """The whole point of the flag: the operator sees the cost before agreeing to it."""
    budget, provider = RecordingBudget(remaining=37), RefusingProvider()

    code = cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--area", "Koramangala",
            "--niche", "salon",
            "--niche", "cafe",
            "--dry-run",
        ],
        settings=make_settings(searchapi_key=KEY),
        engine=dry_run_engine(budget, provider),
    )

    assert code == cli.EXIT_OK
    # Nothing was charged, and nothing was asked of the provider. Both, because either one
    # alone would pass against a run that spent without searching or searched without
    # spending.
    assert budget.spends == []

    out = capsys.readouterr().out
    assert "Indiranagar, Bangalore, Karnataka, India" in out
    assert "Koramangala, Bangalore, Karnataka, India" in out
    assert "12.978400" in out
    # 2 areas x 2 niches, one billed search apiece under the breadth-first default.
    assert "Planned searches: 4" in out
    assert "Search budget remaining: 37" in out
    assert "no credit was spent" in out


def test_a_dry_run_says_when_the_allowance_is_unknown(capsys):
    """None is not zero. A caller told "0 remaining" would abort a run that could proceed."""
    engine = Engine(
        make_settings(searchapi_key=KEY), maps_provider=RefusingProvider(), resolver=SeedResolver()
    )

    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--dry-run"],
        settings=make_settings(searchapi_key=KEY),
        engine=engine,
    )

    assert code == cli.EXIT_OK
    assert "Search budget remaining: unknown" in capsys.readouterr().out


def test_a_dry_run_against_fixtures_says_the_searches_are_free(capsys):
    engine = Engine(
        make_settings(),
        maps_provider=cli.fixture_provider(CORPUS),
        resolver=SeedResolver(),
        budget=RecordingBudget(),
    )

    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--fixtures", str(CORPUS), "--dry-run"],
        settings=make_settings(),
        engine=engine,
    )

    assert code == cli.EXIT_OK
    assert "served from fixtures, free" in capsys.readouterr().out


def test_an_unresolvable_area_exits_2_before_anything_is_built(capsys):
    """No fallback to the city centre. A sweep of ground nobody asked for looks like success."""
    budget = RecordingBudget()

    code = cli.main(
        ["--city", "Bangalore", "--area", "Nowhereville", "--niche", "salon"],
        settings=make_settings(searchapi_key=KEY),
        engine=dry_run_engine(budget, RefusingProvider()),
    )

    assert code == cli.EXIT_CANNOT_START
    assert "location_not_found" in capsys.readouterr().err
    assert budget.spends == []


def test_a_plan_that_costs_more_than_remains_is_refused_before_it_starts(capsys):
    """Refused whole, not started and stopped halfway.

    A pass that ran out mid-sweep would leave a sheet covering some of the areas asked for,
    with nothing in it saying which.
    """
    budget = RecordingBudget(remaining=1)

    code = cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--area", "Koramangala",
            "--niche", "salon",
            "--niche", "cafe",
        ],
        settings=make_settings(searchapi_key=KEY),
        engine=dry_run_engine(budget, RefusingProvider()),
    )

    assert code == cli.EXIT_CANNOT_START
    assert budget.spends == []
    assert "budget_exhausted" in capsys.readouterr().err


def test_a_provider_failure_mid_run_exits_1(capsys, tmp_path):
    """Not 2: this run could start, it just could not finish. A wrapper must not retry it
    the way it would retry a configuration mistake."""

    class BrokenEngine(Engine):
        def execute_run(self, *args, **kwargs):
            raise ProviderError("provider_rate_limited")

        def open_run(self, *args, **kwargs):
            return uuid.uuid4(), uuid.uuid4()

        def fail_run(self, run_id, reason):
            self.failed = (run_id, reason)

    engine = BrokenEngine(
        make_settings(searchapi_key=KEY),
        maps_provider=RefusingProvider(),
        resolver=SeedResolver(),
        budget=RecordingBudget(),
    )

    code = cli.main(
        ["--city", "Bangalore", "--niche", "salon", "--output-dir", str(tmp_path)],
        settings=make_settings(searchapi_key=KEY),
        engine=engine,
    )

    assert code == cli.EXIT_NO_RESULTS
    assert "provider_rate_limited" in capsys.readouterr().err
    # The run is closed rather than left `running` for a reaper to wonder about.
    assert engine.failed[1] == "provider_rate_limited"


# --- provider construction ------------------------------------------------------------


def test_fixtures_swaps_the_provider_for_recorded_responses():
    engine = cli.build_cli_engine(
        args("--city", "Bangalore", "--niche", "salon", "--fixtures", str(CORPUS)),
        make_settings(),
    )
    provider = engine.maps_provider()

    assert isinstance(provider, FixtureMapsProvider)
    assert provider.fixture_dir == CORPUS
    # Non-strict: a cursor that has walked past the last recorded page should stop, not
    # fail. The second fixture run of a goal does exactly that.
    assert provider.strict is False
    # A fixture provider is configured, so nothing downstream reports the system as unwired.
    assert engine.maps_configured() is True


def test_without_fixtures_the_billed_client_is_built():
    engine = cli.build_cli_engine(
        args("--city", "Bangalore", "--niche", "salon"), make_settings(searchapi_key=KEY)
    )
    provider = engine.maps_provider()

    assert isinstance(provider, SearchApiClient)
    assert provider.api_key == KEY
    # Metered by construction: a client built without a ledger is one that can spend fifty
    # credits in a loop. There is no database here, so there is no ledger and no way to
    # reach `search_places` -- which is what `configuration_problem` refuses first.
    assert provider.budget is None


# --- integration: the whole pipeline, for free ------------------------------------------


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

    pool = ConnectionPool(
        DSN,
        min_size=1,
        max_size=4,
        open=True,
        kwargs={"options": f"-c search_path={schema},public"},
    )
    try:
        yield pool
    finally:
        pool.close()
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def fixture_engine(pool) -> Engine:
    """The engine `--fixtures` builds, with the offline resolver pinned.

    `build_cli_engine` is unit-tested above for the provider swap; here the resolver is
    injected rather than chained so that a seed miss can never become a network request.
    """
    return Engine(
        make_settings(lead_engine_dsn=DSN, searchapi_key=KEY),
        pool=pool,
        maps_provider=cli.fixture_provider(CORPUS),
        resolver=SeedResolver(),
    )


@integration
@needs_postgres
def test_a_fixture_run_discovers_scores_and_writes_the_sheet(pool, tmp_path, capsys):
    code = cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--niche", "salon",
            "--fixtures", str(CORPUS),
            "--output-dir", str(tmp_path),
        ],
        settings=make_settings(lead_engine_dsn=DSN, searchapi_key=KEY),
        engine=fixture_engine(pool),
    )

    assert code == cli.EXIT_OK
    out = capsys.readouterr().out
    # The pre-flight block comes before anything billable, on every path.
    assert "Indiranagar, Bangalore, Karnataka, India" in out
    assert "Planned searches: 1" in out

    with pool.connection() as connection:
        businesses = connection.execute(
            "SELECT name, niche_id, city, search_area FROM businesses"
        ).fetchall()
        scores = connection.execute("SELECT count(*) FROM scores").fetchone()[0]
        run = connection.execute("SELECT status, stats FROM runs").fetchone()
        cells = connection.execute(
            "SELECT status, next_page, new_yield FROM search_cells"
        ).fetchall()

    assert businesses, "the fixture corpus qualified nothing at all"
    assert {row[1] for row in businesses} == {"salon"}
    assert {row[2] for row in businesses} == {"Bangalore"}
    assert {row[3] for row in businesses} == {"Indiranagar"}
    # Every discovered business is scored, so the sheet has something to rank by.
    assert scores == len(businesses)

    assert run[0] == "succeeded"
    assert run[1]["searches_spent"] == 1
    assert run[1]["new_businesses"] == len(businesses)
    # The cursor moved: tomorrow's run of the same sweep reads page 2 rather than re-buying
    # page 1 for a credit.
    assert len(cells) == 1
    assert cells[0][1] == 2

    sheets = list(tmp_path.glob("leads-*.xlsx"))
    assert len(sheets) == 1
    assert sheets[0].stat().st_size > 0
    assert str(sheets[0]) in out


@integration
@needs_postgres
def test_a_discovery_run_stamps_its_run_id_on_the_google_maps_enrichment_row(pool, tmp_path):
    """Found end-to-end: `--enrich` reported zero businesses enriched, zero scored, zero
    offers, on ten genuinely new leads from a real SearchAPI call. Root cause traced to
    `Engine.execute_run`: it built `DiscoveryRequest` without `run_id=run_id`, even though
    `run_id` is right there and passed to `self._score(...)` two lines later. Every
    `google_maps` enrichment row `DiscoveryService.record_evidence` writes therefore carried
    `run_id = NULL`, and `_enrichment_targets`'s `run_id`-keyed lookup --

        b.id IN (SELECT e.business_id FROM enrichments e WHERE e.run_id = %(run_id)s)

    -- could never match a single business through that path, no matter how many were
    found. `--enrich business_ids=[...]` explicitly still worked (the query's other OR
    branch), which is exactly why this stayed invisible: every existing test either supplied
    business_ids directly or used a fake `_enrichment_targets` that never touched this SQL.
    """
    code = cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--niche", "salon",
            "--fixtures", str(CORPUS),
            "--output-dir", str(tmp_path),
        ],
        settings=make_settings(lead_engine_dsn=DSN, searchapi_key=KEY),
        engine=fixture_engine(pool),
    )
    assert code == cli.EXIT_OK

    with pool.connection() as connection:
        run_id = connection.execute("SELECT id FROM runs").fetchone()[0]
        rows = connection.execute(
            "SELECT business_id, run_id FROM enrichments WHERE source = 'google_maps'"
        ).fetchall()

    assert rows, "discovery wrote no google_maps enrichment rows at all"
    for business_id, stamped_run_id in rows:
        assert stamped_run_id == run_id, f"business {business_id} was not stamped with its run"


@integration
@needs_postgres
def test_a_second_pass_over_the_same_ground_finds_nothing_and_exits_1(pool, tmp_path, capsys):
    """The boundary dedupe, from the operator's side.

    The second run returns the same twenty places and hands back none of them, because this
    system already holds them. That is exit 1 -- a completed run with nothing new -- and it
    must not read as a failure or as an empty neighbourhood.
    """
    argv = [
        "--city", "Bangalore",
        "--area", "Indiranagar",
        "--niche", "salon",
        "--fixtures", str(CORPUS),
        "--output-dir", str(tmp_path),
    ]
    settings = make_settings(lead_engine_dsn=DSN, searchapi_key=KEY)

    assert cli.main(argv, settings=settings, engine=fixture_engine(pool)) == cli.EXIT_OK
    capsys.readouterr()

    assert cli.main(argv, settings=settings, engine=fixture_engine(pool)) == cli.EXIT_NO_RESULTS

    out = capsys.readouterr().out
    assert "No new businesses" in out
    with pool.connection() as connection:
        runs = connection.execute("SELECT status FROM runs ORDER BY started_at").fetchall()
        already = connection.execute(
            "SELECT stats->>'new_businesses' FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()[0]

    assert [row[0] for row in runs] == ["succeeded", "empty"]
    assert already == "0"


@integration
@needs_postgres
def test_a_fixture_run_never_touches_the_ledger(pool, tmp_path):
    """Fixtures spend nothing, and the remaining-credit figure must not say otherwise.

    A fixture run that decremented the ledger would make the number the operator aborts on
    a lie, in the safe direction today and the expensive direction the moment they trust it.
    """
    engine = fixture_engine(pool)
    engine.ensure_budget()

    cli.main(
        [
            "--city", "Bangalore",
            "--area", "Indiranagar",
            "--niche", "salon",
            "--fixtures", str(CORPUS),
            "--output-dir", str(tmp_path),
        ],
        settings=make_settings(lead_engine_dsn=DSN, searchapi_key=KEY),
        engine=engine,
    )

    with pool.connection() as connection:
        used = connection.execute("SELECT sum(used) FROM search_budget").fetchone()[0]
    assert used == 0
    assert engine.remaining_budget() == 50


@integration
@needs_postgres
def test_main_seeds_the_ledger_for_a_key_it_has_never_seen():
    """A key used only through the CLI must be able to spend its first search.

    Found by actually running the real path against a real, never-before-used key: the API
    server seeds a fresh key's ledger row once at process startup (`api/app.py`'s startup
    hook), and `cli.main()` had no equivalent moment, so `SearchBudget.spend()` -- which
    refuses on principle rather than seeding itself -- raised `BudgetNotConfigured` on the
    very first real search a CLI-only key ever tried to make. `--dry-run` never caught it,
    because planning only READS the ledger and tolerates an absent row as "unknown".

    This test builds the engine the way an operator actually does -- through `cli.main`
    with no `engine=` override, so `build_cli_engine` runs and the fix's `ensure_budget()`
    call is the one under test, not bypassed the way `fixture_engine()` bypasses it above.
    `--dry-run` is deliberate too: it never touches discovery (so nothing here depends on
    what the geocoder or the fixture corpus decides is "new"), but `ensure_budget()` runs
    before `--dry-run`'s early return either way -- exactly the ordering the bug was in.

    Deliberately NOT using the `pool` fixture's isolated schema: `build_cli_engine` builds
    its own pool straight from `settings.dsn`, with no schema pinning, exactly as a real
    deployment's one-DSN-one-schema CLI invocation would -- so this connects to that same
    target directly, with its own unique, disposable key so it cannot collide with anything
    else and cleans its own row up afterward rather than leaving it in a shared schema.
    """
    key = f"sk-test-ledger-seed-{uuid.uuid4().hex}"
    settings = make_settings(lead_engine_dsn=DSN, searchapi_key=key)

    try:
        code = cli.main(
            ["--city", "Bangalore", "--area", "Indiranagar", "--niche", "salon", "--dry-run"],
            settings=settings,
        )

        assert code == cli.EXIT_OK
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        with psycopg.connect(DSN) as connection:
            row = connection.execute(
                "SELECT limit_total, used FROM search_budget "
                "WHERE provider = 'searchapi' AND purpose = 'discover' "
                "AND key_fingerprint = %s",
                (fingerprint,),
            ).fetchone()
        assert row is not None, "no ledger row was seeded for a key the CLI just used"
        limit_total, used = row
        assert limit_total == settings.search_budget
        assert used == 0  # a dry run spends nothing; only the ceiling should exist
    finally:
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(
                "DELETE FROM search_budget WHERE key_fingerprint = %s",
                (hashlib.sha256(key.encode()).hexdigest()[:12],),
            )
