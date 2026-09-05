"""The `--enrich` flag: the CLI's only new surface for this phase.

`lead_engine.cli` holds no behaviour of its own -- everything here is really testing that
the flag is wired to `Engine.execute_enrichment` with the right arguments, at the right
moment, and not otherwise. So this mirrors `tests/test_cli.py`'s own convention exactly:
subclass `Engine`, replace `execute_run` (and `open_run`) with a canned result, and assert
on what the CLI did around it -- never on discovery or enrichment themselves, which have
their own suites.

NO TEST HERE REACHES THE NETWORK OR A DATABASE. `execute_run` and `execute_enrichment` are
both replaced outright.
"""

from __future__ import annotations

import uuid

from lead_engine import cli
from lead_engine.api.deps import Engine
from lead_engine.config import Settings
from lead_engine.geo.resolver import SeedResolver

KEY = "sk-live-cli-enrich-4b7e2f9a1c3d5e60"


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
    return Settings(_env_file=None, **base)


class RefusingProvider:
    """A maps provider that fails the test if a search is actually issued."""

    api_key = "configured-but-must-not-be-called"

    def search_places(self, *args, **kwargs):
        raise AssertionError("no real search should run in this suite")


class _Business:
    def __init__(self, business_id):
        self.business_id = business_id


class _Outcome:
    """Just enough of `DiscoveryOutcome`'s shape for `cli.print_report` to render."""

    def __init__(self, businesses):
        self.businesses = businesses
        self.searches_spent = 0
        self.cells_searched = 0
        self.cells_planned = 0
        self.stopped = "complete"
        self.niches = ()
        self.starved_niches = ()


class _FakeRunReport:
    """Just enough of `RunReport`'s shape for `print_report`/`--enrich` to read."""

    def __init__(self, run_id, business_ids):
        self.run_id = run_id
        self.goal_id = uuid.uuid4()
        self.outcome = _Outcome([_Business(bid) for bid in business_ids])
        self.scored = len(business_ids)
        self.export_path = None

    @property
    def found(self):
        return len(self.outcome.businesses)


class _FakeEnrichmentReport:
    def __init__(self):
        self.enriched = 1
        self.scored = 1
        self.offers_detected = 1
        self.outreach_written = 2
        self.usage = {"primary_calls": 3, "fallback_calls": 1, "fallback_unavailable": 0}


class EnrichingEngine(Engine):
    """`execute_run` and `open_run` return canned results; `execute_enrichment` records
    every call it receives instead of doing anything -- the seam `tests/test_cli.py`'s
    `BrokenEngine` already established for `execute_run` alone."""

    def __init__(self, *args, business_ids=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._business_ids = list(business_ids)
        self.enrichment_calls: list[dict] = []

    def open_run(self, *args, **kwargs):
        return uuid.uuid4(), uuid.uuid4()

    def execute_run(self, *args, **kwargs):
        self.last_run_id = kwargs["run_id"]
        return _FakeRunReport(kwargs["run_id"], self._business_ids)

    def execute_enrichment(self, **kwargs):
        self.enrichment_calls.append(kwargs)
        return _FakeEnrichmentReport()


def engine_with(business_ids) -> EnrichingEngine:
    return EnrichingEngine(
        make_settings(searchapi_key=KEY),
        maps_provider=RefusingProvider(),
        resolver=SeedResolver(),
        business_ids=business_ids,
    )


def run_cli(args, engine, **extra):
    return cli.main(
        ["--city", "Bangalore", "--niche", "salon", *args],
        settings=make_settings(searchapi_key=KEY),
        engine=engine,
        **extra,
    )


def test_enrich_flag_calls_execute_enrichment_with_the_runs_own_ids(capsys, tmp_path):
    business_ids = [uuid.uuid4(), uuid.uuid4()]
    engine = engine_with(business_ids)

    code = run_cli(["--enrich", "--output-dir", str(tmp_path)], engine)

    assert code == cli.EXIT_OK
    assert len(engine.enrichment_calls) == 1
    call = engine.enrichment_calls[0]
    assert call["business_ids"] == business_ids
    assert call["run_id"] == engine.last_run_id
    assert call["use_ai"] is True
    out = capsys.readouterr().out
    assert "Enrichment: 2 business(es) queued" in out
    assert "Enrichment pass:" in out
    assert "automation offers   : 1" in out
    assert "outreach drafted    : 2" in out


def test_without_the_flag_execute_enrichment_is_never_called(capsys, tmp_path):
    engine = engine_with([uuid.uuid4()])

    code = run_cli(["--output-dir", str(tmp_path)], engine)

    assert code == cli.EXIT_OK
    assert engine.enrichment_calls == []
    assert "Enrichment" not in capsys.readouterr().out


def test_no_ai_is_threaded_through_to_execute_enrichment(capsys, tmp_path):
    engine = engine_with([uuid.uuid4()])

    run_cli(["--enrich", "--no-ai", "--output-dir", str(tmp_path)], engine)

    assert len(engine.enrichment_calls) == 1
    assert engine.enrichment_calls[0]["use_ai"] is False


def test_a_run_that_found_nothing_never_reaches_the_enrich_step(capsys, tmp_path):
    engine = engine_with([])  # no businesses found

    code = run_cli(["--enrich", "--output-dir", str(tmp_path)], engine)

    assert code == cli.EXIT_NO_RESULTS
    assert engine.enrichment_calls == []


def test_enrich_plan_is_printed_before_the_enrichment_report(capsys, tmp_path):
    engine = engine_with([uuid.uuid4()])

    run_cli(["--enrich", "--output-dir", str(tmp_path)], engine)

    out = capsys.readouterr().out
    plan_at = out.index("Enrichment: 1 business(es) queued")
    report_at = out.index("Enrichment pass:")
    assert plan_at < report_at
