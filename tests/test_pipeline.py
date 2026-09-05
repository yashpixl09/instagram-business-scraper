"""`Engine.execute_enrichment` -- wiring three already-tested subsystems into one pass.

Before this phase, `EnrichmentService`, `automations.firing_offers` and `LLMRouter` were
each fully built and fully tested in isolation, and none of them had a single caller in
`api/` or `cli.py`. This suite is not about re-proving any of those three -- their own test
files already do that -- it is about the ASSEMBLY: does enrichment's findings reach the
scorer in the shape it needs, does the scorer's output reach `firing_offers` in the
vocabulary it expects, does a firing offer actually produce an `automation_opportunities`
row and a pitch, and does every one of the pipeline's own stated rules survive the trip
through `Engine.execute_enrichment` rather than living only in the subsystems it calls.

THE RULES THIS FILE PROVES, NOT ASSUMES
----------------------------------------
*The verdict gate.* `automations.py`'s own docstring: the automation pitch is expensive to
generate and worthless before the operator has judged the lead worth pursuing. A business
with no verdict, or a `low`/`skip` verdict, must receive a `website_pitch` (the first offer
carries no such gate) and exactly zero `automation_opportunities` rows or automation pitches
-- however strong its evidence.

*Never fabricate.* `--no-ai` (`use_ai=False`) must never touch `LLMRouter` at all -- proved
by handing it a router that raises on `.generate()` -- and every pitch body must be exactly
the deterministic text `lead_engine.copy`/`automations.AutomationOffer.pitch_line` already
produced. With AI enabled, the router's OWN text is what gets stored, never re-derived.

*Idempotent re-detection.* Running the pass twice over evidence that has not changed must
not raise and must not grow a second `automation_opportunities` row for the same offer --
`Repository.insert_automation_opportunity`'s `ON CONFLICT (business_id, opportunity_id) DO
UPDATE` is the mechanism; this file proves the orchestration actually calls it that way.

*No new evidence, no crash.* A business enrichment could tell nothing new about still gets
a re-score and a website pitch from whatever is already on file.

*Honest signal translation.* `_translate_enrichment_signals` is deliberately a PARTIAL
translation of `automations.ENRICHMENT_SIGNALS` -- five of its eleven signals ("high review
count", "owner replies absent", "active social presence", "instagram-only catalogue", "dm
ordering") are not observable from anything `EnrichmentService` gathers today, and must
never be asserted. `SignalTranslationTests` pins that directly.

NO TEST HERE REACHES THE NETWORK, and the unit tests below reach no database either --
`Engine.repository` and `Engine._enrichment_targets` are both seams a subclass or a direct
attribute assignment can replace, exactly as `tests/test_cli.py`'s `BrokenEngine` replaces
`execute_run`. `ExecuteEnrichmentAgainstPostgresTests` at the bottom is the part that cannot
be faked -- real SQL, a real `uuid[]` parameter, real jsonb -- and is skipped without
`LEAD_ENGINE_TEST_DSN`, exactly as `tests/test_queue.py` is.
"""

from __future__ import annotations

import os
import unittest
import uuid
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from lead_engine.api.deps import (
    AUTOMATION_PITCH,
    DEFAULT_OUTREACH_CHANNEL,
    ENRICHED_SCORER_VERSION,
    WEBSITE_PITCH,
    Engine,
    _translate_enrichment_signals,
)
from lead_engine.api.schemas import ApiProblem
from lead_engine.automations import AUTOMATION_OFFERS
from lead_engine.config import Settings
from lead_engine.enrichment.ads import read_ad_library
from lead_engine.enrichment.service import AdFinding, BusinessEnrichment, HandleFinding
from lead_engine.enrichment.website import assess as website_assess
from lead_engine.enrichment.website import grade_site

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db.migrate import apply_migrations
    from lead_engine.db.repository import Repository
    from lead_engine.db.rows import AutomationOpportunityRow, OutreachRow
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)


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


# --- fakes ---------------------------------------------------------------------------------


class Result:
    """One search hit, in the shape `TinyFishClient.search` returns."""

    def __init__(self, url: str, title: str = "", snippet: str = "") -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


AD_LIBRARY_EMPTY = (
    "Meta Ad Library. No ads match your search criteria for this business, in this "
    "country, over the selected time range. Try changing or removing your filters."
)

#: Long enough to pass `ads.MIN_PAGE_CHARS` (120) and carrying a result count the regex
#: reads as "this advertiser is running ads".
AD_LIBRARY_FIVE_RESULTS = (
    "Meta Ad Library. Showing ads for this business across every active placement and "
    "region on the platform. ~5 results found for this advertiser over the selected time "
    "range and country."
)

#: Long enough to grade (`MIN_GRADEABLE_CHARS`), names a catalogue ("menu") and names
#: neither an enquiry/booking mechanism nor an ordering one -- so `grade_site` reports
#: `gaps=("enquiry", "ordering")` and nothing else.
WEAK_SITE_TEXT = (
    "Welcome to our shop. We offer a full range of services and a seasonal menu of "
    "products for every occasion. Follow us for the latest looks, offers and updates "
    "from our team throughout the year. Our stylists have decades of combined experience "
    "and pride themselves on quality, warmth and attention to every detail of a visit."
)


class FakeWeb:
    """A search-and-fetch provider over canned answers, keyed by business name.

    `search` matches on the quoted business name inside `website.search_query`'s own
    `'"{name}" {city}'` shape, so several businesses can share one instance without
    colliding -- exactly the shape `tests/test_contacts.py`'s `FakeWeb` uses.
    """

    def __init__(self, sites: dict[str, tuple[str, str]], ads_text: str = AD_LIBRARY_EMPTY) -> None:
        self.sites = dict(sites)
        self.ads_text = ads_text
        self.fetches: list[str] = []
        self.api_key = "fake"

    def search(self, query: str, limit: int = 10):
        for name, (url, _text) in self.sites.items():
            if f'"{name}"' in query:
                return [Result(url=url, title=name)]
        return []

    def fetch(self, url: str) -> str:
        self.fetches.append(url)
        for _name, (site_url, text) in self.sites.items():
            if url == site_url:
                return text
        if "ads/library" in url:
            return self.ads_text
        raise AssertionError(f"unexpected fetch: {url}")


class FakeRepository:
    """`Repository`'s writes this pass makes, in memory.

    `automation_opportunities` is a dict keyed by `(business_id, opportunity_id)`, which is
    exactly what `queries.UPSERT_AUTOMATION_OPPORTUNITY`'s `ON CONFLICT` target does in
    Postgres -- a second write with the same key replaces the row rather than adding one.
    `outreach` stays a plain list because the schema is append-only there by design.
    """

    def __init__(self) -> None:
        self._next_id = 0
        self.enrichments: list[SimpleNamespace] = []
        self.contacts: list[SimpleNamespace] = []
        self.scores: list[dict] = []
        self.automation_opportunities: dict[tuple, dict] = {}
        self.outreach: list[dict] = []

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def insert_enrichment(self, business_id, source, status, data, *, source_url=None, run_id=None):
        row = SimpleNamespace(id=self._id(), business_id=business_id, source=source, status=status)
        self.enrichments.append(row)
        return row

    def insert_contact(self, business_id, name, source, **kwargs):
        row = SimpleNamespace(id=self._id(), business_id=business_id, name=name, source=source, **kwargs)
        self.contacts.append(row)
        return row

    def insert_score(self, business_id, scorer_version, **kwargs):
        row = {"business_id": business_id, "scorer_version": scorer_version, **kwargs}
        self.scores.append(row)
        return row

    def insert_automation_opportunity(self, business_id, opportunity_id, *, confidence, trigger_signals, evidence):
        key = (business_id, opportunity_id)
        self.automation_opportunities[key] = {
            "business_id": business_id,
            "opportunity_id": opportunity_id,
            "confidence": confidence,
            "trigger_signals": list(trigger_signals),
            "evidence": evidence,
        }
        return self.automation_opportunities[key]

    def insert_outreach(self, business_id, kind, channel, body, *, evidence=None):
        row = {
            "business_id": business_id,
            "kind": kind,
            "channel": channel,
            "body": body,
            "evidence": evidence or {},
        }
        self.outreach.append(row)
        return row


class FakeLLMRouter:
    """Records every call and echoes the fallback back, prefixed -- proof the AI path's
    OWN text is what gets stored, and proof of exactly what it was grounded in."""

    def __init__(self, prefix: str = "AI: ") -> None:
        self.calls: list[SimpleNamespace] = []
        self.prefix = prefix

    def generate(self, prompt: str, fallback: str, **kwargs):
        self.calls.append(SimpleNamespace(prompt=prompt, fallback=fallback))
        return SimpleNamespace(text=f"{self.prefix}{fallback}")


class RaisingRouter:
    """A router that fails the test if `--no-ai`'s promise is broken."""

    def generate(self, *args, **kwargs):
        raise AssertionError("LLMRouter.generate must never be called when use_ai=False")


class TargetsEngine(Engine):
    """An `Engine` whose `_enrichment_targets` is a fixture, not a query -- the same seam
    `tests/test_cli.py`'s `BrokenEngine` uses on `execute_run`, applied here so the whole
    orchestration can be exercised with no pool and no database at all."""

    def __init__(self, *args, targets, **kwargs):
        super().__init__(*args, **kwargs)
        self._fake_targets = list(targets)

    def _enrichment_targets(self, *, run_id, business_ids):
        return list(self._fake_targets)


def target_row(**overrides) -> dict:
    base = dict(
        id=uuid.uuid4(),
        place_id="place-1",
        name="Sunrise Salon",
        niche_id="salon",
        country="IN",
        state=None,
        city="Pune",
        address="MG Road",
        lat=18.5,
        lng=73.8,
        phone="9123456789",
        email=None,
        website=None,
        instagram_handle=None,
        facebook_url=None,
        my_verdict=None,
        google_evidence={},
    )
    base.update(overrides)
    return base


def build_engine(targets, **engine_kwargs) -> TargetsEngine:
    engine_kwargs.setdefault("enrichment_provider", FakeWeb({}))
    engine = TargetsEngine(make_settings(), targets=targets, **engine_kwargs)
    engine.repository = FakeRepository()
    return engine


# --- the verdict gate ------------------------------------------------------------------


class VerdictGateTests(unittest.TestCase):
    def _run(self, my_verdict):
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict=my_verdict)
        engine = build_engine([row], enrichment_provider=provider)

        report = engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        return engine, report

    def test_a_high_verdict_business_gets_an_automation_opportunity_and_pitch(self):
        engine, report = self._run("high")
        repo = engine.repository
        self.assertEqual(report.offers_detected, 1)
        [(_business_id, opportunity_id)] = repo.automation_opportunities.keys()
        self.assertEqual(opportunity_id, "appointment_booking")
        kinds = [row["kind"] for row in repo.outreach]
        self.assertIn(AUTOMATION_PITCH, kinds)
        self.assertIn(WEBSITE_PITCH, kinds)

    def test_a_medium_verdict_business_also_qualifies(self):
        _engine, report = self._run("medium")
        self.assertEqual(report.offers_detected, 1)

    def test_a_low_verdict_business_gets_no_automation_opportunity(self):
        engine, report = self._run("low")
        repo = engine.repository
        self.assertEqual(report.offers_detected, 0)
        self.assertEqual(repo.automation_opportunities, {})
        kinds = [row["kind"] for row in repo.outreach]
        self.assertEqual(kinds, [WEBSITE_PITCH])

    def test_a_business_with_no_verdict_at_all_gets_no_automation_opportunity(self):
        engine, report = self._run(None)
        repo = engine.repository
        self.assertEqual(report.offers_detected, 0)
        self.assertEqual(repo.automation_opportunities, {})

    def test_a_skip_verdict_business_gets_no_automation_opportunity(self):
        engine, report = self._run("skip")
        repo = engine.repository
        self.assertEqual(report.offers_detected, 0)

    def test_every_enriched_business_gets_a_website_pitch_regardless_of_verdict(self):
        for verdict in (None, "low", "skip", "medium", "high"):
            with self.subTest(verdict=verdict):
                engine, report = self._run(verdict)
                self.assertEqual(report.scored, 1)
                kinds = [row["kind"] for row in engine.repository.outreach]
                self.assertIn(WEBSITE_PITCH, kinds)


# --- never fabricate: the --no-ai / no-provider path ------------------------------------


class DeterministicFallbackTests(unittest.TestCase):
    def _engine(self, verdict="high"):
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict=verdict)
        return build_engine([row], enrichment_provider=provider), row

    def test_use_ai_false_never_touches_the_router(self):
        engine, row = self._engine()
        engine._llm_router = RaisingRouter()  # would raise on any .generate() call
        report = engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        self.assertEqual(report.scored, 1)  # completed without the router ever firing

    def test_the_automation_pitch_is_exactly_the_offers_own_pitch_line_without_ai(self):
        engine, row = self._engine()
        engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        automation_rows = [r for r in engine.repository.outreach if r["kind"] == AUTOMATION_PITCH]
        self.assertEqual(len(automation_rows), 1)
        self.assertEqual(
            automation_rows[0]["body"], AUTOMATION_OFFERS["appointment_booking"].pitch_line
        )

    def test_the_website_pitch_is_the_deterministic_template_without_ai(self):
        engine, row = self._engine()
        engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        website_rows = [r for r in engine.repository.outreach if r["kind"] == WEBSITE_PITCH]
        self.assertEqual(len(website_rows), 1)
        # `copy.build_fallback_outreach`'s own template: "Hi {name},\n\n..."
        self.assertTrue(website_rows[0]["body"].startswith("Hi Sunrise Salon,"))

    def test_no_configured_llm_provider_also_falls_back_without_raising(self):
        # `use_ai=True` but zero providers configured: `LLMRouter.from_settings` still
        # returns a working router (the chain's own guarantee), never raising here either.
        engine, row = self._engine()
        report = engine.execute_enrichment(business_ids=[row["id"]], use_ai=True)
        self.assertEqual(report.scored, 1)
        automation_rows = [r for r in engine.repository.outreach if r["kind"] == AUTOMATION_PITCH]
        self.assertEqual(
            automation_rows[0]["body"], AUTOMATION_OFFERS["appointment_booking"].pitch_line
        )


class AiExpansionTests(unittest.TestCase):
    def test_use_ai_true_stores_the_routers_own_text_grounded_in_the_fallback(self):
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict="high")
        router = FakeLLMRouter()
        engine = build_engine([row], enrichment_provider=provider, llm_router=router)

        engine.execute_enrichment(business_ids=[row["id"]], use_ai=True)

        website_rows = [r for r in engine.repository.outreach if r["kind"] == WEBSITE_PITCH]
        automation_rows = [r for r in engine.repository.outreach if r["kind"] == AUTOMATION_PITCH]
        self.assertTrue(website_rows[0]["body"].startswith("AI: Hi Sunrise Salon,"))
        self.assertEqual(
            automation_rows[0]["body"],
            f"AI: {AUTOMATION_OFFERS['appointment_booking'].pitch_line}",
        )
        # Two prompts: one per pitch. Every one is grounded in the business's own name.
        self.assertEqual(len(router.calls), 2)
        for call in router.calls:
            self.assertIn("Sunrise Salon", call.prompt)

    def test_the_automation_prompt_repeats_the_never_fabricate_rule(self):
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict="high")
        router = FakeLLMRouter()
        engine = build_engine([row], enrichment_provider=provider, llm_router=router)

        engine.execute_enrichment(business_ids=[row["id"]], use_ai=True)

        automation_prompt = next(
            call.prompt for call in router.calls if "Automation offered" in call.prompt
        )
        self.assertIn("Do not invent", automation_prompt)


# --- no new evidence, no crash -----------------------------------------------------------


class NoNewEvidenceTests(unittest.TestCase):
    def test_a_business_with_no_website_no_handle_and_no_ads_still_gets_scored_and_pitched(self):
        # An empty `FakeWeb`: the search returns nothing for anyone, and the ad library
        # answers "no ads" for whatever URL it is asked about.
        provider = FakeWeb({})
        row = target_row(niche_id="cafe", my_verdict="high")
        engine = build_engine([row], enrichment_provider=provider)

        report = engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)

        self.assertEqual(report.enriched, 1)
        self.assertEqual(report.scored, 1)
        self.assertEqual(report.offers_detected, 0)  # no evidence -> no offer, per the spec
        kinds = [r["kind"] for r in engine.repository.outreach]
        self.assertEqual(kinds, [WEBSITE_PITCH])
        self.assertEqual(engine.repository.scores[0]["scorer_version"], ENRICHED_SCORER_VERSION)

    def test_google_reviews_and_rating_survive_a_rescore(self):
        # `businesses`/`scores` carry no raw reviews column; the only place a rescore can
        # recover them is the `google_maps` enrichment row `_enrichment_targets` reads back.
        provider = FakeWeb({})
        row = target_row(google_evidence={"reviews": 340, "rating": 4.6})
        engine = build_engine([row], enrichment_provider=provider)

        engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)

        self.assertIsNotNone(engine.repository.scores[0]["audience_index"])


class EmptyTargetsTests(unittest.TestCase):
    def test_no_run_id_and_no_business_ids_is_refused(self):
        # `TargetsEngine._enrichment_targets` is a fixture (see its own docstring), so this
        # exercises the real guard method the plain `Engine` uses -- the one thing a
        # fixture-backed subclass would otherwise never let this suite reach.
        engine = Engine(make_settings())
        with self.assertRaises(ApiProblem) as ctx:
            engine._enrichment_targets(run_id=None, business_ids=None)
        self.assertEqual(ctx.exception.code, "invalid_request")

    def test_an_empty_target_list_reports_zero_without_touching_anything(self):
        engine = build_engine([])
        report = engine.execute_enrichment(business_ids=[uuid.uuid4()], use_ai=False)
        self.assertEqual(report.enriched, 0)
        self.assertEqual(report.scored, 0)
        self.assertEqual(report.offers_detected, 0)
        self.assertEqual(report.outreach_written, 0)
        self.assertEqual(engine.repository.outreach, [])


# --- idempotent re-detection -------------------------------------------------------------


class IdempotentReDetectionTests(unittest.TestCase):
    def test_running_enrichment_twice_on_unchanged_evidence_does_not_duplicate_the_offer(self):
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict="high")
        engine = build_engine([row], enrichment_provider=provider)

        first = engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        second = engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)

        self.assertEqual(first.offers_detected, 1)
        self.assertEqual(second.offers_detected, 1)
        # One row, not two: `FakeRepository.insert_automation_opportunity` mirrors the real
        # `ON CONFLICT (business_id, opportunity_id) DO UPDATE`.
        self.assertEqual(len(engine.repository.automation_opportunities), 1)

    def test_outreach_stays_append_only_across_re_detection(self):
        # A deliberate asymmetry, stated in `queries.INSERT_OUTREACH`'s own comment: unlike
        # the opportunity row, a redrafted pitch is a NEW row, never an overwrite of text
        # that may already have gone out.
        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        provider = FakeWeb({"Sunrise Salon": site})
        row = target_row(my_verdict="high")
        engine = build_engine([row], enrichment_provider=provider)

        engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)
        engine.execute_enrichment(business_ids=[row["id"]], use_ai=False)

        self.assertEqual(len(engine.repository.outreach), 4)  # 2 passes x (website + automation)


# --- honest signal translation ------------------------------------------------------------


class SignalTranslationTests(unittest.TestCase):
    """`_translate_enrichment_signals` in isolation -- no Engine, no provider, no LLM."""

    UNOBSERVABLE = (
        "high review count",
        "owner replies absent",
        "active social presence",
        "instagram-only catalogue",
        "dm ordering",
    )

    def test_never_asserts_any_of_the_five_unobservable_signals(self):
        results = [Result(url="https://example.com", title="Example")]
        finding = website_assess(results, name="Example")
        grade = grade_site(WEAK_SITE_TEXT * 3)
        website = replace(finding, grade=grade)
        ads = read_ad_library(AD_LIBRARY_FIVE_RESULTS, url="https://x")
        enrichment = BusinessEnrichment(
            business_id=uuid.uuid4(),
            website=website,
            social=HandleFinding(status="no_data"),
            ads=ads,
        )
        signals, _evidence = _translate_enrichment_signals(
            enrichment, {"tinyfish_web": 1, "ad_library": 2}
        )
        for signal in self.UNOBSERVABLE:
            with self.subTest(signal=signal):
                self.assertNotIn(signal, signals)

    def test_runs_meta_ads_traces_to_the_ad_library_row(self):
        ads = read_ad_library(AD_LIBRARY_FIVE_RESULTS, url="https://x")
        enrichment = BusinessEnrichment(business_id=uuid.uuid4(), website=None, social=None, ads=ads)
        signals, evidence = _translate_enrichment_signals(enrichment, {"ad_library": 42})
        self.assertIn("runs meta ads", signals)
        self.assertEqual(evidence["runs meta ads"], [42])

    def test_no_ads_finding_and_no_row_id_asserts_nothing(self):
        enrichment = BusinessEnrichment(business_id=uuid.uuid4(), website=None, social=None, ads=None)
        signals, evidence = _translate_enrichment_signals(enrichment, {})
        self.assertEqual(signals, set())
        self.assertEqual(evidence, {})

    def test_an_enquiry_gap_asserts_the_three_correlated_intake_signals_together(self):
        results = [Result(url="https://example.com", title="Example")]
        finding = website_assess(results, name="Example")
        grade = grade_site(WEAK_SITE_TEXT * 3)
        assert grade is not None and "enquiry" in grade.gaps and "ordering" in grade.gaps
        website = replace(finding, grade=grade)
        enrichment = BusinessEnrichment(business_id=uuid.uuid4(), website=website, social=None, ads=None)
        signals, evidence = _translate_enrichment_signals(enrichment, {"tinyfish_web": 7})
        self.assertEqual(
            {"no booking link", "no enquiry form", "manual enquiry flow", "no online ordering"},
            signals,
        )
        for signal in ("no booking link", "no enquiry form", "manual enquiry flow", "no online ordering"):
            self.assertEqual(evidence[signal], [7])


# --- against real Postgres -----------------------------------------------------------------


@needs_postgres
class ExecuteEnrichmentAgainstPostgresTests(unittest.TestCase):
    """The part fakes cannot prove: real SQL, a real `uuid[]` parameter, real jsonb, and a
    real `ON CONFLICT` doing the idempotent-refresh work `IdempotentReDetectionTests` above
    only simulates."""

    def setUp(self):
        self.schema = "test_" + uuid.uuid4().hex
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.addCleanup(self._drop_schema)

        with psycopg.connect(DSN) as connection:
            connection.execute(f"SET search_path = {self.schema}, public")
            apply_migrations(connection)
            connection.commit()

        self.pool = ConnectionPool(
            DSN,
            min_size=1,
            max_size=4,
            open=True,
            kwargs={"options": f"-c search_path={self.schema},public"},
        )
        self.addCleanup(self.pool.close)
        self.repo = Repository(self.pool)

    def _drop_schema(self):
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')

    def sql(self, statement, params=None):
        with self.pool.connection() as connection:
            return connection.execute(statement, params).fetchall()

    def exec_sql(self, statement, params=None) -> None:
        """Run a statement with no result set to fetch -- an INSERT with no RETURNING."""
        with self.pool.connection() as connection:
            connection.execute(statement, params)

    def set_verdict(self, business_id, my_verdict) -> None:
        self.exec_sql(
            "INSERT INTO verdicts (business_id, my_verdict) VALUES (%s, %s)",
            (business_id, my_verdict),
        )

    def a_business(self, **overrides):
        base = dict(name="Sunrise Salon", niche_id="salon", city="Pune")
        base.update(overrides)
        return self.repo.upsert_business(**base)

    def test_row_dataclasses_match_their_tables_exactly(self):
        for table, row_class in (
            ("automation_opportunities", AutomationOpportunityRow),
            ("outreach", OutreachRow),
        ):
            with self.subTest(table=table):
                columns = {
                    row[0]
                    for row in self.sql(
                        "SELECT column_name FROM information_schema.columns"
                        " WHERE table_schema = current_schema() AND table_name = %s",
                        (table,),
                    )
                }
                declared = {field.name for field in fields(row_class)}
                self.assertEqual(declared, columns)

    def test_re_detecting_the_same_offer_refreshes_rather_than_duplicates(self):
        business = self.a_business()
        first = self.repo.insert_automation_opportunity(
            business.id,
            "appointment_booking",
            confidence=1.0,
            trigger_signals=["no booking link"],
            evidence={"enrichment_ids": [1]},
        )
        second = self.repo.insert_automation_opportunity(
            business.id,
            "appointment_booking",
            confidence=1.0,
            trigger_signals=["no booking link"],
            evidence={"enrichment_ids": [1, 2]},
        )
        rows = self.sql(
            "SELECT id, evidence FROM automation_opportunities WHERE business_id = %s",
            (business.id,),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.evidence, {"enrichment_ids": [1, 2]})

    def test_outreach_is_append_only(self):
        business = self.a_business()
        self.repo.insert_outreach(business.id, "website_pitch", "email", "first draft")
        self.repo.insert_outreach(business.id, "website_pitch", "email", "second draft")
        rows = self.sql("SELECT body FROM outreach WHERE business_id = %s", (business.id,))
        self.assertEqual(len(rows), 2)

    def test_execute_enrichment_end_to_end_against_a_real_database(self):
        engine = Engine(make_settings(), pool=self.pool)
        run_row = self.repo.create_run(
            self.repo.create_goal("pune salons", {"city": "Pune", "niche_ids": ["salon"]}).id
        )
        business = self.a_business()
        self.repo.insert_enrichment(
            business.id, "google_maps", "ok", {"reviews": 200, "rating": 4.5}, run_id=run_row.id
        )
        self.set_verdict(business.id, "high")

        site = ("https://sunrisesalon.example", WEAK_SITE_TEXT)
        engine._enrichment_provider = FakeWeb({"Sunrise Salon": site})

        report = engine.execute_enrichment(business_ids=[business.id], use_ai=False)

        self.assertEqual(report.scored, 1)
        self.assertEqual(report.offers_detected, 1)
        scores = self.sql(
            "SELECT scorer_version, audience_index FROM scores WHERE business_id = %s",
            (business.id,),
        )
        self.assertEqual(scores[-1][0], ENRICHED_SCORER_VERSION)
        self.assertIsNotNone(scores[-1][1])
        offers = self.sql(
            "SELECT opportunity_id FROM automation_opportunities WHERE business_id = %s",
            (business.id,),
        )
        self.assertEqual([row[0] for row in offers], ["appointment_booking"])
        outreach = self.sql(
            "SELECT kind FROM outreach WHERE business_id = %s ORDER BY kind", (business.id,)
        )
        self.assertEqual(sorted(row[0] for row in outreach), [AUTOMATION_PITCH, WEBSITE_PITCH])


if __name__ == "__main__":
    unittest.main()
