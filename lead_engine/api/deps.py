"""The composition root, and the service layer both surfaces call.

Two jobs live here, and they are one file because they are one decision: *what this
process is wired to*, and *what it can therefore do*.

    Engine          every operation the system offers, as plain methods over injected
                    collaborators. `routes.py` adapts HTTP to these; `cli.py` calls the
                    same ones with the same arguments.
    dependencies    how a request gets hold of the Engine, and how a process builds one.

WHY THE SERVICE LAYER IS NOT IN THE ROUTES
------------------------------------------
A web frontend arrives later and must not need a capability the API lacks. The only way to
guarantee that is to make the CLI a client of the same methods rather than a second
implementation of them -- so `cli.py` contains argument parsing, printing, and exit codes,
and not one line of what a search *is*. Anything the operator can do from a terminal is
therefore reachable from HTTP by writing a route, never by writing logic.

The one asymmetry is stated rather than hidden: `execute_run` performs a discovery pass
inline, and no route calls it, because an HTTP request that spends a non-renewing credit is
one a browser refresh can spend twice. `POST /api/search` enqueues; a worker (or the CLI)
performs. The capability is here, in the service layer, ready for the worker that will call
it.

WHY THE POOL IS BUILT HERE AND INJECTED
---------------------------------------
`Repository(pool)` takes a pool the caller built -- that is the seam the tests use to point
every statement at a throwaway schema. Several collaborators (`SearchBudget`,
`SearchCellStore`, `NicheStatusStore`, `RepositoryBusinesses`) want a *connection factory*
rather than a repository, and the pool is the only object that is both. So this module owns
the pool, hands it to `Repository`, and hands `pool.connection` to everything else.

SQL THAT IS NOT IN `db/queries.py`
----------------------------------
`lead_engine.db.queries` holds every statement the repository runs, and the repository has
no reads for runs, leads or verdicts -- those are the API's own queries and they live in
this module as named constants, never assembled at a call site. `lead_bands` (0009) is the
driving relation for leads for the same reason `excel.py` uses it: a band is computed per
read against its cohort, and a second implementation here would drift from the sheet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import Request
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..automations import AutomationOffer, firing_offers
from ..config import SEARCHAPI, Settings
from ..copy import build_fallback_outreach, build_prompt
from ..db.repository import Repository
from ..discovery.cells import BREADTH_FIRST, CellPolicy, SearchCellStore
from ..discovery.service import (
    DiscoveryOutcome,
    DiscoveryRequest,
    DiscoveryService,
    LocationResolver,
    MapsProvider,
    RepositoryBusinesses,
)
from ..discovery.status import NicheStatusStore
from ..enrichment import ads as ads_module
from ..enrichment import website as website_module
from ..enrichment.cache import InMemoryCache, PostgresCache
from ..enrichment.service import (
    SOURCE_ADS,
    SOURCE_FIRECRAWL_WEB,
    SOURCE_TINYFISH_WEB,
    BusinessEnrichment,
    EnrichmentRequest,
    EnrichmentService,
    EvidenceStore,
    WebProvider,
)
from ..export import excel
from ..geo.resolver import (
    CachingResolver,
    ChainResolver,
    NominatimResolver,
    SeedResolver,
    resolve_scope,
)
from ..geo.scope import GeoScope, ResolvedLocation, normalize
from ..llm.cache import ResponseCache
from ..llm.router import DatabaseCircuitStore, LLMRouter
from ..models import Evidence, Lead
from ..niches import NICHE_PROFILES, niche_payload
from ..providers.budget import BudgetNotConfigured, SearchBudget, fingerprint
from ..providers.fixtures import FixtureMapsProvider
from ..providers.searchapi import SearchApiClient
from ..providers.tinyfish import TinyFishClient
from ..scoring import audience_index, score_lead
from .schemas import (
    DEFAULT_VERIFICATION,
    ApiProblem,
    BudgetOut,
    ConfigStatusOut,
    LeadListOut,
    LeadOut,
    NicheListOut,
    NicheOut,
    ResolvedLocationOut,
    RunListOut,
    RunOut,
    SearchAcceptedOut,
    SearchPlanOut,
    SearchSpec,
    VerdictUpdate,
)

logger = logging.getLogger("lead_engine.api")

#: The queue's name for a discovery unit of work. One task is one (area x niche).
DISCOVER = "discover"

#: How many runs a listing returns unless asked otherwise, and the ceiling on that ask.
DEFAULT_RUN_LIMIT = 50
MAX_RUN_LIMIT = 200

#: The same, for leads. The sheet is the place to read forty thousand rows; this is an API.
DEFAULT_LEAD_LIMIT = 500
MAX_LEAD_LIMIT = 2000

#: What a freshly accepted search is called before a worker touches it.
QUEUED = "queued"


def utc_now() -> datetime:
    return datetime.now(UTC)


# --- SQL ------------------------------------------------------------------------------

_RUN_COLUMNS = "id, goal_id, status, trigger, stats, started_at, finished_at"

# `started_at` is nullable, so NULLS LAST keeps a run that never started off the top of the
# list. `id DESC` is the tiebreak: two runs created in one transaction share `now()`.
SELECT_RUNS = f"""
SELECT {_RUN_COLUMNS}
  FROM runs
 ORDER BY started_at DESC NULLS LAST, id DESC
 LIMIT %(limit)s
"""

SELECT_RUN = f"SELECT {_RUN_COLUMNS} FROM runs WHERE id = %(id)s"

SELECT_LATEST_RUN = "SELECT id FROM runs ORDER BY started_at DESC NULLS LAST, id DESC LIMIT 1"

# The goal a run belongs to, which is where its scope is written down. `businesses` carries
# no run id (see `Engine.leads`), so this spec is what makes "the leads for this run" a
# question with an answer.
SELECT_RUN_SPEC = """
SELECT g.spec
  FROM runs r
  JOIN goals g ON g.id = r.goal_id
 WHERE r.id = %(id)s
"""

_LEAD_COLUMNS = """
       b.id, b.place_id, b.name, b.niche_id, b.country, b.state, b.city, b.search_area,
       b.address, b.lat, b.lng, b.phone, b.email, b.website, b.instagram_handle,
       b.facebook_url, b.first_seen_at, b.last_seen_at,
       lb.total          AS score_total,
       lb.audience_index AS audience_index,
       lb.audience_band  AS audience_band,
       lb.banding_method AS banding_method,
       lb.cohort_size    AS cohort_size,
       lb.reviews        AS reviews,
       lb.scorer_version AS scorer_version,
       lb.scored_at      AS scored_at,
       v.my_verdict, v.notes, v.contacted_on, v.channel, v.outcome,
       v.updated_at      AS verdict_updated_at
"""

# The scope filter is expressed as "NULL means everything" rather than by assembling a
# WHERE clause at the call site: one statement, one plan, and nothing that can be built
# wrongly from a parameter. A goal whose spec names no city (an older goal, a goal written
# by something else) therefore lists every lead rather than none.
SELECT_LEADS = f"""
SELECT {_LEAD_COLUMNS}
  FROM businesses b
  LEFT JOIN lead_bands lb ON lb.business_id = b.id
  LEFT JOIN verdicts v ON v.business_id = b.id
 WHERE (
         %(run_id)s::uuid IS NULL
         OR b.id IN (
              SELECT e.business_id FROM enrichments e WHERE e.run_id = %(run_id)s::uuid
            )
       )
   AND (%(city)s::text IS NULL OR lower(b.city) = lower(%(city)s::text))
   AND (%(niche_ids)s::text[] IS NULL OR b.niche_id = ANY(%(niche_ids)s::text[]))
 ORDER BY lb.total DESC NULLS LAST, b.first_seen_at, b.id
 LIMIT %(limit)s
"""

SELECT_LEAD = f"""
SELECT {_LEAD_COLUMNS}
  FROM businesses b
  LEFT JOIN lead_bands lb ON lb.business_id = b.id
  LEFT JOIN verdicts v ON v.business_id = b.id
 WHERE b.id = %(id)s
"""

# INSERT ... SELECT over `businesses`, so a verdict on a lead that does not exist inserts
# nothing and returns nothing -- a 404 rather than a foreign-key violation surfacing as a
# 500. The existence check and the write are one statement, so there is no window between
# them.
UPSERT_VERDICT = """
INSERT INTO verdicts (business_id, my_verdict, notes, contacted_on, channel, outcome,
                      updated_at)
SELECT b.id, %(my_verdict)s, %(notes)s, %(contacted_on)s::date, %(channel)s, %(outcome)s,
       now()
  FROM businesses b
 WHERE b.id = %(business_id)s
-- COALESCE, not assignment: this is a PATCH, and an omitted field means "leave it", not
-- "clear it". Without it, an operator recording "I called them today" would silently erase
-- the `high` they set last week -- and Phase 8 gates automation pitches on exactly that
-- value, so the lead would quietly stop qualifying for the second offer.
--
-- The cost is that a field cannot be cleared by sending null. Clearing a verdict is rare and
-- an explicit sentinel can be added when something actually needs it; losing one silently is
-- neither rare nor recoverable.
ON CONFLICT (business_id) DO UPDATE
   SET my_verdict   = COALESCE(excluded.my_verdict, verdicts.my_verdict),
       notes        = COALESCE(NULLIF(excluded.notes, ''), verdicts.notes),
       contacted_on = COALESCE(excluded.contacted_on, verdicts.contacted_on),
       channel      = COALESCE(excluded.channel, verdicts.channel),
       outcome      = COALESCE(excluded.outcome, verdicts.outcome),
       updated_at   = now()
RETURNING business_id
"""

SELECT_NICHE_STATES = "SELECT niche_id, state FROM niche_status"

# What `Engine.execute_enrichment` needs about each business it is asked to enrich: enough
# of `businesses` to rebuild a `Lead`, the operator's verdict (the automation gate), and the
# reviews/rating Google handed over at discovery -- which live in `enrichments`, not on
# `businesses` or `scores`, and would otherwise be lost the moment a later pass re-scores.
#
# Scoped exactly like `SELECT_LEADS`: a NULL `run_id` means "ignore this filter", so a
# caller naming only `business_ids` still gets rows, and one naming only `run_id` gets
# every business that run's discovery pass actually stored an enrichment row for.
SELECT_ENRICHMENT_TARGETS = """
WITH latest_google AS (
    -- `DISTINCT ON` picks the freshest sighting, same rule `lead_bands` (0009) uses for
    -- the same table -- a business seen in March and again in August should be scored on
    -- August's numbers, not averaged or arbitrarily chosen.
    SELECT DISTINCT ON (e.business_id)
           e.business_id, e.data
      FROM enrichments e
     WHERE e.source = 'google_maps'
     ORDER BY e.business_id, e.fetched_at DESC, e.id DESC
)
SELECT b.id, b.place_id, b.name, b.niche_id, b.country, b.state, b.city, b.address,
       b.lat, b.lng, b.phone, b.email, b.website, b.instagram_handle, b.facebook_url,
       v.my_verdict,
       coalesce(lg.data, '{}'::jsonb) AS google_evidence
  FROM businesses b
  LEFT JOIN verdicts v ON v.business_id = b.id
  LEFT JOIN latest_google lg ON lg.business_id = b.id
 WHERE (
         %(run_id)s::uuid IS NULL
         OR b.id IN (
              SELECT e.business_id FROM enrichments e WHERE e.run_id = %(run_id)s::uuid
            )
       )
   AND (%(business_ids)s::uuid[] IS NULL OR b.id = ANY(%(business_ids)s::uuid[]))
 ORDER BY b.first_seen_at, b.id
"""


# --- what a pass produced -------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentReport:
    """What one enrichment pass did. Mirrors `RunReport`'s spirit: counts the CLI prints
    and an API response returns, not the findings themselves -- those are already durable,
    in `enrichments`, `scores`, `automation_opportunities` and `outreach`."""

    business_ids: tuple[UUID, ...]
    enriched: int
    scored: int
    offers_detected: int
    outreach_written: int
    usage: dict[str, Any]


@dataclass(frozen=True)
class RunReport:
    """What one executed pass did. The CLI prints it; a worker will record it."""

    run_id: UUID
    goal_id: UUID
    outcome: DiscoveryOutcome
    scored: int
    export_path: Path | None = None

    @property
    def found(self) -> int:
        return len(self.outcome.businesses)


# --- the enrichment pass ----------------------------------------------------------------

#: The name `_score` never uses -- "google-only" says the discovery pass wrote it, and this
#: says an enrichment pass did, so `lead_bands` and any operator reading `scores` can tell
#: which claims came from a paid-listing snapshot and which came from actually looking at
#: the business's own site, ads and social presence.
ENRICHED_SCORER_VERSION = "enriched"

#: The verdict gate on automation-opportunity detection and the automation pitch, stated in
#: `automations.py`'s own docstring: the pitch is expensive to generate and worthless before
#: the operator has judged the lead worth pursuing. `website_pitch` carries no such gate --
#: it is the first offer, made to every enriched business regardless of verdict.
AUTOMATION_ELIGIBLE_VERDICTS = frozenset({"high", "medium"})

#: `AutomationOffer` carries no confidence of its own, and detection is not a partial-credit
#: process: `firing_offers` (via `offer_fires`/`missing_signals`) requires every one of an
#: offer's `required_signals` to be observed, with no loose or partial match. There is
#: therefore no fractional number to report -- an offer either fired on complete evidence or
#: it is not recorded at all -- so every firing offer is written at full confidence. This is
#: a stated choice, not a measurement: it says "every required signal was actually observed"
#: and nothing more.
AUTOMATION_OFFER_CONFIDENCE = 1.0

#: `outreach.channel` is NOT NULL and this phase does not yet choose one per business -- that
#: is a later decision (which contact method is on file, which the operator prefers). Email
#: is the least intrusive default for drafted prose nobody has approved to send yet.
DEFAULT_OUTREACH_CHANNEL = "email"

WEBSITE_PITCH = "website_pitch"
AUTOMATION_PITCH = "automation_pitch"


class _RecordingEvidenceStore:
    """Wraps a `Repository` and remembers which `enrichments` row backs which source.

    `EnrichmentService` writes through `EvidenceStore` and returns only counts -- see its
    own docstring on why: enrichment and scoring are its whole job, and reconciling
    observations into a different table's evidence is a decision for whatever reads them
    later. This wrapper is that later reader's seam: `automation_opportunities.evidence` and
    `outreach.evidence` are both required to name the row a claim traces back to, per
    `migrations/0005_automation.sql`'s own comment, and the only way to learn a row's id
    without changing `EnrichmentService` is to intercept the write it already makes.

    Never constructed with more than one business's writes interleaved in a way that would
    confuse it: `EnrichmentService.enrich()` writes at most one row per source per business
    per call, so `rows_by_business[business_id][source]` is never overwritten by anything
    but a genuine repeat for the same business, which does not happen within one pass.
    """

    def __init__(self, repository: Repository) -> None:
        self._repository = repository
        self.rows_by_business: dict[UUID, dict[str, int]] = {}

    def insert_enrichment(
        self,
        business_id: UUID,
        source: str,
        status: str,
        data: dict[str, Any],
        *,
        source_url: str | None = None,
        run_id: UUID | None = None,
    ) -> Any:
        row = self._repository.insert_enrichment(
            business_id, source, status, data, source_url=source_url, run_id=run_id
        )
        self.rows_by_business.setdefault(business_id, {})[source] = row.id
        return row

    def insert_contact(
        self,
        business_id: UUID,
        name: str,
        source: str,
        *,
        role: str = "unknown",
        phone: str | None = None,
        email: str | None = None,
        source_url: str | None = None,
        confidence: float = 0.5,
    ) -> Any:
        return self._repository.insert_contact(
            business_id,
            name,
            source,
            role=role,
            phone=phone,
            email=email,
            source_url=source_url,
            confidence=confidence,
        )


# --- the service layer ----------------------------------------------------------------


class Engine:
    """Every operation this system offers, over collaborators it was handed.

    Nothing here constructs a billed provider unless asked to: `maps_provider()` is the one
    factory that can, and it refuses -- 503, not 500 -- when no key is configured, because
    "the operator forgot a key" and "the server is broken" call for different reactions from
    whoever is reading the response.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        pool: ConnectionPool | None = None,
        maps_provider: MapsProvider | None = None,
        resolver: LocationResolver | None = None,
        budget: SearchBudget | None = None,
        policy: CellPolicy = BREADTH_FIRST,
        owns_pool: bool = False,
        clock: Callable[[], datetime] = utc_now,
        enrichment_provider: WebProvider | None = None,
        enrichment_fallback: WebProvider | None = None,
        llm_router: LLMRouter | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self._pool = pool
        self._owns_pool = owns_pool
        self._maps_provider = maps_provider
        self._resolver = resolver
        self._budget = budget
        self._clock = clock
        self._base_resolver: LocationResolver | None = resolver
        self._enrichment_provider = enrichment_provider
        self._enrichment_fallback = enrichment_fallback
        self._llm_router = llm_router
        self.repository: Repository | None = Repository(pool) if pool is not None else None

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        """Dispose of the pool, but only the one this object opened."""
        if self._owns_pool and self._pool is not None:
            self._pool.close()

    # -- plumbing ----------------------------------------------------------------------

    @property
    def connect(self) -> Callable[[], Any]:
        """The connection factory every store in this project takes."""
        return self.require_pool().connection

    def require_pool(self) -> ConnectionPool:
        if self._pool is None:
            raise ApiProblem(
                503,
                "database_not_configured",
                "No database is configured. Set LEAD_ENGINE_DSN and apply the migrations.",
            )
        return self._pool

    def require_repository(self) -> Repository:
        repository = self.repository
        if repository is None:
            self.require_pool()
            raise ApiProblem(503, "database_not_configured", "No database is configured.")
        return repository

    def _select(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Read rows as dicts, so a response model validates by name rather than position.

        `dict_row` rather than `class_row`: these queries join `lead_bands` and `verdicts`
        onto `businesses` and their result has no row dataclass, and naming the columns is
        what stops a reordered SELECT list shifting `phone` into `email`.
        """
        with self.require_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # -- providers ---------------------------------------------------------------------

    def maps_configured(self) -> bool:
        """Whether a search could be issued at all.

        Duck-typed on `api_key`, which both `SearchApiClient` and `FixtureMapsProvider`
        expose -- a fixture provider is always configured, and a fixture-driven demo must
        not report the system as unwired.
        """
        if self._maps_provider is not None:
            return bool(getattr(self._maps_provider, "api_key", None))
        return self.settings.searchapi_key_value is not None

    def maps_provider(self) -> MapsProvider:
        """The provider a search would go through. 503 when there is no key.

        Constructing the live client costs nothing and opens no socket; the credit is spent
        inside `search_places` and nowhere else.
        """
        if self._maps_provider is not None:
            return self._maps_provider
        key = self.settings.searchapi_key_value
        if key is None:
            raise ApiProblem(
                503,
                "provider_not_configured",
                "SEARCHAPI_KEY is not configured, so no search can be issued. This is a "
                "missing key, not a failed search.",
                {"provider": SEARCHAPI},
            )
        # Kept, not rebuilt per call: the client owns an `httpx.Client`, and a plan followed
        # by a run would otherwise open two and close neither.
        self._maps_provider = SearchApiClient(key, budget=self.budget())
        return self._maps_provider

    def require_maps_provider(self) -> MapsProvider:
        return self.maps_provider()

    def budget(self) -> SearchBudget | None:
        """The ledger, or None when there is no database to keep it in."""
        if self._budget is not None:
            return self._budget
        if self._pool is None:
            return None
        return SearchBudget(self._pool.connection)

    def key_fingerprint(self) -> str:
        """Which allowance this process draws down. '' when no key is configured."""
        key = self.settings.searchapi_key_value
        return fingerprint(key) if key else ""

    def remaining_budget(self) -> int | None:
        """Searches left on this key, or None when that cannot be known.

        None is not zero and must not be printed as one: no ledger row means nobody has
        seeded an allowance yet, and a caller told "0 remaining" would abort a run that
        could have gone ahead.
        """
        budget = self.budget()
        if budget is None:
            return None
        try:
            return budget.remaining_total(SEARCHAPI, key_fingerprint=self.key_fingerprint())
        except BudgetNotConfigured:
            return None

    def ensure_budget(self) -> None:
        """Seed this key's allowance if the ledger has never seen it. Idempotent.

        Called at startup, which is where the budget module says it belongs: a rotated key
        legitimately has no row until this runs, and a provisioning script someone has to
        remember is a step that gets forgotten exactly once.
        """
        budget = self.budget()
        if budget is None or not self.maps_configured():
            return
        try:
            budget.ensure_allowance(
                SEARCHAPI,
                self.settings.search_budget,
                key_fingerprint=self.key_fingerprint(),
            )
        except Exception:
            # A ledger that cannot be seeded is a reason to log, not a reason to refuse to
            # boot: every read-only endpoint still works, and the search endpoint fails
            # loudly on its own terms.
            logger.warning("could not seed the search budget", exc_info=True)

    # -- geography ---------------------------------------------------------------------

    def _default_resolver(self) -> LocationResolver:
        if self._base_resolver is None:
            # Seed first: Indian metro neighbourhoods are the common case and the seed makes
            # them free, offline and instant. Nominatim answers everything else, once, behind
            # the cache below.
            self._base_resolver = ChainResolver(SeedResolver(), NominatimResolver())
        return self._base_resolver

    @contextmanager
    def resolver(self) -> Iterator[LocationResolver]:
        """A resolver for the duration of one operation, cached when there is a database.

        `CachingResolver` takes a live connection rather than a factory, so the connection
        is held for the whole operation and released with it. Without a database the chain
        still resolves; it just pays the geocoder again next time.
        """
        base = self._default_resolver()
        if self._pool is None:
            yield base
            return
        with self._pool.connection() as conn:
            yield CachingResolver(base, conn)

    def resolve(self, scope: GeoScope) -> tuple[ResolvedLocation, ...]:
        """Every point this scope fans out to. Raises rather than returning a partial list."""
        with self.resolver() as resolver:
            return resolve_scope(scope, resolver)

    # -- reporting ---------------------------------------------------------------------

    def niche_states(self) -> dict[str, str]:
        """`niche_id -> verification state` for every niche the table knows about."""
        if self._pool is None:
            return {}
        return {row["niche_id"]: row["state"] for row in self._select(SELECT_NICHE_STATES, {})}

    def niches(self) -> NicheListOut:
        """The registry, in registry order, each with how far its evidence has got.

        Order is `NICHE_PROFILES`', not the database's: the registry is the catalogue a
        caller renders, and sorting it by whatever `niche_status` happens to hold would
        reorder the list every time a niche was searched.
        """
        states = self.niche_states()
        return NicheListOut(
            niches=[
                NicheOut(**payload, verification=states.get(payload["id"], DEFAULT_VERIFICATION))
                for payload in niche_payload()
            ]
        )

    def config_status(self) -> ConfigStatusOut:
        """Which providers are wired up, and what is left of the billed allowance."""
        providers = self.settings.provider_status()
        providers[SEARCHAPI] = self.maps_configured()
        return ConfigStatusOut(
            database_configured=self._pool is not None or self.settings.database_configured,
            providers=providers,
            llm_configured=self.settings.llm_configured,
            search_budget=BudgetOut(
                provider=SEARCHAPI,
                configured=providers[SEARCHAPI],
                remaining=self.remaining_budget(),
            ),
        )

    # -- searching ---------------------------------------------------------------------

    def plan(self, spec: SearchSpec) -> SearchPlanOut:
        """What a search WOULD cost. Resolves geography; spends nothing.

        The provider is checked but never called, so a plan is honest about whether the run
        it describes could actually be performed.
        """
        self.require_maps_provider()
        locations = self.resolve(spec.scope)
        return SearchPlanOut(
            niches=list(spec.niche_ids),
            locations=_locations_out(locations),
            planned_searches=self.planned_searches(len(locations), len(spec.niche_ids)),
            search_budget_remaining=self.remaining_budget(),
        )

    def planned_searches(self, locations: int, niches: int) -> int:
        """Billed searches a pass over this many (area x niche) cells would issue."""
        return locations * niches * self.policy.searches_per_area_niche

    def open_run(
        self, spec: SearchSpec, *, trigger: str = "cli", status: str = "running"
    ) -> tuple[UUID, UUID]:
        """Record the goal and the run for a search, and queue nothing. Returns both ids.

        Split out from `start_search` because the two callers must not both happen. The API
        opens a run AND enqueues tasks, for a worker to perform. The CLI opens a run and
        performs the work itself -- if it also enqueued, the same cells would be searched a
        second time whenever a worker was eventually started, at a credit apiece.
        """
        repository = self.require_repository()
        goal_id, run_id = uuid4(), uuid4()
        with repository.transaction():
            repository.create_goal(
                _goal_name(spec), _goal_spec(spec, self.policy), goal_id=goal_id
            )
            repository.create_run(goal_id, trigger=trigger, status=status, run_id=run_id)
        return goal_id, run_id

    def fail_run(self, run_id: UUID, reason: str) -> None:
        """Close a run that could not finish, so it does not sit `running` forever."""
        self.require_repository().finish_run(run_id, "failed", stats={"error": reason})

    def start_search(self, spec: SearchSpec, *, trigger: str = "api") -> SearchAcceptedOut:
        """Accept a search: goal, run, and one queued task per (area x niche).

        NOTHING IS SEARCHED HERE. The provider is checked for configuration and never
        called, the geography is resolved (free, cached, and required so the receipt can
        state what a worker will actually search), and the billed work is enqueued. An HTTP
        request that spent a credit would be one a browser refresh could spend twice.
        """
        self.require_maps_provider()
        repository = self.require_repository()
        locations = self.resolve(spec.scope)
        areas: tuple[str | None, ...] = spec.scope.areas or (None,) * len(locations)

        goal_id, run_id = uuid4(), uuid4()
        enqueued = 0
        with repository.transaction():
            repository.create_goal(
                _goal_name(spec),
                _goal_spec(spec, self.policy),
                goal_id=goal_id,
            )
            repository.create_run(goal_id, trigger=trigger, status=QUEUED, run_id=run_id)
            for area, location in zip(areas, locations, strict=True):
                for niche_id in spec.niche_ids:
                    inserted = repository.enqueue(
                        run_id,
                        DISCOVER,
                        _task_payload(spec, niche_id, area, location, self.policy),
                        # Names the unit of work, not this call: a resumed run has to
                        # recognise the cells it already queued rather than queue them
                        # again at a credit apiece.
                        idem_key=f"{DISCOVER}:{goal_id}:{niche_id}:{normalize(area) or 'city'}",
                    )
                    enqueued += 1 if inserted else 0
            repository.append_event(
                run_id,
                "run.queued",
                {
                    "niches": list(spec.niche_ids),
                    "locations": [location.label for location in locations],
                    "enqueued_tasks": enqueued,
                    "policy": self.policy.name,
                },
            )

        planned = enqueued * self.policy.searches_per_area_niche
        return SearchAcceptedOut(
            run_id=run_id,
            goal_id=goal_id,
            status=QUEUED,
            niches=list(spec.niche_ids),
            locations=_locations_out(locations),
            enqueued_tasks=enqueued,
            planned_searches=planned,
            search_budget_remaining=self.remaining_budget(),
            message=(
                f"Queued {enqueued} discovery task(s). Nothing has been searched and no "
                f"credit has been spent; running them will cost up to {planned} billed "
                "searches."
            ),
        )

    # -- runs and leads ----------------------------------------------------------------

    def runs(self, limit: int = DEFAULT_RUN_LIMIT) -> RunListOut:
        rows = self._select(SELECT_RUNS, {"limit": _bounded(limit, 1, MAX_RUN_LIMIT)})
        return RunListOut(runs=[RunOut.model_validate(row) for row in rows])

    def run(self, run_id: UUID) -> RunOut:
        rows = self._select(SELECT_RUN, {"id": run_id})
        if not rows:
            raise ApiProblem(404, "run_not_found", f"No run with id {run_id}.")
        return RunOut.model_validate(rows[0])

    def latest_run_id(self) -> UUID | None:
        rows = self._select(SELECT_LATEST_RUN, {})
        return rows[0]["id"] if rows else None

    def run_scope(self, run_id: UUID) -> dict[str, Any]:
        """The goal spec behind a run, which is the only record of what it was looking for."""
        rows = self._select(SELECT_RUN_SPEC, {"id": run_id})
        if not rows:
            raise ApiProblem(404, "run_not_found", f"No run with id {run_id}.")
        spec = rows[0]["spec"]
        return spec if isinstance(spec, dict) else {}

    def leads(self, run_id: UUID | None = None, *, limit: int = DEFAULT_LEAD_LIMIT) -> LeadListOut:
        """Every lead in a run's scope, latest run when none is named.

        Scoped through `enrichments.run_id`, not through the goal's city and niches.

        `businesses` deliberately carries no run id: a business is discovered once and lives
        on, so a single column could only ever record which run FIRST found it. What a run
        actually owns is the observations it paid for, and every business a pass stores gets
        an enrichment row stamped with that run.

        This is also why a second run does not re-list the first run's leads. Boundary dedup
        means a pass only ever stores businesses the system did not already hold, so a
        re-encountered business writes no new enrichment and belongs to the run that bought
        it. "What did this run find" and "what have I already worked" stay different
        questions -- which is the whole point of paying for discovery only once.

        The goal's scope still applies as a second filter, so an older goal or one written by
        something else still lists sensibly.
        """
        if run_id is None:
            run_id = self.latest_run_id()
            if run_id is None:
                raise ApiProblem(
                    404,
                    "run_not_found",
                    "No runs exist yet. Start one with POST /api/search.",
                )
        scope = self.run_scope(run_id)
        niche_ids = scope.get("niche_ids") or None
        rows = self._select(
            SELECT_LEADS,
            {
                "run_id": run_id,
                "city": scope.get("city"),
                "niche_ids": list(niche_ids) if niche_ids else None,
                "limit": _bounded(limit, 1, MAX_LEAD_LIMIT),
            },
        )
        leads = [LeadOut.model_validate(row) for row in rows]
        return LeadListOut(run_id=run_id, count=len(leads), leads=leads)

    def lead(self, lead_id: UUID) -> LeadOut:
        rows = self._select(SELECT_LEAD, {"id": lead_id})
        if not rows:
            raise ApiProblem(404, "lead_not_found", f"No lead with id {lead_id}.")
        return LeadOut.model_validate(rows[0])

    def set_verdict(self, lead_id: UUID, update: VerdictUpdate) -> None:
        """Record the operator's verdict. 404 when the lead does not exist."""
        with self.require_pool().connection() as conn:
            row = conn.execute(
                UPSERT_VERDICT,
                {
                    "business_id": lead_id,
                    "my_verdict": update.my_verdict,
                    "notes": update.notes,
                    "contacted_on": update.contacted_on,
                    "channel": update.channel,
                    "outcome": update.outcome,
                },
            ).fetchone()
        if row is None:
            raise ApiProblem(404, "lead_not_found", f"No lead with id {lead_id}.")

    # -- performing a run --------------------------------------------------------------

    def execute_run(
        self,
        spec: SearchSpec,
        *,
        goal_id: UUID,
        run_id: UUID,
        max_searches: int | None = None,
        output_dir: Path | str | None = None,
    ) -> RunReport:
        """Perform a discovery pass, score what it found, and write the sheet.

        THE BILLED PATH. Every credit this system spends is spent inside the provider call
        this reaches, which is why nothing in the HTTP layer calls it: `POST /api/search`
        enqueues, and a worker -- or the CLI, deliberately, in front of an operator who has
        just been shown the remaining budget -- calls this.
        """
        repository = self.require_repository()
        connect = self.connect
        provider = self.require_maps_provider()

        with self.resolver() as resolver:
            service = DiscoveryService(
                provider=provider,
                resolver=resolver,
                businesses=RepositoryBusinesses(repository, connect),
                cells=SearchCellStore(connect),
                niche_status=NicheStatusStore(connect),
                budget=self.budget(),
                clock=self._clock,
            )
            outcome = service.discover(
                DiscoveryRequest(
                    goal_id=goal_id,
                    scope=spec.scope,
                    niche_ids=list(spec.niche_ids),
                    limit=spec.limit,
                    policy=self.policy,
                    max_searches=max_searches,
                    # Every `google_maps` enrichment row this pass writes is stamped with
                    # this run's id -- omitted here, it silently defaulted to NULL, and
                    # `_enrichment_targets`'s run_id lookup (added when execute_enrichment
                    # was wired in) can only ever find a business through that stamp. A
                    # business discovered without this would be permanently invisible to
                    # `--enrich` unless the caller separately named it by business_ids.
                    run_id=run_id,
                )
            )

        scored = self._score(outcome, run_id=run_id)
        path = self._export(output_dir) if output_dir is not None else None

        repository.finish_run(
            run_id,
            "succeeded" if outcome.businesses else "empty",
            stats={
                "searches_spent": outcome.searches_spent,
                "cells_searched": outcome.cells_searched,
                "cells_planned": outcome.cells_planned,
                "new_businesses": len(outcome.businesses),
                "scored": scored,
                "stopped": outcome.stopped,
                "policy": outcome.policy,
                "starved_niches": list(outcome.starved_niches),
                "export": str(path) if path else None,
            },
        )
        return RunReport(
            run_id=run_id, goal_id=goal_id, outcome=outcome, scored=scored, export_path=path
        )

    def _score(self, outcome: DiscoveryOutcome, *, run_id: UUID) -> int:
        """Score every business this pass added, and index the audience it arrived with.

        The score reads the GAP -- no website, a social-only link, a public phone. The index
        reads the DEMAND, from the review count and rating that came free in the same billed
        response. Both halves are needed: without the index every business bands `unknown`
        and the sheet ranks on gap alone, which puts a dead salon with no site above a packed
        one with a bad site. Exactly backwards.

        `audience_index` is None only when Google gave no numbers at all. The banding view
        reads that as 'unknown', which is a different and more useful statement than the 0.0
        that would mean "we looked, and nobody goes there".

        Followers and engagement are still absent -- the Instagram pass supplies those and
        rescores. `audience_index` renormalises over whatever it is given, so a
        reviews-and-rating index is on the same 0-1 scale as a fully enriched one rather than
        a deflated version of it.
        """
        repository = self.require_repository()
        written = 0
        for found in outcome.businesses:
            if found.business_id is None:
                continue
            profile = NICHE_PROFILES.get(found.niche_id)
            breakdown = score_lead(found.lead, [profile] if profile else None)
            signals = found.evidence or {}
            index = audience_index(
                Evidence(reviews=signals.get("reviews"), rating=signals.get("rating"))
            )
            repository.insert_score(
                found.business_id,
                "google-only",
                total=breakdown.total,
                demand=breakdown.demand,
                website_gap=breakdown.website_gap,
                budget=breakdown.budget,
                reachability=breakdown.reachability,
                signals=list(breakdown.signals),
                evidence={"pitch_angle": breakdown.pitch_angle, "run_id": str(run_id)},
                audience_index=index,
            )
            written += 1
        return written

    def _export(self, output_dir: Path | str) -> Path:
        directory = Path(output_dir)
        stamp = self._clock().strftime("%Y%m%d-%H%M%S")
        target = directory / f"leads-{stamp}.xlsx"
        with self.require_pool().connection() as conn:
            return excel.export(conn, target)

    # -- enrichment ----------------------------------------------------------------------

    def enrichment_configured(self) -> bool:
        """Whether an enrichment pass could look anything up at all.

        Duck-typed on `api_key`, same as `maps_configured()` -- an injected fake provider
        (a fixture, a test double) is always "configured", because the point of injecting
        one is to run the pipeline without a real key.
        """
        if self._enrichment_provider is not None:
            return bool(getattr(self._enrichment_provider, "api_key", None))
        return self.settings.tinyfish_key_value is not None

    def enrichment_provider(self) -> WebProvider:
        """The primary web provider an enrichment pass searches and fetches through.

        Unlike `maps_provider()`, this never raises for a missing key: TinyFish's own
        `search`/`fetch` calls raise `ProviderError` on a real 401, and
        `EnrichmentService._attempt` already turns that into a `blocked`/`error` row per
        business rather than aborting the pass -- refusing up front here would only turn a
        graceful per-business degradation into a hard stop for the whole run.
        """
        if self._enrichment_provider is not None:
            return self._enrichment_provider
        self._enrichment_provider = TinyFishClient(self.settings.tinyfish_key_value)
        return self._enrichment_provider

    def enrichment_service(self, *, store: EvidenceStore) -> EnrichmentService:
        """A freshly wired `EnrichmentService` for one pass.

        `fallback` stays whatever was injected (usually None): this codebase has no
        Firecrawl client yet, so there is nothing to build here even though
        `FIRECRAWL_API_KEY` is a recognised setting -- `EnrichmentService` already treats a
        missing fallback as a supported configuration (`fallback_unavailable`, not a crash).

        The cache is `PostgresCache` when a database is configured -- the durable negative
        cache `enrich_many` is documented to respect -- and an in-memory one otherwise, the
        same "no database is still a supported configuration" rule `resolver()` follows.
        """
        cache = PostgresCache(self.connect) if self._pool is not None else InMemoryCache()
        return EnrichmentService(
            self.enrichment_provider(),
            fallback=self._enrichment_fallback,
            store=store,
            cache=cache,
            clock=self._clock,
        )

    def llm_router(self) -> LLMRouter:
        """The provider chain outreach prose is generated over.

        `LLMRouter.from_settings` already skips every provider without a key and still
        returns a router -- zero configured providers is the supported "template only"
        configuration, not an error, so there is nothing to gate here the way
        `maps_provider()` gates a missing SearchAPI key.

        Circuit state and the response cache are Postgres-backed when a database is
        configured, for the reason `router.py`'s own module docstring gives: an in-memory
        breaker resets on every restart, and a persisted cache is what stops a retried
        request re-billing a provider for a prompt it already answered.
        """
        if self._llm_router is not None:
            return self._llm_router
        store = DatabaseCircuitStore(self.connect) if self._pool is not None else None
        cache = ResponseCache(self.connect) if self._pool is not None else None
        self._llm_router = LLMRouter.from_settings(self.settings, store=store, cache=cache)
        return self._llm_router

    def _enrichment_targets(
        self, *, run_id: UUID | None, business_ids: Sequence[UUID] | None
    ) -> list[dict[str, Any]]:
        """Everything `execute_enrichment` needs about the businesses it was asked for.

        A protected method rather than inlined SQL in `execute_enrichment`, so a test can
        override it on a subclass -- exactly how `tests/test_cli.py`'s `BrokenEngine`
        overrides `execute_run` -- and exercise the orchestration below without a database.
        """
        if run_id is None and not business_ids:
            raise ApiProblem(
                422,
                "invalid_request",
                "execute_enrichment needs a run_id, business_ids, or both.",
            )
        return self._select(
            SELECT_ENRICHMENT_TARGETS,
            {
                "run_id": run_id,
                "business_ids": list(business_ids) if business_ids else None,
            },
        )

    def execute_enrichment(
        self,
        *,
        run_id: UUID | None = None,
        business_ids: Sequence[UUID] | None = None,
        use_ai: bool = True,
        channel: str = DEFAULT_OUTREACH_CHANNEL,
    ) -> EnrichmentReport:
        """Enrich, re-score, detect automation opportunities, and draft outreach.

        The chain this assembles -- already individually proven, per the phase's own
        design note -- is: `EnrichmentService.enrich_many` over the named businesses ->
        `score_lead` on what it found, written as a `scores` row versioned `"enriched"` ->
        `firing_offers(niche_id, breakdown.signals | enrichment_signals)` for every
        `high`/`medium`-verdict business -> one `automation_opportunities` row per firing
        offer -> `outreach` prose over a `website_pitch` for every enriched business and an
        `automation_pitch` for every firing offer.

        WHAT THIS PASS CAN HONESTLY CLAIM
        ----------------------------------
        `automations.ENRICHMENT_SIGNALS` names five signals this pass never asserts:
        "high review count", "owner replies absent", "active social presence",
        "instagram-only catalogue" and "dm ordering". None of them are observable from what
        `EnrichmentService` gathers today -- follower counts, engagement and post content
        belong to the Instagram *browser* stage (`enrichment/social.py`'s own docstring
        says so: only the handle is stored here), and nothing anywhere reads review
        replies. Rather than approximate them, `_translate_enrichment_signals` simply never
        claims them, which means `order_intake`, `catalog_whatsapp` and `review_response`
        can never fire from this pass alone. That is the honest answer -- no data, no claim,
        no offer -- not a bug; a later pass with browser-sourced evidence can extend the
        translation without touching this method's contract.

        Never raises for a business the enrichment or LLM step could not fully answer:
        `EnrichmentService` degrades per-question to `blocked`/`error` rows rather than
        raising, and `LLMRouter.generate` is guaranteed to return text -- cache, a live
        model, or the deterministic fallback -- never an exception for a provider failure.
        So a business with no new evidence this pass still gets re-scored and pitched from
        whatever it already has on file.
        """
        repository = self.require_repository()
        targets = self._enrichment_targets(run_id=run_id, business_ids=business_ids)
        if not targets:
            return EnrichmentReport(
                business_ids=(), enriched=0, scored=0, offers_detected=0,
                outreach_written=0, usage={},
            )

        store = _RecordingEvidenceStore(repository)
        service = self.enrichment_service(store=store)
        router = self.llm_router() if use_ai else None

        requests = [
            EnrichmentRequest(
                business_id=row["id"],
                name=row["name"],
                city=row["city"],
                listed_website=row["website"],
                listed_handle=row["instagram_handle"],
                country=row["country"] or ads_module.DEFAULT_COUNTRY,
                run_id=run_id,
            )
            for row in targets
        ]
        summary = service.enrich_many(requests)

        scored = 0
        offers_detected = 0
        outreach_written = 0
        done: list[UUID] = []

        for row, enrichment in zip(targets, summary.businesses, strict=True):
            business_id = row["id"]
            done.append(business_id)
            row_ids = store.rows_by_business.get(business_id, {})

            profile = NICHE_PROFILES.get(row["niche_id"])
            lead = _lead_from_row(row, enrichment)
            breakdown = score_lead(lead, [profile] if profile else None)
            evidence = _rescored_evidence(row, enrichment)
            index = audience_index(evidence)
            enrichment_signals, signal_evidence = _translate_enrichment_signals(
                enrichment, row_ids
            )
            observed = set(breakdown.signals) | enrichment_signals

            repository.insert_score(
                business_id,
                ENRICHED_SCORER_VERSION,
                total=breakdown.total,
                demand=breakdown.demand,
                website_gap=breakdown.website_gap,
                budget=breakdown.budget,
                reachability=breakdown.reachability,
                signals=list(breakdown.signals),
                evidence={
                    "pitch_angle": breakdown.pitch_angle,
                    "run_id": str(run_id) if run_id else None,
                },
                audience_index=index,
            )
            scored += 1

            website_body = _generate_website_pitch(router, lead, breakdown, profile)
            repository.insert_outreach(
                business_id,
                WEBSITE_PITCH,
                channel,
                website_body,
                evidence={"signals": list(breakdown.signals)},
            )
            outreach_written += 1

            if row["my_verdict"] in AUTOMATION_ELIGIBLE_VERDICTS:
                for offer in firing_offers(row["niche_id"], observed):
                    trigger_signals = list(offer.required_signals)
                    evidence_ids = sorted(
                        {
                            enrichment_id
                            for signal in trigger_signals
                            for enrichment_id in signal_evidence.get(signal, ())
                        }
                    )
                    repository.insert_automation_opportunity(
                        business_id,
                        offer.id,
                        confidence=AUTOMATION_OFFER_CONFIDENCE,
                        trigger_signals=trigger_signals,
                        evidence={"enrichment_ids": evidence_ids},
                    )
                    offers_detected += 1

                    automation_body = _generate_automation_pitch(
                        router, row["name"], offer, trigger_signals
                    )
                    repository.insert_outreach(
                        business_id,
                        AUTOMATION_PITCH,
                        channel,
                        automation_body,
                        evidence={
                            "opportunity_id": offer.id,
                            "trigger_signals": trigger_signals,
                        },
                    )
                    outreach_written += 1

        return EnrichmentReport(
            business_ids=tuple(done),
            enriched=len(summary.businesses),
            scored=scored,
            offers_detected=offers_detected,
            outreach_written=outreach_written,
            usage=summary.usage.as_dict(),
        )


# --- construction ---------------------------------------------------------------------


def build_pool(dsn: str, *, min_size: int = 1, max_size: int = 8, **kwargs: Any) -> ConnectionPool:
    """The one place a pool is opened. `Repository(pool)` is built from it, never beside it."""
    return ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True, **kwargs)


def build_engine(
    settings: Settings,
    *,
    maps_provider: MapsProvider | None = None,
    resolver: LocationResolver | None = None,
    policy: CellPolicy = BREADTH_FIRST,
) -> Engine:
    """An Engine wired to whatever this environment actually has.

    A missing DSN is not an error here. `/health`, `/api/niches` and `/api/config/status`
    answer without a database, and the endpoints that cannot say so with a 503 that names
    the missing piece.
    """
    dsn = settings.dsn
    pool = build_pool(dsn) if dsn else None
    return Engine(
        settings,
        pool=pool,
        maps_provider=maps_provider,
        resolver=resolver,
        policy=policy,
        owns_pool=pool is not None,
    )


def fixture_provider(fixture_dir: Path | str) -> FixtureMapsProvider:
    """Recorded responses instead of billed ones.

    `strict=False`: a sweep whose cursor has walked past the last recorded page should stop,
    not fail. That is exactly the second run of a fixture demo, and a `FixtureNotFound`
    there would look like a broken pipeline rather than the end of the corpus.
    """
    return FixtureMapsProvider(fixture_dir, strict=False)


# --- helpers ---------------------------------------------------------------------------


def _bounded(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _locations_out(locations: Sequence[ResolvedLocation]) -> list[ResolvedLocationOut]:
    return [
        ResolvedLocationOut.model_validate(location, from_attributes=True)
        for location in locations
    ]


def _goal_name(spec: SearchSpec) -> str:
    where = spec.scope.city
    if spec.scope.areas:
        where = f"{where} ({len(spec.scope.areas)} area(s))"
    return f"{where}: {', '.join(spec.niche_ids)}"[:200]


def _goal_spec(spec: SearchSpec, policy: CellPolicy) -> dict[str, Any]:
    """What the run was asked for, in the goal's own words.

    This is the only durable record of a run's scope, and `Engine.leads` reads it back, so
    the key names here are a contract with that query rather than free-form metadata.
    """
    return {
        "city": spec.scope.city,
        "state": spec.scope.state,
        "country": spec.scope.country,
        "areas": list(spec.scope.areas),
        "niche_ids": list(spec.niche_ids),
        "limit": spec.limit,
        "use_ai": spec.use_ai,
        "policy": policy.name,
    }


def _location_payload(location: ResolvedLocation) -> dict[str, Any]:
    return {
        "label": location.label,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "radius_meters": location.radius_meters,
        "precision": location.precision,
        "gl": location.gl,
        "source_query": location.source_query,
    }


def _task_payload(
    spec: SearchSpec,
    niche_id: str,
    area: str | None,
    location: ResolvedLocation,
    policy: CellPolicy,
) -> dict[str, Any]:
    """Everything a worker needs to perform one cell without re-deciding anything.

    The resolved point travels with the task. A worker that geocoded again could get a
    different answer from the one the receipt promised the operator, and the receipt is what
    the operator approved.
    """
    return {
        "niche_id": niche_id,
        "area": area,
        "location": _location_payload(location),
        "scope": {
            "city": spec.scope.city,
            "state": spec.scope.state,
            "country": spec.scope.country,
        },
        "limit": spec.limit,
        "use_ai": spec.use_ai,
        "policy": policy.name,
    }


# --- enrichment helpers ------------------------------------------------------------------


def _lead_from_row(row: dict[str, Any], enrichment: BusinessEnrichment) -> Lead:
    """Rebuild the `Lead` `score_lead` wants from a stored business plus this pass's finding.

    `businesses` carries no `category`/`raw_categories` -- those are provider fields never
    persisted past discovery's scoring pass -- so they are supplied empty here rather than
    guessed. The only effect is that the "detailed business listing" bonus signal, which
    reads `len(raw_categories) > 1`, cannot fire on a re-score; every other signal is
    unaffected because an explicit `profile` (from `niche_id`) is always passed to
    `score_lead`, so nothing here depends on `category` to resolve one.
    """
    niche_id = row["niche_id"]
    return Lead(
        name=row["name"],
        category=niche_id or "",
        address=row["address"] or "",
        city=row["city"],
        latitude=row["lat"],
        longitude=row["lng"],
        phone=row["phone"],
        website=_effective_website(row, enrichment.website),
        source_url=None,
        raw_categories=[],
        provider_id=row["place_id"],
        matched_niches=[niche_id] if niche_id else [],
    )


def _effective_website(row: dict[str, Any], website: Any) -> str | None:
    """The website `score_lead` should see: what this pass confirmed, or the discovery
    listing when this pass could not tell.

    `website.told` is False exactly when the verdict is `unknown` -- the search never ran,
    or came back refused -- and reporting that as "no website" would claim ground this pass
    never actually covered.
    """
    if website is None or not website.told:
        return row["website"]
    if website.verdict == website_module.NO_SITE:
        return None
    return website.url or row["website"]


def _rescored_evidence(row: dict[str, Any], enrichment: BusinessEnrichment) -> Evidence:
    """Audience evidence for `audience_index`, carried forward across a rescore.

    `reviews`/`rating` are never re-fetched here -- they are Google's, bought once at
    discovery -- so they come from the `google_maps` enrichment row `_enrichment_targets`
    already read back. Without this, an enrichment pass would silently regress a business's
    audience index to "unknown" the moment it was rescored, purely because this pass does
    not itself ask Google anything.
    """
    google = row.get("google_evidence") or {}
    ads = enrichment.ads
    return Evidence(
        reviews=google.get("reviews"),
        rating=google.get("rating"),
        runs_ads=ads.runs_ads if ads is not None else None,
    )


def _translate_enrichment_signals(
    enrichment: BusinessEnrichment, row_ids: dict[str, int]
) -> tuple[set[str], dict[str, list[int]]]:
    """What `automations.ENRICHMENT_SIGNALS` this pass can honestly assert, and which
    `enrichments` row backs each one.

    This is deliberately a partial translation. `SiteGrade` (`enrichment/website.py`) grades
    one boolean each for `catalogue`, `enquiry` and `ordering` -- it cannot distinguish "no
    way to book an appointment" from "no way to submit a general enquiry" from "no
    structured intake of any kind". All three catalogue tokens for that family --
    "no booking link", "no enquiry form", "manual enquiry flow" -- are therefore asserted
    together from the single `enquiry` gap: they describe the same observed fact, and the
    niche gate in `automations.py` (appointment offers only reach appointment-driven niches,
    quotation offers only reach quote-driven ones) is what stops that from over-claiming,
    since no business's niche is gated by more than one of those families at once.

    Five more vocabulary entries are never asserted here at all: "high review count",
    "owner replies absent", "active social presence", "instagram-only catalogue" and
    "dm ordering". Nothing this codebase runs today measures review replies, follower
    counts, engagement or post content -- see `execute_enrichment`'s docstring for why that
    is a phase boundary, not an oversight.
    """
    signals: set[str] = set()
    evidence: dict[str, list[int]] = {}

    def claim(signal: str, row_id: int | None) -> None:
        if row_id is None:
            return
        signals.add(signal)
        evidence.setdefault(signal, []).append(row_id)

    website = enrichment.website
    web_id = row_ids.get(SOURCE_TINYFISH_WEB, row_ids.get(SOURCE_FIRECRAWL_WEB))
    if website is not None and web_id is not None:
        if website.verdict in (website_module.NO_SITE, website_module.SOCIAL_ONLY):
            claim("no landing page", web_id)
        grade = website.grade
        if grade is not None:
            if "ordering" in grade.gaps:
                claim("no online ordering", web_id)
            if "enquiry" in grade.gaps:
                claim("no booking link", web_id)
                claim("no enquiry form", web_id)
                claim("manual enquiry flow", web_id)
            if "catalogue" in grade.gaps and "enquiry" in grade.gaps:
                claim("no landing page", web_id)

    ads = enrichment.ads
    if ads is not None and ads.runs_ads:
        claim("runs meta ads", row_ids.get(SOURCE_ADS))

    return signals, evidence


def _automation_prompt(business_name: str, offer: AutomationOffer, signals: Sequence[str]) -> str:
    """The LLM prompt for expanding one automation offer's baseline pitch.

    `copy.py` has no automation-pitch prompt of its own -- it is scoped to the website
    pitch -- so this is written here, grounded exactly the way `copy.build_prompt` grounds
    the website one: name the business, the offer, the observed evidence and the baseline
    pitch, and repeat the project's non-negotiable rule verbatim.
    """
    return (
        "You are writing a short outreach message pitching one workflow automation to a "
        "local business, grounded only in evidence already observed about it.\n\n"
        f"Business: {business_name}\n"
        f"Automation offered: {offer.label}\n"
        f"Observed evidence: {', '.join(signals)}\n"
        f"Baseline pitch: {offer.pitch_line}\n\n"
        "Expand the baseline pitch into a warm, concise outreach message of three to four "
        "sentences. Do not invent reviews, followers, revenue, demand, or any fact not "
        "shown above.\n"
    )


def _generate_website_pitch(
    router: LLMRouter | None,
    lead: Lead,
    breakdown: Any,
    profile: Any,
) -> str:
    """The `website_pitch` body: `copy.build_fallback_outreach` verbatim without AI, or an
    LLM's expansion of `copy.build_prompt` with that same text as the guaranteed fallback.

    `build_fallback_outreach` -- not `build_fallback_summary` -- is the fallback, because it
    is the one function in `copy.py` actually shaped as a message to the business
    (`export/view.py` confirms this: `outreach_message` is what fills `website_pitch`).
    `build_fallback_summary` is read but not called here; nothing in this schema has a place
    to put a second, summary-shaped text yet.
    """
    profiles = [profile] if profile else None
    fallback = build_fallback_outreach(lead, breakdown, profiles)
    if router is None:
        return fallback
    prompt = build_prompt(lead, breakdown, profiles)
    return router.generate(prompt, fallback).text


def _generate_automation_pitch(
    router: LLMRouter | None,
    business_name: str,
    offer: AutomationOffer,
    signals: Sequence[str],
) -> str:
    """The `automation_pitch` body: the offer's own `pitch_line` without AI, or an LLM's
    expansion of it, grounded in exactly the signals that fired it."""
    fallback = offer.pitch_line
    if router is None:
        return fallback
    prompt = _automation_prompt(business_name, offer, signals)
    return router.generate(prompt, fallback).text


# --- FastAPI dependencies ----------------------------------------------------------------


def get_engine(request: Request) -> Engine:
    """The process's Engine, per request.

    Request-scoped by dependency and process-scoped by lifetime: the pool is the thing that
    must outlive a request, and a connection is borrowed from it per statement rather than
    held for the duration of one. A request that held a connection open across a slow
    provider call would exhaust an eight-connection pool with eight readers.
    """
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - the lifespan always builds one
        raise ApiProblem(503, "not_ready", "The service is still starting up.")
    return engine


def get_repository(request: Request) -> Repository:
    """The repository, for routes that write through it."""
    return get_engine(request).require_repository()
