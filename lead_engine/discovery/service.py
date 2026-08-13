"""One discovery pass: geography -> cells -> billed searches -> qualified, new businesses.

    resolve the scope        an unresolvable area ABORTS; there is no fallback point
    select cells             the cursor decides where each credit goes (cells.py)
    call the provider        injected, one call == one billed search, and the ONLY thing in
                             this system that spends a credit
    read the outcome         the provider already qualified the page; this accumulates it
    drop the sentinel        "Unnamed business" is a placeholder, not a lead
    dedupe_leads()           collapse the same place returned twice in one pass
    dedupe at the BOUNDARY   against businesses.place_id, so the caller only ever sees
                             businesses this system did not already hold
    persist                  through the injected repository
    update cell yield        measured, from what was actually gained

WHERE THE MONEY IS
------------------
Fifty searches, one-time, non-renewing. Three rules keep that allowance from evaporating,
and all three are structural rather than advisory:

1. **This service does not spend.** There is no `spend()` call anywhere below, and that is
   the point rather than an oversight. The client charges the ledger immediately before the
   HTTP request goes out, which is the only position from which no caller can bypass it --
   and a second caller that also charged would halve the allowance silently. That is not
   hypothetical: this module and the client were written in parallel and both spent, so one
   search cost two credits and the operator would have hit a wall at what looked like 25
   searches with nothing to explain it. `BudgetExhausted` still arrives here, raised out of
   the provider call, and is still a clean stop.

2. **One cell, one search.** `searches_spent` is incremented once per provider call, the
   provider is contractually forbidden to paginate internally (see `MapsProvider`), and the
   default policy plans exactly one cell per (area, niche). So the bill for a pass is the
   number of cells it searched, and that number is visible in the outcome.

3. **`scan_multiplier` never multiplies searches.** In the prototype it widened
   `candidate_limit`, which was handed to a provider that paginated to satisfy it -- so a
   multiplier of 3 turned one billed search into three. Here the widened limit is applied to
   candidates *already paid for* (how many results from a response are worth examining) and
   to cell PRIORITY (who gets the scarce credits first). It is never passed to the provider
   and never changes the cell count. `test_discovery.py` pins that a multiplier of 3 issues
   exactly as many searches as a multiplier of 1.

QUALIFICATION HAPPENS ONCE, AND NOT HERE
----------------------------------------
A `SearchOutcome` arrives already judged: `.leads` are the places `matches_niche` kept,
`.rejected_types` and `.name_gate_rejected` are why the rest failed. This module used to
re-run that judgement against the same profile to produce the same answer. Two
implementations of one rule do not disagree on the day they are written; they disagree the
first time either one is edited, and then the run summary and the sheet describe different
runs. What is left here is the part that is genuinely discovery's: accumulating those
verdicts across passes, and deciding where the next credit goes.

A run must be countable, so `DiscoveryService` refuses to start unless it has either a
`SearchBudget` or an explicit `max_searches`. A sweep of 22 Bangalore areas x 24 niches is
528 searches with nothing counting them; an accidental one of those is the whole allowance
ten times over, and "I forgot to pass a ceiling" is not a mistake this system can afford to
make quietly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from lead_engine.dedupe import dedupe_key as compute_dedupe_key
from lead_engine.dedupe import dedupe_leads
from lead_engine.geo.scope import GeoScope, ResolvedLocation
from lead_engine.models import Lead
from lead_engine.niches import NICHE_PROFILES, NicheProfile, resolve_niche_ids
from lead_engine.providers.budget import BudgetExhausted

# `UNNAMED` is what SearchAPI returns for a place whose name it could not read. It is a
# placeholder, not a business: it cannot be dialled, matched to an Instagram handle, or
# greeted in an outreach message, and it is dropped here before anything is counted --
# including before the niche tally, since a nameless place is evidence about the response
# rather than about the registry. Imported rather than re-typed, because a filter comparing
# against a copy of a literal is a filter that stops working the day the literal moves.
from lead_engine.providers.searchapi import UNNAMED as UNNAMED_SENTINEL
from lead_engine.providers.searchapi import SearchOutcome

from .cells import (
    BREADTH_FIRST,
    CellPolicy,
    CellUpdate,
    Clock,
    ConnectionFactory,
    SearchCell,
    SearchCellStore,
    advance,
    plan_cells,
    selection_key,
)
from .status import NicheObservation, NicheStatusStore

#: Handed to the provider as `limit`. The provider applies it client-side, to a page that has
#: already been paid for and cannot be bought any smaller, so a low number here throws away
#: leads this run has already been billed for. The interface has no None, so this is
#: "everything the page returned" written as an integer -- an order of magnitude above the
#: 100-120 results a single query point can produce at all. Discovery does its own capping
#: further down, against what it is allowed to KEEP.
WHOLE_PAGE = 1000

#: The prototype's widening, unchanged. A niche whose profile sets `scan_multiplier > 1`
#: (cloud_kitchen and manufacturer at 3, salon/spa/tutor_class at 2) is one where Google's
#: results are known to be mostly noise, so more of them must be examined to find the same
#: number of real businesses. Both bounds are the prototype's.
MIN_WIDENED_SCAN = 50
MAX_WIDENED_SCAN = 200

#: The metered provider, named here only for a reader. Nothing below passes it to a ledger:
#: this service does not spend, so it has no reason to know which allowance is being drawn
#: down. The client that charges the credit names its own provider.
PROVIDER_NAME = "searchapi"


def utc_now() -> datetime:
    return datetime.now(UTC)


# --- the seams ---------------------------------------------------------------------------


class MapsProvider(Protocol):
    """The maps vendor. `SearchApiClient` or `FixtureMapsProvider`, injected.

    ONE CALL IS ONE BILLED SEARCH, and the implementation is the thing that bills it. An
    implementation must never paginate internally to satisfy `limit`, because this caller's
    entire cost model is "searches issued == cells consumed", and a provider that quietly
    fetched three pages would make the ledger, the budget and the run summary all agree on a
    number that is three times too small.

    IT ALSO OWNS THE SPEND. The credit is charged inside `search_places`, immediately before
    the request leaves, and this service adds nothing to that. `BudgetExhausted` propagates
    out of the call untouched: it means the ledger refused *before* anything was billed, so
    it is a stop condition and not a failure.

    `page` is 1-based, matching `search_cells.next_page`. `query_variant` is passed as the
    cell's own query string, which is what makes the cell key and the search that was
    actually issued the same fact -- an int index would mean the cursor recorded one query
    and the credit bought another whenever the registry's list was reordered.

    `limit` truncates ONE already-paid-for response; it never asks the vendor for less.
    """

    def search_places(
        self,
        location: ResolvedLocation,
        profile: NicheProfile,
        limit: int = 20,
        page: int = 1,
        query_variant: int | str | None = None,
    ) -> SearchOutcome: ...


class LocationResolver(Protocol):
    """`lead_engine.geo` resolvers, structurally. Raises `location_not_found` on a miss."""

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation: ...


class BusinessStore(Protocol):
    """The tool boundary: what this system already holds, and how a new business lands.

    `known` is the reason a caller can trust that everything returned is new. Without it,
    the second run of an unchanged sweep hands back the same fifty businesses it handed back
    yesterday, and an operator works a sheet they have already worked.
    """

    def known(
        self, place_ids: Sequence[str], dedupe_keys: Sequence[str]
    ) -> tuple[set[str], set[str]]: ...

    def store(
        self,
        lead: Lead,
        *,
        niche_id: str,
        dedupe_key: str,
        city: str,
        country: str | None = None,
        state: str | None = None,
        search_area: str | None = None,
    ) -> Any: ...


class SearchLedger(Protocol):
    """`providers.budget.SearchBudget`, narrowed to what discovery is allowed to do with it.

    There is deliberately no `spend` in this protocol, and this service holds a ledger only
    as proof that a ceiling exists. The client spends; a second spender is how a 50-search
    allowance silently becomes 25.
    """

    def remaining_total(self, provider: str, *, key_fingerprint: str = "") -> int: ...


# --- results -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredBusiness:
    """A business this pass added. By construction, one nothing else had stored."""

    lead: Lead
    niche_id: str
    dedupe_key: str
    area: str | None = None
    business_id: UUID | None = None


@dataclass(frozen=True)
class NicheOutcome:
    """What one niche cost and what it bought."""

    niche_id: str
    searches: int = 0
    candidates: int = 0
    qualified: int = 0
    new_businesses: int = 0
    already_stored: int = 0
    state: str | None = None
    #: Google type slugs this niche refused, and how often. The taxonomy bucket.
    rejected_types: dict[str, int] = field(default_factory=dict)
    #: Places whose types qualified and whose NAME did not. Kept out of `rejected_types`
    #: because it points at `qualification_terms` and at nothing in the type lists: a niche
    #: starving here and a niche starving on its slugs need opposite repairs.
    name_gate_rejected: int = 0


#: Why a pass stopped. Only `complete` means every planned cell was searched.
COMPLETE = "complete"
BUDGET_EXHAUSTED = "budget_exhausted"
MAX_SEARCHES = "max_searches"


@dataclass(frozen=True)
class DiscoveryOutcome:
    businesses: tuple[DiscoveredBusiness, ...] = ()
    searches_spent: int = 0
    cells_searched: int = 0
    cells_planned: int = 0
    stopped: str = COMPLETE
    policy: str = BREADTH_FIRST.name
    niches: tuple[NicheOutcome, ...] = ()
    locations: tuple[ResolvedLocation, ...] = ()

    @property
    def starved_niches(self) -> tuple[str, ...]:
        """Niches whose type slugs the accumulated evidence says are wrong."""
        return tuple(n.niche_id for n in self.niches if n.state == "starved")


@dataclass(frozen=True)
class DiscoveryRequest:
    goal_id: UUID
    scope: GeoScope
    niche_ids: Sequence[str]
    #: Leads wanted across every niche. Split evenly, exactly as the prototype did.
    limit: int = 60
    #: THE money knob. `BREADTH_FIRST` unless a caller explicitly names something else.
    policy: CellPolicy = BREADTH_FIRST
    #: Hard ceiling on billed searches for this pass. Required when no budget is injected.
    max_searches: int | None = None


# --- the prototype's allocation, ported ---------------------------------------------------


def per_niche_limit(limit: int, niche_count: int) -> int:
    """How many leads each niche contributes to the sheet. `ceil`, as in the prototype."""
    if niche_count < 1:
        raise ValueError("at least one niche is required")
    return max(1, math.ceil(max(1, int(limit)) / niche_count))


def candidate_limit(profile: NicheProfile, per_niche: int) -> int:
    """How many CANDIDATES this niche is worth examining. Never how many searches to buy.

    Straight from `lead_finder/search.py`, with one change that is the whole point: there,
    this number was handed to a provider that paginated until it had that many results, so
    `scan_multiplier=3` billed three searches instead of one. Here the candidates are the
    ones a single already-paid-for response returned, and the multiplier's real job -- get
    this niche's credits allocated first -- is done by `selection_key`, not by this number.
    """
    if profile.scan_multiplier > 1:
        return min(MAX_WIDENED_SCAN, max(MIN_WIDENED_SCAN, per_niche * profile.scan_multiplier))
    return per_niche


def niche_priority(profiles: Iterable[NicheProfile]) -> dict[str, int]:
    """`scan_multiplier` as a cell-selection priority. Ordering only; never a count."""
    return {profile.id: max(1, int(profile.scan_multiplier)) for profile in profiles}


# --- the adapter over the real database ---------------------------------------------------


_KNOWN = """
SELECT place_id, dedupe_key
  FROM businesses
 WHERE (place_id IS NOT NULL AND place_id = ANY(%s))
    OR (dedupe_key IS NOT NULL AND dedupe_key = ANY(%s))
"""


class RepositoryBusinesses:
    """`BusinessStore` over the real `businesses` table.

    Writes go through `Repository.upsert_business` rather than SQL of its own: the upsert's
    coalescing rules (a thin later sighting must not blank out a fat earlier one) are not
    something this module should own a second copy of. The boundary read is its own SELECT
    because the repository has none, and it is a read.
    """

    def __init__(self, repository: Any, connect: ConnectionFactory) -> None:
        self._repository = repository
        self._connect = connect

    def known(
        self, place_ids: Sequence[str], dedupe_keys: Sequence[str]
    ) -> tuple[set[str], set[str]]:
        if not place_ids and not dedupe_keys:
            return set(), set()
        with self._connect() as conn:
            rows = conn.execute(_KNOWN, (list(place_ids), list(dedupe_keys))).fetchall()
            conn.commit()
        return (
            {row[0] for row in rows if row[0] is not None},
            {row[1] for row in rows if row[1] is not None},
        )

    def store(
        self,
        lead: Lead,
        *,
        niche_id: str,
        dedupe_key: str,
        city: str,
        country: str | None = None,
        state: str | None = None,
        search_area: str | None = None,
    ) -> Any:
        return self._repository.upsert_business(
            name=lead.name,
            niche_id=niche_id,
            city=lead.city or city,
            dedupe_key=dedupe_key,
            place_id=lead.provider_id,
            country=country,
            state=state,
            search_area=search_area,
            address=lead.address or None,
            lat=lead.latitude,
            lng=lead.longitude,
            phone=lead.phone,
            website=lead.website,
        )


# --- one unit of billable work -------------------------------------------------------------


@dataclass
class _Work:
    profile: NicheProfile
    location: ResolvedLocation
    cell: SearchCell
    area: str | None


class DiscoveryService:
    """One discovery pass. Every collaborator is injected; none is constructed here.

    That is not ceremony. The provider is the only object in this system that spends money,
    and a service that could build its own would be one import away from a test that bills
    the live account.

    `budget` is held and never spent -- see the module docstring. It is here so that a pass
    with a real ledger behind it does not also have to be handed a `max_searches`, not so
    that this class can charge anything.
    """

    def __init__(
        self,
        *,
        provider: MapsProvider,
        resolver: LocationResolver,
        businesses: BusinessStore,
        cells: SearchCellStore,
        niche_status: NicheStatusStore | None = None,
        budget: SearchLedger | None = None,
        clock: Clock = utc_now,
    ) -> None:
        self._provider = provider
        self._resolver = resolver
        self._businesses = businesses
        self._cells = cells
        self._niche_status = niche_status
        self._budget = budget
        self._clock = clock

    # -- public ---------------------------------------------------------------------------

    def discover(self, request: DiscoveryRequest) -> DiscoveryOutcome:
        if self._budget is None and request.max_searches is None:
            raise ValueError(
                "discover() needs a ceiling: inject a SearchBudget or set "
                "DiscoveryRequest.max_searches. The allowance is 50 lifetime searches and a "
                "pass with nothing counting them can spend all of it."
            )

        # Raises UnsupportedNicheError rather than substituting a near niche.
        niche_ids = resolve_niche_ids(list(request.niche_ids))
        profiles = [NICHE_PROFILES[niche_id] for niche_id in niche_ids]
        priority = niche_priority(profiles)
        per_niche = per_niche_limit(request.limit, len(profiles))
        scan_caps = {p.id: candidate_limit(p, per_niche) for p in profiles}

        now = self._clock()
        # An unresolvable area raises location_not_found and aborts the whole pass, before
        # a single credit is spent. Half a sweep is a sweep with a hole nothing can see.
        located = self._resolve(request.scope)

        self._cells.revive_expired(request.goal_id, now)
        work, planned = self._plan(request, profiles, located, priority)

        state = _PassState(per_niche=per_niche, scan_caps=scan_caps)
        stopped = COMPLETE
        for item in work:
            if state.finished_for(item.profile.id):
                continue
            if request.max_searches is not None and state.searches >= request.max_searches:
                stopped = MAX_SEARCHES
                break
            outcome = self._sweep(request, item, state)
            if outcome is not None:
                stopped = outcome
                break

        niches = self._finish(niche_ids, state, now)
        return DiscoveryOutcome(
            businesses=tuple(state.businesses),
            searches_spent=state.searches,
            cells_searched=state.cells_searched,
            cells_planned=planned,
            stopped=stopped,
            policy=request.policy.name,
            niches=niches,
            locations=tuple(location for _, location in located),
        )

    # -- geography ------------------------------------------------------------------------

    def _resolve(self, scope: GeoScope) -> list[tuple[str | None, ResolvedLocation]]:
        """(area name, point) per target. The area name is what `search_cells.area` and
        `businesses.search_area` record, and `resolve_scope` does not return it."""
        targets = scope.targets()
        areas: tuple[str | None, ...] = scope.areas or (None,) * len(targets)
        located: list[tuple[str | None, ResolvedLocation]] = []
        for (query, precision), area in zip(targets, areas, strict=True):
            location = self._resolver.resolve(query, precision)
            _verify_country(scope, location, query)
            located.append((area, location))
        return located

    # -- planning -------------------------------------------------------------------------

    def _plan(
        self,
        request: DiscoveryRequest,
        profiles: Sequence[NicheProfile],
        located: Sequence[tuple[str | None, ResolvedLocation]],
        priority: dict[str, int],
    ) -> tuple[list[_Work], int]:
        """Every cell this pass may search, in the order the credits should reach them."""
        work: list[_Work] = []
        planned = 0
        for area, location in located:
            for profile in profiles:
                specs = plan_cells(
                    request.policy,
                    goal_id=request.goal_id,
                    niche_id=profile.id,
                    queries=profile.queries,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    radius_meters=location.radius_meters,
                    area=area,
                )
                planned += len(specs)
                cells = self._cells.ensure(specs)
                # An exhausted cell is skipped, and skipping it is the entire return on this
                # table: it is ground that has twice returned nothing new, and re-buying it
                # would cost a credit to be told so a third time.
                live = [cell for cell in cells if not cell.exhausted]
                chosen = sorted(live, key=lambda cell: selection_key(cell, priority))
                for cell in chosen[: request.policy.cells_per_area_niche]:
                    work.append(_Work(profile=profile, location=location, cell=cell, area=area))

        work.sort(key=lambda item: selection_key(item.cell, priority))
        return work, planned

    # -- the billed part -------------------------------------------------------------------

    def _sweep(self, request: DiscoveryRequest, item: _Work, state: _PassState) -> str | None:
        """Search one cell, up to `pages_per_pass` times. Returns a stop reason, or None.

        Every page is a separate credit, a separate measurement and a separate cursor
        advance, so a policy that pages three times is honestly billed three times.
        """
        for _ in range(request.policy.pages_per_pass):
            if request.max_searches is not None and state.searches >= request.max_searches:
                return MAX_SEARCHES
            if state.finished_for(item.profile.id):
                return None

            try:
                # THE billed line. The credit is charged inside this call, immediately before
                # the request leaves and never refunded, and nothing here adds to it. The
                # cell's own query string goes back out as `query_variant`, so what the
                # cursor recorded and what the credit bought are the same string.
                outcome = self._provider.search_places(
                    item.location,
                    item.profile,
                    limit=WHOLE_PAGE,
                    page=item.cell.next_page,
                    query_variant=item.cell.spec.query_variant,
                )
            except BudgetExhausted:
                # A clean stop, not a failure. The ledger refuses before the request goes
                # out, so this call cost nothing and everything gathered so far stands.
                return BUDGET_EXHAUSTED

            state.searches += 1
            state.cells_searched += 1
            update = self._absorb(request, item, outcome, state)
            item.cell = self._cells.record(item.cell, update)
            if update.new_yield == 0:
                # Stop paying for this cell in this pass. A zero-yield page does not advance
                # the cursor -- an empty page is no evidence that the next one is fuller --
                # so the next turn of this loop would re-issue the SAME page, seconds later,
                # against a vendor that returns the same ranked results for the same query
                # point. That is a credit bought to be told what the last one just said. The
                # cell is not written off either: a single zero is routinely a transient, and
                # the second strike that exhausts it has to come from a later pass.
                return None
        return None

    def _absorb(
        self,
        request: DiscoveryRequest,
        item: _Work,
        outcome: SearchOutcome,
        state: _PassState,
    ) -> CellUpdate:
        """Accumulate, dedupe, persist, and measure what the credit actually bought.

        Nothing here re-qualifies anything: `outcome.leads` is what `matches_niche` already
        kept, and the counters beside it are why the rest went. See the module docstring.
        """
        profile = item.profile
        # The sentinel goes first: a nameless placeholder is not a business and must not be
        # counted as a qualified one.
        named = [lead for lead in outcome.leads if lead.name != UNNAMED_SENTINEL]
        observation = NicheObservation.from_outcome(
            outcome, dropped=len(outcome.leads) - len(named)
        )
        state.observe(observation)
        state.candidates[profile.id] = state.candidates.get(profile.id, 0) + observation.candidates

        # The leads arrive tagged with the niche by `build_outcome`; the merge below keeps
        # that tag when one pass returns the same place twice.
        unique = dedupe_leads(list(named))

        fresh = self._only_new(unique, profile.id, state)
        # ONE cap, on what this niche is allowed to KEEP. `scan_caps` is not a second one:
        # it bounds how many candidates are worth EXAMINING, which `finished_for` enforces,
        # and applying it here as well capped a scan_multiplier niche's leads at the number
        # of results it was meant to be reading MORE of -- the exact inversion of its point.
        room = max(0, state.per_niche - state.kept.get(profile.id, 0))
        fresh = fresh[:room]

        for lead in fresh:
            key = compute_dedupe_key(lead)
            row = self._businesses.store(
                lead,
                niche_id=profile.id,
                dedupe_key=key,
                city=request.scope.city,
                country=request.scope.country,
                state=request.scope.state,
                search_area=item.area,
            )
            state.remember(lead, key)
            state.businesses.append(
                DiscoveredBusiness(
                    lead=lead,
                    niche_id=profile.id,
                    dedupe_key=key,
                    area=item.area,
                    business_id=getattr(row, "id", None),
                )
            )
        state.kept[profile.id] = state.kept.get(profile.id, 0) + len(fresh)
        state.new_by_niche[profile.id] = state.new_by_niche.get(profile.id, 0) + len(fresh)
        state.searches_by_niche[profile.id] = state.searches_by_niche.get(profile.id, 0) + 1

        # THE measurement. `new_count` is businesses gained, not places returned: a cell that
        # hands back the same twenty places for the third time has yielded nothing, and the
        # cursor has to see that as a strike or it will keep buying them.
        return advance(
            item.cell,
            new_count=len(fresh),
            total_count=observation.candidates,
            now=self._clock(),
        )

    def _only_new(self, leads: Sequence[Lead], niche_id: str, state: _PassState) -> list[Lead]:
        """THE TOOL BOUNDARY. Drop everything this system already holds.

        Checked against `businesses.place_id` first, which is Google's own identity for a
        place and the strongest key available, and against `businesses.dedupe_key` as well,
        because a lead Google gave no place id to would otherwise look new forever.

        The in-run memory matters as much as the query: two overlapping tiles in one pass
        return the same restaurant, and the second tile must not be credited with having
        found it.
        """
        candidates = [lead for lead in leads if not state.seen(lead, compute_dedupe_key(lead))]
        if not candidates:
            return []
        place_ids = [lead.provider_id for lead in candidates if lead.provider_id]
        keys = [compute_dedupe_key(lead) for lead in candidates]
        stored_places, stored_keys = self._businesses.known(place_ids, keys)

        fresh: list[Lead] = []
        for lead in candidates:
            key = compute_dedupe_key(lead)
            # Attributed to the niche whose search paid for the response, not to
            # `lead.matched_niches[0]`: the niche is what the caller asked for and is always
            # known, while the tag is something the provider set and an empty one would raise
            # IndexError in the middle of a pass that had already spent its credits.
            already = lead.provider_id and lead.provider_id in stored_places
            if already or key in stored_keys:
                state.already_stored[niche_id] = state.already_stored.get(niche_id, 0) + 1
                continue
            fresh.append(lead)
        return fresh

    # -- reporting --------------------------------------------------------------------------

    def _finish(
        self, niche_ids: Sequence[str], state: _PassState, now: datetime
    ) -> tuple[NicheOutcome, ...]:
        outcomes: list[NicheOutcome] = []
        for niche_id in niche_ids:
            observation = state.observations.get(niche_id, NicheObservation(niche_id))
            status_state: str | None = None
            if self._niche_status is not None and niche_id in state.observations:
                # Only niches this pass actually looked at are recorded. Stamping
                # `last_seen_at` for a niche whose every cell was exhausted would claim a
                # live response that never happened.
                status_state = self._niche_status.record(observation, seen_at=now).state
            outcomes.append(
                NicheOutcome(
                    niche_id=niche_id,
                    searches=state.searches_by_niche.get(niche_id, 0),
                    candidates=state.candidates.get(niche_id, 0),
                    qualified=observation.qualified,
                    new_businesses=state.new_by_niche.get(niche_id, 0),
                    already_stored=state.already_stored.get(niche_id, 0),
                    state=status_state,
                    rejected_types=dict(observation.rejected_types),
                    name_gate_rejected=observation.name_gate_rejected,
                )
            )
        return tuple(outcomes)


@dataclass
class _PassState:
    """Everything one pass accumulates. Mutable and private on purpose."""

    per_niche: int
    scan_caps: dict[str, int]
    searches: int = 0
    cells_searched: int = 0
    businesses: list[DiscoveredBusiness] = field(default_factory=list)
    observations: dict[str, NicheObservation] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)
    new_by_niche: dict[str, int] = field(default_factory=dict)
    already_stored: dict[str, int] = field(default_factory=dict)
    searches_by_niche: dict[str, int] = field(default_factory=dict)
    _seen_places: set[str] = field(default_factory=set)
    _seen_keys: set[str] = field(default_factory=set)

    def observe(self, observation: NicheObservation) -> None:
        existing = self.observations.get(observation.niche_id)
        self.observations[observation.niche_id] = (
            observation if existing is None else existing.merge(observation)
        )

    def remember(self, lead: Lead, key: str) -> None:
        if lead.provider_id:
            self._seen_places.add(lead.provider_id)
        self._seen_keys.add(key)

    def seen(self, lead: Lead, key: str) -> bool:
        if lead.provider_id and lead.provider_id in self._seen_places:
            return True
        return key in self._seen_keys

    def finished_for(self, niche_id: str) -> bool:
        """This niche has all the leads it was allocated, or has seen all it was worth.

        Both ceilings end the niche's spending early, which is the point: a credit spent on
        a niche whose quota is full buys a lead the sheet will not carry.
        """
        if self.kept.get(niche_id, 0) >= self.per_niche:
            return True
        return self.candidates.get(niche_id, 0) >= self.scan_caps.get(niche_id, 0)


def _verify_country(scope: GeoScope, location: ResolvedLocation, query: str) -> None:
    """A resolution in the wrong country is a miss, not a result.

    `geo.resolver.resolve_scope` does this and is not reusable here -- it returns points
    without the area names this module has to record -- so the check is repeated rather than
    dropped. There is one Jaipur in Rajasthan and another in Texas.
    """
    if scope.gl and location.gl and scope.gl != location.gl:
        from lead_engine.providers.errors import ProviderError

        raise ProviderError(
            "location_not_found",
            f"{query!r} resolved to country {location.gl!r}, but the scope declares "
            f"{scope.gl!r}. Refusing to search the wrong country.",
        )
