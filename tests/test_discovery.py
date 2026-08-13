"""One discovery pass: what it costs, and what it is allowed to hand back.

THE TEST THAT MATTERS MOST IS `test_the_budget_is_charged_once_per_search_by_the_provider`.

`discovery/service.py` and `providers/searchapi.py` were written in parallel by people who
could not see each other's work, and both spent the budget: the client immediately before the
HTTP request, the service immediately before calling the client. Each was correct alone.
Wired together, one search cost two credits, a 50-search lifetime allowance silently became
25, and the only symptom would have been a worker stopping half way through a sweep with
nothing anywhere to explain why. Nothing caught it because nothing had wired them together
yet, and once the credits are gone they do not come back.

So the ledger here is a recording fake, the fake provider spends it exactly as the live
client does, and the count is asserted as a number rather than as "not too many". If the
service ever grows a `spend()` again, `test_a_service_holding_the_ledger_spends_nothing`
fails on the next run: it hands the service a ledger and the provider none, so any charge at
all is the service's.

NOTHING HERE TOUCHES THE NETWORK. `FakeMapsProvider` serves canned SearchAPI bodies through
`build_outcome` -- the live client's own parser and qualification gate -- so what these tests
qualify is what a real response would qualify, and a change to either would show up here
rather than after the credits were spent.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from lead_engine.discovery.cells import (
    BREADTH_FIRST,
    DEEP_SWEEP,
    EXHAUSTED,
    EXHAUSTION_TTL_DAYS,
    PENDING,
    STRUCK,
    CellPolicy,
    CellSpec,
    SearchCell,
    apply,
    is_revivable,
    revive,
)
from lead_engine.discovery.service import (
    BUDGET_EXHAUSTED,
    COMPLETE,
    MAX_SEARCHES,
    UNNAMED_SENTINEL,
    DiscoveryRequest,
    DiscoveryService,
    candidate_limit,
    niche_priority,
    per_niche_limit,
)
from lead_engine.discovery.status import (
    NAME_GATE_KEY,
    STARVATION_THRESHOLD,
    STARVED,
    UNVERIFIED,
    VERIFIED,
    NicheObservation,
    NicheStatus,
    NicheStatusStore,
    derive_state,
    split_buckets,
)
from lead_engine.geo.scope import GeoScope, ResolvedLocation, radius_for
from lead_engine.niches import NICHE_PROFILES, UnsupportedNicheError
from lead_engine.providers.budget import BudgetExhausted
from lead_engine.providers.errors import ProviderError
from lead_engine.providers.searchapi import PROVIDER, build_outcome, resolve_query

try:
    import psycopg

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

NOW = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
GOAL = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

#: The allowance actually in hand. Fifty, once, ever.
ALLOWANCE = 50


# --- canned SearchAPI bodies ---------------------------------------------------------------


def place(place_id: str, title: str, *types: str, phone: str | None = "080 4111 2222") -> dict:
    """One `local_results[]` entry, in SearchAPI's shape.

    `types` are Google's DISPLAY labels ("Hair salon"), because that is what the vendor
    returns and what `niches.slugify_type` has to turn back into a `type_id`. Writing slugs
    here instead would test the registry against a spelling Google never sends.
    """
    return {
        "place_id": place_id,
        "title": title,
        "type": types[0] if types else "",
        "types": list(types),
        "address": f"{title}, 100 Feet Road, Indiranagar, Bengaluru",
        "gps_coordinates": {"latitude": 12.9784, "longitude": 77.6408},
        "phone": phone,
        "rating": 4.4,
        "reviews": 210,
    }


def body(places: list[dict]) -> dict:
    return {
        "search_metadata": {"id": "search_1", "status": "Success"},
        "search_parameters": {"engine": "google_maps"},
        "local_results": places,
    }


def cafes(count: int = 6, prefix: str = "cafe") -> list[dict]:
    """Places that genuinely pass `matches_niche` for the (non-strict) cafe profile."""
    return [
        place(f"{prefix}-{i}", f"Third Wave Coffee {i}", "Cafe", "Coffee shop")
        for i in range(count)
    ]


def hardware(count: int) -> list[dict]:
    """Real businesses that are not cafes. Neutral types: the missing-slug shape."""
    return [place(f"hw-{i}", f"Sri Ganesh Hardware {i}", "Hardware store") for i in range(count)]


# --- the fakes -------------------------------------------------------------------------------


class RecordingLedger:
    """`SearchBudget`, in memory, remembering every charge.

    The whole point is `spends`. A count of one per search is the property under test; a
    count of two per search is the defect this file exists for.
    """

    def __init__(self, allowance: int = ALLOWANCE) -> None:
        self.allowance = allowance
        self.used = 0
        self.spends: list[tuple[str, int]] = []

    def spend(self, provider: str, n: int = 1, *, key_fingerprint: str = "") -> int:
        if self.used + n > self.allowance:
            raise BudgetExhausted(provider, self.allowance, self.used, n)
        self.used += n
        self.spends.append((provider, n))
        return self.used

    def remaining_total(self, provider: str, *, key_fingerprint: str = "") -> int:
        return self.allowance - self.used


class FakeMapsProvider:
    """The maps vendor's contract over canned bodies. Never a socket, always a credit.

    Two things make it a fair stand-in for `SearchApiClient`. It qualifies with the client's
    own `build_outcome`, so what these tests keep is what a live response would keep; and it
    charges the ledger before it answers, in the same position and with the same "spend
    first, never refund" rule. One call is one page and one credit -- it cannot paginate.
    """

    api_key = "fake"

    def __init__(self, pages: dict, *, ledger: RecordingLedger | None = None) -> None:
        # niche id -> a list of places (every page returns it, exactly as Google returns the
        # same ranked twenty for the same point) or {page: places} for a scripted sweep.
        self.pages = dict(pages)
        self.ledger = ledger
        self.calls: list[dict] = []
        self.outcomes: list = []

    def search_places(self, location, profile, limit=20, page=1, query_variant=None):
        query = resolve_query(profile, query_variant)
        if self.ledger is not None:
            # Exactly where the live client charges: before the request, never refunded.
            self.ledger.spend(PROVIDER, 1)
        self.calls.append(
            {
                "niche_id": profile.id,
                "page": page,
                "query": query,
                "limit": limit,
                "label": location.label,
                "radius": location.radius_meters,
            }
        )
        outcome = build_outcome(
            body(self._places(profile.id, page)),
            profile,
            location,
            limit=limit,
            page=page,
            query=query,
        )
        self.outcomes.append(outcome)
        return outcome

    def _places(self, niche_id: str, page: int) -> list[dict]:
        entry = self.pages.get(niche_id, [])
        return list(entry.get(page, [])) if isinstance(entry, dict) else list(entry)


class MemoryCells:
    """`SearchCellStore` in a dict. The SQL twin of every rule here is in test_cells.py."""

    def __init__(self) -> None:
        self.rows: dict[tuple, SearchCell] = {}
        self.revived: list[SearchCell] = []
        self._next_id = 1

    def ensure(self, specs):
        cells = []
        for spec in specs:
            existing = self.rows.get(spec.key)
            if existing is None:
                existing = SearchCell(spec=spec, id=self._next_id)
                self._next_id += 1
                self.rows[spec.key] = existing
            cells.append(existing)
        return cells

    def record(self, cell, update):
        updated = apply(cell, update)
        self.rows[cell.spec.key] = updated
        return updated

    def revive_expired(self, goal_id, now):
        revived = []
        for key, cell in list(self.rows.items()):
            if cell.spec.goal_id == goal_id and is_revivable(cell, now):
                self.rows[key] = revive(cell)
                revived.append(self.rows[key])
        self.revived.extend(revived)
        return revived

    def only(self) -> SearchCell:
        assert len(self.rows) == 1, f"expected one cell, have {len(self.rows)}"
        return next(iter(self.rows.values()))


class MemoryBusinesses:
    """The `businesses` table at the tool boundary: what is already held, and what lands."""

    def __init__(self) -> None:
        self.place_ids: set[str] = set()
        self.dedupe_keys: set[str] = set()
        self.stored: list[dict] = []

    def known(self, place_ids, dedupe_keys):
        return (
            {p for p in place_ids if p in self.place_ids},
            {k for k in dedupe_keys if k in self.dedupe_keys},
        )

    def store(self, lead, *, niche_id, dedupe_key, city, country=None, state=None,
              search_area=None):
        if lead.provider_id:
            self.place_ids.add(lead.provider_id)
        self.dedupe_keys.add(dedupe_key)
        self.stored.append(
            {
                "name": lead.name,
                "niche_id": niche_id,
                "dedupe_key": dedupe_key,
                "city": city,
                "search_area": search_area,
            }
        )
        return SimpleNamespace(id=uuid.uuid4())


class MemoryNicheStatus:
    """`niche_status`, accumulated in a dict and derived exactly as the SQL derives it.

    Goes through `stored_counts()` and `split_buckets()` rather than keeping the two buckets
    as separate attributes, so the encoding the jsonb column actually holds is exercised on
    every write.
    """

    def __init__(self) -> None:
        self.totals: dict[str, NicheObservation] = {}
        self.last_seen: dict[str, datetime] = {}

    def record(self, observation, *, seen_at):
        current = self.totals.get(observation.niche_id)
        merged = observation if current is None else current.merge(observation)
        self.totals[observation.niche_id] = merged
        self.last_seen[observation.niche_id] = seen_at
        return self._status(merged, seen_at)

    def get(self, niche_id: str) -> NicheStatus | None:
        observation = self.totals.get(niche_id)
        if observation is None:
            return None
        return self._status(observation, self.last_seen[niche_id])

    @staticmethod
    def _status(observation: NicheObservation, seen_at: datetime) -> NicheStatus:
        slugs, name_gate = split_buckets(observation.stored_counts())
        return NicheStatus(
            niche_id=observation.niche_id,
            state=derive_state(observation.qualified, observation.rejected),
            qualified=observation.qualified,
            rejected=observation.rejected,
            rejected_types=slugs,
            last_seen_at=seen_at,
            name_gate_rejected=name_gate,
        )


class StubResolver:
    """A geocoder that always answers, unless it is told to fail."""

    def __init__(self, *, gl: str | None = "in", error: Exception | None = None) -> None:
        self.gl = gl
        self.error = error
        self.queries: list[tuple[str, str]] = []
        self.points: dict[str, float] = {}

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        self.queries.append((query, precision))
        if self.error is not None:
            raise self.error
        # A distinct point per area, so two areas are two cells rather than one bought twice
        # -- and the SAME point for the same area every time, which is what makes tomorrow's
        # run recognise today's cell instead of opening a second one beside it. The real
        # resolver gets that from `geo_cache`; a stub that drifted would hide every
        # cursor bug in this file.
        if query not in self.points:
            self.points[query] = round(12.97 + len(self.points) / 100, 6)
        return ResolvedLocation(
            label=query,
            latitude=self.points[query],
            longitude=77.64,
            radius_meters=radius_for(precision),
            precision=precision,
            gl=self.gl,
            source_query=query,
        )


def build(pages: dict, *, ledger=None, provider_ledger="same", **kwargs):
    """A service wired to fakes, plus the fakes, so a test can assert on any of them."""
    ledger = RecordingLedger() if ledger is None else ledger
    provider = FakeMapsProvider(
        pages, ledger=ledger if provider_ledger == "same" else provider_ledger
    )
    cells = kwargs.pop("cells", None) or MemoryCells()
    businesses = kwargs.pop("businesses", None) or MemoryBusinesses()
    status = kwargs.pop("niche_status", None) or MemoryNicheStatus()
    resolver = kwargs.pop("resolver", None) or StubResolver()
    service = DiscoveryService(
        provider=provider,
        resolver=resolver,
        businesses=businesses,
        cells=cells,
        niche_status=status,
        budget=ledger,
        clock=kwargs.pop("clock", lambda: NOW),
    )
    return SimpleNamespace(
        service=service,
        provider=provider,
        ledger=ledger,
        cells=cells,
        businesses=businesses,
        status=status,
        resolver=resolver,
    )


def request(*, niches=("cafe",), areas=("Indiranagar",), **kwargs) -> DiscoveryRequest:
    return DiscoveryRequest(
        goal_id=GOAL,
        scope=GeoScope(city="Bangalore", state="Karnataka", country="India", areas=areas),
        niche_ids=list(niches),
        limit=kwargs.pop("limit", 60),
        **kwargs,
    )


# ============================================================================================
# THE MONEY
# ============================================================================================


def test_one_area_and_one_niche_is_exactly_one_billed_search():
    """The bluntest test in the suite, and the one to read first.

    One neighbourhood, one niche, the default policy: ONE search. Not "about one", not
    "at most a few". The allowance is fifty for the lifetime of the key, so a pass that
    quietly issued two would halve every sweep this system will ever run.
    """
    rig = build({"cafe": cafes(6)})

    outcome = rig.service.discover(request())

    assert len(rig.provider.calls) == 1
    assert outcome.searches_spent == 1
    assert outcome.cells_searched == 1
    assert outcome.cells_planned == 1
    assert outcome.stopped == COMPLETE
    # Cells searched IS the bill. If these three ever disagree, one of them is lying about
    # what was bought.
    assert outcome.searches_spent == outcome.cells_searched == len(rig.provider.calls)


def test_the_budget_is_charged_once_per_search_by_the_provider():
    """THE regression test for the double spend.

    The provider charges the ledger, exactly as the live client does. The service is handed
    the same ledger. One search must leave one charge on it; a service that spent as well
    would leave two, and fifty credits would run out at twenty-five searches.
    """
    rig = build({"cafe": cafes(6)})

    outcome = rig.service.discover(request())

    assert rig.ledger.spends == [(PROVIDER, 1)]
    assert rig.ledger.used == 1
    assert rig.ledger.used == outcome.searches_spent == len(rig.provider.calls)


def test_a_service_holding_the_ledger_spends_nothing():
    """The same defect, caught from the other side and without the provider's help.

    The service gets the ledger; the provider does not. Every charge on it is therefore the
    service's own, and there must be none: spending belongs to the object that sits closest
    to the billed request, because that is the only place a caller cannot bypass.
    """
    rig = build({"cafe": cafes(6)}, provider_ledger=None)

    outcome = rig.service.discover(request())

    assert rig.ledger.spends == []
    assert rig.ledger.used == 0
    assert outcome.searches_spent == 1  # the search still happened; nothing charged it here


def test_every_cell_is_charged_exactly_once_across_a_wider_sweep():
    rig = build({"cafe": cafes(6), "bakery": []})

    outcome = rig.service.discover(
        request(niches=("cafe", "bakery"), areas=("Indiranagar", "Koramangala", "Jayanagar"))
    )

    # 3 areas x 2 niches, one credit each. No tiling, no variants, no second page.
    assert len(rig.provider.calls) == 6
    assert outcome.searches_spent == 6
    assert rig.ledger.used == 6
    assert len(rig.ledger.spends) == 6


def test_scan_multiplier_does_not_buy_a_single_extra_search():
    """`scan_multiplier` is a priority and a reading width. It is never a search count.

    In the prototype it widened the candidate limit, which was handed to a provider that
    paginated to satisfy it, so a multiplier of 3 turned one billed search into three.
    """
    plain = build({"cafe": cafes(6)})
    plain.service.discover(request(niches=("cafe",)))

    # cloud_kitchen carries scan_multiplier=3, salon carries 2.
    assert NICHE_PROFILES["cloud_kitchen"].scan_multiplier == 3
    assert NICHE_PROFILES["salon"].scan_multiplier == 2
    noisy = build({"cloud_kitchen": [], "salon": []})
    noisy.service.discover(request(niches=("cloud_kitchen", "salon")))

    assert len(plain.provider.calls) == 1
    assert len(noisy.provider.calls) == 2  # two niches, one credit each. Not 3 + 2.
    assert noisy.ledger.used == 2


def test_the_default_policy_asks_for_page_one_of_the_first_query_and_stops():
    """Breadth-first: no tiling, no variant rotation, no pagination."""
    rig = build({"cafe": cafes(20)})

    outcome = rig.service.discover(request())

    call = rig.provider.calls[0]
    assert call["page"] == 1
    assert call["query"] == NICHE_PROFILES["cafe"].queries[0]
    # The resolved area radius, not a generated tile's.
    assert call["radius"] == radius_for("area")
    assert outcome.policy == BREADTH_FIRST.name
    assert len(rig.provider.calls) == 1


def test_no_area_ever_gets_a_second_query_variant():
    # The registry lists five queries per niche. Rotating them is five credits per niche per
    # neighbourhood instead of one, and with fifty in total that is a decision, not a default.
    rig = build({"cafe": cafes(6)})

    rig.service.discover(request(areas=("Indiranagar", "Koramangala", "Jayanagar")))

    assert {call["query"] for call in rig.provider.calls} == {NICHE_PROFILES["cafe"].queries[0]}
    assert {call["page"] for call in rig.provider.calls} == {1}
    assert len(rig.provider.calls) == 3


def test_a_full_page_does_not_tempt_the_pass_into_buying_the_next_one():
    # `SearchOutcome.has_more` is a hint and deliberately not acted on: page 2 is another
    # billed search, and nothing here is allowed to buy one on a hunch.
    rig = build({"cafe": cafes(20)})
    rig.service.discover(request())
    assert [call["page"] for call in rig.provider.calls] == [1]


def test_the_deep_sweep_is_the_only_way_to_spend_more_and_it_says_so():
    rig = build({"cafe": cafes(6)})

    outcome = rig.service.discover(request(policy=DEEP_SWEEP, max_searches=4))

    # 13 tiles x 5 variants of ground planned, and the ceiling is what stops it. The default
    # policy would have planned one cell in total.
    assert outcome.cells_planned == DEEP_SWEEP.cells_per_area_niche == 65
    assert outcome.searches_spent == 4
    assert outcome.stopped == MAX_SEARCHES


def test_a_pass_with_no_ceiling_refuses_to_start():
    rig = build({"cafe": cafes(6)})
    service = DiscoveryService(
        provider=rig.provider,
        resolver=rig.resolver,
        businesses=rig.businesses,
        cells=rig.cells,
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="ceiling"):
        service.discover(request())
    assert rig.provider.calls == []


def test_max_searches_stops_the_pass_at_the_number_it_names():
    rig = build({"cafe": cafes(6)})

    outcome = rig.service.discover(
        request(areas=("Indiranagar", "Koramangala", "Jayanagar"), max_searches=2)
    )

    assert len(rig.provider.calls) == 2
    assert outcome.stopped == MAX_SEARCHES
    assert outcome.cells_planned == 3  # planned three, bought two


# --- running out ------------------------------------------------------------------------------


def test_budget_exhausted_stops_the_pass_cleanly_and_keeps_what_it_gathered():
    """Out of credits is a stop condition, not a failure.

    The ledger refuses before the request leaves, so the refused call cost nothing, and
    everything the earlier searches bought stands. Turning this into an exception -- or into
    an empty result -- would report "no cafes in Koramangala" when the truth is "we stopped
    buying", and an operator would act on the difference.
    """
    ledger = RecordingLedger(allowance=1)
    rig = build({"cafe": cafes(6)}, ledger=ledger)

    outcome = rig.service.discover(
        request(areas=("Indiranagar", "Koramangala", "Jayanagar"))
    )

    assert outcome.stopped == BUDGET_EXHAUSTED
    assert len(outcome.businesses) == 6  # the first area's leads survive the stop
    assert outcome.searches_spent == 1
    # The refused call issued no search and charged nothing, and nothing was tried after it.
    assert len(rig.provider.calls) == 1
    assert ledger.used == 1


def test_an_exhausted_budget_reports_the_niches_it_did_reach():
    ledger = RecordingLedger(allowance=1)
    rig = build({"cafe": cafes(6), "bakery": cafes(6)}, ledger=ledger)

    outcome = rig.service.discover(request(niches=("cafe", "bakery")))

    reached = {n.niche_id: n.searches for n in outcome.niches}
    assert sum(reached.values()) == 1
    assert set(reached) == {"cafe", "bakery"}  # both reported, one searched


# ============================================================================================
# THE TOOL BOUNDARY
# ============================================================================================


def test_a_second_pass_over_the_same_responses_returns_no_new_businesses():
    """What the boundary dedupe is for.

    Google returns the same ranked twenty for the same point every time, so a sweep with no
    memory hands the operator the same sheet it handed them yesterday. The second pass here
    sees byte-identical responses and must return NOTHING -- and must say so as
    `already_stored`, not as an empty page.
    """
    rig = build({"cafe": cafes(6)})

    first = rig.service.discover(request())
    second = rig.service.discover(request())

    assert len(first.businesses) == 6
    assert second.businesses == ()
    assert second.niches[0].new_businesses == 0
    assert second.niches[0].already_stored == 6
    assert len(rig.businesses.stored) == 6  # nothing was written twice
    # It still cost a credit to learn that, which is exactly why the cell now gets a strike.
    assert second.searches_spent == 1
    assert rig.cells.only().status == STRUCK


def test_the_same_place_returned_twice_in_one_response_is_one_business():
    duplicated = cafes(2) + [place("cafe-0", "Third Wave Coffee 0", "Cafe", "Coffee shop")]
    rig = build({"cafe": duplicated})

    outcome = rig.service.discover(request())

    assert len(outcome.businesses) == 2
    assert len({b.dedupe_key for b in outcome.businesses}) == 2


def test_a_place_this_system_already_holds_is_counted_not_returned():
    rig = build({"cafe": cafes(3)})
    rig.businesses.place_ids.add("cafe-1")

    outcome = rig.service.discover(request())

    assert [b.lead.provider_id for b in outcome.businesses] == ["cafe-0", "cafe-2"]
    assert outcome.niches[0].already_stored == 1
    assert outcome.niches[0].new_businesses == 2


def test_the_unnamed_sentinel_never_becomes_a_business():
    """A place SearchAPI could not read a name for is a placeholder, not a lead: it cannot
    be dialled, matched to a handle, or greeted in an outreach message."""
    nameless = place("cafe-nameless", "", "Cafe", "Coffee shop")
    assert nameless["title"] == ""
    rig = build({"cafe": [*cafes(2), nameless]})

    outcome = rig.service.discover(request())

    assert len(outcome.businesses) == 2
    assert all(b.lead.name != UNNAMED_SENTINEL for b in outcome.businesses)
    # And it is not counted as a qualified one either, in the tally that judges the registry.
    assert outcome.niches[0].qualified == 2


def test_the_niche_and_the_area_are_stamped_on_what_is_stored():
    rig = build({"cafe": cafes(1)})

    outcome = rig.service.discover(request(areas=("Koramangala",)))

    assert rig.businesses.stored[0]["niche_id"] == "cafe"
    assert rig.businesses.stored[0]["search_area"] == "Koramangala"
    assert rig.businesses.stored[0]["city"] == "Bangalore"
    assert outcome.businesses[0].area == "Koramangala"


def test_a_niche_stops_buying_once_its_share_of_the_sheet_is_full():
    # limit=4 over two niches is two leads each; a credit spent past that buys a row the
    # sheet will not carry.
    rig = build({"cafe": cafes(6), "bakery": []})

    outcome = rig.service.discover(request(niches=("cafe", "bakery"), limit=4))

    cafe = next(n for n in outcome.niches if n.niche_id == "cafe")
    assert cafe.new_businesses == 2
    assert per_niche_limit(4, 2) == 2


# ============================================================================================
# THE CURSOR, THROUGH THE SERVICE
# ============================================================================================


def test_a_productive_pass_advances_the_cursor_so_tomorrow_reads_page_two():
    rig = build({"cafe": cafes(6)})

    rig.service.discover(request())

    cell = rig.cells.only()
    assert cell.next_page == 2
    assert cell.new_yield == 6

    rig.service.discover(request())
    assert rig.provider.calls[-1]["page"] == 2


def test_two_zero_yield_passes_exhaust_the_cell():
    rig = build({"cafe": []})

    first = rig.service.discover(request())
    assert rig.cells.only().status == STRUCK
    assert rig.cells.only().next_page == 1  # an empty page is no reason to buy the next one

    second = rig.service.discover(request())
    assert rig.cells.only().status == EXHAUSTED
    assert (first.searches_spent, second.searches_spent) == (1, 1)


def test_an_exhausted_cell_is_never_searched_again():
    """The entire return on the search_cells table: ground that has twice returned nothing
    new is not bought a third time to be told so again."""
    rig = build({"cafe": []})
    rig.service.discover(request())
    rig.service.discover(request())
    assert rig.cells.only().exhausted

    outcome = rig.service.discover(request())

    assert outcome.searches_spent == 0
    assert len(rig.provider.calls) == 2  # unchanged by the third pass
    assert outcome.cells_planned == 1


def test_thirty_days_later_the_cell_is_swept_again_from_page_one():
    """A permanently exhausted cell is a permanent blind spot, and it grows: re-sweeping
    page 1 is the only path by which a business that opened last month ever arrives."""
    rig = build({"cafe": []})
    rig.service.discover(request())
    rig.service.discover(request())
    assert rig.cells.only().status == EXHAUSTED

    later = NOW + timedelta(days=EXHAUSTION_TTL_DAYS + 1)
    revived = build(
        {"cafe": cafes(6)},
        cells=rig.cells,
        businesses=rig.businesses,
        clock=lambda: later,
    )
    outcome = revived.service.discover(request())

    assert [c.spec.query_variant for c in revived.cells.revived] == ["cafe"]
    assert revived.cells.revived[0].status == PENDING
    assert revived.provider.calls[0]["page"] == 1
    assert outcome.searches_spent == 1
    assert len(outcome.businesses) == 6


def test_an_exhausted_cell_that_has_not_served_its_time_is_left_alone():
    rig = build({"cafe": []})
    rig.service.discover(request())
    rig.service.discover(request())

    soon = NOW + timedelta(days=EXHAUSTION_TTL_DAYS - 1)
    later = build({"cafe": cafes(6)}, cells=rig.cells, clock=lambda: soon)
    outcome = later.service.discover(request())

    assert later.cells.revived == []
    assert outcome.searches_spent == 0


# ============================================================================================
# THE NICHE REGISTRY SIGNAL
# ============================================================================================


def test_a_niche_that_qualifies_nothing_out_of_twenty_is_starved():
    """The loud signal that replaces silent emptiness.

    None of the 259 type slugs in `niches.py` has been seen in a live response. A single
    wrong one produces a niche that returns twenty real places, qualifies none of them, and
    reports a successful run that found nothing -- indistinguishable from a neighbourhood
    with no cafes in it, and it costs a credit every time.
    """
    rig = build({"cafe": hardware(STARVATION_THRESHOLD)})

    outcome = rig.service.discover(request())

    assert outcome.niches[0].state == STARVED
    assert outcome.starved_niches == ("cafe",)
    assert outcome.niches[0].qualified == 0
    assert outcome.niches[0].candidates == STARVATION_THRESHOLD
    # And it names the slug to add, which is the only correction data this system ever gets.
    assert outcome.niches[0].rejected_types == {"hardware_store": STARVATION_THRESHOLD}


def test_too_few_candidates_is_not_yet_evidence_against_the_registry():
    rig = build({"cafe": hardware(STARVATION_THRESHOLD - 1)})
    outcome = rig.service.discover(request())
    assert outcome.niches[0].state == UNVERIFIED
    assert outcome.starved_niches == ()


def test_one_qualified_place_verifies_the_vocabulary_for_good():
    rig = build({"cafe": [*cafes(1), *hardware(STARVATION_THRESHOLD)]})
    outcome = rig.service.discover(request())
    assert outcome.niches[0].state == VERIFIED


def test_starvation_accumulates_across_areas_and_runs_that_are_each_inconclusive():
    """A niche failing quietly in four places is the same finding as one failing badly in
    one, and no single search can see it. Five rejects is nothing; twenty is a bug report."""
    rig = build({"cafe": hardware(5)})
    areas = ("Indiranagar", "Koramangala")

    first = rig.service.discover(request(areas=areas))
    second = rig.service.discover(request(areas=areas))

    assert first.niches[0].candidates == 10
    assert first.niches[0].state == UNVERIFIED
    assert second.niches[0].state == STARVED
    assert rig.status.get("cafe").rejected == 20
    assert rig.status.get("cafe").starved is True


def test_rejected_types_accumulate_across_runs():
    rig = build({"cafe": hardware(3)})
    rig.service.discover(request())
    rig.service.discover(request())
    assert rig.status.get("cafe").rejected_types == {"hardware_store": 6}


def test_the_name_gate_is_counted_apart_from_the_type_slugs():
    """A niche starving on `qualification_terms` and a niche starving on `include_types`
    need opposite repairs, and one merged number cannot tell them apart.

    `salon` is strict: a threading service whose name carries no salon evidence passes the
    type gate and fails the name gate. Nothing is wrong with `threading_service`, so
    recording it as a rejected type would send the maintainer to the wrong field entirely.

    The type label has to be one that carries no qualification term itself: `has_name_evidence`
    reads the category alongside the name, so a place typed "Hair salon" corroborates itself
    and can never reach the name gate at all.
    """
    rig = build(
        {
            "salon": [
                place("s-1", "Sri Lakshmi Enterprises", "Threading service"),  # name gate
                place("s-2", "Ink Age Studio", "Tattoo shop"),  # type gate
            ]
        }
    )

    outcome = rig.service.discover(request(niches=("salon",)))

    niche = outcome.niches[0]
    assert niche.rejected_types == {"tattoo_shop": 1}
    assert niche.name_gate_rejected == 1
    assert niche.qualified == 0

    # And the two survive storage in separate buckets rather than being merged on the way in.
    stored = rig.status.get("salon")
    assert stored.rejected_types == {"tattoo_shop": 1}
    assert stored.name_gate_rejected == 1


def test_a_qualifying_salon_still_gets_through_the_strict_gate():
    # Guarding the guard above: if nothing could pass, the split would be meaningless.
    rig = build({"salon": [place("s-3", "Glow Unisex Salon", "Hair salon", "Beauty salon")]})
    outcome = rig.service.discover(request(niches=("salon",)))
    assert outcome.niches[0].qualified == 1
    assert outcome.niches[0].name_gate_rejected == 0


def test_the_status_is_only_stamped_for_niches_a_pass_actually_looked_at():
    # Stamping `last_seen_at` for a niche whose every cell was exhausted would claim a live
    # response that never happened.
    rig = build({"cafe": []})
    rig.service.discover(request())
    rig.service.discover(request())
    rig.status.totals.clear()
    rig.status.last_seen.clear()

    outcome = rig.service.discover(request())

    assert rig.status.totals == {}
    assert outcome.niches[0].state is None


# --- the accumulation rules, pure --------------------------------------------------------------


def test_derive_state_matches_the_threshold_at_every_boundary():
    assert derive_state(0, 0) == UNVERIFIED
    assert derive_state(0, STARVATION_THRESHOLD - 1) == UNVERIFIED
    assert derive_state(0, STARVATION_THRESHOLD) == STARVED
    assert derive_state(1, 1000) == VERIFIED  # sticky: one live qualification proves the slugs


def test_an_observation_reads_the_finished_outcome_and_nothing_else():
    profile = NICHE_PROFILES["cafe"]
    outcome = build_outcome(
        body([*cafes(2), *hardware(3)]),
        profile,
        StubResolver().resolve("Indiranagar, Bangalore, Karnataka, India"),
        limit=100,
        page=1,
        query="cafe",
    )
    observation = NicheObservation.from_outcome(outcome)
    assert (observation.qualified, observation.rejected) == (2, 3)
    assert observation.candidates == 5
    assert observation.rejected_types == {"hardware_store": 3}


def test_a_dropped_sentinel_leaves_the_qualified_column_not_the_rejected_one():
    profile = NICHE_PROFILES["cafe"]
    outcome = build_outcome(
        body([*cafes(2), place("x", "", "Cafe"), *hardware(3)]),
        profile,
        StubResolver().resolve("Indiranagar, Bangalore, Karnataka, India"),
        limit=100,
        page=1,
        query="cafe",
    )
    observation = NicheObservation.from_outcome(outcome, dropped=1)
    assert (observation.qualified, observation.rejected) == (2, 3)


def test_observations_merge_both_buckets():
    first = NicheObservation("cafe", qualified=1, rejected=2, rejected_types={"bar": 2},
                             name_gate_rejected=1)
    second = NicheObservation("cafe", qualified=0, rejected=3,
                              rejected_types={"bar": 1, "pub": 2}, name_gate_rejected=4)
    merged = first.merge(second)
    assert (merged.qualified, merged.rejected) == (1, 5)
    assert merged.rejected_types == {"bar": 3, "pub": 2}
    assert merged.name_gate_rejected == 5


def test_observations_of_different_niches_do_not_merge():
    with pytest.raises(ValueError, match="cannot merge"):
        NicheObservation("cafe").merge(NicheObservation("bakery"))


def test_the_two_buckets_round_trip_through_one_flat_object():
    observation = NicheObservation("salon", rejected_types={"tattoo_shop": 2},
                                   name_gate_rejected=7)
    counts = observation.stored_counts()
    assert counts == {"tattoo_shop": 2, NAME_GATE_KEY: 7}
    assert split_buckets(counts) == ({"tattoo_shop": 2}, 7)
    # The reserved key cannot collide with a Google type slug: `slugify_type` strips leading
    # underscores, so no display label can ever produce one.
    assert NAME_GATE_KEY.startswith("_")


def test_nothing_is_stored_for_a_name_gate_that_never_fired():
    assert NicheObservation("cafe", rejected_types={"bar": 1}).stored_counts() == {"bar": 1}


# ============================================================================================
# GEOGRAPHY: A MISS ABORTS, IT NEVER FALLS BACK
# ============================================================================================


def test_an_unresolvable_area_aborts_before_a_single_credit_is_spent():
    """Half a sweep is a sweep with a hole nothing can see, and a silent fall back to the
    city centre sends a day of fieldwork to the wrong neighbourhood undetected."""
    resolver = StubResolver(error=ProviderError("location_not_found", "no such place"))
    rig = build({"cafe": cafes(6)}, resolver=resolver)

    with pytest.raises(ProviderError):
        rig.service.discover(request(areas=("Indiranagr",)))

    assert rig.provider.calls == []
    assert rig.ledger.used == 0


def test_a_resolution_in_the_wrong_country_aborts():
    # There is one Jaipur in Rajasthan and another in Texas.
    rig = build({"cafe": cafes(6)}, resolver=StubResolver(gl="us"))

    with pytest.raises(ProviderError, match="wrong country"):
        rig.service.discover(request())

    assert rig.provider.calls == []


def test_an_unsupported_niche_is_refused_rather_than_replaced_with_a_near_one():
    rig = build({"cafe": cafes(6)})
    with pytest.raises(UnsupportedNicheError):
        rig.service.discover(request(niches=("artisanal cheese cave",)))
    assert rig.provider.calls == []


def test_every_area_in_the_scope_is_resolved_and_searched_once():
    rig = build({"cafe": cafes(2)})
    outcome = rig.service.discover(request(areas=("Indiranagar", "Koramangala")))
    assert [q for q, _ in rig.resolver.queries] == [
        "Indiranagar, Bangalore, Karnataka, India",
        "Koramangala, Bangalore, Karnataka, India",
    ]
    assert len(outcome.locations) == 2
    assert len(rig.provider.calls) == 2


# ============================================================================================
# THE PROTOTYPE'S ALLOCATION
# ============================================================================================


def test_per_niche_limit_splits_the_sheet_and_never_reaches_zero():
    assert per_niche_limit(60, 3) == 20
    assert per_niche_limit(10, 3) == 4  # ceil, as in the prototype
    assert per_niche_limit(1, 24) == 1  # every niche is worth at least one lead
    with pytest.raises(ValueError):
        per_niche_limit(60, 0)


def test_candidate_limit_widens_reading_not_spending():
    plain = NICHE_PROFILES["cafe"]
    noisy = NICHE_PROFILES["cloud_kitchen"]
    assert candidate_limit(plain, 20) == 20
    # Widened, and bounded by the prototype's floor and ceiling.
    assert candidate_limit(noisy, 20) == 60
    assert candidate_limit(noisy, 1) == 50
    assert candidate_limit(noisy, 1000) == 200


def test_niche_priority_is_the_multiplier_and_only_orders():
    priority = niche_priority([NICHE_PROFILES["cafe"], NICHE_PROFILES["cloud_kitchen"]])
    assert priority == {"cafe": 1, "cloud_kitchen": 3}


def test_a_noisy_niche_is_served_first_when_the_credits_run_short():
    ledger = RecordingLedger(allowance=1)
    rig = build({"cafe": cafes(2), "cloud_kitchen": []}, ledger=ledger)

    outcome = rig.service.discover(request(niches=("cafe", "cloud_kitchen")))

    assert outcome.stopped == BUDGET_EXHAUSTED
    assert [call["niche_id"] for call in rig.provider.calls] == ["cloud_kitchen"]


def test_a_widened_niche_is_still_allowed_its_whole_share_of_the_sheet():
    """The redundant cap that used to sit in `_absorb` capped a scan_multiplier niche's
    KEPT leads by its candidate width, which is the inversion of what the widening is for."""
    profile = NICHE_PROFILES["cloud_kitchen"]
    kitchens = [
        place(f"ck-{i}", f"Biryani Express {i} Cloud Kitchen", "Meal delivery")
        for i in range(6)
    ]
    rig = build({"cloud_kitchen": kitchens})

    outcome = rig.service.discover(request(niches=("cloud_kitchen",), limit=6))

    assert profile.scan_multiplier == 3
    assert len(outcome.businesses) == 6
    assert len(rig.provider.calls) == 1


# ============================================================================================
# THE POLICY OBJECT ITSELF
# ============================================================================================


def test_a_policy_that_pages_is_billed_for_every_page():
    # Not the default, and it costs what it says: three pages, three credits.
    paging = CellPolicy(name="three_pages", pages_per_pass=3)
    rig = build({"cafe": {1: cafes(6, "p1"), 2: cafes(6, "p2"), 3: cafes(6, "p3")}})

    outcome = rig.service.discover(request(policy=paging))

    assert [call["page"] for call in rig.provider.calls] == [1, 2, 3]
    assert outcome.searches_spent == 3
    assert rig.ledger.used == 3
    assert outcome.cells_searched == 3  # one cell, three billed visits


def test_a_paging_policy_stops_paying_the_moment_a_page_stops_yielding():
    paging = CellPolicy(name="three_pages", pages_per_pass=3)
    # Page 1 is new, page 2 is the same twenty again: a strike, then a stop.
    rig = build({"cafe": {1: cafes(6), 2: cafes(6), 3: cafes(6, "p3")}})

    outcome = rig.service.discover(request(policy=paging))

    assert outcome.searches_spent == 2
    assert len(outcome.businesses) == 6


# ============================================================================================
# THE SQL, AGAINST A REAL POSTGRES
# ============================================================================================

integration = pytest.mark.integration
needs_db = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)


@pytest.fixture(scope="session")
def vector_extension():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


@pytest.fixture
def db(vector_extension):
    """A migrated, throwaway schema, dropped at the end of the test. See test_migrations.py."""
    schema = "test_" + uuid.uuid4().hex
    connection = psycopg.connect(DSN)
    connection.execute(f'CREATE SCHEMA "{schema}"')
    connection.execute(f'SET search_path = "{schema}", public')
    connection.commit()
    try:
        apply_migrations(connection)
        yield connection
    finally:
        connection.rollback()
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.commit()
        connection.close()


@pytest.fixture
def status_store(db):
    @contextmanager
    def connect():
        yield db

    return NicheStatusStore(connect)


@integration
@needs_db
def test_the_upsert_accumulates_both_counts_and_both_buckets(status_store):
    status_store.record(
        NicheObservation("salon", qualified=0, rejected=4, rejected_types={"tattoo_shop": 3},
                         name_gate_rejected=1),
        seen_at=NOW,
    )
    stored = status_store.record(
        NicheObservation("salon", qualified=0, rejected=5, rejected_types={"tattoo_shop": 2,
                                                                          "spa": 3},
                         name_gate_rejected=2),
        seen_at=NOW + timedelta(hours=1),
    )

    assert stored.rejected == 9
    assert stored.rejected_types == {"tattoo_shop": 5, "spa": 3}
    assert stored.name_gate_rejected == 3
    assert stored.last_seen_at == NOW + timedelta(hours=1)


@integration
@needs_db
def test_starvation_is_derived_from_post_merge_totals(status_store):
    """0-of-8 twice is 0-of-16, and neither run can see that on its own -- which is why the
    state is computed in the statement rather than passed in by the caller."""
    for _ in range(2):
        row = status_store.record(NicheObservation("cafe", rejected=8), seen_at=NOW)
        assert row.state == UNVERIFIED

    row = status_store.record(NicheObservation("cafe", rejected=8), seen_at=NOW)
    assert row.state == STARVED
    assert row.rejected == 24
    assert [s.niche_id for s in status_store.starved()] == ["cafe"]


@integration
@needs_db
def test_one_qualified_place_verifies_a_niche_and_keeps_it_verified(status_store):
    status_store.record(NicheObservation("cafe", qualified=1, rejected=19), seen_at=NOW)
    row = status_store.record(NicheObservation("cafe", rejected=40), seen_at=NOW)
    assert row.state == VERIFIED
    assert status_store.starved() == []


@integration
@needs_db
def test_the_python_rule_and_the_sql_function_agree(db):
    """`derive_state`, the CASE in the upsert, and `niche_state()` in migration 0012 are
    three spellings of one rule. Drift between them is a disagreement about which niches are
    broken, so it fails here rather than going unnoticed."""
    for qualified in (0, 1, 5):
        for rejected in (0, 1, 19, 20, 21, 100):
            in_sql = db.execute(
                "SELECT niche_state(%s, %s)", (qualified, rejected)
            ).fetchone()[0]
            assert in_sql == derive_state(qualified, rejected), (qualified, rejected)


@integration
@needs_db
def test_the_stored_shape_is_the_one_the_table_documents(status_store, db):
    status_store.record(
        NicheObservation("cafe", rejected=2, rejected_types={"hardware_store": 2}),
        seen_at=NOW,
    )
    raw = db.execute("SELECT rejected_types FROM niche_status WHERE niche_id = 'cafe'").fetchone()
    stored = raw[0] if isinstance(raw[0], dict) else json.loads(raw[0])
    # Google's own type slugs, exactly as 0012's column comment promises -- the string an
    # operator would paste into `include_types`.
    assert stored == {"hardware_store": 2}


@integration
@needs_db
def test_a_niche_that_was_never_searched_has_no_row(status_store):
    assert status_store.get("cafe") is None
    assert status_store.all() == []


# ============================================================================================
# THE REAL FIXTURE PROVIDER, END TO END
# ============================================================================================

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "searchapi")

needs_fixtures = pytest.mark.skipif(
    not os.path.exists(os.path.join(FIXTURES, "salon-p1.json")),
    reason="the recorded SearchAPI corpus is not present",
)


@needs_fixtures
def test_a_pass_over_the_recorded_corpus_costs_one_search_and_charges_nothing():
    """The same pass, driven by the real `FixtureMapsProvider` rather than by this file's fake.

    Two properties, and they belong together. One recorded page is one call, exactly as one
    live page is one credit -- the fixture provider exists so that building the rest of the
    system does not spend the allowance. And a fixture run must leave the ledger untouched,
    or the operator's remaining-credit figure becomes a lie in the safe direction, which is
    the direction nobody checks.
    """
    from lead_engine.providers.fixtures import FixtureMapsProvider

    provider = FixtureMapsProvider(FIXTURES, strict=False)
    ledger = RecordingLedger()
    cells, businesses = MemoryCells(), MemoryBusinesses()
    service = DiscoveryService(
        provider=provider,
        resolver=StubResolver(),
        businesses=businesses,
        cells=cells,
        niche_status=MemoryNicheStatus(),
        budget=ledger,
        clock=lambda: NOW,
    )

    first = service.discover(request(niches=("salon",)))

    assert len(provider.calls) == 1
    assert first.searches_spent == 1
    assert ledger.spends == []
    assert len(first.businesses) > 0
    assert len({b.dedupe_key for b in first.businesses}) == len(first.businesses)

    # And the second pass over the same recording hands back nothing: everything it returned
    # is already held. That is the boundary dedupe, against a real provider.
    second = service.discover(request(niches=("salon",)))
    assert second.businesses == ()
    assert len(provider.calls) == 2


@needs_fixtures
def test_the_recorded_corpus_agrees_with_the_niche_registry_it_was_recorded_for():
    # If a recorded salon page qualifies nothing, either the fixture or `include_types` is
    # wrong -- and that is exactly the finding `starved` exists to raise, so it must not be
    # discovered for the first time on live credits.
    from lead_engine.providers.fixtures import FixtureMapsProvider

    provider = FixtureMapsProvider(FIXTURES, strict=False)
    rig_status = MemoryNicheStatus()
    service = DiscoveryService(
        provider=provider,
        resolver=StubResolver(),
        businesses=MemoryBusinesses(),
        cells=MemoryCells(),
        niche_status=rig_status,
        budget=RecordingLedger(),
        clock=lambda: NOW,
    )

    outcome = service.discover(request(niches=("salon",)))

    assert outcome.niches[0].qualified > 0
    assert outcome.niches[0].state == VERIFIED
    assert outcome.starved_niches == ()


# --- a guard on the fakes themselves -----------------------------------------------------------


def test_the_fake_provider_matches_the_live_clients_signature():
    """These tests are only worth anything if the fake is call-compatible with the client.

    Checked structurally rather than by comment: parameter names and order, because
    `service._sweep` passes `location` and `profile` positionally and everything else by
    keyword, and a fake that had drifted would make every assertion above meaningless.
    """
    import inspect

    from lead_engine.providers.fixtures import FixtureMapsProvider
    from lead_engine.providers.searchapi import SearchApiClient

    live = inspect.signature(SearchApiClient.search_places).parameters
    fixture = inspect.signature(FixtureMapsProvider.search_places).parameters
    fake = inspect.signature(FakeMapsProvider.search_places).parameters
    assert list(fake) == list(live) == list(fixture)
    assert list(fake) == ["self", "location", "profile", "limit", "page", "query_variant"]


def test_the_recording_ledger_matches_the_spend_signature_the_client_uses():
    import inspect

    from lead_engine.providers.budget import SearchBudget

    real = inspect.signature(SearchBudget.spend).parameters
    fake = inspect.signature(RecordingLedger.spend).parameters
    # The client calls `spend(PROVIDER, 1)` positionally and passes nothing else.
    assert list(fake)[:3] == list(real)[:3] == ["self", "provider", "n"]


def test_a_lead_carries_the_niche_it_was_found_for():
    rig = build({"cafe": cafes(1)})
    outcome = rig.service.discover(request())
    assert outcome.businesses[0].niche_id == "cafe"
    assert outcome.businesses[0].lead.matched_niches == ["cafe"]
    # And the spec it was bought under is the cell's own query.
    assert rig.cells.only().spec.query_variant == NICHE_PROFILES["cafe"].queries[0]


def test_a_cell_spec_and_the_query_that_was_issued_are_the_same_string():
    """The cursor records `query_variant` as a string and the credit buys that string. An
    index would drift the day the registry's query list was reordered."""
    rig = build({"cafe": cafes(1)})
    rig.service.discover(request())
    spec: CellSpec = rig.cells.only().spec
    assert rig.provider.calls[0]["query"] == spec.query_variant


def test_the_provider_is_asked_for_the_whole_page_it_was_paid_for():
    # A low `limit` truncates results the credit already bought; the provider cannot fetch
    # a smaller page, so asking for one only throws leads away.
    rig = build({"cafe": cafes(20)})

    outcome = rig.service.discover(request(limit=2))

    assert rig.provider.calls[0]["limit"] >= 20
    # Nothing the credit bought is thrown away at the provider boundary. Discovery's own cap
    # on what it KEEPS is a separate and deliberate thing, and it still ran here.
    assert [o.truncated for o in rig.provider.outcomes] == [0]
    assert len(outcome.businesses) == 2
