"""The search-cell cursor: where the next credit goes, and when to stop paying.

Every assertion in this file is ultimately about the same fifty non-renewing searches. The
three that matter most:

  * `test_breadth_first_plans_exactly_one_cell` -- the default policy buys ONE search per
    (area, niche). The deep sweep is 195 for the same ground, so a change that made tiling
    or variant rotation the default would spend the entire lifetime allowance four times
    over on one niche in one neighbourhood, and would look like a tidy refactor in a diff.
  * `test_a_zero_yield_pass_does_not_advance_the_page` -- an empty page is not evidence that
    the next page is fuller. Advancing past it pays to find out.
  * `test_ensure_never_disturbs_a_cursor_it_already_has` -- the INSERT is DO NOTHING. A DO
    UPDATE would reset `next_page` on every plan, so every run would re-buy page 1 forever
    while looking like it was working.

The pure half runs anywhere. The SQL half asserts the same transitions through the
statements rather than through the functions, because a cursor that only advances in memory
is a cursor that re-buys page 1 tomorrow, and no pure test can see that.
"""

from __future__ import annotations

import math
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

from lead_engine.discovery.cells import (
    BREADTH_FIRST,
    COOLING,
    DEEP_SWEEP,
    EXHAUSTED,
    EXHAUSTION_TTL_DAYS,
    HOT,
    HOT_THRESHOLD,
    PENDING,
    STRUCK,
    CellPolicy,
    CellSpec,
    CellUpdate,
    SearchCell,
    SearchCellStore,
    advance,
    apply,
    hex_tiles,
    is_revivable,
    offset,
    order_cells,
    plan_cells,
    revive,
    selection_key,
    tile_spacing,
)

try:
    import psycopg

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

NOW = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
GOAL = uuid.UUID("11111111-2222-3333-4444-555555555555")

#: Indiranagar, and the radius `geo.scope` resolves an area to.
LAT, LNG, RADIUS = 12.9784, 77.6408, 3000

QUERIES = ("cafe", "coffee shop", "coffee house", "espresso bar", "tea room")


def spec(**overrides) -> CellSpec:
    fields = {
        "goal_id": GOAL,
        "niche_id": "cafe",
        "tile_lat": LAT,
        "tile_lng": LNG,
        "tile_radius_m": RADIUS,
        "query_variant": "cafe",
        "area": "Indiranagar",
    }
    fields.update(overrides)
    return CellSpec(**fields)


def cell(**overrides) -> SearchCell:
    spec_fields = {k: overrides.pop(k) for k in list(overrides) if k in CellSpec.__annotations__}
    return SearchCell(spec=spec(**spec_fields), **overrides)


# --- the identity of a cell ---------------------------------------------------------------


def test_the_key_is_the_tables_unique_constraint():
    assert spec().key == (GOAL, "cafe", LAT, LNG, RADIUS, "cafe")


def test_coordinates_are_rounded_to_the_precision_the_key_compares():
    # `double precision` equality is what decides whether tomorrow's run recognises today's
    # cell or opens a second one beside it and pays for the same ground twice.
    assert spec(tile_lat=12.97840000001, tile_lng=77.64079999999).key[2:4] == (12.9784, 77.6408)


def test_the_area_is_carried_but_is_not_part_of_the_key():
    # Two spellings of one neighbourhood resolve to one point, and paying twice for it is
    # the mistake the key prevents.
    assert spec(area="Indiranagar").key == spec(area="indira nagar").key


def test_a_cell_with_no_query_buys_nothing():
    with pytest.raises(ValueError, match="query_variant"):
        spec(query_variant="   ")


def test_a_radius_must_be_positive():
    with pytest.raises(ValueError, match="tile_radius_m"):
        spec(tile_radius_m=0)


def test_strikes_are_read_back_out_of_the_status():
    assert cell(status=PENDING).strikes == 0
    assert cell(status=COOLING).strikes == 0
    assert cell(status=STRUCK).strikes == 1
    assert cell(status=EXHAUSTED).strikes == 2


# --- the policies: what one pass is allowed to cost ----------------------------------------


def test_breadth_first_costs_one_search_per_area_and_niche():
    assert BREADTH_FIRST.searches_per_area_niche == 1
    assert BREADTH_FIRST.cells_per_area_niche == 1


def test_the_deep_sweep_costs_the_whole_allowance_four_times_over():
    # 13 tiles x 5 variants x 3 pages, against an allowance of 50. Pinned as a number so
    # that anyone tempted to make it the default has to edit this line and read it.
    assert DEEP_SWEEP.searches_per_area_niche == 195
    assert DEEP_SWEEP.cells_per_area_niche == 65


def test_a_multi_tile_policy_must_shrink_its_tiles():
    # Sub-tiles at the parent's radius are the same search thirteen times, at thirteen
    # slightly different centres, for thirteen credits.
    with pytest.raises(ValueError, match="tile_radius_m"):
        CellPolicy(name="broken", tiles=13)


def test_a_policy_cannot_ask_for_less_than_one_of_anything():
    for field_name in ("tiles", "variants", "pages_per_pass"):
        with pytest.raises(ValueError, match=field_name):
            CellPolicy(name="broken", **{field_name: 0})


# --- planning: cells are what a pass may buy -----------------------------------------------


def test_breadth_first_plans_exactly_one_cell():
    """THE default-policy money test. One (area, niche) is one cell is one search."""
    specs = plan_cells(
        BREADTH_FIRST,
        goal_id=GOAL,
        niche_id="cafe",
        queries=QUERIES,
        latitude=LAT,
        longitude=LNG,
        radius_meters=RADIUS,
        area="Indiranagar",
    )
    assert len(specs) == 1
    # The registry lists queries best-first, and with one credit the ranked guess is the
    # only guess worth making. No rotation, no tiling, and the resolved radius.
    assert specs[0].query_variant == QUERIES[0]
    assert (specs[0].tile_lat, specs[0].tile_lng) == (LAT, LNG)
    assert specs[0].tile_radius_m == RADIUS


def test_the_deep_sweep_plans_every_tile_against_every_variant():
    specs = plan_cells(
        DEEP_SWEEP,
        goal_id=GOAL,
        niche_id="cafe",
        queries=QUERIES,
        latitude=LAT,
        longitude=LNG,
        radius_meters=RADIUS,
    )
    assert len(specs) == 65
    assert len({s.key for s in specs}) == 65
    assert {s.tile_radius_m for s in specs} == {DEEP_SWEEP.tile_radius_m}


def test_a_registry_that_lists_one_query_twice_is_not_billed_twice():
    specs = plan_cells(
        CellPolicy(name="two_variants", variants=2),
        goal_id=GOAL,
        niche_id="cafe",
        queries=("cafe", "cafe"),
        latitude=LAT,
        longitude=LNG,
        radius_meters=RADIUS,
    )
    assert len(specs) == 1


def test_a_niche_with_no_queries_is_refused():
    with pytest.raises(ValueError, match="nothing to ask"):
        plan_cells(
            BREADTH_FIRST,
            goal_id=GOAL,
            niche_id="cafe",
            queries=(),
            latitude=LAT,
            longitude=LNG,
            radius_meters=RADIUS,
        )


# --- tiling --------------------------------------------------------------------------------


def metres_from_centre(lat: float, lng: float) -> float:
    """Distance in METRES, which is the axis `hex_tiles` orders on. Degrees would not do:
    a degree of longitude is narrower than a degree of latitude here, so two tiles equally
    far away in metres are measurably different distances in degree space."""
    north = (lat - LAT) * 111_320.0
    east = (lng - LNG) * 111_320.0 * math.cos(math.radians(LAT))
    return math.hypot(east, north)


def test_hex_tiles_returns_the_count_asked_for_nearest_first():
    tiles = hex_tiles(LAT, LNG, 13, tile_spacing(1200))
    assert len(tiles) == 13
    assert tiles[0] == (round(LAT, 6), round(LNG, 6))
    # Non-decreasing to within the storage precision. The centres are rounded to 6 decimal
    # places before they are written -- they are part of the table's unique key -- so two
    # tiles in one ring land a few centimetres apart, in either order.
    distances = [metres_from_centre(lat, lng) for lat, lng in tiles]
    pairs = zip(distances, distances[1:], strict=False)
    assert all(later >= earlier - 0.5 for earlier, later in pairs)
    # One ring of six at the lattice spacing, then the next. Nothing is buying ground it has
    # already covered from the centre.
    assert distances[1] == pytest.approx(tile_spacing(1200), rel=0.01)


def test_hex_tiles_is_deterministic():
    # The order decides which ground gets bought when the credits run out part way, so two
    # runs of one request must not spend them differently for no reason.
    assert hex_tiles(LAT, LNG, 7, 2000.0) == hex_tiles(LAT, LNG, 7, 2000.0)


def test_an_eastward_offset_shrinks_with_the_cosine_of_latitude():
    # Ignoring it stretches every tile east-west by ~13% at Bangalore's latitude.
    _, lng = offset(LAT, LNG, east_meters=1000, north_meters=0)
    naive = LNG + 1000 / 111_320.0
    assert lng > LNG
    assert lng > naive  # a degree of longitude is narrower here than a degree of latitude


def test_a_polar_offset_does_not_become_an_infinity():
    lat, lng = offset(90.0, 0.0, east_meters=1000, north_meters=0)
    assert lng == 0.0
    assert math.isfinite(lat)


# --- the measured transitions --------------------------------------------------------------


def test_a_productive_pass_advances_the_page_and_stays_hot():
    update = advance(cell(), new_count=HOT_THRESHOLD, total_count=20, now=NOW)
    assert (update.next_page, update.status) == (2, HOT)
    assert update.new_yield == HOT_THRESHOLD
    assert update.exhausted_at is None


def test_a_thinning_pass_advances_the_page_and_cools():
    update = advance(cell(), new_count=1, total_count=20, now=NOW)
    assert (update.next_page, update.status) == (2, COOLING)


def test_a_zero_yield_pass_does_not_advance_the_page():
    """An empty page is not evidence that page N+1 is fuller. Paying to find out is the
    guess this system is not rich enough to make."""
    update = advance(cell(next_page=3), new_count=0, total_count=20, now=NOW)
    assert update.next_page == 3
    assert update.status == STRUCK
    assert update.exhausted is False


def test_two_consecutive_zero_yield_passes_exhaust_the_cell():
    struck = apply(cell(), advance(cell(), new_count=0, total_count=20, now=NOW))
    assert struck.status == STRUCK

    update = advance(struck, new_count=0, total_count=20, now=NOW)
    assert update.status == EXHAUSTED
    assert update.exhausted_at == NOW
    assert apply(struck, update).exhausted is True


def test_any_yield_at_all_clears_a_strike():
    # A single zero is routinely transient -- a rate-limited response, a niche whose one new
    # opening this month was already in the database.
    struck = apply(cell(), advance(cell(), new_count=0, total_count=20, now=NOW))
    assert apply(struck, advance(struck, new_count=1, total_count=20, now=NOW)).strikes == 0


def test_total_yield_accumulates_while_new_yield_is_only_the_last_pass():
    first = apply(cell(), advance(cell(), new_count=9, total_count=20, now=NOW))
    second = apply(first, advance(first, new_count=2, total_count=20, now=NOW))
    # The difference between "returned nothing" and "returned the same twenty for the
    # fourth time" is the whole reason both numbers exist.
    assert (second.total_yield, second.new_yield) == (40, 2)


def test_a_negative_count_cannot_move_the_cursor_backwards():
    update = advance(cell(), new_count=-5, total_count=-20, now=NOW)
    assert (update.new_yield, update.total_yield, update.status) == (0, 0, STRUCK)


# --- exhaustion expires ----------------------------------------------------------------------


def test_exhaustion_expires_after_thirty_days():
    stale = cell(status=EXHAUSTED, exhausted_at=NOW - timedelta(days=EXHAUSTION_TTL_DAYS, hours=1))
    fresh = cell(status=EXHAUSTED, exhausted_at=NOW - timedelta(days=EXHAUSTION_TTL_DAYS - 1))
    assert is_revivable(stale, NOW) is True
    assert is_revivable(fresh, NOW) is False


def test_a_live_cell_is_never_revived():
    assert is_revivable(cell(status=STRUCK, exhausted_at=None), NOW) is False


def test_an_exhausted_cell_with_no_timestamp_cannot_prove_it_served_its_time():
    assert is_revivable(cell(status=EXHAUSTED, exhausted_at=None), NOW) is False


def test_revival_goes_back_to_page_one_and_drops_the_stale_measurement():
    """Re-sweeping page 1 is the ONLY path by which a business that opened last month ever
    enters this system, and a months-old `new_yield` must not let a revived cell outrank
    ground that is producing today."""
    revived = revive(cell(status=EXHAUSTED, next_page=4, new_yield=7, total_yield=80,
                          exhausted_at=NOW))
    assert (revived.status, revived.next_page, revived.new_yield) == (PENDING, 1, 0)
    assert revived.total_yield == 80
    assert revived.exhausted_at is None


# --- selection -------------------------------------------------------------------------------


def test_niche_priority_outranks_measured_yield():
    # `scan_multiplier` decides who gets the scarce credits FIRST. It never decides how many
    # there are -- see test_discovery.py.
    noisy = cell(niche_id="cloud_kitchen", new_yield=0, id=2)
    proven = cell(niche_id="cafe", new_yield=9, id=1)
    ordered = order_cells([proven, noisy], {"cloud_kitchen": 3, "cafe": 1})
    assert [c.niche_id for c in ordered] == ["cloud_kitchen", "cafe"]


def test_the_highest_yielding_cell_goes_first_at_equal_priority():
    cells = [cell(new_yield=1, id=1), cell(new_yield=8, id=2), cell(new_yield=4, id=3)]
    assert [c.id for c in order_cells(cells)] == [2, 3, 1]


def test_an_unmeasured_cell_outranks_one_that_has_already_been_struck():
    # An unmeasured cell might be anything; a struck one has just said it is empty.
    cells = [cell(status=STRUCK, id=1), cell(status=PENDING, id=2), cell(status=HOT, id=3)]
    assert [c.id for c in order_cells(cells)] == [3, 2, 1]


def test_selection_is_deterministic_when_everything_else_ties():
    cells = [cell(id=9, query_variant="tea room"), cell(id=4, query_variant="cafe")]
    assert [c.id for c in order_cells(cells)] == [4, 9]
    assert order_cells(cells) == order_cells(list(reversed(cells)))


def test_order_cells_is_selection_key_and_nothing_else():
    # `service._plan` sorts work items by `selection_key` directly. If the two disagreed,
    # the cell that was chosen and the cell that was searched could differ.
    cells = [cell(new_yield=y, id=i) for i, y in enumerate([3, 0, 7, 1], start=1)]
    priority = {"cafe": 2}
    assert order_cells(cells, priority) == sorted(cells, key=lambda c: selection_key(c, priority))


# --- the SQL, against a real Postgres --------------------------------------------------------


integration = pytest.mark.integration
needs_db = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)


@pytest.fixture(scope="session")
def vector_extension():
    """pgvector, installed once for the session -- `CREATE EXTENSION` is database-scoped."""
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
        connection.execute(
            "INSERT INTO goals (id, name, spec) VALUES (%s, 'discovery test', '{}')", (GOAL,)
        )
        connection.commit()
        yield connection
    finally:
        connection.rollback()
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.commit()
        connection.close()


@pytest.fixture
def store(db):
    @contextmanager
    def connect():
        yield db

    return SearchCellStore(connect)


@integration
@needs_db
def test_ensure_is_idempotent(store):
    first = store.ensure([spec()])
    second = store.ensure([spec()])
    assert [c.id for c in first] == [c.id for c in second]
    assert first[0].status == PENDING and first[0].next_page == 1


@integration
@needs_db
def test_ensure_never_disturbs_a_cursor_it_already_has(store):
    """The INSERT is DO NOTHING. A DO UPDATE would reset `next_page` every time a pass
    planned its cells, so every run would re-buy page 1 forever and look like it worked."""
    created = store.ensure([spec()])[0]
    store.record(created, advance(created, new_count=9, total_count=20, now=NOW))

    again = store.ensure([spec()])[0]
    assert (again.next_page, again.status, again.new_yield) == (2, HOT, 9)


@integration
@needs_db
def test_two_radii_at_one_point_are_two_cells(store):
    # The radius is one of the four discovery axes, and 0004 puts it in the unique key.
    cells = store.ensure([spec(), spec(tile_radius_m=1200)])
    assert len({c.id for c in cells}) == 2


@integration
@needs_db
def test_recording_a_pass_writes_every_measured_column(store):
    created = store.ensure([spec()])[0]
    written = store.record(created, advance(created, new_count=2, total_count=20, now=NOW))
    assert (written.next_page, written.status, written.new_yield, written.total_yield) == (
        2,
        COOLING,
        2,
        20,
    )
    assert written.last_searched_at == NOW
    assert store.ensure([spec()])[0] == written


@integration
@needs_db
def test_a_cell_that_was_never_persisted_cannot_record_a_pass(store):
    with pytest.raises(ValueError, match="never persisted"):
        store.record(cell(), CellUpdate(2, HOT, 5, 20, NOW, None))


@integration
@needs_db
def test_exhaustion_survives_the_round_trip_and_leaves_the_live_set(store):
    created = store.ensure([spec()])[0]
    struck = store.record(created, advance(created, new_count=0, total_count=20, now=NOW))
    done = store.record(struck, advance(struck, new_count=0, total_count=20, now=NOW))

    assert done.status == EXHAUSTED
    assert done.exhausted_at is not None
    assert store.live(GOAL, ["cafe"]) == []


@integration
@needs_db
def test_revive_expired_resets_only_cells_that_served_their_thirty_days(store):
    stale, fresh = store.ensure([spec(), spec(query_variant="tea room")])
    old = NOW - timedelta(days=EXHAUSTION_TTL_DAYS + 1)
    store.record(stale, CellUpdate(4, EXHAUSTED, 0, 80, old, old))
    store.record(fresh, CellUpdate(2, EXHAUSTED, 0, 20, NOW, NOW - timedelta(days=1)))

    revived = store.revive_expired(GOAL, NOW)

    assert [c.spec.query_variant for c in revived] == ["cafe"]
    assert (revived[0].status, revived[0].next_page, revived[0].new_yield) == (PENDING, 1, 0)
    # The lifetime record survives the reset; the stale measurement does not.
    assert revived[0].total_yield == 80
    assert revived[0].exhausted_at is None
    assert store.live(GOAL, ["cafe"]) == revived


@integration
@needs_db
def test_the_live_query_orders_by_measured_yield(store):
    low, high = store.ensure([spec(), spec(query_variant="tea room")])
    store.record(low, advance(low, new_count=1, total_count=20, now=NOW))
    store.record(high, advance(high, new_count=9, total_count=20, now=NOW))
    assert [c.new_yield for c in store.live(GOAL, ["cafe"])] == [9, 1]
