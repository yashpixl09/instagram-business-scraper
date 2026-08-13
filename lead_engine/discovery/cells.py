"""The search-cell cursor: where the next billed search goes, and when to stop paying.

Google Maps returns the same twenty ranked results for the same query point every time. A
sweep with no memory therefore re-buys those twenty on every run, filters all twenty away
as duplicates, and reports that it found nothing new -- having spent a credit to learn it.
A *cell* is one position in the four-dimensional discovery space:

    page  x  tile  x  query_variant  x  radius

and this module is the cursor over that space, persisted in `search_cells` (0004).

THE DEFAULT IS BREADTH-FIRST, AND THAT IS A MONEY DECISION
----------------------------------------------------------
The operator has FIFTY searches. One-time, non-renewing. An exhaustive sweep of a single
niche in a single neighbourhood -- 13 tiles x 5 query variants x 3 pages -- is 195 searches:
the entire lifetime allowance, four times over, for one niche on one patch of one city.

So `BREADTH_FIRST` is the default policy: **one cell per (area, niche), page 1, one query
variant, the radius the geocoder resolved. No tiling, no variant rotation, no pagination.**
`DEEP_SWEEP` exists, is implemented, and is off unless a caller names it. Both are named
objects rather than a pile of keyword defaults precisely so that switching costs a visible
line in a diff and shows up in a code review as what it is: a decision to spend 195 credits
instead of 1.

The cursor still moves *between* passes. A cell that produced results advances its
`next_page`, so tomorrow's run of the same breadth-first sweep reads page 2 rather than
re-buying page 1. Breadth-first constrains what one pass costs; it does not freeze the
cursor.

EXHAUSTION IS MEASURED, NEVER PREDICTED
---------------------------------------
Nothing here estimates how many salons a neighbourhood holds. The only evidence admitted is
how many NEW businesses the last search actually produced:

    >= 5 new   ->  advance `next_page`, stay HOT
    1..4 new   ->  advance `next_page`, drop to COOLING (deprioritised, still searched)
    0 new      ->  a strike; `next_page` does NOT advance
    2 consecutive strikes -> EXHAUSTED

A strike does not advance the page because there is nothing to advance past: an empty page
is not evidence that page N+1 is fuller. Two consecutive strikes, not one, because a single
zero is routinely a transient -- a rate-limited response, a niche whose one new opening this
month happened to already be in the database.

The strike count needs no column of its own: `STRUCK` *is* one consecutive strike, held in
the `status` the schema already has. Any non-zero yield clears it.

EXHAUSTION EXPIRES AFTER 30 DAYS
--------------------------------
An exhausted cell resets to PENDING at page 1 after `EXHAUSTION_TTL_DAYS`. The yield will be
low, and that is accepted knowingly: re-sweeping page 1 is the ONLY path by which a business
that opened last month ever enters this system. A permanently exhausted cell is a permanent
blind spot, and it grows.

`new_yield` is deliberately reset by that revival. It is a measurement, and a months-old
measurement must not let a revived cell jump the selection queue ahead of ground that is
producing today. `total_yield` is the lifetime record and survives.

SELECTION
---------
Highest `new_yield` among unexhausted cells first, so a sweep self-organises toward
productive ground with no manual tuning. `scan_multiplier` from the niche registry enters
HERE, as a priority multiplier on the ordering -- never as a multiplier on how many searches
are issued. See `order_cells` and the note in `service.py`.

WHY THIS FILE IMPORTS NO DATABASE DRIVER
----------------------------------------
Same reason `providers/budget.py` does not: the decisions above are pure functions of a
cell's recorded state, and they are worth testing without a Postgres anywhere near them.
`SearchCellStore` talks to whatever `execute`/`commit` object its connection factory yields.
The SQL is still real SQL and still has to be proved against a real database -- see the
integration half of `tests/test_cells.py`, which asserts the same transitions through the
statements rather than through the pure functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

# --- the measured thresholds -----------------------------------------------------------

#: New businesses in one pass at or above which a cell stays HOT.
HOT_THRESHOLD = 5

#: Consecutive zero-yield passes that exhaust a cell.
STRIKES_TO_EXHAUST = 2

#: How long an exhausted cell stays exhausted before it is swept again from page 1.
EXHAUSTION_TTL_DAYS = 30

# --- the status vocabulary -------------------------------------------------------------
#
# `search_cells.status` is plain `text NOT NULL DEFAULT 'pending'` in 0004, with no CHECK,
# so this tuple is the only definition of the vocabulary. STRUCK exists so that "one
# consecutive strike" is durable state rather than a counter the schema has no column for.

PENDING = "pending"  # never searched
HOT = "hot"  # last pass produced >= HOT_THRESHOLD new businesses
COOLING = "cooling"  # last pass produced 1..HOT_THRESHOLD-1: real, but thinning
STRUCK = "struck"  # last pass produced nothing. One more zero and it is done.
EXHAUSTED = "exhausted"  # two consecutive zeros. Not searched again for 30 days.

STATUSES: tuple[str, ...] = (PENDING, HOT, COOLING, STRUCK, EXHAUSTED)

#: Selection order among unexhausted cells when their measured yields tie. A never-searched
#: cell outranks one that has already been struck, because an unmeasured cell might be
#: anything while a struck one has just told us it is empty.
_STATUS_RANK: dict[str, int] = {HOT: 0, PENDING: 1, COOLING: 2, STRUCK: 3, EXHAUSTED: 4}

#: Tile centres are rounded before they are written, because they are part of the table's
#: unique key. `search_cells` keys on (goal_id, niche_id, tile_lat, tile_lng, tile_radius_m,
#: query_variant), and `double precision` equality is what decides whether tomorrow's run
#: recognises today's cell or silently opens a second one beside it and pays for it again.
#: Six decimal places is ~11cm -- far below any radius this system uses.
COORDINATE_PRECISION = 6

#: Metres per degree of latitude. Spherical-earth approximation, which is accurate to well
#: under a percent at the tile sizes here (1-15km) and does not warrant pulling in a
#: geodesy dependency for a value that only positions a search centre.
METERS_PER_DEGREE_LATITUDE = 111_320.0


# --- the connection seam ---------------------------------------------------------------


class Cursor(Protocol):
    """The sliver of DBAPI this package uses."""

    def fetchone(self) -> Sequence[Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...


class Connection(Protocol):
    """The sliver of a psycopg connection this package uses."""

    def execute(self, query: str, params: Sequence[Any] | None = ..., /) -> Cursor: ...

    def commit(self) -> None: ...


#: Something that yields a connection per `with` block.
#: `psycopg_pool.ConnectionPool.connection` is exactly this shape.
ConnectionFactory = Callable[[], AbstractContextManager[Connection]]

Clock = Callable[[], datetime]


# --- what a cell is --------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """The identity of a cell: exactly the columns of the table's unique key, plus `area`.

    `area` is carried but is NOT part of the key, matching 0004. Two areas that resolve to
    the same point at the same radius are the same search, and paying for it twice because
    an operator spelled the neighbourhood differently is the mistake the key prevents.
    """

    goal_id: UUID
    niche_id: str
    tile_lat: float
    tile_lng: float
    tile_radius_m: int
    query_variant: str
    area: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_lat", round(float(self.tile_lat), COORDINATE_PRECISION))
        object.__setattr__(self, "tile_lng", round(float(self.tile_lng), COORDINATE_PRECISION))
        radius = int(self.tile_radius_m)
        if radius <= 0:
            raise ValueError(f"tile_radius_m must be positive, got {self.tile_radius_m!r}")
        object.__setattr__(self, "tile_radius_m", radius)
        if not str(self.query_variant).strip():
            raise ValueError("query_variant is required: a cell with no query buys nothing")
        if not str(self.niche_id).strip():
            raise ValueError("niche_id is required")

    @property
    def key(self) -> tuple[Any, ...]:
        """The tuple the table's UNIQUE constraint compares."""
        return (
            self.goal_id,
            self.niche_id,
            self.tile_lat,
            self.tile_lng,
            self.tile_radius_m,
            self.query_variant,
        )


@dataclass(frozen=True)
class SearchCell:
    """A row of `search_cells`: a spec plus everything the cursor has measured about it."""

    spec: CellSpec
    next_page: int = 1
    status: str = PENDING
    new_yield: int = 0
    total_yield: int = 0
    last_searched_at: datetime | None = None
    exhausted_at: datetime | None = None
    id: int | None = None

    @property
    def niche_id(self) -> str:
        return self.spec.niche_id

    @property
    def area(self) -> str | None:
        return self.spec.area

    @property
    def exhausted(self) -> bool:
        return self.status == EXHAUSTED

    @property
    def strikes(self) -> int:
        """Consecutive zero-yield passes, read back out of `status`."""
        if self.status == STRUCK:
            return 1
        return STRIKES_TO_EXHAUST if self.status == EXHAUSTED else 0


@dataclass(frozen=True)
class CellUpdate:
    """What one measured pass does to a cell. Pure data, so it can be asserted on."""

    next_page: int
    status: str
    new_yield: int
    total_yield: int
    last_searched_at: datetime
    exhausted_at: datetime | None

    @property
    def exhausted(self) -> bool:
        return self.status == EXHAUSTED


# --- the policies ----------------------------------------------------------------------


@dataclass(frozen=True)
class CellPolicy:
    """How much of the discovery space one pass is allowed to buy.

    Every field multiplies directly into money: the searches one (area, niche) may cost in
    a single pass is `tiles * variants * pages_per_pass`. That is why this is one named,
    frozen object per policy instead of four defaulted keyword arguments -- so the cost of a
    pass is a thing you can read, print in a run summary, and diff.
    """

    name: str
    tiles: int = 1
    variants: int = 1
    pages_per_pass: int = 1
    #: Radius for a generated tile. None means "use the radius the geocoder resolved", which
    #: is the only correct answer when there is exactly one tile.
    tile_radius_m: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("tiles", "variants", "pages_per_pass"):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        if self.tile_radius_m is not None and int(self.tile_radius_m) <= 0:
            raise ValueError("tile_radius_m must be positive when set")
        if self.tiles > 1 and self.tile_radius_m is None:
            # Sub-tiles at the parent's radius are not sub-tiles; they are the same search
            # thirteen times, at thirteen slightly different centres, for thirteen credits.
            raise ValueError("a multi-tile policy must set tile_radius_m")

    @property
    def searches_per_area_niche(self) -> int:
        """Upper bound on billed searches for one (area, niche) in one pass."""
        return self.tiles * self.variants * self.pages_per_pass

    @property
    def cells_per_area_niche(self) -> int:
        return self.tiles * self.variants


#: THE DEFAULT. One search per (area, niche) per pass. See the module docstring.
BREADTH_FIRST = CellPolicy(name="breadth_first", tiles=1, variants=1, pages_per_pass=1)

#: The deep sweep, off unless explicitly passed. 13 x 5 x 3 = 195 searches per (area,
#: niche) -- nearly four times the entire lifetime allowance. It exists because a
#: neighbourhood genuinely worth exhausting should not require new code, not because it is
#: ever the right default.
DEEP_SWEEP = CellPolicy(
    name="deep_sweep", tiles=13, variants=5, pages_per_pass=3, tile_radius_m=1200
)


# --- tiling ----------------------------------------------------------------------------


def hex_tiles(
    latitude: float,
    longitude: float,
    count: int,
    spacing_meters: float,
) -> tuple[tuple[float, float], ...]:
    """`count` centres on a hex lattice around a point, nearest-first.

    Hex packing, not a square grid: circles on a hexagonal lattice cover a plane with the
    least overlap of any regular arrangement, and overlap here is duplicate results bought
    twice. For full coverage by circles of radius r the lattice spacing is r*sqrt(3), which
    is what `tile_spacing` computes.

    Ordering is nearest-to-centre, then by angle, and it is deterministic. That matters
    beyond tidiness: the order decides which tiles get bought first when the budget runs out
    part-way, and a non-deterministic order would make two runs of the same request spend
    their credits on different ground for no reason.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if spacing_meters <= 0:
        raise ValueError("spacing_meters must be positive")

    # Axial hex coordinates -> plane offsets in metres. `rings` is chosen so the lattice
    # holds at least `count` points: ring k contributes 6k, totalling 3k(k+1)+1.
    rings = 0
    while 3 * rings * (rings + 1) + 1 < count:
        rings += 1

    points: list[tuple[float, float, float, float]] = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if abs(q + r) > rings:  # the hexagon, not the rhombus
                continue
            east = spacing_meters * (q + r / 2.0)
            north = spacing_meters * (math.sqrt(3.0) / 2.0) * r
            points.append((math.hypot(east, north), math.atan2(north, east), east, north))

    points.sort(key=lambda item: (round(item[0], 3), round(item[1], 6)))
    return tuple(offset(latitude, longitude, east, north) for _, _, east, north in points[:count])


def tile_spacing(tile_radius_meters: float) -> float:
    """Hex-lattice spacing that covers the plane with circles of this radius."""
    return tile_radius_meters * math.sqrt(3.0)


def offset(latitude: float, longitude: float, east_meters: float, north_meters: float):
    """Move a coordinate by a metre offset, rounded to the key's precision.

    Longitude degrees shrink with the cosine of latitude; ignoring that would stretch every
    tile east-west by ~13% at Bangalore's latitude and by more further north.
    """
    latitude = float(latitude)
    delta_lat = north_meters / METERS_PER_DEGREE_LATITUDE
    scale = math.cos(math.radians(latitude))
    # A degree of longitude at the pole is zero metres wide; refusing to divide by ~0 keeps
    # a polar coordinate from becoming an infinity that the database would happily store.
    if abs(scale) < 1e-9:
        delta_lng = 0.0
    else:
        delta_lng = east_meters / (METERS_PER_DEGREE_LATITUDE * scale)
    return (
        round(latitude + delta_lat, COORDINATE_PRECISION),
        round(float(longitude) + delta_lng, COORDINATE_PRECISION),
    )


def plan_cells(
    policy: CellPolicy,
    *,
    goal_id: UUID,
    niche_id: str,
    queries: Sequence[str],
    latitude: float,
    longitude: float,
    radius_meters: int,
    area: str | None = None,
) -> tuple[CellSpec, ...]:
    """Every cell one (area, niche) is allowed under this policy. Pure; buys nothing.

    Under `BREADTH_FIRST` this returns exactly one spec: `queries[0]` at the resolved point
    and the resolved radius. `queries[0]` rather than a rotation because the registry lists
    each niche's queries best-first, and with one credit to spend the ranked guess is the
    only guess worth making.
    """
    if not queries:
        raise ValueError(f"niche {niche_id!r} declares no queries; there is nothing to ask")

    radius = int(policy.tile_radius_m or radius_meters)
    if policy.tiles == 1:
        centres: tuple[tuple[float, float], ...] = (
            (round(float(latitude), COORDINATE_PRECISION),
             round(float(longitude), COORDINATE_PRECISION)),
        )
    else:
        centres = hex_tiles(latitude, longitude, policy.tiles, tile_spacing(radius))

    variants = [queries[index % len(queries)] for index in range(policy.variants)]
    # dict.fromkeys keeps the order while collapsing a registry that lists the same query
    # twice -- two identical variants at one point are one search bought twice.
    variants = list(dict.fromkeys(variants))

    return tuple(
        CellSpec(
            goal_id=goal_id,
            niche_id=niche_id,
            tile_lat=lat,
            tile_lng=lng,
            tile_radius_m=radius,
            query_variant=variant,
            area=area,
        )
        for lat, lng in centres
        for variant in variants
    )


# --- the measured transitions ----------------------------------------------------------


def advance(cell: SearchCell, *, new_count: int, total_count: int, now: datetime) -> CellUpdate:
    """What one searched pass did to this cell. Pure, and the whole exhaustion rule.

    `new_count` is businesses this pass added that the database did not already hold -- the
    only honest measure of what the credit bought. `total_count` is everything the provider
    returned, duplicates included, and is accumulated so that a later reader can see the
    difference between a cell returning nothing and a cell returning the same twenty places
    for the fourth time.
    """
    new_count = max(0, int(new_count))
    total_yield = cell.total_yield + max(0, int(total_count))

    if new_count >= HOT_THRESHOLD:
        return CellUpdate(cell.next_page + 1, HOT, new_count, total_yield, now, None)
    if new_count > 0:
        return CellUpdate(cell.next_page + 1, COOLING, new_count, total_yield, now, None)

    # Zero. The page does not advance: an empty page is not evidence that the next one is
    # fuller, and paying to find out is the guess this system is not rich enough to make.
    struck_before = cell.status in (STRUCK, EXHAUSTED)
    status = EXHAUSTED if struck_before else STRUCK
    return CellUpdate(
        next_page=cell.next_page,
        status=status,
        new_yield=0,
        total_yield=total_yield,
        last_searched_at=now,
        exhausted_at=now if status == EXHAUSTED else None,
    )


def apply(cell: SearchCell, update: CellUpdate) -> SearchCell:
    """The updated cell, in memory. What `SearchCellStore.record` writes to the row."""
    return replace(
        cell,
        next_page=update.next_page,
        status=update.status,
        new_yield=update.new_yield,
        total_yield=update.total_yield,
        last_searched_at=update.last_searched_at,
        exhausted_at=update.exhausted_at,
    )


def revival_cutoff(now: datetime) -> datetime:
    """Cells exhausted before this instant have served their 30 days."""
    return now - timedelta(days=EXHAUSTION_TTL_DAYS)


def is_revivable(cell: SearchCell, now: datetime) -> bool:
    if not cell.exhausted:
        return False
    # An exhausted cell with no timestamp cannot prove it has served its time. Reviving it
    # would re-sweep on no evidence; leaving it is the conservative half of a bug that
    # should not happen, and `record` always writes the timestamp with the status.
    return cell.exhausted_at is not None and cell.exhausted_at < revival_cutoff(now)


def revive(cell: SearchCell) -> SearchCell:
    """Back to the start of the cursor: pending, page 1, no stale yield.

    `new_yield` resets and `total_yield` does not. The first is a measurement of what this
    ground produced *last time it was searched*, and a months-old one must not let a revived
    cell outrank ground that is producing now. The second is the lifetime record.
    """
    return replace(cell, status=PENDING, next_page=1, new_yield=0, exhausted_at=None)


def order_cells(
    cells: Iterable[SearchCell],
    priority: dict[str, int] | None = None,
) -> list[SearchCell]:
    """Selection order: priority first, then measured yield, then status, then id.

    `priority` is keyed by niche id and comes from `NicheProfile.scan_multiplier`. It orders
    cells; it never creates them. Under the breadth-first policy the number of billed
    searches is exactly the number of cells, so a multiplier that reached the cell *count*
    would turn a 1-credit niche into a 3-credit one -- which is precisely the bug this
    argument exists to avoid. What it does instead is decide who gets the credits first when
    there are not enough to go round.
    """
    priority = priority or {}

    def key(cell: SearchCell) -> tuple[Any, ...]:
        return (
            -int(priority.get(cell.niche_id, 1)),
            -cell.new_yield,
            _STATUS_RANK.get(cell.status, len(_STATUS_RANK)),
            cell.id if cell.id is not None else 0,
            cell.spec.query_variant,
        )

    return sorted(cells, key=key)


# --- SQL -------------------------------------------------------------------------------
#
# Every statement this module runs, as a module-level constant, following the discipline of
# `lead_engine/db/queries.py`: SQL is greppable in one place and never assembled from
# fragments at a call site.

_COLUMNS = (
    "id, goal_id, niche_id, area, tile_lat, tile_lng, tile_radius_m, query_variant, "
    "next_page, status, new_yield, total_yield, last_searched_at, exhausted_at"
)

# DO NOTHING, never DO UPDATE. A pass re-planning the cells it planned yesterday must find
# yesterday's cursor exactly where it left it; a DO UPDATE that reset `next_page` or
# `new_yield` would silently re-buy page 1 forever and look like it was working.
_INSERT = """
INSERT INTO search_cells (goal_id, niche_id, area, tile_lat, tile_lng, tile_radius_m,
                          query_variant)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (goal_id, niche_id, tile_lat, tile_lng, tile_radius_m, query_variant)
DO NOTHING
"""

_SELECT_ONE = f"""
SELECT {_COLUMNS}
  FROM search_cells
 WHERE goal_id = %s AND niche_id = %s AND tile_lat = %s AND tile_lng = %s
   AND tile_radius_m = %s AND query_variant = %s
"""

# Unexhausted only, highest measured yield first -- which is exactly the column order of
# 0004's `(goal_id, status, new_yield DESC)` index. Niche priority is applied afterwards in
# `order_cells`, in process, because it comes from the registry rather than the database.
_SELECT_LIVE = f"""
SELECT {_COLUMNS}
  FROM search_cells
 WHERE goal_id = %s AND niche_id = ANY(%s) AND status <> 'exhausted'
 ORDER BY new_yield DESC, id
"""

_RECORD = f"""
UPDATE search_cells
   SET next_page = %s,
       status = %s,
       new_yield = %s,
       total_yield = %s,
       last_searched_at = %s,
       exhausted_at = %s
 WHERE id = %s
RETURNING {_COLUMNS}
"""

# The 30-day reset, as one statement over the whole goal. `now` is a parameter rather than
# `now()` so that a test can prove the boundary without waiting a month, and so that one
# pass reads a single consistent instant.
_REVIVE_EXPIRED = f"""
UPDATE search_cells
   SET status = 'pending',
       next_page = 1,
       new_yield = 0,
       exhausted_at = NULL
 WHERE goal_id = %s
   AND status = 'exhausted'
   AND exhausted_at IS NOT NULL
   AND exhausted_at < %s::timestamptz - interval '{EXHAUSTION_TTL_DAYS} days'
RETURNING {_COLUMNS}
"""


def _row(values: Sequence[Any]) -> SearchCell:
    (
        cell_id,
        goal_id,
        niche_id,
        area,
        tile_lat,
        tile_lng,
        tile_radius_m,
        query_variant,
        next_page,
        status,
        new_yield,
        total_yield,
        last_searched_at,
        exhausted_at,
    ) = values
    return SearchCell(
        spec=CellSpec(
            goal_id=goal_id,
            niche_id=niche_id,
            tile_lat=tile_lat,
            tile_lng=tile_lng,
            tile_radius_m=tile_radius_m,
            query_variant=query_variant,
            area=area,
        ),
        next_page=next_page,
        status=status,
        new_yield=new_yield,
        total_yield=total_yield,
        last_searched_at=last_searched_at,
        exhausted_at=exhausted_at,
        id=cell_id,
    )


class SearchCellStore:
    """`search_cells`, as a cursor. Stateless; all the state is in Postgres.

    Takes a connection factory (`pool.connection`) and commits its own writes, for the same
    reason `SearchBudget` does: a cursor advance that is still uncommitted when the worker
    dies is a credit spent whose position was never recorded, and the next run pays for the
    same page again.
    """

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect

    def ensure(self, specs: Iterable[CellSpec]) -> list[SearchCell]:
        """Create any cell that does not exist, and return all of them as they now stand.

        Idempotent, and the returned cells carry whatever the cursor already knew, which is
        the point: planning is cheap and repeatable, and it must never disturb measurement.
        """
        specs = list(specs)
        if not specs:
            return []
        cells: list[SearchCell] = []
        with self._connect() as conn:
            for spec in specs:
                conn.execute(
                    _INSERT,
                    (
                        spec.goal_id,
                        spec.niche_id,
                        spec.area,
                        spec.tile_lat,
                        spec.tile_lng,
                        spec.tile_radius_m,
                        spec.query_variant,
                    ),
                )
            for spec in specs:
                row = conn.execute(_SELECT_ONE, spec.key).fetchone()
                if row is None:  # pragma: no cover - the insert above guarantees a row
                    raise RuntimeError(f"search cell vanished after insert: {spec.key!r}")
                cells.append(_row(row))
            conn.commit()
        return cells

    def live(self, goal_id: UUID, niche_ids: Sequence[str]) -> list[SearchCell]:
        """Every unexhausted cell for these niches, highest measured yield first."""
        with self._connect() as conn:
            rows = conn.execute(_SELECT_LIVE, (goal_id, list(niche_ids))).fetchall()
            conn.commit()
        return [_row(row) for row in rows]

    def record(self, cell: SearchCell, update: CellUpdate) -> SearchCell:
        """Write one measured pass to the row."""
        if cell.id is None:
            raise ValueError("cannot record a pass against a cell that was never persisted")
        with self._connect() as conn:
            row = conn.execute(
                _RECORD,
                (
                    update.next_page,
                    update.status,
                    update.new_yield,
                    update.total_yield,
                    update.last_searched_at,
                    update.exhausted_at,
                    cell.id,
                ),
            ).fetchone()
            conn.commit()
        if row is None:
            raise LookupError(f"no search cell with id {cell.id}")
        return _row(row)

    def revive_expired(self, goal_id: UUID, now: datetime) -> list[SearchCell]:
        """Reset every cell whose exhaustion has served its 30 days. Returns the revived."""
        with self._connect() as conn:
            rows = conn.execute(_REVIVE_EXPIRED, (goal_id, now)).fetchall()
            conn.commit()
        return [_row(row) for row in rows]
