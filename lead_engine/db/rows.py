"""One frozen dataclass per table this layer reads back.

These exist to be handed to psycopg's `class_row`, which builds a row as
`cls(**dict(zip(column_names, values)))`. That construction is unforgiving in a useful
way: every selected column must be a field, and every field must be selected. So none of
these dataclasses carries a default. A query that forgets a column raises `TypeError:
missing required argument` on the first row rather than quietly handing back an object
with a plausible-looking zero in it.

The same rule is why `lead_engine.db.queries` never writes `SELECT *`. Under a star, the
day someone adds a column to `tasks` is the day every claim in the system starts raising
`unexpected keyword argument` -- at runtime, in a worker, in production. The column lists
in `queries` are derived from `dataclasses.fields` of these classes, so the two cannot
drift; `tests/test_queue.py` then checks those field names against
`information_schema.columns` so that a migration adding a column fails a test instead of
a worker.

Field order mirrors the DDL in `migrations/0002_control.sql` and `0003_data.sql`. Nothing
depends on the order -- columns are named explicitly everywhere -- but a reader comparing
this file against the migration should not have to hunt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

# jsonb comes back from psycopg already parsed. Every jsonb column in this schema holds an
# object at the top level, so `dict` is the honest annotation; nothing enforces it.
Json = dict[str, Any]


@dataclass(frozen=True)
class GoalRow:
    id: UUID
    name: str
    spec: Json
    schedule: str | None
    status: str
    created_at: datetime


@dataclass(frozen=True)
class RunRow:
    id: UUID
    goal_id: UUID
    status: str
    trigger: str
    stats: Json
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class TaskRow:
    """A row of `tasks`, which is the queue.

    `attempts` counts *claims*, not failures -- see `Repository.claim`. `lease_expires` is
    the only mutable-in-place field the control plane has, and `result`/`error` are the
    terminal record of what the worker did with the task.
    """

    id: int
    run_id: UUID
    type: str
    idem_key: str
    payload: Json
    status: str
    priority: int
    available_at: datetime
    attempts: int
    max_attempts: int
    locked_by: str | None
    lease_expires: datetime | None
    result: Json | None
    error: str | None


@dataclass(frozen=True)
class EventRow:
    """A row of the append-only `events` log.

    Not in the module's original brief, but `append_event` has to return something and an
    opaque integer is a worse answer than the row it just wrote -- a replay needs `seq`,
    and `seq` is assigned by the database, not by the caller.
    """

    id: int
    run_id: UUID
    task_id: int | None
    seq: int
    type: str
    payload: Json
    created_at: datetime


@dataclass(frozen=True)
class BusinessRow:
    id: UUID
    place_id: str | None
    name: str
    niche_id: str
    country: str | None
    state: str | None
    city: str
    search_area: str | None
    address: str | None
    lat: float | None
    lng: float | None
    phone: str | None
    email: str | None
    website: str | None
    instagram_handle: str | None
    facebook_url: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    dedupe_key: str | None


@dataclass(frozen=True)
class EnrichmentRow:
    id: int
    business_id: UUID
    source: str
    status: str
    data: Json
    source_url: str | None
    fetched_at: datetime
    run_id: UUID | None


@dataclass(frozen=True)
class ContactRow:
    """A row of `contacts` -- a named PERSON, distinct from `businesses.email`'s general inbox.

    Append-only, like `EnrichmentRow`: a page that names two owners over two different
    fetches is two facts, not one overwritten by the other. `role` is `owner|manager|
    marketing|unknown` and `confidence` is `real` in Postgres, hence `float` here.
    """

    id: int
    business_id: UUID
    name: str | None
    role: str | None
    phone: str | None
    email: str | None
    source: str
    source_url: str | None
    confidence: float
    found_at: datetime


@dataclass(frozen=True)
class AutomationOpportunityRow:
    """A row of `automation_opportunities` (0005) -- one firing offer for one business.

    Unique on `(business_id, opportunity_id)`, so re-detecting the same offer on a later
    enrichment pass is a refresh of this row, not a second one -- see
    `queries.UPSERT_AUTOMATION_OPPORTUNITY`.
    """

    id: int
    business_id: UUID
    opportunity_id: str
    confidence: float
    trigger_signals: Json
    evidence: Json
    detected_at: datetime


@dataclass(frozen=True)
class OutreachRow:
    """A row of `outreach` (0003) -- one piece of generated pitch prose.

    Append-only, like `EnrichmentRow`: a regenerated pitch is a new row, never an overwrite
    of text that may already have been sent to the business.
    """

    id: int
    business_id: UUID
    kind: str
    channel: str
    body: str
    evidence: Json
    created_at: datetime


@dataclass(frozen=True)
class ScoreRow:
    """A row of `scores`.

    `audience_index` is `numeric` in Postgres and therefore `Decimal` here, not `float`.
    Tests that compare it against a float must say so explicitly; silently widening it to
    a float in the mapping layer would hide that the database rounds differently.
    """

    id: int
    business_id: UUID
    scorer_version: str
    total: int | None
    demand: int | None
    website_gap: int | None
    budget: int | None
    reachability: int | None
    signals: Json
    evidence: Json
    scored_at: datetime
    audience_index: Decimal | None
