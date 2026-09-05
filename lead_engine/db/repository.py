"""The durable task queue and the repository over the existing schema.

Synchronous psycopg3 on a `ConnectionPool`. Sync, not async, because the workers this
serves are CPU-cheap and IO-bound on one upstream at a time, and an async pool would buy
concurrency the queue already provides by handing different tasks to different processes.

The pool is injected. `Repository(pool)` takes one you built -- which is how the tests
point every statement at a throwaway schema without a single environment variable, and how
a future worker points at a read replica -- and `Repository.from_dsn(...)` is the
convenience for callers that just want a pool built for them.

Transactions
------------
Every method runs in its own transaction and commits, which is what a queue wants: a claim
that is not committed still holds its `FOR UPDATE` locks, and every other worker blocks
behind it.

When several writes have to land together, `Repository.transaction()` binds one connection
for the duration and every call inside the block joins it:

    with repo.transaction():
        run = repo.create_run(goal.id)
        repo.enqueue(run.id, "discover", payload, idem_key=key)
        repo.append_event(run.id, "run.started", {})

Anything raised inside rolls the whole block back. The binding is thread-local, so the
threads of a multi-worker process do not see each other's open transactions.

At-least-once, not exactly-once
-------------------------------
A lease can expire while its worker is still alive -- a long upstream call, a paused VM --
and the reaper will then hand the same task to somebody else. Both workers may finish and
both may call `complete`; the last write wins and the work was done twice. This is
inherent to a lease-based queue, and the honest fix belongs in the task handlers, which
must be idempotent, not in a fencing token here that would only narrow the window.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any

from psycopg import Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import queries
from .rows import (
    BusinessRow,
    ContactRow,
    EnrichmentRow,
    EventRow,
    GoalRow,
    RunRow,
    ScoreRow,
    TaskRow,
)


def _json(value: Any) -> Jsonb | None:
    """Wrap for a jsonb parameter, keeping None as SQL NULL.

    `Jsonb(None)` is the JSON value `null`, which is not the same thing as no value at all
    and would happily land in a nullable column as `'null'::jsonb`. Nothing downstream
    would notice until something asked `WHERE result IS NULL` and got the wrong answer.
    """
    return None if value is None else Jsonb(value)


class Repository:
    """Reads and writes for the control and data planes. One instance per process."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._owns_pool = False
        # Thread-local, so `transaction()` in one worker thread cannot capture statements
        # issued by another. A plain attribute here would silently interleave two threads'
        # writes into one transaction and roll back work that succeeded.
        self._local = threading.local()

    @classmethod
    def from_dsn(
        cls, dsn: str, *, min_size: int = 1, max_size: int = 8, **kwargs: Any
    ) -> Repository:
        """Build a pool for this DSN and own it, so `close()` disposes of it."""
        pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True, **kwargs)
        repository = cls(pool)
        repository._owns_pool = True
        return repository

    def close(self) -> None:
        """Close the pool, but only if this object made it. Injected pools are the
        caller's to close -- shutting one down here would break every other user of it."""
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> Repository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- plumbing -------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        bound = getattr(self._local, "connection", None)
        if bound is not None:
            # Inside `transaction()`: no commit here, the block owns the outcome.
            yield bound
            return
        with self._pool.connection() as connection:
            # psycopg's connection context manager commits on a clean exit and rolls back
            # on an exception, so one call is one transaction.
            yield connection

    @contextmanager
    def transaction(self) -> Iterator[Repository]:
        """Run a block of repository calls as one transaction."""
        if getattr(self._local, "connection", None) is not None:
            # Silently joining the outer transaction would make the inner block's `except`
            # look like it recovered, when in fact the outer commit still has to succeed.
            raise RuntimeError("Repository.transaction() does not nest")
        with self._pool.connection() as connection:
            self._local.connection = connection
            try:
                yield self
            finally:
                self._local.connection = None

    def _one(self, sql: str, params: dict[str, Any], row_class: type) -> Any:
        with self._connection() as connection, connection.cursor(
            row_factory=class_row(row_class)
        ) as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def _all(self, sql: str, params: dict[str, Any], row_class: type) -> list[Any]:
        with self._connection() as connection, connection.cursor(
            row_factory=class_row(row_class)
        ) as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    # --- queue ----------------------------------------------------------------------

    def enqueue(
        self,
        run_id: uuid.UUID,
        type: str,
        payload: dict[str, Any],
        *,
        priority: int = 100,
        idem_key: str,
        available_at: datetime | None = None,
    ) -> bool:
        """Add a task unless `idem_key` is already taken. True if this call inserted it.

        `idem_key` is required and has no default on purpose. A generated one -- a uuid, a
        hash of the payload plus a timestamp -- makes every enqueue unique and turns the
        whole mechanism off while still looking like it is on. The key has to name the
        *unit of work* ("discover:blr:salon:tile-7") for a resumed run to recognise what it
        already queued.
        """
        with self._connection() as connection:
            row = connection.execute(
                queries.ENQUEUE_TASK,
                {
                    "run_id": run_id,
                    "type": type,
                    "idem_key": idem_key,
                    "payload": Jsonb(payload),
                    "priority": priority,
                    "available_at": available_at,
                },
            ).fetchone()
        return row is not None

    def claim(
        self,
        worker_id: str,
        types: Sequence[str],
        batch: int = 1,
        lease: str = "60 seconds",
    ) -> list[TaskRow]:
        """Take up to `batch` claimable tasks of these types for `worker_id`.

        Returns fewer rows than asked for, or none at all, whenever other workers hold the
        rows this one would have taken -- `SKIP LOCKED` skips them rather than waiting. An
        empty list therefore means "nothing for me right now", never "the queue is empty".

        `lease` is any Postgres interval literal, and with `batch > 1` it must exceed the
        worst case for the WHOLE BATCH, not for one task.

        Every row in a batch is stamped from a single `now()`, evaluated once for the
        statement -- so the last task in a batch of five is already four tasks' worth of
        runtime into its lease before the worker looks at it. Size the lease per task and
        the reaper reclaims the tail of every batch while the worker is still working it,
        a second worker picks those tasks up, and each one is performed twice. For a
        discovery task that is two billed Google Maps searches for one cell, on an
        allowance that does not renew.

        Either size the lease as `batch x per-task worst case`, or call `renew_lease()` as
        each task begins, which is what a worker processing a batch serially should do.
        """
        return self._all(
            queries.CLAIM_TASKS,
            # A list, not a tuple: psycopg adapts a list to an array and a tuple to a
            # composite, and `= ANY(record)` is a type error at the far end.
            {"types": list(types), "batch": batch, "worker_id": worker_id, "lease": lease},
            TaskRow,
        )

    def renew_lease(
        self, task_id: int, worker_id: str, lease: str = "60 seconds"
    ) -> TaskRow | None:
        """Push this task's deadline out from now. Returns None if the caller no longer owns it.

        A worker handed a batch holds leases that all began at the same instant, so a serial
        worker should renew as each task starts -- otherwise the tail of every batch is
        reclaimed mid-flight and done twice.

        None means the reaper already took the task back, or another worker holds it. That is
        the signal to STOP, not to retry: whatever this worker was about to do, someone else
        is doing. Continuing is how one cell becomes two billed searches.
        """
        rows = self._all(
            queries.RENEW_LEASE,
            {"task_id": task_id, "worker_id": worker_id, "lease": lease},
            TaskRow,
        )
        return rows[0] if rows else None

    def complete(self, task_id: int, result: dict[str, Any] | None = None) -> TaskRow | None:
        """Mark a task done and record what it produced. None if no such task."""
        return self._one(
            queries.COMPLETE_TASK, {"task_id": task_id, "result": _json(result)}, TaskRow
        )

    def fail(self, task_id: int, error: str) -> TaskRow | None:
        """Record a failure and schedule the retry, or bury the task at `max_attempts`.

        Returns the row so the caller can see which happened: `status` is 'retry' with
        `available_at` pushed into the future, or 'dead' and never claimable again.
        """
        return self._one(queries.FAIL_TASK, {"task_id": task_id, "error": error}, TaskRow)

    def reap_expired_leases(self) -> list[TaskRow]:
        """Return every task whose lease ran out to the queue. Call it on a timer.

        This is the only thing that recovers work from a worker that died without saying
        so. Without it a killed process leaves its tasks 'running' forever and the run
        never finishes.
        """
        return self._all(queries.REAP_EXPIRED_LEASES, {}, TaskRow)

    def get_task(self, task_id: int) -> TaskRow | None:
        return self._one(queries.SELECT_TASK, {"task_id": task_id}, TaskRow)

    # --- control plane --------------------------------------------------------------

    def create_goal(
        self,
        name: str,
        spec: dict[str, Any],
        *,
        schedule: str | None = None,
        status: str = "active",
        goal_id: uuid.UUID | None = None,
    ) -> GoalRow:
        return self._one(
            queries.CREATE_GOAL,
            {
                "id": goal_id or uuid.uuid4(),
                "name": name,
                "spec": Jsonb(spec),
                "schedule": schedule,
                "status": status,
            },
            GoalRow,
        )

    def create_run(
        self,
        goal_id: uuid.UUID,
        *,
        trigger: str = "manual",
        status: str = "running",
        stats: dict[str, Any] | None = None,
        run_id: uuid.UUID | None = None,
    ) -> RunRow:
        return self._one(
            queries.CREATE_RUN,
            {
                "id": run_id or uuid.uuid4(),
                "goal_id": goal_id,
                "status": status,
                "trigger": trigger,
                "stats": Jsonb(stats or {}),
            },
            RunRow,
        )

    def finish_run(
        self, run_id: uuid.UUID, status: str, *, stats: dict[str, Any] | None = None
    ) -> RunRow | None:
        """Close a run. `stats` merges into whatever is already there."""
        return self._one(
            queries.FINISH_RUN,
            {"id": run_id, "status": status, "stats": _json(stats)},
            RunRow,
        )

    def append_event(
        self,
        run_id: uuid.UUID,
        type: str,
        payload: dict[str, Any],
        *,
        task_id: int | None = None,
    ) -> EventRow:
        """Append to the run's log. There is no update and no delete, by design.

        `seq` is assigned here, under an advisory lock on the run, because the schema does
        not default it and a replay that reads two events numbered 4 cannot tell which came
        first.
        """
        with self._connection() as connection, connection.cursor(
            row_factory=class_row(EventRow)
        ) as cursor:
            cursor.execute(queries.LOCK_RUN_EVENTS, {"run_id": run_id})
            cursor.execute(
                queries.APPEND_EVENT,
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "type": type,
                    "payload": Jsonb(payload),
                },
            )
            return cursor.fetchone()

    # --- data plane -----------------------------------------------------------------

    def upsert_business(
        self,
        *,
        name: str,
        niche_id: str,
        city: str,
        dedupe_key: str | None = None,
        business_id: uuid.UUID | None = None,
        place_id: str | None = None,
        country: str | None = None,
        state: str | None = None,
        search_area: str | None = None,
        address: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
        phone: str | None = None,
        email: str | None = None,
        website: str | None = None,
        instagram_handle: str | None = None,
        facebook_url: str | None = None,
    ) -> BusinessRow:
        """Insert, or refresh the row that already carries this `dedupe_key`.

        The key is computed by `lead_engine.dedupe` and arrives here ready. This layer does
        not import the pure core to derive one -- that is the dependency direction the
        package docstring promises -- and a caller that passes None gets a plain insert,
        because NULL keys do not collide.
        """
        return self._one(
            queries.UPSERT_BUSINESS,
            {
                "id": business_id or uuid.uuid4(),
                "place_id": place_id,
                "name": name,
                "niche_id": niche_id,
                "country": country,
                "state": state,
                "city": city,
                "search_area": search_area,
                "address": address,
                "lat": lat,
                "lng": lng,
                "phone": phone,
                "email": email,
                "website": website,
                "instagram_handle": instagram_handle,
                "facebook_url": facebook_url,
                "dedupe_key": dedupe_key,
            },
            BusinessRow,
        )

    def insert_enrichment(
        self,
        business_id: uuid.UUID,
        source: str,
        status: str,
        data: dict[str, Any],
        *,
        source_url: str | None = None,
        run_id: uuid.UUID | None = None,
    ) -> EnrichmentRow:
        """Record one fetch. Failures are recorded too -- `status` says which.

        A source that returned nothing is evidence about the business (no website to
        scrape, a dead Instagram handle) and the next run needs to know the attempt
        happened rather than trying it again from scratch.
        """
        return self._one(
            queries.INSERT_ENRICHMENT,
            {
                "business_id": business_id,
                "source": source,
                "status": status,
                "data": Jsonb(data),
                "source_url": source_url,
                "run_id": run_id,
            },
            EnrichmentRow,
        )

    def insert_contact(
        self,
        business_id: uuid.UUID,
        name: str,
        source: str,
        *,
        role: str = "unknown",
        phone: str | None = None,
        email: str | None = None,
        source_url: str | None = None,
        confidence: float = 0.5,
    ) -> ContactRow:
        """Record one named contact. Append-only, like `insert_enrichment`.

        A blank name is never passed here -- an extractor that could not find one records
        nothing at all, and this layer trusts that decision rather than re-deriving a rule
        that belongs to whichever module found the name.
        """
        return self._one(
            queries.INSERT_CONTACT,
            {
                "business_id": business_id,
                "name": name,
                "role": role,
                "phone": phone,
                "email": email,
                "source": source,
                "source_url": source_url,
                "confidence": confidence,
            },
            ContactRow,
        )

    def insert_score(
        self,
        business_id: uuid.UUID,
        scorer_version: str,
        *,
        total: int | None = None,
        demand: int | None = None,
        website_gap: int | None = None,
        budget: int | None = None,
        reachability: int | None = None,
        signals: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        audience_index: Decimal | float | None = None,
    ) -> ScoreRow:
        """Append a score. `scorer_version` names the pass, so two rows can disagree.

        `audience_index` stays None when nothing is known about the audience -- the banding
        view reads that as 'unknown', which is a different and more useful statement than
        the 0.0 that means "we looked and there is no audience".
        """
        return self._one(
            queries.INSERT_SCORE,
            {
                "business_id": business_id,
                "scorer_version": scorer_version,
                "total": total,
                "demand": demand,
                "website_gap": website_gap,
                "budget": budget,
                "reachability": reachability,
                "signals": Jsonb(signals or {}),
                "evidence": Jsonb(evidence or {}),
                "audience_index": audience_index,
            },
            ScoreRow,
        )
