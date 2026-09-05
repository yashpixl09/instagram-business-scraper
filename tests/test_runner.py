"""The queue worker: what it does with one task, and what it must never do twice.

`workers/runner.py` is the only process that spends money. Everything it gets wrong costs a
non-renewing credit or a lead nobody ever hears about, so the tests here are organised
around the four decisions it actually makes:

  * **perform, or hand back?** -- the lease is renewed as each task BEGINS, and a `None`
    from `renew_lease` means the reaper already gave this task to somebody else. The
    handler must not run. `LeaseIsTheRightToSpend` pins that, and pins the far side too:
    `complete()` and `fail()` returning `None` are dropped, never retried, because a retry
    is a second worker's result written over the first one's.
  * **which of the three stops is this?** -- `BudgetExhausted` is a completion and a
    `stopped` run, `ProviderError` is a failure with the schema's backoff, and anything
    else is a bug whose message is never allowed near `tasks.error`.
    `TheThreeWaysAHandlerStops` and `NothingUpstreamReachesTheRow` cover those.
  * **what did this worker actually do?** -- every claim, completion, failure, abandonment
    and loss appends to `events`. A task that vanished into a worker with no trace is a run
    nobody can explain.
  * **what happens after a crash?** -- `ResumeAfterACrash` is the spec's Resume row
    (`docs/superpowers/specs/2026-08-13-lead-engine-design.md`): kill a run mid-flight,
    restart, and assert the completed tasks are skipped and no work repeats. It is asserted
    as a count of billed searches, not as "roughly right", because the difference between
    one and two is the whole point of the lease.

WHY MOST OF THIS RUNS WITHOUT POSTGRES
`Worker` takes the `TaskQueue` protocol, not a `Repository`, and the module docstring says
outright that this is so a unit test can exercise every branch with no database anywhere
near it. `FakeQueue` below is that queue, and every guard in it is copied from
`lead_engine/db/queries.py` -- the status guard on `COMPLETE_TASK` and `FAIL_TASK`, the
ownership guard on `RENEW_LEASE`, the `attempts >= max_attempts` death rule, the backoff.
It is a stand-in for Postgres's *answers*, not for Postgres: whether Postgres really gives
those answers under concurrency is `tests/test_queue.py`'s claim, against a real database,
and nothing here is allowed to imply it.

`WorkerAgainstPostgres` at the bottom is the part that cannot be faked -- a real reaper
returning a real expired lease to a real queue, with the worker loop on top of it. It
skips unless `LEAD_ENGINE_TEST_DSN` is set, exactly as `tests/test_queue.py` does.

NOTHING HERE TOUCHES A NETWORK, A BROWSER OR A REAL PROVIDER. The handlers are functions
that return dicts or raise.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import threading
import types
import unittest
import uuid
from collections import Counter
from contextlib import contextmanager, redirect_stderr
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pytest

from lead_engine.db.rows import TaskRow
from lead_engine.providers.budget import BudgetExhausted
from lead_engine.providers.errors import ProviderError, redact
from lead_engine.workers import runner
from lead_engine.workers.runner import (
    ABANDONED,
    BUDGET_EXHAUSTED,
    BUDGET_STOPPED,
    CLAIMED,
    COMPLETED,
    DEFAULT_LEASE,
    DEFAULT_POLL_INTERVAL,
    EXIT_CANNOT_START,
    EXIT_OK,
    FAILED,
    LOST,
    STOPPED,
    Worker,
    _worker_id,
    build_parser,
    default_worker_id,
    provider_failure,
    redacted_failure,
    redacted_traceback,
    requested_types,
    run_workers,
    stop_on_signals,
)

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db.migrate import apply_migrations
    from lead_engine.db.repository import Repository
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

DISCOVER = "discover"
ENRICH = "enrich"

RUN = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
NOW = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)

#: A URL with a key in the query string -- what every provider in this project produces, and
#: what an httpx exception carries in `.request.url`. Used as the payload of a deliberately
#: leaky handler bug; the assertion is that none of it reaches `tasks.error`.
LEAKY_URL = "https://www.searchapi.io/api/v1/search?api_key=sk-live-9f3c2a7b4e1d8c60&q=cafe"
LEAKED_KEY = "sk-live-9f3c2a7b4e1d8c60"


# --- the fake queue -------------------------------------------------------------------------


_INTERVAL = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*(seconds?|minutes?)\s*")


def interval(lease: str) -> timedelta:
    """The sliver of Postgres interval syntax the worker actually passes.

    Deliberately narrow. A lease literal this cannot parse raises here rather than being
    silently treated as zero, because a test that thinks it granted two minutes and in fact
    granted nothing would prove the opposite of what it claims.
    """
    match = _INTERVAL.fullmatch(lease)
    if match is None:
        raise ValueError(f"the fake queue cannot parse the interval {lease!r}")
    amount = float(match.group(1))
    return timedelta(minutes=amount) if match.group(2).startswith("minute") else timedelta(
        seconds=amount
    )


class FakeQueue:
    """`Repository`'s six queue methods in memory, with the guards its SQL has.

    Every rule below is transcribed from `lead_engine/db/queries.py`:

      * `claim`  takes only 'pending'/'retry' rows of the requested types whose
        `available_at` has arrived, ordered by (priority, id), and stamps every row in the
        batch from ONE clock reading -- which is the whole reason the worker renews.
      * `renew_lease` matches on `locked_by` AND `status = 'running'`.
      * `complete`/`fail` match on `status = 'running'`, so a worker that lost the row gets
        `None` rather than overwriting whoever holds it now.
      * `fail`/`reap` bury the task at `attempts >= max_attempts` and otherwise schedule
        `now() + 2^attempts` seconds.

    The clock is virtual: `advance()` moves it. Nothing in this file sleeps.
    """

    def __init__(self, clock: datetime = NOW) -> None:
        self.clock = clock
        self.tasks: dict[int, TaskRow] = {}
        self.events: list[SimpleNamespace] = []
        self.runs: dict[uuid.UUID, SimpleNamespace] = {}
        #: Ordered record of everything that happened, queue calls and handler calls alike.
        #: Sequencing assertions ("renewed before the handler ran") read this.
        self.trace: list[str] = []
        self.reaps = 0
        self._lock = threading.RLock()
        self._next_id = 1
        self._seq: dict[uuid.UUID, int] = {}

    # --- fixtures ---------------------------------------------------------------------

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.clock += timedelta(seconds=seconds)

    def enqueue(
        self,
        type: str = DISCOVER,
        *,
        run_id: uuid.UUID = RUN,
        payload: dict | None = None,
        idem_key: str | None = None,
        priority: int = 100,
        max_attempts: int = 3,
    ) -> TaskRow:
        with self._lock:
            row = TaskRow(
                id=self._next_id,
                run_id=run_id,
                type=type,
                idem_key=idem_key or f"{type}:{self._next_id}",
                payload=payload or {},
                status="pending",
                priority=priority,
                available_at=self.clock,
                attempts=0,
                max_attempts=max_attempts,
                locked_by=None,
                lease_expires=None,
                result=None,
                error=None,
            )
            self._next_id += 1
            self.tasks[row.id] = row
            return row

    def enqueue_many(self, count: int, type: str = DISCOVER, **kwargs) -> list[TaskRow]:
        return [self.enqueue(type, **kwargs) for _ in range(count)]

    # --- the protocol -----------------------------------------------------------------

    def claim(
        self, worker_id: str, types, batch: int = 1, lease: str = "60 seconds"
    ) -> list[TaskRow]:
        with self._lock:
            self.trace.append(f"claim:{','.join(types)}")
            wanted = set(types)
            deadline = self.clock + interval(lease)
            claimable = [
                task
                for task in sorted(self.tasks.values(), key=lambda t: (t.priority, t.id))
                if task.status in ("pending", "retry")
                and task.type in wanted
                and task.available_at <= self.clock
            ]
            taken = []
            for task in claimable[:batch]:
                row = replace(
                    task,
                    status="running",
                    locked_by=worker_id,
                    lease_expires=deadline,
                    attempts=task.attempts + 1,
                )
                self.tasks[row.id] = row
                taken.append(row)
            return taken

    def renew_lease(self, task_id: int, worker_id: str, lease: str = "60 seconds"):
        with self._lock:
            self.trace.append(f"renew:{task_id}")
            task = self.tasks.get(task_id)
            if task is None or task.status != "running" or task.locked_by != worker_id:
                return None
            row = replace(task, lease_expires=self.clock + interval(lease))
            self.tasks[row.id] = row
            return row

    def complete(self, task_id: int, result: dict | None = None):
        with self._lock:
            self.trace.append(f"complete:{task_id}")
            task = self.tasks.get(task_id)
            if task is None or task.status != "running":
                return None
            row = replace(
                task,
                status="done",
                result=result,
                error=None,
                locked_by=None,
                lease_expires=None,
            )
            self.tasks[row.id] = row
            return row

    def fail(self, task_id: int, error: str):
        with self._lock:
            self.trace.append(f"fail:{task_id}")
            task = self.tasks.get(task_id)
            if task is None or task.status != "running":
                return None
            self.tasks[task_id] = self._failed(task, error)
            return self.tasks[task_id]

    def reap_expired_leases(self) -> list[TaskRow]:
        with self._lock:
            self.trace.append("reap")
            self.reaps += 1
            reaped = []
            for task in list(self.tasks.values()):
                if task.status == "running" and task.lease_expires is not None:
                    if task.lease_expires < self.clock:
                        row = self._failed(task, "lease expired")
                        self.tasks[row.id] = row
                        reaped.append(row)
            return reaped

    def append_event(self, run_id: uuid.UUID, type: str, payload: dict, *, task_id=None):
        with self._lock:
            self._seq[run_id] = self._seq.get(run_id, 0) + 1
            event = SimpleNamespace(
                run_id=run_id,
                task_id=task_id,
                seq=self._seq[run_id],
                type=type,
                payload=payload,
            )
            self.events.append(event)
            return event

    def finish_run(self, run_id: uuid.UUID, status: str, *, stats: dict | None = None):
        with self._lock:
            self.trace.append(f"finish_run:{status}")
            run = SimpleNamespace(id=run_id, status=status, stats=dict(stats or {}))
            self.runs[run_id] = run
            return run

    # --- reading state back -----------------------------------------------------------

    def _failed(self, task: TaskRow, error: str) -> TaskRow:
        status = "dead" if task.attempts >= task.max_attempts else "retry"
        backoff = min(2**task.attempts, 3600)
        return replace(
            task,
            status=status,
            error=error,
            locked_by=None,
            lease_expires=None,
            available_at=self.clock + timedelta(seconds=backoff),
        )

    def status_of(self, task_id: int) -> str:
        return self.tasks[task_id].status

    def statuses(self) -> list[str]:
        return [self.tasks[key].status for key in sorted(self.tasks)]

    def event_types(self, task_id: int | None = None) -> list[str]:
        return [
            event.type
            for event in self.events
            if task_id is None or event.task_id == task_id
        ]

    def event(self, type: str) -> SimpleNamespace:
        matching = [event for event in self.events if event.type == type]
        if not matching:
            raise AssertionError(f"no {type} event was written; got {self.event_types()}")
        return matching[0]


# --- handlers -------------------------------------------------------------------------------


class RecordingHandler:
    """A task handler that remembers every task it was handed.

    `performed` is the assertion in half this file: a task that appears in it twice was
    performed twice, and for a discovery task that is a second billed search.
    """

    def __init__(self, result: dict | None = None, queue: FakeQueue | None = None) -> None:
        self.result = result
        self.queue = queue
        self.performed: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, task: TaskRow) -> dict | None:
        with self._lock:
            self.performed.append(task.id)
        if self.queue is not None:
            self.queue.trace.append(f"handle:{task.id}")
        return self.result


class RaisingHandler(RecordingHandler):
    """Records the task, then raises. `raises` may be an exception or a callable."""

    def __init__(self, exception, queue: FakeQueue | None = None) -> None:
        super().__init__(queue=queue)
        self.exception = exception

    def __call__(self, task: TaskRow):
        super().__call__(task)
        raise self.exception() if callable(self.exception) else self.exception


class Ledger:
    """The fifty non-renewing credits, counted. One handler call is one billed search."""

    def __init__(self, allowance: int = 50) -> None:
        self.allowance = allowance
        self.used = 0
        self._lock = threading.Lock()

    def spend(self, n: int = 1) -> None:
        with self._lock:
            if self.used + n > self.allowance:
                raise BudgetExhausted("searchapi", self.allowance, self.used, n)
            self.used += n


def exhausted(used: int = 50, allowance: int = 50) -> BudgetExhausted:
    return BudgetExhausted("searchapi", allowance, used, 1, purpose="discover")


class WorkerTestCase(unittest.TestCase):
    """A fake queue, and a worker over it that never sleeps and never signals."""

    def setUp(self):
        self.queue = FakeQueue()
        self.waits: list[float] = []

    def worker(self, handlers=None, **kwargs) -> Worker:
        handlers = handlers if handlers is not None else {DISCOVER: RecordingHandler({"ok": 1})}
        kwargs.setdefault("worker_id", "worker-a")
        kwargs.setdefault("sleeper", self.waits.append)
        return Worker(self.queue, handlers, **kwargs)


# --- construction -----------------------------------------------------------------------------


class Construction(WorkerTestCase):
    def test_a_worker_with_no_handlers_is_refused(self):
        # It would claim work it cannot perform, hold it under a lease for nothing, and hand
        # it back one attempt poorer when the lease expired.
        with self.assertRaises(ValueError):
            Worker(self.queue, {})

    def test_a_type_with_no_handler_is_refused_and_named(self):
        with self.assertRaises(ValueError) as caught:
            Worker(self.queue, {DISCOVER: RecordingHandler()}, types=[DISCOVER, ENRICH])
        self.assertIn(ENRICH, str(caught.exception))

    def test_types_default_to_the_handler_names(self):
        worker = self.worker({ENRICH: RecordingHandler(), DISCOVER: RecordingHandler()})
        # Sorted, so two workers built from the same dict claim in the same order whatever
        # order the mapping happened to be in.
        self.assertEqual(worker.types, (DISCOVER, ENRICH))

    def test_a_batch_below_one_is_refused(self):
        for batch in (0, -1):
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError):
                    self.worker(batch=batch)

    def test_a_worker_id_is_generated_when_none_is_given(self):
        worker = Worker(self.queue, {DISCOVER: RecordingHandler()})
        self.assertTrue(worker.worker_id)
        self.assertIn(str(os.getpid()), worker.worker_id)

    def test_two_generated_ids_in_one_process_differ(self):
        # The random tail is the point: a container restarted onto the same host with the
        # same pid would otherwise inherit its predecessor's right to complete tasks the
        # reaper has already handed to somebody else.
        self.assertNotEqual(default_worker_id(), default_worker_id())
        self.assertEqual(len(default_worker_id().split(":")), 3)

    def test_a_stopped_type_is_no_longer_claimable(self):
        worker = self.worker({DISCOVER: RecordingHandler(), ENRICH: RecordingHandler()})
        worker.stopped_types.add(DISCOVER)
        self.assertEqual(worker.claimable_types, (ENRICH,))

    def test_stop_is_visible_through_the_property(self):
        worker = self.worker()
        self.assertFalse(worker.stopping)
        worker.stop()
        self.assertTrue(worker.stopping)


# --- one task, end to end -----------------------------------------------------------------


class OneTask(WorkerTestCase):
    def test_the_lease_is_renewed_before_the_handler_runs(self):
        # THE ordering property. The batch was stamped from one clock reading, so pushing
        # the deadline out from now is the only thing that stops the tail of a batch being
        # reaped mid-flight and performed a second time by somebody else.
        task = self.queue.enqueue()
        handler = RecordingHandler({"leads": 3}, queue=self.queue)

        self.assertEqual(self.worker({DISCOVER: handler}).run_once(), 1)

        self.assertEqual(
            self.queue.trace,
            [f"claim:{DISCOVER}", f"renew:{task.id}", f"handle:{task.id}", f"complete:{task.id}"],
        )

    def test_the_handlers_result_is_what_lands_in_the_row(self):
        task = self.queue.enqueue()
        self.worker({DISCOVER: RecordingHandler({"leads": 12})}).run_once()

        self.assertEqual(self.queue.status_of(task.id), "done")
        self.assertEqual(self.queue.tasks[task.id].result, {"leads": 12})
        self.assertIsNone(self.queue.tasks[task.id].locked_by)

    def test_a_claim_and_a_completion_are_both_recorded(self):
        # A task that vanished into a worker with no trace is a run nobody can explain.
        task = self.queue.enqueue()
        self.worker({DISCOVER: RecordingHandler({"leads": 12})}).run_once()

        self.assertEqual(self.queue.event_types(task.id), [CLAIMED, COMPLETED])
        claimed = self.queue.event(CLAIMED)
        self.assertEqual(claimed.run_id, RUN)
        self.assertEqual(claimed.task_id, task.id)
        self.assertEqual(claimed.payload["worker"], "worker-a")
        # The attempt this delivery is, as the claim counted it.
        self.assertEqual(claimed.payload["attempt"], 1)
        completed = self.queue.event(COMPLETED)
        self.assertEqual(completed.payload["result"], {"leads": 12})
        self.assertEqual(completed.payload["attempts"], 1)
        self.assertEqual(completed.payload["type"], DISCOVER)

    def test_a_handler_that_returns_nothing_still_completes(self):
        task = self.queue.enqueue()
        self.worker({DISCOVER: RecordingHandler(None)}).run_once()

        self.assertEqual(self.queue.status_of(task.id), "done")
        self.assertIsNone(self.queue.tasks[task.id].result)
        # The event carries {} rather than null, so a replay reading `result` never has to
        # decide what None meant.
        self.assertEqual(self.queue.event(COMPLETED).payload["result"], {})

    def test_run_once_counts_the_tasks_it_performed(self):
        self.queue.enqueue_many(3)
        handler = RecordingHandler({"ok": True})

        self.assertEqual(self.worker({DISCOVER: handler}, batch=3).run_once(), 3)
        self.assertEqual(handler.performed, [1, 2, 3])
        self.assertEqual(self.queue.statuses(), ["done", "done", "done"])

    def test_an_empty_queue_performs_nothing_and_writes_nothing(self):
        handler = RecordingHandler()
        self.assertEqual(self.worker({DISCOVER: handler}).run_once(), 0)
        self.assertEqual(handler.performed, [])
        self.assertEqual(self.queue.events, [])

    def test_only_the_types_this_worker_serves_are_claimed(self):
        discover = self.queue.enqueue(DISCOVER)
        enrich = self.queue.enqueue(ENRICH)
        handler = RecordingHandler()

        self.worker({DISCOVER: handler}, batch=5).run_once()

        self.assertEqual(handler.performed, [discover.id])
        self.assertEqual(self.queue.status_of(enrich.id), "pending")

    def test_a_worker_whose_every_type_has_stopped_does_not_even_claim(self):
        self.queue.enqueue()
        worker = self.worker()
        worker.stopped_types.add(DISCOVER)

        self.assertEqual(worker.run_once(), 0)
        # Not "claimed and released": never claimed. A claim would have spent an attempt.
        self.assertEqual(self.queue.trace, [])
        self.assertEqual(self.queue.status_of(1), "pending")

    def test_the_lease_the_worker_was_built_with_reaches_the_queue(self):
        self.queue.enqueue()
        self.worker(lease="90 seconds").run_once()

        # Renewed from now, which is what the fake stamps: 90 seconds past the virtual clock.
        self.assertEqual(self.queue.tasks[1].result, {"ok": 1})
        self.assertIn("renew:1", self.queue.trace)


# --- the lease --------------------------------------------------------------------------------


class LeaseIsTheRightToSpend(WorkerTestCase):
    """`renew_lease` returning None is a stop signal, not a hiccup.

    Whatever the task was about to buy, somebody else is buying it. Continuing "because we
    were nearly there anyway" is precisely how one cell becomes two billed searches.
    """

    def test_a_task_reclaimed_before_it_begins_is_never_handed_to_the_handler(self):
        task = self.queue.enqueue()
        handler = RecordingHandler({"leads": 9})
        worker = self.worker({DISCOVER: handler})
        claimed = self.queue.claim("worker-a", [DISCOVER])
        self.assertEqual(len(claimed), 1)
        # The reaper took it back and worker-b now holds it, exactly as an expired lease
        # leaves things. worker-a's renew will not match.
        self.queue.tasks[task.id] = replace(self.queue.tasks[task.id], locked_by="worker-b")
        self.queue.trace.clear()

        # Hand the stale row straight to _perform through the public loop: the claim above
        # is what run_once would have done.
        self.assertFalse(worker._perform(claimed[0]))

        self.assertEqual(handler.performed, [])
        self.assertEqual(self.queue.event_types(task.id), [CLAIMED, ABANDONED])
        self.assertEqual(self.queue.event(ABANDONED).payload["reason"], "lease_lost")
        # Not failed and not completed: nothing went wrong and nothing was done.
        self.assertEqual(self.queue.status_of(task.id), "running")
        self.assertNotIn(f"fail:{task.id}", self.queue.trace)
        self.assertNotIn(f"complete:{task.id}", self.queue.trace)

    def test_an_abandoned_task_is_not_counted_as_performed(self):
        self.queue.enqueue()
        worker = self.worker({DISCOVER: RecordingHandler()})
        self.queue.claim("someone-else", [DISCOVER])
        # The queue hands worker-a a row it no longer owns -- what a claim that raced the
        # reaper returns. run_once must report 0 tasks performed, not 1.
        with mock.patch.object(
            self.queue, "claim", return_value=[self.queue.tasks[1]]
        ):
            self.assertEqual(worker.run_once(), 0)

    def reaped_mid_handler(self, task_id: int) -> None:
        """What a slow handler comes back to: its lease ran out and the reaper took the row.

        The renew at the start of the task is what normally prevents this; a handler that
        outruns even the renewed lease reaches `complete()`/`fail()` holding nothing.
        """
        self.queue.advance(seconds=600)
        self.assertEqual([task.id for task in self.queue.reap_expired_leases()], [task_id])

    def test_a_row_lost_before_the_completion_lands_is_dropped_not_retried(self):
        # Recording the result anyway would overwrite whoever holds the task now; retrying
        # would perform it a second time.
        task = self.queue.enqueue()

        def handler(row):
            self.reaped_mid_handler(row.id)
            return {"leads": 4}

        with self.assertLogs("lead_engine.workers", level="WARNING") as logs:
            self.assertEqual(self.worker({DISCOVER: handler}).run_once(), 1)

        self.assertEqual(self.queue.event_types(task.id)[-1], LOST)
        self.assertEqual(self.queue.event(LOST).payload["phase"], "complete")
        self.assertIn("dropping the outcome without retrying", "\n".join(logs.output))
        # Back in the queue for whoever claims it next, with this worker's result nowhere.
        self.assertEqual(self.queue.status_of(task.id), "retry")
        self.assertIsNone(self.queue.tasks[task.id].result)

    def test_a_row_lost_before_the_failure_lands_is_dropped_too(self):
        task = self.queue.enqueue()

        def handler(row):
            self.reaped_mid_handler(row.id)
            raise ProviderError("provider_unavailable")

        with self.assertLogs("lead_engine.workers", level="WARNING"):
            self.worker({DISCOVER: handler}).run_once()

        self.assertEqual(self.queue.event(LOST).payload["phase"], "fail")
        # The reaper already spent the attempt and set its own error. A second failure
        # written on top would take this task one step closer to `dead` for one incident.
        self.assertEqual(self.queue.status_of(task.id), "retry")
        self.assertEqual(self.queue.tasks[task.id].error, "lease expired")
        self.assertEqual(self.queue.tasks[task.id].attempts, 1)

    def test_the_tail_of_a_batch_is_left_to_the_reaper_when_the_worker_is_stopping(self):
        self.queue.enqueue_many(3)
        handler = RecordingHandler({"ok": True})
        worker = self.worker({DISCOVER: handler}, batch=3)

        class StoppingHandler(RecordingHandler):
            def __call__(inner, task):  # noqa: N805 - a nested handler, not a method
                result = super().__call__(task)
                worker.stop()
                return result

        worker.handlers[DISCOVER] = stopping = StoppingHandler({"ok": True})
        self.assertEqual(worker.run_once(), 1)

        self.assertEqual(stopping.performed, [1])
        # There is no un-claim. Failing them would spend an attempt each for nothing;
        # completing them would mark cells searched that nobody searched.
        self.assertEqual(self.queue.statuses(), ["done", "running", "running"])
        self.assertEqual(self.queue.event_types(2), [])
        self.assertEqual(self.queue.event_types(3), [])


# --- the three ways a handler stops -----------------------------------------------------------


class TheThreeWaysAHandlerStops(WorkerTestCase):
    def test_a_retryable_provider_error_fails_the_task_with_its_code(self):
        task = self.queue.enqueue()
        error = ProviderError("provider_rate_limited")
        self.worker({DISCOVER: RaisingHandler(error)}).run_once()

        self.assertEqual(self.queue.status_of(task.id), "retry")
        self.assertEqual(
            self.queue.tasks[task.id].error,
            "provider_rate_limited: The provider rate limit was reached. Try again later.",
        )
        failed = self.queue.event(FAILED)
        self.assertEqual(failed.payload["status"], "retry")
        self.assertEqual(failed.payload["attempts"], 1)
        # The backoff is the schema's, computed from attempts. 2 seconds after the first.
        self.assertEqual(
            self.queue.tasks[task.id].available_at - self.queue.clock, timedelta(seconds=2)
        )

    def test_a_non_retryable_provider_error_fails_too_but_says_it_is_pointless(self):
        # There is no state between retry and dead, so inventing "parked" would mean a task
        # nobody ever looks at. What changes is the message an operator reads.
        for code in ("provider_auth_failed", "location_not_found"):
            with self.subTest(code=code):
                queue = FakeQueue()
                task = queue.enqueue()
                worker = Worker(
                    queue,
                    {DISCOVER: RaisingHandler(ProviderError(code))},
                    worker_id="worker-a",
                )
                worker.run_once()

                self.assertEqual(queue.status_of(task.id), "retry")
                self.assertIn("will not succeed on retry", queue.tasks[task.id].error)
                self.assertTrue(queue.tasks[task.id].error.startswith(f"{code}: "))

    def test_the_last_attempt_is_death_not_another_retry(self):
        # The queue owns that decision, in one statement, so two workers cannot disagree
        # about it -- the worker only reports what the row came back as.
        task = self.queue.enqueue(max_attempts=1)
        self.worker({DISCOVER: RaisingHandler(ProviderError("provider_unavailable"))}).run_once()

        self.assertEqual(self.queue.status_of(task.id), "dead")
        self.assertEqual(self.queue.event(FAILED).payload["status"], "dead")

    def test_a_handler_bug_does_not_take_the_worker_down_with_it(self):
        self.queue.enqueue_many(3)
        calls: list[int] = []

        def handler(task):
            calls.append(task.id)
            if task.id == 2:
                raise ZeroDivisionError("a bug in the handler")
            return {"ok": True}

        self.assertEqual(self.worker({DISCOVER: handler}, batch=3).run_once(), 3)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(self.queue.statuses(), ["done", "retry", "done"])

    def test_a_base_exception_is_not_swallowed(self):
        # KeyboardInterrupt is not a handler bug, and catching it here would make Ctrl-C
        # look like a failed task. It leaves the row running, and the reaper returns it.
        task = self.queue.enqueue()
        worker = self.worker({DISCOVER: RaisingHandler(KeyboardInterrupt)})

        with self.assertRaises(KeyboardInterrupt):
            worker.run_once()

        self.assertEqual(self.queue.status_of(task.id), "running")
        self.assertEqual(self.queue.event_types(task.id), [CLAIMED])


class NothingUpstreamReachesTheRow(WorkerTestCase):
    """`tasks.error` is served by the API and pasted into bug reports. It gets a class name."""

    def test_an_unhandled_exceptions_message_never_reaches_the_row(self):
        exception = RuntimeError(f"GET {LEAKY_URL} failed with 500")
        # The fixture is a genuine leak vector, not a straw man: redact() would change it.
        self.assertNotEqual(redact(str(exception)), str(exception))
        task = self.queue.enqueue()

        with self.assertLogs("lead_engine.workers", level="ERROR"):
            self.worker({DISCOVER: RaisingHandler(exception)}).run_once()

        stored = self.queue.tasks[task.id].error
        self.assertIn("unhandled RuntimeError", stored)
        self.assertNotIn(LEAKED_KEY, stored)
        self.assertNotIn("searchapi.io", stored)
        self.assertNotIn("500", stored)
        # And the same text is what the event carries, since that is read by the same API.
        self.assertEqual(self.queue.event(FAILED).payload["error"], stored)

    def test_the_traceback_that_reaches_the_log_is_redacted(self):
        task = self.queue.enqueue()
        exception = RuntimeError(f"GET {LEAKY_URL} failed with 500")

        with self.assertLogs("lead_engine.workers", level="ERROR") as logs:
            self.worker({DISCOVER: RaisingHandler(exception)}).run_once()

        written = "\n".join(logs.output)
        self.assertNotIn(LEAKED_KEY, written)
        self.assertIn("[redacted]", written)
        # The detail still has to go somewhere or nothing is debuggable.
        self.assertIn("RuntimeError", written)
        self.assertIn(str(task.id), written)

    def test_redacted_failure_says_where_the_detail_went(self):
        text = redacted_failure(ValueError("anything at all"))
        self.assertIn("unhandled ValueError", text)
        self.assertNotIn("anything at all", text)
        self.assertIn("worker log", text)

    def test_redacted_traceback_keeps_the_frames_and_drops_the_credentials(self):
        try:
            raise ProviderError("provider_bad_response", f"upstream said {LEAKY_URL}")
        except ProviderError as exc:
            text = redacted_traceback(exc)
        self.assertIn("test_redacted_traceback_keeps_the_frames", text)
        self.assertNotIn(LEAKED_KEY, text)

    def test_provider_failure_is_redacted_on_the_way_out(self):
        # ProviderError already refuses a message redact() would change, so this is the
        # second pass -- the last place before the text becomes a row the API serves back.
        error = ProviderError("provider_bad_response", f"upstream said {LEAKY_URL}")
        text = provider_failure(error)
        self.assertNotIn(LEAKED_KEY, text)
        self.assertEqual(redact(text), text)
        self.assertTrue(text.startswith("provider_bad_response: "))


# --- the allowance running out ----------------------------------------------------------------


class BudgetExhaustionIsACleanStop(WorkerTestCase):
    """`BudgetExhausted` is the budget working, not the budget breaking.

    The ledger refused before the request went out, so nothing was billed and nothing broke.
    It must not become a `dead` task, a `failed` run, or an ERROR log line.
    """

    def setUp(self):
        super().setUp()
        self.task = self.queue.enqueue()
        self.handler = RaisingHandler(exhausted(used=50), queue=self.queue)

    def run_it(self, **kwargs) -> Worker:
        worker = self.worker({DISCOVER: self.handler}, **kwargs)
        worker.run_once()
        return worker

    def test_the_task_is_completed_not_failed(self):
        self.run_it()
        self.assertEqual(self.queue.status_of(self.task.id), "done")
        self.assertNotIn(f"fail:{self.task.id}", self.queue.trace)
        self.assertNotIn(FAILED, self.queue.event_types())

    def test_the_result_names_the_allowance_that_ran_out(self):
        self.run_it()
        self.assertEqual(
            self.queue.tasks[self.task.id].result,
            {
                "stopped": BUDGET_EXHAUSTED,
                "provider": "searchapi",
                "purpose": "discover",
                "limit_total": 50,
                "used": 50,
                "remaining": 0,
            },
        )
        # One spelling of one fact: the same string the discovery service writes.
        self.assertEqual(BUDGET_EXHAUSTED, "budget_exhausted")

    def test_the_run_is_stopped_and_not_failed(self):
        self.run_it()
        run = self.queue.runs[RUN]
        self.assertEqual(run.status, STOPPED)
        self.assertEqual(STOPPED, "stopped")
        self.assertEqual(run.stats["used"], 50)
        self.assertEqual(run.stats["stopped"], BUDGET_EXHAUSTED)

    def test_the_events_read_budget_stop_then_completion(self):
        self.run_it()
        self.assertEqual(
            self.queue.event_types(self.task.id), [CLAIMED, BUDGET_STOPPED, COMPLETED]
        )
        self.assertEqual(self.queue.event(BUDGET_STOPPED).payload["worker"], "worker-a")

    def test_nothing_is_logged_at_error(self):
        with self.assertLogs("lead_engine.workers", level="INFO") as logs:
            self.run_it()
        self.assertTrue(logs.records)
        self.assertEqual([r for r in logs.records if r.levelno >= logging.ERROR], [])
        self.assertIn("Nothing failed", "\n".join(logs.output))

    def test_the_type_stops_being_claimed(self):
        worker = self.run_it()
        self.assertEqual(worker.stopped_types, {DISCOVER})
        self.assertEqual(worker.claimable_types, ())

        self.queue.enqueue()
        self.queue.trace.clear()
        # A worker that kept polling would claim, refuse and release each queued task in
        # turn, burning one attempt apiece until they all died.
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.trace, [])

    def test_the_rest_of_the_batch_is_left_to_the_reaper(self):
        self.queue.enqueue_many(2)
        self.assertEqual(self.run_it(batch=3).run_once(), 0)

        self.assertEqual(self.handler.performed, [self.task.id])
        self.assertEqual(self.queue.statuses(), ["done", "running", "running"])

    def test_a_second_type_carries_on(self):
        # Only the exhausted type stops. Enrichment does not spend the maps allowance.
        enrich = self.queue.enqueue(ENRICH)
        enricher = RecordingHandler({"enriched": True})
        worker = self.worker({DISCOVER: self.handler, ENRICH: enricher}, batch=2)

        worker.run_once()
        self.assertEqual(worker.claimable_types, (ENRICH,))
        worker.run_once()

        self.assertEqual(enricher.performed, [enrich.id])
        self.assertEqual(self.queue.status_of(enrich.id), "done")

    def test_run_forever_exits_once_every_type_has_stopped(self):
        worker = self.worker({DISCOVER: self.handler})
        with self.assertLogs("lead_engine.workers", level="INFO"):
            performed = worker.run_forever(install_signal_handlers=False)

        self.assertEqual(performed, 1)
        # It exited rather than polling, so it never waited.
        self.assertEqual(self.waits, [])


# --- resume -----------------------------------------------------------------------------------


class ResumeAfterACrash(WorkerTestCase):
    """The spec's Resume row: kill a run mid-flight, restart, repeat no work.

    The counter that matters is `Ledger.used`. Fifty credits, once, ever: a restart that
    re-performs a completed task is not a slow restart, it is money gone.
    """

    def test_a_restart_repeats_nothing_that_was_already_done(self):
        tasks = self.queue.enqueue_many(4)
        ledger = Ledger()
        performed: list[int] = []

        def handler(task):
            # Where the credit goes: before the work, never refunded.
            ledger.spend()
            performed.append(task.id)
            if task.id == 3 and performed.count(3) == 1:
                # The kill. A process that dies here says nothing at all; the row stays
                # `running` with a lease that stops being renewed.
                raise SystemExit("killed mid-flight")
            return {"leads": task.id}

        worker_a = self.worker({DISCOVER: handler})
        self.assertEqual(worker_a.run_once(), 1)
        self.assertEqual(worker_a.run_once(), 1)
        with self.assertRaises(SystemExit):
            worker_a.run_once()

        self.assertEqual(self.queue.statuses(), ["done", "done", "running", "pending"])
        self.assertEqual(ledger.used, 3)

        # --- restart -------------------------------------------------------------------
        # Nothing renewed task 3's lease, so it expires and the reaper returns it. The two
        # completed tasks are `done` and are not reaped, however old their leases look.
        self.queue.advance(seconds=300)
        reaped = self.queue.reap_expired_leases()
        self.assertEqual([task.id for task in reaped], [3])
        self.queue.advance(seconds=10)  # past the retry backoff

        worker_b = Worker(
            self.queue, {DISCOVER: handler}, worker_id="worker-b", sleeper=self.waits.append
        )
        while worker_b.run_once():
            pass

        self.assertEqual(self.queue.statuses(), ["done", "done", "done", "done"])
        repeats = {task_id for task_id, count in Counter(performed).items() if count > 1}
        # Task 3 is performed twice and everything else once: the interrupted task is the
        # only work the restart repeats, and it repeats because it never finished.
        self.assertEqual(repeats, {3})
        self.assertEqual(ledger.used, 5)
        self.assertEqual(sorted(performed), [1, 2, 3, 3, 4])
        # The tasks that had completed before the crash were never handed out again.
        for task in tasks[:2]:
            with self.subTest(task=task.id):
                self.assertEqual(performed.count(task.id), 1)

    def test_the_worker_that_took_over_keeps_its_result(self):
        # The zombie: worker-a's lease expires mid-handler, the reaper returns the task,
        # worker-b performs it, and worker-a then finishes and tries to report.
        task = self.queue.enqueue()
        taken_over: list[int] = []

        def slow_handler(row):
            # Long enough that the lease runs out while the handler is still working.
            self.queue.advance(seconds=600)
            self.queue.reap_expired_leases()
            self.queue.advance(seconds=10)
            worker_b = Worker(
                self.queue,
                {DISCOVER: RecordingHandler({"leads": "worker-b"})},
                worker_id="worker-b",
                sleeper=self.waits.append,
            )
            taken_over.append(worker_b.run_once())
            return {"leads": "worker-a"}

        self.assertEqual(self.worker({DISCOVER: slow_handler}).run_once(), 1)

        self.assertEqual(taken_over, [1])
        self.assertEqual(self.queue.tasks[task.id].result, {"leads": "worker-b"})
        self.assertEqual(self.queue.status_of(task.id), "done")
        # worker-a's outcome was dropped, and dropped loudly.
        lost = [event for event in self.queue.events if event.type == LOST]
        self.assertEqual([event.payload["worker"] for event in lost], ["worker-a"])
        self.assertEqual(lost[0].payload["phase"], "complete")

    def test_a_task_is_never_claimed_by_two_workers_at_once(self):
        # The fake's claim is the same predicate the SQL uses; that Postgres really enforces
        # it under `FOR UPDATE SKIP LOCKED` is tests/test_queue.py's claim, not this one's.
        self.queue.enqueue_many(2)
        first = self.worker({DISCOVER: RecordingHandler()}, batch=2)
        claimed = self.queue.claim(first.worker_id, [DISCOVER], batch=2)

        self.assertEqual(len(claimed), 2)
        self.assertEqual(self.queue.claim("worker-b", [DISCOVER], batch=2), [])


# --- polling, signals and the reaper ----------------------------------------------------------


class ThePollingLoop(WorkerTestCase):
    def test_it_waits_only_when_it_performed_nothing(self):
        self.queue.enqueue_many(2)
        worker = self.worker(batch=2)
        polls = {"n": 0}

        def sleeper(seconds):
            self.waits.append(seconds)
            polls["n"] += 1
            if polls["n"] == 2:
                worker.stop()

        worker._sleeper = sleeper
        performed = worker.run_forever(install_signal_handlers=False)

        self.assertEqual(performed, 2)
        # Two tasks performed in the first pass, then two idle polls at the configured
        # interval. Nothing waited between the tasks themselves.
        self.assertEqual(self.waits, [DEFAULT_POLL_INTERVAL, DEFAULT_POLL_INTERVAL])

    def test_stop_ends_the_loop_and_the_total_comes_back(self):
        self.queue.enqueue_many(3)
        worker = self.worker(batch=1)
        worker._sleeper = lambda seconds: worker.stop()

        self.assertEqual(worker.run_forever(install_signal_handlers=False), 3)
        self.assertTrue(worker.stopping)

    def test_a_signal_stops_every_worker_it_was_given(self):
        first, second = self.worker(worker_id="a"), self.worker(worker_id="b")
        before = signal.getsignal(signal.SIGINT)

        with stop_on_signals([first, second]):
            installed = signal.getsignal(signal.SIGINT)
            self.assertIsNot(installed, before)
            # What the OS would call. Clean means the task in hand is finished and recorded
            # before the process exits, so this only sets the flag.
            installed(signal.SIGINT, None)
            self.assertTrue(first.stopping)
            self.assertTrue(second.stopping)

        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_no_workers_means_no_handlers_are_touched(self):
        before = signal.getsignal(signal.SIGINT)
        with stop_on_signals([]):
            self.assertIs(signal.getsignal(signal.SIGINT), before)
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_run_forever_can_be_told_to_leave_the_handlers_alone(self):
        # `--concurrency 2` runs workers in threads, where signal.signal raises; the main
        # thread installs one set of handlers for all of them.
        before = signal.getsignal(signal.SIGINT)
        worker = self.worker()
        worker._sleeper = lambda seconds: worker.stop()

        worker.run_forever(install_signal_handlers=False)

        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_a_worker_started_in_a_thread_still_runs(self):
        # signal.signal raises ValueError off the main thread. That is caught, not avoided.
        self.queue.enqueue()
        worker = self.worker()
        worker._sleeper = lambda seconds: worker.stop()
        performed: list[int] = []

        thread = threading.Thread(target=lambda: performed.append(worker.run_forever()))
        thread.start()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(performed, [1])


class TheReaper(WorkerTestCase):
    """Off unless asked for: a library that reaped by default would surprise a test that
    deliberately holds a lease."""

    def run_polls(self, worker: Worker, count: int, monotonic: list[float]) -> None:
        polls = {"n": 0}

        def sleeper(seconds):
            polls["n"] += 1
            if polls["n"] >= count:
                worker.stop()

        worker._sleeper = sleeper
        with mock.patch.object(runner, "_monotonic", side_effect=monotonic):
            worker.run_forever(install_signal_handlers=False)

    def test_reaping_is_off_by_default(self):
        worker = self.worker()
        self.run_polls(worker, 3, [0.0, 1.0, 2.0])
        self.assertIsNone(worker.reap_interval)
        self.assertEqual(self.queue.reaps, 0)

    def test_expired_leases_are_swept_at_most_once_per_interval(self):
        worker = self.worker(reap_interval=60.0)
        self.run_polls(worker, 3, [0.0, 10.0, 20.0])
        self.assertEqual(self.queue.reaps, 1)

    def test_the_sweep_runs_again_once_the_interval_has_passed(self):
        worker = self.worker(reap_interval=60.0)
        self.run_polls(worker, 3, [0.0, 100.0, 200.0])
        self.assertEqual(self.queue.reaps, 3)

    def test_a_reaped_task_is_reported(self):
        self.queue.enqueue()
        self.queue.claim("a-worker-that-died", [DISCOVER], lease="-1 seconds")
        worker = self.worker(reap_interval=0.0)

        with self.assertLogs("lead_engine.workers", level="WARNING") as logs:
            self.run_polls(worker, 1, [0.0, 1.0])

        self.assertIn("returned 1 expired task(s)", "\n".join(logs.output))
        self.assertEqual(self.queue.status_of(1), "retry")


class SeveralWorkers(WorkerTestCase):
    def test_no_workers_performs_nothing(self):
        self.assertEqual(run_workers([]), 0)

    def test_one_worker_runs_in_this_thread(self):
        self.queue.enqueue_many(2)
        worker = self.worker(batch=2)
        worker._sleeper = lambda seconds: worker.stop()
        self.assertEqual(run_workers([worker]), 2)

    def test_two_workers_share_one_queue_without_repeating_a_task(self):
        self.queue.enqueue_many(6)
        handler = RecordingHandler({"ok": True})
        workers = [
            Worker(self.queue, {DISCOVER: handler}, worker_id=f"worker-{index}")
            for index in range(2)
        ]
        for worker in workers:
            # Each worker stops at its own first idle poll, which cannot happen while it
            # still has claimable work.
            worker._sleeper = (lambda w: lambda seconds: w.stop())(worker)

        total = run_workers(workers)

        self.assertEqual(total, 6)
        self.assertEqual(sorted(handler.performed), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.queue.statuses(), ["done"] * 6)
        # Each worker writes under its own identity, and one task is never recorded under
        # two of them -- which is what the events have to say for anyone working out after
        # the fact who held what. (Which worker got how many is a race, and not asserted.)
        owners = {event.payload["worker"] for event in self.queue.events}
        self.assertTrue(owners <= {"worker-0", "worker-1"}, owners)
        for task_id in range(1, 7):
            with self.subTest(task=task_id):
                per_task = {
                    event.payload["worker"]
                    for event in self.queue.events
                    if event.task_id == task_id
                }
                self.assertEqual(len(per_task), 1, f"task {task_id} was recorded twice")


# --- the entry point ----------------------------------------------------------------------


@contextmanager
def stub_discover(handler):
    """Stand in for `lead_engine.workers.discover`, which `main()` imports lazily.

    The module is not in this package yet -- see the summary in the module docstring of the
    runner -- and even once it is, importing it drags in the whole composition root. What
    `main()` owes it is a contract: build one handler from the settings, the fixture
    directory and the output directory, and close it on the way out. That is what this stub
    records, and it pins the contract whether or not the real module exists.
    """
    name = "lead_engine.workers.discover"
    module = types.ModuleType(name)
    module.DISCOVER = DISCOVER
    module.calls = []

    def build_discover_handler(settings, *, fixture_dir=None, output_dir=None):
        module.calls.append(
            {"settings": settings, "fixture_dir": fixture_dir, "output_dir": output_dir}
        )
        return handler

    module.build_discover_handler = build_discover_handler
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:  # pragma: no cover - only once the real module exists
            sys.modules[name] = previous


class ClosingHandler(RecordingHandler):
    """A handler holding something that has to be released -- an HTTP client, a pool."""

    def __init__(self, result: dict | None = None) -> None:
        super().__init__(result)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class Arguments(unittest.TestCase):
    def test_repeated_and_comma_separated_types_mean_the_same_thing(self):
        for argv, expected in (
            ([], ("discover",)),
            (["discover"], ("discover",)),
            (["discover", "enrich"], ("discover", "enrich")),
            (["discover,enrich"], ("discover", "enrich")),
            (["discover, enrich"], ("discover", "enrich")),
            (["discover", "discover"], ("discover",)),
            (["", " "], ("discover",)),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(requested_types(argv, default="discover"), expected)

    def test_the_parser_defaults_are_the_cheap_ones(self):
        args = build_parser().parse_args([])
        # batch 1 because a batch's leases all start together, so anything above 1 leaves
        # the tail of an interrupted batch to the reaper.
        self.assertEqual(args.batch, 1)
        self.assertEqual(args.concurrency, 1)
        self.assertEqual(args.lease, DEFAULT_LEASE)
        self.assertEqual(args.poll_interval, DEFAULT_POLL_INTERVAL)
        self.assertEqual(args.reap_interval, 60.0)
        self.assertFalse(args.once)
        self.assertEqual(args.types, [])
        self.assertIsNone(args.fixtures)
        self.assertIsNone(args.output_dir)

    def test_a_worker_id_is_suffixed_only_when_there_is_more_than_one(self):
        for concurrency, expected in ((1, "hand-picked"), (2, "hand-picked-0")):
            with self.subTest(concurrency=concurrency):
                args = build_parser().parse_args(
                    ["--worker-id", "hand-picked", "--concurrency", str(concurrency)]
                )
                self.assertEqual(_worker_id(args, 0), expected)

    def test_without_an_override_every_worker_gets_its_own_id(self):
        args = build_parser().parse_args(["--concurrency", "2"])
        self.assertNotEqual(_worker_id(args, 0), _worker_id(args, 1))


class TheEntryPoint(unittest.TestCase):
    """`main()` wires a process up and returns an exit code rather than raising."""

    def setUp(self):
        self.queue = FakeQueue()
        self.settings = SimpleNamespace(dsn=None)
        # main() calls logging.basicConfig, which mutates the root logger for whatever runs
        # next. Put it back.
        root = logging.getLogger()
        handlers, level = list(root.handlers), root.level
        self.addCleanup(lambda: (root.handlers.clear(), root.handlers.extend(handlers)))
        self.addCleanup(root.setLevel, level)

    def main(self, argv, **kwargs):
        """Run main() with stderr captured. Returns (exit code, stderr)."""
        stderr = StringIO()
        with redirect_stderr(stderr):
            code = runner.main(argv, settings=self.settings, **kwargs)
        return code, stderr.getvalue()

    def test_a_type_with_no_handler_cannot_start(self):
        code, stderr = self.main(
            ["--types", "enrich"],
            repository=self.queue,
            handlers={DISCOVER: RecordingHandler()},
        )
        self.assertEqual(code, EXIT_CANNOT_START)
        self.assertIn("unknown_task_type", stderr)
        self.assertIn("enrich", stderr)
        # And it says what this process CAN serve, so the operator does not have to guess.
        self.assertIn(DISCOVER, stderr)
        self.assertEqual(self.queue.trace, [])

    def test_without_a_dsn_it_refuses_to_start_rather_than_running_on_nothing(self):
        code, stderr = self.main(["--once"], handlers={DISCOVER: RecordingHandler()})
        self.assertEqual(code, EXIT_CANNOT_START)
        self.assertIn("database_not_configured", stderr)
        self.assertIn("LEAD_ENGINE_DSN", stderr)

    def test_once_performs_one_batch_and_exits_zero(self):
        self.queue.enqueue_many(2)
        handler = RecordingHandler({"leads": 1})

        code, _ = self.main(
            ["--once", "--batch", "2"], repository=self.queue, handlers={DISCOVER: handler}
        )

        self.assertEqual(code, EXIT_OK)
        self.assertEqual(handler.performed, [1, 2])
        self.assertEqual(self.queue.statuses(), ["done", "done"])

    def test_concurrency_gives_each_worker_its_own_identity(self):
        self.queue.enqueue_many(2)
        handler = RecordingHandler({"leads": 1})

        code, _ = self.main(
            ["--once", "--concurrency", "2", "--worker-id", "w"],
            repository=self.queue,
            handlers={DISCOVER: handler},
        )

        self.assertEqual(code, EXIT_OK)
        self.assertEqual(sorted(handler.performed), [1, 2])
        owners = {event.payload["worker"] for event in self.queue.events}
        self.assertEqual(owners, {"w-0", "w-1"})

    def test_what_it_was_given_it_does_not_close(self):
        # An injected repository or handler belongs to the caller -- a test, or an embedding
        # process -- and closing it would break whatever else is using it.
        handler = ClosingHandler({"leads": 1})
        repository = SimpleNamespace(
            claim=self.queue.claim,
            renew_lease=self.queue.renew_lease,
            complete=self.queue.complete,
            fail=self.queue.fail,
            append_event=self.queue.append_event,
            finish_run=self.queue.finish_run,
            reap_expired_leases=self.queue.reap_expired_leases,
            closed=0,
        )
        repository.close = lambda: setattr(repository, "closed", repository.closed + 1)

        code, _ = self.main(["--once"], repository=repository, handlers={DISCOVER: handler})

        self.assertEqual(code, EXIT_OK)
        self.assertEqual(handler.closed, 0)
        self.assertEqual(repository.closed, 0)

    def test_what_it_built_itself_it_closes(self):
        handler = ClosingHandler({"leads": 1})
        self.queue.enqueue()

        with stub_discover(handler) as module:
            code, _ = self.main(
                ["--once", "--fixtures", "tests/fixtures/searchapi", "--output-dir", "exports"],
                repository=self.queue,
            )

        self.assertEqual(code, EXIT_OK)
        self.assertEqual(handler.performed, [1])
        self.assertEqual(handler.closed, 1)
        # The handler is built from the settings it was given, and the fixture directory is
        # how the queue is demonstrated end to end for free.
        self.assertEqual(len(module.calls), 1)
        self.assertIs(module.calls[0]["settings"], self.settings)
        self.assertEqual(module.calls[0]["fixture_dir"], "tests/fixtures/searchapi")
        self.assertEqual(module.calls[0]["output_dir"], "exports")

    @pytest.mark.xfail(
        reason=(
            "PROVEN BUG, runner.py:691-706. Both `return EXIT_CANNOT_START` paths sit "
            "BEFORE the try/finally that closes the handlers main() built, so a process "
            "that constructs the discover handler and then refuses to start leaks whatever "
            "that handler holds -- an httpx client, a pool. Not fixed here on purpose: see "
            "the summary. The same leak is reachable through the missing-DSN return."
        ),
        strict=True,
    )
    def test_an_early_exit_closes_the_handler_it_built(self):
        handler = ClosingHandler()
        with stub_discover(handler):
            code, _ = self.main(["--types", "enrich"], repository=self.queue)

        self.assertEqual(code, EXIT_CANNOT_START)
        self.assertEqual(handler.closed, 1)


class ThePackage(unittest.TestCase):
    def test_the_loop_is_importable_without_the_service_layer(self):
        # `runner` is imported eagerly precisely so `Worker` can be unit-tested with a fake
        # queue and no driver installed; everything in this file depends on that.
        from lead_engine import workers

        self.assertIs(workers.Worker, Worker)
        self.assertIs(workers.Handler, runner.Handler)
        self.assertIs(workers.TaskQueue, runner.TaskQueue)

    def test_an_unknown_attribute_is_an_attribute_error(self):
        from lead_engine import workers

        with self.assertRaises(AttributeError):
            getattr(workers, "nonexistent")  # noqa: B009 - the lookup is the assertion


# --- against a real Postgres --------------------------------------------------------------


@unittest.skipIf(
    psycopg is None or not DSN,
    "needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)
class WorkerAgainstPostgres(unittest.TestCase):
    """The parts of the loop that are only true if the database says so.

    Everything above proves the worker asks the right questions. This proves the answers it
    gets from a real queue -- a real reaper returning a real expired lease, a real status
    guard refusing a late completion -- are the ones the loop was written against.

    Isolation is a throwaway schema per test, as in tests/test_queue.py. Nothing sleeps: a
    lease is expired by claiming with a negative interval, and backoff is moved with an
    explicit UPDATE.
    """

    pytestmark = pytest.mark.integration

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
        goal = self.repo.create_goal("blr salons", {"city": "Bangalore", "niche": "salon"})
        self.run = self.repo.create_run(goal.id, trigger="manual")

    def _drop_schema(self):
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')

    def sql(self, statement, params=None):
        with self.pool.connection() as connection:
            return connection.execute(statement, params).fetchall()

    def enqueue(self, count: int = 1, type: str = DISCOVER) -> list[int]:
        ids = []
        for index in range(count):
            key = f"{type}:tile-{uuid.uuid4()}"
            self.assertTrue(self.repo.enqueue(self.run.id, type, {"tile": index}, idem_key=key))
            ids.append(self.sql("SELECT id FROM tasks WHERE idem_key = %s", (key,))[0][0])
        return ids

    def make_claimable(self):
        with self.pool.connection() as connection:
            connection.execute("UPDATE tasks SET available_at = now()")

    def worker(self, handlers, **kwargs) -> Worker:
        kwargs.setdefault("worker_id", "worker-a")
        kwargs.setdefault("sleeper", lambda seconds: None)
        return Worker(self.repo, handlers, **kwargs)

    def event_types(self):
        return [row[0] for row in self.sql("SELECT type FROM events ORDER BY seq")]

    def test_a_batch_is_claimed_performed_and_recorded(self):
        ids = self.enqueue(3)
        handler = RecordingHandler({"leads": 2})

        self.assertEqual(self.worker({DISCOVER: handler}, batch=3).run_once(), 3)

        self.assertEqual(sorted(handler.performed), sorted(ids))
        statuses = {row[0] for row in self.sql("SELECT status FROM tasks")}
        self.assertEqual(statuses, {"done"})
        self.assertEqual(
            self.sql("SELECT result FROM tasks ORDER BY id")[0][0], {"leads": 2}
        )
        self.assertEqual(
            self.event_types(), [CLAIMED, COMPLETED, CLAIMED, COMPLETED, CLAIMED, COMPLETED]
        )

    def test_a_restart_after_a_kill_repeats_only_the_unfinished_task(self):
        # The spec's Resume row, against the real queue: two tasks done, one killed
        # mid-flight, restart, and assert the done ones are never handed out again.
        ids = self.enqueue(3)
        performed: list[int] = []

        def handler(task):
            performed.append(task.id)
            if task.id == ids[2] and performed.count(ids[2]) == 1:
                raise SystemExit("killed mid-flight")
            return {"leads": 1}

        worker_a = self.worker({DISCOVER: handler}, batch=1)
        self.assertEqual(worker_a.run_once(), 1)
        self.assertEqual(worker_a.run_once(), 1)
        # The kill: claim with a lease that has already run out, which is the state a dead
        # worker leaves behind, and then die inside the handler.
        killed = self.repo.claim("worker-a", [DISCOVER], lease="-1 seconds")[0]
        with self.assertRaises(SystemExit):
            handler(killed)

        self.assertEqual(
            {row[0] for row in self.sql("SELECT status FROM tasks WHERE id = %s", (ids[2],))},
            {"running"},
        )

        reaped = self.repo.reap_expired_leases()
        self.assertEqual([task.id for task in reaped], [ids[2]])
        self.make_claimable()

        worker_b = self.worker({DISCOVER: handler}, worker_id="worker-b")
        while worker_b.run_once():
            pass

        self.assertEqual(
            {row[0] for row in self.sql("SELECT status FROM tasks")}, {"done"}
        )
        self.assertEqual(sorted(performed), sorted(ids + [ids[2]]))
        self.assertEqual(Counter(performed)[ids[0]], 1)
        self.assertEqual(Counter(performed)[ids[1]], 1)

    def test_a_worker_whose_lease_was_reaped_abandons_the_task_unperformed(self):
        task_id = self.enqueue(1)[0]
        handler = RecordingHandler({"leads": 1})
        worker = self.worker({DISCOVER: handler})
        stale = self.repo.claim("worker-a", [DISCOVER], lease="-1 seconds")[0]
        self.repo.reap_expired_leases()

        self.assertFalse(worker._perform(stale))

        self.assertEqual(handler.performed, [])
        self.assertEqual(self.event_types(), [CLAIMED, ABANDONED])
        self.assertEqual(
            self.sql("SELECT status FROM tasks WHERE id = %s", (task_id,))[0][0], "retry"
        )

    def test_a_late_completion_cannot_overwrite_the_worker_that_took_over(self):
        task_id = self.enqueue(1)[0]
        worker = self.worker({DISCOVER: RecordingHandler({"leads": "worker-a"})})
        stale = self.repo.claim("worker-a", [DISCOVER], lease="-1 seconds")[0]
        self.repo.reap_expired_leases()
        self.make_claimable()
        taken_over = self.repo.claim("worker-b", [DISCOVER])[0]
        self.repo.complete(taken_over.id, {"leads": "worker-b"})

        # worker-a finishing anyway. The status guard refuses it, and the worker drops the
        # outcome instead of retrying -- a retry would perform the task a second time.
        worker._complete(stale, {"leads": "worker-a"})

        row = self.sql("SELECT status, result FROM tasks WHERE id = %s", (task_id,))[0]
        self.assertEqual(row[0], "done")
        self.assertEqual(row[1], {"leads": "worker-b"})
        self.assertEqual(self.event_types()[-1], LOST)

    def test_the_allowance_running_out_stops_the_run_without_failing_anything(self):
        self.enqueue(2)
        worker = self.worker({DISCOVER: RaisingHandler(exhausted(used=50))}, batch=2)

        self.assertEqual(worker.run_once(), 1)

        statuses = [row[0] for row in self.sql("SELECT status FROM tasks ORDER BY id")]
        # The first is done and the second is left claimed for the reaper -- not failed,
        # because nothing failed and an attempt spent on it would be an attempt wasted.
        self.assertEqual(statuses, ["done", "running"])
        run = self.sql("SELECT status, stats FROM runs")[0]
        self.assertEqual(run[0], STOPPED)
        self.assertEqual(run[1]["stopped"], BUDGET_EXHAUSTED)
        self.assertNotIn(FAILED, self.event_types())

    def test_a_provider_error_lands_in_the_row_with_the_schemas_backoff(self):
        task_id = self.enqueue(1)[0]
        worker = self.worker({DISCOVER: RaisingHandler(ProviderError("provider_rate_limited"))})

        worker.run_once()

        row = self.sql(
            "SELECT status, error, attempts, available_at > now() FROM tasks WHERE id = %s",
            (task_id,),
        )[0]
        self.assertEqual(row[0], "retry")
        self.assertTrue(row[1].startswith("provider_rate_limited: "))
        self.assertEqual(row[2], 1)
        self.assertTrue(row[3], "the retry was not pushed into the future")

    def test_an_unhandled_exception_writes_no_credential_to_the_row(self):
        task_id = self.enqueue(1)[0]
        handler = RaisingHandler(RuntimeError(f"GET {LEAKY_URL} failed"))

        with self.assertLogs("lead_engine.workers", level="ERROR"):
            self.worker({DISCOVER: handler}).run_once()

        stored = self.sql("SELECT error FROM tasks WHERE id = %s", (task_id,))[0][0]
        self.assertIn("unhandled RuntimeError", stored)
        self.assertNotIn(LEAKED_KEY, stored)


if __name__ == "__main__":
    unittest.main()
