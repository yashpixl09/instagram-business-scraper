"""The durable queue and the repository, against a real Postgres.

Skipped unless `LEAD_ENGINE_TEST_DSN` is set, for the reason given in
`tests/test_migrations.py`: most of this suite is pure and has to keep running with no
database anywhere near it.

Isolation is a throwaway schema per test. The pool handed to the `Repository` carries
`options=-c search_path=<schema>,public`, so every connection it lends out -- including the
eight the concurrency test uses at once -- lands in that schema and nowhere else.

Nothing here sleeps. Waiting for a lease to expire in real time would add seconds per test
and still be a race on a loaded machine, so expiry is forced by claiming with a negative
lease interval and the backoff clock is moved by an explicit UPDATE. Both are called out
where they happen.
"""

from __future__ import annotations

import os
import re
import threading
import time
import unittest
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db import queries
    from lead_engine.db.migrate import apply_migrations
    from lead_engine.db.repository import Repository
    from lead_engine.db.rows import (
        BusinessRow,
        EnrichmentRow,
        EventRow,
        GoalRow,
        RunRow,
        ScoreRow,
        TaskRow,
    )
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        psycopg is None or not DSN,
        reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
    ),
]

class QueueTestCase(unittest.TestCase):
    """A migrated scratch schema, a pool pointed at it, and a repository over the pool."""

    # Eight, because the concurrency test runs eight threads and a pool smaller than the
    # thread count would serialise them into a queue of its own -- and then pass whether or
    # not the SQL is safe.
    POOL_SIZE = 8

    def setUp(self):
        self.schema = "test_" + uuid.uuid4().hex
        with psycopg.connect(DSN, autocommit=True) as connection:
            # Database-scoped, so it survives this schema and 0001 becomes a no-op here.
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(f'CREATE SCHEMA "{self.schema}"')
        # Registered first so it runs last: cleanup is LIFO, and dropping the schema while
        # the pool still holds connections to it is how you get a hang instead of a result.
        self.addCleanup(self._drop_schema)

        with psycopg.connect(DSN) as connection:
            connection.execute(f"SET search_path = {self.schema}, public")
            apply_migrations(connection)
            connection.commit()

        self.pool = ConnectionPool(
            DSN,
            min_size=1,
            max_size=self.POOL_SIZE,
            open=True,
            kwargs={"options": f"-c search_path={self.schema},public"},
        )
        self.addCleanup(self.pool.close)
        # The injected pool IS the seam. No environment variable reaches the repository.
        self.repo = Repository(self.pool)

    def _drop_schema(self):
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')

    # --- helpers --------------------------------------------------------------------

    def sql(self, statement, params=None):
        """Read state back without going through the code under test."""
        with self.pool.connection() as connection:
            return connection.execute(statement, params).fetchall()

    def db_now(self):
        """The database's clock, not this machine's. Backoff is computed server-side."""
        return self.sql("SELECT now()")[0][0]

    def make_claimable(self, task_id):
        """Stand in for the passage of time: drop the task's backoff to zero.

        The alternative is sleeping through a real exponential backoff, which by the third
        attempt is eight seconds of a test suite doing nothing.
        """
        with self.pool.connection() as connection:
            connection.execute("UPDATE tasks SET available_at = now() WHERE id = %s", (task_id,))

    def a_run(self):
        goal = self.repo.create_goal("blr salons", {"city": "Bangalore", "niche": "salon"})
        return self.repo.create_run(goal.id, trigger="manual")

    def status_of(self, task_id):
        return self.sql("SELECT status FROM tasks WHERE id = %s", (task_id,))[0][0]

    def enqueue(self, run, type="discover", *, idem_key=None, **kwargs):
        key = idem_key or f"{type}:{uuid.uuid4()}"
        inserted = self.repo.enqueue(run.id, type, {"key": key}, idem_key=key, **kwargs)
        self.assertTrue(inserted, "fixture enqueue was suppressed by a duplicate idem_key")
        return self.sql("SELECT id FROM tasks WHERE idem_key = %s", (key,))[0][0]


class RowMappingTests(QueueTestCase):
    def test_every_row_dataclass_matches_its_table_exactly(self):
        # `class_row` builds a row by keyword, so a field the table does not have raises
        # TypeError on the first fetch -- and a column the class does not have raises one
        # too, but only in whichever query happens to run first, in whichever worker
        # happens to run it. Checked here instead, once, against the live catalog.
        for table, row_class in (
            ("goals", GoalRow),
            ("runs", RunRow),
            ("tasks", TaskRow),
            ("events", EventRow),
            ("businesses", BusinessRow),
            ("enrichments", EnrichmentRow),
            ("scores", ScoreRow),
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
                self.assertEqual(
                    declared,
                    columns,
                    f"{row_class.__name__} and {table} have drifted; a migration changed"
                    " the table and rows.py did not follow",
                )

    def test_no_query_selects_a_star(self):
        # The rule the explicit column lists exist to enforce, checked on the SQL itself so
        # that adding a convenient `SELECT *` fails here rather than in a worker six months
        # after the migration that breaks it.
        for name in dir(queries):
            statement = getattr(queries, name)
            if name.startswith("_") or not isinstance(statement, str):
                continue
            with self.subTest(query=name):
                self.assertIsNone(re.search(r"select\s+\*", statement, re.IGNORECASE))


class EnqueueTests(QueueTestCase):
    def test_a_duplicate_idem_key_inserts_once(self):
        run = self.a_run()
        first = self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="blr:salon:tile-7")
        second = self.repo.enqueue(run.id, "discover", {"tile": 2}, idem_key="blr:salon:tile-7")

        self.assertTrue(first)
        self.assertFalse(second)
        rows = self.sql("SELECT payload FROM tasks")
        self.assertEqual(len(rows), 1)
        # DO NOTHING, not DO UPDATE: the second enqueue must not rewrite the first task's
        # payload, which by then may already have been claimed and acted on.
        self.assertEqual(rows[0][0], {"tile": 1})

    def test_a_duplicate_enqueue_cannot_resurrect_a_finished_task(self):
        run = self.a_run()
        self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-7")
        claimed = self.repo.claim("w1", ["discover"])
        self.repo.complete(claimed[0].id, {"found": 3})

        self.assertFalse(self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-7"))
        self.assertEqual(self.repo.get_task(claimed[0].id).status, "done")
        self.assertEqual(self.repo.claim("w1", ["discover"]), [])

    def test_the_same_key_is_free_again_once_the_row_is_gone(self):
        # Idempotency is scoped to rows that still exist, which is what makes a purge of
        # old tasks a legitimate operation rather than a permanent poisoning of the keys.
        run = self.a_run()
        self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-7")
        self.sql("DELETE FROM tasks WHERE idem_key = 'tile-7' RETURNING id")
        self.assertTrue(self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-7"))


class ClaimTests(QueueTestCase):
    def test_concurrent_claim_never_double_issues_a_task(self):
        """The one that matters. Eight threads, one queue, real connections.

        A claim implemented as SELECT-then-UPDATE passes every single-threaded test in this
        file and hands the same task to two workers here.
        """
        run = self.a_run()
        for index in range(60):
            self.enqueue(run, idem_key=f"tile-{index}")
        expected = {row[0] for row in self.sql("SELECT id FROM tasks")}

        workers = 8
        # All eight threads reach the queue in the same instant instead of trickling in
        # behind each other's startup, which is the only way the interleaving under test
        # actually happens.
        barrier = threading.Barrier(workers, timeout=30)
        lock = threading.Lock()
        drained = threading.Event()
        counted = {"claimed": 0}
        deadline = time.monotonic() + 30

        def worker(name):
            mine = []
            barrier.wait()
            while not drained.is_set() and time.monotonic() < deadline:
                batch = self.repo.claim(name, ["discover"], batch=3)
                mine.extend(task.id for task in batch)
                with lock:
                    counted["claimed"] += len(batch)
                    if counted["claimed"] >= len(expected):
                        # Loop on the total rather than on an empty batch: SKIP LOCKED
                        # hands back nothing whenever a peer holds the rows, so "empty"
                        # does not mean "drained" and a worker that stopped there would
                        # leave the test asserting on a partial claim.
                        drained.set()
            return name, mine

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(worker, [f"worker-{n}" for n in range(workers)]))

        claimed = [task_id for _, ids in results for task_id in ids]
        duplicates = [task_id for task_id, count in Counter(claimed).items() if count > 1]
        self.assertEqual(duplicates, [], f"tasks issued to more than one worker: {duplicates}")
        self.assertEqual(set(claimed), expected, "a task was lost or invented")
        self.assertEqual(len(claimed), len(expected))

        # Independent of what the workers reported: the database's own count of deliveries.
        # Every task was handed out exactly once, so every task was claimed exactly once.
        attempts = {row[0] for row in self.sql("SELECT DISTINCT attempts FROM tasks")}
        self.assertEqual(attempts, {1})
        statuses = {row[0] for row in self.sql("SELECT DISTINCT status FROM tasks")}
        self.assertEqual(statuses, {"running"})

        owners = dict(self.sql("SELECT id, locked_by FROM tasks"))
        for name, ids in results:
            for task_id in ids:
                self.assertEqual(owners[task_id], name)

    def test_claim_takes_only_the_requested_types(self):
        run = self.a_run()
        wanted = self.enqueue(run, "discover")
        self.enqueue(run, "enrich")

        claimed = self.repo.claim("w1", ["discover"], batch=10)
        self.assertEqual([task.id for task in claimed], [wanted])
        self.assertEqual(self.repo.get_task(claimed[0].id).type, "discover")

    def test_claim_serves_lower_priority_numbers_first(self):
        run = self.a_run()
        ordinary = self.enqueue(run, idem_key="ordinary")
        urgent = self.enqueue(run, idem_key="urgent", priority=1)
        also_ordinary = self.enqueue(run, idem_key="also-ordinary")

        claimed = self.repo.claim("w1", ["discover"], batch=3)
        # Priority first, then id: a tie must resolve to the older task, so nothing at the
        # default priority can starve behind newer work.
        self.assertEqual([task.id for task in claimed], [urgent, ordinary, also_ordinary])

    def test_claim_marks_the_row_running_and_leases_it(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        before = self.db_now()

        claimed = self.repo.claim("worker-1", ["discover"], lease="90 seconds")

        self.assertEqual(len(claimed), 1)
        task = claimed[0]
        self.assertEqual(task.id, task_id)
        self.assertEqual(task.status, "running")
        self.assertEqual(task.locked_by, "worker-1")
        # attempts moves on the claim, not on the failure. A worker that dies silently has
        # still spent one.
        self.assertEqual(task.attempts, 1)
        self.assertGreaterEqual(task.lease_expires - before, timedelta(seconds=89))
        self.assertLessEqual(task.lease_expires - before, timedelta(seconds=95))
        self.assertEqual(self.repo.claim("worker-2", ["discover"]), [])

    def test_a_task_scheduled_for_later_is_not_claimable(self):
        run = self.a_run()
        task_id = self.enqueue(run, available_at=datetime.now(UTC) + timedelta(hours=1))

        self.assertEqual(self.repo.claim("w1", ["discover"], batch=10), [])
        self.assertEqual(self.repo.get_task(task_id).status, "pending")

        self.make_claimable(task_id)
        self.assertEqual([task.id for task in self.repo.claim("w1", ["discover"])], [task_id])

    def test_claim_returns_an_empty_list_on_an_empty_queue(self):
        self.assertEqual(self.repo.claim("w1", ["discover"], batch=5), [])


class CompletionTests(QueueTestCase):
    def test_complete_records_the_result_and_releases_the_lease(self):
        run = self.a_run()
        self.enqueue(run)
        task = self.repo.claim("w1", ["discover"])[0]

        done = self.repo.complete(task.id, {"found": 12})

        self.assertEqual(done.status, "done")
        self.assertEqual(done.result, {"found": 12})
        self.assertIsNone(done.locked_by)
        self.assertIsNone(done.lease_expires)
        # A completed task is not reaped, however long ago its lease was written.
        self.assertEqual(self.repo.reap_expired_leases(), [])

    def test_complete_on_a_missing_task_returns_none(self):
        self.assertIsNone(self.repo.complete(999999, {"found": 0}))


class OwnershipGuardTests(QueueTestCase):
    """A worker may only finish a task it still holds.

    Without these guards a late duplicate -- a worker whose lease the reaper reclaimed,
    finishing anyway -- writes over a task somebody else now owns. The worst direction is
    `fail()` on a task already `done`: the row flips back to `retry` keeping its stale
    result, so the task is delivered and performed a second time. For a discovery task that
    is a second billed Google Maps search for a cell already swept, on an allowance that does
    not renew.
    """

    def test_completing_a_task_you_no_longer_hold_changes_nothing(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        claimed = self.repo.claim("worker-a", ["discover"])[0]
        self.repo.complete(claimed.id, {"ok": True})

        # The late duplicate arrives and tries to finish the same task.
        self.assertIsNone(self.repo.complete(task_id, {"ok": "from the zombie"}))
        row = self.sql("SELECT status, result FROM tasks WHERE id = %s", (task_id,))[0]
        self.assertEqual(row[0], "done")
        self.assertEqual(row[1], {"ok": True}, "the first worker's result was overwritten")

    def test_failing_a_finished_task_does_not_resurrect_it(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        claimed = self.repo.claim("worker-a", ["discover"])[0]
        self.repo.complete(claimed.id, {"leads": 12})

        self.assertIsNone(self.repo.fail(task_id, "zombie worker reporting a timeout"))
        status = self.sql("SELECT status FROM tasks WHERE id = %s", (task_id,))[0][0]
        self.assertEqual(status, "done", "a done task was flipped back and would run twice")

    def test_a_dead_task_cannot_be_revived_by_a_late_completion(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        with self.pool.connection() as connection:
            connection.execute("UPDATE tasks SET max_attempts = 1 WHERE id = %s", (task_id,))
        claimed = self.repo.claim("worker-a", ["discover"])[0]
        self.repo.fail(claimed.id, "provider timeout")
        self.assertEqual(self.status_of(task_id), "dead")

        self.assertIsNone(self.repo.complete(task_id, {"ok": True}))
        self.assertEqual(self.status_of(task_id), "dead")


class RenewLeaseTests(QueueTestCase):
    """Every row in a batch is stamped from one `now()`.

    So the last task of a batch of five is four tasks' runtime into its lease before the
    worker looks at it. A serial worker renews as each task begins; otherwise the reaper
    reclaims the tail of every batch mid-flight and each of those tasks is done twice.
    """

    def test_renewing_pushes_the_deadline_out(self):
        run = self.a_run()
        self.enqueue(run)
        claimed = self.repo.claim("worker-a", ["discover"], lease="5 seconds")[0]
        renewed = self.repo.renew_lease(claimed.id, "worker-a", lease="10 minutes")
        self.assertIsNotNone(renewed)
        self.assertGreater(renewed.lease_expires, claimed.lease_expires)

    def test_renewing_a_task_another_worker_holds_returns_none(self):
        # The signal to STOP. Whatever this worker was about to do, someone else is doing --
        # and continuing is exactly how one cell becomes two billed searches.
        run = self.a_run()
        self.enqueue(run)
        claimed = self.repo.claim("worker-a", ["discover"])[0]
        self.assertIsNone(self.repo.renew_lease(claimed.id, "worker-b"))

    def test_renewing_a_reclaimed_task_returns_none(self):
        run = self.a_run()
        self.enqueue(run)
        claimed = self.repo.claim("worker-a", ["discover"], lease="-1 seconds")[0]
        self.repo.reap_expired_leases()
        self.assertIsNone(self.repo.renew_lease(claimed.id, "worker-a"))


class BackoffTests(QueueTestCase):
    def assert_backoff(self, task, expected_seconds, *, measured_from):
        """`available_at` is `measured_from` plus the backoff, within a second of slack.

        The slack absorbs the round trip only. It is deliberately smaller than the gap
        between consecutive backoffs (2s, 4s, 8s), so a test that expects 4 cannot pass on
        a 2 that happened to be measured late.
        """
        delay = task.available_at - measured_from
        self.assertGreaterEqual(delay, timedelta(seconds=expected_seconds))
        self.assertLess(delay, timedelta(seconds=expected_seconds + 1))

    def test_failure_schedules_a_retry_with_exponential_backoff(self):
        run = self.a_run()
        task_id = self.enqueue(run)

        self.repo.claim("w1", ["discover"])
        before = self.db_now()
        first = self.repo.fail(task_id, "upstream returned 503")

        self.assertEqual(first.status, "retry")
        self.assertEqual(first.attempts, 1)
        self.assertEqual(first.error, "upstream returned 503")
        self.assertIsNone(first.locked_by)
        self.assertIsNone(first.lease_expires)
        self.assert_backoff(first, 2, measured_from=before)

        # Still in backoff, so still not claimable -- that is the point of the delay.
        self.assertEqual(self.repo.claim("w1", ["discover"]), [])

        self.make_claimable(task_id)
        self.repo.claim("w1", ["discover"])
        before = self.db_now()
        second = self.repo.fail(task_id, "upstream returned 503 again")

        self.assertEqual(second.attempts, 2)
        self.assert_backoff(second, 4, measured_from=before)

    def test_the_third_failure_is_death_not_another_retry(self):
        run = self.a_run()
        task_id = self.enqueue(run)

        statuses = []
        for attempt in range(3):
            self.make_claimable(task_id)
            claimed = self.repo.claim("w1", ["discover"])
            self.assertEqual([task.attempts for task in claimed], [attempt + 1])
            statuses.append(self.repo.fail(task_id, "upstream returned 503").status)

        # max_attempts is 3 in the schema. The third delivery was the last one it was
        # entitled to, so the third failure buries it instead of queueing a fourth.
        self.assertEqual(statuses, ["retry", "retry", "dead"])

        dead = self.repo.get_task(task_id)
        self.assertEqual(dead.attempts, 3)
        self.make_claimable(task_id)
        self.assertEqual(self.repo.claim("w1", ["discover"], batch=10), [])


class ReaperTests(QueueTestCase):
    def claim_with_an_expired_lease(self, worker="w1"):
        """Claim, but backdate the lease so the reaper sees it as abandoned.

        A negative interval instead of a sleep: `now() + '-1 seconds'` is a lease that ran
        out a second ago, which is exactly the state a killed worker leaves behind and
        costs no wall-clock time to reach.
        """
        return self.repo.claim(worker, ["discover"], lease="-1 seconds")

    def test_an_expired_lease_returns_the_task_to_the_queue(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        self.claim_with_an_expired_lease()

        # Until the reaper runs the task is nobody's: 'running' is not a claimable status,
        # so a dead worker's work stays stuck without this.
        self.assertEqual(self.repo.claim("w2", ["discover"]), [])

        before = self.db_now()
        reaped = self.repo.reap_expired_leases()

        self.assertEqual([task.id for task in reaped], [task_id])
        self.assertEqual(reaped[0].status, "retry")
        self.assertIsNone(reaped[0].locked_by)
        self.assertIsNone(reaped[0].lease_expires)
        self.assertEqual(reaped[0].error, "lease expired")
        # Same backoff as an explicit failure, and for the same reason: whatever killed the
        # worker may still be happening.
        delay = reaped[0].available_at - before
        self.assertGreaterEqual(delay, timedelta(seconds=2))
        self.assertLess(delay, timedelta(seconds=3))

        self.assertEqual(self.repo.claim("w2", ["discover"]), [])
        self.make_claimable(task_id)
        requeued = self.repo.claim("w2", ["discover"])
        self.assertEqual([task.id for task in requeued], [task_id])
        self.assertEqual(requeued[0].attempts, 2)

    def test_a_live_lease_is_left_alone(self):
        run = self.a_run()
        self.enqueue(run)
        self.repo.claim("w1", ["discover"], lease="60 seconds")

        self.assertEqual(self.repo.reap_expired_leases(), [])
        self.assertEqual({row[0] for row in self.sql("SELECT status FROM tasks")}, {"running"})

    def test_the_reaper_buries_a_task_that_has_used_every_attempt(self):
        # Three crashed workers, no error message ever reported. The cap has to hold on
        # this path too, or one payload that segfaults its worker loops forever.
        run = self.a_run()
        task_id = self.enqueue(run)

        statuses = []
        for attempt in range(3):
            self.make_claimable(task_id)
            claimed = self.claim_with_an_expired_lease()
            self.assertEqual([task.attempts for task in claimed], [attempt + 1])
            statuses.append(self.repo.reap_expired_leases()[0].status)

        self.assertEqual(statuses, ["retry", "retry", "dead"])
        self.assertEqual(self.repo.get_task(task_id).attempts, 3)

    def test_the_reaper_touches_nothing_when_nothing_expired(self):
        self.assertEqual(self.repo.reap_expired_leases(), [])


class TransactionTests(QueueTestCase):
    def test_a_failing_insert_rolls_back_the_whole_run(self):
        business = self.repo.upsert_business(
            name="Cafe Noir", niche_id="cafe", city="Bangalore", dedupe_key="provider|gplace123"
        )

        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.repo.transaction():
                goal = self.repo.create_goal("blr cafes", {"city": "Bangalore"})
                run = self.repo.create_run(goal.id)
                self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-1")
                self.repo.append_event(run.id, "run.started", {})
                self.repo.insert_enrichment(business.id, "google_maps", "ok", {"reviews": 42})
                # audience_index is constrained to [0,1]. Anything that fails here would
                # do: the point is that the four writes above are already on the wire.
                self.repo.insert_score(business.id, "google-only", audience_index=1.5)

        for table in ("goals", "runs", "tasks", "events", "enrichments", "scores"):
            with self.subTest(table=table):
                self.assertEqual(self.sql(f"SELECT count(*) FROM {table}")[0][0], 0)
        # The write from before the block is untouched: the rollback is scoped to the
        # block, not to the connection's whole history.
        self.assertEqual(self.sql("SELECT count(*) FROM businesses")[0][0], 1)

    def test_a_transaction_that_completes_commits_everything(self):
        with self.repo.transaction():
            goal = self.repo.create_goal("blr cafes", {"city": "Bangalore"})
            run = self.repo.create_run(goal.id)
            self.repo.enqueue(run.id, "discover", {"tile": 1}, idem_key="tile-1")

        self.assertEqual(self.sql("SELECT count(*) FROM tasks")[0][0], 1)
        self.assertEqual([task.type for task in self.repo.claim("w1", ["discover"])], ["discover"])

    def test_transactions_do_not_nest(self):
        with self.repo.transaction():
            with self.assertRaises(RuntimeError):
                with self.repo.transaction():
                    pass


class ControlPlaneTests(QueueTestCase):
    def test_create_goal_and_run(self):
        goal = self.repo.create_goal("blr salons", {"niche": "salon"}, schedule="0 3 * * *")
        self.assertEqual(goal.name, "blr salons")
        self.assertEqual(goal.spec, {"niche": "salon"})
        self.assertEqual(goal.status, "active")

        run = self.repo.create_run(goal.id, trigger="schedule")
        self.assertEqual(run.goal_id, goal.id)
        self.assertEqual(run.status, "running")
        self.assertEqual(run.trigger, "schedule")
        self.assertIsNotNone(run.started_at)
        self.assertIsNone(run.finished_at)

    def test_finish_run_merges_stats_instead_of_replacing_them(self):
        run = self.repo.create_run(
            self.repo.create_goal("blr salons", {}).id, stats={"discovered": 40}
        )
        finished = self.repo.finish_run(run.id, "done", stats={"qualified": 12})

        self.assertEqual(finished.status, "done")
        self.assertIsNotNone(finished.finished_at)
        # A stage that recorded its own counter earlier must still find it afterwards.
        self.assertEqual(finished.stats, {"discovered": 40, "qualified": 12})

    def test_finish_run_without_stats_keeps_what_is_there(self):
        run = self.repo.create_run(
            self.repo.create_goal("blr salons", {}).id, stats={"discovered": 40}
        )
        self.assertEqual(self.repo.finish_run(run.id, "failed").stats, {"discovered": 40})

    def test_events_are_numbered_in_order_within_a_run(self):
        first_run = self.a_run()
        second_run = self.a_run()

        one = self.repo.append_event(first_run.id, "run.started", {"n": 1})
        two = self.repo.append_event(first_run.id, "task.done", {"n": 2}, task_id=None)
        other = self.repo.append_event(second_run.id, "run.started", {"n": 1})
        three = self.repo.append_event(first_run.id, "run.finished", {"n": 3})

        self.assertEqual([one.seq, two.seq, three.seq], [1, 2, 3])
        # Sequences are per run, so a second run starts at 1 rather than continuing.
        self.assertEqual(other.seq, 1)
        self.assertEqual(one.type, "run.started")
        self.assertEqual(two.payload, {"n": 2})

    def test_an_event_can_name_the_task_it_describes(self):
        run = self.a_run()
        task_id = self.enqueue(run)
        event = self.repo.append_event(run.id, "task.claimed", {}, task_id=task_id)
        self.assertEqual(event.task_id, task_id)

    def test_concurrent_appends_to_one_run_do_not_collide(self):
        # `events` has no unique index on (run_id, seq), so nothing but the advisory lock
        # in APPEND_EVENT stops two threads reading the same max(seq) and both writing it.
        run = self.a_run()
        threads = 8
        barrier = threading.Barrier(threads, timeout=30)

        def append(index):
            barrier.wait()
            return self.repo.append_event(run.id, "task.done", {"n": index}).seq

        with ThreadPoolExecutor(max_workers=threads) as executor:
            seqs = list(executor.map(append, range(threads)))

        self.assertEqual(sorted(seqs), list(range(1, threads + 1)))


class DataPlaneTests(QueueTestCase):
    def test_upsert_business_dedupes_on_the_key(self):
        first = self.repo.upsert_business(
            name="Cafe Noir",
            niche_id="cafe",
            city="Bangalore",
            dedupe_key="provider|gplace123",
            phone="9123456789",
        )
        # The same cafe, arriving from a second overlapping search tile under a different
        # spelling and carrying a website the first sighting did not have.
        second = self.repo.upsert_business(
            name="Cafe-Noir",
            niche_id="cafe",
            city="Bangalore",
            dedupe_key="provider|gplace123",
            website="https://cafenoir.example",
        )

        self.assertEqual(second.id, first.id)
        self.assertEqual(self.sql("SELECT count(*) FROM businesses")[0][0], 1)
        self.assertEqual(second.website, "https://cafenoir.example")
        # The thin second sighting must not blank the phone number the first one found.
        self.assertEqual(second.phone, "9123456789")
        self.assertGreaterEqual(second.last_seen_at, first.last_seen_at)
        self.assertEqual(second.first_seen_at, first.first_seen_at)

    def test_businesses_without_a_key_never_collide(self):
        # NULLs are distinct in Postgres, which is what lets a row be written before its
        # dedupe key has been derived. Documented here so the behaviour is a decision
        # rather than a surprise.
        one = self.repo.upsert_business(name="One", niche_id="cafe", city="Bangalore")
        two = self.repo.upsert_business(name="One", niche_id="cafe", city="Bangalore")
        self.assertNotEqual(one.id, two.id)
        self.assertEqual(self.sql("SELECT count(*) FROM businesses")[0][0], 2)

    def test_enrichments_and_scores_append_rather_than_overwrite(self):
        business = self.repo.upsert_business(
            name="Cafe Noir", niche_id="cafe", city="Bangalore", dedupe_key="k"
        )
        run = self.a_run()

        march = self.repo.insert_enrichment(
            business.id, "google_maps", "ok", {"reviews": 180}, run_id=run.id
        )
        august = self.repo.insert_enrichment(
            business.id, "google_maps", "ok", {"reviews": 340}, run_id=run.id
        )
        self.assertNotEqual(march.id, august.id)
        self.assertEqual(self.sql("SELECT count(*) FROM enrichments")[0][0], 2)
        self.assertEqual(march.run_id, run.id)

        discovery = self.repo.insert_score(
            business.id, "google-only", total=61, audience_index=0.42
        )
        enriched = self.repo.insert_score(business.id, "instagram", total=78, audience_index=0.91)
        self.assertNotEqual(discovery.id, enriched.id)
        self.assertEqual(float(enriched.audience_index), 0.91)
        self.assertEqual(enriched.scorer_version, "instagram")

    def test_a_score_with_nothing_known_keeps_a_null_audience_index(self):
        # NULL is "we know nothing", which the banding view reads as 'unknown'. Coercing it
        # to 0.0 would claim the opposite -- that we looked and found no audience.
        business = self.repo.upsert_business(name="X", niche_id="cafe", city="Bangalore")
        score = self.repo.insert_score(business.id, "google-only")
        self.assertIsNone(score.audience_index)
        self.assertEqual(score.signals, {})
        self.assertEqual(score.evidence, {})

    def test_a_failed_enrichment_is_still_recorded(self):
        business = self.repo.upsert_business(name="X", niche_id="cafe", city="Bangalore")
        row = self.repo.insert_enrichment(
            business.id, "firecrawl", "error", {"reason": "timeout"}, source_url="https://x.example"
        )
        self.assertEqual(row.status, "error")
        self.assertEqual(row.data, {"reason": "timeout"})
        self.assertEqual(row.source_url, "https://x.example")


if __name__ == "__main__":
    unittest.main()
