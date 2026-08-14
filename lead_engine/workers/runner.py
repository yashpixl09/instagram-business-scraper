"""The queue worker: claim, renew, perform, record. Generic over what a task actually is.

    worker = Worker(repository, {"discover": handler}, "worker-1", ["discover"])
    worker.run_forever()

    python -m lead_engine.workers.runner --types discover --concurrency 1

Nothing below knows what discovery is. It knows that a task is claimed under a lease, that
the lease can be lost, that a handler may fail in three distinguishable ways, and that every
one of those has to leave a record. `discover.py` supplies the meaning.

WHY THE LEASE IS RENEWED AT THE START OF EVERY TASK
---------------------------------------------------
`claim(batch=n)` stamps every row it returns from ONE `now()`, evaluated once for the
statement. A worker that takes five tasks and performs them serially is therefore four
tasks' runtime into the fifth one's lease before it looks at it, and the reaper reclaims the
tail of every batch while the worker is still working it. A second worker then picks those
tasks up and performs them again -- for a discovery task, a second billed Google Maps search
for a cell already swept, on an allowance of fifty that never renews.

So the lease is pushed out from now as each task BEGINS, and `renew_lease` returning None is
treated as the stop signal the repository says it is: the reaper has already taken the task
back and somebody else owns it. The task is abandoned, not performed. Continuing "because we
were nearly there anyway" is precisely how one cell becomes two billed searches.

`complete()` and `fail()` carry the same guard on the far side: both return None when this
worker no longer owns the row. That is logged and dropped. It is never retried -- a retry
would be a second worker's result written over the first one's.

THE THREE WAYS A HANDLER CAN STOP, AND WHY THEY ARE NOT ONE
-----------------------------------------------------------
`BudgetExhausted`  is NOT a failure. The ledger refused before the request went out, so
                   nothing was billed and nothing broke. The task is COMPLETED, the run is
                   marked `stopped`, and this worker stops claiming that type. It must never
                   become a `dead` task or a 5xx: see `providers/budget.py`.

`ProviderError`    is a failure, and `fail()` applies the backoff and `max_attempts` the
                   schema owns. `retryable=False` fails too -- there is no "park it" state --
                   but the message says so, because a row that reads like a transient will be
                   retried by hand until someone reads the code.

anything else      is a bug, and its message NEVER reaches `tasks.error` verbatim. Every
                   provider in this project authenticates by query string, and an exception
                   from an HTTP client carries `.request.url`. `tasks.error` is read by the
                   API, printed in run summaries and pasted into bug reports, so what lands
                   there is the exception's class name and nothing else. The worker log gets
                   the full traceback with `redact()` run over it.

Every claim, completion and failure appends to `events`, which is the log a run is replayed
from. A task that vanished into a worker with no trace is a run nobody can explain.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import signal
import sys
import threading
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from typing import Any, Protocol
from uuid import UUID

from ..db.rows import TaskRow
from ..providers.budget import BudgetExhausted
from ..providers.errors import ProviderError, redact

logger = logging.getLogger("lead_engine.workers")

#: What a handler is: one task in, the JSON to store in `tasks.result` out.
Handler = Callable[[TaskRow], "dict[str, Any] | None"]

#: Long enough that a discovery pass (one billed search, a handful of writes, a sheet)
#: finishes inside it, short enough that a killed worker's tasks come back in two minutes
#: rather than an hour. It is renewed as each task begins regardless.
DEFAULT_LEASE = "120 seconds"

#: How long an idle worker waits before asking again. There is no LISTEN/NOTIFY here: at one
#: search per cell and fifty credits in total, two seconds of latency is not the bottleneck.
DEFAULT_POLL_INTERVAL = 2.0

#: Event types this loop appends. Prefixed `task.` so a replay can tell the worker's own
#: bookkeeping from the domain events a handler writes.
CLAIMED = "task.claimed"
COMPLETED = "task.completed"
FAILED = "task.failed"
ABANDONED = "task.abandoned"
LOST = "task.lost"
BUDGET_STOPPED = "task.budget_exhausted"

#: What `runs.status` becomes when the allowance runs out. Not 'failed': nothing failed.
STOPPED = "stopped"

#: `tasks.result.stopped` for the same event, matching `discovery.service.BUDGET_EXHAUSTED`
#: so a reader does not have to learn two spellings of one fact.
BUDGET_EXHAUSTED = "budget_exhausted"

EXIT_OK = 0
EXIT_CANNOT_START = 2


class TaskQueue(Protocol):
    """What this loop needs from `db.repository.Repository`, and nothing more.

    Structural rather than the concrete class, for two reasons. A unit test can hand the
    worker an in-memory queue and exercise every branch below with no database anywhere
    near it -- including the ones that are awkward to provoke against real Postgres -- and
    the loop stays honest about the fact that it uses six methods out of twenty.
    """

    def claim(
        self, worker_id: str, types: Sequence[str], batch: int = ..., lease: str = ...
    ) -> list[TaskRow]: ...

    def renew_lease(self, task_id: int, worker_id: str, lease: str = ...) -> TaskRow | None: ...

    def complete(self, task_id: int, result: dict[str, Any] | None = ...) -> TaskRow | None: ...

    def fail(self, task_id: int, error: str) -> TaskRow | None: ...

    def append_event(
        self, run_id: UUID, type: str, payload: dict[str, Any], *, task_id: int | None = ...
    ) -> Any: ...

    def finish_run(
        self, run_id: UUID, status: str, *, stats: dict[str, Any] | None = ...
    ) -> Any: ...

    def reap_expired_leases(self) -> list[TaskRow]: ...


def default_worker_id() -> str:
    """`host:pid:random`. Unique per process AND per restart.

    The random tail is the part that matters: a container restarted onto the same host with
    the same pid would otherwise inherit its predecessor's identity and be able to renew and
    complete tasks the reaper had already handed to somebody else.
    """
    return f"{platform.node() or 'worker'}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def provider_failure(exc: ProviderError) -> str:
    """The failure text for a classified provider error. Safe by construction.

    `ProviderError` refuses to keep a message that `redact()` would change, so `exc.message`
    is either something this codebase wrote or the vendor-neutral default. `redact()` runs
    again here anyway: this is the last place before the text becomes a database row that
    the API will serve back, and one more pass over already-clean prose costs nothing.

    A non-retryable error still fails the task -- there is no state between `retry` and
    `dead`, and inventing one would mean a task nobody ever looks at. What changes is the
    message: an operator who reads "rate limited" retries by hand and is right to; an
    operator who reads a bad key as a transient retries three times and then files a bug
    against the queue.
    """
    text = f"{exc.code}: {exc.message}"
    if not exc.retryable:
        text += (
            " This will not succeed on retry: the request itself is the problem, not the "
            "moment it was made. The attempts left will buy the same answer."
        )
    return redact(text)


def redacted_failure(exc: BaseException) -> str:
    """What an unhandled exception is allowed to write into `tasks.error`: its class name.

    Not `str(exc)`, and not `redact(str(exc))` either. Scrubbing an interpolated response
    body leaves the rest of the body behind -- quota numbers, account fields, whatever the
    vendor echoed back -- and the residue reads as sanitised, so nobody looks again. The
    same reasoning as `providers/errors.py:_safe_message`, applied one layer further out.

    The class name is written by this codebase or by a library, never by an upstream, and it
    is run through `redact()` regardless because it costs one function call to be sure.
    """
    return (
        f"unhandled {redact(type(exc).__name__)} in the task handler. The message is "
        "withheld deliberately -- an exception from an HTTP client carries the request URL, "
        "and every provider here authenticates by query string. The worker log has the "
        "redacted traceback."
    )


def redacted_traceback(exc: BaseException) -> str:
    """The whole traceback, with URLs, bearer tokens and credential-shaped runs stripped.

    The detail has to go somewhere or nothing is debuggable, and a log line this process
    controls is the right somewhere -- but "it is only a log" is how keys reach aggregators,
    so it is scrubbed on the way out rather than on the way in.
    """
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return redact(formatted)


class Worker:
    """One claim/perform/complete loop over one queue.

    `types` defaults to the handler names, and a type with no handler is refused at
    construction: claiming work this process cannot perform would hold it under a lease for
    nothing and hand it back to the queue when the lease expired, one attempt poorer.
    """

    def __init__(
        self,
        repository: TaskQueue,
        handlers: Mapping[str, Handler],
        worker_id: str | None = None,
        types: Sequence[str] | None = None,
        batch: int = 1,
        lease: str = DEFAULT_LEASE,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        *,
        reap_interval: float | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("a worker with no handlers would claim work it cannot perform")
        self.repository = repository
        self.handlers: dict[str, Handler] = dict(handlers)
        self.worker_id = worker_id or default_worker_id()
        self.types: tuple[str, ...] = tuple(types) if types is not None else tuple(
            sorted(self.handlers)
        )
        unknown = [name for name in self.types if name not in self.handlers]
        if unknown:
            raise ValueError(f"no handler for task type(s): {', '.join(sorted(unknown))}")
        if int(batch) < 1:
            raise ValueError("batch must be at least 1")
        self.batch = int(batch)
        self.lease = lease
        self.poll_interval = float(poll_interval)
        self.reap_interval = None if reap_interval is None else float(reap_interval)
        #: Types whose allowance ran out. Claiming them again would buy nothing: the ledger
        #: refuses before the request goes out, so every task would come straight back here.
        self.stopped_types: set[str] = set()
        self._stopping = threading.Event()
        self._sleeper = sleeper
        self._reaped_at: float | None = None

    # --- state ---------------------------------------------------------------------------

    @property
    def claimable_types(self) -> tuple[str, ...]:
        return tuple(name for name in self.types if name not in self.stopped_types)

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def stop(self) -> None:
        """Ask the loop to finish the task in hand and exit. Safe from a signal handler."""
        self._stopping.set()

    # --- the loop ------------------------------------------------------------------------

    def run_once(self) -> int:
        """Claim one batch, perform it, and return how many tasks this worker performed.

        Zero means there was nothing to do -- or nothing this worker could take, since
        `SKIP LOCKED` hands back an empty batch whenever peers hold the claimable rows.
        Either way the caller should wait before asking again.

        The count is of tasks HANDED TO A HANDLER, so a task abandoned because its lease had
        already been reclaimed is not counted: nothing was performed, and counting it would
        report work this worker did not do.
        """
        types = self.claimable_types
        if not types:
            return 0

        tasks = self.repository.claim(self.worker_id, types, batch=self.batch, lease=self.lease)
        performed = 0
        for index, task in enumerate(tasks):
            if self.stopping or task.type in self.stopped_types:
                self._leave_to_the_reaper(tasks[index:])
                break
            if self._perform(task):
                performed += 1
        return performed

    def run_forever(self, *, install_signal_handlers: bool = True) -> int:
        """Poll until stopped, and return how many tasks were performed in total.

        Stops on SIGINT/SIGTERM, and stops on its own once every type it serves has hit its
        budget: a worker that kept polling after the allowance was gone would claim, refuse
        and release each task in turn, burning an attempt apiece until they all died.
        """
        total = 0
        with stop_on_signals([self] if install_signal_handlers else []):
            while not self.stopping:
                if not self.claimable_types:
                    logger.info(
                        "%s: every type this worker serves has stopped; exiting", self.worker_id
                    )
                    break
                self._maybe_reap()
                performed = self.run_once()
                total += performed
                if performed == 0 and not self.stopping:
                    self._wait(self.poll_interval)
        logger.info("%s: stopped after performing %d task(s)", self.worker_id, total)
        return total

    # --- one task ------------------------------------------------------------------------

    def _perform(self, task: TaskRow) -> bool:
        """Renew, run the handler, record what happened. True when the handler ran."""
        self._event(
            task,
            CLAIMED,
            {"type": task.type, "worker": self.worker_id, "attempt": task.attempts},
        )

        # Push this task's deadline out from NOW, before any work begins. See the module
        # docstring: the batch was stamped from one clock reading and this is the only thing
        # that stops the tail of it being reaped mid-flight.
        if self.repository.renew_lease(task.id, self.worker_id, self.lease) is None:
            # The reaper already took it back, or another worker holds it. Whatever this
            # task was about to buy, somebody else is buying it.
            logger.warning(
                "%s: task %s (%s) was reclaimed before it began; abandoning it unperformed",
                self.worker_id,
                task.id,
                task.type,
            )
            self._event(
                task,
                ABANDONED,
                {"type": task.type, "worker": self.worker_id, "reason": "lease_lost"},
            )
            return False

        try:
            result = self.handlers[task.type](task)
        except BudgetExhausted as exc:
            self._budget_stop(task, exc)
            return True
        except ProviderError as exc:
            logger.warning(
                "%s: task %s (%s) failed: %s (retryable=%s)",
                self.worker_id,
                task.id,
                task.type,
                exc.code,
                exc.retryable,
            )
            self._fail(task, provider_failure(exc))
            return True
        except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the worker
            logger.error(
                "%s: task %s (%s) raised %s; redacted traceback follows:\n%s",
                self.worker_id,
                task.id,
                task.type,
                type(exc).__name__,
                redacted_traceback(exc),
            )
            self._fail(task, redacted_failure(exc))
            return True

        self._complete(task, result)
        return True

    def _complete(self, task: TaskRow, result: dict[str, Any] | None) -> None:
        row = self.repository.complete(task.id, result)
        if row is None:
            # Only a `running` task may be completed, so None means this worker lost the
            # row -- reaped, and by now possibly finished by somebody else. Recording the
            # result anyway would overwrite theirs; retrying would perform the task twice.
            self._lost(task, "complete")
            return
        self._event(
            task,
            COMPLETED,
            {
                "type": task.type,
                "worker": self.worker_id,
                "attempts": row.attempts,
                "result": result if isinstance(result, dict) else {},
            },
        )

    def _fail(self, task: TaskRow, error: str) -> None:
        row = self.repository.fail(task.id, error)
        if row is None:
            self._lost(task, "fail")
            return
        # 'retry' with the backoff the schema computes, or 'dead' at max_attempts. The
        # decision is Postgres's, in one statement, so two workers cannot disagree about it.
        self._event(
            task,
            FAILED,
            {
                "type": task.type,
                "worker": self.worker_id,
                "attempts": row.attempts,
                "status": row.status,
                "error": error,
            },
        )

    def _budget_stop(self, task: TaskRow, exc: BudgetExhausted) -> None:
        """The clean stop. Complete the task, close the run, stop claiming this type.

        Not a failure in any of the three senses that matter: the task is `done` and not
        `retry`, the run is `stopped` and not `failed`, and the loop ends rather than
        spending the remaining attempts of every queued task discovering the same thing.
        """
        result = {
            "stopped": BUDGET_EXHAUSTED,
            "provider": exc.provider,
            "purpose": exc.purpose,
            "limit_total": exc.limit_total,
            "used": exc.used,
            "remaining": exc.remaining,
        }
        self.stopped_types.add(task.type)
        logger.info(
            "%s: %s budget exhausted (%s/%s used); %r tasks stop here. Nothing failed.",
            self.worker_id,
            exc.provider,
            exc.used,
            exc.limit_total,
            task.type,
        )
        self._event(task, BUDGET_STOPPED, dict(result, worker=self.worker_id))
        self._complete(task, result)
        self.repository.finish_run(task.run_id, STOPPED, stats=result)

    def _lost(self, task: TaskRow, phase: str) -> None:
        logger.warning(
            "%s: task %s (%s) was no longer ours at %s; dropping the outcome without retrying",
            self.worker_id,
            task.id,
            task.type,
            phase,
        )
        self._event(
            task, LOST, {"type": task.type, "worker": self.worker_id, "phase": phase}
        )

    def _leave_to_the_reaper(self, tasks: Sequence[TaskRow]) -> None:
        """Log the tail of a batch this worker will not perform.

        There is no "un-claim": a claimed row is `running` until it is completed, failed, or
        reaped. Failing these would be a lie (nothing went wrong) and would spend an attempt
        each toward `dead`; completing them would mark cells searched that nobody searched,
        and on a non-renewing allowance a cell wrongly marked done is a cell never bought.
        So they are left, and the reaper returns them when the lease runs out. This is the
        reason `batch` defaults to 1.
        """
        if not tasks:
            return
        logger.info(
            "%s: leaving %d claimed task(s) unperformed for the reaper: %s",
            self.worker_id,
            len(tasks),
            ", ".join(str(task.id) for task in tasks),
        )

    def _event(self, task: TaskRow, type: str, payload: dict[str, Any]) -> None:
        self.repository.append_event(task.run_id, type, payload, task_id=task.id)

    # --- plumbing ------------------------------------------------------------------------

    def _maybe_reap(self) -> None:
        """Return other workers' expired leases to the queue, at most every `reap_interval`.

        Off unless asked for. A crashed worker reports nothing at all, so an expired lease is
        the only evidence its tasks exist, and something has to be on that timer -- but a
        library that reaped by default would surprise a test that deliberately holds a lease.
        """
        if self.reap_interval is None:
            return
        now = _monotonic()
        if self._reaped_at is not None and now - self._reaped_at < self.reap_interval:
            return
        self._reaped_at = now
        reaped = self.repository.reap_expired_leases()
        if reaped:
            logger.warning(
                "%s: returned %d expired task(s) to the queue: %s",
                self.worker_id,
                len(reaped),
                ", ".join(str(task.id) for task in reaped),
            )

    def _wait(self, seconds: float) -> None:
        if self._sleeper is not None:
            self._sleeper(seconds)
            return
        # `Event.wait`, not `sleep`: a stop set by another thread ends the wait immediately.
        self._stopping.wait(seconds)


def _monotonic() -> float:
    from time import monotonic

    return monotonic()


@contextmanager
def stop_on_signals(workers: Iterable[Worker]):
    """Turn SIGINT/SIGTERM into `Worker.stop()` for the duration of a block.

    Clean means the task in hand is finished and recorded before the process exits. Dying
    mid-task is not fatal -- the lease expires and the reaper returns the task -- but it
    costs an attempt and, for discovery, re-buys whatever the task had already spent.

    Handlers can only be installed from the main thread; `signal.signal` raises ValueError
    anywhere else. That is caught rather than avoided so that a Worker running inside a
    thread (a test, or `--concurrency 2`) behaves identically minus the handlers, which the
    main thread installs on its behalf.
    """
    targets = list(workers)
    previous: dict[Any, Any] = {}

    def handle(signum: int, frame: Any) -> None:  # pragma: no cover - needs a real signal
        logger.info("signal %s received: finishing the task in hand, then stopping", signum)
        for worker in targets:
            worker.stop()

    for name in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if name is None or not targets:
            continue
        try:
            previous[name] = signal.signal(name, handle)
        except (ValueError, OSError, RuntimeError):
            logger.debug("cannot install a handler for %s here; the main thread owns it", name)
    try:
        yield
    finally:
        for name, handler in previous.items():
            with suppress(ValueError, OSError, RuntimeError, TypeError):
                signal.signal(name, handler)


def run_workers(workers: Sequence[Worker]) -> int:
    """Run every worker until it stops. One set of signal handlers covers all of them."""
    if not workers:
        return 0
    if len(workers) == 1:
        return workers[0].run_forever()

    performed = [0] * len(workers)

    def run(index: int) -> None:
        performed[index] = workers[index].run_forever(install_signal_handlers=False)

    with stop_on_signals(workers):
        threads = [
            threading.Thread(target=run, args=(index,), name=worker.worker_id)
            for index, worker in enumerate(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return sum(performed)


# --- the entry point --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lead-engine-worker",
        description=(
            "Perform the tasks POST /api/search enqueues. One task is one (area x niche) "
            "cell and at least one billed search, so nothing here runs without a ledger."
        ),
    )
    parser.add_argument(
        "--types",
        action="append",
        default=[],
        metavar="TYPE",
        help="Task type to serve; repeatable, or comma-separated. Default: discover.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Worker threads, each with its own id and its own claims (default 1).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "Tasks to claim at once (default 1). A batch's leases all start together, so "
            "anything above 1 leaves the tail of an interrupted batch to the reaper."
        ),
    )
    parser.add_argument(
        "--lease",
        default=DEFAULT_LEASE,
        help=f"Postgres interval a claimed task is held for (default {DEFAULT_LEASE!r}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds to wait when the queue is empty (default {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--reap-interval",
        type=float,
        default=60.0,
        help=(
            "Seconds between sweeps that return expired leases to the queue; 0 disables it. "
            "Without this, a worker killed mid-task leaves its tasks running forever."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim and perform one batch, then exit. For cron and for smoke tests.",
    )
    parser.add_argument(
        "--fixtures",
        default=None,
        metavar="DIR",
        help=(
            "Serve recorded responses from DIR instead of calling SearchAPI. Spends "
            "nothing; this is how the queue is demonstrated end to end for free."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Write a workbook per completed task into DIR. Off by default.",
    )
    parser.add_argument("--worker-id", default=None, help="Override the generated worker id.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default INFO).")
    return parser


def requested_types(values: Sequence[str], default: str) -> tuple[str, ...]:
    """`--types discover --types enrich` and `--types discover,enrich` mean the same thing."""
    names: list[str] = []
    for value in values:
        for part in str(value).split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return tuple(names) or (default,)


def main(
    argv: Sequence[str] | None = None,
    *,
    repository: TaskQueue | None = None,
    handlers: Mapping[str, Handler] | None = None,
    settings: Any | None = None,
) -> int:
    """Wire a process up and run it. Returns an exit code rather than raising SystemExit.

    This function is the only part of the module that knows what a database or a discovery
    task is -- everything above is the loop -- which is why `Repository`, `Settings` and the
    discover handler are imported here rather than at the top.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    from ..config import Settings

    settings = settings if settings is not None else Settings()
    owns_handlers = handlers is None
    if handlers is None:
        from .discover import DISCOVER, build_discover_handler

        handlers = {
            DISCOVER: build_discover_handler(
                settings, fixture_dir=args.fixtures, output_dir=args.output_dir
            )
        }

    types = requested_types(args.types, default=next(iter(handlers)))
    missing = [name for name in types if name not in handlers]
    if missing:
        print(
            f"unknown_task_type: no handler for {', '.join(missing)}. "
            f"This process serves: {', '.join(sorted(handlers))}.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START

    owns_repository = repository is None
    if repository is None:
        if not settings.dsn:
            print(
                "database_not_configured: LEAD_ENGINE_DSN is not set. The queue, the leads "
                "and the ledger all live in Postgres; there is nothing to work from.",
                file=sys.stderr,
            )
            return EXIT_CANNOT_START
        from ..db.repository import Repository

        # One spare connection above the thread count: every worker holds at most one at a
        # time, and the spare absorbs the event append that follows a completion.
        repository = Repository.from_dsn(
            settings.dsn, min_size=1, max_size=max(2, int(args.concurrency) + 1)
        )

    try:
        workers = [
            Worker(
                repository,
                handlers,
                worker_id=_worker_id(args, index),
                types=types,
                batch=args.batch,
                lease=args.lease,
                poll_interval=args.poll_interval,
                reap_interval=args.reap_interval if args.reap_interval > 0 else None,
            )
            for index in range(max(1, int(args.concurrency)))
        ]
        logger.info(
            "serving %s with %d worker(s), batch=%d, lease=%s",
            ", ".join(types),
            len(workers),
            args.batch,
            args.lease,
        )
        if args.once:
            performed = sum(worker.run_once() for worker in workers)
        else:
            performed = run_workers(workers)
        logger.info("performed %d task(s)", performed)
        return EXIT_OK
    finally:
        if owns_handlers:
            for handler in handlers.values():
                close = getattr(handler, "close", None)
                if callable(close):
                    close()
        if owns_repository:
            close_repository = getattr(repository, "close", None)
            if callable(close_repository):
                close_repository()


def _worker_id(args: argparse.Namespace, index: int) -> str:
    if not args.worker_id:
        return default_worker_id()
    return args.worker_id if args.concurrency == 1 else f"{args.worker_id}-{index}"


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
