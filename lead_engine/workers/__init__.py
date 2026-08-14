"""The processes that perform queued work.

`POST /api/search` enqueues `discover` tasks and returns a receipt; nothing in the HTTP
layer performs them, because a request that spends a non-renewing credit is one a browser
refresh can spend twice. This package is the other half of that arrangement:

    runner    a generic claim/perform/complete loop over `Repository`. It knows about
              leases, retries, redaction and signals, and nothing about discovery.
    discover  the handler for one `discover` task: payload in, `Engine.execute_run` out.

    python -m lead_engine.workers.runner --types discover --concurrency 1

`runner` is imported eagerly because it is pure-ish -- stdlib, the row dataclasses, and the
error taxonomy -- so `Worker` can be unit-tested with a fake queue and no driver installed.
`discover` is imported lazily, for the reason `lead_engine.providers` defers `budget`:
reaching it drags in the whole composition root (FastAPI, pydantic, psycopg), and a module
that only wants the loop should not pay for the service layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .runner import Handler, TaskQueue, Worker

if TYPE_CHECKING:  # pragma: no cover
    from .discover import DiscoverHandler, build_discover_handler

_DISCOVER_NAMES = {"DiscoverHandler", "build_discover_handler"}

__all__ = [
    "DiscoverHandler",
    "Handler",
    "TaskQueue",
    "Worker",
    "build_discover_handler",
]


def __getattr__(name: str):
    if name in _DISCOVER_NAMES:
        from . import discover

        return getattr(discover, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
