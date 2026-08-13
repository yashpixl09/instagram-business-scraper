"""The seam every external service sits behind.

Two things live here and nothing else yet:

  `errors`  -- the five-code taxonomy every provider client raises, and only that. A client
               that lets an httpx exception escape has moved its vendor's failure modes
               into the API layer, which then has to know what a `ReadTimeout` means.
  `budget`  -- the lifetime ledger for billed searches, because SearchAPI's 100-search
               allocation does not renew and a per-process counter cannot protect it.

The clients themselves take an injectable transport (`transport=` with
`httpx.MockTransport`) so no test in this repo ever opens a socket. Nothing in this package
does I/O of its own: `errors` is pure, and `budget` talks only to the connection factory it
is handed.

`budget` is imported lazily below. It needs no driver at import time, but keeping the
package's own import free of database concepts means `from lead_engine.providers import
errors` stays a pure import for the API layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import (
    CODES,
    DEFAULT_MESSAGE,
    RETRYABLE_BY_CODE,
    STATUS_BY_CODE,
    ProviderError,
    auth_failed,
    bad_response,
    from_status,
    location_not_found,
    rate_limited,
    redact,
    unavailable,
)

if TYPE_CHECKING:  # pragma: no cover
    from .budget import BudgetExhausted, BudgetNotConfigured, BudgetState, SearchBudget

_BUDGET_NAMES = {"SearchBudget", "BudgetExhausted", "BudgetNotConfigured", "BudgetState"}

__all__ = [
    "CODES",
    "DEFAULT_MESSAGE",
    "RETRYABLE_BY_CODE",
    "STATUS_BY_CODE",
    "BudgetExhausted",
    "BudgetNotConfigured",
    "BudgetState",
    "ProviderError",
    "SearchBudget",
    "auth_failed",
    "bad_response",
    "from_status",
    "location_not_found",
    "rate_limited",
    "redact",
    "unavailable",
]


def __getattr__(name: str):
    if name in _BUDGET_NAMES:
        from . import budget

        return getattr(budget, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
