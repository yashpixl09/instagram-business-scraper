"""The response cache, which is also the call ledger. One table, `llm_calls`.

WHY ONE TABLE IS BOTH
---------------------
`migrations/0007_llm.sql` puts a unique index on `prompt_hash WHERE status = 'ok'`. That
single line makes the accounting log into a cache: a successful call is recorded once and
can be read back by its prompt, and a failed call is recorded as many times as it fails.
Splitting them into two tables would mean writing the same row twice and inventing a rule
for what to do when the two disagree.

The partial predicate is the whole design. Caching a 429 as the answer to a prompt would
make one throttled minute permanent -- every re-run for the rest of the deployment's life
would serve the failure. Errors stay outside the index and therefore stay retryable.

WHAT THE HASH COVERS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------------
The digest is over the prompt text alone. Not the provider, not the model.

That is what makes a crash cheap. A run that got half its pitches from Groq before the
process died comes back, hashes the same prompts, and finds the same answers -- even for
the leads that would now route to Gemini because Groq's circuit is open. Folding the model
into the key would give every prompt three cache entries and a re-run would re-spend on
each provider in turn, which is precisely the bill the cache exists to avoid.

The cost is real and worth stating: change the model and the cache keeps serving the old
model's prose. That is correct for this system -- the artifact was already reviewed by the
operator and its evidence ids still point at the same rows -- and an operator who wants a
fresh generation deletes the row or edits the prompt. It would be wrong for a system where
the model choice is the point.

WHAT NEVER REACHES A ROW
------------------------
The prompt itself is not stored -- only its digest. Neither is a key, a header, a URL or an
upstream error body. For `status='error'` rows, `response` holds the error CODE from this
project's own five-code taxonomy (`provider_rate_limited`, and so on) and nothing else:
that string is written by `providers/errors.py`, not by a vendor, so it is safe by
construction rather than by inspection.

This module imports no database driver. It talks to whatever `execute`/`commit` object the
injected factory yields -- the same arrangement as `providers/budget.py`, and for the same
reasons: the SQL stays visible, and the non-concurrency tests run against a stub.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

#: Written to `llm_calls.status`. `ok` is the value the partial unique index keys on;
#: changing it here without changing 0007 silently turns the cache off.
STATUS_OK = "ok"
STATUS_ERROR = "error"


def prompt_hash(prompt: str) -> str:
    """The cache key: sha256 of the prompt text, hex.

    Full 64 characters, not truncated. Truncation is a readability trade in a run summary
    (see `providers/budget.fingerprint`); here a collision would serve one business's pitch
    under another business's name, which is the single worst thing this system could do.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    """A previously successful generation, replayed."""

    provider: str
    model: str
    text: str


class Cursor(Protocol):
    def fetchone(self) -> Sequence[Any] | None: ...


class Connection(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = ..., /) -> Cursor: ...

    def commit(self) -> None: ...


ConnectionFactory = Callable[[], AbstractContextManager[Connection]]


class Cache(Protocol):
    """What the router needs. `NullCache` and `ResponseCache` both satisfy it."""

    def get(self, key: str) -> CachedResponse | None: ...

    def record_success(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> None: ...

    def record_failure(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        code: str,
        latency_ms: int | None = None,
    ) -> None: ...


_LOOKUP = f"""
SELECT provider, model, response
  FROM llm_calls
 WHERE prompt_hash = %s AND status = '{STATUS_OK}'
 LIMIT 1
"""

# `ON CONFLICT (prompt_hash) WHERE status = 'ok'` infers the partial index from 0007. The
# predicate is not optional: without it Postgres looks for a total unique index on
# `prompt_hash`, finds none, and raises rather than doing nothing.
#
# DO NOTHING rather than DO UPDATE. Two workers racing on the same prompt both generated a
# valid answer; the first one committed is as good as the second, and overwriting means the
# text an operator is reading can change under them mid-run.
_RECORD_OK = f"""
INSERT INTO llm_calls
       (prompt_hash, provider, model, input_tokens, output_tokens, latency_ms, status, response)
VALUES (%s, %s, %s, %s, %s, %s, '{STATUS_OK}', %s)
ON CONFLICT (prompt_hash) WHERE status = '{STATUS_OK}' DO NOTHING
"""

_RECORD_ERROR = f"""
INSERT INTO llm_calls (prompt_hash, provider, model, latency_ms, status, response)
VALUES (%s, %s, %s, %s, '{STATUS_ERROR}', %s)
"""


class NullCache:
    """The no-database configuration: remembers nothing, costs nothing, never fails.

    Not a test double. A run against a machine with no Postgres still has to produce a
    sheet, so "no cache" is a supported configuration in exactly the way "no LLM provider"
    is, and the router must not branch on which one it was handed.
    """

    def get(self, key: str) -> CachedResponse | None:
        return None

    def record_success(self, key: str, **_: Any) -> None:
        return None

    def record_failure(self, key: str, **_: Any) -> None:
        return None


class ResponseCache:
    """`llm_calls`, read as a cache and written as a ledger.

    Stateless apart from the connection factory, so one instance is safe to share across
    the threads of a worker. It commits its own writes: a generation that was paid for and
    left uncommitted when the process died is a credit spent for nothing, which is the
    failure this table exists to prevent.
    """

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect

    def get(self, key: str) -> CachedResponse | None:
        """The recorded answer for this prompt, or None.

        Rows whose `response` is NULL are treated as a miss. A successful call always has
        text; a row that says otherwise is damaged, and serving an empty pitch from it is
        worse than paying for the call again.
        """
        with self._connect() as conn:
            row = conn.execute(_LOOKUP, (key,)).fetchone()
            conn.commit()
        if row is None or row[2] is None:
            return None
        return CachedResponse(provider=str(row[0]), model=str(row[1]), text=str(row[2]))

    def record_success(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                _RECORD_OK,
                (key, provider, model, input_tokens, output_tokens, latency_ms, text),
            )
            conn.commit()

    def record_failure(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        code: str,
        latency_ms: int | None = None,
    ) -> None:
        """Record a failed attempt. Never cached -- see the partial index in 0007.

        `code` is one of this project's five error codes. It is not a message, and it is
        never upstream text: the whole point of `providers/errors.py` is that the vendor's
        words do not travel, and a ledger row is exactly the kind of place they would end
        up being read from months later.
        """
        with self._connect() as conn:
            conn.execute(_RECORD_ERROR, (key, provider, model, latency_ms, code))
            conn.commit()
