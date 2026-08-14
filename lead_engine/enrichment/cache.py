"""Negative caching: paying for an absence once, then less and less often.

Discovery's waste is re-searching ground it has already swept. Enrichment's waste is the
exact inverse -- it re-purchases the discovery that there is nothing there. A salon with no
Instagram in March has no Instagram in April, and asking again every run buys the same empty
answer at full price forever.

So absence is cached, and the retry interval grows:

    miss 1   ->  retry in 30 days
    miss 2   ->  retry in 90 days
    miss 3   ->  retry in 180 days
    miss 4+  ->  never, until somebody resets it by hand

`never` is deliberate, and `migrations/0004_discovery.sql` writes it as a NULL
`retry_after`. A business that has been checked four times over a year and has produced
nothing four times is not going to produce something on the fifth automatic attempt; an
automatic retry that keeps finding nothing is a subscription to nothing. `reset()` exists
for the case where an operator knows better.

THE RULE THAT MAKES THIS SAFE
-----------------------------
Only an OBSERVED absence advances the ladder. `service.py` calls `record_miss` for a
`no_data` and for nothing else -- never for `error`, never for `blocked`. The difference is
the whole reason this phase carries four statuses instead of two: a rate-limited afternoon
that recorded misses would buy 180 days of not looking at businesses nobody ever actually
checked, and the backoff would hide the outage rather than survive it.

SUCCESSES ARE NOT NEGATIVE CACHE ENTRIES
----------------------------------------
A success deletes the row -- the ladder resets, because the next absence is a new fact
about a business that has changed. What limits re-asking after a success is ordinary
freshness: the newest `ok` row in `enrichments` for that question, against the TTL table
the spec sets (30 days for the web and Instagram questions, 14 for the ad library, because
ad campaigns turn over faster than websites do).

Two implementations, one protocol. `InMemoryCache` is the seam that lets the backoff ladder
be tested across simulated years in no time at all; `PostgresCache` is the same arithmetic
expressed as one upsert so that eight enrichment workers cannot read-modify-write over each
other. `tests/test_enrichment.py` runs the same ladder assertions against both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

# --- the questions ------------------------------------------------------------------------

#: "Does this business have a website?"
WEBSITE = "website"
#: "What is its Instagram handle?"
INSTAGRAM = "instagram_handle"
#: "Is it buying Meta ads?"
ADS = "ad_library"

QUESTIONS: tuple[str, ...] = (WEBSITE, INSTAGRAM, ADS)

#: Which `enrichments.source` values answer each question.
#:
#: The cache keys the QUESTION and the enrichment row names the ANSWERER, which is why
#: `website` maps to two sources. Keying the cache by answerer instead would let the
#: Firecrawl path re-buy an absence the TinyFish path had already established, at one
#: credit a time, on the one provider in this system that meters.
SOURCES_BY_QUESTION: dict[str, tuple[str, ...]] = {
    WEBSITE: ("tinyfish_web", "firecrawl_web"),
    INSTAGRAM: ("instagram_handle",),
    ADS: ("ad_library",),
}

#: The spec's freshness table. A successful answer is re-asked only after this long.
TTL_DAYS: dict[str, int] = {
    WEBSITE: 30,
    INSTAGRAM: 30,
    ADS: 14,
}

#: The backoff ladder, in days, by consecutive miss count. Running off the end means
#: `never` -- a NULL `retry_after`.
BACKOFF_DAYS: tuple[int, ...] = (30, 90, 180)

# --- why a lookup was skipped --------------------------------------------------------------

#: Backing off: the absence is cached and the retry date has not arrived.
SKIP_NEGATIVE = "negative_cache"
#: The ladder ran out. Manual reset only.
SKIP_EXHAUSTED = "negative_cache_exhausted"
#: A recent success. Nothing is wrong; the answer is simply still good.
SKIP_FRESH = "fresh"


def utc_now() -> datetime:
    return datetime.now(UTC)


def retry_after_for(misses: int, now: datetime) -> datetime | None:
    """When to ask again after `misses` consecutive empty answers. None means never."""
    if misses < 1:
        raise ValueError("misses must be at least 1")
    if misses > len(BACKOFF_DAYS):
        return None
    return now + timedelta(days=BACKOFF_DAYS[misses - 1])


@dataclass(frozen=True)
class CacheEntry:
    """One (business, question) pair's memory of having found nothing."""

    business_id: UUID
    question: str
    misses: int = 0
    retry_after: datetime | None = None
    last_ok_at: datetime | None = None

    @property
    def exhausted(self) -> bool:
        """The ladder ran out: recorded misses, and no date on which to try again."""
        return self.misses >= 1 and self.retry_after is None


class EnrichmentCache(Protocol):
    """What `service.py` is allowed to do with the cache. Deliberately four methods.

    There is no "write an arbitrary retry date" on this interface. The ladder is the
    policy, and a caller that could set its own interval would be a second policy waiting
    to disagree with the first.
    """

    def skip_reason(self, business_id: UUID, question: str, *, now: datetime) -> str | None:
        """Why this lookup should not be issued, or None if it should."""
        ...

    def record_miss(self, business_id: UUID, question: str, *, now: datetime) -> CacheEntry:
        """Record ONE observed absence and advance the ladder."""
        ...

    def record_success(self, business_id: UUID, question: str, *, now: datetime) -> None:
        """Clear the ladder. The next absence starts again at 30 days."""
        ...

    def reset(self, business_id: UUID, question: str) -> None:
        """The manual reset the fourth miss requires."""
        ...


def _validate(question: str) -> str:
    if question not in SOURCES_BY_QUESTION:
        raise ValueError(f"unknown enrichment question: {question!r}")
    return question


def _ttl(question: str) -> timedelta:
    return timedelta(days=TTL_DAYS[question])


# --- in memory ------------------------------------------------------------------------------


@dataclass
class InMemoryCache:
    """The ladder without a database. Used by the unit suite and by dry runs.

    Not a stub: it is the same arithmetic as `PostgresCache`, and the ladder tests run
    against both so the SQL and the Python cannot drift.
    """

    entries: dict[tuple[UUID, str], CacheEntry] = field(default_factory=dict)

    def skip_reason(self, business_id: UUID, question: str, *, now: datetime) -> str | None:
        entry = self.entries.get((business_id, _validate(question)))
        if entry is None:
            return None
        if entry.misses >= 1:
            if entry.retry_after is None:
                return SKIP_EXHAUSTED
            if now < entry.retry_after:
                return SKIP_NEGATIVE
            return None
        if entry.last_ok_at is not None and now - entry.last_ok_at < _ttl(question):
            return SKIP_FRESH
        return None

    def record_miss(self, business_id: UUID, question: str, *, now: datetime) -> CacheEntry:
        key = (business_id, _validate(question))
        current = self.entries.get(key)
        misses = (current.misses if current else 0) + 1
        entry = CacheEntry(
            business_id=business_id,
            question=question,
            misses=misses,
            retry_after=retry_after_for(misses, now),
            last_ok_at=current.last_ok_at if current else None,
        )
        self.entries[key] = entry
        return entry

    def record_success(self, business_id: UUID, question: str, *, now: datetime) -> None:
        key = (business_id, _validate(question))
        current = self.entries.get(key)
        self.entries[key] = (
            replace(current, misses=0, retry_after=None, last_ok_at=now)
            if current
            else CacheEntry(business_id=business_id, question=question, last_ok_at=now)
        )

    def reset(self, business_id: UUID, question: str) -> None:
        self.entries.pop((business_id, _validate(question)), None)


class NullCache:
    """Never skips, never remembers. For a forced re-check of a whole cohort.

    Exists so that "ignore the cache" is a different OBJECT rather than a boolean argument
    threaded through the service -- a flag would end up defaulting the wrong way in one
    call site and quietly re-buying every absence in the corpus.
    """

    def skip_reason(self, business_id: UUID, question: str, *, now: datetime) -> str | None:
        return None

    def record_miss(self, business_id: UUID, question: str, *, now: datetime) -> CacheEntry:
        return CacheEntry(business_id=business_id, question=_validate(question), misses=1)

    def record_success(self, business_id: UUID, question: str, *, now: datetime) -> None:
        return None

    def reset(self, business_id: UUID, question: str) -> None:
        return None


# --- postgres -------------------------------------------------------------------------------

ConnectionFactory = Callable[[], AbstractContextManager[Any]]

SELECT_ENTRY = """
SELECT misses, retry_after
  FROM negative_cache
 WHERE business_id = %(business_id)s AND source = %(question)s
"""

# The whole ladder in one statement. The alternative -- SELECT the miss count, compute the
# next date in Python, UPDATE -- is a read-modify-write, and `worker-enrich` runs eight of
# these at once. Two workers recording a miss for the same business would both read 1, both
# write 2, and the business would sit at 90 days having actually been checked three times.
#
# The days array is passed as a parameter and indexed by the POST-increment miss count.
# Postgres arrays are 1-based, so `days[1]` is the first rung. Running past the end yields
# NULL, which is exactly `never`.
RECORD_MISS = """
INSERT INTO negative_cache AS nc (business_id, source, misses, retry_after, updated_at)
VALUES (
    %(business_id)s,
    %(question)s,
    1,
    %(now)s::timestamptz + (%(days)s::int[])[1] * interval '1 day',
    now()
)
ON CONFLICT (business_id, source) DO UPDATE
   SET misses = nc.misses + 1,
       retry_after = CASE
           WHEN nc.misses + 1 <= coalesce(array_length(%(days)s::int[], 1), 0)
           THEN %(now)s::timestamptz + (%(days)s::int[])[nc.misses + 1] * interval '1 day'
           ELSE NULL
       END,
       updated_at = now()
RETURNING misses, retry_after
"""

DELETE_ENTRY = """
DELETE FROM negative_cache
 WHERE business_id = %(business_id)s AND source = %(question)s
"""

# Freshness is read from `enrichments`, not from a column here, because `enrichments` is
# append-only and already holds one row per observation. A `last_ok_at` cached beside the
# miss count would be a second copy of a fact the data plane owns.
SELECT_LAST_OK = """
SELECT max(fetched_at)
  FROM enrichments
 WHERE business_id = %(business_id)s
   AND source = ANY(%(sources)s)
   AND status = 'ok'
"""


class PostgresCache:
    """`EnrichmentCache` over the real `negative_cache` and `enrichments` tables.

    Takes a connection factory rather than a `Repository`, for the same reason
    `discovery.RepositoryBusinesses` does: the repository has no negative-cache methods,
    these are three statements, and adding them to the repository would put queue plumbing
    and enrichment policy in one class. The factory is the seam -- an integration test
    hands it a pool bound to a throwaway schema and nothing here reads an environment
    variable.
    """

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect

    def skip_reason(self, business_id: UUID, question: str, *, now: datetime) -> str | None:
        _validate(question)
        params = {"business_id": business_id, "question": question}
        with self._connect() as connection:
            row = connection.execute(SELECT_ENTRY, params).fetchone()
            if row is not None:
                misses, retry_after = row[0], row[1]
                if misses >= 1:
                    if retry_after is None:
                        return SKIP_EXHAUSTED
                    return SKIP_NEGATIVE if _aware(now) < _aware(retry_after) else None
            fresh = connection.execute(
                SELECT_LAST_OK,
                {
                    "business_id": business_id,
                    "sources": list(SOURCES_BY_QUESTION[question]),
                },
            ).fetchone()
        last_ok = fresh[0] if fresh else None
        if last_ok is not None and _aware(now) - _aware(last_ok) < _ttl(question):
            return SKIP_FRESH
        return None

    def record_miss(self, business_id: UUID, question: str, *, now: datetime) -> CacheEntry:
        _validate(question)
        with self._connect() as connection:
            row = connection.execute(
                RECORD_MISS,
                {
                    "business_id": business_id,
                    "question": question,
                    "now": now,
                    "days": list(BACKOFF_DAYS),
                },
            ).fetchone()
            connection.commit()
        return CacheEntry(
            business_id=business_id,
            question=question,
            misses=row[0],
            retry_after=row[1],
        )

    def record_success(self, business_id: UUID, question: str, *, now: datetime) -> None:
        # A success deletes the ladder. The freshness half of `skip_reason` reads the
        # `enrichments` row the service writes, so there is nothing to store here.
        self.reset(business_id, question)

    def reset(self, business_id: UUID, question: str) -> None:
        _validate(question)
        with self._connect() as connection:
            connection.execute(
                DELETE_ENTRY, {"business_id": business_id, "question": question}
            )
            connection.commit()


def _aware(moment: datetime) -> datetime:
    """Postgres `timestamptz` comes back aware; a caller's clock might not be.

    Comparing the two raises `TypeError`, and the place that would surface is a worker at
    3am rather than a test, so a naive datetime is read as UTC here.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def questions_for(sources: Sequence[str]) -> Iterator[str]:
    """Which questions these `enrichments.source` values answer. For a manual reset."""
    for question, mapped in SOURCES_BY_QUESTION.items():
        if any(source in mapped for source in sources):
            yield question


__all__ = [
    "ADS",
    "BACKOFF_DAYS",
    "INSTAGRAM",
    "QUESTIONS",
    "SKIP_EXHAUSTED",
    "SKIP_FRESH",
    "SKIP_NEGATIVE",
    "SOURCES_BY_QUESTION",
    "TTL_DAYS",
    "WEBSITE",
    "CacheEntry",
    "EnrichmentCache",
    "InMemoryCache",
    "NullCache",
    "PostgresCache",
    "questions_for",
    "retry_after_for",
    "utc_now",
]
