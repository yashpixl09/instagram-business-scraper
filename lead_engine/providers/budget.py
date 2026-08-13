"""The search budget: a lifetime ledger for billed provider calls, held in Postgres.

WHY THIS IS NOT A COUNTER
-------------------------
SearchAPI's free searches are granted ONCE per key and never renew. The allowance in hand is
50; a rotated key brings its own, and that is the only way more arrive. Geoapify's category
filters were free, so the previous generation of this code could sweep a city exhaustively
and pay nothing. Here every query is billed, and the arithmetic is brutal: 13 tiles x 5
query variants x 3 pages is 195 searches -- roughly four times the allowance, to cover one
niche in one neighbourhood. There is no monthly reset to forgive that.

So the ceiling cannot be a field on a worker object. A process-local counter is reset by a
restart, by a crash loop, by a second worker, by a developer running the pipeline by hand to
check something -- and each reset spends real, unrecoverable credits. The count lives in
Postgres and only ever goes up.

    budget = SearchBudget(pool.connection)          # psycopg_pool.ConnectionPool
    key = fingerprint(os.environ["SEARCHAPI_KEY"])  # the digest, never the key

    # All 50 to discovery, which is the default split -- see DEFAULT_SPLIT.
    budget.ensure_allowance("searchapi", 50, key_fingerprint=key)

    while tiles:
        try:
            budget.spend("searchapi", key_fingerprint=key)
        except BudgetExhausted:
            break                                   # a clean stop, not a failure
        ...

THREE THINGS KEY A BUDGET
-------------------------
`provider`, `purpose`, and `key_fingerprint` together identify one allowance.

`purpose` is 'discover' or 'refresh'. Splitting them is what stops refresh -- which grows
with every business already in the database -- from quietly eating the credits discovery
needs. They are two allowances, not one allowance and a good intention. Until discovery has
swept the target areas, `refresh` is worth 0: re-checking known ground only earns its
credits once there is no new ground left. That is a default to configure, not a rule this
module enforces.

`key_fingerprint` is sha256(key)[:12] -- see `fingerprint()`. It is part of the key because
the operator rotates keys, and rotation has to be automatic: an unseen fingerprint starts a
fresh allowance on the next `ensure()`. A manual reset command is a step someone forgets,
and forgetting means the new key inherits a spent budget and the worker refuses to run. The
retired key's row stays behind, so what each allowance actually bought survives the
rotation. Pass '' for providers where rotation is not modelled.

Because a provider now spans several rows, `remaining()` answers for one purpose and
`remaining_total()` sums them for one key. Nothing sums across fingerprints on purpose:
credits on a retired key are not credits you have.

THE ONE-STATEMENT RULE
----------------------
`spend()` checks the limit and increments the counter in a single UPDATE:

    UPDATE search_budget SET used = used + n
     WHERE provider = %s AND purpose = %s AND key_fingerprint = %s
       AND used + n <= limit_total

Not read-then-write. Under READ COMMITTED, two workers that both read `used = 49` would both
conclude they had room and both write 50, and a credit is spent twice. The single statement
takes a row lock; the loser re-evaluates its WHERE against the winner's committed row,
matches nothing, and raises BudgetExhausted.

What the CHECK constraint does and does not cover is worth stating precisely, because both
halves of it have been used to justify a mistake. Two ways to break `spend()`, measured
against real Postgres with 24 threads:

  * Move the arithmetic into Python -- SELECT, decide, UPDATE with the computed total. The
    CHECK is blind to this and stays silent: 1148 credits taken from a 50-credit allowance
    while the stored `used` column sat at a legal 50. A lost update never writes an
    *illegal* value. Each racer reads 40, computes 41, writes 41; they satisfy the
    constraint and overwrite each other. A constraint validates what gets written, never
    what gets overwritten.

  * Keep the arithmetic in SQL but drop the guard -- `SET used = used + n` with no
    `used + n <= limit_total`. Here the CHECK does fire. `used + n` re-reads the winner's
    committed row under EvalPlanQual, so the loser computes 51 against a limit of 50 and
    gets a CheckViolation: an ugly failure, but a failure, not 1148 billed requests.

So the constraint is NOT SUFFICIENT ON ITS OWN, which is not the same as inert: it cannot
substitute for the predicate, and it is also the last line against a SQL-side increment that
forgot the guard. Both mutations are pinned in tests/test_budget.py, because either half of
that sentence quoted alone justifies a mistake -- the first half reads as licence to delete
a constraint that does catch something, the second as licence to trust one that cannot.

WHAT THE SCHEMA CANNOT HOLD
---------------------------
Every constraint on this table is per-row, and since 0011 the real ceiling is a property
*across* rows: a key's lifetime allowance is the sum of `limit_total` over its purpose rows.
Nothing in the database stops discover=50 and refresh=50 from being seeded against a key
worth 50 in total. Every row-level CHECK is satisfied; the allowance is simply doubled, and
the first symptom is SearchAPI returning 401 in the middle of a run.

That is the same shape of gap as the lost update above -- a check cannot see past its own
scope -- and it is answered the same way: not by a better constraint, but by making the bad
state unrepresentable at the one place that writes a ceiling. `ensure_allowance()` is that
place. It takes a total and a split and derives every `limit_total` from them, so there is
no argument to it that expresses an over-allocation. `ensure()`, which writes a single
purpose, refuses when a split already exists for that key rather than becoming the back door
around it.


SPEND FIRST, NEVER REFUND
-------------------------
Record the spend *before* issuing the billed call, and never give a credit back when the
call fails. A timeout does not prove the request was not billed -- the response is what got
lost, not necessarily the query. Over-counting by a handful of failed calls costs a handful
of searches. Under-counting silently overspends an allowance that cannot be topped up. That
is why there is no `refund()` here, and why adding one would be a mistake.

THE LEDGER COMMITS ITS OWN WRITES
---------------------------------
`SearchBudget` takes a *connection factory*, not a connection, and commits inside it. A
spend that is still uncommitted when the worker crashes is a search that was paid for and
not recorded -- exactly the failure the table exists to prevent. Owning the transaction also
means the ledger can never be rolled back by whatever else the caller was doing.

This module imports no database driver. It talks to whatever `execute`/`commit` object the
factory hands it, which keeps the SQL honest (there is nowhere to hide an ORM) and lets the
non-concurrency tests run against a stub. The atomicity guarantee above is a property of
Postgres, not of this file, so the test that asserts it uses real threads and a real
database or it does not run at all.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

#: The two legal purposes. Kept in sync with the CHECK constraint in 0011 -- validating here
#: too is not redundancy for its own sake: it turns a typo into a ValueError naming the legal
#: values, instead of a CheckViolation naming a constraint.
DISCOVER = "discover"
REFRESH = "refresh"
PURPOSES = frozenset({DISCOVER, REFRESH})

#: Everything to discovery, nothing to refresh, until told otherwise. At fifty lifetime
#: credits, re-checking ground already swept is only worth paying for once there is no new
#: ground left. The spec's 70/30 is the shape for budgets that renew -- TinyFish, Firecrawl
#: -- not for an allowance that is gone when it is gone. Read-only so a caller that mutates
#: what it was handed cannot change the default for every worker in the process.
DEFAULT_SPLIT: Mapping[str, float] = MappingProxyType({DISCOVER: 1.0, REFRESH: 0.0})

#: `0.7 + 0.3` is 0.9999999999999999 in binary floating point, so the split from the spec
#: fails an exact comparison against 1.0. The tolerance exists for that, not for sloppiness:
#: it is far tighter than any split anyone would write by hand.
_SPLIT_TOLERANCE = 1e-9

#: What `fingerprint()` produces, and the only thing the ledger will store. '' is also legal
#: and means "rotation is not modelled for this provider".
FINGERPRINT_LENGTH = 12
_FINGERPRINT = re.compile(rf"^[0-9a-f]{{{FINGERPRINT_LENGTH}}}$")


def fingerprint(api_key: str) -> str:
    """Digest an API key into the 12 hex characters the ledger stores.

    The point is that the ledger can tell two keys apart without ever holding one. sha256 is
    one-way, so this row survives in logs, backups and screenshots without carrying the
    credential; truncation to 12 hex characters is about readability in a run summary, and
    is nowhere near a collision risk for the handful of keys one operator rotates through.

    Every caller must use THIS function rather than hashing by hand. A digest truncated to a
    different length, or upper-cased, is a different primary key -- which the ledger reads
    as a brand new key and hands a brand new allowance. The CHECK constraint in 0011 rejects
    the upper-cased case outright; this function is how you avoid meeting it.

    Raises ValueError on an empty key. An unset key is a configuration bug, and quietly
    fingerprinting "" would give every misconfigured deployment the same ledger row.
    """
    if not api_key:
        raise ValueError("cannot fingerprint an empty API key -- the key is not configured")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


class Cursor(Protocol):
    """The sliver of DBAPI this module uses."""

    def fetchone(self) -> Sequence[Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...


class Connection(Protocol):
    """The sliver of a psycopg connection this module uses."""

    def execute(self, query: str, params: Sequence[Any] | None = ..., /) -> Cursor: ...

    def commit(self) -> None: ...


#: What the constructor wants: something that yields a connection per `with` block.
#: `psycopg_pool.ConnectionPool.connection` is exactly this shape. For a single existing
#: connection, `lambda: contextlib.nullcontext(conn)` works -- but read the note about
#: transactions above before reaching for it.
ConnectionFactory = Callable[[], AbstractContextManager[Connection]]


def _key_name(provider: str, key_fingerprint: str) -> str:
    """How one key's allowance is named in an error or a log line."""
    return f"{provider} [key {key_fingerprint}]" if key_fingerprint else provider


def _describe(provider: str, purpose: str, key_fingerprint: str) -> str:
    """How one purpose's allowance is named in an error or a log line."""
    name = f"{provider}/{purpose}"
    return f"{name} [key {key_fingerprint}]" if key_fingerprint else name


class BudgetExhausted(Exception):
    """The allocation is spent. THIS IS NOT AN ERROR CONDITION.

    The discovery worker treats it as a clean stop: finish the leads already in hand, write
    the sheet, exit zero. It is the budget working, not the budget breaking.

    Concretely, it must not be logged at ERROR, must not be reported as a failed run, must
    not be retried, and must never be converted into a 5xx. It is the loop condition of the
    discovery sweep, expressed as an exception because the check has to happen inside the
    call that would otherwise spend the credit.

    It is deliberately NOT a `ProviderError`: nothing upstream went wrong, and there is no
    HTTP status that means "we decided to stop".
    """

    def __init__(
        self,
        provider: str,
        limit_total: int,
        used: int,
        requested: int,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> None:
        self.provider = provider
        self.purpose = purpose
        self.key_fingerprint = key_fingerprint
        self.limit_total = limit_total
        self.used = used
        self.requested = requested
        self.remaining = limit_total - used
        # Naming the purpose and the key is not decoration. During a rotation there are
        # several rows in play, and "searchapi is exhausted" sends the operator to look at
        # the wrong one.
        super().__init__(
            f"{_describe(provider, purpose, key_fingerprint)} search budget exhausted: "
            f"{used}/{limit_total} used, {self.remaining} remaining, {requested} requested"
        )


class AllowanceAlreadySplit(ValueError):
    """`ensure()` was asked to add a purpose beside an allowance that is already split.

    Raised rather than accepted because the alternative is silent: the new row passes every
    per-row CHECK, and the only symptom is that the sum of the ceilings for that key now
    exceeds the allowance they were divided from. Use `ensure_allowance()`, which cannot
    express an over-allocation because it derives every ceiling from one total.
    """


class BudgetNotConfigured(LookupError):
    """No ledger row exists for this provider, purpose and key.

    Separate from BudgetExhausted on purpose. A worker whose `except BudgetExhausted` also
    swallowed a missing row would exit zero having done nothing, and would look like a
    successful run that simply found no leads -- the most expensive kind of silent failure,
    because nobody investigates a green build. A missing row is a configuration bug and
    should be loud. Call `ensure()` at startup.

    Note that a rotated key legitimately has no row until `ensure()` runs, which is why
    `ensure()` belongs at worker startup rather than in a provisioning script someone runs
    once.
    """


@dataclass(frozen=True)
class BudgetState:
    """A point-in-time read of one allowance."""

    provider: str
    limit_total: int
    used: int
    purpose: str = DISCOVER
    key_fingerprint: str = ""

    @property
    def remaining(self) -> int:
        return self.limit_total - self.used

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


# `used + %s <= limit_total` is the whole safety property. Keep it in the WHERE clause.
_SPEND = """
UPDATE search_budget
   SET used = used + %s,
       updated_at = now()
 WHERE provider = %s
   AND purpose = %s
   AND key_fingerprint = %s
   AND used + %s <= limit_total
RETURNING used, limit_total
"""

# Every purpose row in ONE statement. Separate INSERTs would let a crash land discovery's
# share without refresh's, and would let two callers with different splits interleave into a
# pair that sums to neither.
_ALLOCATE = """
INSERT INTO search_budget (provider, purpose, key_fingerprint, limit_total)
VALUES {values}
ON CONFLICT (provider, purpose, key_fingerprint) DO NOTHING
"""

# What the ledger holds for one key, whatever this call did or did not create.
_READ_ALLOWANCE = """
SELECT purpose, limit_total
  FROM search_budget
 WHERE provider = %s AND key_fingerprint = %s
 ORDER BY purpose
"""

# `ensure()` writes one purpose, so it must not be the way a second purpose gets added
# behind `ensure_allowance()`'s back -- that is the seeding path that doubles an allowance.
# The NOT EXISTS makes the refusal part of the same statement as the insert rather than a
# check the caller could race past. See the method docstring for what that does not cover.
_ENSURE_ONLY_PURPOSE = """
INSERT INTO search_budget (provider, purpose, key_fingerprint, limit_total)
SELECT %s, %s, %s, %s
 WHERE NOT EXISTS (
   SELECT 1 FROM search_budget
    WHERE provider = %s AND key_fingerprint = %s AND purpose <> %s
 )
ON CONFLICT (provider, purpose, key_fingerprint) DO NOTHING
RETURNING provider
"""

_KEYED = "WHERE provider = %s AND purpose = %s AND key_fingerprint = %s"

_READ = f"SELECT used, limit_total FROM search_budget {_KEYED}"

_SET_LIMIT = f"""
UPDATE search_budget
   SET limit_total = %s,
       updated_at = now()
 {_KEYED}
RETURNING used, limit_total
"""

# count(*) alongside the sum, because sum() over no rows is NULL and "nothing configured"
# must not be reported as "nothing left" -- see BudgetNotConfigured.
_READ_TOTAL = """
SELECT coalesce(sum(limit_total - used), 0), count(*)
  FROM search_budget
 WHERE provider = %s AND key_fingerprint = %s
"""


class SearchBudget:
    """A lifetime ledger of billed searches, one row per (provider, purpose, key).

    Stateless: it holds a connection factory and nothing else, so a single instance is safe
    to share across threads and workers. All the state is in Postgres, which is the point.

    `purpose` and `key_fingerprint` are keyword-only on every method. They are both plain
    strings with defaults, sitting next to each other, and a positional call site that
    swapped them would write to a legal-looking row that no one else ever reads.
    """

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect

    # --- setup ---------------------------------------------------------------------------

    def ensure_allowance(
        self,
        provider: str,
        total: int,
        *,
        key_fingerprint: str = "",
        split: Mapping[str, float] = DEFAULT_SPLIT,
    ) -> dict[str, int]:
        """Divide ONE key's lifetime allowance across purposes. The only writer of a ceiling.

        Call this at worker startup. It returns what the ledger now holds for this key, as
        {purpose: limit_total} -- which is what was already there if the rows existed, not
        necessarily what this call computed.

        WHY IT TAKES A TOTAL AND A SPLIT, NOT PER-PURPOSE LIMITS
        Since 0011 the ceiling is a sum across rows, and every CHECK on the table is
        per-row. Seeding discover=50 and refresh=50 against a key worth 50 satisfies every
        constraint in the database while handing the system twice the searches it has, and
        nothing finds out until SearchAPI starts returning 401s mid-run. Deriving both
        numbers from one `total` makes that unrepresentable rather than merely checked:
        there is no argument to this method that expresses an over-allocation.

        The split must sum to 1.0 and may only name known purposes. Rounding uses largest
        remainder, so the parts always add back to exactly `total` -- neither inventing a
        credit nor stranding one, and a stranded credit on a non-renewable allowance is a
        real loss, not a rounding detail.

        Every row is written in one statement, so a crash cannot fund discovery and lose
        refresh. Idempotent exactly as `ensure()` is: ON CONFLICT DO NOTHING per row, so a
        restart cannot reset `used` and cannot re-cut a split that is already in force.

        A fingerprint the ledger has not seen gets fresh rows here -- that is the rotation
        mechanism, and the retired key's rows stay behind with their history.

        What it still cannot enforce: a hand-written INSERT, or `set_limit()` raising one
        row afterwards. A cross-row invariant in a per-row schema cannot be made airtight
        from application code -- only expensive to violate by accident.
        """
        _text(provider, "provider")
        total = _non_negative(total, "total")
        key = _checked_fingerprint(key_fingerprint)
        allocation = _divide(total, split)

        values = ", ".join(["(%s, %s, %s, %s)"] * len(allocation))
        params: list[Any] = []
        for purpose, limit_total in allocation.items():
            params += [provider, purpose, key, limit_total]

        with self._connect() as conn:
            conn.execute(_ALLOCATE.format(values=values), params)
            rows = conn.execute(_READ_ALLOWANCE, (provider, key)).fetchall()
            allowance = {str(row[0]): int(row[1]) for row in rows}

            # Verify the readback rather than trusting the arithmetic above. `_divide` cannot
            # over-allocate, but `_ALLOCATE` is ON CONFLICT DO NOTHING, so a row that already
            # existed keeps its own ceiling and is not replaced by this split's share.
            #
            # That is not hypothetical: migration 0011 widens the key by giving the old
            # single-row ledger `purpose='discover'` and `key_fingerprint=''`, preserving its
            # limit. Splitting 100 as 70/30 afterwards leaves discover at its inherited 100
            # and inserts refresh at 30 -- 130 billable searches on a 100-search key, every
            # per-row CHECK satisfied, no error anywhere, and the overspend only visible when
            # SearchAPI starts refusing at a number the operator cannot explain.
            granted = sum(allowance.values())
            if granted > total:
                conn.rollback()
                raise AllowanceAlreadySplit(
                    f"{provider}/{key or 'no key'} already holds ceilings summing to "
                    f"{granted}, which exceeds the {total} being allocated: "
                    f"{allowance}. An existing row was not overwritten. Reconcile it with "
                    f"set_limit() before splitting, or allocate under a fresh key."
                )
            conn.commit()
        return allowance

    def ensure(
        self,
        provider: str,
        limit_total: int,
        *,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> bool:
        """Create a single-purpose ledger row if it is missing. True if it created one.

        For providers whose allowance is not split -- one purpose, one row. Where a split
        exists, use `ensure_allowance()`: it is the only method that can see a total.

        Idempotent and safe to call on every boot. It will NOT touch `used`, and will NOT
        change `limit_total` on a row that already exists -- raising the ceiling is a
        deliberate act with a bill attached, so it lives in `set_limit()` where it is
        visible in a diff.

        It REFUSES when another purpose already exists for this provider and key, raising
        AllowanceAlreadySplit. That refusal is the point: without it, `ensure()` is a back
        door that adds a second purpose row beside a split `ensure_allowance()` computed,
        and the sum of the ceilings quietly exceeds the allowance the split was cut from.

        The refusal shares a statement with the insert rather than preceding it, so it is
        not a check-then-act a caller can race past. It is not proof against two `ensure()`
        calls for different purposes committing in the same instant -- under READ COMMITTED
        the NOT EXISTS takes no lock -- but that is a boot-time call, and the useful
        property is that the ordinary path cannot do it by accident.
        """
        key = _row_key(provider, purpose, key_fingerprint)
        limit_total = _non_negative(limit_total, "limit_total")
        provider, purpose, digest = key
        with self._connect() as conn:
            row = conn.execute(
                _ENSURE_ONLY_PURPOSE,
                (provider, purpose, digest, limit_total, provider, digest, purpose),
            ).fetchone()
            if row is not None:
                conn.commit()
                return True
            # Nothing inserted: either this exact row already exists (idempotent no-op) or
            # another purpose is in the way (a refusal). Read back to say which.
            existing = {str(r[0]) for r in conn.execute(_READ_ALLOWANCE, (provider, digest))}
            conn.commit()

        if existing - {purpose}:
            raise AllowanceAlreadySplit(
                f"{_key_name(provider, digest)} already has an allowance across "
                f"{sorted(existing)}; adding {purpose!r} through ensure() would raise the "
                "total ceiling above whatever it was split from. Use ensure_allowance()."
            )
        return False

    def set_limit(
        self,
        provider: str,
        limit_total: int,
        *,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> BudgetState:
        """Move the ceiling for an allowance that already has a row.

        Lowering it below `used` is rejected by the CHECK constraint in 0010 rather than
        silently accepted, because the alternative is a ledger that reports negative
        remaining and a `spend()` that can never succeed again without anyone noticing why.
        """
        key = _row_key(provider, purpose, key_fingerprint)
        limit_total = _non_negative(limit_total, "limit_total")
        with self._connect() as conn:
            row = conn.execute(_SET_LIMIT, (limit_total, *key)).fetchone()
            conn.commit()
        if row is None:
            raise BudgetNotConfigured(f"no search budget configured for {_describe(*key)}")
        return BudgetState(provider, int(row[1]), int(row[0]), purpose, key_fingerprint)

    # --- the hot path --------------------------------------------------------------------

    def spend(
        self,
        provider: str,
        n: int = 1,
        *,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> int:
        """Charge `n` searches to one allowance and return what is left afterwards.

        Atomic: the limit check and the increment are one statement, so concurrent workers
        cannot both squeeze past the same remaining credit.

        Raises BudgetExhausted -- a clean stop, see the class docstring -- if the charge
        would take `used` past `limit_total`. It is all-or-nothing: a `spend(3)` with 2 left
        charges nothing and raises, rather than charging 2 and reporting success, because a
        partial charge means the caller makes 3 calls having paid for 2.
        """
        key = _row_key(provider, purpose, key_fingerprint)
        n = _positive(n, "n")
        with self._connect() as conn:
            row = conn.execute(_SPEND, (n, *key, n)).fetchone()
            if row is not None:
                conn.commit()
                used, limit_total = int(row[0]), int(row[1])
                return limit_total - used

            # Nothing matched: either the row is missing, or there was not enough left.
            # Re-read to say which, and commit either way so the connection goes back to
            # the pool clean.
            current = conn.execute(_READ, key).fetchone()
            conn.commit()

        if current is None:
            raise BudgetNotConfigured(f"no search budget configured for {_describe(*key)}")
        raise BudgetExhausted(
            provider,
            limit_total=int(current[1]),
            used=int(current[0]),
            requested=n,
            purpose=purpose,
            key_fingerprint=key_fingerprint,
        )

    def remaining(
        self,
        provider: str,
        *,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> int:
        """Searches still available for ONE purpose. Never negative -- 0010 forbids it.

        Advisory only. Between this read and a `spend()` another worker may take the last
        credit, which is exactly why `spend()` re-checks rather than trusting a caller who
        looked first. Use it for reporting and for deciding whether to start a sweep, not
        as a guard.
        """
        return self.snapshot(
            provider, purpose=purpose, key_fingerprint=key_fingerprint
        ).remaining

    def remaining_total(self, provider: str, *, key_fingerprint: str = "") -> int:
        """Searches still available across every purpose, for ONE key.

        This is the question a free-text key could not answer: "how much of this key's
        allowance is left, whatever it was earmarked for". Answering it is why the ledger
        has three columns instead of one string.

        It deliberately does not sum across fingerprints. Credits stranded on a retired key
        are not credits anyone can spend, and adding them to a total would overstate what
        the run can afford -- which, with an allowance this small, is the whole budget.
        """
        _text(provider, "provider")
        key_fingerprint = _checked_fingerprint(key_fingerprint)
        with self._connect() as conn:
            row = conn.execute(_READ_TOTAL, (provider, key_fingerprint)).fetchone()
            conn.commit()
        if row is None or int(row[1]) == 0:
            raise BudgetNotConfigured(
                f"no search budget configured for {_key_name(provider, key_fingerprint)}"
            )
        return int(row[0])

    def snapshot(
        self,
        provider: str,
        *,
        purpose: str = DISCOVER,
        key_fingerprint: str = "",
    ) -> BudgetState:
        """The whole ledger row, for logs and run summaries."""
        key = _row_key(provider, purpose, key_fingerprint)
        with self._connect() as conn:
            row = conn.execute(_READ, key).fetchone()
            conn.commit()
        if row is None:
            raise BudgetNotConfigured(f"no search budget configured for {_describe(*key)}")
        return BudgetState(provider, int(row[1]), int(row[0]), purpose, key_fingerprint)


def _divide(total: int, split: Mapping[str, float]) -> dict[str, int]:
    """Cut `total` into whole searches per purpose. The parts always sum back to `total`.

    Largest remainder, not naive rounding, and not floor-and-forget. Floor alone strands a
    credit -- on an allowance that never renews, a stranded credit is money spent and never
    used, which is the same size of mistake as spending one twice. Rounding each part up
    invents credits the key does not have, which is worse.

    Binary floating point makes this less academic than it sounds: 0.7 * 50 is
    34.99999999999999, so flooring the spec's own 70/30 split would silently produce 34/15
    and lose a search. The remainder pass puts it back.

    Ties break alphabetically so the same split always cuts the same way, whatever order the
    caller's dict happened to be in.
    """
    if not split:
        raise ValueError("split must name at least one purpose")
    unknown = set(split) - PURPOSES
    if unknown:
        raise ValueError(
            f"unknown purpose(s) in split: {sorted(unknown)}; legal purposes are "
            f"{sorted(PURPOSES)}"
        )
    for purpose, fraction in split.items():
        if fraction < 0:
            raise ValueError(f"split fraction for {purpose!r} must be >= 0, got {fraction!r}")
    if abs(sum(split.values()) - 1.0) > _SPLIT_TOLERANCE:
        raise ValueError(
            f"split must sum to 1.0, got {sum(split.values())} from {dict(split)}. A split "
            "that sums to less strands credits; one that sums to more allocates searches "
            "the key does not have."
        )

    exact = {purpose: total * fraction for purpose, fraction in split.items()}
    parts = {purpose: int(value) for purpose, value in exact.items()}
    shortfall = total - sum(parts.values())
    by_remainder = sorted(parts, key=lambda p: (-(exact[p] - parts[p]), p))
    for purpose in by_remainder[:shortfall]:
        parts[purpose] += 1
    return {purpose: parts[purpose] for purpose in sorted(parts)}


def _row_key(provider: str, purpose: str, key_fingerprint: str) -> tuple[str, str, str]:
    """Validate the three columns that identify a row, in the order the SQL wants them."""
    return (
        _text(provider, "provider"),
        _checked_purpose(purpose),
        _checked_fingerprint(key_fingerprint),
    )


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _checked_purpose(purpose: str) -> str:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(PURPOSES)}, got {purpose!r}")
    return purpose


def _checked_fingerprint(key_fingerprint: str) -> str:
    """Reject anything that is not a digest -- without ever echoing the value.

    The value is rejected precisely when it might BE the API key, so putting it in the
    exception message would write the credential into the traceback, the log aggregator and
    the ticket someone pastes it into. The shape and the length are enough to debug with.

    Checking here rather than leaving it to the CHECK constraint also keeps a mistyped key
    out of the driver's parameter list, and so out of the database's statement log.
    """
    if not isinstance(key_fingerprint, str):
        raise ValueError(f"key_fingerprint must be a string, got {type(key_fingerprint).__name__}")
    if key_fingerprint and not _FINGERPRINT.match(key_fingerprint):
        raise ValueError(
            f"key_fingerprint must be {FINGERPRINT_LENGTH} lowercase hex characters "
            f"(or '' where rotation is not modelled); got a {len(key_fingerprint)}-character "
            "value that does not match. Use fingerprint(api_key) -- and note that if you "
            "passed the API key itself, it was NOT logged here."
        )
    return key_fingerprint


def _non_negative(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _positive(value: int, name: str) -> int:
    number = int(value)
    if number < 1:
        # A spend of 0 is always a bug -- either a miscomputed page count or a loop that
        # should not have run. Charging nothing and returning success hides it.
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return number
