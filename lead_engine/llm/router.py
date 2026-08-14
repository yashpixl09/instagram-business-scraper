"""The chain: Groq -> Gemini -> NVIDIA -> deterministic template.

    groq/llama-3.3-70b     30 RPM · 14.4k RPD · free
      └─429/5xx─> gemini-2.5-flash
          └─429/5xx─> nvidia nim
              └─> deterministic template   cannot fail

THE LAST LINK IS THE ONLY ONE THAT MATTERS
-------------------------------------------
`generate()` takes the fallback text as an ARGUMENT, not as a callable to invoke later, and
validates it before the first request leaves the process. That is not a style choice. A
fallback computed lazily at the bottom of the chain is a fallback that can raise at the
bottom of the chain -- after three providers have already declined -- and the guarantee
this phase is built on is that a run completes with every provider down. Requiring the
answer up front makes "the terminal fallback failed" unrepresentable: by the time anything
can go wrong, the text is already in hand.

`lead_engine/copy.py` produces those strings. They are not a degraded path; they are the
guarantee, and `providers = []` is a supported configuration rather than an error.

WHY THE ACCOUNTING LIVES IN POSTGRES
------------------------------------
Circuit state and token buckets both persist in `llm_rate_buckets`. An in-memory limiter
resets to full on every restart, and the failure that produces is specific and nasty: a
worker gets throttled, crashes or is redeployed, comes back believing it has a full bucket,
and sends straight into the 429 it was already being punished for. That trips the breaker,
which trips the restart, which is how one bad minute becomes an outage.

So a fresh process reads the state a dead one left behind. `MemoryCircuitStore` exists for
the no-database configuration and for tests; it is honest about what it is, and the
restart property is the one thing it cannot give you.

NO RETRY, ANYWHERE
------------------
One attempt per provider per call. A 429 counts as a failure against the breaker AND
drains that provider's bucket by a full window, so the next call skips it on the rate check
rather than sleeping on it. There is no backoff loop, because a backoff loop in front of
three providers and a free template is just a slower way to arrive at the template.

`max_wait_seconds` is the whole anti-storm policy in one number: if the bucket says a call
would have to wait longer than that, the chain advances instead of blocking. The template
is always available, so there is never a reason to make an operator's run sit still.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from lead_engine.providers.errors import ProviderError

from .cache import Cache, NullCache, prompt_hash

if TYPE_CHECKING:  # pragma: no cover
    from .providers import ChatClient

#: The provider name recorded when the chain fell through. Not a real provider, and
#: deliberately not an empty string: a run summary has to be able to say how many pitches
#: were written by a model and how many by the templates.
TEMPLATE = "template"

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

#: Three consecutive failures open a circuit; a probe is admitted 60s later. Both from the
#: spec. One failure is noise -- a single timeout is not evidence a provider is down -- and
#: a threshold of three costs at most two wasted calls before the chain stops trying.
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 60.0

#: The longest this router will ever block a caller on a rate limit. Past this, advance.
MAX_WAIT_SECONDS = 5.0

WINDOW_SECONDS = 60.0

# Attempt outcomes, in the order they appear walking the chain.
SKIPPED_OPEN = "circuit_open"
SKIPPED_RATE = "rate_limited_local"
ERROR = "error"
OK = "ok"


def utcnow() -> datetime:
    """Wall clock, timezone aware. Injectable everywhere below so tests cost no real time."""
    return datetime.now(UTC)


Clock = Callable[[], datetime]


@dataclass(frozen=True)
class BucketSpec:
    """One provider's rate ceiling. Not stored -- `llm_rate_buckets` holds the balance,
    configuration holds the capacity, and a plan change is then a deploy rather than a
    migration."""

    capacity: int
    window_seconds: float = WINDOW_SECONDS

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    @property
    def rate(self) -> float:
        """Tokens regained per second. Refill is continuous, as a token bucket's is: at
        30/60s the bucket regains one token every two seconds rather than handing back all
        thirty on a minute boundary."""
        return self.capacity / self.window_seconds


@dataclass(frozen=True)
class BucketState:
    """A row of `llm_rate_buckets`, read back."""

    provider: str
    tokens: float
    refilled_at: datetime
    circuit_state: str = CLOSED
    consecutive_failures: int = 0
    opened_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.circuit_state == OPEN


class CircuitStore(Protocol):
    """Where the router's state lives. Postgres in production, a dict in tests."""

    def ensure(self, provider: str, spec: BucketSpec) -> None: ...

    def state(self, provider: str) -> BucketState | None: ...

    def admit(self, provider: str, *, cooldown_seconds: float = COOLDOWN_SECONDS) -> bool: ...

    def reserve(
        self, provider: str, spec: BucketSpec, *, max_wait_seconds: float = MAX_WAIT_SECONDS
    ) -> float | None: ...

    def penalise(self, provider: str, spec: BucketSpec) -> None: ...

    def record_success(self, provider: str) -> None: ...

    def record_failure(
        self, provider: str, *, threshold: int = FAILURE_THRESHOLD
    ) -> str: ...


# --- the in-memory store ------------------------------------------------------------------


class MemoryCircuitStore:
    """Circuit state and buckets in a dict, guarded by one lock.

    Correct for a single process and honest about the one thing it cannot do: a restart
    wipes it, which is exactly the failure `llm_rate_buckets` exists to prevent. Use it when
    no database is configured, and in tests that are not about persistence.

    The deduction model is `providers/tinyfish.TokenBucket`'s: deduct under the lock, let
    the balance go negative, and sleep off your own deficit outside the lock. That is what
    makes N concurrent callers queue on the rate rather than on each other.
    """

    def __init__(self, *, clock: Clock = utcnow) -> None:
        self._clock = clock
        self._rows: dict[str, BucketState] = {}
        self._lock = threading.Lock()

    def ensure(self, provider: str, spec: BucketSpec) -> None:
        with self._lock:
            if provider not in self._rows:
                self._rows[provider] = BucketState(
                    provider=provider, tokens=float(spec.capacity), refilled_at=self._clock()
                )

    def state(self, provider: str) -> BucketState | None:
        with self._lock:
            return self._rows.get(provider)

    def admit(self, provider: str, *, cooldown_seconds: float = COOLDOWN_SECONDS) -> bool:
        now = self._clock()
        with self._lock:
            row = self._rows.get(provider)
            if row is None:
                return False
            if row.circuit_state == CLOSED:
                return True
            if row.opened_at is None or now - row.opened_at < timedelta(seconds=cooldown_seconds):
                return False
            # The transition IS the probe, and it re-arms `opened_at`. One caller flips the
            # state and goes; everyone else sees a fresh timestamp and waits another
            # cooldown. If the probe never reports back -- the worker died mid-call -- the
            # next cooldown re-arms rather than leaving the provider half-open forever.
            self._rows[provider] = replace(row, circuit_state=HALF_OPEN, opened_at=now)
            return True

    def reserve(
        self, provider: str, spec: BucketSpec, *, max_wait_seconds: float = MAX_WAIT_SECONDS
    ) -> float | None:
        now = self._clock()
        floor = -max_wait_seconds * spec.rate
        with self._lock:
            row = self._rows.get(provider)
            if row is None:
                return None
            balance = _refilled(row.tokens, row.refilled_at, now, spec) - 1.0
            if balance < floor:
                return None
            self._rows[provider] = replace(row, tokens=balance, refilled_at=now)
        return -balance / spec.rate if balance < 0 else 0.0

    def penalise(self, provider: str, spec: BucketSpec) -> None:
        now = self._clock()
        with self._lock:
            row = self._rows.get(provider)
            if row is not None:
                self._rows[provider] = replace(
                    row, tokens=min(row.tokens, -float(spec.capacity)), refilled_at=now
                )

    def record_success(self, provider: str) -> None:
        with self._lock:
            row = self._rows.get(provider)
            if row is not None:
                self._rows[provider] = replace(
                    row, circuit_state=CLOSED, consecutive_failures=0, opened_at=None
                )

    def record_failure(self, provider: str, *, threshold: int = FAILURE_THRESHOLD) -> str:
        now = self._clock()
        with self._lock:
            row = self._rows.get(provider)
            if row is None:
                return CLOSED
            failures = row.consecutive_failures + 1
            if row.circuit_state == HALF_OPEN or failures >= threshold:
                self._rows[provider] = replace(
                    row, circuit_state=OPEN, consecutive_failures=failures, opened_at=now
                )
                return OPEN
            self._rows[provider] = replace(row, consecutive_failures=failures)
            return row.circuit_state


def _refilled(tokens: float, refilled_at: datetime, now: datetime, spec: BucketSpec) -> float:
    """Continuous refill, capped at capacity, never running backwards.

    A clock that went backwards is not a reason to confiscate tokens, so negative elapsed
    time contributes nothing rather than draining the bucket.
    """
    elapsed = max(0.0, (now - refilled_at).total_seconds())
    return min(float(spec.capacity), tokens + elapsed * spec.rate)


# --- the durable store --------------------------------------------------------------------


class Cursor(Protocol):
    def fetchone(self) -> Sequence[Any] | None: ...


class Connection(Protocol):
    def execute(
        self, query: str, params: Mapping[str, Any] | Sequence[Any] | None = ..., /
    ) -> Cursor: ...

    def commit(self) -> None: ...


ConnectionFactory = Callable[[], AbstractContextManager[Connection]]

# The refill expression, written once and interpolated twice, because it has to appear
# identically in the SET and in the WHERE. Typing it out twice is how the guard and the
# assignment drift apart.
_BALANCE = """least(
        %(capacity)s::numeric,
        tokens + greatest(
            0,
            extract(epoch from (%(now)s::timestamptz - refilled_at))
        ) * %(rate)s::numeric
    )"""

_ENSURE = """
INSERT INTO llm_rate_buckets (provider, tokens, refilled_at)
VALUES (%(provider)s, %(tokens)s, %(now)s)
ON CONFLICT (provider) DO NOTHING
"""

_READ = """
SELECT provider, tokens, refilled_at, circuit_state, consecutive_failures, opened_at
  FROM llm_rate_buckets
 WHERE provider = %(provider)s
"""

# One statement, so the check and the deduction cannot be separated. Two workers that both
# read a balance of 1 would both conclude they had a token; here the loser re-evaluates its
# WHERE against the winner's committed row -- the same property `providers/budget._SPEND`
# depends on, and for the same reason.
_RESERVE = f"""
UPDATE llm_rate_buckets
   SET tokens = {_BALANCE} - 1,
       refilled_at = %(now)s
 WHERE provider = %(provider)s
   AND {_BALANCE} - 1 >= %(floor)s::numeric
RETURNING tokens
"""

# A 429 costs a full window of silence on that provider. Not a sleep -- a skip: the next
# call finds the bucket below the floor and advances the chain.
_PENALISE = """
UPDATE llm_rate_buckets
   SET tokens = least(tokens, -%(capacity)s::numeric),
       refilled_at = %(now)s
 WHERE provider = %(provider)s
"""

# Half-open is entered by exactly one caller, because exactly one UPDATE can match a row in
# state 'open' with an elapsed cooldown. Re-stamping `opened_at` is what makes it
# self-healing: a probe whose worker died leaves the provider re-armed, not stuck.
_ADMIT_PROBE = """
UPDATE llm_rate_buckets
   SET circuit_state = '{half_open}',
       opened_at = %(now)s
 WHERE provider = %(provider)s
   AND circuit_state IN ('{open}', '{half_open}')
   AND opened_at IS NOT NULL
   AND %(now)s::timestamptz - opened_at >= make_interval(secs => %(cooldown)s)
RETURNING circuit_state
""".format(open=OPEN, half_open=HALF_OPEN)

_RECORD_SUCCESS = """
UPDATE llm_rate_buckets
   SET circuit_state = '{closed}',
       consecutive_failures = 0,
       opened_at = NULL
 WHERE provider = %(provider)s
""".format(closed=CLOSED)

# The CASE arms read the row's OLD values, which is what makes this one statement rather
# than a read, a decision and a write. A failed half-open probe re-opens immediately: the
# probe WAS the evidence, and counting to three again would send two more calls into a
# provider that has just told us it is still down.
_RECORD_FAILURE = """
UPDATE llm_rate_buckets
   SET consecutive_failures = consecutive_failures + 1,
       circuit_state = CASE
           WHEN circuit_state = '{half_open}' THEN '{open}'
           WHEN consecutive_failures + 1 >= %(threshold)s THEN '{open}'
           ELSE circuit_state
       END,
       opened_at = CASE
           WHEN circuit_state = '{half_open}' THEN %(now)s
           WHEN consecutive_failures + 1 >= %(threshold)s THEN %(now)s
           ELSE opened_at
       END
 WHERE provider = %(provider)s
RETURNING circuit_state
""".format(open=OPEN, half_open=HALF_OPEN)


class DatabaseCircuitStore:
    """`llm_rate_buckets`: the half of the router that survives a restart.

    Stateless apart from the connection factory, and it commits its own writes -- a circuit
    that opened but was rolled back with whatever else the caller was doing is a circuit
    that never opened.

    Imports no database driver, same as `providers/budget.py`: it talks to whatever
    `execute`/`commit` object the factory yields.
    """

    def __init__(self, connect: ConnectionFactory, *, clock: Clock = utcnow) -> None:
        self._connect = connect
        self._clock = clock

    def ensure(self, provider: str, spec: BucketSpec) -> None:
        """Create the row if it is missing. Never resets one that exists -- that would be
        the restart-clears-the-limiter bug wearing a helpful face."""
        with self._connect() as conn:
            conn.execute(
                _ENSURE,
                {"provider": provider, "tokens": float(spec.capacity), "now": self._clock()},
            )
            conn.commit()

    def state(self, provider: str) -> BucketState | None:
        with self._connect() as conn:
            row = conn.execute(_READ, {"provider": provider}).fetchone()
            conn.commit()
        if row is None:
            return None
        return BucketState(
            provider=str(row[0]),
            tokens=float(row[1]),
            refilled_at=row[2],
            circuit_state=str(row[3]),
            consecutive_failures=int(row[4]),
            opened_at=row[5],
        )

    def admit(self, provider: str, *, cooldown_seconds: float = COOLDOWN_SECONDS) -> bool:
        now = self._clock()
        with self._connect() as conn:
            probe = conn.execute(
                _ADMIT_PROBE,
                {"provider": provider, "now": now, "cooldown": float(cooldown_seconds)},
            ).fetchone()
            if probe is not None:
                conn.commit()
                return True
            row = conn.execute(_READ, {"provider": provider}).fetchone()
            conn.commit()
        return row is not None and str(row[3]) == CLOSED

    def reserve(
        self, provider: str, spec: BucketSpec, *, max_wait_seconds: float = MAX_WAIT_SECONDS
    ) -> float | None:
        now = self._clock()
        params = {
            "provider": provider,
            "now": now,
            "capacity": float(spec.capacity),
            "rate": spec.rate,
            "floor": -max_wait_seconds * spec.rate,
        }
        with self._connect() as conn:
            row = conn.execute(_RESERVE, params).fetchone()
            conn.commit()
        if row is None:
            return None
        balance = float(row[0])
        return -balance / spec.rate if balance < 0 else 0.0

    def penalise(self, provider: str, spec: BucketSpec) -> None:
        with self._connect() as conn:
            conn.execute(
                _PENALISE,
                {"provider": provider, "capacity": float(spec.capacity), "now": self._clock()},
            )
            conn.commit()

    def record_success(self, provider: str) -> None:
        with self._connect() as conn:
            conn.execute(_RECORD_SUCCESS, {"provider": provider})
            conn.commit()

    def record_failure(self, provider: str, *, threshold: int = FAILURE_THRESHOLD) -> str:
        with self._connect() as conn:
            row = conn.execute(
                _RECORD_FAILURE,
                {"provider": provider, "threshold": int(threshold), "now": self._clock()},
            ).fetchone()
            conn.commit()
        return CLOSED if row is None else str(row[0])


# --- the router ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """One link of the chain, and what it did. Kept so a run summary can explain a
    template-written pitch without anyone reading a log."""

    provider: str
    outcome: str
    code: str | None = None


@dataclass(frozen=True)
class Generation:
    """What came back, and where from."""

    text: str
    provider: str
    model: str = ""
    cached: bool = False
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)

    @property
    def from_template(self) -> bool:
        return self.provider == TEMPLATE


class LLMRouter:
    """Walks the chain and always returns text.

        router = LLMRouter(build_clients(groq_key=...), store=store, cache=cache)
        result = router.generate(prompt, fallback=build_fallback_outreach(lead, score))

    `clients` may be empty. That is the configuration the headline test of this phase
    exercises, and it is not a degraded mode: it returns the fallback, marked
    `provider='template'`, having issued no request and touched no bucket.
    """

    def __init__(
        self,
        clients: Iterable[ChatClient] = (),
        *,
        store: CircuitStore | None = None,
        cache: Cache | None = None,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        failure_threshold: int = FAILURE_THRESHOLD,
        max_wait_seconds: float = MAX_WAIT_SECONDS,
        window_seconds: float = WINDOW_SECONDS,
        clock: Clock = utcnow,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clients: tuple[ChatClient, ...] = tuple(clients)
        self.store: CircuitStore = store if store is not None else MemoryCircuitStore(clock=clock)
        self.cache: Cache = cache if cache is not None else NullCache()
        self.cooldown_seconds = float(cooldown_seconds)
        self.failure_threshold = int(failure_threshold)
        self.max_wait_seconds = float(max_wait_seconds)
        self._sleep = sleep
        self._monotonic = monotonic
        self.specs: dict[str, BucketSpec] = {
            client.name: BucketSpec(client.requests_per_minute, window_seconds)
            for client in self.clients
        }
        # Seed the rows once, at construction. `ensure` never resets an existing row, so a
        # fresh process after a crash inherits the balance and the open circuit the dead one
        # left rather than starting full.
        for client in self.clients:
            self.store.ensure(client.name, self.specs[client.name])

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        store: CircuitStore | None = None,
        cache: Cache | None = None,
        transport: Any = None,
        only: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> LLMRouter:
        """Build the chain from `config.Settings`, skipping every provider without a key."""
        clients = clients_from_settings(settings, transport=transport, only=only)
        return cls(clients, store=store, cache=cache, **kwargs)

    # -- the one public method ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        fallback: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generation:
        """Return generated prose, or `fallback`. Never raises for a provider failure.

        `fallback` is validated first, before the cache is consulted and long before a
        socket is opened. By the time any provider can fail, the answer to "what if they all
        do" is already a string in this frame. A blank fallback is a programming error --
        the caller forgot to build the deterministic copy -- and it is raised here, at the
        top, where it costs nothing, rather than discovered at the bottom of a dead chain.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        terminal = _checked_fallback(fallback)

        key = prompt_hash(prompt)
        cached = self.cache.get(key)
        if cached is not None:
            # No request, no token, no bucket. A re-run after a crash re-spends nothing.
            return Generation(
                text=cached.text, provider=cached.provider, model=cached.model, cached=True
            )

        options: dict[str, Any] = {"system": system}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature

        attempts: list[Attempt] = []
        for client in self.clients:
            attempt, generation = self._try(client, prompt, key, options)
            attempts.append(attempt)
            if generation is not None:
                return replace(generation, attempts=tuple(attempts))

        return Generation(text=terminal, provider=TEMPLATE, attempts=tuple(attempts))

    # -- one link -------------------------------------------------------------------

    def _try(
        self,
        client: ChatClient,
        prompt: str,
        key: str,
        options: Mapping[str, Any],
    ) -> tuple[Attempt, Generation | None]:
        name = client.name
        spec = self.specs[name]

        if not self.store.admit(name, cooldown_seconds=self.cooldown_seconds):
            return Attempt(name, SKIPPED_OPEN), None

        wait = self.store.reserve(name, spec, max_wait_seconds=self.max_wait_seconds)
        if wait is None:
            # The bucket says this call would have to wait longer than the policy allows.
            # Advancing costs nothing; blocking costs the operator their afternoon.
            return Attempt(name, SKIPPED_RATE), None
        if wait > 0:
            self._sleep(wait)

        started = self._monotonic()
        try:
            completion = client.complete(prompt, **options)
        except ProviderError as exc:
            self._failed(name, client.model, key, exc.code, started)
            if exc.code == "provider_rate_limited":
                # Do not come back to this provider for a full window. The circuit may still
                # be closed -- one 429 is not three failures -- and without this the next
                # lead in the run walks into the same 429, which is the storm.
                self.store.penalise(name, spec)
            return Attempt(name, ERROR, exc.code), None
        except Exception:
            # A client that raises anything but a ProviderError has broken its own contract.
            # That is a bug to fix, but it must not be able to take down a run whose whole
            # promise is that it completes with every provider unavailable. Classified as
            # the taxonomy's "upstream answered with something we could not use", recorded,
            # and the chain advances. Note the deliberate `except Exception`, not
            # `BaseException`: KeyboardInterrupt and SystemExit still stop the process.
            self._failed(name, client.model, key, "provider_bad_response", started)
            return Attempt(name, ERROR, "provider_bad_response"), None

        latency = self._elapsed_ms(started)
        self.store.record_success(name)
        self.cache.record_success(
            key,
            provider=name,
            model=completion.model,
            text=completion.text,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=latency,
        )
        return (
            Attempt(name, OK),
            Generation(text=completion.text, provider=name, model=completion.model),
        )

    def _failed(self, name: str, model: str, key: str, code: str, started: float) -> None:
        self.store.record_failure(name, threshold=self.failure_threshold)
        self.cache.record_failure(
            key, provider=name, model=model, code=code, latency_ms=self._elapsed_ms(started)
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._monotonic() - started) * 1000))

    # -- lifecycle ------------------------------------------------------------------

    def close(self) -> None:
        for client in self.clients:
            client.close()

    def __enter__(self) -> LLMRouter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _checked_fallback(fallback: str) -> str:
    if not isinstance(fallback, str) or not fallback.strip():
        raise ValueError(
            "fallback must be non-empty text. The terminal link of the chain is a guarantee, "
            "not an error path: build it with lead_engine.copy before calling generate()."
        )
    return fallback
