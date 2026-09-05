"""The LLM router: three interchangeable providers in front of a guarantee.

The spec asks for one thing from this layer -- "fake providers returning 429 and 5xx;
asserts circuit opens, chain advances, deterministic fallback terminates" -- and the reason
it asks is not tidiness. Everything above this module writes prose that a business owner
reads. A router that raises takes the sheet with it; a router that returns an empty string
puts a blank pitch in front of an operator who then sends it.

So the assertions here are weighted the way the spec weights them: the negative cases carry
the file.

  * `TheTerminalFallback` is the headline. `providers = []` is a supported configuration,
    not a degraded one, and `generate()` validates the fallback before the cache is even
    consulted so that "the last link failed" is unrepresentable.
  * `ChainAdvances` and `TheSpecScenario` walk the chain with 429s and 5xx, once through
    fake clients and once through the real client classes over `httpx.MockTransport`, so the
    thing that advances is the code that will advance in production.
  * `CircuitBreakerOpens` pins the state machine at the boundaries: three failures, not two;
    a probe at sixty seconds, not fifty-nine point nine; a failed probe re-opens immediately
    rather than counting to three again.
  * `TheTokenBucket` pins the anti-storm policy, including the one behaviour that surprises:
    a burst of 429s does NOT trip the breaker, because `penalise` drains the bucket and the
    following calls are skipped on the rate check before the breaker ever sees them.
  * `MalformedProviderResponses` is the biggest class in the file on purpose. Every shape a
    provider can answer with that is not prose has to become an error and an advance, never
    a return value.

Time is injected everywhere -- `clock`, `sleep` and `monotonic` are all constructor
arguments -- so nothing in this file waits, and the sixty-second cooldown is asserted at
59.9 and at 60.0 rather than approximated.

No test here opens a socket. The fake clients never had one, and every real client is
constructed on an `httpx.MockTransport`.

Two sentinels, and the difference between them is the point, exactly as in
`tests/test_searchapi.py`:

  `SENTINEL_KEY` is credential-shaped, so `errors.redact` would catch it even if these
  clients were careless.
  `SENTINEL_BODY` is ordinary prose that `redact` leaves alone -- deliberately, so that the
  no-leak guarantee has to come from these modules not interpolating response bodies.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import httpx
import pytest

from lead_engine.llm import (
    CLOSED,
    COOLDOWN_SECONDS,
    FAILURE_THRESHOLD,
    HALF_OPEN,
    OPEN,
    STATUS_ERROR,
    STATUS_OK,
    TEMPLATE,
    Attempt,
    BucketSpec,
    BucketState,
    CachedResponse,
    DatabaseCircuitStore,
    Generation,
    LLMRouter,
    MemoryCircuitStore,
    NullCache,
    ResponseCache,
    prompt_hash,
)
from lead_engine.llm import cache as cache_module
from lead_engine.llm import router as router_module
from lead_engine.llm.providers import (
    CHAIN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GEMINI,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ,
    GROQ_MODEL,
    GROQ_URL,
    NVIDIA,
    NVIDIA_MODEL,
    NVIDIA_URL,
    ChatClient,
    Completion,
    GeminiClient,
    GroqClient,
    NvidiaClient,
    build_clients,
    clients_from_settings,
    secret_value,
    status_error,
)
from lead_engine.llm.router import (
    ERROR,
    OK,
    SKIPPED_OPEN,
    SKIPPED_RATE,
    _refilled,
)
from lead_engine.providers.errors import (
    DEFAULT_MESSAGE,
    RETRYABLE_BY_CODE,
    STATUS_BY_CODE,
    ProviderError,
    auth_failed,
    bad_response,
    rate_limited,
    redact,
    unavailable,
)

ROOT = Path(__file__).resolve().parent.parent

#: Credential-shaped, so its absence in an error is guaranteed twice over.
SENTINEL_KEY = "gsk_live_9f2c7b31e4d8a6c5f0b3"
#: Planted in every error response body. Deliberately NOT credential-shaped.
SENTINEL_BODY = "the peacock ate our quota"

#: What `lead_engine/copy.py` hands `generate()`. Not a degraded path -- the guarantee.
FALLBACK = "Kumar Studio has no website and 638 reviews. Offer: a one-page booking site."

PROMPT = "Write a qualification summary for Kumar Studio."

EPOCH = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)


# --- time, injected -------------------------------------------------------------------------


class Clock:
    """A wall clock that only moves when a test moves it.

    The cooldown and the refill are both measured in seconds of this clock, so every
    boundary in the file is asserted exactly rather than slept through.
    """

    def __init__(self, start: datetime = EPOCH) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class Monotonic:
    """A latency source that ticks a fixed amount per read."""

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def never_sleeps(seconds: float) -> None:  # pragma: no cover - the failure it guards
    raise AssertionError(f"the router blocked for {seconds}s; it was meant to advance")


# --- the fakes ------------------------------------------------------------------------------


class FakeClient:
    """A provider that answers from a script, and records what it was asked.

    One script entry per `complete()` call. An entry that is an exception is raised, which
    is how a 429 or a 5xx is expressed without a socket; anything else is returned. When the
    script runs out the client succeeds, so a test that only cares about the first two links
    does not have to spell out the third.

    Deliberately not a `Mock`: the router's contract with a provider is three attributes and
    two methods, and a fake that cannot satisfy `ChatClient` would prove nothing about the
    clients that will actually be passed in. `FakesAreFaithful` pins that.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str | None = None,
        requests_per_minute: int = 30,
        script: object = (),
    ) -> None:
        self.name = name
        self.model = model if model is not None else f"{name}-model"
        self.requests_per_minute = requests_per_minute
        self.script = list(script)
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def complete(self, prompt: str, **options: object) -> Completion:
        self.calls.append((prompt, dict(options)))
        answer = self.script.pop(0) if self.script else self.answer()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def answer(self, text: str | None = None) -> Completion:
        return Completion(
            provider=self.name,
            model=self.model,
            text=text if text is not None else f"{self.name} wrote this pitch.",
            input_tokens=120,
            output_tokens=64,
        )

    def close(self) -> None:
        self.closed = True


class DictCache:
    """The cache in a dict, with every ledger write kept in order.

    It mirrors the two properties of `llm_calls` that the router depends on and that a
    counter could not express: a success is written once and readable by prompt, and a
    failure is written every time and never becomes a cache entry. The partial unique index
    in `0007` is what makes that true in Postgres; here it is written out by hand so that a
    router which started caching its failures fails a test rather than a deployment.
    """

    def __init__(self, seeded: dict[str, CachedResponse] | None = None) -> None:
        self.rows: dict[str, CachedResponse] = dict(seeded or {})
        self.lookups: list[str] = []
        self.successes: list[dict] = []
        self.failures: list[dict] = []

    def get(self, key: str) -> CachedResponse | None:
        self.lookups.append(key)
        return self.rows.get(key)

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
        self.successes.append(
            {
                "key": key,
                "provider": provider,
                "model": model,
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            }
        )
        # ON CONFLICT DO NOTHING: the first committed answer is the one an operator reads.
        self.rows.setdefault(key, CachedResponse(provider=provider, model=model, text=text))

    def record_failure(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        code: str,
        latency_ms: int | None = None,
    ) -> None:
        self.failures.append(
            {
                "key": key,
                "provider": provider,
                "model": model,
                "code": code,
                "latency_ms": latency_ms,
            }
        )
        # And nothing is written to `rows`. Caching a 429 as the answer to a prompt would
        # make one throttled minute permanent.


class ExplodingCache:
    """A cache that must never be reached. Any call is the test's failure."""

    def get(self, key: str) -> CachedResponse | None:  # pragma: no cover - the guard
        raise AssertionError("the cache was consulted before the arguments were validated")

    def record_success(self, key: str, **_: object) -> None:  # pragma: no cover - the guard
        raise AssertionError("a success was recorded when nothing was generated")

    def record_failure(self, key: str, **_: object) -> None:  # pragma: no cover - the guard
        raise AssertionError("a failure was recorded when nothing was attempted")


class StubCursor:
    def __init__(self, rows) -> None:
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class StubConnection:
    """Records every statement and hands back canned result sets, one per execute().

    The same shape `tests/test_budget.py` uses, and for the same reason: the properties that
    matter about `llm_rate_buckets` and `llm_calls` are properties of the SQL -- the check
    is inside the UPDATE, the CASE arms read the old row -- and those are assertable without
    a database.
    """

    def __init__(self, results=None) -> None:
        self.statements: list[tuple[str, object]] = []
        self.commits = 0
        self._results = list(results or [])

    def execute(self, query, params=None):
        self.statements.append((query, params))
        return StubCursor(self._results.pop(0) if self._results else [])

    def commit(self) -> None:
        self.commits += 1

    def factory(self):
        connection = self

        @contextmanager
        def connect():
            yield connection

        return connect

    def collapsed(self, index: int = 0) -> str:
        return " ".join(self.statements[index][0].split()).lower()


# --- shared plumbing ------------------------------------------------------------------------


class RouterTestCase(unittest.TestCase):
    """Every router in this file gets a frozen clock and a sleep that fails the test."""

    def build(self, *clients, **kwargs) -> LLMRouter:
        kwargs.setdefault("clock", self.clock)
        kwargs.setdefault("sleep", never_sleeps)
        kwargs.setdefault("monotonic", Monotonic())
        return LLMRouter(clients, **kwargs)

    def setUp(self) -> None:
        self.clock = Clock()

    def assertAttempts(self, generation: Generation, expected: list[Attempt]) -> None:
        self.assertEqual(list(generation.attempts), expected)

    def mock_client(self, cls, handler, **kwargs):
        """A real provider client on a mock transport, closed when the test ends."""
        client = cls(SENTINEL_KEY, transport=httpx.MockTransport(handler), **kwargs)
        self.addCleanup(client.close)
        return client


def responder(payload, status_code: int = 200):
    return lambda request: httpx.Response(status_code, json=payload)


def groq_body(text: str = "Groq wrote this.", **usage) -> dict:
    body: dict = {"choices": [{"message": {"content": text}}]}
    if usage:
        body["usage"] = dict(usage)
    return body


def gemini_body(parts: list[dict], **usage) -> dict:
    body: dict = {"candidates": [{"content": {"parts": parts}}]}
    if usage:
        body["usageMetadata"] = dict(usage)
    return body


# --- the fakes have to be worth trusting ----------------------------------------------------


class FakesAreFaithful(unittest.TestCase):
    """A fake that accepts less than the real thing hides the defect it was built to catch.

    `tests/test_searchapi.py` records how that goes wrong in practice: a narrow `FakeBudget`
    accepted the narrow call, every test passed, and the client was charging a ledger row
    nobody seeded. So the fakes here are checked against the interfaces they stand in for
    before anything else in the file relies on them.
    """

    def test_the_fake_client_satisfies_the_protocol_the_router_declares(self):
        self.assertIsInstance(FakeClient("groq"), ChatClient)
        # And so does a real one, which is what makes the comparison mean anything.
        real = GroqClient(None)
        self.addCleanup(real.close)
        self.assertIsInstance(real, ChatClient)

    def test_the_fake_client_accepts_every_keyword_a_real_client_does(self):
        real = inspect.signature(GroqClient.complete)
        fake = FakeClient("groq")
        for name, parameter in real.parameters.items():
            if name in ("self", "prompt"):
                continue
            with self.subTest(parameter=name):
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                # `**options` has to actually take it. A TypeError here is a fake that would
                # turn a router change into a green suite and a red run.
                fake.complete(PROMPT, **{name: parameter.default})

    def test_the_dict_cache_accepts_every_argument_the_real_cache_does(self):
        for method in ("get", "record_success", "record_failure"):
            real = inspect.signature(getattr(ResponseCache, method))
            fake = inspect.signature(getattr(DictCache, method))
            for name, parameter in real.parameters.items():
                if name == "self":
                    continue
                with self.subTest(method=method, parameter=name):
                    self.assertIn(name, fake.parameters)
                    self.assertEqual(parameter.kind, fake.parameters[name].kind)

    def test_the_body_sentinel_is_one_redact_would_not_have_caught(self):
        # Guarding the guard, as in test_searchapi.py. If SENTINEL_BODY were
        # credential-shaped, every leak assertion below would be testing `errors.redact`
        # rather than these clients.
        self.assertEqual(redact(SENTINEL_BODY), SENTINEL_BODY)
        self.assertNotEqual(redact(SENTINEL_KEY), SENTINEL_KEY)


# --- the guarantee --------------------------------------------------------------------------


class TheTerminalFallback(RouterTestCase):
    """The run completes with every provider down. Everything else is an optimisation.

    `generate()` takes the fallback text as an argument rather than as a callable, and
    validates it before the first request leaves the process. That is what makes "the
    terminal fallback failed" unrepresentable, and it is only true if the validation really
    is first -- so the tests below assert on ordering, not just on the exception.
    """

    def test_no_providers_at_all_is_a_supported_configuration(self):
        # The headline case of this phase. Not a degraded mode: no request, no bucket, no
        # attempt, and a full artifact at the end of it.
        result = self.build().generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, FALLBACK)
        self.assertEqual(result.provider, TEMPLATE)
        self.assertTrue(result.from_template)
        self.assertFalse(result.cached)
        self.assertEqual(result.attempts, ())
        self.assertEqual(result.model, "")

    def test_the_template_provider_is_named_rather_than_blank(self):
        # A run summary has to be able to say how many pitches a model wrote and how many
        # the templates did. An empty string would make those two indistinguishable.
        self.assertEqual(TEMPLATE, "template")
        self.assertTrue(Generation(text="x", provider=TEMPLATE).from_template)
        self.assertFalse(Generation(text="x", provider=GROQ).from_template)

    def test_a_blank_fallback_is_rejected_before_anything_is_consulted(self):
        # The caller forgot to build the deterministic copy. Raised at the top, where it
        # costs nothing, rather than discovered at the bottom of a dead chain.
        client = FakeClient(GROQ)
        router = self.build(client, cache=ExplodingCache())

        for fallback in ("", "   ", "\n\t ", None, 5, b"bytes"):
            with self.subTest(fallback=fallback):
                with self.assertRaises(ValueError):
                    router.generate(PROMPT, fallback)

        # No cache lookup (ExplodingCache would have raised), and no request.
        self.assertEqual(client.calls, [])

    def test_the_refusal_names_where_the_copy_is_supposed_to_come_from(self):
        # The message is the fix. "fallback must be non-empty text" alone sends someone
        # looking for a default; naming `lead_engine.copy` sends them to the module that
        # already builds the string they are missing.
        with self.assertRaises(ValueError) as caught:
            self.build().generate(PROMPT, "")

        message = str(caught.exception)
        self.assertIn("lead_engine.copy", message)
        self.assertIn("guarantee", message)

    def test_an_unusable_prompt_is_rejected_before_the_fallback_is_read(self):
        router = self.build(cache=ExplodingCache())

        for prompt in ("", "   ", None, 5, b"bytes"):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError) as caught:
                    # Both arguments are bad. The prompt is the one reported, which pins
                    # the order the two checks run in.
                    router.generate(prompt, "")
                self.assertIn("prompt", str(caught.exception))

    def test_every_provider_down_still_returns_the_deterministic_text(self):
        clients = [
            FakeClient(GROQ, script=[rate_limited("throttled", provider=GROQ)]),
            FakeClient(GEMINI, script=[unavailable("down", provider=GEMINI)]),
            FakeClient(NVIDIA, script=[bad_response("nonsense", provider=NVIDIA)]),
        ]

        result = self.build(*clients).generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, FALLBACK)
        self.assertEqual(result.provider, TEMPLATE)
        self.assertAttempts(
            result,
            [
                Attempt(GROQ, ERROR, "provider_rate_limited"),
                Attempt(GEMINI, ERROR, "provider_unavailable"),
                Attempt(NVIDIA, ERROR, "provider_bad_response"),
            ],
        )

    def test_the_chain_terminates_after_exactly_one_pass(self):
        # "Deterministic fallback terminates", spelled out as a bound rather than as an
        # absence of hanging: three providers, three calls, no retry, no second lap.
        clients = [
            FakeClient(name, script=[unavailable("down", provider=name)]) for name in CHAIN
        ]

        result = self.build(*clients).generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, TEMPLATE)
        self.assertEqual(len(result.attempts), len(clients))
        for client in clients:
            with self.subTest(provider=client.name):
                self.assertEqual(len(client.calls), 1)

    def test_a_client_that_breaks_its_contract_cannot_take_down_the_run(self):
        # A provider raising anything but a ProviderError is a bug to fix. It must not be
        # able to end a run whose whole promise is that it completes with every provider
        # unavailable, so it is classified, recorded and stepped over.
        class Broken(FakeClient):
            def complete(self, prompt, **options):
                raise RuntimeError(f"undeclared failure: {SENTINEL_BODY}")

        cache = DictCache()
        result = self.build(Broken(GROQ), FakeClient(GEMINI), cache=cache).generate(
            PROMPT, FALLBACK
        )

        self.assertEqual(result.provider, GEMINI)
        self.assertAttempts(
            result,
            [Attempt(GROQ, ERROR, "provider_bad_response"), Attempt(GEMINI, OK)],
        )
        # Recorded as this project's own code, not as the vendor's words.
        self.assertEqual(cache.failures[0]["code"], "provider_bad_response")
        self.assertNotIn(SENTINEL_BODY, str(cache.failures))

    def test_a_keyboard_interrupt_still_stops_the_process(self):
        # `except Exception`, deliberately not `except BaseException`. An operator pressing
        # ctrl-c must not be answered with a template and a green run.
        class Interrupted(FakeClient):
            def complete(self, prompt, **options):
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.build(Interrupted(GROQ)).generate(PROMPT, FALLBACK)

    def test_a_system_exit_is_not_swallowed_either(self):
        class Exiting(FakeClient):
            def complete(self, prompt, **options):
                raise SystemExit(1)

        with self.assertRaises(SystemExit):
            self.build(Exiting(GROQ)).generate(PROMPT, FALLBACK)


# --- walking the chain ----------------------------------------------------------------------


class ChainAdvances(RouterTestCase):
    """One attempt per provider per call, in order, stopping at the first answer."""

    def test_a_429_on_the_first_link_advances_to_the_second(self):
        groq = FakeClient(GROQ, script=[rate_limited("throttled", provider=GROQ)])
        gemini = FakeClient(GEMINI)
        nvidia = FakeClient(NVIDIA)

        result = self.build(groq, gemini, nvidia).generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, GEMINI)
        self.assertEqual(result.model, "gemini-model")
        self.assertEqual(result.text, "gemini wrote this pitch.")
        self.assertFalse(result.from_template)
        self.assertAttempts(
            result, [Attempt(GROQ, ERROR, "provider_rate_limited"), Attempt(GEMINI, OK)]
        )
        # The third link is never touched: the chain stops at the first answer.
        self.assertEqual(nvidia.calls, [])

    def test_a_5xx_on_the_first_two_links_lands_on_the_third(self):
        groq = FakeClient(GROQ, script=[unavailable("502", provider=GROQ)])
        gemini = FakeClient(GEMINI, script=[unavailable("503", provider=GEMINI)])
        nvidia = FakeClient(NVIDIA)

        result = self.build(groq, gemini, nvidia).generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, NVIDIA)
        self.assertAttempts(
            result,
            [
                Attempt(GROQ, ERROR, "provider_unavailable"),
                Attempt(GEMINI, ERROR, "provider_unavailable"),
                Attempt(NVIDIA, OK),
            ],
        )

    def test_every_failure_shape_advances_the_chain(self):
        # Including auth failure, which is NOT retryable in the taxonomy. Retryability is
        # about coming back to the same provider; the chain advancing is a different
        # question, and a wrong key must not strand a run on the template while two working
        # providers sit unasked.
        failures = [
            rate_limited("throttled", provider=GROQ),
            unavailable("down", provider=GROQ),
            bad_response("nonsense", provider=GROQ),
            auth_failed("bad key", provider=GROQ),
            RuntimeError("undeclared"),
            ValueError("undeclared"),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__ + getattr(failure, "code", "")):
                gemini = FakeClient(GEMINI)
                result = self.build(
                    FakeClient(GROQ, script=[failure]), gemini
                ).generate(PROMPT, FALLBACK)

                self.assertEqual(result.provider, GEMINI)
                self.assertEqual(len(gemini.calls), 1)

    def test_no_link_is_ever_attempted_twice_in_one_call(self):
        # There is no retry anywhere in this system. A backoff loop in front of three
        # providers and a free template is a slower way to arrive at the template.
        groq = FakeClient(GROQ, script=[unavailable("down", provider=GROQ)] * 5)

        result = self.build(groq).generate(PROMPT, FALLBACK)

        self.assertEqual(len(groq.calls), 1)
        self.assertEqual(len(result.attempts), 1)

    def test_the_attempt_log_explains_a_template_written_pitch(self):
        # The reason `Attempt` exists: a run summary must be able to say why a pitch came
        # from a template without anyone reading a log.
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ, requests_per_minute=1)
        gemini = FakeClient(GEMINI, script=[unavailable("down", provider=GEMINI)])
        router = self.build(groq, gemini, store=store, max_wait_seconds=0.0)

        router.generate("first prompt", FALLBACK)  # spends groq's single token
        result = router.generate("second prompt", FALLBACK)

        self.assertEqual(result.provider, TEMPLATE)
        self.assertAttempts(
            result,
            [Attempt(GROQ, SKIPPED_RATE), Attempt(GEMINI, ERROR, "provider_unavailable")],
        )
        # The four outcomes are distinct strings, so a summary can group by them.
        self.assertEqual(len({SKIPPED_OPEN, SKIPPED_RATE, ERROR, OK}), 4)

    def test_the_generation_options_reach_the_provider(self):
        groq = FakeClient(GROQ)
        router = self.build(groq)

        router.generate(PROMPT, FALLBACK)
        router.generate(PROMPT + "!", FALLBACK, system="Be terse.", max_tokens=42,
                        temperature=0.9)

        # Unset options are omitted entirely rather than sent as None, so each client keeps
        # its own documented default for max_tokens and temperature.
        self.assertEqual(groq.calls[0], (PROMPT, {"system": None}))
        self.assertEqual(
            groq.calls[1],
            (PROMPT + "!", {"system": "Be terse.", "max_tokens": 42, "temperature": 0.9}),
        )

    def test_closing_the_router_closes_every_client(self):
        clients = [FakeClient(name) for name in CHAIN]

        with self.build(*clients) as router:
            self.assertIsInstance(router, LLMRouter)
            router.generate(PROMPT, FALLBACK)

        for client in clients:
            with self.subTest(provider=client.name):
                self.assertTrue(client.closed)


# --- the circuit breaker --------------------------------------------------------------------


class CircuitBreakerOpens(RouterTestCase):
    """closed -> open after three consecutive failures, half-open probe after sixty seconds.

    The thresholds are asserted at their boundaries. Two failures must not open the circuit
    and a probe at 59.9 seconds must not be admitted, because an off-by-one in either
    direction is invisible in production: too eager and the chain stops using a provider
    that works, too slow and it keeps hammering one that does not.

    Every failure here is a 5xx rather than a 429 on purpose. A 429 also drains the token
    bucket, which changes what the following calls do -- see `TheTokenBucket`.
    """

    def failing(self, name: str = GROQ, times: int = 10) -> FakeClient:
        return FakeClient(name, script=[unavailable("down", provider=name)] * times)

    def test_three_consecutive_failures_open_the_circuit(self):
        store = MemoryCircuitStore(clock=self.clock)
        router = self.build(self.failing(), store=store)

        for expected_failures, expected_state in ((1, CLOSED), (2, CLOSED), (3, OPEN)):
            with self.subTest(failures=expected_failures):
                router.generate(f"prompt {expected_failures}", FALLBACK)
                state = store.state(GROQ)
                self.assertEqual(state.consecutive_failures, expected_failures)
                self.assertEqual(state.circuit_state, expected_state)

        self.assertEqual(FAILURE_THRESHOLD, 3)
        self.assertTrue(store.state(GROQ).is_open)
        self.assertEqual(store.state(GROQ).opened_at, self.clock.now)

    def test_an_open_circuit_is_skipped_without_a_request(self):
        # The point of the breaker. Two wasted calls at most, then nothing.
        client = self.failing()
        router = self.build(client, store=MemoryCircuitStore(clock=self.clock))

        for _ in range(3):
            router.generate(PROMPT + str(_), FALLBACK)
        calls_when_opened = len(client.calls)

        result = router.generate("after", FALLBACK)

        self.assertEqual(calls_when_opened, 3)
        self.assertEqual(len(client.calls), 3, "a request was issued into an open circuit")
        self.assertAttempts(result, [Attempt(GROQ, SKIPPED_OPEN)])
        self.assertEqual(result.provider, TEMPLATE)

    def test_two_failures_and_a_success_leave_the_circuit_closed(self):
        # "Consecutive" is the whole word. A provider that fails twice, works, then fails
        # twice is a provider with a flaky minute, not one that is down.
        store = MemoryCircuitStore(clock=self.clock)
        client = FakeClient(GROQ)
        down = unavailable("down", provider=GROQ)
        client.script = [down, down, client.answer(), down, down]
        router = self.build(client, store=store)

        for index in range(5):
            router.generate(f"prompt {index}", FALLBACK)

        state = store.state(GROQ)
        self.assertEqual(state.circuit_state, CLOSED)
        self.assertEqual(state.consecutive_failures, 2)

    def test_the_probe_is_admitted_at_sixty_seconds_and_not_before(self):
        store = MemoryCircuitStore(clock=self.clock)
        client = self.failing()
        router = self.build(client, store=store)
        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)
        calls_at_open = len(client.calls)

        self.clock.advance(59.9)
        early = router.generate("early", FALLBACK)

        self.assertAttempts(early, [Attempt(GROQ, SKIPPED_OPEN)])
        self.assertEqual(len(client.calls), calls_at_open)
        self.assertEqual(store.state(GROQ).circuit_state, OPEN)

        self.clock.advance(0.1)
        probe = router.generate("probe", FALLBACK)

        self.assertEqual(COOLDOWN_SECONDS, 60.0)
        self.assertEqual(probe.attempts[0].outcome, ERROR)
        self.assertEqual(len(client.calls), calls_at_open + 1)

    def test_a_failed_probe_re_opens_immediately_rather_than_counting_to_three(self):
        # The probe WAS the evidence. Counting to three again would send two more calls
        # into a provider that has just said it is still down.
        store = MemoryCircuitStore(clock=self.clock)
        client = self.failing()
        router = self.build(client, store=store)
        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)

        self.clock.advance(COOLDOWN_SECONDS)
        router.generate("probe", FALLBACK)

        state = store.state(GROQ)
        self.assertEqual(state.circuit_state, OPEN)
        self.assertEqual(state.consecutive_failures, 4)
        self.assertEqual(state.opened_at, self.clock.now)

        # And the cooldown restarts from the failed probe, not from the original opening.
        self.clock.advance(59.9)
        self.assertAttempts(router.generate("still open", FALLBACK), [Attempt(GROQ, SKIPPED_OPEN)])

    def test_a_successful_probe_closes_the_circuit_and_clears_the_count(self):
        store = MemoryCircuitStore(clock=self.clock)
        client = FakeClient(GROQ, script=[unavailable("down", provider=GROQ)] * 3)
        router = self.build(client, store=store)
        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)

        self.clock.advance(COOLDOWN_SECONDS)
        result = router.generate("probe", FALLBACK)

        self.assertEqual(result.provider, GROQ)
        state = store.state(GROQ)
        self.assertEqual(state.circuit_state, CLOSED)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertIsNone(state.opened_at)

    def test_entering_half_open_re_arms_the_cooldown_for_everyone_else(self):
        # Exactly one caller flips the state and goes. If the probe never reports back --
        # the worker died mid-call -- the next cooldown re-arms rather than leaving the
        # provider half-open forever.
        store = MemoryCircuitStore(clock=self.clock)
        store.ensure(GROQ, BucketSpec(30))
        for _ in range(FAILURE_THRESHOLD):
            store.record_failure(GROQ)
        self.assertTrue(store.state(GROQ).is_open)

        self.clock.advance(COOLDOWN_SECONDS)
        self.assertTrue(store.admit(GROQ))
        self.assertEqual(store.state(GROQ).circuit_state, HALF_OPEN)
        self.assertEqual(store.state(GROQ).opened_at, self.clock.now)

        # A second caller in the same instant is refused, and stays refused right up to the
        # next cooldown boundary.
        self.assertFalse(store.admit(GROQ))
        self.clock.advance(59.9)
        self.assertFalse(store.admit(GROQ))
        self.clock.advance(0.1)
        self.assertTrue(store.admit(GROQ))

    def test_an_open_first_link_does_not_stop_the_chain(self):
        store = MemoryCircuitStore(clock=self.clock)
        groq = self.failing(GROQ)
        gemini = FakeClient(GEMINI)
        router = self.build(groq, gemini, store=store)
        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)

        result = router.generate("after", FALLBACK)

        self.assertEqual(result.provider, GEMINI)
        self.assertAttempts(result, [Attempt(GROQ, SKIPPED_OPEN), Attempt(GEMINI, OK)])

    def test_the_breaker_is_per_provider(self):
        store = MemoryCircuitStore(clock=self.clock)
        router = self.build(self.failing(GROQ), FakeClient(GEMINI), store=store)

        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)

        self.assertEqual(store.state(GROQ).circuit_state, OPEN)
        self.assertEqual(store.state(GEMINI).circuit_state, CLOSED)
        self.assertEqual(store.state(GEMINI).consecutive_failures, 0)

    def test_a_provider_nobody_seeded_is_never_admitted(self):
        # `admit` is the first thing `_try` calls. A missing row means the router was handed
        # a store it never ensured against, and guessing "closed" there would send traffic
        # at a provider with no bucket to charge it to.
        store = MemoryCircuitStore(clock=self.clock)

        self.assertFalse(store.admit("never-seeded"))
        self.assertIsNone(store.state("never-seeded"))
        self.assertIsNone(store.reserve("never-seeded", BucketSpec(30)))
        self.assertEqual(store.record_failure("never-seeded"), CLOSED)
        store.penalise("never-seeded", BucketSpec(30))  # a no-op, not a KeyError
        store.record_success("never-seeded")


# --- the token bucket -----------------------------------------------------------------------


class TheTokenBucket(RouterTestCase):
    """`max_wait_seconds` is the whole anti-storm policy in one number.

    Past it, the chain advances rather than blocking, because the template is always
    available and there is never a reason to make an operator's run sit still.
    """

    def test_a_burst_of_429s_is_absorbed_by_the_bucket_before_the_breaker_sees_it(self):
        # Worth pinning precisely because it reads as a bug at first glance and is not.
        # A 429 penalises the bucket by a full window, so the second and third calls never
        # reach the provider and never count against the breaker. The circuit is still
        # CLOSED after three consecutive `generate()` calls -- the provider has simply been
        # taken out of the chain by the rate check, which is the cheaper way to say no.
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ, script=[rate_limited("throttled", provider=GROQ)] * 3)
        router = self.build(groq, store=store)

        results = [router.generate(f"prompt {index}", FALLBACK) for index in range(3)]

        self.assertEqual(len(groq.calls), 1, "the bucket did not absorb the storm")
        self.assertEqual(store.state(GROQ).circuit_state, CLOSED)
        self.assertEqual(store.state(GROQ).consecutive_failures, 1)
        self.assertEqual(results[0].attempts[0], Attempt(GROQ, ERROR, "provider_rate_limited"))
        self.assertEqual(results[1].attempts[0], Attempt(GROQ, SKIPPED_RATE))
        self.assertEqual(results[2].attempts[0], Attempt(GROQ, SKIPPED_RATE))
        for result in results:
            self.assertEqual(result.text, FALLBACK)

    def test_repeated_429s_do_open_the_circuit_once_the_bucket_lets_them_through(self):
        # The other half of the sentence above: the breaker still opens, it just counts
        # attempts rather than calls. Three throttled attempts, each a full window apart.
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ, script=[rate_limited("throttled", provider=GROQ)] * 3)
        slept: list[float] = []
        router = self.build(groq, store=store, sleep=slept.append)

        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)
            self.clock.advance(60.0)

        self.assertEqual(len(groq.calls), 3)
        self.assertEqual(store.state(GROQ).circuit_state, OPEN)
        self.assertEqual(store.state(GROQ).consecutive_failures, 3)

    def test_a_429_costs_a_full_window_of_silence_and_a_5xx_does_not(self):
        # The penalty is what stops the next lead in the run walking into the same 429.
        # A 5xx is a different failure -- upstream is down, not annoyed -- and draining the
        # bucket for it would punish a provider that is about to come back.
        spec = BucketSpec(30, 60.0)
        for failure, expected in (
            (rate_limited("throttled", provider=GROQ), True),
            (unavailable("down", provider=GROQ), False),
            (bad_response("nonsense", provider=GROQ), False),
        ):
            with self.subTest(code=failure.code):
                clock = Clock()
                store = MemoryCircuitStore(clock=clock)
                router = LLMRouter(
                    [FakeClient(GROQ, script=[failure])],
                    store=store,
                    clock=clock,
                    sleep=never_sleeps,
                )
                router.generate(PROMPT, FALLBACK)

                drained = store.state(GROQ).tokens <= -float(spec.capacity)
                self.assertEqual(drained, expected)

    def test_the_penalised_provider_is_skipped_until_the_window_has_passed(self):
        spec = BucketSpec(30, 60.0)
        store = MemoryCircuitStore(clock=self.clock)
        store.ensure(GROQ, spec)
        store.penalise(GROQ, spec)

        # A full capacity of tokens has to refill before a single call fits under the
        # five-second wait ceiling: 57 seconds at 30 per minute. A refused reservation
        # deducts nothing, so one store can be walked forward through all four instants.
        for elapsed, admitted in ((0.0, False), (30.0, False), (56.0, False), (57.0, True)):
            with self.subTest(elapsed=elapsed):
                self.clock.now = EPOCH + timedelta(seconds=elapsed)
                self.assertEqual(store.reserve(GROQ, spec) is not None, admitted)

        self.assertEqual(store.state(GROQ).tokens, -2.5)

    def test_a_wait_longer_than_the_policy_advances_instead_of_blocking(self):
        # `never_sleeps` is the assertion: if the router blocked here the test fails rather
        # than passing slowly.
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ, requests_per_minute=2)
        gemini = FakeClient(GEMINI)
        router = self.build(groq, gemini, store=store, max_wait_seconds=0.0)

        first = router.generate("one", FALLBACK)
        second = router.generate("two", FALLBACK)
        third = router.generate("three", FALLBACK)

        self.assertEqual([first.provider, second.provider], [GROQ, GROQ])
        self.assertEqual(third.provider, GEMINI)
        self.assertAttempts(third, [Attempt(GROQ, SKIPPED_RATE), Attempt(GEMINI, OK)])
        self.assertEqual(len(groq.calls), 2)

    def test_a_wait_inside_the_policy_is_slept_off_rather_than_skipped(self):
        # Two tokens over a two-second window, so the rate is one per second and the
        # arithmetic is readable: the third call is a second in debt, the fourth two, the
        # fifth three -- and the sixth is past the ceiling and advances.
        slept: list[float] = []
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ, requests_per_minute=2)
        router = self.build(
            groq, store=store, window_seconds=2.0, max_wait_seconds=3.0, sleep=slept.append
        )

        providers = [router.generate(f"prompt {index}", FALLBACK).provider for index in range(6)]

        self.assertEqual(providers, [GROQ] * 5 + [TEMPLATE])
        self.assertEqual(slept, [1.0, 2.0, 3.0])
        self.assertEqual(len(groq.calls), 5)

    def test_the_bucket_refills_continuously_rather_than_on_a_boundary(self):
        # At 30 per minute the bucket regains one token every two seconds. Handing back all
        # thirty on the minute is what produces a burst straight into a 429.
        spec = BucketSpec(30, 60.0)
        self.assertAlmostEqual(spec.rate, 0.5)
        self.assertAlmostEqual(_refilled(0.0, EPOCH, EPOCH + timedelta(seconds=2), spec), 1.0)
        self.assertAlmostEqual(_refilled(0.0, EPOCH, EPOCH + timedelta(seconds=10), spec), 5.0)
        # And never above capacity, however long the process idled.
        self.assertAlmostEqual(_refilled(0.0, EPOCH, EPOCH + timedelta(days=1), spec), 30.0)

    def test_a_clock_that_went_backwards_does_not_confiscate_tokens(self):
        # NTP steps backwards. Draining a bucket because of it would throttle a healthy
        # provider for reasons nobody could ever reconstruct from a log.
        spec = BucketSpec(30, 60.0)
        self.assertAlmostEqual(
            _refilled(10.0, EPOCH, EPOCH - timedelta(seconds=3600), spec), 10.0
        )

    def test_the_bucket_is_not_drained_while_the_circuit_is_open(self):
        # `admit` runs before `reserve`. A skipped call that still deducted a token would
        # keep an already-open provider rate limited for a window after it recovered.
        store = MemoryCircuitStore(clock=self.clock)
        router = self.build(
            FakeClient(GROQ, script=[unavailable("down", provider=GROQ)] * 3), store=store
        )
        for index in range(3):
            router.generate(f"prompt {index}", FALLBACK)
        tokens_when_opened = store.state(GROQ).tokens

        for index in range(5):
            router.generate(f"after {index}", FALLBACK)

        self.assertEqual(store.state(GROQ).tokens, tokens_when_opened)
        self.assertEqual(tokens_when_opened, 27.0)

    def test_a_provider_with_no_rate_at_all_is_rejected_at_construction(self):
        # Loudly, and before a run starts. A capacity of zero read as "unlimited" is the
        # reading that produces the 429 storm this whole module exists to prevent.
        for rpm in (0, -1):
            with self.subTest(requests_per_minute=rpm):
                with self.assertRaises(ValueError):
                    self.build(FakeClient(GROQ, requests_per_minute=rpm))

        with self.assertRaises(ValueError):
            BucketSpec(30, 0.0)
        with self.assertRaises(ValueError):
            BucketSpec(30, -1.0)

    def test_the_spec_comes_from_the_client_rather_than_a_constant(self):
        router = self.build(
            FakeClient(GROQ, requests_per_minute=30),
            FakeClient(GEMINI, requests_per_minute=10),
            FakeClient(NVIDIA, requests_per_minute=40),
        )

        self.assertEqual(
            {name: spec.capacity for name, spec in router.specs.items()},
            {GROQ: 30, GEMINI: 10, NVIDIA: 40},
        )

    def test_a_second_router_inherits_the_state_the_first_left_behind(self):
        # The restart property, in the only form the in-memory store can offer it. A fresh
        # process must not read a full bucket and send straight into the 429 it was already
        # being punished for.
        store = MemoryCircuitStore(clock=self.clock)
        first = self.build(FakeClient(GROQ, requests_per_minute=2), store=store)
        first.generate("one", FALLBACK)
        first.generate("two", FALLBACK)

        second = self.build(FakeClient(GROQ, requests_per_minute=2), store=store,
                            max_wait_seconds=0.0)
        result = second.generate("three", FALLBACK)

        self.assertAttempts(result, [Attempt(GROQ, SKIPPED_RATE)])
        self.assertEqual(store.state(GROQ).tokens, 0.0)

    def test_concurrent_callers_queue_on_the_rate_not_on_each_other(self):
        # The deduction happens under the lock and the balance is allowed to go negative,
        # which is what makes N threads share one bucket correctly. With the clock frozen
        # and no wait allowed, exactly `capacity` reservations can succeed -- a store that
        # read the balance, decided, then wrote it would hand out more than eight.
        spec = BucketSpec(8, 60.0)
        store = MemoryCircuitStore(clock=self.clock)
        store.ensure(GROQ, spec)
        workers = 32
        barrier = Barrier(workers)
        outcomes: list[float | None] = []
        lock = threading.Lock()

        def reserve() -> None:
            barrier.wait(timeout=30)
            granted = store.reserve(GROQ, spec, max_wait_seconds=0.0)
            with lock:
                outcomes.append(granted)

        threads = [threading.Thread(target=reserve) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(len(outcomes), workers)
        self.assertEqual(sum(1 for granted in outcomes if granted is not None), spec.capacity)
        self.assertEqual(store.state(GROQ).tokens, 0.0)


# --- the durable store ----------------------------------------------------------------------


class TheDurableStore(unittest.TestCase):
    """`llm_rate_buckets`: the half of the router that survives a restart.

    Postgres itself is not available in the pure suite, so what is asserted here is the
    shape of the SQL -- and the shape is where the safety lives. The reservation check has
    to be inside the UPDATE, and the failure CASE has to read the row's old state, for the
    same reason `providers/budget._SPEND` does: two workers that both read a balance would
    both conclude they could spend it.
    """

    def store(self, results=None):
        stub = StubConnection(results=results)
        return DatabaseCircuitStore(stub.factory(), clock=Clock()), stub

    def test_ensure_never_resets_a_row_a_dead_process_left_behind(self):
        # ON CONFLICT DO UPDATE here would be the restart-clears-the-limiter bug wearing a
        # helpful face: every crash would hand the worker a full bucket and a closed circuit.
        store, stub = self.store()
        store.ensure(GROQ, BucketSpec(30))

        collapsed = stub.collapsed()
        self.assertTrue(collapsed.startswith("insert into llm_rate_buckets"), collapsed)
        self.assertIn("on conflict (provider) do nothing", collapsed)
        self.assertNotIn("do update", collapsed)
        self.assertEqual(stub.commits, 1)

    def test_the_reservation_check_lives_inside_the_update(self):
        # THE safety property, and checkable without a database. A read-then-write
        # implementation passes every single-threaded test and hands two workers the same
        # last token in production.
        store, stub = self.store(results=[[(-1.0,)]])
        store.reserve(GROQ, BucketSpec(30, 60.0), max_wait_seconds=5.0)

        collapsed = stub.collapsed()
        self.assertTrue(collapsed.startswith("update llm_rate_buckets"), collapsed)
        self.assertIn("- 1 >= %(floor)s", collapsed)
        self.assertIn("returning tokens", collapsed)
        self.assertEqual(len(stub.statements), 1, "a select preceded the reservation")
        self.assertNotIn("select", collapsed)

    def test_the_refill_expression_is_identical_in_the_set_and_the_where(self):
        # Written once and interpolated twice on purpose. Typing it out twice is how the
        # guard and the assignment drift apart, and the drift is silent: the WHERE admits a
        # call the SET then charges differently.
        store, stub = self.store(results=[[(0.0,)]])
        store.reserve(GROQ, BucketSpec(30, 60.0))

        query = stub.statements[0][0]
        self.assertEqual(query.count(router_module._BALANCE), 2)

    def test_the_reservation_carries_every_parameter_its_sql_names(self):
        # A missing key is a psycopg error at the first throttle rather than at import.
        store, stub = self.store(results=[[(2.0,)]])
        spec = BucketSpec(30, 60.0)
        store.reserve(GROQ, spec, max_wait_seconds=5.0)

        params = stub.statements[0][1]
        self.assertEqual(set(params), {"provider", "now", "capacity", "rate", "floor"})
        self.assertEqual(params["provider"], GROQ)
        self.assertEqual(params["capacity"], 30.0)
        self.assertAlmostEqual(params["rate"], 0.5)
        self.assertAlmostEqual(params["floor"], -2.5)

    def test_a_reservation_the_database_refused_is_reported_as_no_token(self):
        store, stub = self.store(results=[[]])
        self.assertIsNone(store.reserve(GROQ, BucketSpec(30, 60.0)))

    def test_the_wait_is_derived_from_the_balance_the_database_returned(self):
        store, _ = self.store(results=[[(-2.0,)]])
        self.assertAlmostEqual(store.reserve(GROQ, BucketSpec(30, 60.0)), 4.0)

        store, _ = self.store(results=[[(7.0,)]])
        self.assertEqual(store.reserve(GROQ, BucketSpec(30, 60.0)), 0.0)

    def test_the_probe_transition_is_one_statement_that_exactly_one_caller_can_win(self):
        store, stub = self.store(results=[[(HALF_OPEN,)]])

        self.assertTrue(store.admit(GROQ))

        collapsed = stub.collapsed()
        self.assertTrue(collapsed.startswith("update llm_rate_buckets"), collapsed)
        self.assertIn(f"set circuit_state = '{HALF_OPEN}'", collapsed)
        self.assertIn(f"circuit_state in ('{OPEN}', '{HALF_OPEN}')", collapsed)
        self.assertIn("opened_at is not null", collapsed)
        self.assertIn("make_interval(secs => %(cooldown)s)", collapsed)
        self.assertIn("returning circuit_state", collapsed)
        # Re-stamping opened_at is what makes it self-healing rather than stuck half-open.
        self.assertIn("opened_at = %(now)s", collapsed)
        self.assertEqual(len(stub.statements), 1)

    def test_admit_falls_back_to_a_read_when_no_probe_matched(self):
        row = (GROQ, 30.0, EPOCH, CLOSED, 0, None)
        store, stub = self.store(results=[[], [row]])

        self.assertTrue(store.admit(GROQ))
        self.assertEqual(len(stub.statements), 2)
        self.assertIn("select", stub.collapsed(1))

        opened = (GROQ, 30.0, EPOCH, OPEN, 3, EPOCH)
        store, _ = self.store(results=[[], [opened]])
        self.assertFalse(store.admit(GROQ))

        missing, _ = self.store(results=[[], []])
        self.assertFalse(missing.admit(GROQ))

    def test_a_failed_probe_re_opens_in_the_same_statement_that_counts_it(self):
        # The CASE arms read the row's OLD values, which is what makes this one statement
        # rather than a read, a decision and a write.
        store, stub = self.store(results=[[(OPEN,)]])

        self.assertEqual(store.record_failure(GROQ, threshold=FAILURE_THRESHOLD), OPEN)

        collapsed = stub.collapsed()
        self.assertIn("consecutive_failures = consecutive_failures + 1", collapsed)
        self.assertIn(f"when circuit_state = '{HALF_OPEN}' then '{OPEN}'", collapsed)
        self.assertIn(f"when consecutive_failures + 1 >= %(threshold)s then '{OPEN}'", collapsed)
        self.assertIn("returning circuit_state", collapsed)
        self.assertEqual(stub.statements[0][1]["threshold"], FAILURE_THRESHOLD)

    def test_a_missing_row_reports_closed_rather_than_raising(self):
        store, _ = self.store(results=[[]])
        self.assertEqual(store.record_failure(GROQ), CLOSED)

    def test_success_clears_the_state_a_failure_left(self):
        store, stub = self.store()
        store.record_success(GROQ)

        collapsed = stub.collapsed()
        self.assertIn(f"circuit_state = '{CLOSED}'", collapsed)
        self.assertIn("consecutive_failures = 0", collapsed)
        self.assertIn("opened_at = null", collapsed)

    def test_the_penalty_never_tops_a_bucket_up(self):
        # `least`, not an assignment. A provider already 40 tokens in debt must not be
        # handed a shallower deficit by a second 429.
        store, stub = self.store()
        store.penalise(GROQ, BucketSpec(30))

        collapsed = stub.collapsed()
        self.assertIn("set tokens = least(tokens, -%(capacity)s::numeric)", collapsed)
        self.assertEqual(stub.statements[0][1]["capacity"], 30.0)

    def test_every_write_commits_its_own_transaction(self):
        # A circuit that opened but was rolled back with whatever else the caller was doing
        # is a circuit that never opened.
        row = (GROQ, 30.0, EPOCH, OPEN, 3, EPOCH)
        cases = [
            ("ensure", [], lambda store: store.ensure(GROQ, BucketSpec(30))),
            ("penalise", [], lambda store: store.penalise(GROQ, BucketSpec(30))),
            ("record_success", [], lambda store: store.record_success(GROQ)),
            ("record_failure", [[(OPEN,)]], lambda store: store.record_failure(GROQ)),
            ("reserve", [[(2.0,)]], lambda store: store.reserve(GROQ, BucketSpec(30, 60.0))),
            ("state", [[row]], lambda store: store.state(GROQ)),
            ("admit", [[(HALF_OPEN,)]], lambda store: store.admit(GROQ)),
            # The two-statement path through admit: no probe matched, so it reads instead.
            ("admit (no probe)", [[], [row]], lambda store: store.admit(GROQ)),
        ]
        for name, results, call in cases:
            with self.subTest(call=name):
                store, stub = self.store(results=results)
                call(store)
                self.assertEqual(stub.commits, 1)

    def test_state_reads_the_row_back_as_the_router_understands_it(self):
        row = (GROQ, "27.5", EPOCH, OPEN, 3, EPOCH)
        store, _ = self.store(results=[[row]])

        state = store.state(GROQ)

        self.assertEqual(
            state,
            BucketState(
                provider=GROQ,
                tokens=27.5,
                refilled_at=EPOCH,
                circuit_state=OPEN,
                consecutive_failures=3,
                opened_at=EPOCH,
            ),
        )
        self.assertTrue(state.is_open)

    def test_a_missing_row_is_none_rather_than_an_empty_state(self):
        store, _ = self.store(results=[[]])
        self.assertIsNone(store.state(GROQ))

    def test_the_module_imports_no_database_driver(self):
        # The store talks to whatever connection it is handed. Importing psycopg here would
        # make the seam a lie and drag a driver into anything that walks the chain.
        source = Path(router_module.__file__ or "").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(import|from)\s+psycopg")
        self.assertNotRegex(source, r"(?m)^\s*(import|from)\s+httpx")


# --- the cache, which is also the ledger ----------------------------------------------------


class TheCache(RouterTestCase):
    """One table, `llm_calls`. A re-run after a crash re-spends nothing on completed work."""

    def test_a_hit_issues_no_request_and_touches_no_bucket(self):
        key = prompt_hash(PROMPT)
        cache = DictCache({key: CachedResponse(provider=GROQ, model=GROQ_MODEL, text="cached")})
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ)
        router = self.build(groq, store=store, cache=cache)
        tokens_before = store.state(GROQ).tokens

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, "cached")
        self.assertEqual(result.provider, GROQ)
        self.assertEqual(result.model, GROQ_MODEL)
        self.assertTrue(result.cached)
        self.assertEqual(result.attempts, ())
        self.assertEqual(groq.calls, [])
        self.assertEqual(store.state(GROQ).tokens, tokens_before)

    def test_a_miss_generates_and_records_what_it_cost(self):
        cache = DictCache()
        router = self.build(FakeClient(GROQ), cache=cache, monotonic=Monotonic(step=0.25))

        result = router.generate(PROMPT, FALLBACK)

        self.assertFalse(result.cached)
        self.assertEqual(cache.lookups, [prompt_hash(PROMPT)])
        self.assertEqual(len(cache.successes), 1)
        recorded = cache.successes[0]
        self.assertEqual(recorded["key"], prompt_hash(PROMPT))
        self.assertEqual(recorded["provider"], GROQ)
        self.assertEqual(recorded["model"], "groq-model")
        self.assertEqual(recorded["text"], result.text)
        self.assertEqual(recorded["input_tokens"], 120)
        self.assertEqual(recorded["output_tokens"], 64)
        self.assertEqual(recorded["latency_ms"], 250)

    def test_the_second_call_for_the_same_prompt_is_served_from_the_ledger(self):
        cache = DictCache()
        groq = FakeClient(GROQ)
        router = self.build(groq, cache=cache)

        first = router.generate(PROMPT, FALLBACK)
        second = router.generate(PROMPT, FALLBACK)

        self.assertEqual(len(groq.calls), 1)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(first.text, second.text)

    def test_the_key_is_the_prompt_and_deliberately_not_the_model(self):
        # A run that got half its pitches from Groq before the process died comes back,
        # hashes the same prompts and finds the same answers -- even for the leads that
        # would now route to Gemini because Groq's circuit is open. Folding the model into
        # the key would give every prompt three entries and re-spend on each in turn.
        cache = DictCache()
        store = MemoryCircuitStore(clock=self.clock)
        groq = FakeClient(GROQ)
        gemini = FakeClient(GEMINI)
        first = self.build(groq, gemini, store=store, cache=cache)
        first.generate(PROMPT, FALLBACK)

        # A second run, different chain order, same prompt.
        second = self.build(gemini, groq, store=store, cache=cache)
        result = second.generate(PROMPT, FALLBACK)

        self.assertTrue(result.cached)
        self.assertEqual(result.provider, GROQ)
        self.assertEqual(gemini.calls, [])

    def test_the_digest_is_the_whole_sha256_of_the_prompt(self):
        # Not truncated. A collision here would serve one business's pitch under another
        # business's name, which is the single worst thing this system could do.
        digest = prompt_hash(PROMPT)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, prompt_hash(PROMPT))
        self.assertNotEqual(digest, prompt_hash(PROMPT + " "))
        self.assertNotIn(PROMPT, digest)

    def test_a_non_string_prompt_is_a_type_error_rather_than_a_digest(self):
        for value in (None, 5, b"bytes", ["prompt"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    prompt_hash(value)

    def test_a_failure_is_recorded_at_every_link_and_cached_at_none_of_them(self):
        # The partial unique index in 0007 is what makes this true in Postgres. Caching a
        # 429 as the answer to a prompt would make one throttled minute permanent: every
        # re-run for the rest of the deployment's life would serve the failure.
        cache = DictCache()
        router = self.build(
            FakeClient(GROQ, script=[rate_limited("throttled", provider=GROQ)]),
            FakeClient(GEMINI, script=[unavailable("down", provider=GEMINI)]),
            cache=cache,
        )

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, TEMPLATE)
        self.assertEqual(
            [(row["provider"], row["code"]) for row in cache.failures],
            [(GROQ, "provider_rate_limited"), (GEMINI, "provider_unavailable")],
        )
        self.assertEqual(cache.successes, [])
        self.assertEqual(cache.rows, {})
        # And the next call still tries: errors stay outside the index, so they stay
        # retryable.
        self.assertIsNone(cache.get(prompt_hash(PROMPT)))

    def test_the_recorded_failure_is_a_taxonomy_code_and_never_a_message(self):
        # `response` on an error row holds the CODE from this project's own five-code
        # taxonomy, written by providers/errors.py rather than by a vendor. That is safe by
        # construction rather than by inspection.
        cache = DictCache()
        failure = rate_limited(f"upstream said {SENTINEL_BODY}", provider=GROQ)
        self.build(FakeClient(GROQ, script=[failure]), cache=cache).generate(PROMPT, FALLBACK)

        recorded = cache.failures[0]
        self.assertEqual(recorded["code"], "provider_rate_limited")
        self.assertIn(recorded["code"], STATUS_BY_CODE)
        self.assertNotIn(SENTINEL_BODY, str(recorded))
        self.assertNotIn(PROMPT, str(recorded))

    def test_the_latency_recorded_is_never_negative(self):
        # `monotonic` is monotonic by contract and not always in practice. A negative
        # latency in the ledger is a number an operator would read as real.
        cache = DictCache()
        backwards = iter([10.0, 1.0, 10.0, 1.0])
        router = self.build(
            FakeClient(GROQ, script=[unavailable("down", provider=GROQ)]),
            cache=cache,
            monotonic=lambda: next(backwards),
        )

        router.generate(PROMPT, FALLBACK)

        self.assertEqual(cache.failures[0]["latency_ms"], 0)

    def test_the_null_cache_remembers_nothing_and_never_fails(self):
        # Not a test double. A run against a machine with no Postgres still has to produce a
        # sheet, so "no cache" is supported in exactly the way "no LLM provider" is.
        cache = NullCache()
        self.assertIsNone(cache.get("anything"))
        self.assertIsNone(cache.record_success("k", provider=GROQ, model="m", text="t"))
        self.assertIsNone(cache.record_failure("k", provider=GROQ, model="m", code="x"))

        groq = FakeClient(GROQ)
        router = self.build(groq)  # the default cache
        self.assertIsInstance(router.cache, NullCache)
        router.generate(PROMPT, FALLBACK)
        router.generate(PROMPT, FALLBACK)
        self.assertEqual(len(groq.calls), 2, "a NullCache started remembering")

    def test_the_router_does_not_re_validate_text_the_cache_hands_back(self):
        # DOCUMENTED GAP, not an endorsement. `generate()` validates the fallback up front
        # precisely so that "no answer" is unrepresentable -- and then discards it without
        # looking, for any CachedResponse the cache returns. A cache that honours its own
        # contract never returns blank text, so this is only reachable through a damaged
        # `llm_calls` row; `TheCacheContract.test_a_row_whose_response_is_blank_is_a_miss`
        # is the xfail that proves such a row is served today. Pinned here so that whoever
        # closes that hole can see both ends of it.
        blank = DictCache({prompt_hash(PROMPT): CachedResponse(GROQ, GROQ_MODEL, "   ")})

        result = self.build(FakeClient(GROQ), cache=blank).generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, "   ")
        self.assertNotEqual(result.text, FALLBACK)
        self.assertTrue(result.cached)


class TheCacheContract(unittest.TestCase):
    """`ResponseCache` against a stub connection: the SQL, and what it refuses to serve."""

    def cache(self, results=None):
        stub = StubConnection(results=results)
        return ResponseCache(stub.factory()), stub

    def test_the_lookup_is_restricted_to_successful_rows(self):
        cache, stub = self.cache(results=[[(GROQ, GROQ_MODEL, "pitch")]])

        cache.get("abc")

        collapsed = stub.collapsed()
        self.assertIn("from llm_calls", collapsed)
        self.assertIn(f"status = '{STATUS_OK}'", collapsed)
        self.assertIn("prompt_hash = %s", collapsed)
        self.assertEqual(stub.statements[0][1], ("abc",))

    def test_a_hit_is_the_three_columns_the_router_replays(self):
        cache, _ = self.cache(results=[[(GROQ, GROQ_MODEL, "pitch")]])
        self.assertEqual(
            cache.get("abc"), CachedResponse(provider=GROQ, model=GROQ_MODEL, text="pitch")
        )

    def test_no_row_is_a_miss(self):
        cache, _ = self.cache(results=[[]])
        self.assertIsNone(cache.get("abc"))

    def test_a_row_whose_response_is_null_is_a_miss(self):
        # A successful call always has text. A row that says otherwise is damaged, and
        # serving an empty pitch from it is worse than paying for the call again.
        cache, _ = self.cache(results=[[(GROQ, GROQ_MODEL, None)]])
        self.assertIsNone(cache.get("abc"))

    @pytest.mark.xfail(
        reason=(
            "PROVEN BUG: ResponseCache.get (cache.py:187) guards only `row[2] is None`, so a "
            "damaged llm_calls row whose response is '' or whitespace is served as a cache "
            "hit. The method's own docstring (cache.py:181-183) says such a row must be a "
            "miss because 'serving an empty pitch from it is worse than paying for the call "
            "again'. LLMRouter.generate then returns that blank text and discards the "
            "validated deterministic fallback. Not fixed here by instruction."
        ),
        strict=True,
    )
    def test_a_row_whose_response_is_blank_is_a_miss_too(self):
        # No subTest here on purpose: pytest-subtests reports each subtest as its own
        # outcome, which turns one honest xfail into an xfail per case plus an XPASS for
        # the enclosing test. Three plain assertions, first failure wins.
        empty, _ = self.cache(results=[[(GROQ, GROQ_MODEL, "")]])
        self.assertIsNone(empty.get("abc"), "a row with an empty response was served")

        spaces, _ = self.cache(results=[[(GROQ, GROQ_MODEL, "   ")]])
        self.assertIsNone(spaces.get("abc"), "a row with a blank response was served")

        newlines, _ = self.cache(results=[[(GROQ, GROQ_MODEL, "\n\t")]])
        self.assertIsNone(newlines.get("abc"), "a row with a whitespace response was served")

    def test_the_success_insert_infers_the_partial_index(self):
        # `ON CONFLICT (prompt_hash) WHERE status = 'ok'`. The predicate is not optional:
        # without it Postgres looks for a total unique index on prompt_hash, finds none, and
        # raises rather than doing nothing.
        cache, stub = self.cache()
        cache.record_success(
            "abc", provider=GROQ, model=GROQ_MODEL, text="pitch",
            input_tokens=10, output_tokens=20, latency_ms=250,
        )

        collapsed = stub.collapsed()
        self.assertTrue(collapsed.startswith("insert into llm_calls"), collapsed)
        self.assertIn(
            f"on conflict (prompt_hash) where status = '{STATUS_OK}' do nothing", collapsed
        )
        # DO NOTHING, not DO UPDATE: two workers racing on the same prompt both generated a
        # valid answer, and overwriting means the text an operator is reading changes
        # under them mid-run.
        self.assertNotIn("do update", collapsed)
        self.assertEqual(
            stub.statements[0][1], ("abc", GROQ, GROQ_MODEL, 10, 20, 250, "pitch")
        )
        self.assertEqual(stub.commits, 1)

    def test_an_error_row_is_a_plain_insert_that_can_repeat(self):
        cache, stub = self.cache()
        cache.record_failure(
            "abc", provider=GROQ, model=GROQ_MODEL, code="provider_rate_limited", latency_ms=12
        )

        collapsed = stub.collapsed()
        self.assertTrue(collapsed.startswith("insert into llm_calls"), collapsed)
        self.assertIn(f"'{STATUS_ERROR}'", collapsed)
        # No ON CONFLICT: a failed call is recorded as many times as it fails.
        self.assertNotIn("on conflict", collapsed)
        self.assertEqual(
            stub.statements[0][1], ("abc", GROQ, GROQ_MODEL, 12, "provider_rate_limited")
        )

    def test_the_prompt_itself_never_reaches_a_row(self):
        # Only the digest. The prompt carries a business name, an address and a phone
        # number, and `llm_calls` is a table someone reads months later.
        cache, stub = self.cache()
        key = prompt_hash(PROMPT)
        cache.record_success("x", provider=GROQ, model=GROQ_MODEL, text="pitch")
        cache.record_failure("y", provider=GROQ, model=GROQ_MODEL, code="provider_unavailable")
        cache.get(key)

        for query, params in stub.statements:
            with self.subTest(query=query.split()[0]):
                self.assertNotIn(PROMPT, str(params))
                self.assertNotIn("Kumar Studio", str(params))

    def test_the_status_values_match_the_index_the_migration_builds(self):
        # Changing these here without changing 0007 silently turns the cache off: every
        # lookup misses, every run re-spends, and nothing raises.
        self.assertEqual((STATUS_OK, STATUS_ERROR), ("ok", "error"))
        migration = (ROOT / "migrations" / "0007_llm.sql").read_text(encoding="utf-8")
        self.assertIn(
            f"CREATE UNIQUE INDEX ON llm_calls (prompt_hash) WHERE status = '{STATUS_OK}'",
            migration,
        )

    def test_the_module_imports_no_database_driver(self):
        source = Path(cache_module.__file__ or "").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(import|from)\s+psycopg")


# --- the provider clients -------------------------------------------------------------------


class TheProviderClients(RouterTestCase):
    """The outbound request, and the one thing about it that could leak a credential."""

    def test_groq_sends_the_documented_openai_envelope(self):
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=groq_body("Groq wrote this."))

        client = self.mock_client(GroqClient, handler)
        completion = client.complete(PROMPT, system="Be terse.", max_tokens=100, temperature=0.5)

        self.assertEqual(str(seen[0].url), GROQ_URL)
        self.assertEqual(seen[0].method, "POST")
        self.assertEqual(seen[0].headers["Authorization"], f"Bearer {SENTINEL_KEY}")
        self.assertEqual(seen[0].headers["Accept"], "application/json")
        self.assertEqual(
            json.loads(seen[0].content),
            {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": PROMPT},
                ],
                "temperature": 0.5,
                "max_completion_tokens": 100,
                "stream": False,
            },
        )
        self.assertEqual(completion.provider, GROQ)
        self.assertEqual(completion.model, GROQ_MODEL)
        self.assertEqual(completion.text, "Groq wrote this.")

    def test_the_max_tokens_field_differs_per_gateway(self):
        # Groq takes the current `max_completion_tokens`; NVIDIA's NIM gateway takes the
        # original `max_tokens`. Sending the wrong one is accepted and ignored, so the
        # symptom is a truncated pitch rather than an error.
        for cls, field, url, model in (
            (GroqClient, "max_completion_tokens", GROQ_URL, GROQ_MODEL),
            (NvidiaClient, "max_tokens", NVIDIA_URL, NVIDIA_MODEL),
        ):
            with self.subTest(client=cls.__name__):
                seen: list[httpx.Request] = []

                def handler(request, seen=seen):
                    seen.append(request)
                    return httpx.Response(200, json=groq_body())

                client = self.mock_client(cls, handler)
                client.complete(PROMPT)

                body = json.loads(seen[0].content)
                self.assertEqual(str(seen[0].url), url)
                self.assertEqual(body["model"], model)
                self.assertEqual(body[field], DEFAULT_MAX_TOKENS)
                self.assertEqual(body["temperature"], DEFAULT_TEMPERATURE)
                self.assertFalse(body["stream"])

    def test_a_prompt_with_no_system_message_sends_one_message(self):
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=groq_body())

        self.mock_client(GroqClient, handler).complete(PROMPT)

        self.assertEqual(
            json.loads(seen[0].content)["messages"], [{"role": "user", "content": PROMPT}]
        )

    def test_gemini_sends_the_documented_generate_content_envelope(self):
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=gemini_body([{"text": "Gemini wrote this."}]))

        client = self.mock_client(GeminiClient, handler)
        completion = client.complete(PROMPT, system="Be terse.")

        self.assertEqual(
            str(seen[0].url), f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
        )
        self.assertEqual(
            json.loads(seen[0].content),
            {
                "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
                "generationConfig": {
                    "temperature": DEFAULT_TEMPERATURE,
                    "maxOutputTokens": DEFAULT_MAX_TOKENS,
                },
                "systemInstruction": {"parts": [{"text": "Be terse."}]},
            },
        )
        self.assertEqual(completion.text, "Gemini wrote this.")

    def test_geminis_key_travels_in_a_header_and_never_in_the_url(self):
        # A deliberate divergence from Gemini's own quickstart, which passes `?key=`. Every
        # error path in this project has to be safe to log, and a URL in a message is a
        # credential in a logfile. A header cannot end up in httpx's `request.url`.
        seen: list[httpx.Request] = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=gemini_body([{"text": "ok"}]))

        client = self.mock_client(GeminiClient, handler)
        client.complete(PROMPT)

        self.assertEqual(seen[0].headers["x-goog-api-key"], SENTINEL_KEY)
        self.assertNotIn(SENTINEL_KEY, str(seen[0].url))
        self.assertNotIn("key", seen[0].url.params)

    def test_no_client_puts_the_key_in_the_url(self):
        for cls, body in (
            (GroqClient, groq_body()),
            (NvidiaClient, groq_body()),
            (GeminiClient, gemini_body([{"text": "ok"}])),
        ):
            with self.subTest(client=cls.__name__):
                seen: list[httpx.Request] = []

                def handler(request, body=body, seen=seen):
                    seen.append(request)
                    return httpx.Response(200, json=body)

                self.mock_client(cls, handler).complete(PROMPT)
                self.assertNotIn(SENTINEL_KEY, str(seen[0].url))

    def test_an_unconfigured_client_sends_no_auth_header_at_all(self):
        # `None` and `""` both mean "not configured" -- an empty env var is how a setting is
        # removed. An empty bearer token buys a 401 instead of a clean skip.
        for cls in (GroqClient, NvidiaClient, GeminiClient):
            for key in (None, ""):
                with self.subTest(client=cls.__name__, key=repr(key)):
                    client = cls(key)
                    self.addCleanup(client.close)
                    self.assertFalse(client.configured)
                    self.assertNotIn("authorization", client._client.headers)
                    self.assertNotIn("x-goog-api-key", client._client.headers)

    def test_gemini_never_gets_a_model_named_empty_string(self):
        for model in (None, ""):
            with self.subTest(model=repr(model)):
                client = GeminiClient(SENTINEL_KEY, model=model)
                self.addCleanup(client.close)
                self.assertEqual(client.model, GEMINI_MODEL)
                self.assertIn(GEMINI_MODEL, client._url())

    def test_a_trailing_slash_on_the_base_url_does_not_double(self):
        client = GeminiClient(SENTINEL_KEY, base_url=GEMINI_BASE_URL + "/")
        self.addCleanup(client.close)
        self.assertEqual(client._url(), f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent")

    def test_an_unusable_prompt_is_refused_before_a_socket_opens(self):
        seen: list[httpx.Request] = []

        def handler(request):  # pragma: no cover - the guard
            seen.append(request)
            return httpx.Response(200, json=groq_body())

        client = self.mock_client(GroqClient, handler)
        for prompt in ("", "   ", None, 5):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    client.complete(prompt)
        for max_tokens in (0, -1):
            with self.subTest(max_tokens=max_tokens):
                with self.assertRaises(ValueError):
                    client.complete(PROMPT, max_tokens=max_tokens)

        self.assertEqual(seen, [])

    def test_no_client_retries(self):
        # Retrying is the router's decision to make, once, with a circuit breaker and a
        # persisted bucket in front of it. A retry loop down here multiplies by whatever the
        # router does above, which is exactly the storm a 429 must not cause.
        attempts: list[httpx.Request] = []

        def handler(request):
            attempts.append(request)
            return httpx.Response(500, json={"error": SENTINEL_BODY})

        client = self.mock_client(GroqClient, handler)
        with self.assertRaises(ProviderError):
            client.complete(PROMPT)

        self.assertEqual(len(attempts), 1)

    def test_the_client_closes_its_transport(self):
        client = GroqClient(SENTINEL_KEY, transport=httpx.MockTransport(responder(groq_body())))
        with client as entered:
            self.assertIs(entered, client)
            entered.complete(PROMPT)

        with self.assertRaises(RuntimeError) as caught:
            client.complete(PROMPT)
        # ProviderError IS a RuntimeError, so assertRaises alone would pass on a closed
        # client that quietly reported "provider unavailable" instead.
        self.assertNotIsInstance(caught.exception, ProviderError)


class StatusMappingTests(RouterTestCase):
    """Every upstream status lands on the agreed code, and leaks nothing on the way."""

    CASES = [
        (429, "provider_rate_limited"),
        (401, "provider_auth_failed"),
        (403, "provider_auth_failed"),
        (400, "provider_bad_response"),
        (404, "provider_bad_response"),
        (422, "provider_bad_response"),
        (500, "provider_unavailable"),
        (502, "provider_unavailable"),
        (503, "provider_unavailable"),
    ]

    def error_body(self) -> dict:
        return {
            "error": {"message": SENTINEL_BODY, "type": "quota"},
            "request_url": f"{GROQ_URL}?key={SENTINEL_KEY}",
        }

    def assertNoSecrets(self, error: ProviderError) -> None:
        for message in (str(error), error.message, repr(error)):
            self.assertNotIn(SENTINEL_KEY, message)
            self.assertNotIn(SENTINEL_BODY, message)
            self.assertNotIn("peacock", message)
            self.assertNotIn("api.groq.com", message)
            self.assertNotIn("googleapis.com", message)

    def test_status_codes_map_onto_the_taxonomy(self):
        for status, expected in self.CASES:
            with self.subTest(status=status):
                client = self.mock_client(GroqClient, responder(self.error_body(), status))
                with self.assertRaises(ProviderError) as caught:
                    client.complete(PROMPT)

                error = caught.exception
                self.assertEqual(error.code, expected)
                self.assertEqual(error.status, STATUS_BY_CODE[expected])
                self.assertEqual(error.retryable, RETRYABLE_BY_CODE[expected])
                self.assertEqual(error.provider, GROQ)
                self.assertIn(str(status), str(error))
                self.assertNoSecrets(error)

    def test_five_hundreds_are_unavailable_here_rather_than_a_bad_response(self):
        # Pinned because `errors.from_status` says the opposite and `test_searchapi.py`
        # pins that opposite. This module follows providers/tinyfish.py instead: 5xx means
        # upstream is down, not that upstream sent us nonsense. Both are retryable and both
        # count as one failure against the breaker, so nothing downstream behaves
        # differently -- the run log just reads honestly.
        error = status_error(GROQ, 503)
        self.assertEqual(error.code, "provider_unavailable")
        self.assertEqual(error.status, 503)
        self.assertTrue(error.retryable)

    def test_the_status_error_map_is_the_same_for_every_provider(self):
        for provider in CHAIN:
            for status, expected in self.CASES:
                with self.subTest(provider=provider, status=status):
                    error = status_error(provider, status)
                    self.assertEqual(error.code, expected)
                    self.assertEqual(error.provider, provider)

    def test_a_timeout_is_unavailable_and_breaks_the_exception_chain(self):
        def explode(request):
            # httpx stringifies `.request.url` into its own exception messages and a
            # traceback prints the whole chain, so `from None` is load-bearing.
            raise httpx.ReadTimeout(f"timed out reading {request.url}?key={SENTINEL_KEY}")

        client = self.mock_client(GroqClient, explode)
        with self.assertRaises(ProviderError) as caught:
            client.complete(PROMPT)

        error = caught.exception
        self.assertEqual(error.code, "provider_unavailable")
        self.assertIn("timed out", str(error))
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)
        self.assertNoSecrets(error)

    def test_a_transport_error_is_unavailable_and_breaks_the_chain(self):
        def explode(request):
            raise httpx.ConnectError(f"connection refused for {request.url}?key={SENTINEL_KEY}")

        client = self.mock_client(GeminiClient, explode)
        with self.assertRaises(ProviderError) as caught:
            client.complete(PROMPT)

        error = caught.exception
        self.assertEqual(error.code, "provider_unavailable")
        self.assertEqual(error.provider, GEMINI)
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)
        self.assertNoSecrets(error)

    def test_every_message_this_module_writes_survives_redaction(self):
        # `_safe_message` silently substitutes the generic default for any message `redact`
        # would alter, so a leaky message would not leak -- it would DISAPPEAR, taking the
        # only diagnostic with it. Every message here is static prose with a status in it.
        messages = [str(status_error(name, status)) for name in CHAIN for status, _ in self.CASES]
        messages += [
            str(bad_response(f"{name} returned {what}.", provider=name))
            for name in CHAIN
            for what in (
                "a body that was not JSON",
                "a JSON body that was not an object",
                "no choices",
                "a malformed choice",
                "a choice with no message",
                "a message with no text content",
                "no candidates",
                "a malformed candidate",
                "a candidate with no content",
                "content with no text part",
                "an empty completion",
            )
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(redact(message), message)
                self.assertNotIn(message, DEFAULT_MESSAGE.values())


class MalformedProviderResponses(RouterTestCase):
    """A 200 that is not a completion raises; it never returns junk, blank text, or a zero.

    This is where the spec's weighting lands. An extractor that guesses puts a fabricated
    number in front of a business owner; a chat client that returns "" puts an empty pitch
    there, and the operator sends it. Both are worse than an error, and an error here costs
    nothing -- the chain advances and the template is still in hand.
    """

    def assertBadResponse(self, client, expected_fragment: str) -> ProviderError:
        with self.assertRaises(ProviderError) as caught:
            client.complete(PROMPT)
        error = caught.exception
        self.assertEqual(error.code, "provider_bad_response")
        # The message is this module's own prose, not the generic fallback. `_safe_message`
        # swaps the default in for anything `redact` would touch, so a client that
        # interpolated a body would pass a no-leak assertion by having its message erased.
        self.assertNotEqual(error.message, DEFAULT_MESSAGE["provider_bad_response"])
        self.assertIn(expected_fragment, str(error))
        self.assertNotIn(SENTINEL_BODY, str(error))
        self.assertNotIn(SENTINEL_KEY, str(error))
        return error

    OPENAI_CASES = [
        ("a body that was not JSON", lambda: httpx.Response(200, text=f"<b>{SENTINEL_BODY}</b>")),
        ("a JSON body that was not an object", lambda: httpx.Response(200, json=[SENTINEL_BODY])),
        ("no choices", lambda: httpx.Response(200, json={"error": SENTINEL_BODY})),
        ("no choices", lambda: httpx.Response(200, json={"choices": []})),
        ("no choices", lambda: httpx.Response(200, json={"choices": SENTINEL_BODY})),
        ("a malformed choice", lambda: httpx.Response(200, json={"choices": [SENTINEL_BODY]})),
        ("a choice with no message", lambda: httpx.Response(200, json={"choices": [{}]})),
        (
            "a choice with no message",
            lambda: httpx.Response(200, json={"choices": [{"message": SENTINEL_BODY}]}),
        ),
        (
            "a message with no text content",
            lambda: httpx.Response(200, json={"choices": [{"message": {}}]}),
        ),
        (
            "a message with no text content",
            lambda: httpx.Response(200, json=groq_body(None)),
        ),
        (
            "a message with no text content",
            lambda: httpx.Response(200, json={"choices": [{"message": {"content": ["a"]}}]}),
        ),
        ("an empty completion", lambda: httpx.Response(200, json=groq_body(""))),
        ("an empty completion", lambda: httpx.Response(200, json=groq_body("   \n\t "))),
    ]

    def test_every_malformed_openai_body_is_an_error(self):
        for fragment, build in self.OPENAI_CASES:
            for cls in (GroqClient, NvidiaClient):
                with self.subTest(client=cls.__name__, expects=fragment):
                    client = self.mock_client(cls, lambda request, build=build: build())
                    self.assertBadResponse(client, fragment)

    GEMINI_CASES = [
        ("no candidates", lambda: httpx.Response(200, json={})),
        ("no candidates", lambda: httpx.Response(200, json={"candidates": []})),
        # What a safety block looks like: promptFeedback and no candidates. A refusal, not
        # an outage -- but it is still "no answer from this provider", and the next link or
        # the template is the right response.
        (
            "no candidates",
            lambda: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}),
        ),
        ("no candidates", lambda: httpx.Response(200, json={"candidates": SENTINEL_BODY})),
        (
            "a malformed candidate",
            lambda: httpx.Response(200, json={"candidates": [SENTINEL_BODY]}),
        ),
        ("a candidate with no content", lambda: httpx.Response(200, json={"candidates": [{}]})),
        (
            "a candidate with no content",
            lambda: httpx.Response(200, json={"candidates": [{"content": SENTINEL_BODY}]}),
        ),
        (
            "content with no text part",
            lambda: httpx.Response(200, json={"candidates": [{"content": {}}]}),
        ),
        ("content with no text part", lambda: httpx.Response(200, json=gemini_body([]))),
        (
            "content with no text part",
            lambda: httpx.Response(200, json={"candidates": [{"content": {"parts": "x"}}]}),
        ),
        ("content with no text part", lambda: httpx.Response(200, json=gemini_body([{}]))),
        (
            "content with no text part",
            lambda: httpx.Response(200, json=gemini_body([{"text": 7}])),
        ),
        # A thinking model that returned only its deliberation. Reasoning is not an answer.
        (
            "content with no text part",
            lambda: httpx.Response(
                200, json=gemini_body([{"text": "let me think", "thought": True}])
            ),
        ),
        ("an empty completion", lambda: httpx.Response(200, json=gemini_body([{"text": "  "}]))),
    ]

    def test_every_malformed_gemini_body_is_an_error(self):
        for fragment, build in self.GEMINI_CASES:
            with self.subTest(expects=fragment):
                client = self.mock_client(GeminiClient, lambda request, build=build: build())
                self.assertBadResponse(client, fragment)

    def test_a_thinking_models_deliberation_never_reaches_the_artifact(self):
        # 2.5 Flash is a thinking model. Splicing `thought: true` parts into the artifact
        # would put the model's reasoning in front of a business owner.
        client = self.mock_client(
            GeminiClient,
            responder(
                gemini_body(
                    [
                        {"text": "The user wants a pitch. First I will...", "thought": True},
                        {"text": "Kumar Studio has no website. "},
                        {"text": "A one-page booking site would fit."},
                    ]
                )
            ),
        )

        completion = client.complete(PROMPT)

        self.assertEqual(
            completion.text,
            "Kumar Studio has no website. A one-page booking site would fit.",
        )
        self.assertNotIn("First I will", completion.text)

    def test_an_empty_completion_is_never_returned_as_a_success(self):
        # Returning "" would let the caller store an empty pitch as a successful generation
        # and cache it forever, behind the partial unique index where nothing retries it.
        for cls, body in (
            (GroqClient, groq_body("")),
            (NvidiaClient, groq_body("\n \t")),
            (GeminiClient, gemini_body([{"text": ""}])),
        ):
            with self.subTest(client=cls.__name__):
                client = self.mock_client(cls, responder(body))
                self.assertBadResponse(client, "an empty completion")

    def test_a_completion_is_stripped_but_not_otherwise_rewritten(self):
        client = self.mock_client(GroqClient, responder(groq_body("  Kumar Studio.\n")))
        self.assertEqual(client.complete(PROMPT).text, "Kumar Studio.")

    def test_token_counts_are_none_rather_than_a_zero_standing_in_for_unknown(self):
        # A zero written where a count is unknown is an invented number in the ledger --
        # the same mistake this phase exists to prevent, one layer down.
        cases = [
            ({}, (None, None)),
            ({"usage": SENTINEL_BODY}, (None, None)),
            ({"usage": {}}, (None, None)),
            ({"usage": {"prompt_tokens": None, "completion_tokens": None}}, (None, None)),
            ({"usage": {"prompt_tokens": "120", "completion_tokens": "64"}}, (None, None)),
            ({"usage": {"prompt_tokens": 1.5, "completion_tokens": 2.5}}, (None, None)),
            # True is an int in Python. A boolean reaching a token column is a 1 nobody meant.
            ({"usage": {"prompt_tokens": True, "completion_tokens": False}}, (None, None)),
            ({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}, (0, 0)),
            ({"usage": {"prompt_tokens": 120, "completion_tokens": 64}}, (120, 64)),
        ]
        for extra, expected in cases:
            with self.subTest(usage=extra.get("usage")):
                body = {"choices": [{"message": {"content": "text"}}], **extra}
                client = self.mock_client(GroqClient, responder(body))
                completion = client.complete(PROMPT)
                self.assertEqual((completion.input_tokens, completion.output_tokens), expected)

    def test_gemini_token_counts_follow_the_same_rule(self):
        for usage, expected in (
            ({}, (None, None)),
            ({"promptTokenCount": "7"}, (None, None)),
            ({"promptTokenCount": 7, "candidatesTokenCount": 3}, (7, 3)),
        ):
            with self.subTest(usage=usage):
                body = gemini_body([{"text": "ok"}])
                if usage:
                    body["usageMetadata"] = usage
                client = self.mock_client(GeminiClient, responder(body))
                completion = client.complete(PROMPT)
                self.assertEqual((completion.input_tokens, completion.output_tokens), expected)

    def test_a_malformed_body_advances_the_chain_rather_than_returning_junk(self):
        # End to end over the real clients: nonsense from Groq, nonsense from Gemini, prose
        # from NVIDIA. Nothing invented, nothing blank, and no exception out of `generate`.
        groq = self.mock_client(GroqClient, responder({"choices": []}))
        gemini = self.mock_client(GeminiClient, lambda r: httpx.Response(200, text="<html/>"))
        nvidia = self.mock_client(NvidiaClient, responder(groq_body("NVIDIA wrote this.")))

        result = self.build(groq, gemini, nvidia).generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, "NVIDIA wrote this.")
        self.assertEqual(result.provider, NVIDIA)
        self.assertAttempts(
            result,
            [
                Attempt(GROQ, ERROR, "provider_bad_response"),
                Attempt(GEMINI, ERROR, "provider_bad_response"),
                Attempt(NVIDIA, OK),
            ],
        )


# --- the scenario the spec names -------------------------------------------------------------


class TheSpecScenario(RouterTestCase):
    """"Fake providers returning 429 and 5xx; asserts circuit opens, chain advances,
    deterministic fallback terminates" -- run against the real client classes.

    The fake clients elsewhere in this file test the router's logic. This class tests that
    the thing which advances in production is the code that advances here: real
    `GroqClient`, `GeminiClient` and `NvidiaClient` objects, on mock transports, answering
    with the statuses a throttled and a broken provider actually send.
    """

    def chain(self, groq_handler, gemini_handler, nvidia_handler, **kwargs):
        clients = [
            self.mock_client(GroqClient, groq_handler),
            self.mock_client(GeminiClient, gemini_handler),
            self.mock_client(NvidiaClient, nvidia_handler),
        ]
        return self.build(*clients, **kwargs)

    def test_a_429_then_a_5xx_lands_on_the_third_provider(self):
        router = self.chain(
            responder({"error": SENTINEL_BODY}, 429),
            responder({"error": SENTINEL_BODY}, 503),
            responder(groq_body("NVIDIA wrote this pitch.")),
        )

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, NVIDIA)
        self.assertEqual(result.model, NVIDIA_MODEL)
        self.assertEqual(result.text, "NVIDIA wrote this pitch.")
        self.assertAttempts(
            result,
            [
                Attempt(GROQ, ERROR, "provider_rate_limited"),
                Attempt(GEMINI, ERROR, "provider_unavailable"),
                Attempt(NVIDIA, OK),
            ],
        )

    def test_every_provider_down_terminates_on_the_deterministic_template(self):
        def timeout(request):
            raise httpx.ConnectTimeout("no route to host")

        router = self.chain(
            responder({"error": SENTINEL_BODY}, 429),
            responder({"error": SENTINEL_BODY}, 500),
            timeout,
        )

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, FALLBACK)
        self.assertTrue(result.from_template)
        self.assertEqual(
            [attempt.code for attempt in result.attempts],
            ["provider_rate_limited", "provider_unavailable", "provider_unavailable"],
        )

    def test_a_sustained_5xx_opens_the_circuit_and_the_run_still_completes(self):
        store = MemoryCircuitStore(clock=self.clock)
        requests: list[httpx.Request] = []

        def down(request):
            requests.append(request)
            return httpx.Response(500, json={"error": SENTINEL_BODY})

        router = self.chain(down, down, down, store=store)

        results = [router.generate(f"prompt {index}", FALLBACK) for index in range(6)]

        for name in CHAIN:
            with self.subTest(provider=name):
                self.assertEqual(store.state(name).circuit_state, OPEN)
        # Three providers, three failures each before the breaker stopped them. The other
        # three runs issued nothing at all.
        self.assertEqual(len(requests), 9)
        for result in results:
            self.assertEqual(result.text, FALLBACK)
        self.assertAttempts(
            results[-1],
            [Attempt(name, SKIPPED_OPEN) for name in CHAIN],
        )


# --- building the chain ---------------------------------------------------------------------


class BuildingTheChain(unittest.TestCase):
    """Zero configured providers is a deployment, not an error."""

    def built(self, **kwargs) -> list:
        clients = build_clients(**kwargs)
        for client in clients:
            self.addCleanup(client.close)
        return clients

    def test_a_provider_without_a_key_is_skipped(self):
        clients = self.built(groq_key=SENTINEL_KEY, gemini_key=None, nvidia_key="")
        self.assertEqual([client.name for client in clients], [GROQ])

    def test_no_keys_at_all_is_an_empty_chain(self):
        self.assertEqual(self.built(), [])
        self.assertEqual(self.built(groq_key="", gemini_key="   ", nvidia_key=None), [])

    def test_an_empty_chain_still_generates(self):
        # The guarantee, reached the way a deployment reaches it rather than by a test
        # constructing `LLMRouter()` directly.
        router = LLMRouter(self.built(), clock=Clock(), sleep=never_sleeps)
        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.text, FALLBACK)
        self.assertEqual(result.provider, TEMPLATE)

    def test_the_chain_is_built_in_the_documented_order(self):
        clients = self.built(groq_key="a", gemini_key="b", nvidia_key="c")
        self.assertEqual(tuple(client.name for client in clients), CHAIN)
        self.assertEqual(CHAIN, (GROQ, GEMINI, NVIDIA))

    def test_only_narrows_the_chain_without_reordering_it(self):
        self.assertEqual(
            [c.name for c in self.built(groq_key="a", gemini_key="b", nvidia_key="c",
                                        only=["nvidia", "groq"])],
            [GROQ, NVIDIA],
        )
        self.assertEqual(
            [c.name for c in self.built(groq_key="a", gemini_key="b", only=[GEMINI])], [GEMINI]
        )
        # `only` narrows; it cannot conjure a provider whose key is missing.
        self.assertEqual(self.built(groq_key="a", only=[NVIDIA]), [])
        self.assertEqual(self.built(groq_key="a", only=[]), [])

    def test_a_blank_key_is_treated_as_absent(self):
        # `GROQ_API_KEY=` in a .env is how a key is removed. Treating it as configured buys
        # a 401 instead of a clean skip.
        for value in (None, "", "   ", "\n", 5, True, ["key"]):
            with self.subTest(value=repr(value)):
                self.assertIsNone(secret_value(value))

        self.assertEqual(secret_value("  sk-abc  "), "sk-abc")

    def test_a_secret_str_is_unwrapped_without_importing_pydantic(self):
        class Secret:
            def get_secret_value(self):
                return f"  {SENTINEL_KEY} "

        self.assertEqual(secret_value(Secret()), SENTINEL_KEY)

        class Blank:
            def get_secret_value(self):
                return "   "

        self.assertIsNone(secret_value(Blank()))

    def test_clients_from_settings_reads_the_documented_attributes(self):
        settings = types.SimpleNamespace(
            groq_api_key=SENTINEL_KEY,
            gemini_api_key=None,
            nvidia_api_key=SENTINEL_KEY,
            gemini_model="gemini-2.5-pro",
        )
        clients = clients_from_settings(settings)
        for client in clients:
            self.addCleanup(client.close)

        self.assertEqual([client.name for client in clients], [GROQ, NVIDIA])

        # And the model override lands where it was asked to.
        settings.gemini_api_key = SENTINEL_KEY
        clients = clients_from_settings(settings, only=[GEMINI])
        for client in clients:
            self.addCleanup(client.close)
        self.assertEqual(clients[0].model, "gemini-2.5-pro")

    def test_settings_with_no_llm_attributes_at_all_yields_an_empty_chain(self):
        # `getattr(..., None)` on every field, so a Settings object from before this phase
        # produces the template chain rather than an AttributeError at run time.
        class Bare:
            pass

        self.assertEqual(clients_from_settings(Bare()), [])

    def test_from_settings_with_nothing_configured_still_generates(self):
        class Bare:
            pass

        router = LLMRouter.from_settings(Bare(), clock=Clock(), sleep=never_sleeps)
        self.addCleanup(router.close)

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(router.clients, ())
        self.assertEqual(result.text, FALLBACK)
        self.assertEqual(result.provider, TEMPLATE)

    def test_from_settings_builds_the_chain_on_the_transport_it_is_given(self):
        settings = types.SimpleNamespace(
            groq_api_key=SENTINEL_KEY, gemini_api_key=None, nvidia_api_key=None,
            gemini_model=None,
        )
        router = LLMRouter.from_settings(
            settings,
            transport=httpx.MockTransport(responder(groq_body("Groq wrote this."))),
            clock=Clock(),
            sleep=never_sleeps,
        )
        self.addCleanup(router.close)

        result = router.generate(PROMPT, FALLBACK)

        self.assertEqual(result.provider, GROQ)
        self.assertEqual(result.text, "Groq wrote this.")


# --- the package surface ---------------------------------------------------------------------


class ThePackageSurface(unittest.TestCase):
    """`__init__.py` promises a lazy provider import. That promise is load-bearing."""

    def test_everything_in_all_is_reachable(self):
        import lead_engine.llm as package

        for name in package.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(package, name), f"{name} is exported but missing")

    def test_an_unknown_attribute_is_an_attribute_error(self):
        import lead_engine.llm as package

        never_exported = "OpenAIClient"
        with self.assertRaises(AttributeError):
            getattr(package, never_exported)

    def test_the_provider_clients_are_imported_lazily(self):
        # `cache` and `router` are pure Python over a `%s` string. `providers` needs httpx,
        # so a caller that only wants `prompt_hash` must not need an HTTP client installed.
        # Asserted in a fresh interpreter, because by the time this file runs everything is
        # already in sys.modules.
        script = (
            "import sys; import lead_engine.llm; "
            "print('lead_engine.llm.providers' in sys.modules, 'httpx' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False False", result.stderr)

    def test_touching_a_provider_name_pulls_the_module_in(self):
        script = (
            "import sys; import lead_engine.llm as m; m.build_clients; "
            "print('lead_engine.llm.providers' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "True", result.stderr)


if __name__ == "__main__":
    unittest.main()
