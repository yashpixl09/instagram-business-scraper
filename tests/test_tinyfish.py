"""Tests for the TinyFish Search and Fetch client.

Everything here runs against `httpx.MockTransport`, so no test touches the network, and
against a fake clock, so no test spends a real second proving a per-minute rate limit.

The assertions deliberately look at the OUTBOUND request -- method, host, params, headers,
JSON body -- and not only at what the client made of the reply. A client that parses a
fixture perfectly while sending the wrong auth header is a client that fails in production
with a green suite behind it.

Two sentinels, and the difference between them is the point:

  `SENTINEL_KEY` is credential-shaped, so `errors.redact` would catch it even if the client
  were careless. It proves the guarantee holds.
  `SENTINEL_BODY` is ordinary prose that `redact` leaves alone -- deliberately, so that the
  guarantee has to come from THIS module not interpolating response bodies, and cannot be
  provided for free by the shared safety net.
"""

from __future__ import annotations

import json
import unittest
from functools import partial

import httpx

from lead_engine.providers.errors import (
    DEFAULT_MESSAGE,
    RETRYABLE_BY_CODE,
    STATUS_BY_CODE,
    ProviderError,
    redact,
)
from lead_engine.providers.tinyfish import (
    FETCH_URL,
    SEARCH_URL,
    SearchResult,
    TinyFishClient,
    TokenBucket,
)

# The API key. Credential-shaped, so its absence is guaranteed twice over.
SENTINEL_KEY = "tf-live-key-DO-NOT-LEAK-9f3a"
# Planted in every error response body. Deliberately NOT credential-shaped: `redact` leaves
# it intact, so if it ever reaches an error message, this suite sees it.
SENTINEL_BODY = "the wombat ate our quota"
# Planted in fetched URLs.
SENTINEL_URL = "https://client-site.example/private-path"


class FakeClock:
    """A monotonic clock that only moves when something sleeps on it, or is told to.

    One object plays both roles the client needs: `clock()` reads the time and
    `clock.sleep(n)` advances it, which is exactly the relationship `time.monotonic` and
    `time.sleep` have with each other -- minus the waiting.
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Recorder:
    """A MockTransport handler that keeps every request it was asked to answer."""

    def __init__(self, responder) -> None:
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


def json_responder(payload, status_code: int = 200):
    return lambda request: httpx.Response(status_code, json=payload)


SEARCH_PAYLOAD = {
    "query": "bakery pune",
    "results": [
        {
            "position": 1,
            "site_name": "Instagram",
            "title": "Sweet Corner Bakery",
            "snippet": "Fresh bakes in Kothrud.",
            "url": "https://www.instagram.com/sweetcornerbakery/",
            "date": "2026-07-01",
        },
        {
            "position": 2,
            "site_name": "JustDial",
            "title": "Sweet Corner Bakery, Kothrud",
            "snippet": "Phone, address, timings.",
            "url": "https://www.justdial.com/Pune/Sweet-Corner-Bakery",
        },
    ],
    "total_results": 2,
    "page": 0,
}

FETCH_PAYLOAD = {
    "results": [
        {
            "url": SENTINEL_URL,
            "final_url": SENTINEL_URL,
            "title": "Sweet Corner Bakery",
            "description": "Fresh bakes",
            "language": "en",
            "format": "markdown",
            "text": "# Sweet Corner Bakery\n\nFresh bakes in Kothrud.",
        }
    ],
    "errors": [],
}


class TinyFishTestCase(unittest.TestCase):
    """Shared plumbing: every client is mock-transported, fake-clocked, and closed."""

    def make_client(self, responder, **kwargs) -> tuple[TinyFishClient, Recorder, FakeClock]:
        recorder = Recorder(responder)
        clock = FakeClock()
        client = TinyFishClient(
            SENTINEL_KEY,
            transport=httpx.MockTransport(recorder),
            clock=clock,
            sleep=clock.sleep,
            **kwargs,
        )
        self.addCleanup(client.close)
        return client, recorder, clock

    def assertNoSecrets(self, error: ProviderError) -> None:
        for message in (str(error), error.message):
            self.assertNotIn(SENTINEL_KEY, message)
            self.assertNotIn(SENTINEL_BODY, message)
            self.assertNotIn("wombat", message)
            self.assertNotIn(SENTINEL_URL, message)
            self.assertNotIn("private-path", message)
            # The provider hostnames are the request URLs; they must not appear either.
            self.assertNotIn("api.search.tinyfish.ai", message)
            self.assertNotIn("api.fetch.tinyfish.ai", message)

        # And the message must be OURS, not the generic fallback. `_safe_message` swaps in
        # the default whenever `redact` wants to touch anything, so a client that
        # interpolated a body or a URL would pass the checks above by having its message
        # silently replaced. This is the assertion that notices.
        self.assertNotEqual(error.message, DEFAULT_MESSAGE[error.code])

    def assertProviderError(self, call, code: str) -> ProviderError:
        with self.assertRaises(ProviderError) as caught:
            call()
        error = caught.exception
        self.assertEqual(error.code, code)
        # `retryable` and `status` belong to the shared taxonomy, not to this client.
        self.assertEqual(error.retryable, RETRYABLE_BY_CODE[code])
        self.assertEqual(error.status, STATUS_BY_CODE[code])
        self.assertEqual(error.provider, "tinyfish")
        self.assertNoSecrets(error)
        return error


class SearchRequestTests(TinyFishTestCase):
    def test_search_sends_the_documented_get_request(self):
        client, recorder, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        client.search("bakery pune")

        request = recorder.last
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url).split("?")[0], SEARCH_URL)
        self.assertEqual(request.url.params["query"], "bakery pune")
        self.assertEqual(request.headers["X-API-Key"], SENTINEL_KEY)
        # A raw key, not a bearer token: the docs' curl example is `-H "X-API-Key: $KEY"`.
        self.assertNotIn("authorization", request.headers)

    def test_optional_filters_are_omitted_unless_asked_for(self):
        client, recorder, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        client.search("bakery pune")

        self.assertEqual(list(recorder.last.url.params.keys()), ["query"])

    def test_optional_filters_are_sent_with_the_documented_names(self):
        client, recorder, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        client.search(
            "bakery",
            page=2,
            location="Pune, India",
            domain_type="web",
            include_domains=["instagram.com", "facebook.com"],
            exclude_domains=["pinterest.com"],
            purpose="find the shop's social profile",
        )

        params = recorder.last.url.params
        self.assertEqual(params["page"], "2")
        self.assertEqual(params["location"], "Pune, India")
        self.assertEqual(params["domain_type"], "web")
        self.assertEqual(params["include_domains"], "instagram.com,facebook.com")
        self.assertEqual(params["exclude_domains"], "pinterest.com")
        self.assertEqual(params["purpose"], "find the shop's social profile")

    def test_search_parses_results_into_dataclasses(self):
        client, _, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        results = client.search("bakery pune")

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].url, "https://www.instagram.com/sweetcornerbakery/")
        self.assertEqual(results[0].title, "Sweet Corner Bakery")
        self.assertEqual(results[0].snippet, "Fresh bakes in Kothrud.")
        self.assertEqual(results[0].site_name, "Instagram")
        self.assertEqual(results[0].position, 1)
        self.assertEqual(results[0].date, "2026-07-01")
        # `date` is optional in the schema and absent from the second row.
        self.assertIsNone(results[1].date)

    def test_limit_truncates_client_side_because_the_api_has_no_count_parameter(self):
        client, recorder, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        results = client.search("bakery pune", limit=1)

        self.assertEqual(len(results), 1)
        # Nothing resembling a count was sent. TinyFish documents no such parameter, and an
        # invented one would be ignored rather than honoured -- which would look like it
        # worked, right up until a page came back with fifty rows.
        params = recorder.last.url.params
        for name in ("limit", "num_results", "max_results", "count", "n", "per_page"):
            self.assertNotIn(name, params)

    def test_empty_query_is_rejected_before_a_request_is_made(self):
        client, recorder, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        with self.assertRaises(ValueError):
            client.search("   ")

        self.assertEqual(recorder.requests, [])


class FetchRequestTests(TinyFishTestCase):
    def test_fetch_sends_the_documented_post_request(self):
        client, recorder, _ = self.make_client(json_responder(FETCH_PAYLOAD))

        client.fetch(SENTINEL_URL)

        request = recorder.last
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), FETCH_URL)
        self.assertEqual(request.headers["X-API-Key"], SENTINEL_KEY)
        self.assertEqual(request.headers["Content-Type"], "application/json")
        self.assertEqual(
            json.loads(request.content),
            {"urls": [SENTINEL_URL], "format": "markdown"},
        )

    def test_fetch_returns_the_markdown_text(self):
        client, _, _ = self.make_client(json_responder(FETCH_PAYLOAD))

        self.assertEqual(
            client.fetch(SENTINEL_URL), "# Sweet Corner Bakery\n\nFresh bakes in Kothrud."
        )

    def test_fetch_many_batches_urls_and_drops_only_the_ones_that_failed(self):
        payload = {
            "results": [
                {"url": "https://a.example/", "text": "page a"},
                {"url": "https://b.example/", "text": "page b"},
            ],
            "errors": [{"url": "https://c.example/", "error": SENTINEL_BODY, "status": 403}],
        }
        client, recorder, _ = self.make_client(json_responder(payload))

        pages = client.fetch_many(["https://a.example/", "https://b.example/", "https://c.example/"])

        self.assertEqual(pages, {"https://a.example/": "page a", "https://b.example/": "page b"})
        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(len(json.loads(recorder.last.content)["urls"]), 3)

    def test_fetch_many_refuses_more_urls_than_one_request_accepts(self):
        client, recorder, _ = self.make_client(json_responder(FETCH_PAYLOAD))

        with self.assertRaises(ValueError):
            client.fetch_many([f"https://site{n}.example/" for n in range(11)])

        self.assertEqual(recorder.requests, [])


class StatusMappingTests(TinyFishTestCase):
    """Every provider-level HTTP status lands on the agreed code, and leaks nothing."""

    CASES = [
        (429, "provider_rate_limited"),
        (401, "provider_auth_failed"),
        (403, "provider_auth_failed"),
        (500, "provider_unavailable"),
        (502, "provider_unavailable"),
        (503, "provider_unavailable"),
        (400, "provider_bad_response"),
        (404, "provider_bad_response"),
        (422, "provider_bad_response"),
    ]

    def error_body(self) -> dict:
        return {"error": {"code": "RATE_LIMIT_EXCEEDED", "message": SENTINEL_BODY}}

    def test_search_status_codes_map_to_provider_error_codes(self):
        for status, expected in self.CASES:
            with self.subTest(status=status):
                client, _, _ = self.make_client(json_responder(self.error_body(), status))
                self.assertProviderError(partial(client.search, "bakery pune"), expected)

    def test_fetch_status_codes_map_to_provider_error_codes(self):
        for status, expected in self.CASES:
            with self.subTest(status=status):
                client, _, _ = self.make_client(json_responder(self.error_body(), status))
                self.assertProviderError(partial(client.fetch, SENTINEL_URL), expected)

    def test_the_brief_mapping_is_pinned_where_it_diverges_from_errors_from_status(self):
        # `errors.from_status` routes 5xx to provider_bad_response. This client routes it
        # to provider_unavailable, per its own brief: a TinyFish 500 is upstream being
        # down, not upstream sending unparseable JSON. Both are retryable, so nothing
        # retries differently; the API status differs, 503 rather than 502. Pinned here so
        # the divergence is a decision on the record rather than a discovery in prod.
        client, _, _ = self.make_client(json_responder({}, 503))
        error = self.assertProviderError(lambda: client.search("x"), "provider_unavailable")

        self.assertEqual(error.status, 503)
        self.assertTrue(error.retryable)

    def test_rate_limiting_is_retryable_and_auth_failure_is_not(self):
        # The two the brief names explicitly. Backing off from a 429 is the whole point;
        # retrying a rejected key just burns the next window too.
        throttled, _, _ = self.make_client(json_responder({}, 429))
        error = self.assertProviderError(partial(throttled.search, "x"), "provider_rate_limited")
        self.assertTrue(error.retryable)

        rejected, _, _ = self.make_client(json_responder({}, 401))
        error = self.assertProviderError(partial(rejected.search, "x"), "provider_auth_failed")
        self.assertFalse(error.retryable)

    def test_a_transport_timeout_is_unavailable(self):
        def explode(request):
            # httpx puts the URL into its own exception messages, and a traceback prints
            # the whole chain, so the client must break the chain with `from None`.
            raise httpx.ConnectTimeout(f"timed out connecting to {SENTINEL_URL}")

        client, _, _ = self.make_client(explode)

        with self.assertRaises(ProviderError) as caught:
            client.fetch(SENTINEL_URL)

        self.assertEqual(caught.exception.code, "provider_unavailable")
        self.assertNoSecrets(caught.exception)
        # `raise ... from None` leaves __cause__ None and sets __suppress_context__; a bare
        # `raise` inside the except block would leave the httpx exception in __context__
        # and every traceback formatter would print it.
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertIsNone(caught.exception.__cause__)

    def test_a_transport_error_is_unavailable(self):
        def explode(request):
            raise httpx.ConnectError(f"connection refused for {SENTINEL_URL}")

        client, _, _ = self.make_client(explode)

        error = self.assertProviderError(
            lambda: client.search("bakery pune"), "provider_unavailable"
        )
        self.assertTrue(error.__suppress_context__)


class LeakTests(TinyFishTestCase):
    """The message discipline, checked at the source rather than case by case."""

    def test_every_error_message_survives_redaction(self):
        # `errors._safe_message` silently substitutes the generic default for any message
        # `redact()` would alter. A message assembled from a response body would therefore
        # not leak -- it would DISAPPEAR, taking the only diagnostic with it. Every message
        # this module can produce is static prose plus at most an HTTP status, and this is
        # what keeps it that way.
        from lead_engine.providers import tinyfish

        messages = [
            tinyfish._status_error(status).message
            for status in (400, 401, 403, 404, 422, 429, 500, 502, 503)
        ]
        messages.append(tinyfish._fetch_failure([]).message)
        messages.append(tinyfish._fetch_failure([{"url": SENTINEL_URL, "status": 403}]).message)
        messages.extend(
            [
                tinyfish._bad_response(text).message
                for text in (
                    "TinyFish search response had no results list",
                    "TinyFish returned a body that was not JSON",
                    "TinyFish returned a JSON body that was not an object",
                )
            ]
        )

        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(redact(message), message)
                self.assertNotIn(message, DEFAULT_MESSAGE.values())

    def test_the_body_sentinel_is_one_redact_would_not_have_caught(self):
        # Guarding the guard. If SENTINEL_BODY were credential-shaped, every leak assertion
        # in this file would be testing errors.redact rather than this client.
        self.assertEqual(redact(SENTINEL_BODY), SENTINEL_BODY)
        # And the key sentinel is one it would catch, which is the belt to that's braces.
        self.assertNotEqual(redact(SENTINEL_KEY), SENTINEL_KEY)

    def test_a_per_url_failure_is_not_reported_as_an_auth_failure(self):
        # The `status` inside errors[] describes the SITE being scraped. Routing it through
        # the provider status table would announce "TinyFish rejected the configured key"
        # every time a shop's website answered 403, and would trip the provider circuit
        # breaker over something entirely outside TinyFish.
        payload = {
            "results": [],
            "errors": [{"url": SENTINEL_URL, "error": SENTINEL_BODY, "status": 403}],
        }
        client, _, _ = self.make_client(json_responder(payload))

        self.assertProviderError(lambda: client.fetch(SENTINEL_URL), "provider_bad_response")


class MalformedBodyTests(TinyFishTestCase):
    """A body that does not match the documented schema raises; it never returns junk."""

    def assertBadResponse(self, call) -> None:
        self.assertProviderError(call, "provider_bad_response")

    def test_search_rejects_a_body_that_is_not_json(self):
        def responder(request):
            return httpx.Response(200, text=f"<html>{SENTINEL_BODY}</html>")

        client, _, _ = self.make_client(responder)
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_a_json_body_that_is_not_an_object(self):
        client, _, _ = self.make_client(json_responder([SENTINEL_BODY]))
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_a_missing_results_list(self):
        client, _, _ = self.make_client(json_responder({"query": "x", "total_results": 0}))
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_results_that_are_not_a_list(self):
        client, _, _ = self.make_client(json_responder({"results": SENTINEL_BODY}))
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_a_result_row_that_is_not_an_object(self):
        client, _, _ = self.make_client(json_responder({"results": [SENTINEL_BODY]}))
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_a_result_row_without_a_url(self):
        payload = {"results": [{"title": "No link here", "snippet": SENTINEL_BODY}]}
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(lambda: client.search("bakery pune"))

    def test_search_rejects_a_malformed_row_even_beyond_the_limit(self):
        # Truncating first would hide a schema change behind a small `limit`.
        payload = {
            "results": [
                {"url": "https://good.example/", "title": "fine"},
                {"title": "broken"},
            ]
        }
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(lambda: client.search("bakery pune", limit=1))

    def test_fetch_rejects_a_body_missing_the_envelope(self):
        client, _, _ = self.make_client(json_responder({"pages": []}))
        self.assertBadResponse(lambda: client.fetch(SENTINEL_URL))

    def test_fetch_rejects_an_envelope_with_neither_results_nor_errors(self):
        client, _, _ = self.make_client(json_responder({"results": [], "errors": []}))
        self.assertBadResponse(lambda: client.fetch(SENTINEL_URL))

    def test_fetch_rejects_a_result_whose_text_is_not_a_string(self):
        payload = {"results": [{"url": SENTINEL_URL, "text": {"body": SENTINEL_BODY}}]}
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(lambda: client.fetch(SENTINEL_URL))

    def test_fetch_rejects_a_result_with_no_text_at_all(self):
        payload = {"results": [{"url": SENTINEL_URL, "title": "Something"}]}
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(lambda: client.fetch(SENTINEL_URL))

    def test_fetch_many_rejects_a_result_without_a_url_to_key_it_by(self):
        payload = {"results": [{"text": "orphan page"}], "errors": []}
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(lambda: client.fetch_many([SENTINEL_URL]))


class RateLimitTests(TinyFishTestCase):
    """The documented caps, enforced client-side, proved without a real second passing."""

    def test_defaults_match_the_documented_free_tier_caps(self):
        client, _, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        self.assertEqual(client.search_limiter.capacity, 30)
        self.assertEqual(client.fetch_limiter.capacity, 150)
        self.assertEqual(client.search_limiter.window_seconds, 60.0)
        self.assertEqual(client.fetch_limiter.window_seconds, 60.0)

    def test_the_thirty_first_search_in_a_minute_waits(self):
        client, recorder, clock = self.make_client(json_responder(SEARCH_PAYLOAD))

        for _ in range(30):
            client.search("bakery pune")

        # Thirty went out back to back, in the same instant, without sleeping once.
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(len(recorder.requests), 30)

        client.search("bakery pune")

        # The thirty-first could not go out until the bucket refilled a token. At 30 per
        # 60s that is one token every two seconds.
        self.assertEqual(clock.sleeps, [2.0])
        self.assertEqual(len(recorder.requests), 31)

    def test_a_minute_later_a_full_burst_goes_out_again(self):
        client, recorder, clock = self.make_client(json_responder(SEARCH_PAYLOAD))

        for _ in range(30):
            client.search("bakery pune")
        clock.advance(60.0)
        for _ in range(30):
            client.search("bakery pune")

        self.assertEqual(clock.sleeps, [])
        self.assertEqual(len(recorder.requests), 60)

    def test_the_search_and_fetch_buckets_are_independent(self):
        client, _, clock = self.make_client(json_responder(FETCH_PAYLOAD))

        for _ in range(30):
            client.search("bakery pune")
        self.assertEqual(clock.sleeps, [])

        # Search is exhausted; fetch has its own, larger bucket and must not be blocked.
        client.fetch(SENTINEL_URL)
        self.assertEqual(clock.sleeps, [])

    def test_a_batch_fetch_spends_one_token_per_url(self):
        payload = {"results": [{"url": "https://a.example/", "text": "a"}], "errors": []}
        client, _, clock = self.make_client(json_responder(payload), fetch_urls_per_minute=4)

        client.fetch_many(["https://a.example/", "https://b.example/", "https://c.example/"])
        self.assertEqual(clock.sleeps, [])

        # One token left, two more URLs wanted: the deficit is one token, and at 4 per 60s
        # a token takes 15 seconds.
        client.fetch_many(["https://a.example/", "https://b.example/"])
        self.assertEqual(clock.sleeps, [15.0])

    def test_the_limiter_runs_before_the_request_not_after(self):
        # A limiter that charged after the fact would let a burst through and only then
        # start waiting -- which is precisely the 429 it exists to avoid.
        seen: list[float] = []
        clock = FakeClock()

        def responder(request):
            seen.append(clock.now)
            return httpx.Response(200, json=SEARCH_PAYLOAD)

        client = TinyFishClient(
            SENTINEL_KEY,
            transport=httpx.MockTransport(responder),
            clock=clock,
            sleep=clock.sleep,
            searches_per_minute=1,
        )
        self.addCleanup(client.close)

        client.search("one")
        client.search("two")

        self.assertEqual(seen, [1000.0, 1060.0])


class TokenBucketTests(unittest.TestCase):
    def make_bucket(self, capacity: int, window: float = 60.0) -> tuple[TokenBucket, FakeClock]:
        clock = FakeClock()
        return TokenBucket(capacity, window, clock=clock, sleep=clock.sleep), clock

    def test_a_full_bucket_hands_out_its_capacity_without_waiting(self):
        bucket, clock = self.make_bucket(5)

        waits = [bucket.take() for _ in range(5)]

        self.assertEqual(waits, [0.0] * 5)
        self.assertEqual(clock.sleeps, [])

    def test_an_empty_bucket_waits_exactly_one_refill_interval(self):
        bucket, clock = self.make_bucket(5)
        for _ in range(5):
            bucket.take()

        self.assertEqual(bucket.take(), 12.0)
        self.assertEqual(clock.sleeps, [12.0])
        self.assertEqual(clock.now, 1012.0)

    def test_refill_is_continuous_not_a_window_boundary(self):
        bucket, clock = self.make_bucket(60)  # one token per second
        for _ in range(60):
            bucket.take()

        clock.advance(3.0)

        self.assertEqual([bucket.take() for _ in range(4)], [0.0, 0.0, 0.0, 1.0])

    def test_the_bucket_never_fills_past_capacity(self):
        bucket, clock = self.make_bucket(5)
        bucket.take()
        clock.advance(3600.0)

        self.assertEqual(bucket.available, 5.0)

    def test_a_multi_token_take_waits_for_the_whole_deficit(self):
        bucket, clock = self.make_bucket(10)  # one token per six seconds
        for _ in range(10):
            bucket.take()

        self.assertEqual(bucket.take(3), 18.0)
        self.assertEqual(clock.sleeps, [18.0])

    def test_a_take_larger_than_the_bucket_is_a_programming_error(self):
        bucket, _ = self.make_bucket(5)

        with self.assertRaises(ValueError):
            bucket.take(6)

    def test_a_clock_that_goes_backwards_does_not_mint_tokens(self):
        bucket, clock = self.make_bucket(5)
        for _ in range(5):
            bucket.take()

        clock.advance(-100.0)

        self.assertEqual(bucket.available, 0.0)

    def test_a_sleep_that_does_not_advance_time_still_terminates(self):
        # Guarding against a retry loop: the wait is computed exactly, so `take` sleeps
        # once and proceeds even if the injected sleep is a no-op.
        calls: list[float] = []
        bucket = TokenBucket(2, 60.0, clock=lambda: 0.0, sleep=calls.append)
        for _ in range(2):
            bucket.take()

        self.assertEqual(bucket.take(), 30.0)
        self.assertEqual(calls, [30.0])


class ConfigurationTests(TinyFishTestCase):
    def test_the_api_key_is_exposed_for_the_api_layer_to_duck_type_on(self):
        client, _, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        self.assertEqual(client.api_key, SENTINEL_KEY)

    def test_an_unconfigured_client_sends_no_auth_header(self):
        # The API layer answers 503 before ever calling this; if it does call, TinyFish
        # answers 401 and that maps to provider_auth_failed. Either way, no empty header.
        recorder = Recorder(json_responder(SEARCH_PAYLOAD))
        client = TinyFishClient(None, transport=httpx.MockTransport(recorder))
        self.addCleanup(client.close)

        client.search("bakery pune")

        self.assertIsNone(client.api_key)
        self.assertNotIn("x-api-key", recorder.last.headers)

    def test_endpoints_are_the_documented_hosts(self):
        self.assertEqual(SEARCH_URL, "https://api.search.tinyfish.ai")
        self.assertEqual(FETCH_URL, "https://api.fetch.tinyfish.ai")

    def test_the_client_closes_its_transport(self):
        client, _, _ = self.make_client(json_responder(SEARCH_PAYLOAD))

        with client as entered:
            self.assertIs(entered, client)
            entered.search("bakery pune")

        with self.assertRaises(RuntimeError) as caught:
            client.search("bakery pune")
        # ProviderError IS a RuntimeError, so assertRaises alone would pass on a closed
        # client that quietly reported "provider unavailable" instead.
        self.assertNotIsInstance(caught.exception, ProviderError)


if __name__ == "__main__":
    unittest.main()
