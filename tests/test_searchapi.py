"""Tests for the SearchAPI Google Maps client -- the only billed provider in this system.

Fifty searches exist, ever, and they do not renew. That single fact decides what this file
spends its assertions on, and it is not "does the parser work":

  * `BudgetTests` is the money. One HTTP call is one credit, a validation failure costs
    nothing, and a failed call is never refunded. Every one of those is a property of the
    ORDER of statements in `search_raw`, which is why they are tested with a budget that
    records rather than a budget that counts.
  * `RequestTests` asserts on the outbound request. `hl=en` in particular is a correctness
    requirement, not a preference: SearchAPI returns Google's localised display labels, the
    registry's 259 slugs were authored from the English ones, and a call that lands with
    another `hl` reports "no supply" while looking perfectly healthy. There is no error to
    see, so there has to be a test.
  * `ParityTests` is the most important class here. It feeds the committed corpus through
    the live client and through `FixtureMapsProvider` and demands equal outcomes. That
    equality is the entire justification for building Phases 2 and 3 against fixtures; if it
    breaks, every fixture-driven test in the repository is measuring something that will not
    happen when a credit is spent.

Everything runs against `httpx.MockTransport`. No test touches the network under any flag.

Two sentinels, and the difference between them is the point:

  `SENTINEL_KEY` is credential-shaped, so `errors.redact` would catch it even if this client
  were careless. It goes in the URL, because `auth_in_query` is the default.
  `SENTINEL_BODY` is ordinary prose that `redact` leaves alone -- deliberately, so that the
  no-leak guarantee has to come from this module not interpolating response bodies, and
  cannot be handed to it for free by the shared safety net.
"""

from __future__ import annotations

import json
import unittest
from functools import partial
from pathlib import Path

import httpx

from lead_engine.geo.scope import ResolvedLocation
from lead_engine.models import Lead
from lead_engine.niches import NICHE_PROFILES
from lead_engine.providers.budget import BudgetExhausted
from lead_engine.providers.errors import (
    DEFAULT_MESSAGE,
    RETRYABLE_BY_CODE,
    STATUS_BY_CODE,
    ProviderError,
    redact,
)
from lead_engine.providers.fixtures import FixtureMapsProvider
from lead_engine.providers.searchapi import (
    DEFAULT_GL,
    DEFAULT_HL,
    ENDPOINT,
    ENGINE,
    MAPS_PLACE_URL,
    MAX_RADIUS_METERS,
    MIN_RADIUS_METERS,
    RESULTS_PER_PAGE,
    UNNAMED,
    SearchApiClient,
    build_outcome,
    city_for,
    distinct_slugs,
    evidence_from_place,
    lead_from_place,
)

CORPUS = Path(__file__).parent / "fixtures" / "searchapi"

# The API key. Credential-shaped, so its absence is guaranteed twice over -- and it travels
# in the query string, which is where this client puts it by default.
SENTINEL_KEY = "sa-live-9f3c7b21e4d8a6c5f0b3"
# Planted in every error response body. Deliberately NOT credential-shaped.
SENTINEL_BODY = "the peacock ate our quota"

LOCATION = ResolvedLocation(
    label="Indiranagar, Bangalore, Karnataka, India",
    latitude=12.9784,
    longitude=77.6408,
    radius_meters=3000,
    precision="area",
    gl="in",
    source_query="Indiranagar, Bangalore, Karnataka, India",
)

PHOTOGRAPHER = NICHE_PROFILES["photographer"]
SALON = NICHE_PROFILES["salon"]
CAKE_SHOP = NICHE_PROFILES["cake_shop"]
CLOUD_KITCHEN = NICHE_PROFILES["cloud_kitchen"]
MANUFACTURER = NICHE_PROFILES["manufacturer"]

#: Every recorded page, as (niche id, page). The parity suite walks all of them.
CORPUS_PAGES = [
    ("cake_shop", 1),
    ("cloud_kitchen", 1),
    ("manufacturer", 1),
    ("photographer", 1),
    ("photographer", 2),
    ("salon", 1),
]


def payload_for(niche_id: str, page: int = 1) -> dict:
    return json.loads((CORPUS / f"{niche_id}-p{page}.json").read_text(encoding="utf-8"))


def envelope(places: list) -> dict:
    """A minimal but *documented* response envelope around some places."""
    return {
        "search_metadata": {"id": "search_test", "status": "Success"},
        "search_parameters": {"engine": ENGINE, "q": "photographer", "hl": "en", "gl": "in"},
        "search_information": {"query_displayed": "photographer"},
        "local_results": places,
    }


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


class FakeBudget:
    """A ledger that records rather than counts.

    The properties under test are about WHEN `spend` is called, not how much is left, so a
    counter would answer the wrong question: it cannot tell "charged once" from "charged
    once, refunded, charged again". Every call is kept, in order.

    `allowance` is all-or-nothing exactly as `SearchBudget.spend` is: an exhausted budget
    raises and records nothing, because a charge that raised was never made.
    """

    def __init__(self, allowance: int = 50) -> None:
        self.allowance = allowance
        self.used = 0
        self.calls: list[tuple[str, int]] = []
        # Which key each charge was booked against. A fake that swallowed the fingerprint
        # would pass whether or not the client sent one, and the client sending none is
        # precisely the defect this fake now guards: the ledger keys rows on the fingerprint,
        # so an unfingerprinted charge draws down a row nobody seeds.
        self.fingerprints: list[str] = []
        self.purposes: list[str] = []

    def spend(
        self, provider: str, n: int = 1, *, purpose: str = "discover", key_fingerprint: str = ""
    ) -> int:
        if self.used + n > self.allowance:
            raise BudgetExhausted(provider, self.allowance, self.used, n)
        self.calls.append((provider, n))
        self.fingerprints.append(key_fingerprint)
        self.purposes.append(purpose)
        self.used += n
        return self.allowance - self.used


class FakeBudgetFidelityTests(unittest.TestCase):
    """The fake must accept everything the real ledger does.

    A fake that had drifted from `SearchBudget.spend` is how the client came to charge an
    unfingerprinted row: the narrow fake accepted the narrow call, every test passed, and the
    mismatch only surfaced against a real ledger seeded the way migration 0011 exists to
    support -- where every search raised `BudgetNotConfigured` before it was issued.
    """

    def test_the_fake_accepts_every_argument_the_real_ledger_does(self):
        import inspect

        from lead_engine.providers.budget import SearchBudget

        real = inspect.signature(SearchBudget.spend)
        fake = inspect.signature(FakeBudget.spend)
        for name, parameter in real.parameters.items():
            if name == "self":
                continue
            with self.subTest(parameter=name):
                self.assertIn(
                    name,
                    fake.parameters,
                    f"FakeBudget.spend is missing {name!r}; a call the real ledger accepts "
                    f"would raise here and the drift would go unnoticed",
                )
                self.assertEqual(parameter.kind, fake.parameters[name].kind)

    def test_the_client_charges_the_fingerprint_of_the_key_it_holds(self):
        from lead_engine.providers.budget import fingerprint

        client = SearchApiClient(SENTINEL_KEY)
        self.assertEqual(client.key_fingerprint, fingerprint(SENTINEL_KEY))
        # No key, no allowance to draw down -- and crucially not some other key's row.
        self.assertEqual(SearchApiClient(None).key_fingerprint, "")


def json_responder(payload, status_code: int = 200):
    return lambda request: httpx.Response(status_code, json=payload)


def corpus_responder(niche_id: str, page: int = 1):
    return json_responder(payload_for(niche_id, page))


class SearchApiTestCase(unittest.TestCase):
    """Shared plumbing: every client is mock-transported, budgeted, and closed."""

    def make_client(self, responder, **kwargs):
        recorder = Recorder(responder)
        kwargs.setdefault("budget", FakeBudget())
        client = SearchApiClient(
            kwargs.pop("api_key", SENTINEL_KEY),
            transport=httpx.MockTransport(recorder),
            **kwargs,
        )
        self.addCleanup(client.close)
        return client, recorder, client.budget

    def assertNoSecrets(self, error: ProviderError) -> None:
        for message in (str(error), error.message, repr(error)):
            self.assertNotIn(SENTINEL_KEY, message)
            self.assertNotIn(SENTINEL_BODY, message)
            self.assertNotIn("peacock", message)
            # The endpoint is the request URL, and with `auth_in_query` the request URL IS
            # the key. Neither the host nor the parameter name may appear.
            self.assertNotIn("searchapi.io", message.lower())
            self.assertNotIn("api_key", message.lower())

    def assertProviderError(self, call, code: str) -> ProviderError:
        with self.assertRaises(ProviderError) as caught:
            call()
        error = caught.exception
        self.assertEqual(error.code, code)
        # `retryable` and `status` belong to the shared taxonomy, not to this client.
        self.assertEqual(error.retryable, RETRYABLE_BY_CODE[code])
        self.assertEqual(error.status, STATUS_BY_CODE[code])
        self.assertEqual(error.provider, "searchapi")
        self.assertNoSecrets(error)
        return error


# --- the outbound request ------------------------------------------------------------------


class RequestTests(SearchApiTestCase):
    def test_search_sends_the_documented_get_request(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER)

        request = recorder.last
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url).split("?")[0], ENDPOINT)
        params = request.url.params
        self.assertEqual(params["engine"], "google_maps")
        self.assertEqual(params["q"], "photographer")
        self.assertEqual(params["ll"], "@12.9784,77.6408,3000m")
        self.assertEqual(params["page"], "1")
        self.assertEqual(params["api_key"], SENTINEL_KEY)
        self.assertEqual(request.headers["Accept"], "application/json")

    def test_hl_and_gl_are_pinned_on_every_call(self):
        # THE test this module's docstring exists for. Google's type labels are localised;
        # `slugify_type` reconstructs `type_id` from the ENGLISH label, so a response that
        # arrived under any other `hl` misses all 259 slugs in the registry and reports an
        # empty market. Nothing raises. Nothing logs. The credit is spent either way.
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER)
        client.search_places(LOCATION, PHOTOGRAPHER, page=2)
        client.search_places(LOCATION, PHOTOGRAPHER, query_variant=3)
        client.search_places(LOCATION, PHOTOGRAPHER, limit=1)
        client.search_places(LOCATION, SALON, query_variant="unisex salon near me")
        client.search_raw(LOCATION, "photographer", page=4)

        self.assertEqual(len(recorder.requests), 6)
        for request in recorder.requests:
            with self.subTest(url=str(request.url)):
                self.assertEqual(request.url.params["hl"], "en")
                self.assertEqual(request.url.params["gl"], "in")
        self.assertEqual((DEFAULT_HL, DEFAULT_GL), ("en", "in"))

    def test_resolved_location_gl_never_overrides_the_pin(self):
        # `ResolvedLocation` carries a `gl` of its own and this client deliberately ignores
        # it: the registry's vocabulary was authored from what these queries return in an
        # Indian metro, so the country bias belongs to `niches.py`, not to whichever point a
        # resolver handed back. Named by the module docstring, which promises this exists.
        american = ResolvedLocation(
            label="Fremont, California, United States",
            latitude=37.5485,
            longitude=-121.9886,
            radius_meters=3000,
            precision="area",
            gl="us",
            source_query="Fremont, San Jose, California, United States",
        )
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(american, PHOTOGRAPHER)

        self.assertEqual(american.gl, "us")
        self.assertEqual(recorder.last.url.params["gl"], "in")

    def test_the_client_level_gl_is_the_one_an_operator_changes(self):
        # Switching country is a deliberate act that comes with re-authoring the registry,
        # so it is a constructor argument rather than a per-call one.
        client, recorder, _ = self.make_client(corpus_responder("photographer"), gl="ae", hl="en")

        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(recorder.last.url.params["gl"], "ae")

    def test_ll_carries_the_metres_suffix_that_resolved_location_omits(self):
        # The known discrepancy with `geo/scope.py`, pinned from both sides. A bare radius
        # is not a documented form; read as a zoom level it silently re-centres every area
        # search on the city core, and twenty identical results per credit is the symptom.
        params = SearchApiClient(None).request_params(LOCATION, "photographer")

        self.assertEqual(params["ll"], "@12.9784,77.6408,3000m")
        self.assertEqual(LOCATION.ll, "@12.9784,77.6408,3000")
        self.assertNotEqual(params["ll"], LOCATION.ll)
        self.assertEqual(params["ll"], f"{LOCATION.ll}m")

    def test_a_radius_outside_the_documented_range_is_clamped_not_sent(self):
        # SearchAPI rejects an out-of-range radius AFTER charging for it, so the nearest
        # legal radius is sent instead. Clamping preserves intent; raising would refuse a
        # search the vendor would have answered.
        def located(radius: int) -> ResolvedLocation:
            return ResolvedLocation(
                label="Indiranagar",
                latitude=12.9784,
                longitude=77.6408,
                radius_meters=radius,
                precision="area",
                gl="in",
                source_query="Indiranagar, Bangalore, Karnataka, India",
            )

        client = SearchApiClient(None)
        self.assertEqual(
            client.request_params(located(1), "q")["ll"],
            f"@12.9784,77.6408,{MIN_RADIUS_METERS}m",
        )
        self.assertEqual(
            client.request_params(located(99_999_999), "q")["ll"],
            f"@12.9784,77.6408,{MAX_RADIUS_METERS}m",
        )

    def test_the_key_travels_in_the_query_string_by_default(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(recorder.last.url.params["api_key"], SENTINEL_KEY)
        self.assertNotIn("authorization", recorder.last.headers)

    def test_the_header_mode_keeps_the_key_out_of_the_url_entirely(self):
        client, recorder, _ = self.make_client(
            corpus_responder("photographer"), auth_in_query=False
        )

        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(recorder.last.headers["Authorization"], f"Bearer {SENTINEL_KEY}")
        self.assertNotIn("api_key", recorder.last.url.params)
        self.assertNotIn(SENTINEL_KEY, str(recorder.last.url))

    def test_an_unconfigured_client_sends_no_key_at_all(self):
        # The API layer answers 503 before ever calling this; if it does call, SearchAPI
        # answers 401 and that maps through the shared taxonomy. Either way, no empty key.
        client, recorder, _ = self.make_client(corpus_responder("photographer"), api_key=None)

        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertIsNone(client.api_key)
        self.assertNotIn("api_key", recorder.last.url.params)
        self.assertNotIn("authorization", recorder.last.headers)

    def test_the_page_parameter_is_sent_as_asked(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer", 2))

        client.search_places(LOCATION, PHOTOGRAPHER, page=2)

        self.assertEqual(recorder.last.url.params["page"], "2")

    def test_the_query_variant_decides_what_is_asked_for(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER, query_variant=2)
        self.assertEqual(recorder.last.url.params["q"], "wedding photographer")

        client.search_places(LOCATION, PHOTOGRAPHER, query_variant="baby photoshoot blr")
        self.assertEqual(recorder.last.url.params["q"], "baby photoshoot blr")

    def test_nothing_resembling_a_page_size_is_invented(self):
        # SearchAPI documents no page-size parameter; `limit` truncates client-side. An
        # invented parameter would be ignored rather than honoured, which looks like it
        # worked right up until a page came back with twenty rows.
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER, limit=3)

        for name in ("limit", "num", "count", "per_page", "results", "n"):
            self.assertNotIn(name, recorder.last.url.params)

    def test_request_params_is_inspectable_without_a_socket(self):
        params = SearchApiClient(SENTINEL_KEY).request_params(LOCATION, " photographer ", page=3)

        self.assertEqual(
            params,
            {
                "engine": "google_maps",
                "q": "photographer",
                "ll": "@12.9784,77.6408,3000m",
                "hl": "en",
                "gl": "in",
                "page": 3,
                "api_key": SENTINEL_KEY,
            },
        )

    def test_an_empty_query_is_refused_before_anything_is_sent(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        with self.assertRaises(ValueError):
            client.search_raw(LOCATION, "   ")

        self.assertEqual(recorder.requests, [])

    def test_a_page_that_is_not_a_positive_integer_is_refused(self):
        client, recorder, _ = self.make_client(corpus_responder("photographer"))

        for page in (0, -1, True, "2", 1.5, None):
            with self.subTest(page=page):
                with self.assertRaises(ValueError):
                    client.search_raw(LOCATION, "photographer", page=page)

        self.assertEqual(recorder.requests, [])

    def test_the_client_closes_its_transport(self):
        client, _, _ = self.make_client(corpus_responder("photographer"))

        with client as entered:
            self.assertIs(entered, client)
            entered.search_places(LOCATION, PHOTOGRAPHER)

        with self.assertRaises(RuntimeError) as caught:
            client.search_places(LOCATION, PHOTOGRAPHER)
        # ProviderError IS a RuntimeError, so assertRaises alone would pass on a closed
        # client that quietly reported "provider unavailable" instead.
        self.assertNotIsInstance(caught.exception, ProviderError)


# --- the money -----------------------------------------------------------------------------


class BudgetTests(SearchApiTestCase):
    """Fifty searches, ever. Every assertion here is about the order of two statements."""

    def test_one_http_call_costs_exactly_one_credit(self):
        client, recorder, budget = self.make_client(corpus_responder("photographer"))

        client.search_places(LOCATION, PHOTOGRAPHER)
        client.search_places(LOCATION, PHOTOGRAPHER, page=2)
        client.search_raw(LOCATION, "photography studio")

        self.assertEqual(len(recorder.requests), 3)
        self.assertEqual(budget.calls, [("searchapi", 1)] * 3)

    def test_nothing_is_spent_when_the_limit_is_invalid(self):
        # `limit` is validated in `search_places` and again in `build_outcome`. The first of
        # those is the one that matters: validating only inside `build_outcome` would put
        # the check on the far side of the spend, and a mistyped limit would cost a credit.
        client, recorder, budget = self.make_client(corpus_responder("photographer"))

        for limit in (0, -1, True, "20", 2.5, None):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    client.search_places(LOCATION, PHOTOGRAPHER, limit=limit)

        self.assertEqual(budget.calls, [])
        self.assertEqual(recorder.requests, [])

    def test_nothing_is_spent_when_the_query_variant_is_invalid(self):
        client, recorder, budget = self.make_client(corpus_responder("photographer"))

        for variant, expected in ((99, ValueError), (-1, ValueError), ("  ", ValueError),
                                  (True, TypeError)):
            with self.subTest(variant=variant):
                with self.assertRaises(expected):
                    client.search_places(LOCATION, PHOTOGRAPHER, query_variant=variant)

        self.assertEqual(budget.calls, [])
        self.assertEqual(recorder.requests, [])

    def test_nothing_is_spent_when_the_page_is_invalid(self):
        # `page` is validated inside `request_params`, which `search_raw` calls BEFORE it
        # charges. That ordering is the only reason this costs nothing.
        client, recorder, budget = self.make_client(corpus_responder("photographer"))

        for page in (0, -1, True, "2"):
            with self.subTest(page=page):
                with self.assertRaises(ValueError):
                    client.search_places(LOCATION, PHOTOGRAPHER, page=page)

        self.assertEqual(budget.calls, [])
        self.assertEqual(recorder.requests, [])

    def test_budget_exhausted_propagates_untouched_and_issues_no_request(self):
        # A clean stop, not a failure. Converting it into an empty result would turn "we are
        # out of credits" into "this neighbourhood has no salons" -- a lie the operator would
        # act on by moving to the next neighbourhood.
        client, recorder, budget = self.make_client(corpus_responder("salon"))
        budget.allowance = 0

        with self.assertRaises(BudgetExhausted) as caught:
            client.search_places(LOCATION, SALON)

        self.assertEqual(recorder.requests, [])
        self.assertEqual(budget.calls, [])
        # Not wrapped, not reclassified. It is deliberately NOT a ProviderError: nothing
        # upstream went wrong and there is no HTTP status that means "we decided to stop".
        self.assertNotIsInstance(caught.exception, ProviderError)
        self.assertEqual(caught.exception.provider, "searchapi")

    def test_the_last_credit_is_spendable_and_the_next_call_stops(self):
        client, recorder, budget = self.make_client(corpus_responder("photographer"))
        budget.allowance = 1

        client.search_places(LOCATION, PHOTOGRAPHER)
        with self.assertRaises(BudgetExhausted):
            client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(budget.calls, [("searchapi", 1)])

    def test_a_failed_call_is_never_refunded(self):
        # budget.py, "SPEND FIRST, NEVER REFUND": a timeout does not prove the request was
        # not billed -- the response is what got lost, not necessarily the query. Refunding
        # here would silently overspend an allowance that cannot be topped up.
        for responder in (
            json_responder({"error": SENTINEL_BODY}, 500),
            json_responder({"error": SENTINEL_BODY}, 429),
            json_responder({"error": SENTINEL_BODY}),
            lambda request: httpx.Response(200, text="<html>maintenance</html>"),
        ):
            with self.subTest(responder=responder):
                client, recorder, budget = self.make_client(responder)

                with self.assertRaises(ProviderError):
                    client.search_places(LOCATION, PHOTOGRAPHER)

                self.assertEqual(len(recorder.requests), 1)
                self.assertEqual(budget.calls, [("searchapi", 1)])

    def test_a_credit_lost_to_a_timeout_stays_spent(self):
        def explode(request):
            raise httpx.ConnectTimeout(f"timed out connecting to {request.url}")

        client, _, budget = self.make_client(explode)

        with self.assertRaises(ProviderError):
            client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(budget.calls, [("searchapi", 1)])

    def test_the_credit_is_charged_before_the_request_leaves(self):
        # Charging afterwards would let a crash between the call and the ledger write hand
        # the operator a remaining-credit figure that is too high, which is the direction
        # that overspends.
        seen: list[int] = []
        budget = FakeBudget()

        def responder(request):
            seen.append(len(budget.calls))
            return httpx.Response(200, json=payload_for("photographer"))

        recorder = Recorder(responder)
        client = SearchApiClient(
            SENTINEL_KEY, transport=httpx.MockTransport(recorder), budget=budget
        )
        self.addCleanup(client.close)

        client.search_places(LOCATION, PHOTOGRAPHER)
        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(seen, [1, 2])

    def test_a_client_with_no_budget_is_unmetered(self):
        # `None` means unmetered and exists for tests and the fixture path. A worker that
        # constructs this without a budget is a worker that can spend fifty credits in a
        # loop -- which is why the seam is explicit rather than defaulted to a real ledger.
        client, recorder, _ = self.make_client(corpus_responder("photographer"), budget=None)

        client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertIsNone(client.budget)
        self.assertEqual(len(recorder.requests), 1)

    def test_there_is_no_way_to_reach_the_network_without_charging(self):
        # `search_places` is a wrapper over `search_raw`, and `search_raw` holds the only
        # `self._client.get` in the module. Both charge.
        client, recorder, budget = self.make_client(corpus_responder("photographer"))

        client.search_raw(LOCATION, "photographer")

        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(len(budget.calls), 1)


# --- failure taxonomy ----------------------------------------------------------------------


class StatusMappingTests(SearchApiTestCase):
    """Every upstream status lands on the agreed code, and leaks nothing on the way."""

    CASES = [
        (429, "provider_rate_limited"),
        (401, "provider_auth_failed"),
        (403, "provider_auth_failed"),
        (400, "provider_bad_response"),
        (404, "provider_bad_response"),
        (422, "provider_bad_response"),
        (500, "provider_bad_response"),
        (502, "provider_bad_response"),
        (503, "provider_bad_response"),
    ]

    def error_body(self) -> dict:
        # Shaped like a real upstream error, with the key echoed back in it -- which is what
        # SearchAPI does, and the reason `from_status` takes a status and not a response.
        return {
            "error": SENTINEL_BODY,
            "request_url": f"{ENDPOINT}?engine=google_maps&api_key={SENTINEL_KEY}",
        }

    def test_status_codes_map_to_provider_error_codes(self):
        for status, expected in self.CASES:
            with self.subTest(status=status):
                client, _, _ = self.make_client(json_responder(self.error_body(), status))
                self.assertProviderError(
                    partial(client.search_places, LOCATION, PHOTOGRAPHER), expected
                )

    def test_five_hundreds_are_a_bad_response_here_not_an_unavailable(self):
        # Pinned because it looks arbitrary and because the TinyFish client does the
        # opposite. `errors.from_status` is the shared ladder and this client uses it
        # unchanged: `provider_unavailable` is reserved for no answer at all -- connection
        # refused, DNS, timeout. Both are retryable, so nothing downstream retries
        # differently; the API status is 502 rather than 503.
        client, _, _ = self.make_client(json_responder({}, 503))

        error = self.assertProviderError(
            partial(client.search_places, LOCATION, PHOTOGRAPHER), "provider_bad_response"
        )

        self.assertEqual(error.status, 502)
        self.assertTrue(error.retryable)

    def test_rate_limiting_is_retryable_and_auth_failure_is_not(self):
        throttled, _, _ = self.make_client(json_responder({}, 429))
        error = self.assertProviderError(
            partial(throttled.search_places, LOCATION, PHOTOGRAPHER), "provider_rate_limited"
        )
        self.assertTrue(error.retryable)

        rejected, _, _ = self.make_client(json_responder({}, 401))
        error = self.assertProviderError(
            partial(rejected.search_places, LOCATION, PHOTOGRAPHER), "provider_auth_failed"
        )
        self.assertFalse(error.retryable)

    def test_a_timeout_is_unavailable_and_breaks_the_exception_chain(self):
        def explode(request):
            # httpx stringifies `.request.url` into its own exception messages, and with
            # `auth_in_query` that URL is the API key. A traceback prints the whole chain.
            raise httpx.ConnectTimeout(f"timed out connecting to {request.url}")

        client, _, _ = self.make_client(explode)

        error = self.assertProviderError(
            partial(client.search_places, LOCATION, PHOTOGRAPHER), "provider_unavailable"
        )
        # `raise ... from None` leaves __cause__ None and sets __suppress_context__; a bare
        # raise inside the except block would leave the httpx exception in __context__ and
        # every traceback formatter would print it, key included.
        self.assertTrue(error.__suppress_context__)
        self.assertIsNone(error.__cause__)

    def test_a_transport_error_is_unavailable_and_breaks_the_chain(self):
        def explode(request):
            raise httpx.ConnectError(f"connection refused for {request.url}")

        client, _, _ = self.make_client(explode)

        error = self.assertProviderError(
            partial(client.search_places, LOCATION, PHOTOGRAPHER), "provider_unavailable"
        )
        self.assertTrue(error.__suppress_context__)
        self.assertIsNone(error.__cause__)

    def test_the_body_sentinel_is_one_redact_would_not_have_caught(self):
        # Guarding the guard. If SENTINEL_BODY were credential-shaped, every leak assertion
        # in this file would be testing `errors.redact` rather than this client.
        self.assertEqual(redact(SENTINEL_BODY), SENTINEL_BODY)
        # And the key sentinel is one it would catch, which is the belt to that's braces.
        self.assertNotEqual(redact(SENTINEL_KEY), SENTINEL_KEY)


class MalformedBodyTests(SearchApiTestCase):
    """A 200 that is not a SearchAPI response raises; it never returns junk or zero."""

    def assertBadResponse(self, call) -> ProviderError:
        error = self.assertProviderError(call, "provider_bad_response")
        # The message is this module's own prose, not the generic fallback. `_safe_message`
        # swaps the default in for anything `redact` would touch, so a client that
        # interpolated a body would pass `assertNoSecrets` by having its message erased.
        self.assertNotEqual(error.message, DEFAULT_MESSAGE["provider_bad_response"])
        return error

    def search(self, client):
        return partial(client.search_places, LOCATION, PHOTOGRAPHER)

    def test_a_body_that_is_not_json_is_rejected(self):
        client, _, _ = self.make_client(
            lambda request: httpx.Response(200, text=f"<html>{SENTINEL_BODY}</html>")
        )
        self.assertBadResponse(self.search(client))

    def test_a_json_body_that_is_not_an_object_is_rejected(self):
        client, _, _ = self.make_client(json_responder([SENTINEL_BODY]))
        self.assertBadResponse(self.search(client))

    def test_a_two_hundred_carrying_an_error_key_is_rejected(self):
        client, _, _ = self.make_client(json_responder({"error": SENTINEL_BODY}))
        self.assertBadResponse(self.search(client))

    def test_local_results_that_is_not_a_list_is_rejected(self):
        payload = dict(envelope([]), local_results=SENTINEL_BODY)
        client, _, _ = self.make_client(json_responder(payload))
        self.assertBadResponse(self.search(client))

    def test_a_body_with_none_of_the_envelope_keys_is_rejected(self):
        # A proxy error page or an HTML login form parses as JSON often enough to matter.
        # Reporting it as "no results" would let an infrastructure fault look like an empty
        # neighbourhood -- and the operator would move on and never come back.
        client, _, _ = self.make_client(json_responder({"message": SENTINEL_BODY}))
        self.assertBadResponse(self.search(client))

    def test_a_place_that_is_not_an_object_is_rejected(self):
        client, _, _ = self.make_client(json_responder(envelope([SENTINEL_BODY])))
        self.assertBadResponse(self.search(client))

    def test_an_envelope_with_no_local_results_is_an_empty_search_not_an_error(self):
        # "No cloud kitchens in this pocket" is a real, billed, useful finding. Only a body
        # that is not a SearchAPI response at all is an error.
        payload = {"search_metadata": {"status": "Success"}, "search_information": {}}
        client, _, budget = self.make_client(json_responder(payload))

        outcome = client.search_places(LOCATION, CLOUD_KITCHEN)

        self.assertEqual(outcome.returned, 0)
        self.assertEqual(outcome.leads, ())
        self.assertFalse(outcome.has_more)
        self.assertEqual(budget.calls, [("searchapi", 1)])

    def test_every_message_this_module_writes_survives_redaction(self):
        # `_safe_message` silently substitutes the generic default for any message `redact`
        # would alter, so a leaky message would not leak -- it would DISAPPEAR, taking the
        # only diagnostic with it. Every message here is static prose.
        from lead_engine.providers import searchapi

        messages = [
            searchapi._bad(text).message
            for text in (
                "SearchAPI returned a place that was not an object",
                "SearchAPI returned a body that was not JSON",
                "SearchAPI returned a JSON body that was not an object",
                "SearchAPI reported an error for this search",
                "SearchAPI returned a response that was not an object",
                "SearchAPI returned local_results that was not a list",
                "SearchAPI returned a body that was not the documented envelope",
            )
        ]
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(redact(message), message)
                self.assertNotIn(message, DEFAULT_MESSAGE.values())


# --- parsing: the one function that knows a SearchAPI field name ---------------------------


class ParsingTests(SearchApiTestCase):
    def lead(self, place: dict) -> Lead:
        return lead_from_place(place, city="Bangalore")

    def test_a_place_becomes_a_lead_field_for_field(self):
        place = payload_for("photographer")["local_results"][0]

        lead = self.lead(place)

        self.assertEqual(lead.name, "Aperture & Co. Wedding Photography")
        # Google's DISPLAY label, verbatim. Storing the derived slug here would bake this
        # system's interpretation into the record and lose what the provider actually said.
        self.assertEqual(lead.category, "Wedding photographer")
        self.assertEqual(lead.raw_categories, ["Wedding photographer", "Photographer",
                                               "Videographer"])
        self.assertEqual(lead.city, "Bangalore")
        self.assertEqual(lead.latitude, 12.9718)
        self.assertEqual(lead.longitude, 77.6412)
        self.assertEqual(lead.phone, "+91 80 4123 8890")
        self.assertEqual(lead.website, "https://apertureandco.in/")
        self.assertEqual(lead.provider_id, "ChIJl0kDbmYWrjsRLxLxjEVfE5o")
        self.assertEqual(lead.source_url, f"{MAPS_PLACE_URL}ChIJl0kDbmYWrjsRLxLxjEVfE5o")

    def test_a_place_with_no_title_becomes_the_shared_sentinel(self):
        # Not dropped and not blank: the place is real, it has an address and a phone number,
        # and an unclaimed listing is exactly the kind of business worth pitching a web
        # presence to. Downstream filters compare against this string literally.
        titleless = [p for p in payload_for("salon")["local_results"] if "title" not in p]
        self.assertEqual(len(titleless), 1)

        lead = self.lead(titleless[0])

        self.assertEqual(lead.name, UNNAMED)
        self.assertEqual(lead.name, "Unnamed business")
        self.assertEqual(lead.category, "Barber shop")

    def test_the_sentinel_reaches_the_leads_a_caller_gets(self):
        client, _, _ = self.make_client(corpus_responder("salon"))

        outcome = client.search_places(LOCATION, SALON)

        self.assertIn(UNNAMED, [lead.name for lead in outcome.leads])

    def test_a_missing_website_or_phone_is_none_rather_than_empty(self):
        lead = self.lead({"title": "Lensmen Studio", "type": "Photography studio"})

        self.assertIsNone(lead.website)
        self.assertIsNone(lead.phone)

    def test_a_blank_website_or_phone_is_also_none(self):
        lead = self.lead({"title": "X", "website": "   ", "phone": ""})

        self.assertIsNone(lead.website)
        self.assertIsNone(lead.phone)

    def test_absent_coordinates_are_none_and_never_zero(self):
        # 0.0 is a real point in the Gulf of Guinea, and a lead silently parked there would
        # survive every range check in the system.
        for value in ({}, {"gps_coordinates": None}, {"gps_coordinates": "12.9,77.6"},
                      {"gps_coordinates": {}}):
            with self.subTest(place=value):
                lead = self.lead({"title": "X", **value})
                self.assertIsNone(lead.latitude)
                self.assertIsNone(lead.longitude)

    def test_string_coordinates_are_coerced(self):
        lead = self.lead(
            {"title": "X", "gps_coordinates": {"latitude": "12.9784", "longitude": "77.6408"}}
        )

        self.assertEqual((lead.latitude, lead.longitude), (12.9784, 77.6408))

    def test_a_place_with_no_id_gets_no_source_url(self):
        lead = self.lead({"title": "X", "type": "Photographer"})

        self.assertIsNone(lead.provider_id)
        self.assertIsNone(lead.source_url)

    def test_blank_and_non_string_types_are_dropped_from_raw_categories(self):
        lead = self.lead({"title": "X", "types": ["Photographer", "", None, 7, "  Videographer "]})

        self.assertEqual(lead.raw_categories, ["Photographer", "Videographer"])

    def test_a_place_that_is_not_an_object_is_a_provider_error(self):
        with self.assertRaises(ProviderError) as caught:
            lead_from_place([SENTINEL_BODY], city="Bangalore")

        self.assertEqual(caught.exception.code, "provider_bad_response")
        self.assertNoSecrets(caught.exception)

    def test_evidence_is_the_two_signals_discovery_can_know(self):
        place = payload_for("salon")["local_results"][0]

        self.assertEqual(evidence_from_place(place), {"reviews": 638, "rating": 4.4})

    def test_evidence_is_none_rather_than_zero_when_a_place_has_no_reviews(self):
        # Zero reviews and unknown reviews score differently, and a new business with no
        # reviews is a lead, not a bad one.
        self.assertEqual(evidence_from_place({}), {"reviews": None, "rating": None})
        self.assertEqual(
            evidence_from_place({"reviews": "n/a", "rating": None}),
            {"reviews": None, "rating": None},
        )

    def test_the_city_comes_from_the_source_query_not_the_resolver_label(self):
        # `label` is whatever the resolver that answered chose to call the place --
        # Nominatim returns "Indiranagar, East Zone, Bengaluru, Bangalore Urban, ..." for the
        # same point -- so a positional rule over it would put "East Zone" in a column called
        # `city` depending on which resolver won.
        self.assertEqual(city_for(LOCATION), "Bangalore")

        city_wide = ResolvedLocation(
            label="Bengaluru, Bangalore Urban, Karnataka, India",
            latitude=12.9716,
            longitude=77.5946,
            radius_meters=15000,
            precision="city",
            gl="in",
            source_query="Bangalore, Karnataka, India",
        )
        self.assertEqual(city_for(city_wide), "Bangalore")

    def test_every_lead_is_stamped_with_the_city_and_the_niche_that_kept_it(self):
        client, _, _ = self.make_client(corpus_responder("manufacturer"))

        outcome = client.search_places(LOCATION, MANUFACTURER)

        for lead in outcome.leads:
            self.assertEqual(lead.city, "Bangalore")
            self.assertEqual(lead.matched_niches, ["manufacturer"])


# --- qualification telemetry, driven by the corpus -----------------------------------------


class QualificationTests(SearchApiTestCase):
    """What one billed search bought, including what it refused and why.

    The numbers below are properties of the committed corpus, which was authored to contain
    the substitutions Google actually returns for these queries in an Indian metro. When a
    real response replaces a fixture, these are the assertions that should be re-derived
    rather than deleted.
    """

    def outcome(self, niche_id: str, page: int = 1, **kwargs):
        profile = NICHE_PROFILES[niche_id]
        client, _, _ = self.make_client(corpus_responder(niche_id, page))
        return client.search_places(LOCATION, profile, page=page, **kwargs)

    def test_a_full_page_of_photographers_splits_into_kept_and_refused(self):
        outcome = self.outcome("photographer", limit=100)

        self.assertEqual(outcome.returned, 20)
        self.assertEqual(outcome.qualified, 12)
        self.assertEqual(outcome.rejected, 8)
        self.assertEqual(outcome.name_gate_rejected, 1)
        self.assertEqual(len(outcome.leads), 12)
        self.assertEqual(outcome.niche_id, "photographer")
        self.assertEqual(outcome.query, "photographer")
        self.assertEqual(outcome.page, 1)
        self.assertEqual(outcome.location_label, LOCATION.label)

    def test_rejected_types_names_why_the_difference_exists(self):
        outcome = self.outcome("photographer", limit=100)

        self.assertEqual(
            outcome.rejected_types,
            {
                "camera_store": 2,
                "photo_lab": 2,
                "art_gallery": 1,
                "art_studio": 1,
                "electronics_store": 1,
                "picture_frame_shop": 1,
                "print_shop": 1,
                "recording_studio": 1,
                "tattoo_studio": 1,
                "video_equipment_rental_service": 1,
                "video_production_service": 1,
            },
        )
        # Biggest offender first, so a run summary that prints only the head of this dict
        # prints the thing worth looking at.
        self.assertEqual(list(outcome.rejected_types)[:2], ["camera_store", "photo_lab"])

    def test_a_qualifying_slug_on_a_rejected_place_is_not_counted(self):
        # "Camera Corner Electronics" is typed {Camera store, Electronics store,
        # Photographer}. It was refused by `camera_store`; logging `photographer` beside it
        # would blame the rule that was working and send someone to widen the include list.
        outcome = self.outcome("photographer", limit=100)

        self.assertIn("camera_store", outcome.rejected_types)
        self.assertNotIn("photographer", outcome.rejected_types)

    def test_a_neutral_slug_on_a_rejected_place_is_counted(self):
        # "Om Sai Photo Frames" is typed only {Photo lab} -- neutral, not excluded -- so it
        # carries no qualifying type and is refused. Counting the neutral slug is what turns
        # a silent miss into "photo_lab: 2, consider including it".
        outcome = self.outcome("photographer", limit=100)

        self.assertEqual(outcome.rejected_types["photo_lab"], 2)

    def test_a_repeated_head_type_is_counted_once_per_place(self):
        # Google repeats the head type in `types`. Double-counting it would inflate exactly
        # the number `rejected_types` exists to be read literally.
        payload = envelope(
            [{"title": "The Camera Shop", "type": "Camera store",
              "types": ["Camera store", "Camera store", "Camera store"]}]
        )
        client, _, _ = self.make_client(json_responder(payload))

        outcome = client.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(outcome.rejected_types, {"camera_store": 1})
        self.assertEqual(distinct_slugs(lead_from_place(payload["local_results"][0],
                                                        city="Bangalore")), ["camera_store"])

    def test_the_name_gate_is_counted_separately_from_the_taxonomy(self):
        # "Sri Venkateshwara Video Coverage" is typed {Videographer} -- a type this niche
        # includes -- and carries no photography term in its name, so `strict` refuses it.
        # The fix is a `qualification_terms` entry, not an include type, and the counters
        # have to say which. Its slug qualified, so it leaves nothing in `rejected_types`.
        outcome = self.outcome("photographer", limit=100)

        self.assertEqual(outcome.name_gate_rejected, 1)
        self.assertNotIn("videographer", outcome.rejected_types)
        self.assertEqual(sum(outcome.rejected_types.values()), 13)
        # Two rejections short of what the slug counts account for: seven type rejections
        # covering thirteen slugs, plus one name-gate rejection that left none.
        self.assertEqual(outcome.rejected, 8)

    def test_a_non_strict_profile_never_rejects_on_a_name(self):
        # `cake_shop` has no `strict` flag and no qualification terms, so the name gate does
        # not run at all. "Corner House Ice Cream" carries no cake term and still qualifies.
        outcome = self.outcome("cake_shop", limit=100)

        self.assertEqual(outcome.name_gate_rejected, 0)
        self.assertEqual(outcome.qualified, 7)
        self.assertIn("Corner House Ice Cream", [lead.name for lead in outcome.leads])

    def test_the_salon_substitutions_are_refused_on_type(self):
        outcome = self.outcome("salon", limit=100)

        self.assertEqual(outcome.returned, 12)
        self.assertEqual(outcome.qualified, 8)
        self.assertEqual(outcome.name_gate_rejected, 1)
        # A beauty supply store is retail and a different pitch; a pet groomer is here
        # because "grooming" is one of this niche's own qualification terms and would have
        # matched the name.
        self.assertIn("beauty_supply_store", outcome.rejected_types)
        self.assertIn("cosmetics_store", outcome.rejected_types)
        self.assertIn("pet_groomer", outcome.rejected_types)

    def test_an_excluded_type_beats_a_perfect_name(self):
        # `cloud_kitchen` sets `allow_name_only`, so no type evidence is required and every
        # exclusion is the only defence there is. "Cloud Kitchen Interiors" is a modular
        # kitchen showroom whose name is flawless evidence; "Hotel Empire Delivery Kitchen"
        # is a restaurant Google types as a hotel. Both must lose.
        outcome = self.outcome("cloud_kitchen", limit=100)
        kept = [lead.name for lead in outcome.leads]

        self.assertNotIn("Cloud Kitchen Interiors", kept)
        self.assertNotIn("Hotel Empire Delivery Kitchen", kept)
        self.assertIn("kitchen_furniture_store", outcome.rejected_types)
        self.assertIn("hotel", outcome.rejected_types)
        # `restaurant` qualified on the hotel's own row, so it is not blamed for it.
        self.assertNotIn("restaurant", outcome.rejected_types)

    def test_allow_name_only_admits_a_place_with_no_qualifying_type(self):
        # "Ghost Kitchen Co." is typed {Caterer} -- deliberately not an include type for this
        # niche, since owning it made every caterer look like a cloud kitchen -- and gets in
        # on its name alone.
        outcome = self.outcome("cloud_kitchen", limit=100)

        self.assertIn("Ghost Kitchen Co.", [lead.name for lead in outcome.leads])

    def test_a_cloud_kitchen_run_starves_on_the_name_gate_not_the_taxonomy(self):
        # The diagnostic this field exists for, visible in the corpus: two of the seven
        # refusals are real delivery kitchens whose Google names carry no term from the
        # list. That is a `qualification_terms` problem, and a run that reported only
        # `rejected_types` would send someone to fix the taxonomy instead.
        outcome = self.outcome("cloud_kitchen", limit=100)

        self.assertEqual(outcome.name_gate_rejected, 2)
        self.assertEqual(outcome.qualified, 5)

    def test_the_manufacturer_substitutions_are_refused_by_the_suffix_rules(self):
        # `manufacturer` also sets `allow_name_only`: "Bharat Industrial Logistics Pvt Ltd"
        # is perfect name evidence for a business that makes nothing. The suffix rules are
        # what let one niche reject all of retail without enumerating two hundred types.
        outcome = self.outcome("manufacturer", limit=100)
        kept = [lead.name for lead in outcome.leads]

        self.assertEqual(outcome.returned, 12)
        self.assertEqual(outcome.qualified, 7)
        self.assertEqual(outcome.name_gate_rejected, 1)
        self.assertNotIn("Bharat Industrial Logistics Pvt Ltd", kept)
        self.assertIn("steel_almirah_dealer", outcome.rejected_types)
        self.assertIn("steel_distributor", outcome.rejected_types)
        self.assertIn("furniture_store", outcome.rejected_types)

    def test_truncated_reports_the_qualified_leads_that_limit_dropped(self):
        outcome = self.outcome("photographer", limit=5)

        self.assertEqual(len(outcome.leads), 5)
        self.assertEqual(outcome.qualified, 12)
        self.assertEqual(outcome.truncated, 7)

    def test_truncated_is_zero_when_the_page_had_nothing_more_to_give(self):
        outcome = self.outcome("photographer", page=2, limit=RESULTS_PER_PAGE)

        self.assertEqual(outcome.truncated, 0)
        self.assertEqual(len(outcome.leads), outcome.qualified)

    def test_limit_truncates_the_leads_and_nothing_else(self):
        # A caller asking for five leads must not blind the run to the other fifteen results
        # it has already paid for. The telemetry is what the registry is judged against.
        full = self.outcome("photographer", limit=100)
        clipped = self.outcome("photographer", limit=1)

        self.assertEqual(clipped.returned, full.returned)
        self.assertEqual(clipped.qualified, full.qualified)
        self.assertEqual(clipped.rejected_types, full.rejected_types)
        self.assertEqual(clipped.name_gate_rejected, full.name_gate_rejected)
        self.assertEqual(len(clipped.leads), 1)

    def test_has_more_is_a_full_page_and_nothing_stronger(self):
        # SearchAPI pages at twenty with a soft ceiling around 100-120 per query point, so a
        # full page is the only evidence available that another one exists. It is
        # deliberately not acted on: page 2 is another billed search.
        self.assertTrue(self.outcome("photographer", limit=100).has_more)
        self.assertFalse(self.outcome("photographer", page=2, limit=100).has_more)
        self.assertFalse(self.outcome("salon", limit=100).has_more)

    def test_page_two_is_a_different_set_of_places(self):
        first = self.outcome("photographer", limit=100)
        second = self.outcome("photographer", page=2, limit=100)

        ids = {lead.provider_id for lead in first.leads}
        self.assertTrue(ids.isdisjoint({lead.provider_id for lead in second.leads}))
        self.assertEqual(second.page, 2)


# --- the test the whole fixture strategy rests on ------------------------------------------


class ParityTests(SearchApiTestCase):
    """The live client and the fixture provider must produce the same outcome.

    If this fails, developing Phase 2 and Phase 3 against recorded responses is developing
    against a fiction, and the first live run is where that would surface -- with fifty
    credits to discover it in.

    It holds by construction rather than by coincidence: `build_outcome` and `resolve_query`
    are free functions in `searchapi`, shared verbatim, and the fixture provider inherits
    the live client's exact parsing without inheriting a socket.
    """

    def both(self, niche_id: str, page: int = 1, **kwargs):
        profile = NICHE_PROFILES[niche_id]
        client, recorder, budget = self.make_client(corpus_responder(niche_id, page))
        provider = FixtureMapsProvider(CORPUS)

        live = client.search_places(LOCATION, profile, page=page, **kwargs)
        recorded = provider.search_places(LOCATION, profile, page=page, **kwargs)
        return live, recorded, recorder, budget, provider

    def test_every_recorded_page_parses_identically_through_both_paths(self):
        for niche_id, page in CORPUS_PAGES:
            with self.subTest(niche=niche_id, page=page):
                live, recorded, _, _, _ = self.both(niche_id, page, limit=100)

                self.assertEqual(live.leads, recorded.leads)
                # Not just the leads: the telemetry a run is steered by has to agree too.
                self.assertEqual(live, recorded)

    def test_parity_holds_under_a_limit(self):
        live, recorded, _, _, _ = self.both("photographer", limit=3)

        self.assertEqual(live.leads, recorded.leads)
        self.assertEqual(len(live.leads), 3)
        self.assertEqual(live.truncated, recorded.truncated)
        self.assertEqual(live.truncated, 9)

    def test_parity_holds_for_a_query_variant(self):
        live, recorded, recorder, _, provider = self.both(
            "photographer", query_variant=2, limit=100
        )

        self.assertEqual(live, recorded)
        self.assertEqual(live.query, "wedding photographer")
        # And the resolved query is what actually went on the wire, so the fixture run and
        # the billed run are answering the same question.
        self.assertEqual(recorder.last.url.params["q"], "wedding photographer")
        self.assertEqual(provider.calls[0]["query"], "wedding photographer")

    def test_the_leads_are_equal_object_for_object_not_merely_in_count(self):
        live, recorded, _, _, _ = self.both("salon", limit=100)

        self.assertEqual(len(live.leads), 8)
        for mine, theirs in zip(live.leads, recorded.leads, strict=True):
            self.assertIsInstance(mine, Lead)
            self.assertEqual(mine, theirs)
            self.assertEqual(mine.provider_id, theirs.provider_id)
            self.assertEqual(mine.raw_categories, theirs.raw_categories)
            self.assertEqual(mine.matched_niches, theirs.matched_niches)

    def test_only_the_live_path_costs_anything(self):
        # The fixture provider never touches the budget. A fixture run that decremented the
        # ledger would make the operator's remaining-credit figure lie in the expensive
        # direction, which is the whole reason the seam is on the client and not shared.
        _, _, recorder, budget, provider = self.both("cake_shop", limit=100)

        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(budget.calls, [("searchapi", 1)])
        self.assertEqual(len(provider.calls), 1)
        self.assertFalse(hasattr(provider, "budget"))

    def test_build_outcome_is_the_shared_function_and_not_a_reimplementation(self):
        # Belt and braces on the parity claim: the same payload through `build_outcome`
        # directly matches both callers, so neither path is quietly transforming anything on
        # the way in.
        payload = payload_for("cloud_kitchen")
        direct = build_outcome(
            payload, CLOUD_KITCHEN, LOCATION, limit=100, page=1, query="cloud kitchen"
        )
        live, recorded, _, _, _ = self.both("cloud_kitchen", limit=100)

        self.assertEqual(direct, live)
        self.assertEqual(direct, recorded)


if __name__ == "__main__":
    unittest.main()
