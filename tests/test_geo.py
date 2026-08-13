"""Geographic resolution: the seed, the geocoder, the cache, and the refusal to guess.

The single most important test in this file is
`FanOutTests.test_an_unresolvable_area_aborts_and_never_falls_back`. Every other test here
protects a cost or a correctness detail; that one protects a day of somebody's fieldwork.
A resolver that answers "Indiranagr" with Bangalore's city centre produces a sheet that is
indistinguishable from a correct one -- real businesses, real phone numbers, real addresses,
all in the wrong place -- and nothing downstream can detect it.

No test here touches the network. Nominatim sits behind `httpx.MockTransport` and the
assertions are on the outbound request, because the two things this project must get right
about that API -- a descriptive User-Agent and one request per second -- are properties of
what is sent, not of what comes back. The seed path asserts the transport was never called
at all, which is the only way to prove the common case is free.
"""

from __future__ import annotations

import json
import os
import unittest
import uuid
from typing import Any

import httpx
import pytest

from lead_engine.geo import resolver as geo
from lead_engine.geo.scope import (
    RADIUS_METERS,
    GeoScope,
    ResolvedLocation,
    compose,
    normalize,
    radius_for,
)
from lead_engine.providers.errors import ProviderError

try:
    import psycopg

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

# The 22 neighbourhoods the launch metro is swept by. Listed here rather than derived from
# seed.json so that deleting one from the seed fails a test instead of shrinking a run.
BANGALORE_AREAS = (
    "Indiranagar",
    "Koramangala",
    "HSR Layout",
    "Whitefield",
    "Jayanagar",
    "JP Nagar",
    "Malleshwaram",
    "Rajajinagar",
    "Basavanagudi",
    "BTM Layout",
    "Marathahalli",
    "Bellandur",
    "Sarjapur Road",
    "Hebbal",
    "Yelahanka",
    "Banashankari",
    "Electronic City",
    "MG Road",
    "Frazer Town",
    "RT Nagar",
    "Vijayanagar",
    "Kalyan Nagar",
)


# --- doubles ------------------------------------------------------------------------------


class Recorder:
    """An `httpx.MockTransport` that remembers every outbound request.

    Requests beyond the scripted responses raise rather than returning something plausible:
    an unplanned call is a test that no longer describes what the code does.
    """

    def __init__(self, *responses: httpx.Response, clock: FakeClock | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.times: list[float] = []
        self._responses = list(responses)
        self._clock = clock
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._clock is not None:
            self.times.append(self._clock())
        if not self._responses:
            raise AssertionError(f"unexpected request to {request.url.path}")
        return self._responses.pop(0)


class FakeClock:
    """A monotonic clock that only moves when something sleeps or the test says so."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeCursor:
    def __init__(self, row: tuple[Any, ...] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class FakeConnection:
    """Enough of a psycopg connection for `CachingResolver`, with the SQL kept for assertions."""

    def __init__(self, rows: dict[str, tuple[Any, ...]] | None = None) -> None:
        self.rows = dict(rows or {})
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.commits = 0

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        statement = " ".join(sql.split())
        self.statements.append((statement, tuple(params)))
        verb = statement.split()[0].upper()
        if verb == "SELECT":
            return FakeCursor(self.rows.get(params[0]))
        # Writes actually take effect. A fake that only recorded them would let a test
        # "prove" a cached row was evicted while the next read still returned it -- which is
        # the precise bug this fake is used to check for.
        if verb == "DELETE":
            self.rows.pop(params[0], None)
        elif verb == "INSERT":
            query, formatted, lat, lng, country_code, *_ = params
            self.rows[query] = (formatted, lat, lng, country_code)
        return FakeCursor()

    def commit(self) -> None:
        self.commits += 1

    def verbs(self) -> list[str]:
        return [statement.split()[0].upper() for statement, _ in self.statements]


class SpyResolver:
    """Delegates, counts, and keeps the wrapped resolver's identity."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.queries: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", type(self.inner).__name__)

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        self.queries.append((query, precision))
        return self.inner.resolve(query, precision)


class ExplodingResolver:
    """A resolver that must never be reached."""

    name = "exploding"

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        raise AssertionError(f"resolver was consulted for {query!r}")


def nominatim_result(
    *,
    lat: str = "12.9784",
    lon: str = "77.6408",
    display_name: str = "Indiranagar, Bengaluru, Karnataka, India",
    country_code: str | None = "in",
) -> list[dict[str, Any]]:
    address = {"country_code": country_code} if country_code is not None else {}
    return [{"lat": lat, "lon": lon, "display_name": display_name, "address": address}]


def nominatim(recorder: Recorder, clock: FakeClock | None = None) -> geo.NominatimResolver:
    clock = clock or FakeClock()
    return geo.NominatimResolver(
        transport=recorder.transport, clock=clock, sleeper=clock.sleep
    )


# --- scope --------------------------------------------------------------------------------


class GeoScopeTests(unittest.TestCase):
    def test_city_is_required(self):
        for missing in (None, "", "   "):
            with self.subTest(city=missing):
                with self.assertRaises(ValueError):
                    GeoScope(city=missing)

    def test_areas_must_not_be_a_bare_string(self):
        # Iterating a string would produce eleven single-character "areas" and eleven paid
        # searches for places that do not exist.
        with self.assertRaises(TypeError):
            GeoScope(city="Bangalore", areas="Indiranagar")

    def test_an_area_scope_composes_most_specific_first(self):
        scope = GeoScope(
            city="Bangalore", state="Karnataka", country="India", areas=("Indiranagar",)
        )
        self.assertEqual(
            scope.targets(), (("Indiranagar, Bangalore, Karnataka, India", "area"),)
        )

    def test_a_scope_without_areas_targets_the_city(self):
        scope = GeoScope(city="Bangalore", state="Karnataka", country="India")
        self.assertEqual(scope.targets(), (("Bangalore, Karnataka, India", "city"),))

    def test_a_component_repeated_by_the_map_is_composed_once(self):
        # Delhi is a city inside a state called Delhi.
        scope = GeoScope(city="Delhi", state="Delhi", country="India")
        self.assertEqual(scope.city_query(), "Delhi, India")

    def test_the_same_area_twice_is_searched_once(self):
        # "Indira Nagar" is not a second neighbourhood, and a duplicate area is a duplicate
        # sweep at full price. The first spelling wins so the composed query stays readable.
        scope = GeoScope(
            city="Bangalore", areas=("Indiranagar", "indiranagar ", "Indira Nagar", "Hebbal")
        )
        self.assertEqual(scope.areas, ("Indiranagar", "Hebbal"))

    def test_country_maps_to_the_searchapi_country_code(self):
        self.assertEqual(GeoScope(city="Bangalore", country="India").gl, "in")
        self.assertEqual(GeoScope(city="Bangalore", country="in").gl, "in")
        self.assertEqual(GeoScope(city="London", country="United Kingdom").gl, "gb")

    def test_an_unknown_country_yields_no_code_rather_than_a_guess(self):
        # The name still disambiguates inside the composed query; inventing `gl` would bias
        # every search toward a country nobody named.
        self.assertIsNone(GeoScope(city="Timbuktu", country="Songhai Empire").gl)
        self.assertIsNone(GeoScope(city="Bangalore").gl)

    def test_blank_disambiguators_are_dropped(self):
        scope = GeoScope(city=" Bangalore ", state="   ", country=None, areas=("", "  Hebbal"))
        self.assertEqual(scope.city, "Bangalore")
        self.assertIsNone(scope.state)
        self.assertEqual(scope.areas, ("Hebbal",))

    def test_compose_and_normalize_agree_on_spelling_noise(self):
        self.assertEqual(normalize("J.P.  Nagar"), "jpnagar")
        self.assertEqual(compose("A", None, "", "B"), "A, B")


class ResolvedLocationTests(unittest.TestCase):
    def location(self, **overrides: Any) -> ResolvedLocation:
        fields: dict[str, Any] = {
            "label": "Indiranagar, Bangalore, Karnataka, India",
            "latitude": 12.9784,
            "longitude": 77.6408,
            "radius_meters": 3000,
            "precision": "area",
            "gl": "in",
            "source_query": "Indiranagar, Bangalore, Karnataka, India",
        }
        fields.update(overrides)
        return ResolvedLocation(**fields)

    def test_ll_is_the_only_geographic_control_searchapi_offers(self):
        self.assertEqual(self.location().ll, "@12.9784,77.6408,3000")

    def test_string_coordinates_from_a_geocoder_are_coerced(self):
        location = self.location(latitude="12.9784", longitude="77.6408")
        self.assertEqual((location.latitude, location.longitude), (12.9784, 77.6408))

    def test_out_of_range_coordinates_are_rejected(self):
        # This catches a swap only when the longitude exceeds 90 -- San Francisco's -122.4
        # below. It cannot catch a Bangalore swap, since 77.6 is a legal latitude, and no
        # validation can: that is why every seeded area is bounding-box tested instead.
        with self.assertRaises(ValueError):
            self.location(latitude=-122.4194, longitude=37.7749)
        with self.assertRaises(ValueError):
            self.location(longitude=181.0)
        self.assertEqual(self.location(latitude=77.6408, longitude=12.9784).latitude, 77.6408)

    def test_unusable_coordinates_are_rejected(self):
        for bad in ("not a number", None, ""):
            with self.subTest(latitude=bad):
                with self.assertRaises(ValueError):
                    self.location(latitude=bad)

    def test_precision_must_be_one_of_the_two_that_exist(self):
        with self.assertRaises(ValueError):
            self.location(precision="street")
        with self.assertRaises(ValueError):
            radius_for("street")

    def test_gl_must_be_a_two_letter_code(self):
        with self.assertRaises(ValueError):
            self.location(gl="India")
        self.assertEqual(self.location(gl="IN").gl, "in")
        self.assertIsNone(self.location(gl=None).gl)

    def test_a_point_with_no_provenance_is_rejected(self):
        with self.assertRaises(ValueError):
            self.location(source_query="")
        with self.assertRaises(ValueError):
            self.location(radius_meters=0)


# --- seed ---------------------------------------------------------------------------------


class SeedResolverTests(unittest.TestCase):
    def setUp(self):
        self.seed = geo.SeedResolver()

    def test_the_shipped_seed_covers_every_named_bangalore_area(self):
        for area in BANGALORE_AREAS:
            with self.subTest(area=area):
                location = self.seed.resolve(f"{area}, Bangalore, Karnataka, India")
                self.assertEqual(location.precision, "area")
                self.assertEqual(location.radius_meters, 3000)
                self.assertEqual(location.gl, "in")
                # Bangalore's bounding box, loosely. A fat-fingered digit in the seed puts a
                # sweep in the Bay of Bengal, and nothing downstream would notice.
                self.assertTrue(12.7 <= location.latitude <= 13.2, location.latitude)
                self.assertTrue(77.4 <= location.longitude <= 77.8, location.longitude)

    def test_seeded_areas_are_distinct_points(self):
        points = {
            self.seed.resolve(f"{area}, Bangalore, Karnataka, India").ll
            for area in BANGALORE_AREAS
        }
        self.assertEqual(len(points), len(BANGALORE_AREAS))

    def test_a_seed_hit_never_touches_the_network(self):
        recorder = Recorder()
        chain = geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))

        location = chain.resolve("Koramangala, Bangalore, Karnataka, India")

        self.assertEqual(recorder.requests, [])
        self.assertEqual(location.label, "Koramangala, Bangalore, Karnataka, India")
        self.assertEqual(location.radius_meters, 3000)

    def test_a_city_resolves_at_the_city_radius(self):
        location = self.seed.resolve("Bangalore, Karnataka, India", "city")
        self.assertEqual((location.precision, location.radius_meters), ("city", 15000))
        self.assertEqual(location.ll, "@12.9716,77.5946,15000")

    def test_spelling_variants_resolve_to_the_same_point(self):
        canonical = self.seed.resolve("JP Nagar, Bangalore, Karnataka, India")
        for variant in (
            "J.P. Nagar, Bangalore, Karnataka, India",
            "jp nagar, bangalore, karnataka, india",
            "JP Nagar, Bengaluru, Karnataka, India",
            "JP Nagar, Bangalore, Karnataka, IN",
            "JP Nagar, Bangalore",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(self.seed.resolve(variant).ll, canonical.ll)

    def test_an_alias_that_is_a_different_word_resolves(self):
        self.assertEqual(
            self.seed.resolve("Malleswaram, Bengaluru, Karnataka, India").label,
            "Malleshwaram, Bangalore, Karnataka, India",
        )

    def test_a_tail_the_entry_is_not_is_a_miss(self):
        # There is a Bangalore in California. The disambiguators are only doing their job if
        # a wrong one refuses to match.
        with self.assertRaises(ProviderError):
            self.seed.resolve("Indiranagar, Bangalore, Ohio, United States")

    def test_asking_the_seed_for_a_precision_it_does_not_hold_is_a_miss(self):
        # An area entry must never be handed back as a 15km city centre, nor a city centre
        # as a 3km neighbourhood: the point would be real and the radius a fabrication.
        with self.assertRaises(ProviderError):
            self.seed.resolve("Indiranagar, Bangalore, Karnataka, India", "city")
        with self.assertRaises(ProviderError):
            self.seed.resolve("Bangalore, Karnataka, India", "area")

    def test_an_unseeded_area_raises_location_not_found(self):
        with self.assertRaises(ProviderError) as caught:
            self.seed.resolve("Ballygunge, Kolkata, West Bengal, India")
        self.assertEqual(caught.exception.code, "location_not_found")

    def test_two_seed_entries_answering_to_one_name_are_rejected(self):
        data = {
            "cities": [],
            "areas": [
                {"area": "Indiranagar", "city": "Bangalore", "lat": 12.9, "lng": 77.6},
                {"area": "Indira Nagar", "city": "Bangalore", "lat": 13.9, "lng": 77.9},
            ],
        }
        with self.assertRaises(ValueError):
            geo.SeedResolver(data=data)

    def test_the_seed_file_is_valid_json_with_both_sections(self):
        raw = json.loads(geo.SEED_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["areas"]), len(BANGALORE_AREAS))
        self.assertTrue(raw["cities"])


# --- nominatim ----------------------------------------------------------------------------


class NominatimResolverTests(unittest.TestCase):
    def test_the_outbound_request_identifies_this_application(self):
        recorder = Recorder(httpx.Response(200, json=nominatim_result()))

        nominatim(recorder).resolve("Ballygunge, Kolkata, West Bengal, India")

        agent = recorder.requests[0].headers["user-agent"]
        self.assertIn("lead-engine", agent)
        # A contact route is the part of the policy that is easy to drop and impossible to
        # notice: the block arrives later, on someone else's run.
        self.assertIn("@", agent)
        self.assertNotIn("python-httpx", agent.lower())

    def test_a_client_that_cannot_identify_itself_is_refused(self):
        for agent in ("", "   ", "python-httpx/0.28"):
            with self.subTest(user_agent=agent):
                with self.assertRaises(ValueError):
                    geo.NominatimResolver(user_agent=agent)

    def test_the_outbound_request_asks_nominatim_the_composed_question(self):
        recorder = Recorder(httpx.Response(200, json=nominatim_result()))

        nominatim(recorder).resolve("Ballygunge, Kolkata, West Bengal, India")

        request = recorder.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.host, "nominatim.openstreetmap.org")
        self.assertEqual(request.url.path, "/search")
        self.assertEqual(dict(request.url.params), {
            "q": "Ballygunge, Kolkata, West Bengal, India",
            "format": "jsonv2",
            "limit": "1",
            "addressdetails": "1",
        })

    def test_consecutive_requests_are_one_second_apart(self):
        clock = FakeClock()
        recorder = Recorder(
            *[httpx.Response(200, json=nominatim_result()) for _ in range(3)], clock=clock
        )
        resolver = nominatim(recorder, clock)

        for query in ("Ballygunge, Kolkata", "Salt Lake, Kolkata", "Alipore, Kolkata"):
            resolver.resolve(query)

        self.assertEqual(clock.sleeps, [1.0, 1.0])
        pairs = zip(recorder.times, recorder.times[1:], strict=False)
        gaps = [later - earlier for earlier, later in pairs]
        self.assertTrue(all(gap >= 1.0 for gap in gaps), gaps)

    def test_time_already_spent_elsewhere_counts_toward_the_second(self):
        # The gate is a rate limit, not a delay: work done between two lookups is not paid
        # for twice.
        clock = FakeClock()
        recorder = Recorder(
            httpx.Response(200, json=nominatim_result()),
            httpx.Response(200, json=nominatim_result()),
            clock=clock,
        )
        resolver = nominatim(recorder, clock)

        resolver.resolve("Ballygunge, Kolkata")
        clock.advance(5.0)
        resolver.resolve("Salt Lake, Kolkata")

        self.assertEqual(clock.sleeps, [])
        self.assertEqual(len(recorder.requests), 2)

    def test_the_country_code_becomes_gl(self):
        recorder = Recorder(httpx.Response(200, json=nominatim_result(country_code="IN")))

        location = nominatim(recorder).resolve("Ballygunge, Kolkata, West Bengal, India")

        self.assertEqual(location.gl, "in")
        self.assertEqual(location.label, "Indiranagar, Bengaluru, Karnataka, India")
        self.assertEqual(location.source_query, "Ballygunge, Kolkata, West Bengal, India")

    def test_a_city_query_gets_the_city_radius(self):
        recorder = Recorder(httpx.Response(200, json=nominatim_result()))
        location = nominatim(recorder).resolve("Kolkata, West Bengal, India", "city")
        self.assertEqual((location.precision, location.radius_meters), ("city", 15000))

    def test_an_empty_result_list_is_location_not_found(self):
        # Nominatim answers a genuine miss with 200 and `[]`, so this is what "no such place"
        # looks like on the wire.
        recorder = Recorder(httpx.Response(200, json=[]))

        with self.assertRaises(ProviderError) as caught:
            nominatim(recorder).resolve("Nowhereville, Bangalore, Karnataka, India")

        self.assertEqual(caught.exception.code, "location_not_found")
        self.assertEqual(caught.exception.status, 422)
        self.assertFalse(caught.exception.retryable)

    def test_upstream_failures_keep_their_own_classification(self):
        # A throttle or an outage must not arrive as `location_not_found`: that code aborts
        # the run permanently, and neither of these is evidence the place does not exist.
        # The codes below are `errors.from_status`'s shared ladder rather than this module's
        # opinion -- pinned here so that a change to the ladder is visible from the one
        # caller whose 422 has an operational meaning.
        for status, code, retryable in (
            (429, "provider_rate_limited", True),
            (503, "provider_bad_response", True),
            (500, "provider_bad_response", True),
            (403, "provider_auth_failed", False),
        ):
            with self.subTest(status=status):
                recorder = Recorder(httpx.Response(status, text="nope"))
                with self.assertRaises(ProviderError) as caught:
                    nominatim(recorder).resolve("Ballygunge, Kolkata")
                self.assertNotEqual(caught.exception.code, "location_not_found")
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.retryable, retryable)

    def test_a_body_that_is_not_json_is_a_bad_response(self):
        recorder = Recorder(httpx.Response(200, text="<html>maintenance</html>"))
        with self.assertRaises(ProviderError) as caught:
            nominatim(recorder).resolve("Ballygunge, Kolkata")
        self.assertEqual(caught.exception.code, "provider_bad_response")

    def test_a_result_without_usable_coordinates_is_a_bad_response(self):
        for payload in ([{"display_name": "somewhere"}], [{"lat": "x", "lon": "y"}], [42], {}):
            with self.subTest(payload=payload):
                recorder = Recorder(httpx.Response(200, json=payload))
                with self.assertRaises(ProviderError) as caught:
                    nominatim(recorder).resolve("Ballygunge, Kolkata")
                self.assertEqual(caught.exception.code, "provider_bad_response")

    def test_a_transport_failure_never_leaks_the_request_url(self):
        def explode(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        resolver = geo.NominatimResolver(transport=httpx.MockTransport(explode))
        with self.assertRaises(ProviderError) as caught:
            resolver.resolve("Ballygunge, Kolkata")

        self.assertEqual(caught.exception.code, "provider_unavailable")
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn("nominatim.openstreetmap.org", str(caught.exception))


class ChainResolverTests(unittest.TestCase):
    def test_an_unseeded_location_falls_through_to_nominatim(self):
        recorder = Recorder(
            httpx.Response(
                200,
                json=nominatim_result(
                    lat="22.5300", lon="88.3639", display_name="Ballygunge, Kolkata, India"
                ),
            )
        )
        chain = geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))

        location = chain.resolve("Ballygunge, Kolkata, West Bengal, India")

        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(
            recorder.requests[0].url.params["q"], "Ballygunge, Kolkata, West Bengal, India"
        )
        self.assertEqual(location.latitude, 22.53)
        self.assertEqual(location.radius_meters, 3000)

    def test_a_throttle_stops_the_chain_instead_of_reading_as_a_miss(self):
        recorder = Recorder(httpx.Response(429, text="slow down"))
        chain = geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))

        with self.assertRaises(ProviderError) as caught:
            chain.resolve("Ballygunge, Kolkata, West Bengal, India")

        self.assertEqual(caught.exception.code, "provider_rate_limited")
        self.assertTrue(caught.exception.retryable)

    def test_every_resolver_missing_raises_the_last_miss(self):
        recorder = Recorder(httpx.Response(200, json=[]))
        chain = geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))

        with self.assertRaises(ProviderError) as caught:
            chain.resolve("Nowhereville, Atlantis, India")

        self.assertEqual(caught.exception.code, "location_not_found")

    def test_an_empty_chain_is_refused(self):
        with self.assertRaises(ValueError):
            geo.ChainResolver()

    def test_the_chain_names_the_stack_that_produced_a_point(self):
        chain = geo.ChainResolver(geo.SeedResolver(), nominatim(Recorder()))
        self.assertEqual(chain.name, "seed+nominatim")


# --- cache --------------------------------------------------------------------------------


class CachingResolverTests(unittest.TestCase):
    QUERY = "Indiranagar, Bangalore, Karnataka, India"

    def test_a_cache_hit_consults_neither_the_seed_nor_the_network(self):
        recorder = Recorder()
        inner = SpyResolver(geo.ChainResolver(geo.SeedResolver(), nominatim(recorder)))
        connection = FakeConnection(
            {self.QUERY: ("Indiranagar, Bengaluru, Karnataka, India", 12.9784, 77.6408, "in")}
        )

        location = geo.CachingResolver(inner, connection).resolve(self.QUERY)

        self.assertEqual(inner.queries, [])
        self.assertEqual(recorder.requests, [])
        self.assertEqual(connection.verbs(), ["SELECT"])
        self.assertEqual(location.label, "Indiranagar, Bengaluru, Karnataka, India")
        self.assertEqual(location.ll, "@12.9784,77.6408,3000")
        self.assertEqual(location.source_query, self.QUERY)

    def test_the_cached_point_takes_the_radius_the_caller_asked_for(self):
        # `geo_cache` stores what the geocoder said, not what the caller wanted with it.
        cached = {"Bangalore, Karnataka, India": ("Bangalore", 12.97, 77.59, "in")}
        connection = FakeConnection(cached)
        location = geo.CachingResolver(ExplodingResolver(), connection).resolve(
            "Bangalore, Karnataka, India", "city"
        )
        self.assertEqual((location.precision, location.radius_meters), ("city", 15000))

    def test_a_miss_resolves_once_and_writes_the_row(self):
        connection = FakeConnection()
        inner = SpyResolver(geo.SeedResolver())

        geo.CachingResolver(inner, connection).resolve(self.QUERY)

        self.assertEqual(inner.queries, [(self.QUERY, "area")])
        self.assertEqual(connection.verbs(), ["SELECT", "INSERT"])
        statement, params = connection.statements[1]
        self.assertIn("ON CONFLICT (query) DO UPDATE", statement)
        label = "Indiranagar, Bangalore, Karnataka, India"
        self.assertEqual(params, (self.QUERY, label, 12.9784, 77.6408, "in", "seed"))

    def test_a_failed_resolution_is_never_cached(self):
        # A cached miss outlives the typo that caused it, and turns one bad run into every
        # subsequent run.
        connection = FakeConnection()
        resolver = geo.CachingResolver(geo.SeedResolver(), connection)

        with self.assertRaises(ProviderError):
            resolver.resolve("Nowhereville, Bangalore, Karnataka, India")

        self.assertEqual(connection.verbs(), ["SELECT"])

    def test_the_cache_never_commits_the_callers_transaction(self):
        connection = FakeConnection()
        geo.CachingResolver(geo.SeedResolver(), connection).resolve(self.QUERY)
        self.assertEqual(connection.commits, 0)


# --- fan-out ------------------------------------------------------------------------------


class CachePoisoningTests(unittest.TestCase):
    """A cache may only be authoritative about answers that were accepted.

    `CachingResolver` writes on the way out, and the country check necessarily runs after --
    it needs the scope, which the resolver does not have. So a wrong-country answer is stored
    and then rejected, and without eviction the rejection becomes permanent: every later run
    is served the same wrong point, aborts identically, and never asks the geocoder again.
    The operator sees a failure that retrying cannot clear and nothing explains.
    """

    QUERY = "Jaipur, Rajasthan, India"

    def wrong_country_resolver(self):
        class TexasResolver:
            name = "texas"

            def resolve(self, query: str, precision: str = "area"):
                # There is a Jaipur in Rajasthan and another in Texas.
                return ResolvedLocation(
                    label="Jaipur, Texas, United States",
                    latitude=29.7,
                    longitude=-95.4,
                    radius_meters=3000,
                    precision=precision,
                    gl="us",
                    source_query=query,
                )

        return TexasResolver()

    def test_a_rejected_resolution_is_evicted_so_the_next_run_can_differ(self):
        connection = FakeConnection()
        resolver = geo.CachingResolver(self.wrong_country_resolver(), connection)
        scope = GeoScope(city="Jaipur", state="Rajasthan", country="India")

        with self.assertRaises(ProviderError) as caught:
            geo.resolve_scope(scope, resolver)
        self.assertEqual(caught.exception.code, "location_not_found")

        self.assertEqual(
            connection.rows,
            {},
            "the wrong-country point is still cached, so every later run fails identically "
            "without ever re-asking the geocoder",
        )
        self.assertIn("DELETE", connection.verbs())

    def test_an_accepted_resolution_stays_cached(self):
        # The counterweight: if eviction were unconditional the cache would never hold
        # anything and every area would be geocoded on every run.
        connection = FakeConnection()
        resolver = geo.CachingResolver(geo.SeedResolver(), connection)
        scope = GeoScope(city="Bangalore", state="Karnataka", country="India")

        geo.resolve_scope(scope, resolver)

        self.assertTrue(connection.rows, "an accepted resolution should have been cached")
        self.assertNotIn("DELETE", connection.verbs())

    def test_a_resolver_without_invalidate_still_works(self):
        # `invalidate` is optional, so a plain uncached resolver must not break -- there is
        # nothing to evict when nothing was stored.
        scope = GeoScope(city="Jaipur", state="Rajasthan", country="India")
        with self.assertRaises(ProviderError):
            geo.resolve_scope(scope, self.wrong_country_resolver())


class FanOutTests(unittest.TestCase):
    def test_three_areas_fan_out_to_three_locations_at_3000m(self):
        recorder = Recorder()
        scope = GeoScope(
            city="Bangalore",
            state="Karnataka",
            country="India",
            areas=("Indiranagar", "Koramangala", "HSR Layout"),
        )

        located = geo.resolve_scope(
            scope, geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))
        )

        self.assertEqual(recorder.requests, [])
        self.assertEqual(len(located), 3)
        self.assertEqual({point.precision for point in located}, {"area"})
        self.assertEqual({point.radius_meters for point in located}, {3000})
        self.assertEqual(len({point.ll for point in located}), 3)
        self.assertEqual(
            [point.source_query for point in located],
            [
                "Indiranagar, Bangalore, Karnataka, India",
                "Koramangala, Bangalore, Karnataka, India",
                "HSR Layout, Bangalore, Karnataka, India",
            ],
        )

    def test_no_areas_gives_one_city_wide_location_at_15000m(self):
        scope = GeoScope(city="Bangalore", state="Karnataka", country="India")

        located = geo.resolve_scope(scope, geo.SeedResolver())

        self.assertEqual(len(located), 1)
        self.assertEqual(located[0].precision, "city")
        self.assertEqual(located[0].radius_meters, RADIUS_METERS["city"])
        self.assertEqual(located[0].ll, "@12.9716,77.5946,15000")

    def test_an_unresolvable_area_aborts_and_never_falls_back(self):
        """The one that matters.

        A resolver that answered this with Bangalore's centre would produce a sheet of real
        businesses in the wrong neighbourhood, and the operator would find out by walking it.
        """
        recorder = Recorder(httpx.Response(200, json=[]))
        scope = GeoScope(
            city="Bangalore",
            state="Karnataka",
            country="India",
            areas=("Indiranagar", "Nowhereville"),
        )
        resolver = geo.ChainResolver(geo.SeedResolver(), nominatim(recorder))

        with self.assertRaises(ProviderError) as caught:
            geo.resolve_scope(scope, resolver)

        self.assertEqual(caught.exception.code, "location_not_found")
        self.assertEqual(caught.exception.status, 422)
        self.assertFalse(caught.exception.retryable)
        # Both halves of the guarantee: it asked, and having been told no, it stopped --
        # rather than substituting the city centre it already had in hand from area one.
        self.assertEqual(len(recorder.requests), 1)
        self.assertEqual(
            recorder.requests[0].url.params["q"], "Nowhereville, Bangalore, Karnataka, India"
        )

    def test_the_city_centre_is_never_substituted_for_an_unknown_area(self):
        # Same guarantee stated the other way round, so that a future "helpful" fallback has
        # to break two tests and read both comments.
        seed = geo.SeedResolver()
        city = seed.resolve("Bangalore, Karnataka, India", "city")

        with self.assertRaises(ProviderError):
            seed.resolve("Nowhereville, Bangalore, Karnataka, India")

        located = geo.resolve_scope(GeoScope(city="Bangalore", state="Karnataka"), seed)
        self.assertEqual(located[0].ll, city.ll)  # only when the city is what was asked for

    def test_a_resolution_in_the_wrong_country_aborts(self):
        # There is a Jaipur in Rajasthan and a Jaipur in Texas. `country` exists to keep them
        # apart, so a geocoder that returns the wrong one is a miss, not a result.
        recorder = Recorder(
            httpx.Response(
                200,
                json=nominatim_result(
                    lat="32.8140",
                    lon="-96.9489",
                    display_name="Jaipur, Texas, United States",
                    country_code="us",
                ),
            )
        )
        scope = GeoScope(city="Jaipur", state="Rajasthan", country="India")

        with self.assertRaises(ProviderError) as caught:
            geo.resolve_scope(scope, nominatim(recorder))

        self.assertEqual(caught.exception.code, "location_not_found")
        self.assertIn("wrong country", str(caught.exception))

    def test_a_scope_with_no_declared_country_accepts_what_it_is_given(self):
        recorder = Recorder(httpx.Response(200, json=nominatim_result(country_code="us")))
        located = geo.resolve_scope(GeoScope(city="Jaipur"), nominatim(recorder))
        self.assertEqual(located[0].gl, "us")


# --- geo_cache, against the real table ------------------------------------------------------


@pytest.mark.integration
@unittest.skipUnless(
    psycopg is not None and DSN, "needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)"
)
class GeoCacheIntegrationTests(unittest.TestCase):
    """`CachingResolver` against the real `geo_cache`.

    The fake connection above proves the control flow; only this proves the SQL. A column
    name invented in good faith passes every unit test in this file and fails on the first
    real run.
    """

    QUERY = "Indiranagar, Bangalore, Karnataka, India"

    @classmethod
    def setUpClass(cls):
        # Database-scoped, so it is installed once into `public` and stays visible from every
        # scratch schema on the search_path.
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def setUp(self):
        self.schema = "test_" + uuid.uuid4().hex
        self.connection = psycopg.connect(DSN)
        self.connection.execute(f'CREATE SCHEMA "{self.schema}"')
        self.connection.execute(f'SET search_path = "{self.schema}", public')
        self.connection.commit()
        apply_migrations(self.connection)
        self.addCleanup(self.drop_schema)

    def drop_schema(self):
        self.connection.rollback()
        self.connection.execute(f'DROP SCHEMA "{self.schema}" CASCADE')
        self.connection.commit()
        self.connection.close()

    def test_a_resolution_round_trips_through_the_real_table(self):
        recorder = Recorder()
        inner = SpyResolver(geo.ChainResolver(geo.SeedResolver(), nominatim(recorder)))
        resolver = geo.CachingResolver(inner, self.connection)

        first = resolver.resolve(self.QUERY)
        row = self.connection.execute(
            "SELECT formatted, lat, lng, country_code, resolver FROM geo_cache WHERE query = %s",
            (self.QUERY,),
        ).fetchone()
        second = resolver.resolve(self.QUERY)

        expected = (first.label, first.latitude, first.longitude, "in", "seed+nominatim")
        self.assertEqual(row, expected)
        self.assertEqual(second.ll, first.ll)
        self.assertEqual(second.label, first.label)
        # One resolution, two calls: the second was served by the row.
        self.assertEqual(len(inner.queries), 1)
        self.assertEqual(recorder.requests, [])

    def test_re_resolving_updates_the_row_instead_of_colliding_with_it(self):
        # `query` is the primary key, so a plain INSERT would raise the second time a run
        # refreshed a point.
        resolver = geo.CachingResolver(geo.SeedResolver(), self.connection)
        resolver.resolve(self.QUERY)
        self.connection.execute("DELETE FROM geo_cache WHERE query = %s", (self.QUERY,))
        resolver.resolve(self.QUERY)
        resolver.resolve(self.QUERY)  # now a hit, then force the upsert path again
        self.connection.execute("UPDATE geo_cache SET lat = 0 WHERE query = %s", (self.QUERY,))
        self.connection.execute("DELETE FROM geo_cache WHERE query = %s", (self.QUERY,))
        resolver.resolve(self.QUERY)

        count = self.connection.execute("SELECT count(*) FROM geo_cache").fetchone()[0]
        self.assertEqual(count, 1)

    def test_an_unresolvable_location_leaves_the_table_empty(self):
        resolver = geo.CachingResolver(geo.SeedResolver(), self.connection)

        with self.assertRaises(ProviderError):
            resolver.resolve("Nowhereville, Bangalore, Karnataka, India")

        self.assertEqual(
            self.connection.execute("SELECT count(*) FROM geo_cache").fetchone()[0], 0
        )


if __name__ == "__main__":
    unittest.main()
