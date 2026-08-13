"""Turning a place name into a coordinate, cheaply, and never quietly wrongly.

Three resolvers, one duck-typed interface -- `resolve(query) -> ResolvedLocation` -- so they
compose in any order without knowing about each other:

    SeedResolver        hand-authored JSON. No network, no quota, no rate limit.
    NominatimResolver   OpenStreetMap, for everywhere the seed does not cover.
    CachingResolver     the `geo_cache` table, so a repeated area is geocoded once ever.
    ChainResolver       tries each in turn; a miss falls through, a failure does not.

Production wiring is `CachingResolver(ChainResolver(SeedResolver(), NominatimResolver()), conn)`.

THE RULE THAT MATTERS
---------------------
An unresolvable location raises `ProviderError("location_not_found", ...)` and aborts the run.
There is no fallback to the city centre, no nearest match, no default point. The cost of the
alternative is not an ugly log line: a sweep aimed at a city centre it was never asked for
produces a sheet of real businesses with real phone numbers, and the operator spends a day
door-knocking the wrong neighbourhood with nothing anywhere signalling that anything went
wrong. This is the same rule that rejects an unsupported niche rather than substituting a
near one, and it is the only reason a resolver may not answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import httpx

from lead_engine.providers.errors import ProviderError, from_status

from .scope import GeoScope, ResolvedLocation, compose, normalize, radius_for

SEED_PATH = Path(__file__).resolve().parent / "seed.json"

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy requires an application that identifies itself and a contact
# route, and it blocks by User-Agent when that is missing. This is a constant rather than a
# setting because a deployment that forgets to configure it does not get a warning -- it gets
# a 403 mid-run, on the box, at the moment coordinates are needed.
NOMINATIM_USER_AGENT = "lead-engine/0.1 (local business lead research; contact: dev@pixlstudio.ai)"

# One request per second, absolute, per the same policy. Hardcoded on purpose: this is a
# condition of use, not a tuning knob, and the only honest way to configure it is downward.
MIN_INTERVAL_SECONDS = 1.0

PROVIDER = "nominatim"


class Resolver(Protocol):
    """The shape every resolver here satisfies. Structural, so nothing has to inherit it."""

    name: str

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation: ...


def not_found(message: str) -> ProviderError:
    """The one refusal a resolver is allowed to make."""
    return ProviderError("location_not_found", message)


def split_query(query: str) -> list[str]:
    """"Indiranagar, Bangalore, Karnataka, India" -> its four components."""
    return [part.strip() for part in str(query).split(",") if part.strip()]


# --- seed ---------------------------------------------------------------------------------


class SeedResolver:
    """Hand-authored coordinates, matched on the normalized area and city.

    Bangalore's neighbourhoods are the common case and this makes the common case free. It
    also makes the happy path testable without a network stub anywhere near it.
    """

    name = "seed"

    def __init__(self, path: Path | None = None, *, data: Mapping[str, Any] | None = None) -> None:
        if data is None:
            data = json.loads((path or SEED_PATH).read_text(encoding="utf-8"))
        self._cities: dict[str, Mapping[str, Any]] = {}
        self._areas: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._load(data)

    def _load(self, data: Mapping[str, Any]) -> None:
        city_keys: dict[str, list[str]] = {}
        for entry in data.get("cities", ()):
            keys = _keys(entry["city"], entry.get("aliases", ()))
            city_keys[normalize(entry["city"])] = keys
            for key in keys:
                _claim(self._cities, key, entry)

        for entry in data.get("areas", ()):
            # An area inherits its city's aliases, so "Indiranagar, Bengaluru" resolves
            # without every area entry restating that Bengaluru is Bangalore.
            city = normalize(entry["city"])
            for area_key in _keys(entry["area"], entry.get("aliases", ())):
                for city_key in city_keys.get(city, [city]):
                    _claim(self._areas, (area_key, city_key), entry)

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        radius = radius_for(precision)
        parts = split_query(query)
        entry = self._lookup(parts, precision)
        if entry is None:
            raise not_found(f"No seeded location matches {query!r}.")
        return ResolvedLocation(
            label=compose(
                entry.get("area"), entry["city"], entry.get("state"), entry.get("country")
            ),
            latitude=entry["lat"],
            longitude=entry["lng"],
            radius_meters=radius,
            precision=precision,
            gl=entry.get("gl"),
            source_query=query,
        )

    def _lookup(self, parts: list[str], precision: str) -> Mapping[str, Any] | None:
        """Only the index matching the requested precision is consulted.

        Asking for a city and being handed an area centre (or the reverse) would attach the
        wrong radius to a real-looking point, which is exactly the silent wrongness this
        module exists to prevent. A precision the seed cannot serve is a miss, and a miss
        falls through to a geocoder that can.
        """
        if precision == "area":
            if len(parts) < 2:
                return None
            entry = self._areas.get((normalize(parts[0]), normalize(parts[1])))
            tail = parts[2:]
        else:
            if not parts:
                return None
            entry = self._cities.get(normalize(parts[0]))
            tail = parts[1:]

        if entry is None or not _tail_agrees(entry, tail):
            return None
        return entry


def _keys(name: str, aliases: Iterable[str]) -> list[str]:
    """Every normalized spelling that should match one entry, canonical name first."""
    keys: list[str] = []
    for value in (name, *aliases):
        key = normalize(value)
        if key and key not in keys:
            keys.append(key)
    return keys


def _claim(index: dict[Any, Mapping[str, Any]], key: Any, entry: Mapping[str, Any]) -> None:
    existing = index.get(key)
    if existing is not None and existing is not entry:
        # Two entries answering to one name is an editing mistake in seed.json, and it would
        # otherwise show up as whichever one happened to be listed last.
        raise ValueError(f"seed.json has two entries for {key!r}")
    index[key] = entry


def _tail_agrees(entry: Mapping[str, Any], tail: list[str]) -> bool:
    """Whatever follows the area and city must be something this entry actually is.

    "Indiranagar, Bangalore, Ohio" is not a seed hit. Without this check the trailing
    components would be ignored and the disambiguators -- the entire job of `state` and
    `country` -- would do nothing.
    """
    identity = (entry.get("state"), entry.get("country"), entry.get("gl"))
    allowed = {normalize(value) for value in identity if value}
    return all(normalize(part) in allowed for part in tail)


# --- nominatim ----------------------------------------------------------------------------


class NominatimResolver:
    """OpenStreetMap's geocoder, used within its published usage policy.

    Two conditions of that policy are encoded here rather than documented: a descriptive
    User-Agent with a contact route, and a hard ceiling of one request per second. Both are
    enforced by construction -- there is no way to build this object without a User-Agent and
    no way to issue two requests inside a second -- because the penalty for breaching them is
    a block on the whole project's traffic, not a slow response.

    The gate is per instance, not per process. One resolver per worker is the intended wiring
    and the only one that honours the policy; four workers each holding their own resolver
    would issue four requests a second between them. Discovery runs behind `geo_cache` and
    the seed precisely so that this path is rare, but a future multi-worker geocoding sweep
    needs a shared gate (an advisory lock or a token in Postgres), not a second instance.
    """

    name = "nominatim"

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = NOMINATIM_USER_AGENT,
        endpoint: str = NOMINATIM_ENDPOINT,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        timeout: float = 10.0,
    ) -> None:
        agent = (user_agent or "").strip()
        if not agent or agent.lower().startswith("python-httpx"):
            raise ValueError(
                "Nominatim requires a descriptive User-Agent identifying this application "
                "and a contact route; the default HTTP client string is blocked"
            )
        self._endpoint = endpoint
        self._clock = clock
        self._sleeper = sleeper
        self._last_request: float | None = None
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": agent, "Accept": "application/json"},
        )

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        radius = radius_for(precision)
        payload = self._get({"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1})

        if not isinstance(payload, list):
            raise ProviderError("provider_bad_response", "Nominatim did not return a result list.")
        if not payload:
            # Nominatim answers a genuine miss with 200 and an empty list, so this -- not an
            # HTTP status -- is what "that place does not exist" looks like.
            raise not_found(f"Nominatim has no match for {query!r}.")

        first = payload[0]
        if not isinstance(first, Mapping):
            raise ProviderError("provider_bad_response", "Nominatim returned a malformed result.")

        address = first.get("address")
        try:
            return ResolvedLocation(
                label=first.get("display_name") or query,
                latitude=first["lat"],
                longitude=first["lon"],
                radius_meters=radius,
                precision=precision,
                gl=(address or {}).get("country_code") if isinstance(address, Mapping) else None,
                source_query=query,
            )
        except (KeyError, TypeError, ValueError) as exc:
            # `str(exc)` is safe to interpolate: it is this module's own validation text, and
            # ProviderError redacts anything URL-shaped regardless.
            raise ProviderError(
                "provider_bad_response", f"Nominatim returned an unusable result: {exc}"
            ) from None

    def _get(self, params: dict[str, Any]) -> Any:
        self._throttle()
        try:
            response = self._client.get(self._endpoint, params=params)
        except httpx.HTTPError:
            # Never chain from an httpx error: it carries `.request.url`, and every provider
            # in this project authenticates by query string.
            raise ProviderError("provider_unavailable", provider=PROVIDER) from None
        finally:
            self._last_request = self._clock()

        if response.status_code != 200:
            raise from_status(response.status_code, provider=PROVIDER) from None
        try:
            return response.json()
        except ValueError:
            raise ProviderError(
                "provider_bad_response", "Nominatim returned a body that is not JSON."
            ) from None

    def _throttle(self) -> None:
        if self._last_request is None:
            return
        remaining = MIN_INTERVAL_SECONDS - (self._clock() - self._last_request)
        if remaining > 0:
            self._sleeper(remaining)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NominatimResolver:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --- cache --------------------------------------------------------------------------------


SELECT_SQL = "SELECT formatted, lat, lng, country_code FROM geo_cache WHERE query = %s"

UPSERT_SQL = """
INSERT INTO geo_cache (query, formatted, lat, lng, country_code, resolver)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (query) DO UPDATE SET
  formatted = EXCLUDED.formatted,
  lat = EXCLUDED.lat,
  lng = EXCLUDED.lng,
  country_code = EXCLUDED.country_code,
  resolver = EXCLUDED.resolver,
  resolved_at = now()
"""


class CachingResolver:
    """`geo_cache`, in front of any other resolver.

    Areas repeat on every run of a goal, and re-geocoding them buys nothing but latency, a
    rate-limit wait, and quota. The cache is keyed on the composed query string exactly as
    resolved, which is why the composition rules live in one place.

    Two deliberate non-behaviours:

    * A failed resolution is never written. Caching "this place does not exist" would turn a
      transient bad query into a permanent one, and the row would outlive the typo.
    * Nothing here commits. The caller owns its transaction; a cache write that quietly
      commits someone else's half-finished unit of work is a much worse bug than a cache
      entry lost to a rollback.
    """

    name = "cache"

    def __init__(self, inner: Resolver, connection: Any) -> None:
        self._inner = inner
        self._connection = connection

    @property
    def inner_name(self) -> str:
        """What goes in `geo_cache.resolver`: which stack produced this coordinate."""
        return getattr(self._inner, "name", type(self._inner).__name__)

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        radius = radius_for(precision)
        row = self._connection.execute(SELECT_SQL, (query,)).fetchone()
        if row is not None:
            formatted, latitude, longitude, country_code = row
            # Radius and precision are not cached: they are the caller's declaration, not a
            # property of the point, and `geo_cache` stores only what the geocoder said.
            return ResolvedLocation(
                label=formatted,
                latitude=latitude,
                longitude=longitude,
                radius_meters=radius,
                precision=precision,
                gl=country_code,
                source_query=query,
            )

        location = self._inner.resolve(query, precision)
        self._connection.execute(
            UPSERT_SQL,
            (
                query,
                location.label,
                location.latitude,
                location.longitude,
                location.gl,
                self.inner_name,
            ),
        )
        return location


# --- chain --------------------------------------------------------------------------------


class ChainResolver:
    """Try each resolver in turn; the first answer wins.

    Only `location_not_found` falls through. A rate limit or an outage is not evidence that
    the place does not exist, and swallowing it would demote a retryable failure into the one
    error code that aborts the run permanently.
    """

    def __init__(self, *resolvers: Resolver) -> None:
        if not resolvers:
            raise ValueError("ChainResolver needs at least one resolver")
        self._resolvers = resolvers

    @property
    def name(self) -> str:
        return "+".join(getattr(r, "name", type(r).__name__) for r in self._resolvers)

    def resolve(self, query: str, precision: str = "area") -> ResolvedLocation:
        # The constructor rejects an empty chain, so this is always rebound before the raise.
        miss = not_found(f"No resolver matched {query!r}.")
        for resolver in self._resolvers:
            try:
                return resolver.resolve(query, precision)
            except ProviderError as exc:
                if exc.code != "location_not_found":
                    raise
                miss = exc
        raise miss


# --- fan-out ------------------------------------------------------------------------------


def resolve_scope(scope: GeoScope, resolver: Resolver) -> tuple[ResolvedLocation, ...]:
    """One scope in, N points out -- one per area, or one city-wide when there are none.

    Raises rather than returning a partial list. Half a sweep is not a smaller sweep; it is a
    sweep with a hole in it that nothing downstream can see.
    """
    resolved: list[ResolvedLocation] = []
    for query, precision in scope.targets():
        location = resolver.resolve(query, precision)
        _verify_country(scope, location, query)
        resolved.append(location)
    return tuple(resolved)


def _verify_country(scope: GeoScope, location: ResolvedLocation, query: str) -> None:
    """A resolution in the wrong country is a miss, not a result.

    Composing the country into the query is what usually prevents this; this is the check
    that notices when it did not. There is one Jaipur in Rajasthan and another in Texas, and
    the failure mode of accepting the wrong one is indistinguishable from success until an
    operator is standing in it.
    """
    if scope.gl and location.gl and scope.gl != location.gl:
        raise not_found(
            f"{query!r} resolved to country {location.gl!r}, but the scope declares "
            f"{scope.gl!r}. Refusing to search the wrong country."
        )
