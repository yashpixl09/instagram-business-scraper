"""Client for SearchAPI's Google Maps engine -- the only *billed* provider in this system.

Written against https://www.searchapi.io/docs/google-maps, read on 2026-08-13.

    GET https://www.searchapi.io/api/v1/search
        engine=google_maps        required
        q=<what you would type into Maps>   required
        api_key=<key>             required (or `Authorization: Bearer <key>`)
        ll=@lat,lng,<radius>m     optional, the ONLY geographic control
        hl=en  gl=in  page=1      optional

Fifty searches exist, ever. They do not renew. Every constant and every early `raise` in
this module is downstream of that one fact, so the shape of the code is unusual in three
places and each is deliberate:

* Nothing here retries. A retry loop against a metered endpoint is how an allowance
  disappears while a developer watches a spinner.
* The budget is spent INSIDE this module, immediately before the HTTP call. There is no way
  for a caller to issue a billed request without charging it, because there is no method
  that reaches the network without going through `_search`.
* Every argument is validated BEFORE the credit is charged. A malformed page number must
  cost nothing.

WHAT `hl` AND `gl` ARE DOING HERE
--------------------------------
Both are pinned, on every call, and neither is read from the resolved location.

`hl=en` is a correctness requirement, not a preference. SearchAPI returns Google's *display
labels* for `type`/`types` ("Hair salon", "Cake shop"), and `niches.slugify_type` turns those
labels back into the `type_id` the registry is written against. Those labels are localised.
One call that lands with `hl=kn` returns "ಹೇರ್ ಸಲೂನ್", every slug in that response misses
every include list in the registry, and the run reports "no supply" while looking perfectly
healthy. There is no error to see. That is the failure this pin exists to prevent.

`gl=in` is pinned for the same class of reason and is a *client-level* setting, not a
per-location one. `ResolvedLocation` carries a `gl` of its own, and this module deliberately
ignores it: the registry's 259 type slugs were authored from what these queries return in an
Indian metro, so the country the search is biased toward is a property of the vocabulary in
`niches.py`, not of whichever point a resolver happened to hand back. An operator working
another country constructs the client with `gl=` set and re-authors the registry to match.
`test_resolved_location_gl_never_overrides_the_pin` pins this against a future helpful edit.

THE `ll` SUFFIX, AND A KNOWN DISCREPANCY WITH `geo/scope.py`
-----------------------------------------------------------
SearchAPI documents exactly two forms:

    @latitude,longitude,<zoom>z     e.g. @40.7009973,-73.994778,12z
    @latitude,longitude,<meters>m   e.g. @40.7009973,-73.994778,500m

`ResolvedLocation.ll` emits `@12.9784,77.6408,3000` -- correct numbers, no unit suffix, which
is not a documented form. A bare number is at best undefined and at worst read as a zoom
level, and a silently-ignored radius means every area search re-centres on the same city
core: twenty identical results per credit, with nothing anywhere signalling it.

Rather than depend on an undocumented reading, this module composes the parameter itself from
the resolved point's own fields -- the same latitude, longitude and radius, with the suffix
the vendor documents. It is not a second geocoder and not a second source of the coordinate;
it is the vendor's serialisation of a point that `lead_engine/geo` resolved. If `scope.py`
later emits the suffix itself, `_ll` becomes a one-line delegation and nothing else changes.

AUTHENTICATION, AND WHY THE KEY IS IN THE QUERY STRING BY DEFAULT
----------------------------------------------------------------
SearchAPI accepts the key either as `?api_key=` or as `Authorization: Bearer`. The header is
the safer of the two -- a key that is never in a URL cannot leak through `httpx`'s exception
strings, an access log, or a recorded fixture. The default here is nonetheless the query
parameter, because the vendor documents `api_key` as *required* and the header as an
alternative, and the first live call has to work: there are fifty credits and no budget for
a debugging loop. `auth_in_query=False` switches to the header, and both paths are tested.

The leak risk that choice accepts is closed on all three routes it could take:
  * errors -- no message here ever contains a URL, a body, or the key, and every `raise`
    from an `httpx` exception uses `from None`, since those exceptions stringify
    `.request.url` and tracebacks print the whole chain (see `errors.py`);
  * fixtures -- `fixtures.scrub_payload` strips the key out of a recorded response before it
    is written, because `search_metadata.request_url` echoes the request back verbatim;
  * logs -- nothing in this module logs.

ONE PLACE PARSES SEARCHAPI'S SHAPE
----------------------------------
`lead_from_place` is the only function in the codebase that knows a SearchAPI field name.
Everything else -- qualification, telemetry, the fixture provider -- works on `Lead`. Swapping
the provider is then a rewrite of one function rather than an archaeology exercise, and the
fixture provider gets the live client's exact parsing for free, which is the property that
makes fixture-driven development of Phase 2 and Phase 3 trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx

from lead_engine.geo.scope import ResolvedLocation
from lead_engine.models import Lead
from lead_engine.niches import (
    DISQUALIFIES,
    QUALIFIES,
    NicheProfile,
    classify_type,
    matches_niche,
    slugify_type,
)
from lead_engine.providers.errors import ProviderError, bad_response, from_status, unavailable

PROVIDER = "searchapi"

ENDPOINT = "https://www.searchapi.io/api/v1/search"
ENGINE = "google_maps"

#: Pinned on every call. See the module docstring -- `hl` is a correctness requirement.
DEFAULT_HL = "en"
DEFAULT_GL = "in"

#: SearchAPI returns 20 places per page and has no page-size parameter, so `limit` is applied
#: client-side. The soft ceiling per query point is around 100-120 results (5-6 pages).
RESULTS_PER_PAGE = 20

#: The documented bounds of the metres form of `ll`. A radius outside them is a request that
#: gets rejected after it has been charged for, so it is clamped rather than sent.
MIN_RADIUS_METERS = 62
MAX_RADIUS_METERS = 18_636_559

DEFAULT_TIMEOUT = 30.0

#: What a place with no `title` is called. Downstream filters compare against this string
#: literally, so it is a constant here rather than a literal at the one site that emits it.
UNNAMED = "Unnamed business"

#: A Google Maps deep link built from a place id. This is Google's own documented form for
#: addressing a place by id, so the link an operator clicks in the sales sheet is the exact
#: listing the row came from.
MAPS_PLACE_URL = "https://www.google.com/maps/place/?q=place_id:"

#: The envelope keys SearchAPI always returns. Seeing none of them means the body is not a
#: SearchAPI response at all, which is a different thing from a search that found nothing.
ENVELOPE_KEYS = ("search_metadata", "search_parameters", "search_information")


class Budget(Protocol):
    """The sliver of `providers.budget.SearchBudget` this module uses.

    Deliberately narrow. The real `spend()` also takes keyword-only `purpose` and
    `key_fingerprint`; this client passes neither, so that widening the ledger's key -- which
    is happening in `budget.py` right now -- cannot require a change here.
    """

    def spend(self, provider: str, n: int = 1) -> int: ...


@dataclass(frozen=True)
class SearchOutcome:
    """Everything one billed search bought, including what it refused.

    HOW TO READ THIS WHEN A RUN FINDS NOTHING
    -----------------------------------------
    `returned` is what Google sent, `qualified` is what `matches_niche` kept, and
    `rejected_types` says *why* the difference exists in the only vocabulary that can be
    acted on -- Google's own type slugs.

    That last field is the reason this dataclass is not just `list[Lead]`. All 259 slugs in
    `niches.py` were authored from expectation; not one has been observed live. A registry
    whose include lists name types Google never emits produces empty runs that are
    indistinguishable from a quiet market, and every empty run costs credits to discover. The
    counter turns that into a diff: `{"photo_lab": 6, "camera_store": 4}` after a photographer
    sweep says the exclusions are earning their keep, while `{"portrait_photographer": 11}`
    says the registry is missing a label and eleven real leads were thrown away.

    Counting rule, stated precisely because a vaguer one would be misleading:

      * only places that FAILED `matches_niche` contribute;
      * within such a place, each DISTINCT slug is counted once (Google repeats the head type
        in `types`, and double-counting it would inflate the very number being read);
      * a slug whose verdict is QUALIFIES is not counted -- a rejected place typed
        {Photographer, Camera store} was refused by `camera_store`, and logging
        `photographer` alongside it would blame the rule that was working.

    Two rejections therefore leave no slug behind, and `rejected` will exceed the sum of the
    counts. Both are visible rather than silent:

      * a place Google returned with no type at all;
      * a place whose types qualified but whose name carried no corroborating evidence for a
        `strict` profile -- counted in `name_gate_rejected`, because a niche starving on the
        name gate is a `qualification_terms` problem and needs a different fix from a
        taxonomy problem.
    """

    niche_id: str
    query: str
    page: int
    location_label: str
    leads: tuple[Lead, ...] = ()
    rejected_types: dict[str, int] = field(default_factory=dict)
    returned: int = 0
    qualified: int = 0
    name_gate_rejected: int = 0
    has_more: bool = False

    @property
    def rejected(self) -> int:
        """Places Google returned that this niche refused."""
        return self.returned - self.qualified

    @property
    def truncated(self) -> int:
        """Qualified places dropped by `limit`. Non-zero means the page had more to give."""
        return self.qualified - len(self.leads)


# --- the SearchAPI shape lives here and nowhere else ------------------------------------


def lead_from_place(place: Any, *, city: str) -> Lead:
    """Convert one raw `local_results[]` entry into a `Lead`.

    THE quarantine function: the only place in the codebase that names a SearchAPI field.

    Every field is defensive, because a raw feed is not a schema. A missing `title` becomes
    the `UNNAMED` sentinel rather than an empty string or a dropped row -- the place is real,
    it has an address and often a phone number, and it is exactly the kind of listing that is
    worth pitching a web presence to. Downstream filters compare against that sentinel
    literally, which is why it is one shared constant.

    Coordinates are `None` rather than 0.0 when absent, since 0.0 is a real point in the Gulf
    of Guinea and a lead silently parked there would survive every range check in the system.
    """
    if not isinstance(place, dict):
        raise _bad("SearchAPI returned a place that was not an object")

    title = _text(place.get("title"))
    latitude, longitude = _coordinates(place.get("gps_coordinates"))
    place_id = _text(place.get("place_id"))

    return Lead(
        name=title or UNNAMED,
        # Google's display label, stored verbatim. `niches.slugify_type` derives the slug at
        # the point of comparison; storing the derived form here would bake this system's
        # interpretation into the record and lose what the provider actually said.
        category=_text(place.get("type")),
        address=_text(place.get("address")),
        city=city,
        latitude=latitude,
        longitude=longitude,
        phone=_text(place.get("phone")) or None,
        website=_text(place.get("website")) or None,
        source_url=f"{MAPS_PLACE_URL}{place_id}" if place_id else None,
        raw_categories=[t for t in (_text(v) for v in _sequence(place.get("types"))) if t],
        provider_id=place_id or None,
    )


def evidence_from_place(place: Any) -> dict[str, Any]:
    """The audience signals a place carries, in `models.Evidence`'s vocabulary.

    Returned as a plain dict rather than an `Evidence`, because discovery only ever knows two
    of its fields and the Instagram pass supplies the rest; building the frozen object here
    would force every caller to unpack and rebuild it. Also part of the quarantine.

    `popular_times` is deliberately not reduced here. `scoring.audience_index` wants a mean
    busyness and the raw block is a nested day/hour structure; folding it is a scoring
    decision, so this returns only what a single field maps onto directly.
    """
    if not isinstance(place, dict):
        return {"reviews": None, "rating": None}
    return {"reviews": _int(place.get("reviews")), "rating": _float(place.get("rating"))}


def city_for(location: ResolvedLocation) -> str:
    """The city name to stamp on every lead from this point.

    Taken from `source_query` -- what the operator's `GeoScope` composed -- and not from
    `label`, which is whatever the resolver that answered chose to call the place. `SeedResolver`
    returns a tidy "Indiranagar, Bangalore, Karnataka, India"; `NominatimResolver` returns
    OpenStreetMap's `display_name`, which for the same point is "Indiranagar, East Zone,
    Bengaluru, Bangalore Urban, Karnataka, 560038, India". A positional rule over `label`
    would put "East Zone" in a column called `city` depending on which resolver answered.

    `GeoScope` composes most-specific-first and `precision` says how deep that goes:
    `area_query()` is (area, city, state, country) and `city_query()` is (city, state,
    country). So the city sits at index 1 for an area search and index 0 for a city one.
    """
    parts = [part.strip() for part in str(location.source_query).split(",") if part.strip()]
    index = 1 if location.precision == "area" else 0
    if len(parts) > index:
        return parts[index]
    # A source_query with no city component should be impossible -- GeoScope requires a city
    # -- but a hand-built ResolvedLocation can do it, and a blank city column is worse than a
    # slightly redundant one.
    return parts[0] if parts else str(location.label)


# --- qualification and telemetry ---------------------------------------------------------


def distinct_slugs(lead: Lead) -> list[str]:
    """Every Google type slug this lead carries, de-duplicated, in the order first seen.

    Google repeats the head type: a place with `type` "Cake shop" and `types` ["Cake shop",
    "Bakery"] is one cake shop, not two. Counting the repeat would inflate exactly the number
    `rejected_types` exists to be read literally.
    """
    slugs = (slugify_type(lead.category), *(slugify_type(t) for t in lead.raw_categories))
    return list(dict.fromkeys(s for s in slugs if s))


def build_outcome(
    payload: Any,
    profile: NicheProfile,
    location: ResolvedLocation,
    *,
    limit: int,
    page: int,
    query: str,
) -> SearchOutcome:
    """Turn a raw SearchAPI response into a `SearchOutcome`. No network, no budget, no clock.

    Shared verbatim by the live client and by `fixtures.FixtureMapsProvider`, which is what
    makes a recorded fixture behave exactly like the call it was recorded from. Keeping it a
    free function rather than a method is what allows that sharing without the fixture
    provider inheriting a socket.
    """
    limit = _limit(limit)
    places = _places(payload)
    city = city_for(location)

    leads: list[Lead] = []
    rejected_types: dict[str, int] = {}
    qualified = 0
    name_gate_rejected = 0

    for place in places:
        lead = lead_from_place(place, city=city)
        if matches_niche(profile, lead):
            qualified += 1
            # `limit` truncates the kept leads and nothing else: the telemetry below still
            # sees the whole page, because what Google returned is what the registry has to
            # be judged against, and a caller asking for 5 leads must not blind the run to
            # the other 15 results it has already paid for.
            if len(leads) < limit:
                leads.append(replace(lead, matched_niches=[profile.id]))
            continue

        slugs = distinct_slugs(lead)
        verdicts = [classify_type(profile, slug) for slug in slugs]
        if QUALIFIES in verdicts and DISQUALIFIES not in verdicts:
            # The types were fine; `strict` refused the name. A taxonomy counter must not
            # absorb this -- the fix is a qualification term, not an include type.
            name_gate_rejected += 1
        for slug, verdict in zip(slugs, verdicts, strict=True):
            if verdict != QUALIFIES:
                rejected_types[slug] = rejected_types.get(slug, 0) + 1

    return SearchOutcome(
        niche_id=profile.id,
        query=query,
        page=page,
        location_label=location.label,
        leads=tuple(leads),
        # Biggest offender first, so a run summary that prints only the head of this dict
        # prints the thing worth looking at. Dict equality ignores order, so tests are
        # unaffected by the sort.
        rejected_types=dict(sorted(rejected_types.items(), key=lambda kv: (-kv[1], kv[0]))),
        returned=len(places),
        qualified=qualified,
        name_gate_rejected=name_gate_rejected,
        # A hint, not a promise. SearchAPI pages at 20 with a soft ceiling around 100-120 per
        # query point, so a full page is the only evidence available that another one exists.
        # It is deliberately not acted on by default: page 2 is another billed search.
        has_more=len(places) >= RESULTS_PER_PAGE,
    )


# --- the client ---------------------------------------------------------------------------


class SearchApiClient:
    """Google Maps place search, metered.

    `api_key` is a plain attribute because the API layer duck-types on it to answer 503 when
    the provider is unconfigured. The client never asserts on it: a missing key must surface
    as the 401 SearchAPI actually returns, mapped through the shared taxonomy, rather than as
    a local exception that pretends to know what the vendor would have said.

    `transport` is the seam that keeps tests off the network -- pass an `httpx.MockTransport`.
    `budget` is the seam that keeps a run inside its allowance; `None` means *unmetered* and
    exists for tests and for the fixture path. A worker that constructs this without a budget
    is a worker that can spend fifty credits in a loop.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        transport: httpx.BaseTransport | None = None,
        budget: Budget | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        endpoint: str = ENDPOINT,
        hl: str = DEFAULT_HL,
        gl: str = DEFAULT_GL,
        auth_in_query: bool = True,
    ) -> None:
        self.api_key = api_key
        self.budget = budget
        self.endpoint = endpoint
        self.hl = hl
        self.gl = gl
        self.auth_in_query = auth_in_query

        headers = {"Accept": "application/json"}
        if api_key and not auth_in_query:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(transport=transport, timeout=timeout, headers=headers)

    # -- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SearchApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the interface the pipeline uses ---------------------------------------------------

    def search_places(
        self,
        location: ResolvedLocation,
        profile: NicheProfile,
        limit: int = RESULTS_PER_PAGE,
        page: int = 1,
        query_variant: int | str | None = None,
    ) -> SearchOutcome:
        """One billed Google Maps search, qualified against `profile`.

        Costs exactly one credit, charged before the request leaves. `BudgetExhausted`
        propagates untouched: it is a clean stop, not a failure, and converting it into an
        empty result here would turn "we are out of credits" into "this neighbourhood has no
        salons" -- a lie the operator would act on.

        `query_variant` selects what to ask Google:

            None  the profile's first query -- the breadth-first default, and the only one a
                  sweep should use without a reason. Five variants against thirteen tiles for
                  three pages is 195 searches: the whole allowance, four times over, for one
                  niche in one neighbourhood.
            int   an index into `profile.queries`, for a deliberate deeper sweep.
            str   sent verbatim, for an operator testing a hunch by hand.

        `limit` caps the leads returned. SearchAPI has no page-size parameter, so it truncates
        what has already been paid for rather than asking for less; `SearchOutcome.truncated`
        reports how many qualified leads were dropped.
        """
        # Both of these can fail, and both are resolved before `search_raw` charges anything.
        # Validating `limit` inside `build_outcome` alone would put the check on the far side
        # of the spend, so a mistyped limit would cost a credit to discover.
        limit = _limit(limit)
        query = self.query_for(profile, query_variant)

        payload = self.search_raw(location, query, page=page)
        return build_outcome(payload, profile, location, limit=limit, page=page, query=query)

    def search_raw(
        self, location: ResolvedLocation, query: str, *, page: int = 1
    ) -> dict[str, Any]:
        """The billed call, returning SearchAPI's response verbatim.

        Public because the fixture recorder needs the untouched body -- a fixture derived from
        parsed output could never catch a parsing bug. Callers wanting leads want
        `search_places`; this one qualifies nothing.
        """
        params = self.request_params(location, query, page=page)

        # Everything above this line can fail without costing anything, and everything that
        # can fail is above this line: an invalid page, an unusable query and a bad radius all
        # raise before a credit moves. Below it, the credit is gone whatever happens -- see
        # budget.py, "SPEND FIRST, NEVER REFUND". A timeout does not prove the query was not
        # billed; the response is what got lost, not necessarily the request.
        if self.budget is not None:
            self.budget.spend(PROVIDER, 1)

        try:
            response = self._client.get(self.endpoint, params=params)
        except httpx.TimeoutException:
            # `from None`, always: an httpx exception carries `.request.url`, and with
            # `auth_in_query` that URL is the API key.
            raise unavailable("SearchAPI timed out.", provider=PROVIDER) from None
        except httpx.HTTPError:
            raise unavailable("SearchAPI could not be reached.", provider=PROVIDER) from None

        if response.status_code >= 400:
            # The shared ladder, not a local one: 429 -> rate limited, 401/403 -> auth,
            # everything else -> bad response. The status is never accompanied by the body.
            raise from_status(response.status_code, provider=PROVIDER) from None

        try:
            payload = response.json()
        except ValueError:
            raise _bad("SearchAPI returned a body that was not JSON") from None
        if not isinstance(payload, dict):
            raise _bad("SearchAPI returned a JSON body that was not an object")
        if payload.get("error"):
            # A 200 carrying an `error` key. The value is upstream text and is not repeated.
            raise _bad("SearchAPI reported an error for this search")
        return payload

    # -- request construction, exposed so tests can assert on it without a socket ----------

    def request_params(
        self, location: ResolvedLocation, query: str, *, page: int = 1
    ) -> dict[str, Any]:
        """Exactly what goes on the wire, in one inspectable place."""
        text = str(query).strip()
        if not text:
            raise ValueError("query must be a non-empty string")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError(f"page must be an integer >= 1, got {page!r}")

        params: dict[str, Any] = {
            "engine": ENGINE,
            "q": text,
            "ll": _ll(location),
            # Pinned. Never `location.gl` -- see the module docstring.
            "hl": self.hl,
            "gl": self.gl,
            "page": page,
        }
        if self.api_key and self.auth_in_query:
            params["api_key"] = self.api_key
        return params

    def query_for(self, profile: NicheProfile, query_variant: int | str | None = None) -> str:
        """Resolve `query_variant` against a profile's query list."""
        return resolve_query(profile, query_variant)


# --- helpers -------------------------------------------------------------------------------


def resolve_query(profile: NicheProfile, query_variant: int | str | None = None) -> str:
    """Resolve `query_variant` against a profile's query list.

    A free function, not a method, for the same reason `build_outcome` is one: the fixture
    provider must resolve a variant identically to the live client, and it cannot do that by
    borrowing a method from a class that owns an HTTP transport.

        None  the profile's first query -- the breadth-first default
        int   an index into `profile.queries`, for a deliberate deeper sweep
        str   sent verbatim, for an operator testing a hunch by hand
    """
    if query_variant is None:
        return profile.queries[0]
    if isinstance(query_variant, bool):
        # `True` would index to queries[1] and quietly buy a different search.
        raise TypeError("query_variant must be an int index, a string, or None")
    if isinstance(query_variant, int):
        if not 0 <= query_variant < len(profile.queries):
            raise ValueError(
                f"query_variant {query_variant} is out of range for niche {profile.id!r}, "
                f"which has {len(profile.queries)} queries"
            )
        return profile.queries[query_variant]
    text = str(query_variant).strip()
    if not text:
        raise ValueError("query_variant must not be blank")
    return text


def _bad(message: str) -> ProviderError:
    return bad_response(message, provider=PROVIDER)


def _ll(location: ResolvedLocation) -> str:
    """`@lat,lng,<metres>m`, the documented metres form.

    Composed here rather than taken from `ResolvedLocation.ll`, which omits the unit. See the
    module docstring: an unsuffixed radius is not a documented form, and the failure it
    invites -- every area search silently re-centring on the city -- is invisible in the
    results and costs a credit per occurrence.
    """
    radius = int(location.radius_meters)
    # Clamping rather than raising: a radius outside the documented range is rejected by
    # SearchAPI *after* the request is charged, and the nearest legal radius preserves the
    # caller's intent (a very tight search stays as tight as the vendor allows).
    radius = max(MIN_RADIUS_METERS, min(MAX_RADIUS_METERS, radius))
    return f"@{location.latitude},{location.longitude},{radius}m"


def _limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"limit must be an integer >= 1, got {limit!r}")
    return limit


def _places(payload: Any) -> list[Any]:
    """`local_results`, with an empty search distinguished from a broken one.

    A search that found nothing is a normal, billed, useful answer -- "no cloud kitchens in
    this pocket" is a real finding -- so an absent `local_results` is zero results, not an
    error. But a body with none of SearchAPI's envelope keys either is not a SearchAPI
    response at all, and reporting that as "no results" would let a proxy error page look
    like an empty neighbourhood.
    """
    if not isinstance(payload, dict):
        raise _bad("SearchAPI returned a response that was not an object")

    results = payload.get("local_results")
    if isinstance(results, list):
        return results
    if results is not None:
        raise _bad("SearchAPI returned local_results that was not a list")
    if any(key in payload for key in ENVELOPE_KEYS):
        return []
    raise _bad("SearchAPI returned a body that was not the documented envelope")


def _coordinates(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return (None, None)
    return (_float(value.get("latitude")), _float(value.get("longitude")))


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None
