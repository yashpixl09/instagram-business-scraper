"""What "where" means to this system: a scope the operator declares, and a point it becomes.

SearchAPI's Google Maps endpoint has no `location` parameter and no geocoding endpoint. The
only geographic control it offers is `ll=@lat,lng,meters`, so every search this system issues
needs a coordinate that had to come from somewhere else. That is the whole reason this module
exists, and it is why `GeoScope` (what a human types) and `ResolvedLocation` (what a search
needs) are two different types instead of one.

`city` is required and anchors everything. `country` and `state` only disambiguate -- they
stop "Jaipur" landing in Texas and "Hyderabad" landing in Pakistan -- so they are composed
into the query string and never used to narrow a search. `areas` are the coverage strategy:
Google returns twenty results biased to the centre point, so one city-wide search of a 40km
metro only ever sees its core, and area-level searches are how the rest is reached.

This module is import-pure by construction -- no HTTP, no database -- but it is deliberately
outside `tests/test_core_purity.py`'s list, which governs `lead_engine/*.py` only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A tight circle around a neighbourhood; the unit of a sweep.
AREA_RADIUS_METERS = 3000

#: A whole metro. Used only when the operator named no areas at all.
CITY_RADIUS_METERS = 15000

#: Since `city` is mandatory, a resolution is only ever one of these two.
RADIUS_METERS: dict[str, int] = {"area": AREA_RADIUS_METERS, "city": CITY_RADIUS_METERS}

PRECISIONS: tuple[str, ...] = tuple(RADIUS_METERS)

# Country name -> SearchAPI `gl`. Deliberately short: this exists to turn the handful of
# names an operator actually types into a code, not to be a world atlas. An unrecognised
# name resolves to None rather than a guess -- see `GeoScope.gl`.
COUNTRY_CODES: dict[str, str] = {
    "india": "in",
    "bharat": "in",
    "unitedstates": "us",
    "unitedstatesofamerica": "us",
    "usa": "us",
    "america": "us",
    "unitedkingdom": "gb",
    "uk": "gb",
    "greatbritain": "gb",
    "england": "gb",
    "unitedarabemirates": "ae",
    "uae": "ae",
    "singapore": "sg",
    "canada": "ca",
    "australia": "au",
    "srilanka": "lk",
    "bangladesh": "bd",
    "nepal": "np",
}

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(value: object) -> str:
    """A comparison key: lowercase, alphanumerics only.

    "J.P. Nagar", "JP Nagar" and "jp  nagar" are the same neighbourhood typed by three
    different people, and this is what makes them the same seed lookup.
    """
    return _NON_ALPHANUMERIC.sub("", str(value).lower())


def clean(value: object | None) -> str:
    """Display form: trimmed, internal whitespace collapsed, punctuation preserved."""
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def radius_for(precision: str) -> int:
    """The radius a precision implies, or a loud failure.

    A typo'd precision must not silently become a default radius: the difference between
    3km and 15km is the difference between a neighbourhood and a city.
    """
    try:
        return RADIUS_METERS[precision]
    except KeyError:
        raise ValueError(
            f"precision must be one of {PRECISIONS!r}, got {precision!r}"
        ) from None


def compose(*parts: object | None) -> str:
    """Join location components most-specific-first: "Indiranagar, Bangalore, Karnataka, India".

    Blank components drop out, and a component identical to the one before it drops out too.
    That second rule is not cosmetic: Delhi is a city inside a state called Delhi, so a
    faithful composition would ask a geocoder for "Delhi, Delhi, India" -- which is the kind
    of string that returns a different answer than "Delhi, India" for no good reason.
    """
    kept: list[str] = []
    for part in parts:
        text = clean(part)
        if not text:
            continue
        if kept and normalize(text) == normalize(kept[-1]):
            continue
        kept.append(text)
    return ", ".join(kept)


@dataclass(frozen=True)
class GeoScope:
    """Where a run is allowed to look, as an operator would describe it."""

    city: str
    country: str | None = None
    state: str | None = None
    areas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # `areas="Indiranagar"` is a plausible typo that iterates into eleven one-character
        # areas and eleven wasted searches. It is a type error, so it is raised as one.
        if isinstance(self.areas, str):
            raise TypeError("GeoScope.areas must be a sequence of names, not a single string")

        city = clean(self.city)
        if not city:
            raise ValueError(
                "GeoScope.city is required: without it there is nothing to anchor a "
                "coordinate to, and an area name alone is ambiguous across every metro"
            )
        object.__setattr__(self, "city", city)
        object.__setattr__(self, "country", clean(self.country) or None)
        object.__setattr__(self, "state", clean(self.state) or None)

        # Two spellings of one area buy two identical searches at full price.
        unique: dict[str, str] = {}
        for area in self.areas:
            name = clean(area)
            if name:
                unique.setdefault(normalize(name), name)
        object.__setattr__(self, "areas", tuple(unique.values()))

    @property
    def gl(self) -> str | None:
        """The SearchAPI `gl` this scope declares, or None if it declares nothing usable.

        None is the honest answer for a country name this module does not know: the name is
        still in the composed query string doing its disambiguating work, and inventing a
        two-letter code would bias every search toward the wrong country silently.
        """
        if not self.country:
            return None
        key = normalize(self.country)
        if key in COUNTRY_CODES:
            return COUNTRY_CODES[key]
        if len(key) == 2 and key.isalpha():
            return key
        return None

    def city_query(self) -> str:
        return compose(self.city, self.state, self.country)

    def area_query(self, area: str) -> str:
        return compose(area, self.city, self.state, self.country)

    def targets(self) -> tuple[tuple[str, str], ...]:
        """Every (query, precision) pair this scope fans out to.

        One per area, or exactly one city-wide target when no areas were given.
        """
        if not self.areas:
            return ((self.city_query(), "city"),)
        return tuple((self.area_query(area), "area") for area in self.areas)


@dataclass(frozen=True)
class ResolvedLocation:
    """A point a search can actually be issued against."""

    label: str
    latitude: float
    longitude: float
    radius_meters: int
    precision: str
    gl: str | None
    source_query: str

    def __post_init__(self) -> None:
        if self.precision not in RADIUS_METERS:
            raise ValueError(f"precision must be one of {PRECISIONS!r}, got {self.precision!r}")

        # Geocoders return coordinates as strings ("12.9784"), so coercion happens here
        # rather than at three call sites that would each have to remember.
        for field_name in ("latitude", "longitude"):
            try:
                object.__setattr__(self, field_name, float(getattr(self, field_name)))
            except (TypeError, ValueError):
                raise ValueError(
                    f"{field_name} must be a number, got {getattr(self, field_name)!r}"
                ) from None

        # Range validation, and only that. It catches a longitude parked in `latitude` when
        # that longitude exceeds 90 -- San Francisco's -122.4, Sydney's 151.2 -- and it does
        # NOT catch a Bangalore swap, because 77.6 is a legal latitude (it is in Siberia).
        # Nothing here can catch that one; the seed's per-area bounding-box test is what
        # covers the coordinates this system actually ships.
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude!r}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude out of range: {self.longitude!r}")

        radius = int(self.radius_meters)
        if radius <= 0:
            raise ValueError(f"radius_meters must be positive, got {self.radius_meters!r}")
        object.__setattr__(self, "radius_meters", radius)

        if self.gl is not None:
            gl = str(self.gl).strip().lower()
            # SearchAPI wants ISO 3166-1 alpha-2. "India" here would be accepted by the
            # dataclass and rejected by the vendor mid-run, which is the expensive place.
            if len(gl) != 2 or not gl.isalpha():
                raise ValueError(f"gl must be a two-letter country code, got {self.gl!r}")
            object.__setattr__(self, "gl", gl)

        if not clean(self.label):
            raise ValueError("label is required")
        if not clean(self.source_query):
            raise ValueError("source_query is required: a point with no provenance is a guess")

    @property
    def ll(self) -> str:
        """The only geographic control SearchAPI offers: `@lat,lng,meters`."""
        return f"@{self.latitude},{self.longitude},{self.radius_meters}"
