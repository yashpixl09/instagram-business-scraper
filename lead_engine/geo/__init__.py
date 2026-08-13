"""Geographic resolution: place names in, search coordinates out.

SearchAPI's Google Maps endpoint accepts no location string and offers no geocoding of its
own -- `ll=@lat,lng,meters` is the whole interface -- so coordinates have to be produced
before any search can be issued. This package is where they come from.

`scope.py` is import-pure and has no dependencies beyond the standard library. `resolver.py`
reaches the network and the database, so importing this package pulls in `httpx` and the
provider error taxonomy; that is deliberate, since a missing dependency should surface at
import rather than at the moment a run needs a coordinate.
"""

from __future__ import annotations

from .resolver import (
    MIN_INTERVAL_SECONDS,
    NOMINATIM_ENDPOINT,
    NOMINATIM_USER_AGENT,
    SEED_PATH,
    CachingResolver,
    ChainResolver,
    NominatimResolver,
    Resolver,
    SeedResolver,
    resolve_scope,
)
from .scope import (
    AREA_RADIUS_METERS,
    CITY_RADIUS_METERS,
    PRECISIONS,
    RADIUS_METERS,
    GeoScope,
    ResolvedLocation,
    compose,
    normalize,
    radius_for,
)

__all__ = [
    "AREA_RADIUS_METERS",
    "CITY_RADIUS_METERS",
    "MIN_INTERVAL_SECONDS",
    "NOMINATIM_ENDPOINT",
    "NOMINATIM_USER_AGENT",
    "PRECISIONS",
    "RADIUS_METERS",
    "SEED_PATH",
    "CachingResolver",
    "ChainResolver",
    "GeoScope",
    "NominatimResolver",
    "ResolvedLocation",
    "Resolver",
    "SeedResolver",
    "compose",
    "normalize",
    "radius_for",
    "resolve_scope",
]
