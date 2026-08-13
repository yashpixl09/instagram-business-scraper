"""A maps provider backed by recorded responses.

This is not a test double. It is how the discovery, scoring and export layers get built and
demonstrated without spending any of fifty non-renewing SearchAPI credits -- an allowance
where a single debugging loop against the live API is the most expensive mistake available.

What makes a fixture trustworthy is that nothing is re-implemented here. Parsing,
qualification and query resolution all come from `searchapi` itself:

    resolve_query   the same variant rules
    build_outcome   the same place -> Lead conversion and the same matches_niche gate

So a fixture recorded from a live call produces byte-identical leads to that call, and a
parsing bug shows up in fixture-driven tests instead of hiding until the credits are gone.
Fixtures are recorded from `SearchApiClient.search_raw`, which returns the response
untouched -- a fixture derived from parsed output could never catch a parsing bug at all.

The provider never touches the network and never touches the budget. Spending is the live
client's job, and a fixture run that decremented the ledger would make the operator's
remaining-credit figure lie.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..geo.scope import ResolvedLocation
from ..niches import NicheProfile
from .errors import ProviderError, bad_response
from .searchapi import PROVIDER, RESULTS_PER_PAGE, build_outcome, resolve_query

__all__ = [
    "FixtureMapsProvider",
    "FixtureNotFound",
    "fixture_name",
    "record_fixture",
    "scrub_payload",
]


class FixtureNotFound(LookupError):
    """No recorded response for this niche and page.

    Deliberately loud. Returning an empty page instead would be indistinguishable from
    "Google knows of no salons in Indiranagar", which is a conclusion a developer would
    act on and a fact this provider is in no position to assert.
    """


def fixture_name(niche_id: str, page: int = 1) -> str:
    return f"{niche_id}-p{page}.json"


class FixtureMapsProvider:
    """Serves recorded SearchAPI responses through the live client's interface.

    Exposes `.api_key` because the API layer duck-types on that attribute to decide whether
    a provider is configured. A fixture provider is always configured.
    """

    api_key = "fixture"

    def __init__(self, fixture_dir: str | Path, *, strict: bool = True) -> None:
        self.fixture_dir = Path(fixture_dir)
        self.strict = strict
        # Requests served, for tests asserting that a run issued the number of searches it
        # claimed. The live client's equivalent is the budget ledger; this is the free
        # counterpart, and the property both must satisfy is one call per page.
        self.calls: list[dict[str, Any]] = []

    def search_places(
        self,
        location: ResolvedLocation,
        profile: NicheProfile,
        limit: int = RESULTS_PER_PAGE,
        page: int = 1,
        query_variant: int | str | None = None,
    ) -> Any:
        """One recorded page, qualified against `profile`. Signature matches the live client."""
        query = resolve_query(profile, query_variant)
        payload = self.load(profile.id, page)
        self.calls.append({"niche_id": profile.id, "page": page, "query": query})
        return build_outcome(payload, profile, location, limit=limit, page=page, query=query)

    def search_raw(
        self, location: ResolvedLocation, query: str, *, page: int = 1
    ) -> dict[str, Any]:
        """The recorded body, unparsed -- mirrors the live client's recording seam."""
        raise FixtureNotFound(
            "search_raw needs a niche to find its fixture; call search_places instead"
        )

    def load(self, niche_id: str, page: int = 1) -> dict[str, Any]:
        path = self.fixture_dir / fixture_name(niche_id, page)
        if not path.exists():
            if self.strict:
                raise FixtureNotFound(
                    f"no fixture {path.name!r} in {self.fixture_dir}. Record one with "
                    f"record_fixture(), or construct the provider with strict=False to "
                    f"treat a missing page as the end of results."
                )
            # Non-strict exists for paging: page 2 legitimately may not have been recorded,
            # and a sweep walking off the end should stop, not fail.
            return {"local_results": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise _bad(f"fixture {path.name!r} is not valid JSON: {exc}") from exc

    def available(self) -> list[str]:
        """Niche ids with at least a page 1 recorded."""
        return sorted(
            path.stem.rsplit("-p", 1)[0]
            for path in self.fixture_dir.glob("*-p1.json")
        )


_SECRET_PARAMS = ("api_key", "apikey", "key", "token", "access_token")
_REDACTED = "REDACTED"


def scrub_payload(payload: Any, api_key: str | None = None) -> Any:
    """Strip credentials out of a response before it is written to disk.

    `SearchApiClient` sends the key as a query parameter by default, and SearchAPI echoes the
    request back in `search_metadata`. Fixtures are committed to the repository. Without this,
    the first recording of a live response is the single most likely place in this codebase
    for a working credential to be published -- and it would look like an ordinary test asset,
    which is exactly why nobody would check it.

    Three independent passes, because each alone has a gap the others cover:

      * any field NAMED like a credential has its value replaced outright. SearchAPI echoes
        `search_parameters.api_key` as a bare string, which is not a URL and which the pass
        below would therefore walk straight past;
      * every URL-shaped string has its secret query parameters replaced. Both of these work
        without knowing the key, so they still cover a key this process never held -- a
        fixture pasted in by hand, or one recorded under a key since rotated;
      * if the key IS known, every remaining occurrence is replaced wherever it sits. This
        covers shapes nobody anticipated, which is the category that matters, since the
        anticipated ones are already handled above.

    Returns a new structure; the caller's payload is not modified. A recorded fixture must be
    the response as parsed, minus only the credential -- scrubbing in place would mean the
    live client saw different data from the fixture derived from it.
    """
    if isinstance(payload, dict):
        return {
            key: _REDACTED
            if isinstance(key, str) and key.lower() in _SECRET_PARAMS
            else scrub_payload(value, api_key)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [scrub_payload(item, api_key) for item in payload]
    if isinstance(payload, str):
        return _scrub_text(payload, api_key)
    return payload


def _scrub_text(text: str, api_key: str | None) -> str:
    if "?" in text and "=" in text:
        head, _, query = text.partition("?")
        pairs = []
        for pair in query.split("&"):
            name, sep, value = pair.partition("=")
            pairs.append(f"{name}={_REDACTED}" if name.lower() in _SECRET_PARAMS else pair)
            del sep, value
        text = f"{head}?{'&'.join(pairs)}"
    if api_key and api_key in text:
        text = text.replace(api_key, _REDACTED)
    return text


def record_fixture(
    payload: dict[str, Any],
    niche_id: str,
    fixture_dir: str | Path,
    page: int = 1,
    *,
    api_key: str | None = None,
) -> Path:
    """Save a live response so the credit that bought it is never spent on the same page twice.

    Takes the body from `SearchApiClient.search_raw` verbatim, minus credentials -- see
    `scrub_payload`. Everything else Google returned is kept, including fields nothing reads
    yet: the next thing to consume `popular_times` or `review_results` should not need a fresh
    credit to see one.

    Pass `api_key` whenever it is known. Scrubbing works without it, but only for the shapes
    that have been anticipated.
    """
    directory = Path(fixture_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / fixture_name(niche_id, page)
    scrubbed = scrub_payload(payload, api_key)
    path.write_text(json.dumps(scrubbed, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _bad(message: str) -> ProviderError:
    return bad_response(message, provider=PROVIDER)
