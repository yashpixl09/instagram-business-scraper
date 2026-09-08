"""The Instagram handle, and nothing else.

THIS MODULE MUST NEVER TALK TO INSTAGRAM.

Not "should not", not "prefers not to". Instagram is reached exactly once in this system,
in Phase 7, through the operator's own logged-in browser session on `worker-browser`, which
is pinned to a single process for a reason: one session, one identity, serialised. A second
path that fetches `instagram.com` from a datacentre IP -- even once, even through a
provider that would happily do it -- puts a challenge on the account the whole Instagram
stage depends on, and the cost of that lands on a human who then cannot log in.

So the guarantee here is structural rather than disciplinary: this module imports no HTTP
client, holds no provider, and every function on it is pure. There is nothing in it that
COULD issue a request. `tests/test_enrichment.py` also holds the belt to those braces, with
a transport that fails the test on any request to an Instagram host.

What it does instead: the search `website.py` already paid a rate-limit token for almost
always surfaces the profile, because a local business's Instagram page outranks nearly
everything else for `"<name>" <city>`. So the handle is read out of the SAME result set --
one search, two answers -- and out of the business's own site text when one was fetched,
where a bio link or a footer icon is stronger evidence than a search hit.

Only the handle is stored. Followers, engagement and bio belong to the browser stage, and
storing a hollow profile row now would make the freshness check believe Instagram had
already been enriched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .website import name_matches

#: Every host that serves Instagram profiles. Used to RECOGNISE urls, never to fetch them.
INSTAGRAM_HOSTS = ("instagram.com", "instagr.am", "ig.me", "cdninstagram.com")

#: A handle is 1-30 characters of letters, digits, dots and underscores. Instagram's own
#: rule, and a useful filter: anything else in the first path segment is not a profile.
HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,30}$")

#: First path segments that are Instagram's, not a business's. A URL like
#: `instagram.com/explore/locations/123/pune` names a place, and `instagram.com/p/Cx1/` is
#: somebody's post -- storing either as a handle produces an outreach message addressed to
#: a URL fragment.
RESERVED_SEGMENTS = frozenset(
    {
        "p",
        "reel",
        "reels",
        "tv",
        "stories",
        "s",
        "explore",
        "directory",
        "accounts",
        "about",
        "developer",
        "developers",
        "legal",
        "privacy",
        "terms",
        "help",
        "web",
        "graphql",
        "ajax",
        "api",
        "oauth",
        "challenge",
        "sessions",
        "emails",
        "session",
        "download",
        "topics",
        "locations",
        "direct",
        "your_activity",
        "lite",
        "static",
        "images",
        "favicon.ico",
        # `instagram.com/popular/<slug>/` is Instagram's own tag/topic aggregation page --
        # the same shape as `/explore/` or `/p/`, just discovered live: four real
        # businesses this session (Shri Krishna Sweets, Monalisa Boutique, Ananya Designer
        # Boutique, Annu Shree Boutique) all got "popular" stored as their handle, each from
        # a URL like `instagram.com/popular/krishna-mysore-pak/` -- Google's indexed copy of
        # a topic page, not a profile. `handle_from_url` took the first path segment without
        # knowing this one is reserved too.
        "popular",
    }
)

#: `instagram.com/<handle>` wherever it appears in text -- a page footer, a bio link, a
#: search snippet. Anchored on the host so a bare `@handle` in prose is never picked up:
#: "@zomato" in a caption is not the shop's account.
_PROFILE_IN_TEXT = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?instagram\.com/([A-Za-z0-9._]{1,30})",
    re.I,
)

PROFILE_URL = "https://www.instagram.com/{handle}/"


def is_instagram_url(url: str | None) -> bool:
    """Whether this URL points at Instagram. The recogniser, not a fetcher."""
    host = _host(url)
    return any(host == domain or host.endswith("." + domain) for domain in INSTAGRAM_HOSTS)


def _host(url: str | None) -> str:
    if not url or not isinstance(url, str):
        return ""
    candidate = url if "://" in url else "https://" + url
    return urlparse(candidate).netloc.lower().split("@")[-1].split(":")[0]


def handle_from_url(url: str | None) -> str | None:
    """The profile handle in an Instagram URL, or None if it is not a profile URL."""
    if not is_instagram_url(url):
        return None
    candidate = url if "://" in (url or "") else "https://" + (url or "")
    segments = [part for part in urlparse(candidate).path.split("/") if part]
    if not segments:
        return None
    return normalise_handle(segments[0])


def normalise_handle(raw: str | None) -> str | None:
    """A handle, lowercased and validated, or None. Never a guess."""
    if not raw or not isinstance(raw, str):
        return None
    handle = raw.strip().lstrip("@").rstrip("/").lower()
    if not handle or handle in RESERVED_SEGMENTS:
        return None
    if not HANDLE_PATTERN.match(handle):
        return None
    # A handle of only dots and underscores is a path artefact, not an account.
    if not any(character.isalnum() for character in handle):
        return None
    return handle


def handles_in_text(text: str | None) -> list[str]:
    """Every Instagram handle mentioned in a block of text, in order, deduplicated."""
    if not text or not isinstance(text, str):
        return []
    found: list[str] = []
    for raw in _PROFILE_IN_TEXT.findall(text):
        handle = normalise_handle(raw)
        if handle and handle not in found:
            found.append(handle)
    return found


@dataclass(frozen=True)
class HandleCandidate:
    handle: str
    url: str
    origin: str
    matched: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "url": self.url,
            "origin": self.origin,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class HandleFinding:
    """One handle, or a recorded absence.

    `status` follows the same rule as everywhere else in this phase: `ok` when a handle was
    found, `no_data` when the search ran and there was none, and `error`/`blocked` when the
    search never happened -- which is NOT the same statement and must not be cached as one.
    """

    status: str
    handle: str | None = None
    url: str | None = None
    confidence: str = "high"
    candidates: tuple[HandleCandidate, ...] = ()
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.handle is not None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "handle": self.handle,
            "url": self.url,
            "confidence": self.confidence,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def unavailable(reason: str, *, status: str = "error") -> HandleFinding:
    """The search did not answer. Distinct from "they have no Instagram"."""
    return HandleFinding(status=status, reason=reason)


def find_handle(
    results: Sequence[Any] = (),
    *,
    name: str,
    site_text: str | None = None,
    listed_handle: str | None = None,
) -> HandleFinding:
    """Read the handle out of results already paid for. Issues no request, ever.

    Sources in descending order of trust:

      1. `listed_handle` -- whatever the business row already carries;
      2. `site_text` -- a link on the business's OWN site, which is the business asserting
         its own handle rather than a search engine guessing;
      3. the search results, preferring a handle whose spelling matches the business name.

    A handle that does not match the name is still returned when it is the only one, at
    `confidence="low"`: a lookalike account is worth an operator's glance, and silently
    dropping it would look identical to having no Instagram at all.
    """
    candidates: list[HandleCandidate] = []
    seen: set[str] = set()

    def add(handle: str | None, url: str, origin: str) -> None:
        handle = normalise_handle(handle)
        if not handle or handle in seen:
            return
        seen.add(handle)
        candidates.append(
            HandleCandidate(
                handle=handle,
                url=url or PROFILE_URL.format(handle=handle),
                origin=origin,
                matched=_handle_matches(handle, name),
            )
        )

    add(listed_handle, "", "listing")
    for handle in handles_in_text(site_text):
        add(handle, PROFILE_URL.format(handle=handle), "own_site")
    for result in results or ():
        url = getattr(result, "url", None)
        if is_instagram_url(url):
            add(handle_from_url(url), url, "search")
    # Snippets carry the profile URL as often as the result row does, and reading them
    # costs nothing -- the page was already bought.
    for result in results or ():
        for handle in handles_in_text(getattr(result, "snippet", "") or ""):
            add(handle, PROFILE_URL.format(handle=handle), "snippet")

    if not candidates:
        return HandleFinding(status="no_data", reason="no instagram profile in the results")

    matched = [candidate for candidate in candidates if candidate.matched]
    best = matched[0] if matched else candidates[0]
    return HandleFinding(
        status="ok",
        handle=best.handle,
        url=best.url or PROFILE_URL.format(handle=best.handle),
        confidence=_confidence(best),
        candidates=tuple(candidates),
    )


def _confidence(candidate: HandleCandidate) -> str:
    if candidate.origin in ("listing", "own_site"):
        return "high"
    if candidate.matched:
        return "high"
    return "low"


def _handle_matches(handle: str, name: str) -> bool:
    """Whether a handle spells the business's name.

    Handles are written `sweet.corner.bakery`, `sweetcornerbakery_blr`, `sweetcorner__`.
    Splitting on separators and re-joining lets the same coverage rule `website.py` uses
    for domains answer here too, so the two modules agree on what "this is theirs" means.
    """
    return name_matches(handle.replace(".", " ").replace("_", " "), name)


def unmatched_handles(finding: HandleFinding) -> Iterable[str]:
    """Handles found but not tied to the name. For an operator to eyeball, not to store."""
    return (candidate.handle for candidate in finding.candidates if not candidate.matched)


__all__ = [
    "HANDLE_PATTERN",
    "INSTAGRAM_HOSTS",
    "PROFILE_URL",
    "RESERVED_SEGMENTS",
    "HandleCandidate",
    "HandleFinding",
    "find_handle",
    "handle_from_url",
    "handles_in_text",
    "is_instagram_url",
    "normalise_handle",
    "unavailable",
    "unmatched_handles",
]
