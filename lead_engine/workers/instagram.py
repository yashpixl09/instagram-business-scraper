"""The Instagram profile, read from a page the operator's own browser already fetched.

THIS MODULE MUST NEVER TALK TO INSTAGRAM EITHER.

`enrichment/social.py` already carries half of this rule -- it finds the HANDLE without
ever fetching a profile. This module is the other half: it turns a profile page someone
ELSE already fetched into follower counts, a bio, a category. It is Phase 7, which the
design spec pins to one place and one identity:

    Instagram is reached exactly once in this system, in Phase 7, through the operator's
    own logged-in browser session on `worker-browser`, which is pinned to a single process
    for a reason: one session, one identity, serialised. A second path that fetches
    instagram.com from a datacentre IP -- even once, even through a provider that would
    happily do it -- puts a challenge on the account the whole Instagram stage depends on,
    and the cost of that lands on a human who then cannot log in.

So, exactly as in `social.py`: this module imports no HTTP client and constructs nothing
that could reach the network on its own. The one thing that CAN issue a request --
`InstagramBrowser.fetch_profile` -- is a `Protocol` this module never implements, only
calls through an object handed to it by whoever assembles the process. A test that wants
to exercise the happy path hands in a fake that returns canned text; there is no
constructor here that would let it hand in anything real by accident.

WHY JSON, NOT RENDERED HTML
---------------------------
A profile page is a single-page app: the markup Chrome first receives carries almost none
of the numbers on it, because the follower count, the bio and the category are filled in
by an XHR the page issues to Instagram's own internal API
(`/api/v1/users/web_profile_info/?username=<handle>`) once the JS runs. A CDP session
already sitting in that page can read the RESPONSE BODY of that request directly --
`Network.getResponseBody` against the request the page itself made -- which is the exact
JSON payload below, before it was ever turned into DOM. Scraping the rendered DOM instead
would mean re-deriving numbers from formatted text ("18.4K followers") that Instagram
already rounds for display, and would break on every markup change the front end ships.
So `extract_profile` is written against that JSON shape, which is also the more honest
statement of what a real `InstagramBrowser` implementation will hand back: the body of one
network response, not a scrape.

WHAT COUNTS AS DATA, AND WHAT DOESN'T
--------------------------------------
Same discipline as `website.py` and `social.py`: a field that is not clearly present comes
back `None`, never a guess, and a bio is stored exactly as written -- including a phone
number or email address sitting inside it. `contacts.py` is the module that turns page
text into a named contact, for a different source (a business's own site) with a different
confidence model; re-implementing a shrunken version of it here, for one field on one
provider, is exactly the kind of duplicate logic that drifts from the original the first
time either one is fixed. This module reads a bio, not a contact.

STATUS, MATCHING THE FOUR-STATE VOCABULARY EVERYWHERE ELSE IN THIS PHASE
--------------------------------------------------------------------------
    ok        a profile was read -- including a private one. "This account is private and
              has this bio" is a fact about the business, not an absence.
    no_data   the handle does not resolve to an account. Also a fact about the business:
              the handle is wrong, or the account is gone.
    blocked   Instagram served a checkpoint/challenge/rate-limit page instead of a profile.
              A fact about the FETCHER, at this moment -- not about the business, and not
              cause to conclude anything about it.
    error     the browser call raised, or the response could not be parsed into anything
              recognisable. A fact about US.

Only `ok` and `no_data` are allowed to move `enrichment/cache.py`'s negative-cache ladder,
for the reason that module's own docstring gives: a `blocked` afternoon that recorded
misses would buy months of not looking at businesses nobody actually checked.

RATE-GATING IS STRUCTURAL, NOT DISCIPLINED
-------------------------------------------
`RateGatedInstagramBrowser` holds a `threading.Lock` for the FULL DURATION of the wrapped
call, not just while it reads a timestamp. That is what makes "only one fetch in flight"
true regardless of who calls it or how many threads a future worker process runs: a second
caller's `fetch_profile` blocks on the lock itself, inside this class, before it can reach
the wrapped browser at all. There is no second method that reaches `self._browser` and no
way to construct a caller that bypasses the wrapper once one is in place -- the guarantee
comes from what the object CAN do, the same shape `social.py`'s docstring describes for
itself ("there is nothing in it that COULD issue a request").

WHAT A HUMAN STILL HAS TO BUILD
--------------------------------
Everything above and below is data-in, data-out. Nobody has written the thing that drives
a real Chrome session -- navigates it to `https://www.instagram.com/<handle>/` under the
operator's own logged-in profile, waits for the `web_profile_info` XHR, and returns its
response body as a string. That is `InstagramBrowser.fetch_profile`'s one job, and it is
deliberately not implemented here: doing so from this agent run would mean writing code
that could, if ever pointed at anything real, become the second path into `instagram.com`
this whole module exists to prevent.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from ..enrichment.cache import INSTAGRAM, EnrichmentCache, utc_now
from ..enrichment.social import PROFILE_URL, normalise_handle

# --- status vocabulary ----------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_NO_DATA = "no_data"
STATUS_BLOCKED = "blocked"
STATUS_ERROR = "error"

#: `enrichments.source` for rows this module writes. Distinct from `social.py`'s
#: `instagram_handle` -- that row says WHICH account a business has; this one says what the
#: account itself contains. Two different questions, two different rows, so that a business
#: whose handle is known but whose profile fetch failed does not read as "no Instagram".
SOURCE_INSTAGRAM_PROFILE = "instagram_profile"

#: Text that shows up on Instagram's checkpoint/rate-limit pages, whether the browser hands
#: back that page's JSON error body or its rendered HTML. Matched case-insensitively against
#: whatever text came back, because a challenge page is exactly the kind of response that
#: does not reliably arrive as clean JSON.
_BLOCKED_MARKERS = re.compile(
    r"checkpoint_required|challenge_required|challenge required|please wait a few minutes"
    r"|unusual activity|rate limit|try again later|restrict certain content",
    re.I,
)

#: Instagram's own wording for "this handle resolves to nothing", seen in the `message`
#: field of a failed `web_profile_info` call.
_NOT_FOUND_MARKERS = re.compile(
    r"not found|no longer available|doesn't exist|does not exist|user not found",
    re.I,
)


# --- the finding ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileFinding:
    """What one profile fetch produced, or a recorded reason it produced nothing.

    `bio` is copied verbatim, including a phone number or email a business put in it --
    see the module docstring for why this is deliberately not `contacts.py`'s job.
    """

    status: str
    handle: str | None = None
    followers: int | None = None
    following: int | None = None
    posts: int | None = None
    bio: str | None = None
    external_url: str | None = None
    is_private: bool | None = None
    category: str | None = None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.status == STATUS_OK

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "handle": self.handle,
            "followers": self.followers,
            "following": self.following,
            "posts": self.posts,
            "bio": self.bio,
            "external_url": self.external_url,
            "is_private": self.is_private,
            "category": self.category,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def unavailable(handle: str | None, reason: str, *, status: str = STATUS_ERROR) -> ProfileFinding:
    """The fetch did not produce a profile. Distinct from "this account has nothing in it"."""
    return ProfileFinding(status=status, handle=handle, reason=reason)


# --- pure extraction --------------------------------------------------------------------------


def extract_profile(raw: str | None, *, handle: str) -> ProfileFinding:
    """Turn one already-fetched `web_profile_info` response into a `ProfileFinding`.

    Pure: no request is made here, ever -- `raw` is text somebody else already fetched.
    Never raises. Bad input produces a distinct, correct status instead of a crash or a
    guessed number; see the module docstring's status table for what each branch means.
    """
    normalised = normalise_handle(handle)
    if normalised is None:
        return unavailable(handle, "not a valid instagram handle", status=STATUS_ERROR)

    if not raw or not isinstance(raw, str) or not raw.strip():
        return unavailable(normalised, "empty response from the browser", status=STATUS_ERROR)

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        if _BLOCKED_MARKERS.search(raw):
            return unavailable(
                normalised, "challenge or rate-limit page returned", status=STATUS_BLOCKED
            )
        return unavailable(normalised, "response was not valid JSON", status=STATUS_ERROR)

    if not isinstance(payload, dict):
        return unavailable(normalised, "unexpected response shape", status=STATUS_ERROR)

    data = payload.get("data")
    user = data.get("user") if isinstance(data, dict) else None

    if not isinstance(user, dict):
        message = payload.get("message") if isinstance(payload.get("message"), str) else ""
        if _BLOCKED_MARKERS.search(message) or _BLOCKED_MARKERS.search(raw):
            return unavailable(
                normalised, "challenge or rate-limit response", status=STATUS_BLOCKED
            )
        if _NOT_FOUND_MARKERS.search(message):
            return unavailable(normalised, "account not found", status=STATUS_NO_DATA)
        return unavailable(
            normalised, "unexpected response shape: no user data", status=STATUS_ERROR
        )

    return ProfileFinding(
        status=STATUS_OK,
        handle=normalised,
        followers=_edge_count(user.get("edge_followed_by")),
        following=_edge_count(user.get("edge_follow")),
        posts=_edge_count(user.get("edge_owner_to_timeline_media")),
        bio=_str_or_none(user.get("biography")),
        external_url=_str_or_none(user.get("external_url")),
        is_private=_bool_or_none(user.get("is_private")),
        category=_str_or_none(user.get("category_name")),
    )


def _edge_count(edge: Any) -> int | None:
    if isinstance(edge, dict):
        count = edge.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return None


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


# --- the browser seam ---------------------------------------------------------------------


class InstagramBrowser(Protocol):
    """What this module needs from a real, operator-supervised browser session.

    One method, mirroring `website.py`'s `WebProvider.fetch(url) -> str`: the caller does
    not need to know whether the real implementation does one CDP round trip or five
    (navigate, wait for the XHR, read its response body) to produce that string. Splitting
    this into `navigate()` / `read()` would expose CDP's shape without buying this module
    anything -- it never needs to navigate without reading, or read without navigating --
    and every extra method here is one more thing a future caller could get wrong about
    which order they are safe to call in.
    """

    def fetch_profile(self, handle: str) -> str: ...


#: Minimum seconds between two fetches through `RateGatedInstagramBrowser`. A STARTING
#: NUMBER, not a validated one -- exactly the caveat the design spec's own "Open Questions"
#: section attaches to its guesses. Thirty seconds is conservative enough that a human
#: reading the account's own traffic would not call it scraping, but nobody has run this
#: against a real, supervised session long enough to say it is right. Raise or lower it once
#: there is operational experience to do that WITH, not before.
MIN_FETCH_INTERVAL_SECONDS = 30.0


class RateGatedInstagramBrowser:
    """Wraps a real `InstagramBrowser` so at most one fetch is ever in flight, spaced apart.

    See the module docstring's "RATE-GATING IS STRUCTURAL" section: the lock below is held
    for the entire wrapped call, not just while a timestamp is read, which is what makes
    "one at a time" true of every caller rather than true of callers that happen to behave.
    """

    def __init__(
        self,
        browser: InstagramBrowser,
        *,
        min_interval: float = MIN_FETCH_INTERVAL_SECONDS,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._browser = browser
        self._min_interval = float(min_interval)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._last_call_at: float | None = None

    def fetch_profile(self, handle: str) -> str:
        with self._lock:
            now = self._clock()
            if self._last_call_at is not None:
                wait = self._min_interval - (now - self._last_call_at)
                if wait > 0:
                    self._sleep(wait)
            try:
                return self._browser.fetch_profile(handle)
            finally:
                self._last_call_at = self._clock()


# --- writing the finding -------------------------------------------------------------------


class EvidenceStore(Protocol):
    """`Repository`, narrowed to the one write this module performs.

    Deliberately re-declared here rather than imported from `enrichment.service` -- this
    module is a new, independent caller of `Repository.insert_enrichment`, not a piece of
    that service, and the two should be able to change their internals without either one's
    tests knowing about the other's Protocol object.
    """

    def insert_enrichment(
        self,
        business_id: UUID,
        source: str,
        status: str,
        data: dict[str, Any],
        *,
        source_url: str | None = None,
        run_id: UUID | None = None,
    ) -> Any: ...


@dataclass
class InstagramProfileLookup:
    """Fetch one profile, extract it, record it. The seam a task handler will call.

    `store` and `cache` are both optional so `extract_profile`'s behaviour can be exercised
    with nothing behind it; when they are supplied, this is where the four-state status
    rule turns into actual side effects: every fetch writes one `enrichments` row, and only
    `ok`/`no_data` ever touch `cache` -- see the module docstring.

    `cache` is `enrichment.cache.EnrichmentCache`, reused rather than reimplemented, keyed
    on the existing `INSTAGRAM` question. That question's only source today is
    `social.py`'s handle finder; sharing its ladder with the profile fetch means a business
    whose profile could never be fetched backs off exactly as a business with no handle
    does, which is the conservative choice given `cache.py`'s question table cannot be
    extended from here (this module may read it, never edit it). Wiring anything more
    specific than that is a decision for whoever adds a queue task type for this handler.
    """

    browser: InstagramBrowser
    store: EvidenceStore | None = None
    cache: EnrichmentCache | None = None
    clock: Callable[[], Any] = utc_now

    def run(self, *, business_id: UUID, handle: str, run_id: UUID | None = None) -> ProfileFinding:
        try:
            raw = self.browser.fetch_profile(handle)
        except Exception as exc:  # noqa: BLE001 - see runner.py's identical reasoning
            # The class name only, never `str(exc)`. A browser client's exception can carry
            # the URL it was navigating, and that URL is the operator's own authenticated
            # session -- the same reasoning `workers/runner.py:redacted_failure` applies to
            # every other provider in this project.
            finding = unavailable(
                normalise_handle(handle) or handle,
                f"browser fetch failed: {type(exc).__name__}",
                status=STATUS_ERROR,
            )
        else:
            finding = extract_profile(raw, handle=handle)

        if self.store is not None:
            self.store.insert_enrichment(
                business_id,
                SOURCE_INSTAGRAM_PROFILE,
                finding.status,
                finding.as_dict(),
                source_url=PROFILE_URL.format(handle=finding.handle or handle),
                run_id=run_id,
            )

        if self.cache is not None:
            now = self.clock()
            if finding.status == STATUS_NO_DATA:
                self.cache.record_miss(business_id, INSTAGRAM, now=now)
            elif finding.status == STATUS_OK:
                self.cache.record_success(business_id, INSTAGRAM, now=now)

        return finding


__all__ = [
    "MIN_FETCH_INTERVAL_SECONDS",
    "SOURCE_INSTAGRAM_PROFILE",
    "STATUS_BLOCKED",
    "STATUS_ERROR",
    "STATUS_NO_DATA",
    "STATUS_OK",
    "EvidenceStore",
    "InstagramBrowser",
    "InstagramProfileLookup",
    "ProfileFinding",
    "RateGatedInstagramBrowser",
    "extract_profile",
    "unavailable",
]
