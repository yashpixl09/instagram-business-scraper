"""Client for the TinyFish Search and Fetch APIs.

Written against the live documentation at https://docs.tinyfish.ai, read on 2026-08-13.
`docs/providers.md` records what those pages say, page by page, so the next person does not
have to re-derive it. The short version:

    GET  https://api.search.tinyfish.ai   ?query=...        header `X-API-Key: <key>`
    POST https://api.fetch.tinyfish.ai    {"urls": [...]}   header `X-API-Key: <key>`

Both are free on every plan and consume zero credits; only the request rate is capped --
30 search queries per minute and 150 fetch URLs per minute on Free and Pay As You Go.
Those caps are the reason this module carries a client-side `TokenBucket`: the cheapest
429 is the one that is never sent.

Two things the docs do not give us, and how this module handles them:

* Search has NO result-count parameter. The reference lists `page` (0..10) and nothing
  that sets a page size. `limit` is therefore applied client-side, as an upper bound on
  one page of results -- never a guarantee of that many. Callers wanting more paginate.
* Per-URL fetch failures come back inside a 200 response, in `errors[]`, each with a
  `status` that reflects the *target site*, not TinyFish. Mapping that status through the
  provider-level table would claim our API key was rejected whenever a scraped site
  answered 403, so every per-URL failure maps to `provider_bad_response` instead. See
  `_fetch_failure`.

No error raised here ever carries the response body, the requested URL, or the API key.
That is a hard rule -- these messages end up in logs and in the sales-sheet run record.
It costs some debuggability: transport errors are re-raised with `from None` because the
underlying httpx exception stringifies the URL, key and all.

Every message here is static prose with at most an HTTP status interpolated, which matters
more than it looks: `errors._safe_message` throws away any message that `redact()` would
touch and substitutes the generic default. A message built from a response body would
therefore not leak -- it would silently vanish, and the run log would say nothing useful.
`test_every_error_message_survives_redaction` pins that none of ours does.

`retryable` is never passed to `ProviderError`. The tables in `errors.py` own it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx

from lead_engine.providers.errors import (
    ProviderError,
    auth_failed,
    bad_response,
    rate_limited,
    unavailable,
)

PROVIDER = "tinyfish"

SEARCH_URL = "https://api.search.tinyfish.ai"
FETCH_URL = "https://api.fetch.tinyfish.ai"

API_KEY_HEADER = "X-API-Key"

# Free and Pay As You Go limits. Starter and Pro are higher (60/300 and 120/600); a caller
# on a paid plan raises them through the constructor rather than editing these.
SEARCHES_PER_MINUTE = 30
FETCH_URLS_PER_MINUTE = 150
RATE_WINDOW_SECONDS = 60.0

# The Fetch API accepts 1..10 URLs per request.
MAX_URLS_PER_FETCH = 10

DEFAULT_TIMEOUT = 30.0

# The only format this client asks for. Fetch also offers "html" and "json", but "json"
# makes `text` an object rather than a string, and the whole point of this path is clean
# text for the extractor, so the choice is fixed rather than exposed.
FETCH_FORMAT = "markdown"


@dataclass(frozen=True)
class SearchResult:
    """One row of `results[]` from the Search API.

    `url` is the only field this client insists on -- a result we cannot fetch is not a
    result. The rest are presentation metadata and default to empty rather than failing a
    whole search over a missing snippet.
    """

    url: str
    title: str = ""
    snippet: str = ""
    site_name: str = ""
    position: int | None = None
    date: str | None = None


class TokenBucket:
    """A token bucket with an injectable clock, so rate limiting is testable in zero time.

    `clock` and `sleep` are separate callables on purpose: a test supplies a fake pair
    where `sleep(n)` advances `clock()` by `n`, and the bucket then behaves exactly as it
    would against `time.monotonic`/`time.sleep` without a real second passing.

    Refill is continuous, which is what "token bucket" means: at 30 per 60s the bucket
    regains a token every two seconds rather than handing back all 30 on a minute
    boundary. A burst of 30 goes out instantly; the 31st waits.
    """

    def __init__(
        self,
        capacity: int,
        window_seconds: float = RATE_WINDOW_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.capacity = capacity
        self.window_seconds = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(capacity)
        self._updated = clock()
        # One bucket, eight enrichment threads. Guards the balance and the watermark
        # together: refilling against a stale watermark hands out tokens time never earned.
        self._lock = threading.Lock()

    @property
    def rate(self) -> float:
        """Tokens regained per second."""
        return self.capacity / self.window_seconds

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        # A clock that went backwards is not a reason to hand out free tokens; leave the
        # watermark where it was and let the next forward tick catch up.
        if elapsed > 0:
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.rate)
            self._updated = now

    def take(self, tokens: int = 1) -> float:
        """Consume `tokens`, sleeping first if the bucket is short. Returns seconds slept.

        THREAD SAFE, and it has to be: `worker-enrich` runs at concurrency 8 against one
        client, so eight threads share one bucket.

        The debt model is what makes that work. Tokens are deducted under the lock
        *unconditionally*, letting the balance go negative, and each caller then sleeps off
        the deficit its own deduction created. So the Nth concurrent caller waits N times as
        long as the first, which is a queue -- and the arithmetic that forms the queue happens
        while the lock is held, while the sleeping happens outside it, so callers wait on the
        rate rather than on each other.

        An earlier version refilled to `float(tokens)` after sleeping. That was an absolute
        assignment, not an increment: two threads sleeping concurrently each restored the
        bucket the other had just spent, minting tokens out of nothing. Measured at 196
        requests per provider-minute against a documented cap of 30 -- which TinyFish answers
        with 429s that stall the whole enrichment stage.

        There is still no retry loop. A deficit of `d` tokens takes `d / rate` seconds, so one
        sleep is always enough and an injected `sleep` that does not really advance time
        cannot spin this method forever.
        """
        if tokens < 1:
            raise ValueError("tokens must be at least 1")
        if tokens > self.capacity:
            raise ValueError(f"cannot take {tokens} tokens from a bucket holding {self.capacity}")

        with self._lock:
            self._refill()
            deficit = tokens - self._tokens
            # Deduct before releasing the lock. A caller that sleeps without having deducted
            # is invisible to everyone else, and every concurrent caller then computes its
            # wait against a balance that nobody has spent yet.
            self._tokens -= tokens
            waited = deficit / self.rate if deficit > 0 else 0.0

        if waited > 0:
            self._sleep(waited)
        return waited


class TinyFishClient:
    """Search and Fetch against TinyFish, rate limited client-side.

    `api_key` is exposed as a plain attribute because the API layer duck-types on it to
    answer 503 when the provider is unconfigured; the client itself never asserts on it,
    so a missing key surfaces as the 401 that TinyFish actually returns.

    `transport` is the seam that keeps tests off the network -- pass an
    `httpx.MockTransport`. `clock` and `sleep` are the seam that keeps them fast.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        search_url: str = SEARCH_URL,
        fetch_url: str = FETCH_URL,
        searches_per_minute: int = SEARCHES_PER_MINUTE,
        fetch_urls_per_minute: int = FETCH_URLS_PER_MINUTE,
    ) -> None:
        self.api_key = api_key
        self.search_url = search_url
        self.fetch_url = fetch_url
        self.search_limiter = TokenBucket(searches_per_minute, clock=clock, sleep=sleep)
        self.fetch_limiter = TokenBucket(fetch_urls_per_minute, clock=clock, sleep=sleep)

        headers = {"Accept": "application/json"}
        if api_key:
            headers[API_KEY_HEADER] = api_key
        self._client = httpx.Client(transport=transport, timeout=timeout, headers=headers)

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TinyFishClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- search -------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        page: int = 0,
        location: str | None = None,
        domain_type: str | None = None,
        include_domains: Iterable[str] | None = None,
        exclude_domains: Iterable[str] | None = None,
        purpose: str | None = None,
    ) -> list[SearchResult]:
        """Run one search query. Costs one token from the 30/min bucket.

        `limit` caps the returned list. TinyFish has no page-size parameter, so this
        truncates rather than asking for fewer -- a `limit` above the page size returns
        whatever the page held.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if page < 0:
            raise ValueError("page must not be negative")

        params: dict[str, str | int] = {"query": query.strip()}
        if page:
            params["page"] = page
        if location:
            params["location"] = location
        if domain_type:
            params["domain_type"] = domain_type
        if include_domains:
            params["include_domains"] = ",".join(include_domains)
        if exclude_domains:
            params["exclude_domains"] = ",".join(exclude_domains)
        if purpose:
            params["purpose"] = purpose

        self.search_limiter.take(1)
        payload = self._request("GET", self.search_url, params=params)

        entries = payload.get("results")
        if not isinstance(entries, list):
            raise _bad_response("TinyFish search response had no results list")

        # Every entry is validated, not just the ones that survive `limit`: a malformed row
        # anywhere means the schema moved under us, and that is worth an error rather than
        # a quietly shorter list.
        results = [_search_result(entry) for entry in entries]
        return results[:limit]

    # -- fetch --------------------------------------------------------------------

    def fetch(self, url: str) -> str:
        """Fetch one URL as markdown. Costs one token from the 150/min bucket.

        Raises rather than returning a placeholder when the page could not be read -- a
        fabricated empty page would be scored as a real one downstream.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")

        payload = self._fetch_payload([url.strip()])
        results = payload["results"]
        if results:
            return _page_text(results[0])
        raise _fetch_failure(payload["errors"])

    def fetch_many(self, urls: Iterable[str]) -> dict[str, str]:
        """Fetch up to 10 URLs in one request, keyed by the URL TinyFish echoes back.

        A URL that failed is absent from the mapping rather than raising, because the
        Fetch API deliberately isolates per-URL failures -- one dead site must not throw
        away the nine pages that came back with it. Transport and provider-level failures
        (auth, rate limit, 5xx) still raise, since none of the batch survived those.
        """
        requested = [url.strip() for url in urls if isinstance(url, str) and url.strip()]
        if not requested:
            raise ValueError("at least one non-empty url is required")
        if len(requested) > MAX_URLS_PER_FETCH:
            raise ValueError(f"fetch accepts at most {MAX_URLS_PER_FETCH} urls per request")

        payload = self._fetch_payload(requested)
        pages: dict[str, str] = {}
        for entry in payload["results"]:
            if not isinstance(entry, dict):
                raise _bad_response("TinyFish fetch response held a malformed result")
            key = entry.get("url")
            if not isinstance(key, str) or not key:
                raise _bad_response("TinyFish fetch result was missing its url")
            pages[key] = _page_text(entry)
        return pages

    def _fetch_payload(self, urls: list[str]) -> dict[str, Any]:
        """Spend the rate-limit tokens, POST, and check the envelope shape."""
        self.fetch_limiter.take(len(urls))
        payload = self._request(
            "POST",
            self.fetch_url,
            json={"urls": urls, "format": FETCH_FORMAT},
        )

        results = payload.get("results")
        errors = payload.get("errors", [])
        if not isinstance(results, list) or not isinstance(errors, list):
            raise _bad_response("TinyFish fetch response was not the documented envelope")
        if not results and not errors:
            raise _bad_response("TinyFish fetch response was empty")
        return {"results": results, "errors": errors}

    # -- transport ----------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, url, params=params, json=json)
        except httpx.TimeoutException:
            # `from None`, per the rule in errors.py: an httpx exception carries
            # `.request.url`, key and all, and tracebacks print the whole chain.
            raise unavailable("TinyFish timed out.", provider=PROVIDER) from None
        except httpx.HTTPError:
            raise unavailable("TinyFish could not be reached.", provider=PROVIDER) from None

        if response.status_code >= 400:
            raise _status_error(response.status_code)

        try:
            payload = response.json()
        except ValueError:
            raise _bad_response("TinyFish returned a body that was not JSON") from None
        if not isinstance(payload, dict):
            raise _bad_response("TinyFish returned a JSON body that was not an object")
        return payload


# -- module-level helpers -------------------------------------------------------------


def _bad_response(message: str) -> ProviderError:
    return bad_response(message, provider=PROVIDER)


def _status_error(status_code: int) -> ProviderError:
    """Map a provider-level HTTP status onto the shared error taxonomy.

    The status code itself is safe to name -- it is neither the key, the body, nor the
    URL -- and it is the single most useful thing to have in a run log.

    `retryable` is never passed. The shared tables in `errors.py` own that decision, and a
    client that hand-typed it would be a second source of truth waiting to disagree.

    DIVERGENCE, deliberate and worth a look: this ladder sends 5xx to
    `provider_unavailable`, whereas `errors.from_status` sends everything that is not 429
    or 401/403 -- 5xx included -- to `provider_bad_response`. Both are retryable, so
    nothing downstream retries differently; the visible difference is the API status, 503
    rather than 502. This module follows its own brief ("5xx -> provider_unavailable")
    because a TinyFish 500 is upstream being down, not upstream sending us bad JSON.
    Reconciling the two is a one-line change here -- see docs/providers.md.
    """
    if status_code == 429:
        return rate_limited(f"TinyFish rate limit reached (HTTP {status_code}).", provider=PROVIDER)
    if status_code in (401, 403):
        return auth_failed(
            f"TinyFish rejected the configured key (HTTP {status_code}).", provider=PROVIDER
        )
    if status_code >= 500:
        return unavailable(f"TinyFish is unavailable (HTTP {status_code}).", provider=PROVIDER)
    # 400, 402, 404, 409, 422 and friends, matching `errors.from_status`.
    return _bad_response(f"TinyFish refused the request (HTTP {status_code}).")


def _fetch_failure(errors: list[Any]) -> ProviderError:
    """Turn an `errors[]` entry into a ProviderError.

    Deliberately does NOT route the entry's `status` through `_status_error`. That status
    describes the site being scraped, so a shop whose website answers 403 would otherwise
    be reported as "TinyFish rejected the API key" and would trip the provider circuit
    breaker over a problem that has nothing to do with TinyFish.
    """
    if not errors:
        return _bad_response("TinyFish fetch returned neither content nor an error")
    return _bad_response("TinyFish could not fetch the requested URL")


def _search_result(entry: Any) -> SearchResult:
    if not isinstance(entry, dict):
        raise _bad_response("TinyFish search response held a malformed result")
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        raise _bad_response("TinyFish search result was missing its url")
    return SearchResult(
        url=url,
        title=_text(entry.get("title")),
        snippet=_text(entry.get("snippet")),
        site_name=_text(entry.get("site_name")),
        position=entry.get("position") if isinstance(entry.get("position"), int) else None,
        date=entry.get("date") if isinstance(entry.get("date"), str) else None,
    )


def _page_text(entry: Any) -> str:
    if not isinstance(entry, dict):
        raise _bad_response("TinyFish fetch response held a malformed result")
    text = entry.get("text")
    if not isinstance(text, str):
        raise _bad_response("TinyFish fetch result carried no markdown text")
    return text


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
