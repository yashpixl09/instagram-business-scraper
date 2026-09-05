"""The enrichment pass: three questions per business, on the cheapest provider that answers.

    does this business really have no website     -> website.py
    what is its Instagram handle                  -> social.py
    is it buying Meta ads                         -> ads.py

THE COST MODEL, WHICH DECIDES EVERYTHING ELSE
---------------------------------------------
SearchAPI's 50 searches are one-time and non-renewing, and discovery owns every one of
them. Nothing in this module may touch them. TinyFish Search and Fetch are free on every
plan and consume zero credits -- they are capped by request RATE alone, 30 queries and 150
URLs a minute, and `TinyFishClient` already enforces both client-side. Firecrawl meters:
1,000 credits a month, two per search and one per page.

So the ordering is not a preference, it is the budget: TinyFish first, always, and
Firecrawl only for the request TinyFish could not complete. That ordering is expressed once,
in `_attempt`, rather than being repeated at each of the four call sites -- a fallback
written out four times is a fallback that is wrong in one of them. `usage` counts both
sides, so the fallback rate a run produces is a measurement rather than an estimate.

ONE SEARCH, TWO ANSWERS
-----------------------
The website question and the Instagram question are answered from the SAME result set. A
local business's Instagram profile is usually the first or second hit for `"<name>" <city>`,
so issuing a second query for it would double the rate-limit cost of the whole stage to
learn something already on the page. `_lookup` therefore searches once and hands the
results to both modules.

A NAMED CONTACT RIDES THE SAME FETCH
-------------------------------------
`_grade` fetches a business's own homepage to grade the site, and that text is briefly in
memory before being dropped. `contacts.py` reads it once more, for a named owner/manager/
marketing contact, before it goes -- zero marginal cost, since the page was already paid
for. See that module's docstring for why this is the only one of the design spec's five
contact sources built so far, and why it never writes a "found nobody" row: an extractor
that cannot find a name records nothing, full stop.

STATUS IS NOT A LOG LEVEL
-------------------------
Every row written here carries one of four statuses, and the distinctions are the point:

    ok        the lookup answered and found something
    no_data   the lookup answered and there was nothing -- a fact about the business
    blocked   upstream refused, gated, or served a shell -- a fact about the fetcher
    error     the call did not complete -- a fact about us

Only `no_data` advances the negative-cache backoff. `blocked` and `error` leave it exactly
where it was, because an outage that recorded misses would buy 180 days of not looking at
businesses nobody ever checked, and the backoff would then be hiding the outage instead of
surviving it. "We asked and there is nothing" and "we never got to ask" have to stay
different rows, or the next pass cannot tell which one is worth spending on.

PROVIDERS ARE INJECTED
----------------------
This class constructs no client. It is handed a primary and (optionally) a fallback, both
duck-typed to `.search(query, limit)` and `.fetch(url)`, which is the shape `TinyFishClient`
already has. That is what lets the whole suite run on `httpx.MockTransport` and on recording
fakes, and what lets a deployment point the fallback at Firecrawl without this file knowing
what Firecrawl is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar
from uuid import UUID

from ..providers.errors import ProviderError
from . import ads as ads_module
from . import contacts as contacts_module
from . import social as social_module
from . import website as website_module
from .ads import AdFinding
from .cache import (
    ADS,
    INSTAGRAM,
    WEBSITE,
    EnrichmentCache,
    InMemoryCache,
    utc_now,
)
from .social import HandleFinding
from .website import SiteGrade, WebsiteFinding

T = TypeVar("T")

#: `enrichments.source` values this phase writes. The web question names its answerer,
#: because "TinyFish said so" and "we spent a Firecrawl credit to find out" are different
#: provenance and a later audit of the credit spend needs to tell them apart.
SOURCE_TINYFISH_WEB = "tinyfish_web"
SOURCE_FIRECRAWL_WEB = "firecrawl_web"
SOURCE_INSTAGRAM = "instagram_handle"
SOURCE_ADS = "ad_library"

#: Which slot answered, and therefore which source name the web row carries.
PRIMARY = "primary"
FALLBACK = "fallback"

STATUS_OK = "ok"
STATUS_NO_DATA = "no_data"
STATUS_BLOCKED = "blocked"
STATUS_ERROR = "error"

#: How many search results to look at. TinyFish has no page-size parameter, so this
#: truncates one already-returned page; it never asks for more or costs more.
SEARCH_LIMIT = 10


# --- the seams -----------------------------------------------------------------------------


class WebProvider(Protocol):
    """A search-and-fetch provider. `TinyFishClient` structurally, Firecrawl by adapter.

    Both methods raise `ProviderError` on failure rather than returning a placeholder. An
    empty string standing in for a page that could not be read would be graded as a real
    page with no menu and no contact form, and the operator would open with a criticism of
    a website nobody looked at.
    """

    def search(self, query: str, limit: int = 10) -> Sequence[Any]: ...

    def fetch(self, url: str) -> str: ...


class EvidenceStore(Protocol):
    """`Repository`, narrowed to the two append-only writes this phase performs.

    `insert_enrichment` and, since this contacts slice, `insert_contact`. Both are
    observations with a timestamp and a provenance; this phase never updates `businesses`
    or promotes a contact onto anything else, because reconciling observations is a
    decision for whatever reads them later, not for the thing that made one.
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

    def insert_contact(
        self,
        business_id: UUID,
        name: str,
        source: str,
        *,
        role: str = "unknown",
        phone: str | None = None,
        email: str | None = None,
        source_url: str | None = None,
        confidence: float = 0.5,
    ) -> Any: ...


# --- requests and results --------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentRequest:
    """One business to enrich.

    `listed_website` is what Google claimed on the map card -- usually None, which is the
    claim this phase exists to check. Passing it in lets a business that already has a
    known site be confirmed and graded rather than re-litigated.
    """

    business_id: UUID
    name: str
    city: str | None = None
    listed_website: str | None = None
    listed_handle: str | None = None
    country: str = ads_module.DEFAULT_COUNTRY
    run_id: UUID | None = None


@dataclass(frozen=True)
class BusinessEnrichment:
    """What this pass learned about one business, and what it deliberately did not ask."""

    business_id: UUID
    website: WebsiteFinding | None = None
    social: HandleFinding | None = None
    ads: AdFinding | None = None
    #: question -> why it was not asked (`fresh`, `negative_cache`, ...). Reading this is
    #: how a caller tells "we skipped it on purpose" from "it failed silently".
    skipped: dict[str, str] = field(default_factory=dict)
    rows_written: int = 0

    @property
    def searched(self) -> bool:
        return self.website is not None or self.social is not None


@dataclass
class ProviderUsage:
    """How many calls each slot took, so the fallback rate is measured, not guessed."""

    attempts: int = 0
    primary_calls: int = 0
    primary_failures: int = 0
    fallback_calls: int = 0
    fallback_failures: int = 0
    #: Times the primary failed and there was no fallback configured to try.
    fallback_unavailable: int = 0
    #: Operation name -> {"primary": n, "fallback": n}. Which JOBS need the metered path.
    by_operation: dict[str, dict[str, int]] = field(default_factory=dict)
    #: `ProviderError.code` -> count. Codes only; never a provider message.
    error_codes: dict[str, int] = field(default_factory=dict)

    @property
    def fallback_rate(self) -> float:
        """Share of attempts that had to reach the metered provider. 0.0 is the target."""
        return (self.fallback_calls / self.attempts) if self.attempts else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "primary_calls": self.primary_calls,
            "primary_failures": self.primary_failures,
            "fallback_calls": self.fallback_calls,
            "fallback_failures": self.fallback_failures,
            "fallback_unavailable": self.fallback_unavailable,
            "fallback_rate": round(self.fallback_rate, 4),
            "by_operation": {name: dict(counts) for name, counts in self.by_operation.items()},
            "error_codes": dict(self.error_codes),
        }

    def record(self, operation: str, slot: str, *, failed: bool, code: str | None = None) -> None:
        counts = self.by_operation.setdefault(operation, {PRIMARY: 0, FALLBACK: 0})
        counts[slot] = counts.get(slot, 0) + 1
        if slot == PRIMARY:
            self.primary_calls += 1
            self.primary_failures += 1 if failed else 0
        else:
            self.fallback_calls += 1
            self.fallback_failures += 1 if failed else 0
        if code:
            self.error_codes[code] = self.error_codes.get(code, 0) + 1


@dataclass(frozen=True)
class EnrichmentSummary:
    """The pass, as a caller sees it. Counts first, findings second."""

    businesses: tuple[BusinessEnrichment, ...] = ()
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    verdicts: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    rows_written: int = 0
    handles_found: int = 0
    advertisers_found: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "businesses": len(self.businesses),
            "rows_written": self.rows_written,
            "verdicts": dict(self.verdicts),
            "statuses": dict(self.statuses),
            "skipped": dict(self.skipped),
            "handles_found": self.handles_found,
            "advertisers_found": self.advertisers_found,
            "usage": self.usage.as_dict(),
        }


@dataclass(frozen=True)
class _Answer:
    """One provider call, resolved. `slot` is None when nobody answered."""

    value: Any = None
    slot: str | None = None
    code: str | None = None

    @property
    def ok(self) -> bool:
        return self.slot is not None


# --- the service ------------------------------------------------------------------------------


class EnrichmentService:
    """Runs the three lookups, writes `enrichments`, and honours the negative cache.

    Nothing here is constructed internally. The providers, the store, the cache and the
    clock all arrive through the constructor, which is what makes the whole of this
    testable without a socket, a credit or a real day passing.
    """

    def __init__(
        self,
        primary: WebProvider,
        *,
        fallback: WebProvider | None = None,
        store: EvidenceStore | None = None,
        cache: EnrichmentCache | None = None,
        clock: Callable[[], datetime] = utc_now,
        search_limit: int = SEARCH_LIMIT,
        grade_sites: bool = True,
        primary_name: str = "tinyfish",
        fallback_name: str = "firecrawl",
    ) -> None:
        if primary is None:
            raise ValueError("a primary provider is required")
        self._primary = primary
        self._fallback = fallback
        self._store = store
        # An in-memory cache by default rather than None: a service with no cache at all
        # would re-buy every absence in the corpus on the next pass, and the default should
        # be the safe one even in a script somebody wrote in a hurry.
        self._cache: EnrichmentCache = cache if cache is not None else InMemoryCache()
        self._clock = clock
        self._search_limit = max(1, int(search_limit))
        self._grade_sites = grade_sites
        self._names = {PRIMARY: primary_name, FALLBACK: fallback_name}
        self.usage = ProviderUsage()

    # -- public ---------------------------------------------------------------------------

    def enrich(self, request: EnrichmentRequest) -> BusinessEnrichment:
        """Answer all three questions for one business, skipping whatever the cache owns."""
        now = self._clock()
        skipped: dict[str, str] = {}
        for question in (WEBSITE, INSTAGRAM, ADS):
            reason = self._cache.skip_reason(request.business_id, question, now=now)
            if reason:
                skipped[question] = reason

        rows = 0
        site: WebsiteFinding | None = None
        handle: HandleFinding | None = None
        advert: AdFinding | None = None

        # One search serves both the website question and the handle question, so it is
        # issued when EITHER is due and skipped entirely when neither is.
        page_text: str | None = None
        if WEBSITE not in skipped or INSTAGRAM not in skipped:
            query = website_module.search_query(request.name, request.city)
            limit = self._search_limit
            answer = self._attempt("search", lambda provider: provider.search(query, limit))
            results = list(answer.value or ()) if answer.ok else []

            if WEBSITE not in skipped:
                site, page_text = self._assess_website(request, results, query, answer)
                rows += self._write_website(request, site, answer)
                rows += self._write_contacts(request, page_text, site.url)
                self._remember(request.business_id, WEBSITE, site.status, now)

            if INSTAGRAM not in skipped:
                handle = self._find_handle(request, results, answer, page_text)
                rows += self._write_handle(request, handle, answer)
                self._remember(request.business_id, INSTAGRAM, handle.status, now)

        if ADS not in skipped:
            advert = self._check_ads(request)
            rows += self._write_ads(request, advert)
            self._remember(request.business_id, ADS, advert.status, now)

        return BusinessEnrichment(
            business_id=request.business_id,
            website=site,
            social=handle,
            ads=advert,
            skipped=skipped,
            rows_written=rows,
        )

    def enrich_many(self, requests: Iterable[EnrichmentRequest]) -> EnrichmentSummary:
        """Enrich a cohort and total it up. One business's failure never stops the rest."""
        results: list[BusinessEnrichment] = []
        verdicts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        skips: dict[str, int] = {}
        rows = 0
        handles = 0
        advertisers = 0

        for request in requests:
            outcome = self.enrich(request)
            results.append(outcome)
            rows += outcome.rows_written
            if outcome.website is not None:
                verdicts[outcome.website.verdict] = verdicts.get(outcome.website.verdict, 0) + 1
                _bump(statuses, outcome.website.status)
            if outcome.social is not None:
                _bump(statuses, outcome.social.status)
                handles += 1 if outcome.social.found else 0
            if outcome.ads is not None:
                _bump(statuses, outcome.ads.status)
                advertisers += 1 if outcome.ads.runs_ads else 0
            for reason in outcome.skipped.values():
                _bump(skips, reason)

        return EnrichmentSummary(
            businesses=tuple(results),
            usage=self.usage,
            verdicts=verdicts,
            statuses=statuses,
            skipped=skips,
            rows_written=rows,
            handles_found=handles,
            advertisers_found=advertisers,
        )

    # -- the three lookups -----------------------------------------------------------------

    def _assess_website(
        self,
        request: EnrichmentRequest,
        results: Sequence[Any],
        query: str,
        answer: _Answer,
    ) -> tuple[WebsiteFinding, str | None]:
        """The verdict, and the page text if one was read. The text is never stored.

        It is handed to `social.py` in memory and dropped: a footer link on the business's
        own site is the strongest evidence of its handle there is, and putting 40kB of
        markdown into `enrichments.data` to keep it would make the data plane a page cache.
        """
        if not answer.ok:
            # The search never happened, so there is no claim to make. `unknown`, not
            # `none` -- reporting "no website" here would be inventing a sales signal out
            # of an outage.
            return website_module.unavailable("the web search did not complete", query=query), None

        finding = website_module.assess(
            results,
            name=request.name,
            listed_website=request.listed_website,
            query=query,
        )
        if not self._grade_sites or finding.url is None:
            return finding, None
        if finding.verdict not in (website_module.OWN_SITE, website_module.FREE_BUILDER):
            return finding, None
        return self._grade(finding)

    def _grade(self, finding: WebsiteFinding) -> tuple[WebsiteFinding, str | None]:
        """Fetch the site and grade it. A failed grade never changes the verdict.

        Whether they HAVE a site was settled by the search. Failing to read the page is a
        missing grade, not a missing website, and downgrading the verdict here would hand
        the sheet a "no website" for a business whose homepage merely timed out.
        """
        target = finding.url or ""
        page = self._attempt("fetch_site", lambda provider: provider.fetch(target))
        if not page.ok:
            return _replace_grade(finding, None, None, "the site could not be fetched"), None

        text = str(page.value or "")
        grade = website_module.grade_site(text)
        if grade is None:
            return (
                _replace_grade(
                    finding, None, self._names[page.slot], "the page held too little text to grade"
                ),
                text,
            )
        return _replace_grade(finding, grade, self._names[page.slot], None), text

    def _find_handle(
        self,
        request: EnrichmentRequest,
        results: Sequence[Any],
        answer: _Answer,
        page_text: str | None,
    ) -> HandleFinding:
        if not answer.ok:
            return social_module.unavailable("the web search did not complete")
        return social_module.find_handle(
            results,
            name=request.name,
            site_text=page_text,
            listed_handle=request.listed_handle,
        )

    def _check_ads(self, request: EnrichmentRequest) -> AdFinding:
        url = ads_module.ad_library_url(request.name, country=request.country)
        answer = self._attempt("fetch_ads", lambda provider: provider.fetch(url))
        if not answer.ok:
            return ads_module.unavailable(
                "the ad library page could not be fetched",
                url=url,
                query=request.name,
                country=request.country,
            )
        return ads_module.read_ad_library(
            str(answer.value or ""),
            url=url,
            query=request.name,
            country=request.country,
            provider=self._names[answer.slot],
        )

    # -- provider ordering ------------------------------------------------------------------

    def _attempt(self, operation: str, call: Callable[[WebProvider], T]) -> _Answer:
        """TinyFish first. Firecrawl only if TinyFish could not answer.

        The single place the ordering is expressed. Both outcomes are counted, which is
        what turns "we mostly use the free one" into a number a run can print.

        Only `ProviderError` is caught. Anything else -- a TypeError from an adapter with
        the wrong signature, a KeyboardInterrupt -- is a bug or an operator, and swallowing
        either as "this business has no website" is how a broken deploy produces a
        confident, wrong sheet.
        """
        self.usage.attempts += 1
        try:
            value = call(self._primary)
        except ProviderError as error:
            self.usage.record(operation, PRIMARY, failed=True, code=error.code)
        else:
            self.usage.record(operation, PRIMARY, failed=False)
            return _Answer(value=value, slot=PRIMARY)

        if self._fallback is None:
            self.usage.fallback_unavailable += 1
            return _Answer()

        try:
            value = call(self._fallback)
        except ProviderError as error:
            self.usage.record(operation, FALLBACK, failed=True, code=error.code)
            # The CODE, never the message. `ProviderError` already refuses to keep a
            # message that redaction would alter, but a payload that carried provider
            # prose would be one schema change away from carrying a body.
            return _Answer(code=error.code)
        self.usage.record(operation, FALLBACK, failed=False)
        return _Answer(value=value, slot=FALLBACK)

    # -- writing -------------------------------------------------------------------------

    def _write_website(
        self, request: EnrichmentRequest, finding: WebsiteFinding, answer: _Answer
    ) -> int:
        # The row is named for whoever answered the SEARCH, because the search is what
        # decides the verdict. Which provider read the page for grading is in
        # `data['graded_by']`, so one row never has to stand for two provenances.
        source = SOURCE_FIRECRAWL_WEB if answer.slot == FALLBACK else SOURCE_TINYFISH_WEB
        payload = finding.as_dict()
        payload["provider"] = self._names[answer.slot] if answer.slot else None
        return self._record(
            request, source, finding.status, payload, source_url=finding.url
        )

    def _write_handle(
        self, request: EnrichmentRequest, finding: HandleFinding, answer: _Answer
    ) -> int:
        payload = finding.as_dict()
        payload["provider"] = self._names[answer.slot] if answer.slot else None
        return self._record(
            request, SOURCE_INSTAGRAM, finding.status, payload, source_url=finding.url
        )

    def _write_contacts(
        self, request: EnrichmentRequest, page_text: str | None, site_url: str | None
    ) -> int:
        """Read named contacts out of the homepage text `_grade` already fetched.

        `page_text` is None whenever there was nothing to read it from: no site was found,
        grading is disabled, or the fetch itself failed. In every one of those cases there
        is no page to read a contact off, so this is a no-op rather than a row recording an
        absence -- `contacts` has no status column, and never guesses a `no_data` in place
        of one either.
        """
        if not page_text:
            return 0
        extraction = contacts_module.find_contacts(page_text, source_url=site_url)
        written = 0
        for candidate in extraction.candidates:
            if self._store is None:
                break
            self._store.insert_contact(
                request.business_id,
                candidate.name,
                candidate.source,
                role=candidate.role,
                phone=candidate.phone,
                email=candidate.email,
                source_url=candidate.source_url,
                confidence=candidate.confidence,
            )
            written += 1
        return written

    def _write_ads(self, request: EnrichmentRequest, finding: AdFinding) -> int:
        return self._record(
            request, SOURCE_ADS, finding.status, finding.as_dict(), source_url=finding.url
        )

    def _record(
        self,
        request: EnrichmentRequest,
        source: str,
        status: str,
        data: dict[str, Any],
        *,
        source_url: str | None,
    ) -> int:
        if self._store is None:
            return 0
        self._store.insert_enrichment(
            request.business_id,
            source,
            status,
            data,
            source_url=source_url,
            run_id=request.run_id,
        )
        return 1

    def _remember(self, business_id: UUID, question: str, status: str, now: datetime) -> None:
        """Move the negative cache, but only on an OBSERVATION.

        `ok` resets the ladder. `no_data` advances it. `blocked` and `error` do neither --
        they are statements about the fetcher and about us, and letting them back off the
        retry would make an afternoon of 429s look like a corpus of businesses with no
        Instagram.
        """
        if status == STATUS_OK:
            self._cache.record_success(business_id, question, now=now)
        elif status == STATUS_NO_DATA:
            self._cache.record_miss(business_id, question, now=now)


# --- small helpers ----------------------------------------------------------------------------


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _replace_grade(
    finding: WebsiteFinding,
    grade: SiteGrade | None,
    graded_by: str | None,
    reason: str | None,
) -> WebsiteFinding:
    """`WebsiteFinding` is frozen, so grading produces a new one rather than mutating."""
    return WebsiteFinding(
        status=finding.status,
        verdict=finding.verdict,
        url=finding.url,
        host=finding.host,
        confidence=finding.confidence,
        grade=grade,
        graded_by=graded_by,
        grade_reason=reason,
        candidates=finding.candidates,
        results_seen=finding.results_seen,
        query=finding.query,
        reason=finding.reason,
        listed_website=finding.listed_website,
        extras=dict(finding.extras),
    )


__all__ = [
    "PRIMARY",
    "SEARCH_LIMIT",
    "SOURCE_ADS",
    "SOURCE_FIRECRAWL_WEB",
    "SOURCE_INSTAGRAM",
    "SOURCE_TINYFISH_WEB",
    "STATUS_BLOCKED",
    "STATUS_ERROR",
    "STATUS_NO_DATA",
    "STATUS_OK",
    "BusinessEnrichment",
    "EnrichmentRequest",
    "EnrichmentService",
    "EnrichmentSummary",
    "EvidenceStore",
    "ProviderUsage",
    "WebProvider",
]
