"""Whether the business is already buying Meta ads.

Why this is worth a fetch at all: a business running ads has already decided that paid
customer acquisition is a thing it does, has a budget line for it, and has somebody who
approves that spend. Every one of those is a stronger qualification signal than follower
count, which measures how many people once tapped a button. "You are already paying to be
seen and sending that traffic to an Instagram bio" is an opening; "you have 4,000
followers" is not.

WHY THE PUBLIC PAGE AND NOT THE API
-----------------------------------
The Meta Ad Library *API* is restricted to political and social-issue advertising
worldwide, plus all advertising in the UK and EU. Indian commercial ads -- which is the
entire corpus this pipeline cares about -- are not in it. They are on the public Ad Library
web page, which Meta publishes deliberately and without a login for exactly this
transparency purpose. So the path is: build the public URL, fetch it like any other page,
and read the result count off it.

WHAT THE COUNT MEANS, AND WHAT IT DOES NOT
------------------------------------------
The page is a keyword search over advertiser names, so a hit is "an advertiser matching
this name is running ads", not "this exact business is". `runs_ads` is therefore a lead,
not a fact, and the payload carries the query and the count so an operator can check. A
zero is the more reliable direction: no advertiser of that name is running anything.

THREE OUTCOMES, KEPT APART
--------------------------
    ok       + runs_ads True/False   the page answered
    blocked  + runs_ads None         the page answered with a login wall or a JS shell
    error    + runs_ads None         the fetch never produced a page

`blocked` exists as its own status because it is the one that says something about the
FETCHER rather than the business. Folding it into `no_data` would let a week of
anti-automation defence be recorded as a week of businesses that buy no ads, and the
negative cache would then stop asking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

AD_LIBRARY_BASE = "https://www.facebook.com/ads/library/"

#: India. The page requires a country, and an unset one silently searches nothing.
DEFAULT_COUNTRY = "IN"

#: The parameters the public page reads. `active_status=active` rather than `all`, because
#: the question is whether they are buying ads NOW -- a campaign that ended in 2021 is not
#: a budget. `search_type=keyword_unordered` matches the words in any order, which is what
#: a shop name needs when the page has it as "Sweet Corner Bakery Pune".
AD_TYPE = "all"
ACTIVE_STATUS = "active"
SEARCH_TYPE = "keyword_unordered"
MEDIA_TYPE = "all"

#: "~14 results", "14 results", "1 result". The tilde is Meta's own approximation marker.
_RESULT_COUNT = re.compile(r"~?\s*([\d,]+)\s*\+?\s*results?\b", re.I)

#: The page's own words for an empty search. Checked before the count, because the empty
#: page also contains the word "results" in its explanatory copy.
EMPTY_MARKERS = (
    "no ads match your search criteria",
    "no results found",
    "0 results",
    "we couldn't find any ads",
    "we could not find any ads",
    "try changing or removing your filters",
)

#: A page that came back without having answered. A login wall, an interstitial, or the
#: JS shell that a text fetcher gets when the content is rendered client-side.
BLOCKED_MARKERS = (
    "log in to continue",
    "you must log in",
    "log into facebook",
    "please log in",
    "create new account",
    "javascript is required",
    "enable javascript",
    "you're temporarily blocked",
    "sorry, something went wrong",
    "this content isn't available",
    "checkpoint required",
    "security check",
    "are you a robot",
)

#: Below this, whatever came back is a shell rather than a page. The real Ad Library page
#: is tens of kilobytes even when it has nothing to show.
MIN_PAGE_CHARS = 120


def ad_library_url(name: str, *, country: str = DEFAULT_COUNTRY) -> str:
    """The public Ad Library search URL for this business name.

    Built with `urlencode` rather than an f-string: business names in these metros carry
    ampersands, apostrophes and Devanagari, and one unescaped `&` turns the rest of the
    name into somebody else's query parameter.
    """
    query = (name or "").strip()
    if not query:
        raise ValueError("a business name is required to search the ad library")
    params = {
        "active_status": ACTIVE_STATUS,
        "ad_type": AD_TYPE,
        "country": (country or DEFAULT_COUNTRY).upper(),
        "media_type": MEDIA_TYPE,
        "q": query,
        "search_type": SEARCH_TYPE,
    }
    return f"{AD_LIBRARY_BASE}?{urlencode(params)}"


@dataclass(frozen=True)
class AdFinding:
    """What the Ad Library page said, and how much of a statement it is.

    `runs_ads` is tri-state and the None is load-bearing: `models.Evidence.runs_ads` is
    also `bool | None`, and the export prints a blank for None and "No" for False. A
    blocked page that reported False would print "No" beside a business that advertises.
    """

    status: str
    runs_ads: bool | None = None
    ad_count: int | None = None
    url: str | None = None
    country: str = DEFAULT_COUNTRY
    query: str = ""
    provider: str | None = None
    reason: str | None = None

    @property
    def answered(self) -> bool:
        return self.runs_ads is not None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runs_ads": self.runs_ads,
            "ad_count": self.ad_count,
            "country": self.country,
            "query": self.query,
            "provider": self.provider,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def unavailable(
    reason: str,
    *,
    status: str = "error",
    url: str | None = None,
    query: str = "",
    country: str = DEFAULT_COUNTRY,
) -> AdFinding:
    """The page was never read. `runs_ads` stays None -- silence is not a "no"."""
    return AdFinding(status=status, url=url, query=query, country=country, reason=reason)


def read_ad_library(
    text: str,
    *,
    url: str | None = None,
    query: str = "",
    country: str = DEFAULT_COUNTRY,
    provider: str | None = None,
) -> AdFinding:
    """Read a fetched Ad Library page. Pure: it parses text and issues no request.

    Order of checks matters. Blocked wins over everything, because a login wall renders
    plenty of chrome text and some of it contains numbers. Then the explicit "no ads"
    copy, because that page's own explanatory prose contains the word "results". The
    count regex runs last and only on a page that got past both.
    """
    common = {"url": url, "query": query, "country": country, "provider": provider}

    if not isinstance(text, str) or len(text.strip()) < MIN_PAGE_CHARS:
        return AdFinding(
            status="blocked", reason="the ad library returned no readable page", **common
        )

    lowered = text.lower()
    for marker in BLOCKED_MARKERS:
        if marker in lowered:
            return AdFinding(
                status="blocked", reason="the ad library page was gated", **common
            )

    for marker in EMPTY_MARKERS:
        if marker in lowered:
            return AdFinding(status="ok", runs_ads=False, ad_count=0, **common)

    match = _RESULT_COUNT.search(lowered)
    if match:
        count = int(match.group(1).replace(",", ""))
        return AdFinding(status="ok", runs_ads=count > 0, ad_count=count, **common)

    # A page that read fine and said neither. Not a "no": the layout moved, or the search
    # landed somewhere unexpected. `no_data` records that we asked and learned nothing,
    # which the negative cache is allowed to back off on, unlike `blocked`.
    return AdFinding(
        status="no_data", reason="no result count on the ad library page", **common
    )


__all__ = [
    "AD_LIBRARY_BASE",
    "BLOCKED_MARKERS",
    "DEFAULT_COUNTRY",
    "EMPTY_MARKERS",
    "AdFinding",
    "ad_library_url",
    "read_ad_library",
    "unavailable",
]
