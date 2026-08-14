"""Confirming -- or refuting -- the "no website" claim, and grading whatever is found.

Google Maps saying a business has no website is a statement about Google's index, not
about the world. Half of the leads this pipeline cares about have a site Google never
learned; the other half have an Instagram page a salesperson would call a website and a
scoring function must not. So the question this module answers is deliberately narrow:

    given one web search for "<name> <city>", is there a page on the open internet that
    THIS BUSINESS OWNS, and if so what kind?

Four answers, and the difference between the last two is the whole point:

    own_site       a domain whose name is theirs. They have a website.
    social_only    Instagram / Facebook / Linktree / wa.me and nothing else.
    free_builder   wixsite, blogspot, sites.google, business.site and friends.
    none           the search ran, and nothing on the page belonged to them.

Plus a fifth that is NOT an answer:

    unknown        the search did not run, or came back refused. We could not tell.

`none` is a sales signal -- "we looked, there is nothing" -- and `unknown` is a gap. They
must never collapse into each other, because only the first is worth putting in front of an
operator and only the second is worth spending another request on. The status field carries
which happened (`no_data` versus `blocked`/`error`) and the verdict carries what it means;
`service.py` refuses to write a negative-cache miss for anything but `no_data`, so an
outage cannot buy 180 days of not looking.

THE TWO GATES
-------------
A result counts as the business's own site only if it passes both:

  1. the host is not a third party -- Justdial, Zomato, a news site and a Google Maps link
     all carry the business's name and none of them are its website;
  2. the identity in the URL matches the business name. For an ordinary domain that is the
     label (`sweetcornerbakery.in`); for a social or builder URL it is the handle or
     subdomain, because `instagram.com` is nobody's name and `sweetcorner.wixsite.com/cakes`
     is somebody's.

Gate 2 is why an acronym domain (`scb.co.in`) is a known false negative. Rather than
loosen the gate -- which would start reporting a food blogger's review as the shop's
website -- unmatched non-third-party results are carried in `candidates` and drop the
finding's confidence to "medium". The claim stays honest and the near-miss stays visible.

GRADING
-------
`grade_site` reads whatever text the fetcher returned and answers three questions a pitch
is built from: is there a menu or catalogue, is there any way to enquire or order, and is
the thing usable on a phone. The third one is the honest limit of the free path: TinyFish
Fetch returns markdown, and `<meta name="viewport">` does not survive the conversion to
markdown, so `mobile_friendly` is None -- *unknown*, not False -- unless HTML was supplied.
The score renormalises over the components it actually has, exactly as
`scoring.audience_index` does, so a markdown-graded site is not punished for a fact nobody
measured.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..scoring import SOCIAL_DOMAINS

# --- verdicts -----------------------------------------------------------------------------

#: They own a real domain.
OWN_SITE = "own_site"
#: Instagram, Facebook, a Linktree, a wa.me link -- a presence, not a website.
SOCIAL_ONLY = "social_only"
#: A free builder's default domain. A website in the sense that a URL resolves.
FREE_BUILDER = "free_builder"
#: The search ran and nothing belonged to them. The sales signal.
NO_SITE = "none"
#: The search did not run, or would not answer. The gap.
UNKNOWN = "unknown"

#: Ordered worst-to-best, which is also the order `scoring.website_gap` rewards.
VERDICTS = (NO_SITE, SOCIAL_ONLY, FREE_BUILDER, OWN_SITE, UNKNOWN)

# --- host kinds ---------------------------------------------------------------------------

KIND_SOCIAL = "social"
KIND_FREE_BUILDER = "free_builder"
KIND_THIRD_PARTY = "third_party"
KIND_OTHER = "other"

#: Extends `scoring.SOCIAL_DOMAINS` rather than replacing it. `scoring` stays the authority
#: on what "social-only" means -- a second list that disagreed would put a business in one
#: band here and a different one in the sheet -- and these are the link-in-bio and
#: second-network hosts that arrived after that constant was written.
EXTRA_SOCIAL_DOMAINS = (
    "linkedin.com",
    "twitter.com",
    "x.com",
    "pinterest.com",
    "youtube.com",
    "youtu.be",
    "threads.net",
    "taplink.cc",
    "beacons.ai",
    "linkin.bio",
    "zaap.bio",
    "t.me",
    "snapchat.com",
)

SOCIAL_HOSTS: tuple[str, ...] = tuple(SOCIAL_DOMAINS) + EXTRA_SOCIAL_DOMAINS

#: The four substrings `scoring.score_lead` tests for. Listed here so that the classifier
#: and the scorer cannot drift apart; `tests/test_enrichment.py` pins that every URL this
#: module calls a free builder also lands in the scorer's "weak/free website" band.
SCORING_FREE_BUILDER_MARKERS = ("wixsite", "blogspot", "sites.google", "business.site")

FREE_BUILDER_DOMAINS = (
    "wixsite.com",
    "wix.com",
    "blogspot.com",
    "blogspot.in",
    "sites.google.com",
    "business.site",
    "weebly.com",
    "webnode.com",
    "webnode.page",
    "godaddysites.com",
    "wordpress.com",
    "tumblr.com",
    "carrd.co",
    "strikingly.com",
    "mystrikingly.com",
    "square.site",
    "myshopify.com",
    "mydukaan.io",
    "dukaan.app",
    "webflow.io",
    "netlify.app",
    "vercel.app",
    "github.io",
    "jimdosite.com",
    "yolasite.com",
    "site123.me",
    "simdif.com",
    "ucraft.site",
    "zohosites.in",
    "instamojo.com",
    "glideapp.io",
)

#: Hosts that carry a business's name without being its website: directories,
#: marketplaces, aggregators, registries, news. A result here is evidence the business
#: EXISTS and no evidence at all that it has a site, which is exactly the confusion this
#: list prevents.
THIRD_PARTY_DOMAINS = (
    # Indian local directories and B2B marketplaces
    "justdial.com",
    "indiamart.com",
    "sulekha.com",
    "tradeindia.com",
    "exportersindia.com",
    "quikr.com",
    "olx.in",
    "yellowpages.in",
    "asklaila.com",
    "grotal.com",
    "connect2india.com",
    # food, travel and hospitality aggregators
    "zomato.com",
    "swiggy.com",
    "dineout.co.in",
    "eazydiner.com",
    "magicpin.in",
    "tripadvisor.com",
    "tripadvisor.in",
    "yelp.com",
    "bookmyshow.com",
    "makemytrip.com",
    "goibibo.com",
    "booking.com",
    "agoda.com",
    "airbnb.co.in",
    # services, health, weddings
    "practo.com",
    "urbancompany.com",
    "urbanclap.com",
    "lybrate.com",
    "credihealth.com",
    "nearbuy.com",
    "wedmegood.com",
    "weddingwire.in",
    "shaadisaga.com",
    # property
    "99acres.com",
    "magicbricks.com",
    "housing.com",
    "nobroker.in",
    "commonfloor.com",
    # registries and employer review sites
    "zaubacorp.com",
    "tofler.in",
    "thecompanycheck.com",
    "indiafilings.com",
    "glassdoor.co.in",
    "ambitionbox.com",
    "mouthshut.com",
    # retail marketplaces and general web properties
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "meesho.com",
    "snapdeal.com",
    "google.com",
    "goo.gl",
    "wikipedia.org",
    "medium.com",
    "quora.com",
    "reddit.com",
    "foursquare.com",
    "mapquest.com",
    "trustpilot.com",
    "bing.com",
    "yahoo.com",
    "archive.org",
    "scribd.com",
    "slideshare.net",
    # news and media
    "indiatimes.com",
    "hindustantimes.com",
    "thehindu.com",
    "ndtv.com",
    "news18.com",
    "deccanherald.com",
    "indianexpress.com",
    "livemint.com",
    "telegraphindia.com",
    "firstpost.com",
    "yourstory.com",
)

#: Trailing labels that are a public suffix rather than anybody's name. Not the full PSL --
#: this only has to reduce `thecakestory.co.in` to `thecakestory`, and a wrong answer costs
#: a slightly noisier token set rather than a wrong verdict.
TLD_LABELS = frozenset(
    {
        "com",
        "co",
        "in",
        "net",
        "org",
        "io",
        "ai",
        "biz",
        "info",
        "shop",
        "store",
        "site",
        "online",
        "app",
        "dev",
        "me",
        "us",
        "uk",
        "xyz",
        "live",
        "cafe",
        "restaurant",
        "company",
        "services",
        "tech",
        "digital",
        "agency",
        "studio",
        "pro",
        "edu",
        "gov",
        "ac",
        "web",
        "page",
        "link",
        "bio",
    }
)

#: Words that appear in half the business names in a metro and identify nobody. Dropped
#: before matching so "Sweet Corner Bakery" is matched on "sweet corner" -- otherwise
#: `bakerywala.in` would answer for every bakery in the city.
GENERIC_NAME_TOKENS = frozenset(
    {
        "the",
        "and",
        "of",
        "for",
        "at",
        "by",
        "pvt",
        "private",
        "ltd",
        "limited",
        "llp",
        "inc",
        "co",
        "company",
        "india",
        "indian",
        "bakery",
        "bakers",
        "cafe",
        "coffee",
        "restaurant",
        "kitchen",
        "salon",
        "spa",
        "studio",
        "clinic",
        "dental",
        "hospital",
        "store",
        "shop",
        "stores",
        "boutique",
        "gym",
        "fitness",
        "hotel",
        "sweets",
        "foods",
        "food",
        "services",
        "service",
        "solutions",
        "enterprises",
        "traders",
        "centre",
        "center",
        "house",
        "point",
        "world",
        "hub",
        "zone",
        "best",
        "new",
        "official",
        "home",
        "online",
    }
)

#: How much of a business name has to show up in a URL's identity before that URL is
#: called theirs. Three-token names need two, two-token names need both.
NAME_MATCH_COVERAGE = 0.6

_TOKEN = re.compile(r"[a-z0-9]+")


# --- grading ------------------------------------------------------------------------------

#: What a local business's site is graded on, and what each is worth. `mobile_friendly` is
#: the heaviest because in these metros the traffic is a phone, and it is also the one the
#: free path cannot see -- see `grade_site`.
GRADE_WEIGHTS: dict[str, float] = {
    "mobile_friendly": 0.30,
    "catalogue": 0.30,
    "enquiry": 0.25,
    "ordering": 0.15,
}

#: Below this much text the page is not a page -- a JS shell, a redirect stub, a cookie
#: wall. Grading it would report "no menu, no contact form" about a document nobody read.
MIN_GRADEABLE_CHARS = 200

CATALOGUE_MARKERS = (
    "menu",
    "our menu",
    "catalogue",
    "catalog",
    "price list",
    "pricelist",
    "our products",
    "products",
    "our services",
    "packages",
    "collections",
    "shop now",
)

ENQUIRY_MARKERS = (
    "contact us",
    "contact",
    "enquiry",
    "enquire",
    "inquiry",
    "get in touch",
    "book now",
    "book a table",
    "appointment",
    "request a quote",
    "call us",
    "mailto:",
    "tel:",
    "wa.me",
    "whatsapp",
)

ORDERING_MARKERS = (
    "order online",
    "order now",
    "add to cart",
    "buy now",
    "checkout",
    "shopping cart",
    "place your order",
    "pay now",
    "razorpay",
    "payu",
    "swiggy",
    "zomato",
)

#: A viewport declaration, or a media query in inline CSS. Both are HTML-only facts.
_VIEWPORT = re.compile(r"""<meta[^>]+name=["']?viewport["']?[^>]*>""", re.I)
_DEVICE_WIDTH = re.compile(r"width\s*=\s*device-width", re.I)
_MEDIA_QUERY = re.compile(r"@media[^{]*\(\s*(max|min)-width", re.I)
_LOOKS_LIKE_HTML = re.compile(r"<\s*(!doctype|html|head|meta|body|div)\b", re.I)


def looks_like_html(text: str) -> bool:
    """Whether this document still has the markup that carries a viewport declaration."""
    return bool(_LOOKS_LIKE_HTML.search(text))


@dataclass(frozen=True)
class SiteGrade:
    """What the site does for the business, and what it does not. The pitch material.

    `mobile_friendly` is a tri-state on purpose. False means "this page declares no
    viewport", which is a thing to say to an owner; None means "we read markdown and
    markdown has no `<head>`", which is a thing to say to nobody.
    """

    mobile_friendly: bool | None
    catalogue: bool
    enquiry: bool
    ordering: bool
    score: int
    signals: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mobile_friendly": self.mobile_friendly,
            "catalogue": self.catalogue,
            "enquiry": self.enquiry,
            "ordering": self.ordering,
            "score": self.score,
            "signals": list(self.signals),
            "gaps": list(self.gaps),
        }


def grade_site(text: str, *, html: bool | None = None) -> SiteGrade | None:
    """Grade a fetched page. None when there is not enough of it to grade honestly.

    `html` overrides the sniffer. Left unset, a document carrying `<meta>`/`<html>` is
    treated as HTML and its viewport is read; anything else is markdown, where
    `mobile_friendly` stays None because the tag it lives in was thrown away upstream --
    `TinyFishClient` fixes `format=markdown` and does not expose the HTML format.
    """
    if not isinstance(text, str) or len(text.strip()) < MIN_GRADEABLE_CHARS:
        return None

    is_html = looks_like_html(text) if html is None else bool(html)
    lowered = text.lower()

    mobile: bool | None = None
    if is_html:
        viewport = _VIEWPORT.search(text)
        mobile = bool(
            (viewport and _DEVICE_WIDTH.search(viewport.group(0)))
            or _MEDIA_QUERY.search(text)
        )

    catalogue = _any_marker(lowered, CATALOGUE_MARKERS)
    enquiry = _any_marker(lowered, ENQUIRY_MARKERS)
    ordering = _any_marker(lowered, ORDERING_MARKERS)

    present: dict[str, bool] = {
        "catalogue": catalogue,
        "enquiry": enquiry,
        "ordering": ordering,
    }
    if mobile is not None:
        present["mobile_friendly"] = mobile

    # Renormalise over what was actually measured, exactly as `scoring.audience_index`
    # does. Otherwise every markdown-graded site loses 30 points to a question nobody
    # asked, and a free-path grade could never be compared with a Firecrawl one.
    total_weight = sum(GRADE_WEIGHTS[name] for name in present)
    earned = sum(GRADE_WEIGHTS[name] for name, value in present.items() if value)
    score = int(round(100 * earned / total_weight)) if total_weight else 0

    signals = tuple(name for name, value in present.items() if value)
    gaps = tuple(name for name, value in present.items() if not value)
    return SiteGrade(
        mobile_friendly=mobile,
        catalogue=catalogue,
        enquiry=enquiry,
        ordering=ordering,
        score=score,
        signals=signals,
        gaps=gaps,
    )


def _any_marker(lowered: str, markers: Sequence[str]) -> bool:
    return any(marker in lowered for marker in markers)


# --- URL classification -------------------------------------------------------------------


def host_of(url: str | None) -> str:
    """The lowercased host, tolerating the scheme-less URLs Google hands out."""
    if not url:
        return ""
    candidate = url if "://" in url else "https://" + url
    return urlparse(candidate).netloc.lower().split("@")[-1].split(":")[0]


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _first_match(host: str, domains: Sequence[str]) -> str | None:
    for domain in domains:
        if _matches_domain(host, domain):
            return domain
    return None


def classify_host(url: str | None) -> str:
    """Which bucket this URL's host falls into. Social wins, then builder, then third party.

    The order is `scoring.score_lead`'s: it tests `is_social_only_url` before the
    free-builder markers, so a hypothetical `something.blogspot.com` link posted on
    Instagram is social, not weak. Keeping the order identical is what keeps the two
    modules from banding the same URL differently.
    """
    host = host_of(url)
    if not host:
        return KIND_OTHER
    if _first_match(host, SOCIAL_HOSTS):
        return KIND_SOCIAL
    if _first_match(host, FREE_BUILDER_DOMAINS):
        return KIND_FREE_BUILDER
    if any(marker in host for marker in SCORING_FREE_BUILDER_MARKERS):
        return KIND_FREE_BUILDER
    if _first_match(host, THIRD_PARTY_DOMAINS):
        return KIND_THIRD_PARTY
    return KIND_OTHER


def domain_label(host: str) -> str:
    """The part of a host that is somebody's name: `www.thecakestory.co.in` -> `thecakestory`."""
    labels = [label for label in host.split(".") if label]
    while labels and labels[0] in ("www", "m", "mobile"):
        labels.pop(0)
    while len(labels) > 1 and labels[-1] in TLD_LABELS:
        labels.pop()
    return ".".join(labels)


def identity_of(url: str | None, kind: str | None = None) -> str:
    """The text in a URL that could be the business's name, and nothing else.

    For an ordinary domain that is the label. For a social or builder URL the registrable
    domain belongs to the platform, so the identity is whatever is left of the host plus
    the first path segment -- `instagram.com/sweetcornerbakery`,
    `sweetcorner.wixsite.com/cakes`.

    Third-party hosts deliberately do NOT contribute their path. `justdial.com` carries
    `/Pune/Sweet-Corner-Bakery` for a business that has no website at all, and a matcher
    that read it would report the directory listing as the shop's site.
    """
    host = host_of(url)
    if not host:
        return ""
    kind = classify_host(url) if kind is None else kind

    if kind in (KIND_SOCIAL, KIND_FREE_BUILDER):
        platform = _first_match(host, SOCIAL_HOSTS) or _first_match(host, FREE_BUILDER_DOMAINS)
        remainder = host[: -len(platform) - 1] if platform and host != platform else ""
        candidate = url if "://" in (url or "") else "https://" + (url or "")
        segments = [part for part in urlparse(candidate).path.split("/") if part][:1]
        return " ".join([remainder, *segments]).strip()

    if kind == KIND_THIRD_PARTY:
        return ""
    return domain_label(host)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def significant_tokens(name: str) -> list[str]:
    """A business name reduced to the words that identify it, falling back to all of them."""
    tokens = _tokens(name)
    strong = [token for token in tokens if len(token) >= 3 and token not in GENERIC_NAME_TOKENS]
    return strong or tokens


def name_matches(identity: str, name: str) -> bool:
    """Whether `identity` is plausibly this business's own name.

    Coverage plus a longest-token rule. Coverage alone lets a three-letter fragment of a
    generic word carry a match ("art" inside "smartsolutions"); requiring the name's
    longest distinguishing word to appear as well costs nothing on real matches and
    removes that whole class of accident.
    """
    wanted = significant_tokens(name)
    if not wanted:
        return False
    haystack = "".join(_tokens(identity))
    if not haystack:
        return False
    hits = [token for token in wanted if token in haystack]
    needed = max(1, math.ceil(len(wanted) * NAME_MATCH_COVERAGE))
    if len(hits) < needed:
        return False
    longest = max(wanted, key=len)
    return longest in haystack


# --- the finding --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One search result, placed. Carried on the finding so a `none` verdict is auditable."""

    url: str
    host: str
    kind: str
    matched: bool
    title: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "host": self.host,
            "kind": self.kind,
            "matched": self.matched,
            "title": self.title,
        }


@dataclass(frozen=True)
class WebsiteFinding:
    """What one search said about whether this business has a website.

    `status` is the enrichment row's status and answers "did the lookup work". `verdict`
    answers "what is true about the business". The pair is what keeps "they have no
    website" (`no_data` + `none`) apart from "we could not tell" (`error`/`blocked` +
    `unknown`), and only the first of those is a thing to say to a prospect.
    """

    status: str
    verdict: str
    url: str | None = None
    host: str | None = None
    confidence: str = "high"
    grade: SiteGrade | None = None
    graded_by: str | None = None
    grade_reason: str | None = None
    candidates: tuple[Candidate, ...] = ()
    results_seen: int = 0
    query: str = ""
    reason: str | None = None
    listed_website: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def has_site(self) -> bool:
        """True only for a real own-domain site. A Linktree is not a website."""
        return self.verdict == OWN_SITE

    @property
    def siteless(self) -> bool:
        """The sales signal: we looked, and there is nothing they own."""
        return self.verdict == NO_SITE

    @property
    def told(self) -> bool:
        """Whether the lookup produced a statement about the world at all."""
        return self.verdict != UNKNOWN

    @property
    def contradicts_listing(self) -> bool:
        """Google said no website and the open web disagrees. Worth surfacing on its own."""
        return not self.listed_website and self.verdict in (OWN_SITE, FREE_BUILDER)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "url": self.url,
            "host": self.host,
            "query": self.query,
            "results_seen": self.results_seen,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "listed_website": self.listed_website,
            "contradicts_listing": self.contradicts_listing,
            "grade": self.grade.as_dict() if self.grade else None,
            "graded_by": self.graded_by,
            "grade_reason": self.grade_reason,
        }
        if self.reason:
            payload["reason"] = self.reason
        payload.update(self.extras)
        return payload


def search_query(name: str, city: str | None) -> str:
    """The one query that serves both this module and `social.py`.

    Quoted, because an unquoted multi-word name matches every business in the city with
    one word in common, and this search is the only one either module gets.
    """
    name = (name or "").strip()
    city = (city or "").strip()
    return f'"{name}" {city}'.strip()


def unavailable(reason: str, *, status: str = "error", query: str = "") -> WebsiteFinding:
    """We could not tell. Never `none`, which would be a claim we did not earn."""
    return WebsiteFinding(status=status, verdict=UNKNOWN, query=query, reason=reason)


def assess(
    results: Sequence[Any],
    *,
    name: str,
    listed_website: str | None = None,
    query: str = "",
) -> WebsiteFinding:
    """Turn one page of search results into a verdict. Pure -- it issues no requests.

    `results` is anything with `.url` and `.title`, which is `tinyfish.SearchResult` and
    also whatever a Firecrawl adapter returns. `listed_website` is what Google claimed, if
    anything; it is folded in as a candidate so that a listing which already names a site
    is confirmed rather than re-litigated.
    """
    candidates: list[Candidate] = []
    seen: set[str] = set()

    def consider(url: str | None, title: str = "") -> None:
        if not url or not isinstance(url, str):
            return
        host = host_of(url)
        if not host or url in seen:
            return
        seen.add(url)
        kind = classify_host(url)
        candidates.append(
            Candidate(
                url=url,
                host=host,
                kind=kind,
                matched=name_matches(identity_of(url, kind), name),
                title=title if isinstance(title, str) else "",
            )
        )

    consider(listed_website, "listed on the map card")
    for result in results or ():
        consider(getattr(result, "url", None), getattr(result, "title", "") or "")

    results_seen = len(results or ())
    owned = [c for c in candidates if c.kind == KIND_OTHER and c.matched]
    builders = [c for c in candidates if c.kind == KIND_FREE_BUILDER and c.matched]
    socials = [c for c in candidates if c.kind == KIND_SOCIAL and c.matched]
    # A non-third-party host we could not tie to the name. Not enough to claim, too much
    # to ignore -- it is what turns a "high" confidence `none` into a "medium" one.
    near_misses = [c for c in candidates if c.kind == KIND_OTHER and not c.matched]

    common = {
        "candidates": tuple(candidates),
        "results_seen": results_seen,
        "query": query,
        "listed_website": listed_website,
    }

    if owned:
        best = owned[0]
        return WebsiteFinding(
            status="ok", verdict=OWN_SITE, url=best.url, host=best.host, **common
        )
    if builders:
        best = builders[0]
        return WebsiteFinding(
            status="ok", verdict=FREE_BUILDER, url=best.url, host=best.host, **common
        )
    if socials:
        best = socials[0]
        return WebsiteFinding(
            status="ok", verdict=SOCIAL_ONLY, url=best.url, host=best.host, **common
        )

    # Nothing they own. `no_data` rather than `ok`, because the lookup produced no data
    # about a website -- and `none` rather than `unknown`, because the search DID run and
    # its silence is the finding.
    return WebsiteFinding(
        status="no_data",
        verdict=NO_SITE,
        confidence="medium" if near_misses else "high",
        reason="unmatched non-directory results present" if near_misses else None,
        **common,
    )
