"""Everything Google Maps cannot tell you about a business.

Discovery answers "does this place exist and how busy is it". This package answers the three
questions that decide whether it is worth pitching:

    website.py   does it REALLY have no website -- Maps saying so is a claim about Google's
                 data, not about the world -- and if it has one, is it any good
    social.py    what is its Instagram handle. The handle only; the profile is Phase 7, read
                 through the operator's own browser, and nothing here may touch instagram.com
    ads.py       is it already buying Meta ads, which proves budget and marketing intent more
                 directly than any follower count

WHY THIS IS FREE AND DISCOVERY IS NOT
-------------------------------------
SearchAPI costs fifty searches, once, ever. TinyFish Search and Fetch cost nothing on any
plan and are capped by rate alone. So enrichment -- which touches every business several
times -- runs on TinyFish, and Firecrawl (1,000 metered credits a month) is the fallback for
what TinyFish cannot do. That ordering is enforced in `service.py` rather than left to
whoever calls it.

ABSENCE IS A RESULT, AND IT IS PAID FOR ONCE
--------------------------------------------
`cache.py` backs off on misses: 30d, 90d, 180d, then never. A business with no Instagram in
March almost certainly has none in April, and re-asking every month is a bill for
re-learning the same nothing. `status` on every enrichment row keeps `no_data` distinct from
never-having-asked, because only the second is worth spending on again.
"""

from __future__ import annotations

from .ads import AdFinding, ad_library_url, read_ad_library
from .cache import (
    CacheEntry,
    EnrichmentCache,
    InMemoryCache,
    NullCache,
    PostgresCache,
    retry_after_for,
)
from .service import (
    BusinessEnrichment,
    EnrichmentRequest,
    EnrichmentService,
    EnrichmentSummary,
    EvidenceStore,
    WebProvider,
)
from .social import HandleFinding, handle_from_url, handles_in_text, normalise_handle
from .website import SiteGrade, classify_host, grade_site, host_of

__all__ = [
    "AdFinding",
    "BusinessEnrichment",
    "CacheEntry",
    "EnrichmentCache",
    "EnrichmentRequest",
    "EnrichmentService",
    "EnrichmentSummary",
    "EvidenceStore",
    "HandleFinding",
    "InMemoryCache",
    "NullCache",
    "PostgresCache",
    "SiteGrade",
    "WebProvider",
    "ad_library_url",
    "classify_host",
    "grade_site",
    "handle_from_url",
    "handles_in_text",
    "host_of",
    "normalise_handle",
    "read_ad_library",
    "retry_after_for",
]
