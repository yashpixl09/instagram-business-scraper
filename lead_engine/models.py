from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lead:
    name: str
    category: str
    address: str
    city: str
    latitude: float | None
    longitude: float | None
    phone: str | None
    website: str | None
    source_url: str | None
    raw_categories: list[str] = field(default_factory=list)
    provider_id: str | None = None
    matched_niches: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Evidence:
    """Audience signals about a lead, kept off `Lead` on purpose.

    A lead is scored twice: once at discovery on Google data alone, and again after
    Instagram enrichment. Only the second pass knows followers or engagement, so folding
    these fields into `Lead` would mean every discovery-time lead carried a row of Nones
    and every consumer had to guess which pass produced it. Every field is optional
    because each pass supplies a different subset.
    """

    reviews: int | None = None
    rating: float | None = None
    followers: int | None = None
    engagement_rate: float | None = None  # likes / followers
    popular_times_density: float | None = None  # mean busyness, 0-100
    runs_ads: bool | None = None
