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
