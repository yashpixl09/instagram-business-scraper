from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import Lead


def dedupe_leads(leads: list[Lead]) -> list[Lead]:
    merged: dict[str, Lead] = {}
    order: list[str] = []
    for lead in leads:
        key = dedupe_key(lead)
        if key not in merged:
            merged[key] = lead
            order.append(key)
        else:
            merged[key] = merge_leads(merged[key], lead)
    return [merged[key] for key in order]


def dedupe_key(lead: Lead) -> str:
    if lead.provider_id:
        return "provider|" + normalize(lead.provider_id)
    name = normalize(lead.name)
    contact = normalize(lead.phone or website_host(lead.website) or "")
    if contact:
        return f"contact|{name}|{contact}"
    address = normalize(lead.address)
    if address:
        return f"address|{name}|{address}"
    coordinates = ""
    if lead.latitude is not None and lead.longitude is not None:
        coordinates = f"{lead.latitude:.5f}|{lead.longitude:.5f}"
    return f"location|{name}|{coordinates}"


def merge_leads(first: Lead, second: Lead) -> Lead:
    categories = list(dict.fromkeys([*first.raw_categories, *second.raw_categories]))
    matched_niches = list(dict.fromkeys([*first.matched_niches, *second.matched_niches]))
    richer = second if richness(second) > richness(first) else first
    other = first if richer is second else second
    return Lead(
        name=richer.name or other.name,
        category=richer.category or other.category,
        address=richer.address or other.address,
        city=richer.city or other.city,
        latitude=richer.latitude if richer.latitude is not None else other.latitude,
        longitude=richer.longitude if richer.longitude is not None else other.longitude,
        phone=richer.phone or other.phone,
        website=richer.website or other.website,
        source_url=richer.source_url or other.source_url,
        raw_categories=categories,
        provider_id=richer.provider_id or other.provider_id,
        matched_niches=matched_niches,
    )


def richness(lead: Lead) -> int:
    score = 0
    for value in (lead.address, lead.phone, lead.website, lead.source_url, lead.provider_id):
        if value:
            score += 1
    if lead.latitude is not None and lead.longitude is not None:
        score += 1
    return score


def website_host(value: str | None) -> str:
    if not value:
        return ""
    return urlparse(value if "://" in value else "https://" + value).netloc.lower()


def normalize(value: object) -> str:
    # str() is load-bearing: providers return numeric phone values.
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
