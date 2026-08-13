from __future__ import annotations

from dataclasses import dataclass
from math import log10
from urllib.parse import urlparse

from .models import Evidence, Lead
from .niches import NICHE_PROFILES, NicheProfile, matches_niche

SOCIAL_DOMAINS = (
    "instagram.com",
    "facebook.com",
    "fb.com",
    "wa.me",
    "whatsapp.com",
    "linktr.ee",
    "bit.ly",
)


@dataclass(frozen=True)
class ScoreBreakdown:
    demand: int
    website_gap: int
    budget: int
    reachability: int
    total: int
    signals: list[str]
    pitch_angle: str


@dataclass(frozen=True)
class ScoredLead:
    lead: Lead
    score: ScoreBreakdown
    ai_summary: str
    outreach_message: str


def is_social_only_url(url: str | None) -> bool:
    if not url:
        return False
    candidate = url if "://" in url else "https://" + url
    host = urlparse(candidate).netloc.lower()
    return any(host == domain or host.endswith("." + domain) for domain in SOCIAL_DOMAINS)


def profiles_for_lead(lead: Lead) -> list[NicheProfile]:
    """Fallback profile resolution.

    Prefer passing `profiles` explicitly to `score_lead`. This function's second branch
    scans the whole registry by category and is therefore taxonomy-coupled; the search
    service always knows which niche matched and should say so.
    """
    profiles = [
        NICHE_PROFILES[niche_id]
        for niche_id in lead.matched_niches
        if niche_id in NICHE_PROFILES
    ]
    if profiles:
        return profiles
    return [profile for profile in NICHE_PROFILES.values() if matches_niche(profile, lead)]


def score_lead(
    lead: Lead,
    profiles: list[NicheProfile] | tuple[NicheProfile, ...] | None = None,
    evidence: Evidence | None = None,
) -> ScoreBreakdown:
    """Score a lead from its public listing.

    `evidence` is accepted so that discovery-pass and enrichment-pass callers share one
    signature, but the band arithmetic below deliberately ignores it: banding against the
    rest of a run's cohort is a percentile question, and it is answered in SQL over the
    stored rows, not here over one lead in isolation. Use `audience_index` to turn
    evidence into a comparable number.
    """
    selected = list(profiles or profiles_for_lead(lead))
    signals: list[str] = []

    if selected:
        demand = max(profile.demand_base for profile in selected)
        budget = max(profile.budget_base for profile in selected)
        for profile in selected:
            signals.append(f"{profile.label.lower()} match")
        demand += 4
        if len(selected) > 1:
            demand += 2
            signals.append("multiple service fit")
    else:
        demand = 12
        budget = 10

    if len(lead.raw_categories) > 1:
        demand += 2
        budget += 2
        signals.append("detailed business listing")
    demand = min(demand, 30)

    if not lead.website:
        website_gap = 25
        signals.append("no website")
    elif is_social_only_url(lead.website):
        website_gap = 23
        budget += 2
        signals.append("social-only")
    elif any(
        marker in lead.website.lower()
        for marker in ("wixsite", "blogspot", "sites.google", "business.site")
    ):
        website_gap = 18
        signals.append("weak/free website")
    else:
        website_gap = 8
        signals.append("has website")

    if lead.phone:
        budget += 2
    if lead.source_url:
        budget += 1
    budget = min(budget, 25)

    reachability = 6
    if lead.phone:
        reachability += 10
        signals.append("public phone")
    if lead.website:
        reachability += 4
        signals.append("public web link")
    reachability = min(reachability, 20)

    pitch_angle = build_pitch_angle(lead, signals, selected)
    total = demand + website_gap + budget + reachability
    return ScoreBreakdown(
        demand=demand,
        website_gap=website_gap,
        budget=budget,
        reachability=reachability,
        total=total,
        signals=list(dict.fromkeys(signals)),
        pitch_angle=pitch_angle,
    )


def build_pitch_angle(
    lead: Lead,
    signals: list[str],
    profiles: list[NicheProfile] | tuple[NicheProfile, ...] | None = None,
) -> str:
    selected = list(profiles or profiles_for_lead(lead))
    profile = selected[0] if selected else None
    if profile and profile.id in {"bakery", "cake_shop"} and (
        "no website" in signals or "social-only" in signals
    ):
        return "Pitch a catalog + advance payment order system for structured custom orders."
    offer = profile.offer if profile else "clear services, lead capture, and automated follow-up"
    if "no website" in signals or "social-only" in signals:
        return f"Pitch {offer} through a focused mobile-first website."
    if "weak/free website" in signals:
        return f"Pitch replacing the weak website with {offer}."
    return f"Pitch a conversion upgrade focused on {offer}."


AUDIENCE_WEIGHTS = {
    "reviews": 0.40,
    "followers": 0.30,
    "engagement_rate": 0.20,
    "popular_times_density": 0.10,
}


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def audience_index(evidence: Evidence) -> float | None:
    """How much real audience a business has, as a number in [0.0, 1.0].

    Reviews and followers are log-transformed because both distributions are heavy-tailed:
    the gap between 10 reviews and 100 says far more about a neighbourhood business than
    the gap between 1,000 and 1,090, and a linear scale would let one viral outlier flatten
    every genuine local lead to zero. The divisors set the ceiling -- 10,000 reviews or
    100,000 followers reads as a full mark.

    Missing components renormalise the weights over the components actually present, which
    is the whole point of the function. A discovery pass knows reviews and popular times
    and nothing else; without renormalisation it would top out at 0.5 and every
    Google-only lead would look weaker than an enriched one purely because it was measured
    earlier. Returns None when nothing at all is known, so callers can tell "no evidence"
    apart from "evidence of no audience".

    `rating` and `runs_ads` carry no weight: rating barely varies across local listings
    (nearly everything sits at 4.x) and `runs_ads` is a qualitative flag, not a magnitude.
    Both stay on `Evidence` for use elsewhere.
    """
    components: list[tuple[float, float]] = []

    if evidence.reviews is not None:
        value = log10(1 + max(0, evidence.reviews)) / 4.0
        components.append((AUDIENCE_WEIGHTS["reviews"], _unit(value)))
    if evidence.followers is not None:
        value = log10(1 + max(0, evidence.followers)) / 5.0
        components.append((AUDIENCE_WEIGHTS["followers"], _unit(value)))
    if evidence.engagement_rate is not None:
        value = evidence.engagement_rate / 0.10
        components.append((AUDIENCE_WEIGHTS["engagement_rate"], _unit(value)))
    if evidence.popular_times_density is not None:
        value = evidence.popular_times_density / 100.0
        components.append((AUDIENCE_WEIGHTS["popular_times_density"], _unit(value)))

    if not components:
        return None

    total_weight = sum(weight for weight, _ in components)
    return sum(weight * value for weight, value in components) / total_weight
