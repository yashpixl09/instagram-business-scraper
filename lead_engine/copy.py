"""Deterministic copy.

These are not a degraded fallback path — they are the guarantee that a run completes with
every LLM provider unavailable. They interpolate validated rows only and invent nothing.
"""

from __future__ import annotations

from .models import Lead
from .niches import NicheProfile
from .scoring import ScoreBreakdown, profiles_for_lead


def build_prompt(
    lead: Lead,
    score: ScoreBreakdown,
    profiles: list[NicheProfile] | None = None,
    location: str | None = None,
) -> str:
    selected = profiles or profiles_for_lead(lead)
    labels = ", ".join(profile.label for profile in selected) or lead.category
    offer = selected[0].offer if selected else score.pitch_angle
    return f"""You are qualifying a local business lead for a website and workflow agency.

Business: {lead.name}
Matched niche: {labels}
Resolved location: {location or lead.city}
Address: {lead.address}
Phone found: {"yes" if lead.phone else "no"}
Website/link: {lead.website or "none"}
Score: {score.total}/100
Observed signals: {", ".join(score.signals)}
Recommended offer: {offer}

Write a concise factual qualification summary. Do not invent reviews, followers, revenue, \
demand, or any fact not shown above.
"""


def build_fallback_summary(
    lead: Lead,
    score: ScoreBreakdown,
    profiles: list[NicheProfile] | None = None,
    location: str | None = None,
) -> str:
    selected = profiles or profiles_for_lead(lead)
    profile = selected[0] if selected else None
    niche = profile.label if profile else lead.category
    offer = profile.offer if profile else score.pitch_angle
    return (
        f"{lead.name} is a {niche} lead in {location or lead.city}. "
        f"Observed signals are {', '.join(score.signals) or 'limited public listing data'}. "
        f"The recommended offer is {offer}."
    )


def build_fallback_outreach(
    lead: Lead,
    score: ScoreBreakdown,
    profiles: list[NicheProfile] | None = None,
    location: str | None = None,
    sender: str = "",
) -> str:
    selected = profiles or profiles_for_lead(lead)
    profile = selected[0] if selected else None
    niche = profile.label.lower() if profile else lead.category
    offer = profile.offer if profile else score.pitch_angle
    body = f"""Hi {lead.name},

I came across your {niche} listing in {location or lead.city}. I help local businesses set \
up practical online systems for {offer}, so customer enquiries are easier to convert and \
manage.

Would you like me to send a short three-point audit based only on your current public \
listing?"""
    if sender:
        return f"{body}\n\nBest,\n{sender}"
    return body
