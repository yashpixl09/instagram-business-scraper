from __future__ import annotations

from .scoring import ScoredLead


def rank_key(item: ScoredLead) -> tuple[int, int, int, int]:
    """Canonical ranking tiebreaker. Always applied with `reverse=True`.

    Reachability breaks total-score ties on purpose: between two equally scored leads,
    the one you can actually phone is worth more than the one you cannot.
    """
    score = item.score
    return (score.total, score.reachability, score.website_gap, score.demand)
