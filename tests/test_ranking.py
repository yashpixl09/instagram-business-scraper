"""Ranking tests.

`lead_engine/ranking.py` is one function and one decision: which of two equally scored
leads the operator is sent to first. The decision is encoded as a tuple of four components
in a fixed order, always sorted with `reverse=True`, and the whole of its behaviour is in
which component breaks which tie -- so that is what these tests are about.

They are written against `sorted(..., key=rank_key, reverse=True)` rather than against the
tuple alone wherever the ordering is the claim, because `reverse=True` is half the
contract: the same key applied ascending would put the worst lead at the top, and a tuple
assertion would not notice.

A separate file from `tests/test_automations.py` because it is a separate subject; ranking
knows nothing about automation offers and vice versa.
"""

from __future__ import annotations

import unittest

from lead_engine.models import Lead
from lead_engine.ranking import rank_key
from lead_engine.scoring import ScoreBreakdown, ScoredLead


def lead(name: str) -> Lead:
    return Lead(
        name=name,
        category="bakery",
        address="MG Road, Pune",
        city="Pune",
        latitude=18.52,
        longitude=73.86,
        phone="9123456789",
        website=None,
        source_url="https://example.invalid/source",
        raw_categories=["bakery"],
    )


def scored(
    name: str,
    *,
    total: int,
    reachability: int = 0,
    website_gap: int = 0,
    demand: int = 0,
    budget: int = 0,
    summary: str = "summary",
    outreach: str = "outreach",
) -> ScoredLead:
    """A `ScoredLead` whose components can be set one at a time.

    The components are deliberately not made to add up to `total`. Ranking reads the four
    numbers it is given; making the fixture enforce the scorer's own arithmetic would mean
    no test could isolate a single tiebreaker.
    """
    return ScoredLead(
        lead=lead(name),
        score=ScoreBreakdown(
            demand=demand,
            website_gap=website_gap,
            budget=budget,
            reachability=reachability,
            total=total,
            signals=[],
            pitch_angle="",
        ),
        ai_summary=summary,
        outreach_message=outreach,
    )


def order(items: list[ScoredLead]) -> list[str]:
    """The names in the order the operator would work them."""
    return [item.lead.name for item in sorted(items, key=rank_key, reverse=True)]


class RankKeyShapeTests(unittest.TestCase):
    def test_key_is_the_four_components_in_declared_order(self):
        item = scored("A", total=71, reachability=16, website_gap=25, demand=30, budget=20)
        self.assertEqual(rank_key(item), (71, 16, 25, 30))

    def test_budget_is_not_part_of_the_key(self):
        # Budget reaches the ranking only through `total`. Two leads differing in budget
        # alone but sharing a total rank equal, which is the intended reading: budget is a
        # component of how good the lead is, not of which door to knock on first.
        rich = scored("Rich", total=60, budget=25)
        poor = scored("Poor", total=60, budget=0)
        self.assertEqual(rank_key(rich), rank_key(poor))

    def test_the_prose_fields_are_not_part_of_the_key(self):
        # `ai_summary` and `outreach_message` are generated text. If they moved the order,
        # a rerun of the LLM pass would reshuffle the operator's day.
        one = scored("One", total=50, summary="a long, enthusiastic summary", outreach="x")
        two = scored("Two", total=50, summary="", outreach="")
        self.assertEqual(rank_key(one), rank_key(two))

    def test_key_is_a_plain_tuple_of_ints(self):
        key = rank_key(scored("A", total=1, reachability=2, website_gap=3, demand=4))
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 4)
        for component in key:
            self.assertIsInstance(component, int)


class RankOrderTests(unittest.TestCase):
    def test_total_decides_when_totals_differ(self):
        items = [
            scored("Middle", total=60, reachability=20),
            scored("Best", total=80, reachability=0),
            scored("Worst", total=40, reachability=20),
        ]
        self.assertEqual(order(items), ["Best", "Middle", "Worst"])

    def test_a_higher_total_outranks_every_tiebreaker(self):
        # The tiebreakers are tiebreakers. An unreachable lead one point ahead still leads;
        # otherwise reachability would be a second score rather than a tiebreak.
        unreachable = scored("Unreachable", total=61, reachability=0, website_gap=0, demand=0)
        reachable = scored("Reachable", total=60, reachability=20, website_gap=25, demand=30)
        self.assertEqual(order([reachable, unreachable]), ["Unreachable", "Reachable"])

    def test_reachability_breaks_a_total_tie(self):
        # The one the docstring calls out: between two equally scored leads, the one you can
        # actually phone is worth more than the one you cannot.
        phoneable = scored("Phoneable", total=60, reachability=16, website_gap=8, demand=10)
        silent = scored("Silent", total=60, reachability=6, website_gap=25, demand=30)
        self.assertEqual(order([silent, phoneable]), ["Phoneable", "Silent"])

    def test_website_gap_breaks_a_total_and_reachability_tie(self):
        wide = scored("Wide", total=60, reachability=16, website_gap=25, demand=10)
        narrow = scored("Narrow", total=60, reachability=16, website_gap=8, demand=30)
        self.assertEqual(order([narrow, wide]), ["Wide", "Narrow"])

    def test_demand_breaks_the_last_tie(self):
        high = scored("High", total=60, reachability=16, website_gap=25, demand=30)
        low = scored("Low", total=60, reachability=16, website_gap=25, demand=12)
        self.assertEqual(order([low, high]), ["High", "Low"])

    def test_fully_tied_leads_keep_the_order_they_arrived_in(self):
        # `sorted` is stable and `reverse=True` does not reverse equal elements, so a tie on
        # all four components is not resolved arbitrarily -- it is left alone. Discovery
        # order survives, which is reproducible; a set-like reshuffle would not be.
        items = [scored(name, total=60, reachability=16, website_gap=25, demand=30)
                 for name in ("First", "Second", "Third")]
        self.assertEqual(order(items), ["First", "Second", "Third"])
        self.assertEqual(order(list(reversed(items))), ["Third", "Second", "First"])

    def test_each_component_outranks_every_component_below_it(self):
        # A sweep of the precedence rule. For each component: the two leads are equal on
        # everything above it, the winner is ahead on it and behind on everything below it.
        # Winning anyway is what "this component decides before that one" means.
        fields = ("total", "reachability", "website_gap", "demand")
        shared = {"total": 60, "reachability": 16, "website_gap": 20, "demand": 20}
        high = {"total": 61, "reachability": 20, "website_gap": 25, "demand": 30}
        low = {"total": 59, "reachability": 6, "website_gap": 8, "demand": 12}
        for index, decider in enumerate(fields):
            with self.subTest(component=decider):
                winner = scored("Winner", **{
                    field: shared[field] if position < index
                    else high[field] if position == index
                    else low[field]
                    for position, field in enumerate(fields)
                })
                loser = scored("Loser", **{
                    field: shared[field] if position < index
                    else low[field] if position == index
                    else high[field]
                    for position, field in enumerate(fields)
                })
                self.assertEqual(order([loser, winner]), ["Winner", "Loser"])

    def test_a_zero_scored_lead_ranks_last_but_is_not_dropped(self):
        # Nothing here filters. A lead that scored nothing is still a row the operator can
        # look at; it simply belongs at the bottom.
        zero = scored("Zero", total=0)
        items = [zero, scored("Some", total=30, reachability=6)]
        self.assertEqual(order(items), ["Some", "Zero"])
        self.assertEqual(len(sorted(items, key=rank_key, reverse=True)), 2)

    def test_negative_components_still_order_below_zero_rather_than_being_clamped(self):
        # `rank_key` reads the numbers it is handed. If a future scorer ever emits a negative
        # component, the ranking must not quietly treat it as zero and promote the lead.
        negative = scored("Negative", total=-5, reachability=-1)
        items = [scored("Zero", total=0), negative]
        self.assertEqual(order(items), ["Zero", "Negative"])

    def test_ranking_an_empty_list_and_a_single_lead(self):
        self.assertEqual(order([]), [])
        self.assertEqual(order([scored("Only", total=42)]), ["Only"])

    def test_a_realistic_mixed_cohort_sorts_the_way_the_operator_expects(self):
        cohort = [
            scored("No phone, big gap", total=64, reachability=6, website_gap=25, demand=30),
            scored("Phone, big gap", total=64, reachability=16, website_gap=25, demand=30),
            scored("Phone, has site", total=64, reachability=20, website_gap=8, demand=30),
            scored("Weak site", total=58, reachability=16, website_gap=18, demand=28),
        ]
        self.assertEqual(
            order(cohort),
            ["Phone, has site", "Phone, big gap", "No phone, big gap", "Weak site"],
        )


if __name__ == "__main__":
    unittest.main()
