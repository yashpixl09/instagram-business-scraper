"""Scoring tests.

`ScoringTests` is ported from the prototype's `tests/test_scoring.py`; only the import
paths and the shared factory moved. The three cases it carries are the product thesis in
test form: copy that is specific to the matched niche, a social-only business outranking a
website-ready one, and a no-website lead getting a concrete pitch angle.

`AudienceIndexTests` covers the second scoring pass, which runs over Instagram-enriched
evidence rather than over the listing.
"""

from __future__ import annotations

import unittest

from lead_engine.copy import build_fallback_outreach, build_fallback_summary
from lead_engine.models import Evidence, Lead
from lead_engine.niches import NICHE_PROFILES, matches_niche
from lead_engine.scoring import audience_index, score_lead
from tests.factories import qualifying_lead


class FactoryTests(unittest.TestCase):
    def test_factory_builds_a_lead_that_actually_qualifies(self):
        # The scoring tests below are only meaningful if the factory's leads pass the same
        # gate the search pipeline applies. Asserting it here means a factory that quietly
        # stopped qualifying shows up as a factory failure, not as a scoring mystery.
        for niche_id, profile in NICHE_PROFILES.items():
            with self.subTest(niche=niche_id):
                self.assertTrue(matches_niche(profile, qualifying_lead(profile)))


class ScoringTests(unittest.TestCase):
    def test_copy_is_niche_specific_for_every_profile(self):
        for niche_id, profile in NICHE_PROFILES.items():
            with self.subTest(niche=niche_id):
                lead = qualifying_lead(profile)
                lead.matched_niches.append(niche_id)
                score = score_lead(lead, [profile])
                summary = build_fallback_summary(lead, score, [profile], "Pune")
                outreach = build_fallback_outreach(lead, score, [profile], "Pune")
                combined = (summary + " " + outreach).lower()
                self.assertIn(profile.offer.split()[0].lower(), combined)
                self.assertNotIn("bangalore", combined)
                if niche_id not in {"cafe", "bakery", "cake_shop", "cloud_kitchen"}:
                    self.assertNotIn("cafes and bakeries", combined)

    def test_scores_social_only_bakery_above_website_ready_business(self):
        social_only = Lead(
            name="Sweet Crumbs",
            category="bakery",
            address="Indiranagar, Bengaluru",
            city="Bangalore",
            latitude=12.978,
            longitude=77.641,
            phone="+91 90000 00000",
            website="https://instagram.com/sweetcrumbs",
            source_url="https://www.openstreetmap.org/way/1",
            raw_categories=["commercial.food_and_drink.bakery"],
        )

        proper_site = Lead(
            name="Chain Cafe",
            category="cafe",
            address="MG Road, Bengaluru",
            city="Bangalore",
            latitude=12.975,
            longitude=77.606,
            phone="+91 91111 11111",
            website="https://chaincafe.example.com",
            source_url="https://www.openstreetmap.org/way/2",
            raw_categories=["catering.cafe"],
        )

        social_score = score_lead(social_only)
        proper_score = score_lead(proper_site)

        self.assertGreater(social_score.total, proper_score.total)
        self.assertIn("social-only", social_score.signals)
        self.assertIn("public phone", social_score.signals)

    def test_no_website_lead_has_clear_pitch_angle(self):
        lead = Lead(
            name="Cloud Cake Studio",
            category="cake shop",
            address="Koramangala, Bengaluru",
            city="Bangalore",
            latitude=12.935,
            longitude=77.624,
            phone=None,
            website=None,
            source_url="https://www.openstreetmap.org/node/3",
            raw_categories=["commercial.food_and_drink.confectionery"],
        )

        score = score_lead(lead)

        self.assertGreaterEqual(score.website_gap, 20)
        self.assertIn("no website", score.signals)
        self.assertIn("catalog + advance payment", score.pitch_angle)

    def test_score_lead_ignores_evidence_for_now(self):
        # Evidence is on the signature so both passes call one function; banding against
        # the cohort happens in SQL. Until then the listing score must not move.
        lead = qualifying_lead(NICHE_PROFILES["bakery"])
        profiles = [NICHE_PROFILES["bakery"]]

        without = score_lead(lead, profiles)
        with_evidence = score_lead(
            lead,
            profiles,
            evidence=Evidence(reviews=800, followers=12000, engagement_rate=0.07),
        )

        self.assertEqual(without, with_evidence)


class AudienceIndexTests(unittest.TestCase):
    def test_returns_none_when_no_component_is_present(self):
        self.assertIsNone(audience_index(Evidence()))

    def test_unweighted_fields_alone_are_not_evidence_of_audience(self):
        # rating and runs_ads carry no weight, so on their own there is still nothing to
        # measure -- the answer is "unknown", not 0.0.
        self.assertIsNone(audience_index(Evidence(rating=4.6, runs_ads=True)))

    def test_combines_every_component_with_its_weight(self):
        index = audience_index(
            Evidence(
                reviews=100,
                rating=4.4,
                followers=10_000,
                engagement_rate=0.05,
                popular_times_density=50.0,
                runs_ads=True,
            )
        )

        self.assertAlmostEqual(index, 0.5904347430148761, places=12)

    def test_each_component_alone_is_its_own_transform(self):
        # With one component present the renormalised weight is 1.0, so the index is that
        # component's transform and nothing else. Each input is chosen to land on 0.5/0.8.
        cases = [
            ("reviews", Evidence(reviews=99), 0.5),
            ("followers", Evidence(followers=9_999), 0.8),
            ("engagement_rate", Evidence(engagement_rate=0.05), 0.5),
            ("popular_times_density", Evidence(popular_times_density=50.0), 0.5),
        ]
        for name, evidence, expected in cases:
            with self.subTest(component=name):
                self.assertAlmostEqual(audience_index(evidence), expected, places=12)

    def test_reviews_only_at_ceiling_renormalises_to_one_not_to_its_weight(self):
        # The headline renormalisation case: 0.4 here would mean a lead measured before
        # enrichment could never look strong, however large its audience.
        self.assertAlmostEqual(audience_index(Evidence(reviews=100_000)), 1.0, places=12)

    def test_google_only_pass_lands_on_the_same_scale_as_a_full_pass(self):
        # Discovery knows reviews and popular times only. Two half-strength components
        # must read as 0.5, not as the 0.25 an unnormalised sum would give.
        google_only = audience_index(Evidence(reviews=99, popular_times_density=50.0))
        full = audience_index(
            Evidence(
                reviews=99,
                followers=9_999,
                engagement_rate=0.05,
                popular_times_density=50.0,
            )
        )

        self.assertAlmostEqual(google_only, 0.5, places=12)
        self.assertGreater(full, google_only)

    def test_is_monotonic_in_reviews(self):
        indices = [audience_index(Evidence(reviews=count)) for count in (0, 1, 10, 100, 1_000)]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(set(indices)), len(indices))
        self.assertAlmostEqual(indices[0], 0.0, places=12)

    def test_clamps_extreme_inputs_to_one(self):
        extreme = Evidence(
            reviews=10**9,
            followers=10**9,
            engagement_rate=5.0,
            popular_times_density=1_000.0,
        )

        self.assertAlmostEqual(audience_index(extreme), 1.0, places=12)

    def test_every_result_stays_inside_the_unit_interval(self):
        samples = [
            Evidence(reviews=0),
            Evidence(reviews=-5),
            Evidence(engagement_rate=-0.2),
            Evidence(popular_times_density=0.0),
            Evidence(reviews=3, followers=250, engagement_rate=0.004),
            Evidence(reviews=25_000, followers=1_000_000, popular_times_density=100.0),
        ]
        for evidence in samples:
            with self.subTest(evidence=evidence):
                index = audience_index(evidence)
                self.assertGreaterEqual(index, 0.0)
                self.assertLessEqual(index, 1.0)


if __name__ == "__main__":
    unittest.main()
