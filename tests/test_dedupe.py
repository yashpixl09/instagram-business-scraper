"""Deduplication tests, ported from the prototype's `tests/test_dedupe.py`.

Only the import paths changed. The Geoapify-style dotted category strings are kept
verbatim from the prototype: `dedupe.py` never interprets a category, it only unions the
lists, so the strings are opaque payload here and keeping them makes this file diffable
against the prototype it came from.
"""

from __future__ import annotations

import unittest

from lead_engine.dedupe import dedupe_leads
from lead_engine.models import Lead


class DedupeTests(unittest.TestCase):
    def test_provider_identity_merges_niche_matches(self):
        first = Lead(
            name="Shared Business",
            category="cafe",
            address="Pune",
            city="Pune",
            latitude=18.52,
            longitude=73.86,
            phone=None,
            website=None,
            source_url=None,
            raw_categories=["catering.cafe"],
            provider_id="place-1",
            matched_niches=["cafe"],
        )
        second = Lead(
            name="Shared Business Bakery",
            category="bakery",
            address="Pune",
            city="Pune",
            latitude=18.52,
            longitude=73.86,
            phone="9000000000",
            website=None,
            source_url=None,
            raw_categories=["commercial.food_and_drink.bakery"],
            provider_id="place-1",
            matched_niches=["bakery"],
        )

        result = dedupe_leads([first, second])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].matched_niches, ["cafe", "bakery"])
        self.assertEqual(result[0].phone, "9000000000")

    def test_same_name_at_different_addresses_is_not_merged_without_contact(self):
        first = Lead(
            name="Popular Classes",
            category="education",
            address="Kothrud, Pune",
            city="Pune",
            latitude=18.50,
            longitude=73.81,
            phone=None,
            website=None,
            source_url=None,
        )
        second = Lead(
            name="Popular Classes",
            category="education",
            address="Viman Nagar, Pune",
            city="Pune",
            latitude=18.56,
            longitude=73.91,
            phone=None,
            website=None,
            source_url=None,
        )

        self.assertEqual(len(dedupe_leads([first, second])), 2)

    def test_merges_duplicate_business_and_keeps_richer_address(self):
        sparse = Lead(
            name="Marzipan Cafe",
            category="bakery",
            address="",
            city="Bangalore",
            latitude=None,
            longitude=None,
            phone="9844422724",
            website=None,
            source_url="source-1",
            raw_categories=["commercial.food_and_drink.bakery"],
        )
        richer = Lead(
            name="Marzipan Cafe",
            category="bakery",
            address="Marzipan Cafe, Halasuru Road, Bengaluru",
            city="Bangalore",
            latitude=12.98,
            longitude=77.62,
            phone="9844422724",
            website=None,
            source_url="source-2",
            raw_categories=["catering.cafe"],
        )

        result = dedupe_leads([sparse, richer])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].address, "Marzipan Cafe, Halasuru Road, Bengaluru")
        self.assertEqual(result[0].source_url, "source-2")
        self.assertIn("commercial.food_and_drink.bakery", result[0].raw_categories)
        self.assertIn("catering.cafe", result[0].raw_categories)

    def test_accepts_numeric_phone_values_from_provider_data(self):
        # Regression guard: providers really do return phone numbers as JSON numbers, and
        # this is why `normalize()` calls `str()` before touching the value.
        lead = Lead(
            name="Numeric Phone Bakery",
            category="bakery",
            address="Bengaluru",
            city="Bangalore",
            latitude=None,
            longitude=None,
            phone=9844422724,  # type: ignore[arg-type]
            website=None,
            source_url=None,
            raw_categories=[],
        )

        result = dedupe_leads([lead])

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
