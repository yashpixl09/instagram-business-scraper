"""Qualification tests for the niche registry.

The suite is unittest-flavoured (pytest is only the runner) to match the ported prototype.

Anti-substitution is the product guarantee -- asking for salons returns salons or nothing --
and three tests defend it at different strengths:

`test_no_include_type_is_claimed_by_two_niches` is the structural one. If no two niches
claim the same Google type, substitution cannot happen by type at all, and the rest is a
matter of exclusions and name evidence. It is the cheapest and sharpest of the three.

`test_no_niche_accepts_another_niches_individual_types` sweeps all 24 x 23 pairs, one type
at a time. The per-type form matters: a bundled lead carrying every one of a niche's types
short-circuits on the first exclusion collision, so it can pass while individual types
still leak.

`test_every_niche_accepts_its_own_canonical_place` is the counterweight. Without it a
registry that excluded everything would pass the other two trivially.
"""

from __future__ import annotations

import unittest

from lead_engine.models import Lead
from lead_engine.niches import (
    DISQUALIFIES,
    NEUTRAL,
    NICHE_PROFILES,
    QUALIFIES,
    NicheProfile,
    UnsupportedNicheError,
    classify_type,
    has_name_evidence,
    matches_niche,
    niche_payload,
    normalize_niche,
    resolve_niche_ids,
    slugify_type,
)

# Pairs whose canonical places genuinely overlap in Google's data. Declared here so the
# exemptions stay small, visible and reviewable rather than dissolving into the registry.
OVERLAP_ALLOWED = {
    frozenset({"bakery", "cake_shop"}),   # an Indian bakery sells the cakes
    frozenset({"salon", "spa"}),          # unisex salons list spa services and vice versa
    frozenset({"cafe", "bakery"}),        # bakery-cafes are one shop with two labels
}

NAME_ONLY_NICHES = {"cloud_kitchen", "manufacturer"}


def make_lead(
    name: str,
    category: str = "",
    raw_categories: list[str] | None = None,
    website: str | None = None,
) -> Lead:
    return Lead(
        name=name,
        category=category,
        address="12 MG Road",
        city="Bangalore",
        latitude=12.97,
        longitude=77.59,
        phone="+91 80 4000 0000",
        website=website,
        source_url=None,
        raw_categories=list(raw_categories or []),
    )


def canonical_lead(profile: NicheProfile) -> Lead:
    """The most representative place this niche could return.

    Its name is the niche label (perfect name evidence) and every one of its declared
    include types is present -- so any other niche that accepts it is substituting.
    """
    types = sorted(profile.include_types)
    primary = profile.id if profile.id in profile.include_types else (
        types[0] if types else profile.label
    )
    return make_lead(profile.label, category=primary, raw_categories=types)


class SlugifyTypeTests(unittest.TestCase):
    def test_folds_accents(self):
        self.assertEqual(slugify_type("Café"), "cafe")
        self.assertEqual(slugify_type("Cafe"), "cafe")
        self.assertEqual(slugify_type("Café"), slugify_type("Cafe"))

    def test_folds_punctuation_and_spaces_to_single_underscores(self):
        self.assertEqual(slugify_type("Car repair & maintenance"), "car_repair_maintenance")
        self.assertEqual(slugify_type("Hair salon"), "hair_salon")
        self.assertEqual(slugify_type("Cold storage facility"), "cold_storage_facility")
        self.assertEqual(slugify_type("  Beauty  Parlour  "), "beauty_parlour")

    def test_is_idempotent_on_already_slugified_values(self):
        self.assertEqual(slugify_type("hair_salon"), "hair_salon")

    def test_empty_and_junk_values_slugify_to_empty(self):
        self.assertEqual(slugify_type(""), "")
        self.assertEqual(slugify_type("---"), "")


class ClassifyTypePrecedenceTests(unittest.TestCase):
    """exact include > exact exclude > suffix exclude > suffix include > neutral."""

    profile = NicheProfile(
        id="t", label="T",
        queries=("t",),
        include_types=frozenset({"plastic_products_supplier", "overlap_type"}),
        exclude_types=frozenset({"overlap_type_excluded", "kitchen_furniture_store"}),
        include_suffixes=("_supplier", "_works"),
        exclude_suffixes=("_store", "_supplier_store"),
    )

    def test_exact_include_beats_exclude_suffix(self):
        # `plastic_products_supplier` ends with nothing excluded, so build the real clash:
        clash = NicheProfile(
            id="t2", label="T2",
            queries=("t2",),
            include_types=frozenset({"kitchen_supply_store"}),
            exclude_suffixes=("_store",),
        )
        self.assertEqual(classify_type(clash, "kitchen_supply_store"), QUALIFIES)
        self.assertEqual(classify_type(clash, "furniture_store"), DISQUALIFIES)

    def test_exact_exclude_beats_include_suffix(self):
        clash = NicheProfile(
            id="t3", label="T3",
            queries=("t3",),
            exclude_types=frozenset({"driving_school"}),
            include_suffixes=("_school",),
        )
        self.assertEqual(classify_type(clash, "driving_school"), DISQUALIFIES)
        self.assertEqual(classify_type(clash, "music_school"), QUALIFIES)

    def test_exclude_suffix_beats_include_suffix(self):
        clash = NicheProfile(
            id="t4", label="T4",
            queries=("t4",),
            include_suffixes=("_supplier",),
            exclude_suffixes=("_equipment_supplier",),
        )
        self.assertEqual(classify_type(clash, "bakery_equipment_supplier"), DISQUALIFIES)
        self.assertEqual(classify_type(clash, "plastic_products_supplier"), QUALIFIES)

    def test_unknown_type_is_neutral(self):
        self.assertEqual(classify_type(self.profile, "historical_landmark"), NEUTRAL)

    # A registry-backed precedence test was removed here deliberately. No profile currently
    # has an `include_types` entry that also matches one of its own `exclude_suffixes`, so
    # any such test passes regardless of precedence order and asserts nothing. The synthetic
    # profiles above cover precedence properly; `test_registry_has_no_precedence_collisions`
    # below pins the fact that made the registry version vacuous.

    def test_registry_has_no_precedence_collisions(self):
        """No profile includes a type its own exclude suffixes would reject.

        If this ever fails, the exact-include-beats-suffix-exclude rule has become
        load-bearing for real data and deserves a registry-level test of its own.
        """
        for profile in NICHE_PROFILES.values():
            for niche_type in sorted(profile.include_types):
                collisions = [s for s in profile.exclude_suffixes if niche_type.endswith(s)]
                with self.subTest(niche=profile.id, type=niche_type):
                    self.assertEqual(collisions, [])


class DisqualifierIsFatalTests(unittest.TestCase):
    def test_one_disqualifying_type_rejects_a_place_that_also_qualifies(self):
        salon = NICHE_PROFILES["salon"]
        qualifying_only = make_lead(
            "Glow Unisex Salon",
            category="Hair salon",
            raw_categories=["Hair salon", "Nail salon"],
        )
        self.assertTrue(matches_niche(salon, qualifying_only))

        with_disqualifier = make_lead(
            "Glow Unisex Salon",
            category="Hair salon",
            raw_categories=["Hair salon", "Nail salon", "Beauty supply store"],
        )
        self.assertFalse(matches_niche(salon, with_disqualifier))

    def test_disqualifier_in_the_primary_type_is_fatal_too(self):
        salon = NICHE_PROFILES["salon"]
        lead = make_lead(
            "Cafe Coiffure Salon",
            category="Coffee shop",
            raw_categories=["Coffee shop", "Hair salon"],
        )
        self.assertFalse(matches_niche(salon, lead))

    def test_the_headline_case_a_cafe_is_never_a_salon(self):
        salon = NICHE_PROFILES["salon"]
        cafe_lead = make_lead(
            "Third Wave Coffee",
            category="Café",
            raw_categories=["Café", "Coffee shop", "Espresso bar"],
        )
        self.assertFalse(matches_niche(salon, cafe_lead))


class StrictAndNameEvidenceTests(unittest.TestCase):
    def test_strict_profile_needs_both_type_and_name_evidence(self):
        gym = NICHE_PROFILES["fitness_gym"]
        typed_but_unnamed = make_lead(
            "Vivek Enterprises",
            category="Sports club",
            raw_categories=["Sports club"],
        )
        self.assertFalse(matches_niche(gym, typed_but_unnamed))

        named = make_lead(
            "Iron House Fitness",
            category="Gym",
            raw_categories=["Gym", "Sports club"],
        )
        self.assertTrue(matches_niche(gym, named))

    def test_non_strict_profile_needs_type_only(self):
        cafe = NICHE_PROFILES["cafe"]
        self.assertFalse(cafe.strict)
        lead = make_lead("Vivek Enterprises", category="Coffee shop",
                         raw_categories=["Coffee shop"])
        self.assertTrue(matches_niche(cafe, lead))

    def test_name_evidence_reads_through_slug_separators(self):
        manufacturer = NICHE_PROFILES["manufacturer"]
        lead = make_lead("Acme Co", category="", raw_categories=["metal_workshop"])
        self.assertTrue(has_name_evidence(manufacturer, lead))

    def test_missing_category_does_not_crash_qualification(self):
        cafe = NICHE_PROFILES["cafe"]
        self.assertFalse(matches_niche(cafe, make_lead("Nameless", category="")))


class AllowNameOnlyTests(unittest.TestCase):
    def test_exactly_two_niches_allow_name_only(self):
        enabled = {p.id for p in NICHE_PROFILES.values() if p.allow_name_only}
        self.assertEqual(enabled, NAME_ONLY_NICHES)

    def test_name_only_niche_qualifies_on_name_without_a_faithful_type(self):
        cloud = NICHE_PROFILES["cloud_kitchen"]
        lead = make_lead(
            "Biryani Cloud Kitchen",
            category="Food court",           # no faithful Google type for this business
            raw_categories=["Food court"],
        )
        self.assertTrue(matches_niche(cloud, lead))

    def test_name_only_cannot_bypass_an_exclusion(self):
        cloud = NICHE_PROFILES["cloud_kitchen"]
        perfect_name_but_excluded = make_lead(
            "Cloud Kitchen Modular Interiors",     # flawless name evidence
            category="Modular kitchen store",      # excluded type
            raw_categories=["Modular kitchen store", "Kitchen furniture store"],
        )
        self.assertTrue(has_name_evidence(cloud, perfect_name_but_excluded))
        self.assertFalse(matches_niche(cloud, perfect_name_but_excluded))

    def test_manufacturer_name_only_cannot_bypass_an_exclusion(self):
        manufacturer = NICHE_PROFILES["manufacturer"]
        lead = make_lead(
            "Sri Balaji Industries Pvt Ltd",
            category="Wholesaler",
            raw_categories=["Wholesaler", "Distributor"],
        )
        self.assertTrue(has_name_evidence(manufacturer, lead))
        self.assertFalse(matches_niche(manufacturer, lead))

    def test_strict_niche_without_name_only_needs_a_qualifying_type(self):
        salon = NICHE_PROFILES["salon"]
        self.assertFalse(salon.allow_name_only)
        lead = make_lead("Sharp Cuts Barber", category="Point of interest",
                         raw_categories=["Point of interest"])
        self.assertTrue(has_name_evidence(salon, lead))
        self.assertFalse(matches_niche(salon, lead))


class NicheResolutionTests(unittest.TestCase):
    def test_resolves_ids_labels_and_aliases(self):
        self.assertEqual(resolve_niche_ids(["salon"]), ["salon"])
        self.assertEqual(resolve_niche_ids(["Hair Salon"]), ["salon"])
        self.assertEqual(resolve_niche_ids(["fitness/gym"]), ["fitness_gym"])
        self.assertEqual(resolve_niche_ids(["Tutor/Class"]), ["tutor_class"])
        self.assertEqual(resolve_niche_ids(["ghost kitchen"]), ["cloud_kitchen"])

    def test_dedupes_while_preserving_first_seen_order(self):
        self.assertEqual(
            resolve_niche_ids(["gym", "salon", "fitness", "beauty salon", "cafe"]),
            ["fitness_gym", "salon", "cafe"],
        )

    def test_unknown_value_raises(self):
        with self.assertRaises(UnsupportedNicheError) as ctx:
            resolve_niche_ids(["scuba diving school"])
        self.assertEqual(ctx.exception.value, "scuba diving school")

    def test_normalize_niche_strips_separators(self):
        self.assertEqual(normalize_niche("Fitness / Gym"), "fitnessgym")


class RegistryShapeTests(unittest.TestCase):
    def test_keys_match_ids_and_registry_is_complete(self):
        # 24, not 23: the prototype registry shipped with 24 ids and every one is kept.
        self.assertEqual(len(NICHE_PROFILES), 24)
        for key, profile in NICHE_PROFILES.items():
            self.assertEqual(key, profile.id)

    def test_registry_keeps_the_prototype_ids_and_their_order(self):
        self.assertEqual(
            list(NICHE_PROFILES),
            ["cafe", "bakery", "cake_shop", "cloud_kitchen", "catering", "salon", "spa",
             "fitness_gym", "dental_clinic", "clinic", "veterinary", "auto_service",
             "preschool", "tutor_class", "driving_school", "photographer", "event_planner",
             "interior_designer", "real_estate", "travel_agency", "professional_services",
             "boutique", "home_decor", "manufacturer"],
        )

    def test_every_profile_has_queries_and_a_primary_query(self):
        for profile in NICHE_PROFILES.values():
            with self.subTest(niche=profile.id):
                self.assertTrue(profile.queries)
                self.assertTrue(profile.queries[0].strip())

    def test_strict_profiles_carry_qualification_terms(self):
        for profile in NICHE_PROFILES.values():
            with self.subTest(niche=profile.id):
                if profile.strict:
                    self.assertTrue(profile.qualification_terms)

    def test_types_are_stored_as_slugs_not_display_labels(self):
        for profile in NICHE_PROFILES.values():
            for slug in (*profile.include_types, *profile.exclude_types):
                with self.subTest(niche=profile.id, slug=slug):
                    self.assertEqual(slug, slugify_type(slug))

    def test_suffixes_start_with_an_underscore(self):
        for profile in NICHE_PROFILES.values():
            for suffix in (*profile.include_suffixes, *profile.exclude_suffixes):
                with self.subTest(niche=profile.id, suffix=suffix):
                    self.assertTrue(suffix.startswith("_"))

    def test_no_type_is_both_included_and_excluded_in_one_profile(self):
        for profile in NICHE_PROFILES.values():
            with self.subTest(niche=profile.id):
                self.assertEqual(profile.include_types & profile.exclude_types, frozenset())

    def test_payload_emits_the_new_field_names(self):
        payload = niche_payload()
        self.assertEqual(len(payload), len(NICHE_PROFILES))
        self.assertEqual(
            set(payload[0]),
            {"id", "label", "queries", "include_types", "exclude_types",
             "include_suffixes", "exclude_suffixes"},
        )
        self.assertEqual([row["id"] for row in payload], list(NICHE_PROFILES))


class AntiSubstitutionTests(unittest.TestCase):
    def test_every_niche_accepts_its_own_canonical_place(self):
        """Counterweight: without this, excluding everything would pass the cross product."""
        for profile in NICHE_PROFILES.values():
            with self.subTest(niche=profile.id):
                self.assertTrue(matches_niche(profile, canonical_lead(profile)))

    def test_no_niche_accepts_another_niches_canonical_place(self):
        for target in NICHE_PROFILES.values():
            for other in NICHE_PROFILES.values():
                if other.id == target.id or frozenset({target.id, other.id}) in OVERLAP_ALLOWED:
                    continue
                intruder = canonical_lead(other)
                with self.subTest(target=target.id, intruder=other.id):
                    self.assertFalse(matches_niche(target, intruder))

    def test_no_include_type_is_claimed_by_two_niches(self):
        """The structural guarantee. If no two niches claim the same Google type,
        substitution cannot happen by type at all.

        This held for every type except `caterer`, which `catering` and `cloud_kitchen`
        both claimed -- and that single collision was enough to make every caterer look
        like a cloud kitchen. `caterer` now belongs to `catering` alone.
        """
        owners: dict[str, list[str]] = {}
        for profile in NICHE_PROFILES.values():
            for niche_type in profile.include_types:
                owners.setdefault(niche_type, []).append(profile.id)
        contested = {t: sorted(ids) for t, ids in owners.items() if len(ids) > 1}
        self.assertEqual(contested, {}, f"types claimed by more than one niche: {contested}")

    def test_no_niche_accepts_another_niches_individual_types(self):
        """The exhaustive sweep, one type at a time.

        A lead bundling every one of a niche's types short-circuits on the first exclusion
        collision, so the bundled cross product can pass while individual types still leak.
        A place typed only "Drivers license training school" was returned for tutor
        requests until this test existed.
        """
        for target in NICHE_PROFILES.values():
            for other in NICHE_PROFILES.values():
                if other.id == target.id or frozenset({target.id, other.id}) in OVERLAP_ALLOWED:
                    continue
                for niche_type in sorted(other.include_types):
                    intruder = make_lead(
                        other.label, category=niche_type, raw_categories=[niche_type]
                    )
                    with self.subTest(target=target.id, intruder=other.id, type=niche_type):
                        self.assertFalse(matches_niche(target, intruder))

    def test_overlap_allowlist_only_names_real_niches(self):
        for pair in OVERLAP_ALLOWED:
            for niche_id in pair:
                self.assertIn(niche_id, NICHE_PROFILES)


class DualLabelTests(unittest.TestCase):
    """Google dual-labels the businesses most worth pitching.

    A disqualifier is fatal, so two niches that exclude each other's include types leave a
    dual-labelled place matching nothing at all. The Kerala-style Ayurveda centre and the
    caterer who also runs a banquet hall are the highest-budget businesses in their niches
    and are exactly the ones Google gives two labels. Orphaning them is the expensive
    failure; matching two niches is harmless, since dedupe unions matched_niches.
    """

    COMBINATIONS = (
        ("Massage spa", "Ayurvedic clinic"),
        ("Ayurvedic clinic", "Wellness center"),
        ("Caterer", "Banquet hall"),
        ("Restaurant", "Caterer"),
        ("Hair salon", "Spa"),
        ("Photographer", "Wedding planner"),
        ("Boutique", "Fashion designer"),
        ("Dentist", "Medical clinic"),
        ("Gym", "Yoga studio"),
        ("Bakery", "Cake shop"),
    )

    def test_dual_labelled_places_match_at_least_one_niche(self):
        for combination in self.COMBINATIONS:
            lead = make_lead(
                " ".join(combination),
                category=combination[0],
                raw_categories=list(combination),
            )
            matched = [p.id for p in NICHE_PROFILES.values() if matches_niche(p, lead)]
            with self.subTest(types=combination):
                self.assertTrue(matched, f"{combination} is orphaned: no niche accepts it")


class RealWorldSubstitutionTests(unittest.TestCase):
    """Places SearchAPI actually returns for these queries in Bangalore."""

    def test_modular_kitchen_showroom_is_not_a_cloud_kitchen(self):
        lead = make_lead(
            "Sleek Modular Kitchens",
            category="Modular kitchen store",
            raw_categories=["Modular kitchen store", "Kitchen furniture store"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["cloud_kitchen"], lead))
        self.assertTrue(matches_niche(NICHE_PROFILES["home_decor"], lead))

    def test_historical_landmark_is_not_a_travel_agency(self):
        lead = make_lead(
            "Bangalore Palace",
            category="Historical landmark",
            raw_categories=["Historical landmark", "Tourist attraction"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["travel_agency"], lead))

    def test_car_dealer_is_not_an_auto_service(self):
        lead = make_lead(
            "Sagar Motors",
            category="Car dealer",
            raw_categories=["Car dealer", "Used car dealer", "Car repair and maintenance service"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["auto_service"], lead))

    def test_driving_school_is_not_a_tuition_centre(self):
        lead = make_lead(
            "Sri Sai Motor Driving School",
            category="Driving school",
            raw_categories=["Driving school"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["tutor_class"], lead))
        self.assertTrue(matches_niche(NICHE_PROFILES["driving_school"], lead))

    def test_beauty_supply_retail_is_not_a_salon(self):
        lead = make_lead(
            "Shahnaz Beauty Products",
            category="Beauty supply store",
            raw_categories=["Beauty supply store", "Cosmetics store"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["salon"], lead))

    def test_furniture_showroom_is_not_an_interior_designer(self):
        lead = make_lead(
            "Home Centre Interiors",
            category="Furniture store",
            raw_categories=["Furniture store", "Home goods store"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["interior_designer"], lead))

    def test_real_estate_consultant_is_not_professional_services(self):
        lead = make_lead(
            "Prestige Property Consultants",
            category="Real estate consultant",
            raw_categories=["Real estate consultant", "Real estate agency"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["professional_services"], lead))
        self.assertTrue(matches_niche(NICHE_PROFILES["real_estate"], lead))

    def test_accented_cafe_label_still_matches_the_cafe_niche(self):
        lead = make_lead("Koshy's", category="Café", raw_categories=["Café", "Coffee shop"])
        self.assertTrue(matches_niche(NICHE_PROFILES["cafe"], lead))

    def test_dental_lab_is_not_a_dental_clinic(self):
        lead = make_lead(
            "Precision Dental Laboratory",
            category="Dental laboratory",
            raw_categories=["Dental laboratory"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["dental_clinic"], lead))

    def test_suffix_rule_rejects_a_retail_type_nobody_enumerated(self):
        """The point of suffix rules: reject the long tail without listing it.

        None of these slugs appear in any `exclude_types` set -- the head noun does the work.
        """
        manufacturer = NICHE_PROFILES["manufacturer"]
        for label in ("Steel almirah dealer", "Industrial pump showroom", "Bearing store"):
            with self.subTest(label=label):
                lead = make_lead(
                    "Sri Balaji Industries Pvt Ltd",   # perfect name evidence
                    category=label,
                    raw_categories=[label],
                )
                self.assertNotIn(slugify_type(label), manufacturer.exclude_types)
                self.assertFalse(matches_niche(manufacturer, lead))

    def test_suffix_rule_keeps_a_genuine_manufacturer_that_nobody_enumerated(self):
        manufacturer = NICHE_PROFILES["manufacturer"]
        lead = make_lead(
            "Ganesh Precision Works",
            category="Gasket manufacturer",
            raw_categories=["Gasket manufacturer", "Rubber products supplier"],
        )
        self.assertNotIn("gasket_manufacturer", manufacturer.include_types)
        self.assertTrue(matches_niche(manufacturer, lead))

    def test_pet_store_is_not_a_veterinary_clinic(self):
        lead = make_lead(
            "Happy Tails Pet Store",
            category="Pet store",
            raw_categories=["Pet store", "Pet supply store"],
        )
        self.assertFalse(matches_niche(NICHE_PROFILES["veterinary"], lead))


if __name__ == "__main__":
    unittest.main()
