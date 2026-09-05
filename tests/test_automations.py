"""Automation catalogue tests.

The catalogue decides which automation offers a business is told it needs, and the LLM
writes prose over whatever these functions return. So the failure this suite is built
around is not a crash: it is a plausible claim about how a business operates -- "there is
no way to book you online", "six hours a week" -- reaching the owner's face without
anything having been observed. A crash costs an afternoon; a fabricated claim costs the
account.

That is why the negative cases outnumber the positive ones here. `OfferFiresTests` and
`AdversarialObservationTests` between them sweep every offer against empty evidence,
partial evidence, misspelt evidence, near-miss evidence and evidence of the wrong shape
entirely, and each asserts the same thing: nothing fires. `FiringOffersTests` carries the
one that matters most in practice -- a lead that has only been through discovery, whose
signals all come from the Google listing, receives no automation pitch at all, because
every offer in the catalogue requires at least one signal an enrichment pass had to go and
observe.

Two claims the module's own docstring makes about the world outside it are checked by
running that world rather than trusting the comment:

* `SignalVocabularyTests.test_scorer_emits_every_signal_the_catalogue_names` runs
  `score_lead` across the registry and asserts `SCORER_SIGNALS` is exactly the non-niche
  vocabulary it can produce. A scorer that renamed a signal would otherwise turn two
  offers into dead rules silently.
* `CatalogueIntegrityTests` checks every declared niche against `NICHE_PROFILES` and every
  required signal against `KNOWN_SIGNALS`, because a rule that can never fire is worse than
  no rule -- it looks like coverage.

`EstimatedHoursTests` pins `est_hours_saved_weekly` as what it currently is: a constant
stated per opportunity type, identical for every business that offer fires on, derived from
no measurement. The tests pin the numbers so a change is deliberate, and assert the
catalogue neither aggregates them nor states one in any pitch line -- they do not certify
that any of them is true.
"""

from __future__ import annotations

import inspect
import json
import unittest
from dataclasses import FrozenInstanceError, replace

from lead_engine import automations
from lead_engine.automations import (
    AUTOMATION_OFFERS,
    ENRICHMENT_SIGNALS,
    ENRICHMENT_SOURCES,
    KNOWN_SIGNALS,
    SCORER_SIGNALS,
    AutomationOffer,
    applies_to_niche,
    automation_payload,
    firing_offers,
    missing_signals,
    offer_fires,
    offers_for_niche,
)
from lead_engine.niches import NICHE_PROFILES
from lead_engine.scoring import score_lead
from tests.factories import qualifying_lead

#: The offers with no niche gate, named here so the coverage tests can talk about "every
#: other offer" without recomputing the thing they are checking.
UNIVERSAL_OFFER_IDS = ("review_response", "lead_capture")

#: A full observation set: every signal the catalogue will ever accept, at once. Feeding
#: this in answers "what is the most this niche could ever be offered", which is the ceiling
#: the negative cases are measured against.
EVERY_SIGNAL = frozenset(KNOWN_SIGNALS)

#: Strings that are not signals. Each one is a mistake an enrichment payload could plausibly
#: make: a near miss, a case change, a stray space, a substring, a translation of the token
#: into another spelling convention.
NEAR_MISSES = (
    "no booking",
    "booking link",
    "no booking link ",
    " no booking link",
    "no  booking link",
    "No Booking Link",
    "NO BOOKING LINK",
    "no_booking_link",
    "no-booking-link",
    "no booking links",
    "nobookinglink",
)


def niche_specific_signals_for(niche_id: str) -> tuple[str, ...]:
    """Every signal required by a niche-gated offer that reaches this niche.

    The universal offers are excluded on purpose: their signals fire for everyone by
    design, so including them would drown out the question of whether one niche's evidence
    leaks into another's pitch.
    """
    return tuple(
        dict.fromkeys(
            signal
            for offer in offers_for_niche(niche_id)
            if offer.niches
            for signal in offer.required_signals
        )
    )


def scorer_signals_of(profile, **overrides) -> list[str]:
    """What `score_lead` actually emits for a lead built from this profile."""
    lead = replace(qualifying_lead(profile), **overrides)
    return list(score_lead(lead, [profile]).signals)


class SignalVocabularyTests(unittest.TestCase):
    """The vocabulary is the join between this module and the two things that fill it."""

    def test_scorer_emits_every_signal_the_catalogue_names(self):
        # The module's docstring promises this is proved by running the scorer rather than
        # by trusting the tuple. A scorer that renamed "manual enquiry flow"'s neighbour
        # `public phone` would leave two offers unable to fire and nothing else would say so.
        websites = (
            None,
            "https://instagram.com/example",
            "https://example.wixsite.com/shop",
            "https://example.com",
        )
        emitted: set[str] = set()
        for profile in NICHE_PROFILES.values():
            other = next(p for p in NICHE_PROFILES.values() if p.id != profile.id)
            for website in websites:
                for phone in (None, "+91 90000 00000"):
                    for categories in ([], ["one"], ["one", "two"]):
                        lead = replace(
                            qualifying_lead(profile),
                            website=website,
                            phone=phone,
                            raw_categories=list(categories),
                        )
                        emitted |= set(score_lead(lead, [profile]).signals)
                        emitted |= set(score_lead(lead, [profile, other]).signals)

        for signal in SCORER_SIGNALS:
            with self.subTest(signal=signal):
                self.assertIn(signal, emitted)

    def test_scorer_emits_nothing_outside_the_vocabulary_but_niche_matches(self):
        # The other half of the same claim. Anything the scorer emits that this module has
        # not declared is a signal no offer can ever require -- invisible coverage.
        niche_matches = {f"{p.label.lower()} match" for p in NICHE_PROFILES.values()}
        for niche_id, profile in NICHE_PROFILES.items():
            with self.subTest(niche=niche_id):
                for website in (None, "https://instagram.com/x", "https://x.com"):
                    emitted = set(scorer_signals_of(profile, website=website))
                    self.assertEqual(emitted - niche_matches - set(SCORER_SIGNALS), set())

    def test_niche_match_strings_are_deliberately_outside_the_vocabulary(self):
        # The niche gate already carries this information; an offer requiring both would
        # state the same condition twice.
        for niche_id, profile in NICHE_PROFILES.items():
            with self.subTest(niche=niche_id):
                self.assertNotIn(f"{profile.label.lower()} match", KNOWN_SIGNALS)

    def test_the_two_signal_families_do_not_overlap(self):
        # Provenance is the point of the split: a signal in both families would have no
        # single answer to "which pass observed this".
        self.assertEqual(set(SCORER_SIGNALS) & set(ENRICHMENT_SIGNALS), set())

    def test_known_signals_is_exactly_the_union_of_the_two_families(self):
        self.assertEqual(KNOWN_SIGNALS, frozenset(SCORER_SIGNALS) | frozenset(ENRICHMENT_SIGNALS))

    def test_every_enrichment_signal_names_a_declared_source(self):
        for signal, source in ENRICHMENT_SIGNALS.items():
            with self.subTest(signal=signal):
                self.assertIn(source, ENRICHMENT_SOURCES)

    def test_every_declared_source_actually_asserts_something(self):
        # A source with no signals is documentation of a pass that cannot contribute.
        self.assertEqual(set(ENRICHMENT_SOURCES), set(ENRICHMENT_SIGNALS.values()))

    def test_signals_are_bare_lowercase_tokens(self):
        # Membership is exact, so a signal carrying leading whitespace or a capital could
        # only ever be satisfied by a payload carrying the identical mistake.
        for signal in sorted(KNOWN_SIGNALS):
            with self.subTest(signal=signal):
                self.assertEqual(signal, signal.strip())
                self.assertEqual(signal, signal.lower())
                self.assertNotIn("  ", signal)
                self.assertTrue(signal)

    def test_scorer_signal_tuple_has_no_duplicates(self):
        self.assertEqual(len(SCORER_SIGNALS), len(set(SCORER_SIGNALS)))


class CatalogueIntegrityTests(unittest.TestCase):
    """Structural claims. Each one, broken, produces a rule that quietly never fires."""

    def test_every_offer_id_matches_its_key(self):
        for key, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=key):
                self.assertEqual(key, offer.id)

    def test_offer_ids_and_labels_are_unique(self):
        labels = [offer.label for offer in AUTOMATION_OFFERS.values()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(AUTOMATION_OFFERS), len({o.id for o in AUTOMATION_OFFERS.values()}))

    def test_every_offer_requires_at_least_one_signal(self):
        # The invariant `offer_fires` names in its own docstring. An offer requiring nothing
        # would fire for every business in its niches, which is guessing with extra steps.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertGreater(len(offer.required_signals), 0)

    def test_no_required_signal_is_outside_the_vocabulary(self):
        # The dead-rule check. A required signal nothing can ever assert makes the offer
        # unreachable while it still reads as coverage in the catalogue.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            for signal in offer.required_signals:
                with self.subTest(offer=offer_id, signal=signal):
                    self.assertIn(signal, KNOWN_SIGNALS)

    def test_no_offer_requires_the_same_signal_twice(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertEqual(
                    len(offer.required_signals), len(set(offer.required_signals))
                )

    def test_every_offer_requires_at_least_one_enrichment_signal(self):
        # This is what makes "a discovery-only lead gets no automation pitch" true, and it
        # is the load-bearing safety property of the whole catalogue: the listing alone can
        # never produce a claim about how the business runs its workflows.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertTrue(set(offer.required_signals) & set(ENRICHMENT_SIGNALS))

    def test_every_niche_an_offer_names_exists_in_the_registry(self):
        # A typo here is a rule that can never fire, since `applies_to_niche` compares
        # against `businesses.niche_id` values the registry produces.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            for niche_id in offer.niches:
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertIn(niche_id, NICHE_PROFILES)

    def test_every_registry_niche_receives_at_least_one_niche_specific_offer(self):
        # The coverage claim the `enquiry_routing` comment appeals to. Without it a niche
        # could be left with the two universal offers and nobody would notice.
        specific = {niche for offer in AUTOMATION_OFFERS.values() for niche in offer.niches}
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                self.assertIn(niche_id, specific)

    def test_the_universal_offers_are_exactly_the_two_declared_ones(self):
        universal = tuple(o.id for o in AUTOMATION_OFFERS.values() if not o.niches)
        self.assertEqual(universal, UNIVERSAL_OFFER_IDS)

    def test_enquiry_routing_niches_are_reached_by_no_other_niche_specific_offer(self):
        # Pins the justification written into the catalogue: these five would otherwise
        # have only the universal offers.
        others = [
            offer
            for offer in AUTOMATION_OFFERS.values()
            if offer.id != "enquiry_routing" and offer.niches
        ]
        for niche_id in AUTOMATION_OFFERS["enquiry_routing"].niches:
            with self.subTest(niche=niche_id):
                self.assertEqual([o.id for o in others if niche_id in o.niches], [])

    def test_pitch_lines_state_no_number(self):
        # The pitch line is the sentence the operator says out loud. A digit in it would be
        # a quantitative claim written into a static catalogue rather than measured, which
        # is precisely the failure this system is most exposed to.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertFalse(any(char.isdigit() for char in offer.pitch_line))
                self.assertNotIn("%", offer.pitch_line)
                self.assertNotIn("hour", offer.pitch_line.lower())

    def test_pitch_lines_and_labels_are_non_empty_prose(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertTrue(offer.label.strip())
                self.assertTrue(offer.pitch_line.strip())
                self.assertEqual(offer.pitch_line, offer.pitch_line.strip())

    def test_offers_are_frozen(self):
        offer = AUTOMATION_OFFERS["order_intake"]
        with self.assertRaises(FrozenInstanceError):
            offer.est_hours_saved_weekly = 99.0  # type: ignore[misc]


class EstimatedHoursTests(unittest.TestCase):
    """`est_hours_saved_weekly` is an estimate per opportunity type, not a measurement.

    Nothing in this repository derives these numbers from data, and no test here asserts
    any of them is true of any business. They are pinned so that changing one is a
    deliberate act with a diff, and surrounded by tests asserting the catalogue does not
    dress them up as measured: no aggregate, no per-business variation, no appearance in
    the pitch prose.
    """

    #: The catalogue's stated values as of this commit. Unvalidated.
    STATED_HOURS = {
        "order_intake": 6.0,
        "appointment_booking": 5.0,
        "enrolment_scheduling": 4.0,
        "quotation_handling": 4.0,
        "enquiry_routing": 3.5,
        "catalog_whatsapp": 5.0,
        "service_reminders": 2.5,
        "review_response": 2.0,
        "lead_capture": 3.0,
    }

    def test_stated_estimates_match_the_catalogue(self):
        actual = {offer.id: offer.est_hours_saved_weekly for offer in AUTOMATION_OFFERS.values()}
        self.assertEqual(actual, self.STATED_HOURS)

    def test_every_estimate_is_a_positive_finite_float(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertIsInstance(offer.est_hours_saved_weekly, float)
                self.assertGreater(offer.est_hours_saved_weekly, 0.0)
                self.assertLess(offer.est_hours_saved_weekly, 168.0)

    def test_the_estimate_is_a_constant_of_the_type_not_of_the_business(self):
        # Fire the same offer for every niche it reaches, on two different observation sets,
        # and the number does not move. That is the honest description of what it is: a
        # property of the opportunity type. Anything presenting it as this business's saving
        # is adding a claim the catalogue never made.
        offer = AUTOMATION_OFFERS["appointment_booking"]
        seen = set()
        for niche_id in offer.niches:
            for observed in (offer.required_signals, EVERY_SIGNAL):
                fired = firing_offers(niche_id, observed)
                match = next(o for o in fired if o.id == offer.id)
                seen.add(match.est_hours_saved_weekly)
        self.assertEqual(seen, {5.0})

    def test_the_catalogue_offers_no_aggregate_of_the_estimates(self):
        # Several offers can fire at once, and their hours are trivially summable -- but the
        # sum would be a number nobody computed from anything. The module deliberately
        # provides no such total; this pins the public surface so one cannot appear quietly.
        defined = {
            name
            for name, obj in inspect.getmembers(automations, inspect.isfunction)
            if obj.__module__ == automations.__name__ and not name.startswith("_")
        }
        self.assertEqual(
            defined,
            {
                "applies_to_niche",
                "missing_signals",
                "offer_fires",
                "offers_for_niche",
                "firing_offers",
                "automation_payload",
            },
        )

    def test_no_payload_row_carries_a_derived_total(self):
        for row in automation_payload():
            with self.subTest(offer=row["id"]):
                self.assertEqual(
                    set(row),
                    {
                        "id",
                        "label",
                        "niches",
                        "required_signals",
                        "pitch_line",
                        "est_hours_saved_weekly",
                    },
                )


class NicheGateTests(unittest.TestCase):
    def test_an_offer_with_no_niches_applies_to_every_registry_niche(self):
        for offer_id in UNIVERSAL_OFFER_IDS:
            offer = AUTOMATION_OFFERS[offer_id]
            for niche_id in NICHE_PROFILES:
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertTrue(applies_to_niche(offer, niche_id))

    def test_a_niche_specific_offer_applies_only_to_its_declared_niches(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            if not offer.niches:
                continue
            for niche_id in NICHE_PROFILES:
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertEqual(
                        applies_to_niche(offer, niche_id), niche_id in offer.niches
                    )

    def test_niche_ids_are_matched_exactly(self):
        # `businesses.niche_id` is a slug written by the registry. Anything that is not that
        # exact slug is not that niche, however close it reads.
        offer = AUTOMATION_OFFERS["order_intake"]
        for niche_id in ("Cafe", "CAFE", " cafe", "cafe ", "cafes", "caf", "cafe_shop", ""):
            with self.subTest(niche=niche_id):
                self.assertFalse(applies_to_niche(offer, niche_id))

    def test_an_unknown_niche_gets_the_universal_offers_and_nothing_else(self):
        # Both readings of an unknown id -- a niche the catalogue has not been extended to,
        # and a corrupt `businesses.niche_id` -- want the same answer.
        for niche_id in ("", "not_a_niche", "cafe;drop", "None"):
            with self.subTest(niche=niche_id):
                self.assertEqual(
                    [o.id for o in offers_for_niche(niche_id)], list(UNIVERSAL_OFFER_IDS)
                )

    def test_a_none_niche_id_is_treated_as_unknown_rather_than_raising(self):
        # A row whose niche was never set arrives as None. It must degrade to the universal
        # offers, not crash the pitch pass and not match a niche.
        self.assertEqual(
            [o.id for o in offers_for_niche(None)],  # type: ignore[arg-type]
            list(UNIVERSAL_OFFER_IDS),
        )

    def test_offers_for_niche_returns_catalogue_order(self):
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                returned = [o.id for o in offers_for_niche(niche_id)]
                self.assertEqual(returned, [i for i in AUTOMATION_OFFERS if i in returned])

    def test_offers_for_niche_ignores_evidence_entirely(self):
        # It answers "what could this niche ever be offered", which is a question about the
        # catalogue. Every offer it returns still has to pass the signal gate afterwards.
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                candidates = offers_for_niche(niche_id)
                self.assertTrue(candidates)
                self.assertEqual(firing_offers(niche_id, ()), [])


class MissingSignalsTests(unittest.TestCase):
    def test_reports_the_missing_signals_in_declared_order(self):
        offer = AUTOMATION_OFFERS["enrolment_scheduling"]
        self.assertEqual(
            missing_signals(offer, []), ("no booking link", "manual enquiry flow")
        )
        self.assertEqual(
            missing_signals(offer, ["manual enquiry flow"]), ("no booking link",)
        )
        self.assertEqual(
            missing_signals(offer, ["no booking link"]), ("manual enquiry flow",)
        )

    def test_reports_nothing_missing_when_every_required_signal_is_observed(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertEqual(missing_signals(offer, offer.required_signals), ())

    def test_an_empty_observation_misses_every_required_signal(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                self.assertEqual(missing_signals(offer, []), offer.required_signals)

    def test_unrelated_and_unknown_observations_are_ignored(self):
        offer = AUTOMATION_OFFERS["review_response"]
        noise = ["cafe match", "no website", "totally invented signal", "", "  "]
        self.assertEqual(missing_signals(offer, noise), offer.required_signals)
        self.assertEqual(
            missing_signals(offer, [*noise, *offer.required_signals]), ()
        )

    def test_membership_is_exact_and_not_a_loose_match(self):
        # Loose matching is how a rule fires on evidence nobody gathered. Each near miss is
        # a plausible enrichment payload mistake and each must leave the signal missing.
        offer = AUTOMATION_OFFERS["appointment_booking"]
        for observation in NEAR_MISSES:
            with self.subTest(observation=observation):
                self.assertEqual(missing_signals(offer, [observation]), ("no booking link",))

    def test_a_required_signal_is_not_satisfied_by_a_longer_string_containing_it(self):
        offer = AUTOMATION_OFFERS["appointment_booking"]
        self.assertEqual(
            missing_signals(offer, ["site has no booking link anywhere"]),
            ("no booking link",),
        )

    def test_accepts_any_iterable_of_observations(self):
        offer = AUTOMATION_OFFERS["catalog_whatsapp"]
        forms = (
            list(offer.required_signals),
            tuple(offer.required_signals),
            set(offer.required_signals),
            frozenset(offer.required_signals),
            (s for s in offer.required_signals),
            iter(offer.required_signals),
        )
        for form in forms:
            with self.subTest(form=type(form).__name__):
                self.assertEqual(missing_signals(offer, form), ())

    def test_duplicate_observations_change_nothing(self):
        offer = AUTOMATION_OFFERS["review_response"]
        doubled = list(offer.required_signals) * 3
        self.assertEqual(missing_signals(offer, doubled), ())


class OfferFiresTests(unittest.TestCase):
    """Both gates, and the many ways a near miss must fail to become a claim."""

    def test_both_gates_must_pass(self):
        offer = AUTOMATION_OFFERS["order_intake"]
        full = offer.required_signals
        partial = full[:1]
        cases = [
            ("right niche, full evidence", "cafe", full, True),
            ("right niche, partial evidence", "cafe", partial, False),
            ("right niche, no evidence", "cafe", (), False),
            ("wrong niche, full evidence", "salon", full, False),
            ("wrong niche, partial evidence", "salon", partial, False),
            ("unknown niche, full evidence", "not_a_niche", full, False),
        ]
        for name, niche_id, observed, expected in cases:
            with self.subTest(case=name):
                self.assertIs(offer_fires(offer, niche_id, observed), expected)

    def test_nothing_fires_on_an_empty_observation_set(self):
        # The headline negative: no evidence, no claim, for every offer and every niche
        # including the ones the offer was written for.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            for niche_id in (*NICHE_PROFILES, "", "not_a_niche"):
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertFalse(offer_fires(offer, niche_id, []))
                    self.assertFalse(offer_fires(offer, niche_id, ()))
                    self.assertFalse(offer_fires(offer, niche_id, set()))

    def test_nothing_fires_on_observations_outside_the_vocabulary(self):
        junk = [
            "no bookings",
            "website is bad",
            "owner seems busy",
            "high review",
            "runs ads",
            "instagram",
            "true",
            "1",
        ]
        for offer_id, offer in AUTOMATION_OFFERS.items():
            for niche_id in NICHE_PROFILES:
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertFalse(offer_fires(offer, niche_id, junk))

    def test_dropping_any_one_required_signal_sinks_the_offer(self):
        # There is no partial credit. For every multi-signal offer, every one-signal-short
        # observation set must fail -- including the set that is short only the signal a
        # reader would call the least important.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            if len(offer.required_signals) < 2:
                continue
            niche_id = offer.niches[0] if offer.niches else "cafe"
            for dropped in offer.required_signals:
                observed = [s for s in offer.required_signals if s != dropped]
                with self.subTest(offer=offer_id, dropped=dropped):
                    self.assertFalse(offer_fires(offer, niche_id, observed))
                    self.assertEqual(missing_signals(offer, observed), (dropped,))

    def test_a_typo_in_one_signal_sinks_the_offer(self):
        for offer_id, offer in AUTOMATION_OFFERS.items():
            niche_id = offer.niches[0] if offer.niches else "cafe"
            for index, signal in enumerate(offer.required_signals):
                mistyped = list(offer.required_signals)
                mistyped[index] = signal.upper()
                with self.subTest(offer=offer_id, signal=signal):
                    self.assertFalse(offer_fires(offer, niche_id, mistyped))

    def test_a_signal_from_the_wrong_offer_does_not_help(self):
        # `order_intake` and `catalog_whatsapp` both reach bakeries and both concern selling
        # online, so the tempting failure is one satisfying the other.
        order_intake = AUTOMATION_OFFERS["order_intake"]
        catalog = AUTOMATION_OFFERS["catalog_whatsapp"]
        self.assertFalse(offer_fires(order_intake, "bakery", catalog.required_signals))
        self.assertFalse(offer_fires(catalog, "bakery", order_intake.required_signals))

    def test_fires_for_every_declared_niche_when_its_evidence_is_complete(self):
        # The counterweight. Without it a catalogue that never fired would pass every
        # negative test above.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            niches = offer.niches or tuple(NICHE_PROFILES)
            for niche_id in niches:
                with self.subTest(offer=offer_id, niche=niche_id):
                    self.assertTrue(offer_fires(offer, niche_id, offer.required_signals))

    def test_accepts_a_generator_for_a_single_call(self):
        offer = AUTOMATION_OFFERS["service_reminders"]
        self.assertTrue(
            offer_fires(offer, "auto_service", (s for s in offer.required_signals))
        )


class FiringOffersTests(unittest.TestCase):
    def test_nothing_fires_for_any_niche_without_evidence(self):
        for niche_id in (*NICHE_PROFILES, "", "not_a_niche"):
            with self.subTest(niche=niche_id):
                self.assertEqual(firing_offers(niche_id, []), [])

    def test_a_discovery_only_lead_receives_no_automation_pitch(self):
        # The most important negative in this file. Discovery scores a lead from its Google
        # listing alone; those signals are real, and none of them is enough to say anything
        # about how the business takes an order or answers an enquiry. Every offer requires
        # an enrichment pass to have gone and looked, so the answer here is silence.
        for niche_id, profile in NICHE_PROFILES.items():
            for website in (None, "https://instagram.com/x", "https://x.example.com"):
                with self.subTest(niche=niche_id, website=website):
                    observed = scorer_signals_of(profile, website=website)
                    self.assertEqual(firing_offers(niche_id, observed), [])

    def test_not_even_every_scorer_signal_at_once_fires_an_offer(self):
        # The strongest form of the same claim: hand every niche the entire listing-side
        # vocabulary simultaneously, which no single lead could ever produce, and still
        # nothing fires.
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                self.assertEqual(firing_offers(niche_id, SCORER_SIGNALS), [])

    def test_full_evidence_fires_exactly_the_offers_the_niche_is_eligible_for(self):
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                self.assertEqual(
                    firing_offers(niche_id, EVERY_SIGNAL), offers_for_niche(niche_id)
                )

    def test_an_unknown_niche_with_full_evidence_gets_only_the_universal_offers(self):
        for niche_id in ("", "not_a_niche", "CAFE"):
            with self.subTest(niche=niche_id):
                self.assertEqual(
                    [o.id for o in firing_offers(niche_id, EVERY_SIGNAL)],
                    list(UNIVERSAL_OFFER_IDS),
                )

    def test_results_come_back_in_catalogue_order(self):
        # The operator reads them top to bottom; a set-ordered result would reorder the
        # pitch between runs for no reason.
        fired = [o.id for o in firing_offers("bakery", EVERY_SIGNAL)]
        self.assertEqual(fired, [i for i in AUTOMATION_OFFERS if i in fired])
        self.assertEqual(fired, ["order_intake", "catalog_whatsapp", "review_response",
                                 "lead_capture"])

    def test_a_generator_is_walked_once_and_reused(self):
        # The docstring's promise. A naive implementation would exhaust the generator on the
        # first offer and report nothing for the rest.
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                streamed = firing_offers(niche_id, (s for s in sorted(EVERY_SIGNAL)))
                self.assertEqual(streamed, offers_for_niche(niche_id))

    def test_evidence_for_one_niches_offer_does_not_leak_into_another(self):
        # A salon's booking evidence must not produce a cafe's ordering pitch, and vice
        # versa. This is the automation-side equivalent of anti-substitution.
        salon_evidence = niche_specific_signals_for("salon")
        cafe_evidence = niche_specific_signals_for("cafe")
        self.assertEqual([o.id for o in firing_offers("cafe", salon_evidence)], [])
        self.assertEqual([o.id for o in firing_offers("salon", cafe_evidence)], [])
        self.assertEqual(
            [o.id for o in firing_offers("salon", salon_evidence)], ["appointment_booking"]
        )
        self.assertEqual([o.id for o in firing_offers("cafe", cafe_evidence)], ["order_intake"])

    def test_one_offers_evidence_fires_that_offer_and_nothing_else(self):
        # Feeding exactly one offer's requirements must not drag a second offer along. It
        # would if any offer's signals were a subset of another's within a shared niche, and
        # the extra pitch would be making a claim on borrowed evidence.
        for offer_id, offer in AUTOMATION_OFFERS.items():
            niche_id = offer.niches[0] if offer.niches else "not_a_niche"
            with self.subTest(offer=offer_id, niche=niche_id):
                fired = [o.id for o in firing_offers(niche_id, offer.required_signals)]
                self.assertEqual(fired, [offer_id])

    def test_two_offers_may_share_evidence_where_their_niches_overlap(self):
        # The legitimate exception, pinned so it stays deliberate: a dental clinic is both
        # appointment-driven and service-due-driven, and "no booking link" plus a listed
        # number is genuine evidence for both pitches. Neither is inferred from the other.
        fired = [
            o.id for o in firing_offers("dental_clinic", ["no booking link", "public phone"])
        ]
        self.assertEqual(fired, ["appointment_booking", "service_reminders"])


class AdversarialObservationTests(unittest.TestCase):
    """Observation sets of the wrong shape. Each must fail safe, or fail loudly."""

    def test_a_single_signal_passed_as_a_bare_string_fires_nothing(self):
        # `observed` is an iterable of signals, so a bare string iterates as characters. The
        # important part is that this mistake produces silence rather than a claim.
        self.assertEqual(firing_offers("salon", "no booking link"), [])

    def test_non_string_observations_are_ignored_rather_than_matched(self):
        offer = AUTOMATION_OFFERS["review_response"]
        junk = [None, 0, 1, True, 3.5, ("high review count",), frozenset()]
        self.assertFalse(offer_fires(offer, "cafe", junk))
        self.assertTrue(offer_fires(offer, "cafe", [*junk, *offer.required_signals]))

    def test_an_unhashable_observation_raises_rather_than_passing_silently(self):
        # A nested payload is a caller bug. Raising is the right failure: the alternative
        # would be an empty observation set that reads as "we looked and found nothing".
        offer = AUTOMATION_OFFERS["review_response"]
        with self.assertRaises(TypeError):
            offer_fires(offer, "cafe", [["high review count"]])

    def test_a_mapping_of_flags_is_read_as_its_keys_not_its_values(self):
        # Documented, not endorsed. `observed` is an iterable of signals that were observed,
        # and iterating a dict yields its keys -- so a payload shaped
        # {"no booking link": False} fires the offer its own value denies. No caller in this
        # repository does this today; the test exists so that if one starts, it starts by
        # failing here rather than in a sales conversation.
        denied = {"no booking link": False}
        self.assertEqual([o.id for o in firing_offers("salon", denied)], ["appointment_booking"])

    def test_whitespace_and_empty_observations_fire_nothing(self):
        for observation in ("", " ", "\t", "\n"):
            with self.subTest(observation=repr(observation)):
                for niche_id in NICHE_PROFILES:
                    self.assertEqual(firing_offers(niche_id, [observation]), [])

    def test_the_entire_near_miss_vocabulary_fires_nothing_anywhere(self):
        for niche_id in NICHE_PROFILES:
            with self.subTest(niche=niche_id):
                self.assertEqual(firing_offers(niche_id, NEAR_MISSES), [])

    def test_observations_are_not_consumed_from_the_callers_collection(self):
        observed = set(EVERY_SIGNAL)
        firing_offers("cafe", observed)
        self.assertEqual(observed, set(EVERY_SIGNAL))


class AutomationPayloadTests(unittest.TestCase):
    def test_one_row_per_offer_in_catalogue_order(self):
        self.assertEqual([row["id"] for row in automation_payload()], list(AUTOMATION_OFFERS))

    def test_every_row_reproduces_its_offer_exactly(self):
        rows = {row["id"]: row for row in automation_payload()}
        for offer_id, offer in AUTOMATION_OFFERS.items():
            with self.subTest(offer=offer_id):
                row = rows[offer_id]
                self.assertEqual(row["label"], offer.label)
                self.assertEqual(row["niches"], list(offer.niches))
                self.assertEqual(row["required_signals"], list(offer.required_signals))
                self.assertEqual(row["pitch_line"], offer.pitch_line)
                self.assertEqual(row["est_hours_saved_weekly"], offer.est_hours_saved_weekly)

    def test_payload_is_json_serialisable_without_a_custom_encoder(self):
        # It is served by the API and read by the operator; a tuple that survived into the
        # payload would only fail at response time.
        self.assertEqual(json.loads(json.dumps(automation_payload())), automation_payload())

    def test_mutating_the_payload_cannot_reach_the_catalogue(self):
        # The payload hands out lists. If they were the offer's own data a single API caller
        # could rewrite the catalogue for the process.
        payload = automation_payload()
        payload[0]["niches"].append("invented_niche")
        payload[0]["required_signals"].clear()
        payload[0]["est_hours_saved_weekly"] = 999.0

        offer = AUTOMATION_OFFERS[payload[0]["id"]]
        self.assertNotIn("invented_niche", offer.niches)
        self.assertTrue(offer.required_signals)
        self.assertNotEqual(offer.est_hours_saved_weekly, 999.0)
        self.assertEqual(automation_payload()[0]["niches"], list(offer.niches))

    def test_payload_niches_are_registry_ids(self):
        for row in automation_payload():
            for niche_id in row["niches"]:
                with self.subTest(offer=row["id"], niche=niche_id):
                    self.assertIn(niche_id, NICHE_PROFILES)


class OfferShapeTests(unittest.TestCase):
    def test_a_hand_built_offer_needs_no_niches(self):
        # `_offer` defaults `niches` to empty, which is what makes an offer universal. Built
        # directly, the same default must hold, or a new universal offer would be a
        # TypeError away from being written as a niche-specific one by accident.
        offer = AutomationOffer(
            id="x",
            label="X",
            niches=(),
            required_signals=("no website",),
            pitch_line="...",
            est_hours_saved_weekly=1.0,
        )
        self.assertTrue(applies_to_niche(offer, "anything"))
        self.assertTrue(offer_fires(offer, "anything", ["no website"]))
        self.assertFalse(offer_fires(offer, "anything", []))

    def test_a_signal_free_offer_would_fire_on_nothing_at_all(self):
        # Why `test_every_offer_requires_at_least_one_signal` exists: an offer with no
        # required signals fires on an empty observation set, for every business in its
        # niches, with no evidence whatsoever. The catalogue contains none, and this
        # demonstrates the consequence if one were added.
        empty = AutomationOffer(
            id="empty",
            label="Empty",
            niches=(),
            required_signals=(),
            pitch_line="...",
            est_hours_saved_weekly=1.0,
        )
        self.assertTrue(offer_fires(empty, "cafe", []))
        self.assertNotIn(empty, AUTOMATION_OFFERS.values())


if __name__ == "__main__":
    unittest.main()
