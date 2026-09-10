"""`best_reach_channel`: the operator's own stated priority -- mobile, then social, then
email last -- built on provenance rather than guessing a phone number's type from its
digits.

THE REAL FAILURE THIS MODULE REFUSES TO REPEAT

A naive "starts with 6/7/8/9 is a mobile" check is wrong on real data from this exact
project: Corner House Ice Cream's real number, a Bangalore LANDLINE ("080 2520 3364"),
strips to "8025203364" -- indistinguishable in shape from a mobile, because Bangalore's own
STD code is "80". `RankingTests` pins that this module never even looks at a phone
number's digits to decide anything.
"""

from __future__ import annotations

import unittest

from lead_engine.reachability import (
    EMAIL,
    INSTAGRAM,
    MOBILE,
    NONE,
    PHONE_UNKNOWN_TYPE,
    best_reach_channel,
)


class RankingTests(unittest.TestCase):
    def test_a_named_contacts_phone_wins_over_everything(self):
        result = best_reach_channel(
            contact_phone="9876543210",
            contact_name="Priya Sharma",
            instagram_handle="cakebee",
            business_phone="08041234567",
            contact_email="priya@cakebee.in",
            business_email="hello@cakebee.in",
        )
        self.assertEqual(result.channel, MOBILE)
        self.assertEqual(result.value, "9876543210")
        self.assertIn("Priya Sharma", result.note)

    def test_instagram_beats_the_businesss_own_ambiguous_phone(self):
        result = best_reach_channel(instagram_handle="cakebee", business_phone="08041234567")
        self.assertEqual(result.channel, INSTAGRAM)
        self.assertEqual(result.value, "cakebee")

    def test_the_businesss_phone_is_ranked_but_never_called_a_mobile(self):
        # The real, proven failure case: a Bangalore landline shaped identically to a
        # mobile number. This module must rank it below instagram and must never claim
        # to know its type.
        result = best_reach_channel(business_phone="+91 80 2520 3364")
        self.assertEqual(result.channel, PHONE_UNKNOWN_TYPE)
        self.assertNotEqual(result.channel, MOBILE)
        self.assertIn("not knowable", result.note)

    def test_a_named_contacts_email_beats_a_general_inbox(self):
        result = best_reach_channel(
            contact_email="priya@cakebee.in", business_email="hello@cakebee.in"
        )
        self.assertEqual(result.channel, EMAIL)
        self.assertEqual(result.value, "priya@cakebee.in")

    def test_a_general_inbox_is_the_last_resort(self):
        result = best_reach_channel(business_email="hello@cakebee.in")
        self.assertEqual(result.channel, EMAIL)
        self.assertEqual(result.value, "hello@cakebee.in")

    def test_nothing_found_is_reported_as_none_not_a_guess(self):
        result = best_reach_channel()
        self.assertEqual(result.channel, NONE)
        self.assertIsNone(result.value)
        self.assertFalse(result.found)

    def test_found_is_true_for_every_real_channel(self):
        for kwargs in (
            {"contact_phone": "9876543210"},
            {"instagram_handle": "cakebee"},
            {"business_phone": "08041234567"},
            {"contact_email": "a@b.in"},
            {"business_email": "a@b.in"},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(best_reach_channel(**kwargs).found)


class NeverGuessesFromDigitsTests(unittest.TestCase):
    def test_a_contact_phone_shaped_like_a_landline_is_still_reported_as_mobile(self):
        # Deliberate: provenance, not shape, decides. A number attached to a named
        # contact is trusted as likely-personal regardless of what it looks like -- the
        # module's whole point is refusing to let digit shape override provenance in
        # EITHER direction.
        result = best_reach_channel(contact_phone="08041234567", contact_name="Priya")
        self.assertEqual(result.channel, MOBILE)

    def test_business_phone_starting_with_a_mobile_looking_digit_is_still_unknown_type(self):
        result = best_reach_channel(business_phone="9876543210")
        self.assertEqual(result.channel, PHONE_UNKNOWN_TYPE)


if __name__ == "__main__":
    unittest.main()
