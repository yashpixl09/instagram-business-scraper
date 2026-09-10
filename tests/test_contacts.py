"""Phase 5 (narrow slice): a named contact, read from the homepage text `_grade` already paid for.

WHAT THESE TESTS ARE DEFENDING

*Never guess.* A fabricated owner name is worse than a blank field the moment an operator
opens with it. `find_contacts` must record a person only when a name is clearly present AND
clearly tied -- in the same line, or a name-only line paired with the very next detail lines
-- to a role, a phone, or an email. A bare name gets nothing. A bare email or phone gets
nothing. Two facts that merely share a page, but not a line, get nothing.

*`website` and `ig_bio` only.* `review_reply`, `directory` and `search` all need capability
or budget this slice does not have -- see `lead_engine/enrichment/contacts.py`'s module
docstring for the individual blockers. `ig_bio` shares `find_contacts`' extraction engine
via `find_bio_contacts`, at 0.6 confidence instead of `website`'s 0.8; `IgBioTests` below
confirms the same never-guess rules hold over bio text and that the two sources never bleed
into each other's confidence or source label. The wiring test at the bottom checks
`EnrichmentService` writes a `contacts` row only when grading text was fetched and a
candidate was found in it -- that wiring is `website`-only; `ig_bio` has no pipeline caller
yet (see the module docstring).

NO TEST HERE REACHES THE NETWORK.
"""

from __future__ import annotations

import unittest
import uuid

from lead_engine.enrichment.cache import InMemoryCache
from lead_engine.enrichment.contacts import (
    CONFIDENCE_IG_BIO,
    ROLE_MANAGER,
    ROLE_MARKETING,
    ROLE_OWNER,
    ROLE_UNKNOWN,
    SOURCE_IG_BIO,
    SOURCE_WEBSITE,
    ContactCandidate,
    ContactExtraction,
    find_bio_contacts,
    find_contacts,
)
from lead_engine.enrichment.service import EnrichmentRequest, EnrichmentService

NOW_KW = {"cache": InMemoryCache()}


def service(primary, **kwargs) -> EnrichmentService:
    kwargs.setdefault("cache", InMemoryCache())
    return EnrichmentService(primary, **kwargs)


class Result:
    """One search hit, in the shape `TinyFishClient.search` returns."""

    def __init__(self, url: str, title: str = "", snippet: str = "") -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


class FakeWeb:
    """A search-and-fetch provider over canned answers. Mirrors `tests/test_enrichment.py`."""

    def __init__(self, results=None, pages=None) -> None:
        self.results = list(results or ())
        self.pages = dict(pages or {})
        self.fetches: list[str] = []

    def search(self, query: str, limit: int = 10):
        return self.results[:limit]

    def fetch(self, url: str) -> str:
        self.fetches.append(url)
        # The ad-library check fetches a URL this suite never seeds; a clean "no ads"
        # page is a real, cheap answer to it and keeps every test focused on contacts.
        return self.pages.get(url, "<html><body>No ads found</body></html>")


class RecordingStore:
    """`enrichments` and `contacts`, each in its own list."""

    def __init__(self) -> None:
        self.enrichment_rows: list[dict] = []
        self.contact_rows: list[dict] = []

    def insert_enrichment(
        self, business_id, source, status, data, *, source_url=None, run_id=None
    ):
        self.enrichment_rows.append(
            {"business_id": business_id, "source": source, "status": status}
        )
        return object()

    def insert_contact(
        self,
        business_id,
        name,
        source,
        *,
        role="unknown",
        phone=None,
        email=None,
        source_url=None,
        confidence=0.5,
    ):
        self.contact_rows.append(
            {
                "business_id": business_id,
                "name": name,
                "role": role,
                "phone": phone,
                "email": email,
                "source": source,
                "source_url": source_url,
                "confidence": confidence,
            }
        )
        return object()


def request(**overrides) -> EnrichmentRequest:
    base = dict(business_id=uuid.uuid4(), name="Cake Bee", city="Bangalore")
    base.update(overrides)
    return EnrichmentRequest(**base)


# --- pure extraction -------------------------------------------------------------------


class NamedBlockTests(unittest.TestCase):
    def test_a_clean_owner_card_yields_name_role_and_email(self):
        text = (
            "Meet the Owner\n\n"
            "Priya Sharma\n"
            "Founder & Owner\n"
            "priya@cakebee.in | +91 98765 43210\n"
        )
        extraction = find_contacts(text, source_url="https://cakebee.in")
        self.assertEqual(len(extraction.candidates), 1)
        candidate = extraction.candidates[0]
        self.assertEqual(candidate.name, "Priya Sharma")
        self.assertEqual(candidate.role, ROLE_OWNER)
        self.assertEqual(candidate.email, "priya@cakebee.in")
        self.assertIsNotNone(candidate.phone)
        self.assertEqual(candidate.source, SOURCE_WEBSITE)
        self.assertEqual(candidate.source_url, "https://cakebee.in")
        self.assertEqual(candidate.confidence, 0.8)

    def test_a_signed_line_pairs_the_role_with_the_name(self):
        text = (
            "Thanks for baking with us all these years.\n"
            "-- Founder Priya Sharma\n"
        )
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)
        self.assertEqual(extraction.candidates[0].name, "Priya Sharma")
        self.assertEqual(extraction.candidates[0].role, ROLE_OWNER)

    def test_a_labelled_contact_line_with_no_role_word_is_still_recorded_as_unknown(self):
        # "Contact:" names a channel, not a role. Per the never-infer-a-role rule, the role
        # stays `unknown` -- the email is still a clean, attributable fact worth keeping.
        text = "Contact: Priya Sharma, priya@cakebee.in"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)
        self.assertEqual(extraction.candidates[0].name, "Priya Sharma")
        self.assertEqual(extraction.candidates[0].role, ROLE_UNKNOWN)
        self.assertEqual(extraction.candidates[0].email, "priya@cakebee.in")

    def test_marketing_manager_is_marketing_not_manager(self):
        # Never infer a role from context -- but the label itself must still be read
        # correctly: "Marketing Manager" is the marketing role, not the generic one.
        text = "Marketing Manager: Raj Kumar, raj@cakebee.in"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)
        self.assertEqual(extraction.candidates[0].role, ROLE_MARKETING)

    def test_a_plain_manager_label_is_the_manager_role(self):
        text = "Manager: Raj Kumar, raj@cakebee.in"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)
        self.assertEqual(extraction.candidates[0].role, ROLE_MANAGER)


class NeverGuessTests(unittest.TestCase):
    def test_a_role_word_inside_a_markdown_heading_records_no_name(self):
        # Found live, on a real suspended-hosting placeholder page: "## For Website
        # Owners" masks "Owners" via the role pattern and leaves "For Website" behind --
        # title case, two words, not a stopword, and not a person. Headings are markup,
        # not prose, and get skipped from extraction entirely rather than trying to
        # enumerate every generic heading phrase as a stopword.
        text = (
            "# Service Suspended\n\n"
            "The website owner or hosting provider has suspended this service.\n\n"
            "## For Website Owners\n\n"
            "If you are the owner of this website, please contact your hosting provider.\n"
        )
        self.assertEqual(find_contacts(text, source_url=None).candidates, ())

    def test_a_heading_does_not_count_as_a_staff_cards_detail_line(self):
        # A heading between a bare name and its real detail must not be treated as detail
        # for that name (it would never match role/phone/email anyway) nor eat into the
        # two-line detail budget that lets a search give up on a genuine non-match.
        text = "Priya Sharma\n\n## Our Team\n\nFounder, priya@cakebee.in\n"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)
        self.assertEqual(extraction.candidates[0].name, "Priya Sharma")
        self.assertEqual(extraction.candidates[0].email, "priya@cakebee.in")

    def test_a_name_with_nothing_nearby_records_nothing(self):
        # Two names in a row with no role, phone or email attached to either -- a staff
        # list, not a lead. Recording either would be a guess about who does what.
        text = (
            "Meet our team.\n\n"
            "Priya Sharma\n"
            "Raj Kumar\n\n"
            "We love serving fresh cakes every day of the week.\n"
        )
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(extraction.candidates, ())

    def test_an_email_with_no_attributable_name_records_nothing(self):
        text = "General enquiries: info@cakebee.in\nCall us on 080 4111 2222 any day.\n"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(extraction.candidates, ())

    def test_a_phone_with_no_attributable_name_records_nothing(self):
        text = "For orders call 9876543210 between 9am and 8pm every day of the week.\n"
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(extraction.candidates, ())

    def test_none_input_records_nothing(self):
        extraction = find_contacts(None, source_url=None)
        self.assertIsInstance(extraction, ContactExtraction)
        self.assertEqual(extraction.candidates, ())

    def test_empty_string_records_nothing(self):
        self.assertEqual(find_contacts("", source_url=None).candidates, ())

    def test_malformed_garbage_does_not_crash_and_records_nothing(self):
        garbage = "\x00\x01 <<>> ???  \n\n\t\t   ---- \n" * 3
        extraction = find_contacts(garbage, source_url=None)
        self.assertEqual(extraction.candidates, ())


class MultipleContactsTests(unittest.TestCase):
    def test_two_distinct_named_contacts_on_one_page_are_both_recorded(self):
        text = (
            "Owner: Priya Sharma, priya@cakebee.in\n"
            "Marketing: Raj Kumar, raj@cakebee.in\n"
        )
        extraction = find_contacts(text, source_url="https://cakebee.in")
        names = {c.name for c in extraction.candidates}
        roles = {c.name: c.role for c in extraction.candidates}
        self.assertEqual(names, {"Priya Sharma", "Raj Kumar"})
        self.assertEqual(roles["Priya Sharma"], ROLE_OWNER)
        self.assertEqual(roles["Raj Kumar"], ROLE_MARKETING)

    def test_the_same_name_mentioned_twice_is_recorded_once(self):
        text = (
            "Owner: Priya Sharma, priya@cakebee.in\n"
            "Questions for Priya Sharma? Call 9876543210.\n"
        )
        extraction = find_contacts(text, source_url=None)
        self.assertEqual(len(extraction.candidates), 1)


class IgBioTests(unittest.TestCase):
    """`find_bio_contacts` -- the `ig_bio` source, sharing `find_contacts`' engine at 0.6."""

    def test_a_named_owner_in_a_bio_is_recorded_at_ig_bio_confidence(self):
        bio = "Owner: Priya Sharma | priya@cakebee.in | Custom cakes, Indiranagar"
        extraction = find_bio_contacts(bio, source_url="https://instagram.com/cakebee/")
        self.assertEqual(len(extraction.candidates), 1)
        candidate = extraction.candidates[0]
        self.assertEqual(candidate.name, "Priya Sharma")
        self.assertEqual(candidate.role, ROLE_OWNER)
        self.assertEqual(candidate.source, SOURCE_IG_BIO)
        self.assertEqual(candidate.confidence, CONFIDENCE_IG_BIO)

    def test_website_confidence_is_unaffected_by_ig_bio_existing(self):
        # A regression guard on find_contacts' new default arguments: an existing website
        # caller that never learned about `source`/`confidence` must see the old behaviour.
        text = "Owner: Priya Sharma, priya@cakebee.in"
        extraction = find_contacts(text, source_url="https://cakebee.in")
        candidate = extraction.candidates[0]
        self.assertEqual(candidate.source, SOURCE_WEBSITE)
        self.assertNotEqual(candidate.confidence, CONFIDENCE_IG_BIO)

    def test_a_bio_with_no_attributable_name_records_nothing(self):
        # Same never-guess rule as a homepage: a bare contact detail in a bio is not a
        # contact just because it is the only thing on the page.
        bio = "DM us or call 9876543210 for custom orders!"
        self.assertEqual(find_bio_contacts(bio).candidates, ())

    def test_a_bio_with_no_name_or_detail_records_nothing(self):
        bio = "Fresh cakes daily. Indiranagar, Bangalore. Order now!"
        self.assertEqual(find_bio_contacts(bio).candidates, ())

    def test_none_bio_records_nothing(self):
        self.assertEqual(find_bio_contacts(None).candidates, ())

    def test_source_url_defaults_to_none_and_is_carried_through(self):
        extraction = find_bio_contacts("Founder: Raj Kumar, raj@cakebee.in")
        self.assertIsNone(extraction.candidates[0].source_url)
        extraction = find_bio_contacts(
            "Founder: Raj Kumar, raj@cakebee.in", source_url="https://instagram.com/cakebee/"
        )
        self.assertEqual(extraction.candidates[0].source_url, "https://instagram.com/cakebee/")


class DataclassTests(unittest.TestCase):
    def test_candidate_as_dict_carries_every_field(self):
        candidate = ContactCandidate(
            name="Priya Sharma",
            role=ROLE_OWNER,
            phone=None,
            email="priya@cakebee.in",
            source=SOURCE_WEBSITE,
            source_url="https://cakebee.in",
            confidence=0.8,
        )
        payload = candidate.as_dict()
        self.assertEqual(payload["name"], "Priya Sharma")
        self.assertEqual(payload["role"], ROLE_OWNER)
        self.assertEqual(payload["email"], "priya@cakebee.in")
        self.assertEqual(payload["source"], SOURCE_WEBSITE)
        self.assertEqual(payload["confidence"], 0.8)


# --- service wiring ----------------------------------------------------------------------


class ServiceWiringTests(unittest.TestCase):
    """`EnrichmentService` writes a `contacts` row only off text it already fetched to grade."""

    def test_a_contact_is_written_when_grading_text_names_one(self):
        site_text = (
            "<html><head><meta name=\"viewport\" content=\"width=device-width\"></head>"
            "<body><h1>Our Menu</h1><p>Fresh cakes baked daily in Indiranagar. Browse our "
            "catalogue of celebration cakes, cupcakes and custom orders for every "
            "occasion we bake for.</p>"
            "<p>Owner: Priya Sharma, priya@cakebee.in</p>"
            "<a href=\"/order\">Order online</a><form action=\"/enquiry\">"
            "<input name=\"phone\"></form></body></html>"
        )
        store = RecordingStore()
        web = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee | Custom cakes")],
            pages={"https://cakebee.in": site_text},
        )
        service(web, store=store).enrich(request())

        self.assertEqual(len(store.contact_rows), 1)
        row = store.contact_rows[0]
        self.assertEqual(row["name"], "Priya Sharma")
        self.assertEqual(row["role"], ROLE_OWNER)
        self.assertEqual(row["email"], "priya@cakebee.in")
        self.assertEqual(row["source"], SOURCE_WEBSITE)
        self.assertEqual(row["source_url"], "https://cakebee.in")
        self.assertEqual(row["confidence"], 0.8)

    def test_no_contact_row_when_the_site_text_names_nobody(self):
        site_text = (
            "<html><head><meta name=\"viewport\" content=\"width=device-width\"></head>"
            "<body><h1>Our Menu</h1><p>Fresh cakes baked daily in Indiranagar. Browse our "
            "catalogue of celebration cakes, cupcakes and custom orders for every "
            "occasion we bake for.</p>"
            "<a href=\"/order\">Order online</a><form action=\"/enquiry\">"
            "<input name=\"phone\"></form>"
            "<p>Call 080 4111 2222 or visit the shop on 100 Feet Road.</p></body></html>"
        )
        store = RecordingStore()
        web = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee | Custom cakes")],
            pages={"https://cakebee.in": site_text},
        )
        service(web, store=store).enrich(request())

        self.assertEqual(store.contact_rows, [])
        # The website row itself must still be written -- absence of a contact must not
        # be confused with absence of a website.
        self.assertTrue(store.enrichment_rows)

    def test_no_contact_row_when_no_site_was_found_at_all(self):
        store = RecordingStore()
        web = FakeWeb(results=[])
        service(web, store=store).enrich(request())

        self.assertEqual(store.contact_rows, [])

    def test_no_contact_row_when_grading_is_disabled(self):
        store = RecordingStore()
        web = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee")],
            pages={"https://cakebee.in": "irrelevant, never fetched"},
        )
        service(web, store=store, grade_sites=False).enrich(request())

        self.assertEqual(store.contact_rows, [])
        self.assertNotIn(
            "https://cakebee.in",
            web.fetches,
            "grading is disabled, so the site page should never be fetched",
        )


if __name__ == "__main__":
    unittest.main()
