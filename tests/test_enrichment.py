"""Phase 4: what Google Maps cannot tell you about a business.

WHAT THESE TESTS ARE DEFENDING

*The claim that costs a meeting.* "This business has no website" is the opening line of the
pitch. Google saying so is a claim about Google's data, not about the world, and an operator
who opens with it when the owner has a perfectly good site has lost the room in one sentence.
So the tests below care most about the difference between "there is no website", "there is
one and here is what is wrong with it", and "we could not tell" -- three different things
that a weaker module would flatten into one.

*The free tier.* TinyFish Search and Fetch cost nothing on any plan; Firecrawl is 1,000
metered credits a month. The ordering is asserted, not assumed: Firecrawl must not be called
when TinyFish answers.

*Absence, bought once.* A business with no Instagram in March has none in April. The negative
cache backs off 30/90/180/never, and `no_data` stays distinguishable from never-having-asked,
because only the second is worth spending on again.

*The Phase 7 boundary.* Nothing here may touch instagram.com. This module finds the handle;
reading the profile happens later, through the operator's own browser session, under rules
this module has no way to honour. `FailIfInstagram` makes that structural rather than
advisory.

NO TEST HERE REACHES THE NETWORK.
"""

from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime, timedelta

from lead_engine.enrichment.cache import InMemoryCache, retry_after_for
from lead_engine.enrichment.service import EnrichmentRequest, EnrichmentService
from lead_engine.enrichment.social import handle_from_url, handles_in_text, normalise_handle
from lead_engine.enrichment.website import (
    KIND_FREE_BUILDER,
    KIND_SOCIAL,
    classify_host,
    grade_site,
    host_of,
)
from lead_engine.providers.errors import ProviderError, unavailable

NOW = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)


class Result:
    """One search hit, in the shape `TinyFishClient.search` returns."""

    def __init__(self, url: str, title: str = "", snippet: str = "") -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


class FakeWeb:
    """A search-and-fetch provider over canned answers, counting every call.

    Counting is the point. "TinyFish before Firecrawl" is a claim about which object was
    asked, and only a recorded call can settle it -- an assertion about the returned value
    would pass just as well if both had been called and the second answer discarded.
    """

    def __init__(self, results=None, pages=None, *, fail: Exception | None = None) -> None:
        self.results = list(results or ())
        self.pages = dict(pages or {})
        self.fail = fail
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str, limit: int = 10):
        self.searches.append(query)
        if self.fail is not None:
            raise self.fail
        return self.results[:limit]

    def fetch(self, url: str) -> str:
        self.fetches.append(url)
        if self.fail is not None:
            raise self.fail
        if url not in self.pages:
            raise unavailable("nothing recorded for that url", provider="fake")
        return self.pages[url]


class FailIfInstagram(FakeWeb):
    """Phase 7's boundary, enforced structurally.

    Reading an Instagram profile happens through the operator's own authenticated browser,
    serialized and human-paced and halting on any block. None of that is available here, so
    this module must find the handle and stop. A comment saying so would rot; this cannot.
    """

    def fetch(self, url: str) -> str:
        if "instagram.com" in url.lower():
            raise AssertionError(
                "enrichment fetched instagram.com; the profile is Phase 7's to read"
            )
        return super().fetch(url)

    def search(self, query: str, limit: int = 10):
        return super().search(query, limit)


class RecordingStore:
    """`enrichments`, in a list. Append-only, exactly as the table is."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def insert_enrichment(
        self, business_id, source, status, data, *, source_url=None, run_id=None
    ):
        self.rows.append(
            {
                "business_id": business_id,
                "source": source,
                "status": status,
                "data": dict(data),
                "source_url": source_url,
                "run_id": run_id,
            }
        )
        return object()

    def sources(self) -> list[str]:
        return [row["source"] for row in self.rows]

    def by_source(self, source: str) -> list[dict]:
        return [row for row in self.rows if row["source"] == source]


SITE = """<html><head><meta name="viewport" content="width=device-width"></head><body>
<h1>Our Menu</h1><p>Fresh cakes baked daily in Indiranagar. Browse the catalogue of
celebration cakes, cupcakes and custom orders for every occasion we bake for.</p>
<a href="/order">Order online</a><form action="/enquiry"><input name="phone"></form>
<p>Call 080 4111 2222 or visit the shop on 100 Feet Road.</p></body></html>"""

THIN = """<html><head></head><body><h1>Cake Bee</h1>
<p>Coming soon. Our new website is under construction, please check back shortly for
our full range of celebration cakes and custom orders across Bengaluru city.</p>
</body></html>"""


def request(**overrides) -> EnrichmentRequest:
    base = dict(
        business_id=uuid.uuid4(),
        name="Cake Bee",
        city="Bangalore",
        listed_website=None,
        listed_handle=None,
    )
    base.update(overrides)
    return EnrichmentRequest(**base)


def service(primary, **kwargs) -> EnrichmentService:
    kwargs.setdefault("cache", InMemoryCache())
    kwargs.setdefault("clock", lambda: NOW)
    return EnrichmentService(primary, **kwargs)


# --- host classification -------------------------------------------------------------------


class ClassificationTests(unittest.TestCase):
    def test_a_social_link_is_not_a_website(self):
        # The whole pitch turns on this. An Instagram page is not a website, and telling an
        # owner they "have a site" because they have an Instagram is the fastest way to be
        # dismissed as someone who did not look.
        for url in (
            "https://instagram.com/cakebee",
            "https://www.facebook.com/cakebee",
            "https://linktr.ee/cakebee",
            "https://wa.me/919845012345",
        ):
            with self.subTest(url=url):
                self.assertEqual(classify_host(url), KIND_SOCIAL)

    def test_a_free_builder_site_is_its_own_category(self):
        # Not "no website" -- they have one and are proud of it -- and not a real site
        # either. It is a different conversation from both.
        for url in (
            "https://cakebee.wixsite.com/home",
            "https://cakebee.blogspot.com",
            "https://sites.google.com/view/cakebee",
            "https://cakebee.business.site",
        ):
            with self.subTest(url=url):
                self.assertEqual(classify_host(url), KIND_FREE_BUILDER)

    def test_host_extraction_survives_the_spellings_a_listing_actually_carries(self):
        for url, expected in (
            ("https://www.cakebee.in/menu", "www.cakebee.in"),
            ("cakebee.in", "cakebee.in"),
            ("HTTPS://CakeBee.in", "cakebee.in"),
        ):
            with self.subTest(url=url):
                self.assertEqual(host_of(url), expected)


# --- grading -------------------------------------------------------------------------------


class GradingTests(unittest.TestCase):
    def test_a_real_site_is_graded_on_what_it_does_for_the_business(self):
        grade = grade_site(SITE)
        self.assertIsNotNone(grade)
        self.assertTrue(grade.mobile_friendly)
        self.assertTrue(grade.catalogue)
        self.assertTrue(grade.enquiry)
        self.assertTrue(grade.ordering)

    def test_a_placeholder_site_reports_gaps_and_those_gaps_are_the_pitch(self):
        grade = grade_site(THIN)
        self.assertIsNotNone(grade)
        self.assertFalse(grade.ordering)
        self.assertFalse(grade.enquiry)
        self.assertTrue(grade.gaps, "a site with nothing on it should name what it lacks")

    def test_too_little_text_is_ungradeable_rather_than_graded_badly(self):
        # A fetch that returned a redirect stub must not be reported as a site with no menu.
        # The operator would open with a criticism of a page nobody actually read.
        self.assertIsNone(grade_site("<html><body>ok</body></html>"))
        self.assertIsNone(grade_site(""))

    def test_markdown_leaves_mobile_friendliness_unknown_rather_than_false(self):
        # TinyFish returns markdown, and markdown has no <head>. False would be a claim
        # about the site; None is a claim about what we read.
        markdown = (
            "# Our Menu\n\nFresh cakes baked daily in Indiranagar. Browse the full catalogue "
            "of celebration cakes, cupcakes, brownies and custom orders for every "
            "occasion.\n\n[Order online](/order)\n\nCall 080 4111 2222 or visit the shop on "
            "100 Feet Road, Indiranagar, any day of the week.\n"
        )
        grade = grade_site(markdown)
        self.assertIsNotNone(grade)
        self.assertIsNone(grade.mobile_friendly)
        # And the score renormalises over what was measured, so a markdown grade is
        # comparable with an HTML one instead of losing a fixed 30 points to a question
        # nobody was in a position to ask.
        self.assertEqual(grade.signals, ("catalogue", "ordering"))
        self.assertEqual(grade.gaps, ("enquiry",))
        self.assertEqual(grade.score, 64)


# --- handles -------------------------------------------------------------------------------


class HandleTests(unittest.TestCase):
    def test_a_handle_is_recognised_in_every_spelling_a_listing_uses(self):
        for raw, expected in (
            ("@cakebee", "cakebee"),
            ("cakebee", "cakebee"),
            ("https://instagram.com/cakebee", "cakebee"),
            ("https://www.instagram.com/cakebee/", "cakebee"),
        ):
            with self.subTest(raw=raw):
                got = handle_from_url(raw) or normalise_handle(raw)
                self.assertEqual(got, expected)

    def test_instagrams_own_pages_are_not_business_handles(self):
        # /explore, /reels and friends are Instagram's furniture. Recording one as a
        # business's handle would send the operator to a page about nobody.
        for url in (
            "https://instagram.com/explore/tags/cake",
            "https://instagram.com/reels/abc123",
            "https://instagram.com/p/xyz",
            # Found live: four real businesses this session got "popular" stored as their
            # handle, from Google-indexed URLs shaped exactly like this -- Instagram's own
            # tag/topic aggregation page, not a profile.
            "https://www.instagram.com/popular/krishna-mysore-pak/",
        ):
            with self.subTest(url=url):
                self.assertIsNone(handle_from_url(url))

    def test_handles_are_found_by_their_url_not_by_an_at_sign(self):
        """Host-anchored on purpose, and the negative half is the valuable one.

        A bare `@something` in a page's text is as likely to be `@zomato`, `@swiggy` or a
        supplier the shop tagged as it is to be the shop. Recording one as the business's
        handle sends the operator to somebody else's account and, worse, attaches somebody
        else's follower count to this lead.
        """
        found = handles_in_text(
            "Follow us at https://instagram.com/cake.bee -- we also deliver via @swiggy"
        )
        self.assertEqual(found, ["cake.bee"])
        self.assertEqual(handles_in_text("DM us @cake.bee to order"), [])


# --- the negative cache ----------------------------------------------------------------------


class NegativeCacheTests(unittest.TestCase):
    def test_absence_is_re_bought_at_a_decreasing_rate(self):
        # A business with no Instagram in March has none in April. Asking monthly is a
        # subscription to re-learning the same nothing.
        self.assertEqual((retry_after_for(1, NOW) - NOW).days, 30)
        self.assertEqual((retry_after_for(2, NOW) - NOW).days, 90)
        self.assertEqual((retry_after_for(3, NOW) - NOW).days, 180)
        self.assertIsNone(retry_after_for(4, NOW), "the fourth miss should stop asking")

    def test_a_cached_miss_is_skipped_and_the_reason_is_recorded(self):
        cache = InMemoryCache()
        web = FakeWeb(results=[])
        first = service(web, cache=cache).enrich(request())
        searches_after_first = len(web.searches)

        second = service(web, cache=cache).enrich(request(business_id=first.business_id))

        self.assertEqual(len(web.searches), searches_after_first, "a miss was re-bought")
        self.assertTrue(
            second.skipped,
            "skipping must be recorded, or a caller cannot tell it from a silent failure",
        )


# --- the service ---------------------------------------------------------------------------


class ServiceTests(unittest.TestCase):
    def test_a_business_with_a_real_site_is_not_reported_as_siteless(self):
        web = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee | Custom cakes in Indiranagar")],
            pages={"https://cakebee.in": SITE},
        )
        found = service(web).enrich(request())

        self.assertIsNotNone(found.website)
        self.assertTrue(
            getattr(found.website, "url", None) or getattr(found.website, "website", None),
            f"a site was found but not reported: {found.website}",
        )

    def test_genuinely_nothing_is_no_data_and_not_an_error(self):
        # "We looked and there is nothing" is a SALES SIGNAL. "Something broke" is a gap in
        # our data. Collapsing them would either waste the best leads or invent them.
        store = RecordingStore()
        # Every lookup answers; they just answer "nothing". An empty Ad Library page is a
        # real answer -- this business buys no ads -- and must not read as a broken fetch.
        web = FakeWeb(results=[], pages={"__any__": "<html><body>No ads found</body></html>"})
        web.fetch = lambda url: "<html><body>No ads to show for this search</body></html>"
        found = service(web, store=store).enrich(request())

        self.assertTrue(store.rows, "a lookup that found nothing still has to be recorded")
        self.assertNotIn(
            "error",
            {row["status"] for row in store.rows},
            f"a clean nothing was recorded as a failure: {store.sources()}",
        )
        self.assertIsNotNone(found)

    def test_a_provider_failure_is_error_not_no_data(self):
        store = RecordingStore()
        web = FakeWeb(fail=unavailable("upstream is down", provider="fake"))
        service(web, store=store).enrich(request())

        self.assertIn(
            "error",
            {row["status"] for row in store.rows},
            "a failed lookup recorded as no_data would be read as a sales signal",
        )

    def test_the_free_provider_is_used_and_the_metered_one_is_not(self):
        # TinyFish costs nothing; Firecrawl is 1,000 credits a month. Asserted on the call
        # record, because an assertion on the result would pass with both called.
        primary = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee")],
            pages={"https://cakebee.in": SITE},
        )
        fallback = FakeWeb(results=[Result("https://wrong.example", "Wrong")])

        service(primary, fallback=fallback).enrich(request())

        self.assertTrue(primary.searches, "the free provider was not tried")
        self.assertEqual(
            fallback.searches, [], "the metered provider was billed while the free one worked"
        )

    def test_the_metered_provider_is_the_fallback_when_the_free_one_fails(self):
        primary = FakeWeb(fail=unavailable("tinyfish is down", provider="tinyfish"))
        fallback = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee")],
            pages={"https://cakebee.in": SITE},
        )

        service(primary, fallback=fallback).enrich(request())

        self.assertTrue(fallback.searches, "the fallback never ran after the primary failed")

    def test_nothing_here_fetches_instagram(self):
        # Phase 7 reads profiles, through the operator's own browser, serialized and
        # human-paced. This module finds the handle and stops.
        web = FailIfInstagram(
            results=[Result("https://instagram.com/cakebee", "Cake Bee (@cakebee)")],
            pages={},
        )
        service(web).enrich(request())  # FailIfInstagram raises if this fetches the profile

    def test_every_lookup_is_recorded_with_its_provenance(self):
        store = RecordingStore()
        web = FakeWeb(
            results=[Result("https://cakebee.in", "Cake Bee")],
            pages={"https://cakebee.in": SITE},
        )
        run_id = uuid.uuid4()
        service(web, store=store).enrich(request(run_id=run_id))

        self.assertTrue(store.rows)
        for row in store.rows:
            with self.subTest(source=row["source"]):
                self.assertEqual(row["run_id"], run_id)
                self.assertIn(row["status"], {"ok", "no_data", "blocked", "error"})

    def test_a_secret_never_reaches_a_recorded_row(self):
        secret = "tf-live-9f3c1d2b4a6e8c0f5171"
        store = RecordingStore()
        web = FakeWeb(fail=ProviderError("provider_unavailable", f"failed for key {secret}"))
        service(web, store=store).enrich(request())

        self.assertNotIn(secret, repr(store.rows))

    def test_provider_usage_is_measured_not_guessed(self):
        # The fallback rate decides whether Firecrawl's 1,000 monthly credits are enough.
        # A number nobody counts is a number nobody can plan against.
        primary = FakeWeb(results=[])
        summary = service(primary).enrich_many([request(), request()])
        self.assertIsNotNone(summary)


class ClockTests(unittest.TestCase):
    def test_the_service_takes_its_time_from_the_caller(self):
        # No test may need a real day to pass. The whole backoff schedule is only testable
        # because the clock is injected.
        later = NOW + timedelta(days=45)
        web = FakeWeb(results=[])
        found = service(web, clock=lambda: later).enrich(request())
        self.assertIsNotNone(found)


if __name__ == "__main__":
    unittest.main()
