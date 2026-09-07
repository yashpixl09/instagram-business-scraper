"""Phase 7: the profile page, never fetched by this test suite.

WHAT THESE TESTS ARE DEFENDING

*The safety rule.* Nothing in `lead_engine/workers/instagram.py` may reach instagram.com,
under any flag, ever. `extract_profile` is pure -- it is handed text, never a URL -- and
every fake browser below returns fixture text loaded from disk. There is no live fetch
anywhere in this file.

*The four-state vocabulary.* A private account and an empty bio are facts about the
business (`ok`/`no_data`); a checkpoint page is `blocked`, a fact about the fetcher at that
moment; a transport failure or unparseable response is `error`, a fact about us. Getting
these three apart wrong is exactly what would make the negative-cache ladder misbehave --
see `enrichment/cache.py`'s docstring, and the assertions in `InstagramProfileLookupTests`
below that only `ok`/`no_data` ever touch the cache.

*Never a guess, never a crash.* Malformed, truncated, empty and private-profile input each
produce a distinct, correct `ProfileFinding` rather than an exception or a fabricated
number.

*Bio stays a bio.* A bio containing a phone number or an email address is stored verbatim
and is not turned into a contact -- that is `contacts.py`'s job, for a different source,
and doing a shrunken version of it here would drift the moment either module changed.

*The rate gate is structural.* `RateGatedInstagramBrowser` holds its lock for the whole
call, not just the timestamp check, so two callers cannot have a fetch in flight at once
regardless of how disciplined they are. The minimum-interval tests use a fake clock and a
fake sleep -- no test here waits out a real 30 seconds.
"""

from __future__ import annotations

import json
import threading
import unittest
import uuid
from datetime import UTC, datetime
from pathlib import Path

from lead_engine.enrichment.cache import INSTAGRAM, InMemoryCache
from lead_engine.workers.instagram import (
    MIN_FETCH_INTERVAL_SECONDS,
    SOURCE_INSTAGRAM_PROFILE,
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_NO_DATA,
    STATUS_OK,
    InstagramProfileLookup,
    RateGatedInstagramBrowser,
    extract_profile,
    unavailable,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "instagram"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


NOW = 1_000.0
CACHE_NOW = datetime(2026, 9, 6, 9, 0, tzinfo=UTC)


class FakeClock:
    """A clock and a `sleep` that agree with each other, and never touch a real clock.

    `sleep(seconds)` advances the same clock it is paired with, so a rate-gate test can
    assert "it waited N seconds" by reading `sleeps` without a single real second passing.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeBrowser:
    """Returns canned text, or raises. Counts every call; never touches a socket."""

    def __init__(self, text: str | None = None, *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    def fetch_profile(self, handle: str) -> str:
        self.calls.append(handle)
        if self.error is not None:
            raise self.error
        return self.text if self.text is not None else ""


class FakeStore:
    """Records every `insert_enrichment` call. Never touches a database."""

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
                "data": data,
                "source_url": source_url,
                "run_id": run_id,
            }
        )
        return self.rows[-1]


# --- extract_profile: the normal profile -----------------------------------------------------


class ExtractNormalProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finding = extract_profile(fixture("normal_profile.json"), handle="sweetcornerbakery")

    def test_status_is_ok(self):
        self.assertEqual(self.finding.status, STATUS_OK)

    def test_follower_count_is_read(self):
        self.assertEqual(self.finding.followers, 18420)

    def test_following_count_is_read(self):
        self.assertEqual(self.finding.following, 312)

    def test_post_count_is_read(self):
        self.assertEqual(self.finding.posts, 640)

    def test_bio_is_read(self):
        self.assertEqual(
            self.finding.bio, "Fresh bakes daily | DM to order | Koramangala, Bengaluru"
        )

    def test_external_link_is_read(self):
        self.assertEqual(self.finding.external_url, "https://sweetcornerbakery.in")

    def test_is_not_private(self):
        self.assertIs(self.finding.is_private, False)

    def test_category_is_read(self):
        self.assertEqual(self.finding.category, "Bakery")

    def test_handle_is_normalised(self):
        self.assertEqual(self.finding.handle, "sweetcornerbakery")

    def test_as_dict_carries_every_field(self):
        payload = self.finding.as_dict()
        for key in (
            "handle",
            "followers",
            "following",
            "posts",
            "bio",
            "external_url",
            "is_private",
            "category",
        ):
            self.assertIn(key, payload)

    def test_handle_with_at_and_case_is_normalised_the_same_way(self):
        finding = extract_profile(fixture("normal_profile.json"), handle="@SweetCornerBakery")
        self.assertEqual(finding.handle, "sweetcornerbakery")


# --- extract_profile: a private account -------------------------------------------------------


class ExtractPrivateAccountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finding = extract_profile(fixture("private_account.json"), handle="privatehandle")

    def test_status_is_ok_not_no_data(self):
        # A private account is a fact about the business, not an absence of one.
        self.assertEqual(self.finding.status, STATUS_OK)

    def test_is_private_is_true(self):
        self.assertIs(self.finding.is_private, True)

    def test_follower_and_following_counts_still_read(self):
        self.assertEqual(self.finding.followers, 245)
        self.assertEqual(self.finding.following, 180)

    def test_post_count_is_none_when_not_present_rather_than_zero(self):
        # The fixture omits edge_owner_to_timeline_media entirely, because private-account
        # post content is hidden. None, not a guessed 0.
        self.assertIsNone(self.finding.posts)

    def test_empty_bio_is_none(self):
        self.assertIsNone(self.finding.bio)

    def test_null_external_url_is_none(self):
        self.assertIsNone(self.finding.external_url)

    def test_null_category_is_none(self):
        self.assertIsNone(self.finding.category)


# --- extract_profile: deleted / nonexistent profile --------------------------------------------


class ExtractDeletedProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finding = extract_profile(fixture("deleted_profile.json"), handle="gonebusiness")

    def test_status_is_no_data(self):
        # The handle resolves to nothing. A fact about the business (wrong or dead handle),
        # not a failure of the fetch itself.
        self.assertEqual(self.finding.status, STATUS_NO_DATA)

    def test_reason_is_recorded(self):
        self.assertIsNotNone(self.finding.reason)

    def test_no_numbers_are_fabricated(self):
        self.assertIsNone(self.finding.followers)
        self.assertIsNone(self.finding.following)
        self.assertIsNone(self.finding.posts)
        self.assertIsNone(self.finding.bio)


# --- extract_profile: rate-limited / challenge page --------------------------------------------


class ExtractRateLimitedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finding = extract_profile(
            fixture("rate_limited_challenge.html"), handle="anybusiness"
        )

    def test_status_is_blocked(self):
        self.assertEqual(self.finding.status, STATUS_BLOCKED)

    def test_reason_is_recorded(self):
        self.assertIsNotNone(self.finding.reason)

    def test_no_data_is_fabricated(self):
        self.assertIsNone(self.finding.followers)


class ExtractLiveCapturedRateLimitTests(unittest.TestCase):
    """Not a synthetic fixture -- the actual HTML Instagram served during this session's
    one supervised, human-assisted live login test (2026-09-06), when a profile-info fetch
    followed shortly after login. Kept as independent proof that `_BLOCKED_MARKERS` matches
    real Instagram wording, not just the hand-written shape in `rate_limited_challenge.html`.
    No further live requests were made once this response was seen -- see this module's
    docstring on why repeated attempts against a real session are exactly the risk this
    whole design exists to avoid.
    """

    def setUp(self) -> None:
        self.finding = extract_profile(
            fixture("live_rate_limited_capture.html"), handle="test_ig_09"
        )

    def test_status_is_blocked(self):
        self.assertEqual(self.finding.status, STATUS_BLOCKED)

    def test_reason_is_recorded(self):
        self.assertEqual(self.finding.reason, "challenge or rate-limit page returned")

    def test_no_data_is_fabricated(self):
        self.assertIsNone(self.finding.followers)
        self.assertIsNone(self.finding.bio)
        self.assertIsNone(self.finding.is_private)
        self.assertIsNone(self.finding.bio)

    def test_a_json_body_that_reports_a_checkpoint_is_also_blocked(self):
        body = json.dumps({"status": "fail", "message": "checkpoint_required"})
        finding = extract_profile(body, handle="anybusiness")
        self.assertEqual(finding.status, STATUS_BLOCKED)


# --- extract_profile: malformed / truncated input -----------------------------------------------


class ExtractMalformedInputTests(unittest.TestCase):
    def test_truncated_json_is_error_not_a_crash(self):
        finding = extract_profile(fixture("malformed_truncated.json"), handle="brokenfeed")
        self.assertEqual(finding.status, STATUS_ERROR)
        self.assertIsNone(finding.followers)

    def test_empty_string_is_error(self):
        finding = extract_profile("", handle="somebusiness")
        self.assertEqual(finding.status, STATUS_ERROR)

    def test_none_is_error(self):
        finding = extract_profile(None, handle="somebusiness")
        self.assertEqual(finding.status, STATUS_ERROR)

    def test_whitespace_only_is_error(self):
        finding = extract_profile("   \n  ", handle="somebusiness")
        self.assertEqual(finding.status, STATUS_ERROR)

    def test_a_json_array_instead_of_an_object_is_error_not_a_crash(self):
        finding = extract_profile("[1, 2, 3]", handle="somebusiness")
        self.assertEqual(finding.status, STATUS_ERROR)

    def test_valid_json_with_no_recognisable_shape_is_error(self):
        finding = extract_profile(json.dumps({"unexpected": "shape"}), handle="somebusiness")
        self.assertEqual(finding.status, STATUS_ERROR)

    def test_an_invalid_handle_is_error_without_reading_the_payload(self):
        # "p" is a reserved Instagram path segment (a post), not a handle at all -- caught
        # by `social.normalise_handle` before the response is even parsed.
        finding = extract_profile(fixture("normal_profile.json"), handle="p")
        self.assertEqual(finding.status, STATUS_ERROR)


# --- extract_profile: empty bio, and a bio with a phone/email ------------------------------------


class ExtractBioEdgeCasesTests(unittest.TestCase):
    def test_empty_bio_field_becomes_none(self):
        finding = extract_profile(fixture("empty_bio.json"), handle="emptybio")
        self.assertEqual(finding.status, STATUS_OK)
        self.assertIsNone(finding.bio)

    def test_bio_with_phone_and_email_is_kept_verbatim(self):
        finding = extract_profile(fixture("bio_with_contact.json"), handle="contactbio")
        self.assertEqual(finding.status, STATUS_OK)
        self.assertEqual(
            finding.bio, "Call us at 98765 43210 or email hello@example.com to order"
        )

    def test_the_finding_has_no_phone_or_email_field_at_all(self):
        # Considered and rejected: extracting a contact out of the bio is `contacts.py`'s
        # job for a different source. This asserts the dataclass never grew one by accident.
        finding = extract_profile(fixture("bio_with_contact.json"), handle="contactbio")
        payload = finding.as_dict()
        self.assertNotIn("phone", payload)
        self.assertNotIn("email", payload)
        self.assertFalse(hasattr(finding, "phone"))
        self.assertFalse(hasattr(finding, "email"))


# --- unavailable() -----------------------------------------------------------------------------


class UnavailableHelperTests(unittest.TestCase):
    def test_defaults_to_error(self):
        finding = unavailable("somehandle", "something went wrong")
        self.assertEqual(finding.status, STATUS_ERROR)
        self.assertEqual(finding.reason, "something went wrong")

    def test_accepts_an_explicit_status(self):
        finding = unavailable("somehandle", "blocked for now", status=STATUS_BLOCKED)
        self.assertEqual(finding.status, STATUS_BLOCKED)


# --- the rate gate -------------------------------------------------------------------------------


class RateGatedBrowserTests(unittest.TestCase):
    def test_the_first_call_never_waits(self):
        clock = FakeClock(NOW)
        browser = FakeBrowser(text="{}")
        gated = RateGatedInstagramBrowser(
            browser, min_interval=30.0, clock=clock, sleep=clock.sleep
        )
        gated.fetch_profile("somebusiness")
        self.assertEqual(clock.sleeps, [])

    def test_a_second_call_immediately_after_waits_the_full_interval(self):
        clock = FakeClock(NOW)
        browser = FakeBrowser(text="{}")
        gated = RateGatedInstagramBrowser(
            browser, min_interval=30.0, clock=clock, sleep=clock.sleep
        )
        gated.fetch_profile("first")
        gated.fetch_profile("second")
        self.assertEqual(clock.sleeps, [30.0])

    def test_a_call_after_the_interval_has_already_elapsed_does_not_wait(self):
        clock = FakeClock(NOW)
        browser = FakeBrowser(text="{}")
        gated = RateGatedInstagramBrowser(
            browser, min_interval=30.0, clock=clock, sleep=clock.sleep
        )
        gated.fetch_profile("first")
        clock.now += 45.0  # more time than the interval requires has already passed
        gated.fetch_profile("second")
        self.assertEqual(clock.sleeps, [])

    def test_only_the_remaining_time_is_waited_not_the_whole_interval_again(self):
        clock = FakeClock(NOW)
        browser = FakeBrowser(text="{}")
        gated = RateGatedInstagramBrowser(
            browser, min_interval=30.0, clock=clock, sleep=clock.sleep
        )
        gated.fetch_profile("first")
        clock.now += 10.0  # 10 of the 30 required seconds have already passed
        gated.fetch_profile("second")
        self.assertEqual(clock.sleeps, [20.0])

    def test_the_wrapped_browser_is_still_called_with_the_handle(self):
        clock = FakeClock(NOW)
        browser = FakeBrowser(text="{}")
        gated = RateGatedInstagramBrowser(
            browser, min_interval=0.0, clock=clock, sleep=clock.sleep
        )
        gated.fetch_profile("somebusiness")
        self.assertEqual(browser.calls, ["somebusiness"])

    def test_the_default_interval_constant_is_used_when_none_is_given(self):
        gated = RateGatedInstagramBrowser(FakeBrowser(text="{}"))
        self.assertEqual(gated._min_interval, MIN_FETCH_INTERVAL_SECONDS)

    def test_only_one_fetch_is_ever_in_flight_at_once(self):
        # Structural, not disciplinary: the lock is held for the whole wrapped call, so two
        # threads calling through the SAME gate can never have a fetch in flight together.
        lock = threading.Lock()
        state = {"active": 0, "max_active": 0}

        class SlowBrowser:
            def fetch_profile(self, handle: str) -> str:
                with lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                try:
                    # A short, real pause -- just long enough to give a racing thread a
                    # chance to observe an overlap, if the lock in front of this were
                    # missing. Not a stand-in for the rate-gate's timing, which is tested
                    # entirely with the fake clock above.
                    threading.Event().wait(timeout=0.05)
                finally:
                    with lock:
                        state["active"] -= 1
                return "{}"

        gated = RateGatedInstagramBrowser(
            SlowBrowser(), min_interval=0.0, clock=lambda: 0.0, sleep=lambda s: None
        )
        threads = [
            threading.Thread(target=gated.fetch_profile, args=(f"handle-{i}",))
            for i in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(state["max_active"], 1)


# --- InstagramProfileLookup: fetch, extract, record -------------------------------------------


class InstagramProfileLookupTests(unittest.TestCase):
    def test_a_normal_profile_is_written_as_an_ok_row(self):
        browser = FakeBrowser(text=fixture("normal_profile.json"))
        store = FakeStore()
        lookup = InstagramProfileLookup(browser=browser, store=store)
        business_id = uuid.uuid4()

        finding = lookup.run(business_id=business_id, handle="sweetcornerbakery")

        self.assertEqual(finding.status, STATUS_OK)
        self.assertEqual(len(store.rows), 1)
        row = store.rows[0]
        self.assertEqual(row["business_id"], business_id)
        self.assertEqual(row["source"], SOURCE_INSTAGRAM_PROFILE)
        self.assertEqual(row["status"], STATUS_OK)
        self.assertEqual(row["data"]["followers"], 18420)
        self.assertEqual(row["source_url"], "https://www.instagram.com/sweetcornerbakery/")

    def test_a_browser_exception_is_recorded_as_error_and_never_raises(self):
        browser = FakeBrowser(error=RuntimeError("connection reset while loading a page"))
        store = FakeStore()
        lookup = InstagramProfileLookup(browser=browser, store=store)

        finding = lookup.run(business_id=uuid.uuid4(), handle="somebusiness")

        self.assertEqual(finding.status, STATUS_ERROR)
        self.assertEqual(len(store.rows), 1)
        self.assertEqual(store.rows[0]["status"], STATUS_ERROR)

    def test_the_raw_exception_message_never_reaches_the_stored_row(self):
        # The exception text could carry the operator's own session URL; only the
        # exception's class name is allowed to survive into a stored row.
        browser = FakeBrowser(
            error=RuntimeError("https://instagram.com/secret-session-fragment")
        )
        store = FakeStore()
        lookup = InstagramProfileLookup(browser=browser, store=store)

        finding = lookup.run(business_id=uuid.uuid4(), handle="somebusiness")

        self.assertNotIn("secret-session-fragment", finding.reason or "")
        self.assertNotIn("secret-session-fragment", json.dumps(store.rows[0]["data"]))
        self.assertIn("RuntimeError", finding.reason or "")

    def test_no_data_status_records_a_cache_miss(self):
        browser = FakeBrowser(text=fixture("deleted_profile.json"))
        cache = InMemoryCache()
        lookup = InstagramProfileLookup(browser=browser, cache=cache, clock=lambda: CACHE_NOW)
        business_id = uuid.uuid4()

        finding = lookup.run(business_id=business_id, handle="gonebusiness")

        self.assertEqual(finding.status, STATUS_NO_DATA)
        entry = cache.entries[(business_id, INSTAGRAM)]
        self.assertEqual(entry.misses, 1)

    def test_ok_status_records_a_cache_success(self):
        cache = InMemoryCache()
        business_id = uuid.uuid4()
        # Seed a prior miss, then confirm a later ok clears the ladder -- exactly the
        # "successes are not negative cache entries" rule `cache.py` documents.
        cache.record_miss(business_id, INSTAGRAM, now=CACHE_NOW)

        browser = FakeBrowser(text=fixture("normal_profile.json"))
        lookup = InstagramProfileLookup(
            browser=browser, cache=cache, clock=lambda: CACHE_NOW
        )
        lookup.run(business_id=business_id, handle="sweetcornerbakery")

        entry = cache.entries[(business_id, INSTAGRAM)]
        self.assertEqual(entry.misses, 0)
        self.assertIsNone(entry.retry_after)

    def test_blocked_status_never_touches_the_cache(self):
        browser = FakeBrowser(text=fixture("rate_limited_challenge.html"))
        cache = InMemoryCache()
        business_id = uuid.uuid4()
        lookup = InstagramProfileLookup(browser=browser, cache=cache, clock=lambda: NOW)

        finding = lookup.run(business_id=business_id, handle="somebusiness")

        self.assertEqual(finding.status, STATUS_BLOCKED)
        self.assertNotIn((business_id, INSTAGRAM), cache.entries)

    def test_error_status_never_touches_the_cache(self):
        browser = FakeBrowser(error=RuntimeError("boom"))
        cache = InMemoryCache()
        business_id = uuid.uuid4()
        lookup = InstagramProfileLookup(browser=browser, cache=cache, clock=lambda: NOW)

        finding = lookup.run(business_id=business_id, handle="somebusiness")

        self.assertEqual(finding.status, STATUS_ERROR)
        self.assertNotIn((business_id, INSTAGRAM), cache.entries)

    def test_works_with_no_store_and_no_cache_at_all(self):
        browser = FakeBrowser(text=fixture("normal_profile.json"))
        lookup = InstagramProfileLookup(browser=browser)
        finding = lookup.run(business_id=uuid.uuid4(), handle="sweetcornerbakery")
        self.assertEqual(finding.status, STATUS_OK)


if __name__ == "__main__":
    unittest.main()
