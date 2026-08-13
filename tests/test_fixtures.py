"""Tests for the recorded-response maps provider.

The provider under test is the reason three later phases can be built for zero credits, so
what these tests protect is not really a class -- it is the claim that developing against
`tests/fixtures/searchapi/*.json` is the same as developing against SearchAPI. Three
properties carry that claim, and each has a test here:

  * a recorded body survives the round trip byte-for-byte, so what a credit bought is what a
    later phase reads (`RoundTripTests`);
  * one `search_places` is one page, which is the unit the whole cost model is denominated
    in -- 13 tiles x 5 variants x 3 pages is 195 searches against an allowance of 50, and
    that arithmetic is only checkable if the counter means what it says (`CallLedgerTests`);
  * `query_variant` resolves exactly as the live client resolves it, because a fixture run
    that quietly asked a different question than the billed run would answer for is worse
    than no fixture at all (`QueryVariantTests`).

The fourth property is the one this file cannot prove on its own: that the *parsing* agrees.
That lives in `tests/test_searchapi.py::ParityTests`, which feeds the same corpus through
both paths and compares the outcomes.

No test here touches the network, and the provider has no seam that could.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from lead_engine.geo.scope import ResolvedLocation
from lead_engine.niches import NICHE_PROFILES
from lead_engine.providers.errors import DEFAULT_MESSAGE, ProviderError
from lead_engine.providers.fixtures import (
    FixtureMapsProvider,
    FixtureNotFound,
    fixture_name,
    record_fixture,
)
from lead_engine.providers.searchapi import SearchApiClient, resolve_query

#: The committed corpus. Hand-authored to be what an Indiranagar sweep actually returns,
#: including the substitutions that make `exclude_types` necessary.
CORPUS = Path(__file__).parent / "fixtures" / "searchapi"

#: Every niche with a recorded page 1.
CORPUS_NICHES = ["cake_shop", "cloud_kitchen", "manufacturer", "photographer", "salon"]

#: Planted inside a deliberately broken fixture. Ordinary prose, so `errors.redact` will not
#: catch it for us -- if it reaches an error message, this suite is what notices.
SENTINEL_CONTENT = "the peacock ate our quota"

LOCATION = ResolvedLocation(
    label="Indiranagar, Bangalore, Karnataka, India",
    latitude=12.9784,
    longitude=77.6408,
    radius_meters=3000,
    precision="area",
    gl="in",
    source_query="Indiranagar, Bangalore, Karnataka, India",
)

PHOTOGRAPHER = NICHE_PROFILES["photographer"]


class FixtureTestCase(unittest.TestCase):
    """Shared plumbing: a scratch fixture directory that cleans itself up."""

    def scratch(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def corpus_provider(self, *, strict: bool = True) -> FixtureMapsProvider:
        return FixtureMapsProvider(CORPUS, strict=strict)


class RoundTripTests(FixtureTestCase):
    """What was recorded is what is served. The credit is spent once, ever."""

    PAYLOAD = {
        "search_metadata": {"id": "search_abc", "status": "Success"},
        "search_parameters": {"engine": "google_maps", "q": "cake shop", "hl": "en"},
        "local_results": [
            {
                "position": 1,
                "title": "Sweet Chariot",
                "type": "Cake shop",
                "types": ["Cake shop", "Bakery"],
                "price": "₹₹",
                "open_state": "Open ⋅ Closes 10 pm",
                "rating": 4.4,
                "reviews": 1284,
                "gps_coordinates": {"latitude": 12.9776, "longitude": 77.6404},
                "service_options": {"delivery": True, "dine_in": False},
                "description": None,
            }
        ],
    }

    def test_record_then_load_returns_an_identical_payload(self):
        directory = self.scratch()

        record_fixture(self.PAYLOAD, "cake_shop", directory)
        loaded = FixtureMapsProvider(directory).load("cake_shop")

        # Equality of the whole body, not of a parsed projection of it. A recorder that
        # dropped a field nothing reads yet would still pass a leads-only comparison, and
        # the next phase to want `service_options` would need a fresh credit to find out.
        self.assertEqual(loaded, self.PAYLOAD)

    def test_the_round_trip_survives_the_characters_indian_listings_are_full_of(self):
        # Rupee signs, the interpunct in `open_state`, en dashes in hours, Kannada shop
        # names. `record_fixture` writes UTF-8 with ensure_ascii=False and `load` reads
        # UTF-8; a platform default encoding creeping into either end breaks exactly here.
        payload = {
            "search_metadata": {"status": "Success"},
            "local_results": [
                {
                    "title": "ಶ್ರೀ ಕೃಷ್ಣ ಸ್ವೀಟ್ಸ್",
                    "price": "₹₹₹",
                    "open_state": "Open ⋅ Closes 9 pm",
                    "hours": "10 am–9 pm",
                    "type": "Café",
                }
            ],
        }
        directory = self.scratch()

        record_fixture(payload, "cafe", directory)

        self.assertEqual(FixtureMapsProvider(directory).load("cafe"), payload)

    def test_the_recorded_file_uses_the_documented_name(self):
        directory = self.scratch()

        path = record_fixture(self.PAYLOAD, "cake_shop", directory, page=2)

        self.assertEqual(path.name, "cake_shop-p2.json")
        self.assertEqual(path.name, fixture_name("cake_shop", 2))
        self.assertEqual(fixture_name("cake_shop"), "cake_shop-p1.json")

    def test_recording_creates_a_directory_that_does_not_exist_yet(self):
        directory = self.scratch() / "nested" / "searchapi"

        path = record_fixture(self.PAYLOAD, "cake_shop", directory)

        self.assertTrue(path.exists())

    def test_pages_are_separate_files_and_do_not_overwrite_each_other(self):
        directory = self.scratch()
        page_two = {"search_metadata": {"status": "Success"}, "local_results": []}

        record_fixture(self.PAYLOAD, "cake_shop", directory, page=1)
        record_fixture(page_two, "cake_shop", directory, page=2)

        provider = FixtureMapsProvider(directory)
        self.assertEqual(provider.load("cake_shop", 1), self.PAYLOAD)
        self.assertEqual(provider.load("cake_shop", 2), page_two)


class MissingFixtureTests(FixtureTestCase):
    """The two ways a missing page can be read, and why both exist."""

    def test_a_missing_fixture_is_loud_when_strict(self):
        provider = FixtureMapsProvider(self.scratch(), strict=True)

        with self.assertRaises(FixtureNotFound) as caught:
            provider.load("photographer")

        # The message has to be actionable: the operator needs to know which file, and that
        # recording one is the fix. Silence here would read as "Google knows of no
        # photographers in Indiranagar", which is a conclusion this provider cannot assert.
        self.assertIn("photographer-p1.json", str(caught.exception))
        self.assertIn("record_fixture", str(caught.exception))

    def test_a_missing_fixture_is_an_empty_page_when_not_strict(self):
        provider = FixtureMapsProvider(self.scratch(), strict=False)

        self.assertEqual(provider.load("photographer"), {"local_results": []})

    def test_paging_off_the_end_stops_rather_than_failing(self):
        # Non-strict exists for exactly this: a sweep asks for page 3, only two were ever
        # recorded, and the loop must terminate instead of raising. `has_more` False is what
        # makes it terminate.
        provider = self.corpus_provider(strict=False)

        outcome = provider.search_places(LOCATION, PHOTOGRAPHER, page=3)

        self.assertEqual(outcome.leads, ())
        self.assertEqual(outcome.returned, 0)
        self.assertFalse(outcome.has_more)

    def test_a_page_that_was_never_recorded_still_counts_as_a_call(self):
        # Deliberate, and worth pinning: the live client would spend a credit to discover
        # that page 3 is empty, so a fixture run that did not count it would understate the
        # cost of the sweep it is modelling.
        provider = self.corpus_provider(strict=False)

        provider.search_places(LOCATION, PHOTOGRAPHER, page=3)

        self.assertEqual(len(provider.calls), 1)

    def test_strict_is_the_default(self):
        self.assertTrue(FixtureMapsProvider(self.scratch()).strict)


class MalformedFixtureTests(FixtureTestCase):
    """A corrupted fixture is a provider error, and it takes nothing with it."""

    def broken(self) -> FixtureMapsProvider:
        directory = self.scratch()
        path = directory / fixture_name("photographer")
        # Truncated mid-write, which is what a killed recorder actually leaves behind.
        path.write_text(
            '{"local_results": [{"title": "' + SENTINEL_CONTENT + '", "type": "Photog',
            encoding="utf-8",
        )
        return FixtureMapsProvider(directory)

    def test_malformed_json_raises_a_provider_error(self):
        with self.assertRaises(ProviderError) as caught:
            self.broken().load("photographer")

        error = caught.exception
        self.assertEqual(error.code, "provider_bad_response")
        self.assertEqual(error.status, 502)
        self.assertEqual(error.provider, "searchapi")

    def test_the_error_message_does_not_contain_the_file_contents(self):
        # A fixture body is a recorded upstream response. Interpolating it into an error is
        # the same mistake as interpolating a live one, and a recorded body is the one place
        # in this system an API key could plausibly have been written to disk.
        with self.assertRaises(ProviderError) as caught:
            self.broken().load("photographer")

        message = caught.exception.message
        self.assertNotIn(SENTINEL_CONTENT, message)
        self.assertNotIn("peacock", message)
        # And the message is still OURS. `errors._safe_message` swaps in the generic default
        # for anything `redact` wants to touch, so a message that HAD leaked would pass the
        # two checks above by being silently replaced. This is the assertion that notices.
        self.assertNotEqual(message, DEFAULT_MESSAGE["provider_bad_response"])
        self.assertIn("photographer-p1.json", message)

    def test_a_malformed_fixture_fails_the_same_way_through_search_places(self):
        with self.assertRaises(ProviderError):
            self.broken().search_places(LOCATION, PHOTOGRAPHER)


class AvailableTests(FixtureTestCase):
    def test_available_lists_niche_ids_that_have_a_page_one(self):
        directory = self.scratch()
        for niche_id in ("salon", "photographer", "cake_shop"):
            record_fixture({"local_results": []}, niche_id, directory)

        self.assertEqual(
            FixtureMapsProvider(directory).available(),
            ["cake_shop", "photographer", "salon"],
        )

    def test_available_ignores_a_niche_with_only_a_later_page(self):
        # A page 2 with no page 1 is a half-recorded niche, not an available one. Listing it
        # would send a sweep to a fixture that does not exist.
        directory = self.scratch()
        record_fixture({"local_results": []}, "salon", directory, page=2)

        self.assertEqual(FixtureMapsProvider(directory).available(), [])

    def test_available_is_empty_for_a_directory_that_does_not_exist(self):
        provider = FixtureMapsProvider(self.scratch() / "never-recorded")

        self.assertEqual(provider.available(), [])

    def test_the_committed_corpus_covers_the_niches_phase_two_needs(self):
        self.assertEqual(self.corpus_provider().available(), CORPUS_NICHES)

    def test_a_niche_id_with_underscores_survives_the_filename_round_trip(self):
        # `available()` recovers the id by splitting on the LAST "-p". Every niche id in the
        # registry uses underscores, so this holds -- but a hyphenated id would not survive,
        # and `cloud_kitchen` is the one that proves the split is on the right separator.
        self.assertIn("cloud_kitchen", self.corpus_provider().available())


class CallLedgerTests(FixtureTestCase):
    """One call is one page. The whole cost model is denominated in this counter."""

    def test_a_new_provider_has_served_nothing(self):
        self.assertEqual(self.corpus_provider().calls, [])

    def test_one_call_is_recorded_per_page(self):
        provider = self.corpus_provider()

        provider.search_places(LOCATION, PHOTOGRAPHER, page=1)
        provider.search_places(LOCATION, PHOTOGRAPHER, page=2)
        provider.search_places(LOCATION, PHOTOGRAPHER, page=1)

        # Three pages, three calls -- including the repeat. The live client would be billed
        # for the repeat too; a ledger that deduplicated would model a discount that the
        # vendor does not give.
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual([call["page"] for call in provider.calls], [1, 2, 1])

    def test_a_call_records_the_niche_the_page_and_the_query_that_was_asked(self):
        provider = self.corpus_provider()

        provider.search_places(LOCATION, PHOTOGRAPHER, page=2, query_variant=2)

        self.assertEqual(
            provider.calls,
            [{"niche_id": "photographer", "page": 2, "query": "wedding photographer"}],
        )

    def test_limit_does_not_change_the_number_of_calls(self):
        # SearchAPI has no page-size parameter, so asking for five leads costs exactly what
        # asking for twenty costs. A limit that looked cheaper here would be a lie about the
        # budget.
        provider = self.corpus_provider()

        provider.search_places(LOCATION, PHOTOGRAPHER, limit=1)
        provider.search_places(LOCATION, PHOTOGRAPHER, limit=20)

        self.assertEqual(len(provider.calls), 2)


class QueryVariantTests(FixtureTestCase):
    """Resolution is the live client's, not a second implementation of it."""

    def test_none_resolves_to_the_first_query(self):
        provider = self.corpus_provider()

        provider.search_places(LOCATION, PHOTOGRAPHER)

        self.assertEqual(provider.calls[0]["query"], PHOTOGRAPHER.queries[0])
        self.assertEqual(provider.calls[0]["query"], "photographer")

    def test_an_int_indexes_the_query_list(self):
        provider = self.corpus_provider()

        provider.search_places(LOCATION, PHOTOGRAPHER, query_variant=1)

        self.assertEqual(provider.calls[0]["query"], "photography studio")

    def test_a_string_is_sent_through_verbatim_once_trimmed(self):
        provider = self.corpus_provider()

        outcome = provider.search_places(LOCATION, PHOTOGRAPHER, query_variant="  baby photos ")

        self.assertEqual(outcome.query, "baby photos")

    def test_true_is_a_type_error_rather_than_the_second_query(self):
        # `True` is an int in Python and would index to queries[1], quietly buying a
        # different search than the one asked for.
        with self.assertRaises(TypeError):
            self.corpus_provider().search_places(LOCATION, PHOTOGRAPHER, query_variant=True)

    def test_an_out_of_range_index_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.corpus_provider().search_places(LOCATION, PHOTOGRAPHER, query_variant=99)
        with self.assertRaises(ValueError):
            self.corpus_provider().search_places(LOCATION, PHOTOGRAPHER, query_variant=-1)

    def test_a_blank_string_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.corpus_provider().search_places(LOCATION, PHOTOGRAPHER, query_variant="   ")

    def test_a_rejected_variant_reads_no_fixture_and_records_no_call(self):
        # The fixture analogue of "nothing is spent when validation fails": resolution
        # happens before the fixture is opened, so a bad variant costs neither a file read
        # nor a line in the ledger.
        provider = self.corpus_provider()

        with self.assertRaises(ValueError):
            provider.search_places(LOCATION, PHOTOGRAPHER, query_variant=99)

        self.assertEqual(provider.calls, [])

    def test_resolution_agrees_with_the_live_client_across_the_whole_matrix(self):
        provider = self.corpus_provider()
        for variant in (None, 0, 1, 4, "custom query"):
            with self.subTest(variant=variant):
                outcome = provider.search_places(LOCATION, PHOTOGRAPHER, query_variant=variant)
                self.assertEqual(outcome.query, resolve_query(PHOTOGRAPHER, variant))


class ProviderContractTests(FixtureTestCase):
    """The fixture provider is a drop-in for the live client, or it is worthless."""

    def test_it_reports_itself_configured_for_the_api_layer_to_duck_type_on(self):
        self.assertEqual(FixtureMapsProvider(CORPUS).api_key, "fixture")

    def test_search_places_takes_the_same_arguments_as_the_live_client(self):
        # The whole point of this class is that Phase 2 can be written against it and then
        # run against SearchAPI unchanged. A drifted signature breaks that silently, at the
        # one moment when there are credits on the line.
        live = inspect.signature(SearchApiClient.search_places).parameters
        fixture = inspect.signature(FixtureMapsProvider.search_places).parameters

        self.assertEqual(list(live), list(fixture))
        self.assertEqual(
            [p.default for p in live.values()],
            [p.default for p in fixture.values()],
        )

    def test_search_raw_refuses_because_a_fixture_is_addressed_by_niche(self):
        # Fixtures are keyed by niche and page; a raw query string cannot find one. Refusing
        # is better than guessing, and it is the one method a recorder would call by habit.
        with self.assertRaises(FixtureNotFound):
            FixtureMapsProvider(CORPUS).search_raw(LOCATION, "photographer")


class CorpusShapeTests(FixtureTestCase):
    """The committed corpus is a SearchAPI response, not a convenient approximation."""

    def payloads(self):
        for path in sorted(CORPUS.glob("*.json")):
            yield path, json.loads(path.read_text(encoding="utf-8"))

    def test_every_fixture_is_the_documented_envelope(self):
        for path, payload in self.payloads():
            with self.subTest(fixture=path.name):
                self.assertIn("search_metadata", payload)
                self.assertIn("search_parameters", payload)
                self.assertIn("search_information", payload)
                # `local_results` is the key `searchapi._places` reads. Everything else in
                # the envelope is context; this one is the contract.
                self.assertIsInstance(payload["local_results"], list)
                for place in payload["local_results"]:
                    self.assertIsInstance(place, dict)

    def test_every_fixture_was_recorded_with_the_pinned_locale(self):
        # A fixture recorded under any other `hl` carries localised type labels, and every
        # slug in it would miss every include list in the registry. Recording one would burn
        # a credit to produce a corpus that teaches the wrong lesson.
        for path, payload in self.payloads():
            with self.subTest(fixture=path.name):
                self.assertEqual(payload["search_parameters"]["hl"], "en")
                self.assertEqual(payload["search_parameters"]["gl"], "in")

    def test_the_page_number_in_the_body_matches_the_filename(self):
        for path, payload in self.payloads():
            with self.subTest(fixture=path.name):
                page = int(path.stem.rsplit("-p", 1)[1])
                self.assertEqual(payload["search_parameters"]["page"], page)

    def test_no_fixture_carries_an_api_key(self):
        # SearchAPI echoes the request back in `search_metadata.request_url`, so a fixture
        # recorded from a live call is the most likely place in this repository for a key to
        # land -- and fixtures are committed. Nothing scrubs them on the way in, so this
        # test is the guard.
        for path, _ in self.payloads():
            with self.subTest(fixture=path.name):
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("api_key", text)
                self.assertNotIn("apikey", text)
                self.assertNotIn("authorization", text)

    def test_one_niche_has_a_second_page_so_paging_can_be_exercised(self):
        pages = sorted(path.stem for path in CORPUS.glob("*-p2.json"))

        self.assertEqual(pages, ["photographer-p2"])

    def test_a_full_page_is_twenty_results_because_that_is_what_paging_means(self):
        # `has_more` is inferred from a full page, so the corpus needs one page that is
        # genuinely full and one that is genuinely not.
        provider = self.corpus_provider()

        self.assertTrue(provider.search_places(LOCATION, PHOTOGRAPHER, page=1).has_more)
        self.assertFalse(provider.search_places(LOCATION, PHOTOGRAPHER, page=2).has_more)

    def test_the_corpus_carries_the_blocks_later_phases_will_want(self):
        # Recorded so that the first consumer of `popular_times` or `reviews_histogram` can
        # see the shape without buying a credit for it.
        blocks = {"popular_times": 0, "reviews_histogram": 0}
        for _, payload in self.payloads():
            for place in payload["local_results"]:
                for block in blocks:
                    if block in place:
                        blocks[block] += 1

        self.assertGreaterEqual(blocks["popular_times"], 1)
        self.assertGreaterEqual(blocks["reviews_histogram"], 1)

    def test_the_corpus_carries_the_gaps_that_make_a_lead_worth_pitching(self):
        # A business with no website is the product's entire target, and a place with no
        # title is what the UNNAMED sentinel exists for. Both must be in the corpus or the
        # code paths that handle them are only ever exercised by invented payloads.
        missing_website = missing_phone = missing_title = 0
        for _, payload in self.payloads():
            for place in payload["local_results"]:
                missing_website += "website" not in place
                missing_phone += "phone" not in place
                missing_title += "title" not in place

        self.assertGreaterEqual(missing_website, 1)
        self.assertGreaterEqual(missing_phone, 1)
        self.assertGreaterEqual(missing_title, 1)


if __name__ == "__main__":
    unittest.main()
