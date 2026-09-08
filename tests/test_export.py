"""The sheet the operator actually works from.

Split the way the module is split, and for the same reason. `view.py` decides which
columns exist, in which order, and which row comes first; every one of those tests is
synthetic records in, tuples out, and runs without Postgres, without openpyxl and without
a temp directory. `excel.py` decides what lands in the file, and those tests write a real
workbook and read it back -- asserting on the bytes that survived rather than on the
values handed in, because a writer that drops a comment or a fill is indistinguishable
from a correct one if you only ever inspect your own inputs.

The integration section at the bottom is the only part that needs a database, and it is
skipped unless `LEAD_ENGINE_TEST_DSN` is set. It exists because two claims in `excel.py`
cannot be checked any other way: that `EXPORT_QUERY` still names the columns
`record_from_row` reads, and that `banding_method` flips from `absolute` to `relative`
when a cohort reaches thirty. Everything else stays infrastructure-free.

Why these assertions and not others
-----------------------------------
The spreadsheet is the product. The operator sorts it, reads a number off it, and says
that number out loud to a business owner. So the tests concentrate on the failures that
are silent in a spreadsheet and expensive in a conversation:

* a dropped or reordered column, which shifts every value after it one cell left and looks
  entirely normal;
* a sort by score alone, which is the obvious sort and sends the operator across a 40km
  city in ranking order;
* a `0` where nothing is known, which is a claim about the business rather than an
  admission of ignorance;
* a half-written workbook replacing a good one, which costs a week of marked-up visits.
"""

from __future__ import annotations

import os
import re
import unittest
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from openpyxl import Workbook, load_workbook

from lead_engine.export import excel, view
from lead_engine.export.excel import (
    COMMENT_AUTHOR,
    EXPORT_QUERY,
    NUMBER_FORMATS,
    OPERATOR_FILL_COLOR,
    OPERATOR_HEADER_FILL_COLOR,
    SHEET_TITLE,
    build_workbook,
    export_records,
    fetch_records,
    record_from_row,
    write_workbook,
)
from lead_engine.export.view import (
    COLUMN_GROUPS,
    COLUMN_INDEX,
    COLUMNS,
    EVIDENCE_COLUMNS,
    GOOGLE_EVIDENCE_COLUMNS,
    INSTAGRAM_EVIDENCE_COLUMNS,
    OPERATOR_COLUMNS,
    Cell,
    ContactPerson,
    ExportRow,
    LeadRecord,
    OperatorVerdict,
    build_row,
    build_rows,
    evidence_sources,
    instagram_profile_url,
    niche_label,
    record_from_scored_lead,
)
from lead_engine.models import Evidence
from lead_engine.niches import NICHE_PROFILES
from lead_engine.scoring import ScoredLead, score_lead
from tests.factories import qualifying_lead

# The layout, written out by hand. Deriving this from `COLUMNS` would assert that the
# module equals itself; the point is that a column silently dropped, renamed or reordered
# fails here in the same commit rather than months later as a blank field the operator has
# already trusted, or as every value in a row shifted one cell to the left.
EXPECTED_COLUMNS = [
    # identity
    "business_name",
    "niche",
    "business_type",
    # geography
    "country",
    "state",
    "city",
    "search_area",
    "address",
    "lat",
    "lng",
    # contact
    "phone",
    "email",
    "website",
    "instagram_handle",
    "facebook_url",
    # people
    "contact_name",
    "contact_role",
    "contact_phone",
    "contact_email",
    "contact_source",
    # evidence
    "reviews",
    "rating",
    "followers",
    "engagement_rate",
    "runs_ads",
    "peak_hours",
    # agent
    "total_score",
    "audience_band",
    "banding_method",
    "signals",
    "ai_summary",
    "website_pitch",
    # automation
    "automation_opportunities",
    "automation_pitch",
    # operator
    "my_verdict",
    "notes",
    "contacted_on",
    "channel",
    "outcome",
]


def record(name: str, **fields) -> LeadRecord:
    """A `LeadRecord` with only what the test cares about set."""
    return LeadRecord(business_name=name, **fields)


def ordered_names(records) -> list[str]:
    return [row.value("business_name") for row in build_rows(records)]


def column_letter(name: str) -> str:
    from openpyxl.utils import get_column_letter

    return get_column_letter(COLUMN_INDEX[name] + 1)


def cell_at(sheet, row: int, column: str):
    return sheet.cell(row=row, column=COLUMN_INDEX[column] + 1)


def written_sheet(records, directory: Path, name: str = "leads.xlsx"):
    """Export, reopen, and hand back the sheet as it exists on disk."""
    path = export_records(records, directory / name)
    return load_workbook(path)[SHEET_TITLE], path


# --- view: the layout ------------------------------------------------------------------


class ColumnLayoutTests(unittest.TestCase):
    def test_columns_are_exactly_this_list_in_this_order(self):
        self.assertEqual(list(COLUMNS), EXPECTED_COLUMNS)

    def test_no_column_is_declared_twice(self):
        self.assertEqual(len(set(COLUMNS)), len(COLUMNS))

    def test_groups_flatten_to_the_column_list(self):
        flattened = [name for _, names in COLUMN_GROUPS for name in names]
        self.assertEqual(flattened, list(COLUMNS))

    def test_column_index_points_at_the_right_position(self):
        for position, name in enumerate(COLUMNS):
            with self.subTest(column=name):
                self.assertEqual(COLUMN_INDEX[name], position)

    def test_the_operator_owns_the_last_five_columns(self):
        # Last because they are written rather than read, and a later phase reads them back
        # by name. Moving them is a compatibility break, not a cosmetic change.
        self.assertEqual(
            list(OPERATOR_COLUMNS),
            ["my_verdict", "notes", "contacted_on", "channel", "outcome"],
        )
        self.assertEqual(list(COLUMNS[-5:]), list(OPERATOR_COLUMNS))

    def test_the_evidence_group_is_the_evidence_columns(self):
        self.assertEqual(
            list(EVIDENCE_COLUMNS),
            ["reviews", "rating", "followers", "engagement_rate", "runs_ads", "peak_hours"],
        )

    def test_every_evidence_column_has_at_most_one_source_page(self):
        google = set(GOOGLE_EVIDENCE_COLUMNS)
        instagram = set(INSTAGRAM_EVIDENCE_COLUMNS)
        self.assertEqual(google & instagram, set())
        self.assertLessEqual(google | instagram, set(EVIDENCE_COLUMNS))

    def test_runs_ads_has_no_verifiable_source(self):
        # Deliberate: there is no page that shows it, and a comment pointing somewhere the
        # operator cannot find the claim is worse than no comment at all.
        self.assertNotIn("runs_ads", GOOGLE_EVIDENCE_COLUMNS)
        self.assertNotIn("runs_ads", INSTAGRAM_EVIDENCE_COLUMNS)


class SortOrderTests(unittest.TestCase):
    """Area first, then score. The operator walks to these businesses."""

    def test_area_is_grouped_before_score_is_ranked(self):
        records = [
            record("Whitefield Best", search_area="Whitefield", total_score=95),
            record("Indiranagar Mid", search_area="Indiranagar", total_score=60),
            record("Whitefield Worst", search_area="Whitefield", total_score=20),
            record("Indiranagar Best", search_area="Indiranagar", total_score=80),
        ]

        self.assertEqual(
            ordered_names(records),
            ["Indiranagar Best", "Indiranagar Mid", "Whitefield Best", "Whitefield Worst"],
        )

    def test_score_only_sorting_would_give_a_different_answer(self):
        # Guarding the guard. The case above is only evidence of anything if the obvious
        # wrong sort disagrees with it; otherwise it would pass against a scoreboard.
        records = [
            record("Whitefield Best", search_area="Whitefield", total_score=95),
            record("Indiranagar Mid", search_area="Indiranagar", total_score=60),
            record("Whitefield Worst", search_area="Whitefield", total_score=20),
            record("Indiranagar Best", search_area="Indiranagar", total_score=80),
        ]
        by_score_alone = [
            item.business_name
            for item in sorted(records, key=lambda item: -(item.total_score or 0))
        ]

        self.assertEqual(
            by_score_alone,
            ["Whitefield Best", "Indiranagar Best", "Indiranagar Mid", "Whitefield Worst"],
        )
        self.assertNotEqual(by_score_alone, ordered_names(records))

    def test_each_area_is_one_contiguous_block(self):
        # The whole justification for the sort: a day's route, not a scattered ranking.
        records = [
            record(f"{area}-{score}", search_area=area, total_score=score)
            for area, score in [
                ("Koramangala", 30),
                ("Indiranagar", 90),
                ("Koramangala", 88),
                ("Jayanagar", 45),
                ("Indiranagar", 10),
                ("Jayanagar", 99),
            ]
        ]

        areas = [name.rsplit("-", 1)[0] for name in ordered_names(records)]

        self.assertEqual(
            areas,
            ["Indiranagar", "Indiranagar", "Jayanagar", "Jayanagar",
             "Koramangala", "Koramangala"],
        )
        # Contiguity, stated independently of the alphabetical expectation above: no area
        # is ever visited, left, and visited again.
        blocks = [
            area for index, area in enumerate(areas) if index == 0 or areas[index - 1] != area
        ]
        self.assertEqual(len(blocks), len(set(blocks)))

    def test_within_an_area_the_ranking_is_by_score_descending(self):
        records = [
            record(f"s{score}", search_area="Indiranagar", total_score=score)
            for score in (10, 90, 55, 72)
        ]

        self.assertEqual(ordered_names(records), ["s90", "s72", "s55", "s10"])

    def test_a_lead_with_no_area_sorts_last_however_good_it_is(self):
        records = [
            record("Unplaceable", search_area=None, total_score=100),
            record("Blank Area", search_area="   ", total_score=99),
            record("Walkable", search_area="Indiranagar", total_score=1),
        ]

        ordered = ordered_names(records)

        self.assertEqual(ordered[0], "Walkable")
        self.assertEqual(set(ordered[1:]), {"Unplaceable", "Blank Area"})

    def test_an_unscored_lead_sorts_below_a_zero_score_in_its_own_area(self):
        # "Not scored" is missing information, not a bad result. It must not displace a
        # real ranking, and it must not be promoted above one either.
        records = [
            record("Unscored", search_area="Indiranagar", total_score=None),
            record("Zero", search_area="Indiranagar", total_score=0),
            record("Ten", search_area="Indiranagar", total_score=10),
        ]

        self.assertEqual(ordered_names(records), ["Ten", "Zero", "Unscored"])

    def test_one_area_spelled_two_ways_stays_one_block(self):
        records = [
            record("lower", search_area="indiranagar", total_score=50),
            record("Upper", search_area="Indiranagar", total_score=90),
            record("Elsewhere", search_area="Hebbal", total_score=70),
        ]

        self.assertEqual(ordered_names(records), ["Elsewhere", "Upper", "lower"])

    def test_surrounding_whitespace_does_not_split_an_area(self):
        records = [
            record("padded", search_area="  Indiranagar  ", total_score=50),
            record("clean", search_area="Indiranagar", total_score=90),
            record("other", search_area="Whitefield", total_score=99),
        ]

        self.assertEqual(ordered_names(records), ["clean", "padded", "other"])

    def test_ties_break_on_name_so_two_exports_of_the_same_data_match(self):
        records = [
            record("Zeta Salon", search_area="Indiranagar", total_score=70),
            record("alpha salon", search_area="Indiranagar", total_score=70),
            record("Mid Salon", search_area="Indiranagar", total_score=70),
        ]

        self.assertEqual(ordered_names(records), ["alpha salon", "Mid Salon", "Zeta Salon"])
        self.assertEqual(ordered_names(records), ordered_names(list(reversed(records))))

    def test_sorting_does_not_mutate_the_caller_s_list(self):
        records = [
            record("B", search_area="Whitefield", total_score=10),
            record("A", search_area="Indiranagar", total_score=10),
        ]
        original = list(records)

        build_rows(records)

        self.assertEqual(records, original)

    def test_an_empty_corpus_produces_no_rows(self):
        self.assertEqual(build_rows([]), [])


class RowWidthTests(unittest.TestCase):
    """Ragged rows are what break a spreadsheet silently."""

    def test_a_lead_with_nothing_but_a_name_is_still_full_width(self):
        row = build_row(record("Bare Minimum"))

        self.assertEqual(len(row.cells), len(COLUMNS))
        self.assertEqual(len(row.values), len(EXPECTED_COLUMNS))
        self.assertEqual(row.value("business_name"), "Bare Minimum")

    def test_every_shape_of_record_produces_the_same_width(self):
        cases = {
            "empty": record("Empty"),
            "no contact": record("No Contact", evidence=Evidence(reviews=5)),
            "no evidence": record("No Evidence", contact=ContactPerson(name="Asha")),
            "no verdict": record("No Verdict", total_score=40),
            "full": record(
                "Full",
                niche="Salon",
                city="Bangalore",
                search_area="Indiranagar",
                contact=ContactPerson(name="Asha", role="owner"),
                evidence=Evidence(reviews=340, rating=4.6),
                verdict=OperatorVerdict(my_verdict="visit"),
                signals=("no website",),
                automation_opportunities=("booking",),
            ),
        }
        for label, item in cases.items():
            with self.subTest(record=label):
                self.assertEqual(len(build_row(item).cells), len(COLUMNS))

    def test_every_column_is_addressable_on_a_bare_row(self):
        row = build_row(record("Bare Minimum"))
        for name in COLUMNS:
            with self.subTest(column=name):
                # Raises rather than returning nothing if the row were short.
                row.cell(name)

    def test_a_value_keyed_to_a_column_that_does_not_exist_raises(self):
        # The other half of the width guarantee: a column renamed in one place and not the
        # other must fail here, not print a sheet with a blank column and a lost value.
        stale = {name: None for name in COLUMNS}
        stale["phone_number"] = "9123456789"

        with patch.object(view, "_row_values", return_value=stale):
            with self.assertRaises(KeyError) as caught:
                build_row(record("Whoever"))

        self.assertIn("phone_number", str(caught.exception))


class BlankNotZeroTests(unittest.TestCase):
    """An empty cell is an admission of ignorance. A 0 is a claim about the business."""

    def test_unknown_evidence_is_empty_not_zero(self):
        row = build_row(record("Unknown Everything"))

        for name in EVIDENCE_COLUMNS:
            with self.subTest(column=name):
                value = row.value(name)
                self.assertIsNone(value)
                self.assertNotEqual(value, 0)

    def test_an_unscored_lead_has_an_empty_score_not_a_zero(self):
        self.assertIsNone(build_row(record("Unscored")).value("total_score"))

    def test_a_genuine_zero_survives(self):
        # The mirror image, and the reason the test above cannot be satisfied by blanking
        # everything falsy: zero reviews is a fact, and the operator should see it.
        row = build_row(
            record(
                "Brand New Cafe",
                total_score=0,
                lat=0.0,
                evidence=Evidence(reviews=0, rating=0.0, followers=0, engagement_rate=0.0),
            )
        )

        self.assertEqual(row.value("reviews"), 0)
        self.assertEqual(row.value("rating"), 0.0)
        self.assertEqual(row.value("followers"), 0)
        self.assertEqual(row.value("engagement_rate"), 0.0)
        self.assertEqual(row.value("total_score"), 0)
        self.assertEqual(row.value("lat"), 0.0)

    def test_no_cell_ever_reads_as_the_word_none(self):
        row = build_row(record("Unknown Everything"))

        for name, value in zip(COLUMNS, row.values, strict=True):
            with self.subTest(column=name):
                self.assertNotEqual(value, "None")
                self.assertNotEqual(value, "null")

    def test_a_missing_value_is_none_and_never_the_empty_string(self):
        # `""` is a non-empty cell: it defeats COUNTBLANK, "go to special > blanks", and
        # every filter the operator would use to find the rows still needing work.
        row = build_row(record("Unknown Everything"))

        for name, value in zip(COLUMNS, row.values, strict=True):
            if name == "business_name":
                continue
            with self.subTest(column=name):
                self.assertIsNone(value)

    def test_whitespace_only_text_becomes_a_blank_cell(self):
        row = build_row(
            record(
                "Whitespace",
                city="   ",
                phone="\t",
                address=" \n ",
                verdict=OperatorVerdict(notes="  "),
            )
        )

        for name in ("city", "phone", "address", "notes"):
            with self.subTest(column=name):
                self.assertIsNone(row.value(name))

    def test_text_is_trimmed_rather_than_blanked_when_there_is_something_there(self):
        row = build_row(record("  Blush Salon  ", city="  Bangalore "))

        self.assertEqual(row.value("business_name"), "Blush Salon")
        self.assertEqual(row.value("city"), "Bangalore")

    def test_an_empty_signal_list_is_a_blank_cell(self):
        row = build_row(record("Quiet", signals=(), automation_opportunities=()))

        self.assertIsNone(row.value("signals"))
        self.assertIsNone(row.value("automation_opportunities"))

    def test_runs_ads_says_yes_no_or_nothing(self):
        # Blank is load-bearing: "never checked" and "does not run ads" are different
        # facts, and FALSE in a cell reads as the second one.
        cases = [(True, "yes"), (False, "no"), (None, None)]
        for flag, expected in cases:
            with self.subTest(runs_ads=flag):
                row = build_row(record("Advertiser", evidence=Evidence(runs_ads=flag)))
                self.assertEqual(row.value("runs_ads"), expected)


class ValueFormattingTests(unittest.TestCase):
    def test_signals_are_joined_with_semicolons(self):
        row = build_row(record("Joined", signals=("no website", "public phone, verified")))

        self.assertEqual(row.value("signals"), "no website; public phone, verified")

    def test_blank_entries_inside_a_signal_list_are_dropped(self):
        row = build_row(record("Joined", signals=("no website", "  ", "public phone")))

        self.assertEqual(row.value("signals"), "no website; public phone")

    def test_engagement_rate_stays_the_fraction_it_is_everywhere_else(self):
        row = build_row(record("Engaged", evidence=Evidence(engagement_rate=0.042)))

        self.assertAlmostEqual(row.value("engagement_rate"), 0.042, places=12)

    def test_niche_label_uses_the_registry_label(self):
        niche_id, profile = next(iter(NICHE_PROFILES.items()))

        self.assertEqual(niche_label(niche_id), profile.label)

    def test_niche_label_falls_back_to_a_retired_id_rather_than_raising(self):
        # A business stored under a niche that has since been renamed still has to appear.
        # An export that aborts on one stale row is worse than one that prints the id.
        self.assertEqual(niche_label("beauty_salon_v1"), "beauty_salon_v1")
        self.assertIsNone(niche_label(None))
        self.assertIsNone(niche_label(""))

    def test_a_contacted_on_date_is_kept_as_a_date(self):
        row = build_row(
            record("Visited", verdict=OperatorVerdict(contacted_on=date(2026, 8, 13)))
        )

        self.assertEqual(row.value("contacted_on"), date(2026, 8, 13))


class InstagramUrlTests(unittest.TestCase):
    def test_a_handle_becomes_a_profile_url(self):
        self.assertEqual(
            instagram_profile_url("@blush.salon"),
            "https://www.instagram.com/blush.salon/",
        )
        self.assertEqual(
            instagram_profile_url("  blush.salon  "),
            "https://www.instagram.com/blush.salon/",
        )

    def test_every_spelling_of_a_profile_link_lands_on_one_canonical_url(self):
        # Normalised, not returned verbatim. These are receipts on a cell the operator will
        # click while deciding whether to trust a number, so they should all reach the same
        # page regardless of how the handle happened to be stored.
        for stored in (
            "@blush.salon",
            "blush.salon",
            "www.instagram.com/blush.salon",
            "instagram.com/blush.salon",
            "https://instagram.com/blush.salon/",
            "https://www.instagram.com/blush.salon",
            "http://instagram.com/blush.salon",
        ):
            with self.subTest(stored=stored):
                self.assertEqual(
                    instagram_profile_url(stored),
                    "https://www.instagram.com/blush.salon/",
                )

    def test_a_scheme_less_profile_link_is_not_double_prefixed(self):
        # The regression this class exists for. The host check once missed the `www.` form,
        # so the link became https://www.instagram.com/www.instagram.com/blush.salon/ -- a
        # plausible URL that 404s, attached to the followers cell as its source. view.py's
        # own docstring calls that worse than no comment: the operator clicks, cannot find
        # the number, and stops trusting the other receipts too.
        self.assertEqual(
            instagram_profile_url("www.instagram.com/blush.salon"),
            "https://www.instagram.com/blush.salon/",
        )

    def test_another_sites_url_is_left_alone(self):
        # A Linktree in the handle column is not an Instagram profile, and rewriting it into
        # one would fabricate a source for a number it never provided.
        self.assertEqual(
            instagram_profile_url("https://linktr.ee/blush"), "https://linktr.ee/blush"
        )

    def test_nothing_in_means_nothing_out(self):
        for empty in (None, "", "   ", "@", "@/"):
            with self.subTest(handle=empty):
                self.assertIsNone(instagram_profile_url(empty))

    def test_a_bare_host_is_not_a_profile(self):
        # instagram.com with no path links to nothing in particular, so it is no evidence
        # for the cell it would be attached to.
        for bare in ("instagram.com", "www.instagram.com", "https://www.instagram.com/"):
            with self.subTest(handle=bare):
                self.assertIsNone(instagram_profile_url(bare))


class EvidenceSourceTests(unittest.TestCase):
    def test_each_page_provenances_its_own_columns(self):
        sources = evidence_sources(
            google_url="https://maps.example/blush",
            instagram_url="https://www.instagram.com/blush/",
        )

        for name in GOOGLE_EVIDENCE_COLUMNS:
            with self.subTest(column=name):
                self.assertEqual(sources[name], "https://maps.example/blush")
        for name in INSTAGRAM_EVIDENCE_COLUMNS:
            with self.subTest(column=name):
                self.assertEqual(sources[name], "https://www.instagram.com/blush/")

    def test_runs_ads_is_given_no_source(self):
        sources = evidence_sources(
            google_url="https://maps.example/blush",
            instagram_url="https://www.instagram.com/blush/",
        )

        self.assertNotIn("runs_ads", sources)

    def test_a_caller_with_a_precise_url_overrides_the_grouping(self):
        sources = evidence_sources(
            google_url="https://maps.example/blush",
            extra={"reviews": "https://maps.example/blush/reviews"},
        )

        self.assertEqual(sources["reviews"], "https://maps.example/blush/reviews")
        self.assertEqual(sources["rating"], "https://maps.example/blush")

    def test_a_blank_override_does_not_erase_the_grouping(self):
        sources = evidence_sources(
            google_url="https://maps.example/blush", extra={"reviews": ""}
        )

        self.assertEqual(sources["reviews"], "https://maps.example/blush")

    def test_no_pages_means_no_sources(self):
        self.assertEqual(evidence_sources(), {})

    def test_a_row_without_sources_carries_no_provenance(self):
        row = build_row(record("Sourceless", evidence=Evidence(reviews=12)))

        for name in COLUMNS:
            with self.subTest(column=name):
                self.assertIsNone(row.source(name))

    def test_provenance_lands_on_the_evidence_cells_only(self):
        row = build_row(
            record(
                "Sourced",
                evidence=Evidence(reviews=340),
                evidence_sources=evidence_sources(
                    google_url="https://maps.example/x",
                    instagram_url="https://www.instagram.com/x/",
                ),
            )
        )

        sourced = {name for name in COLUMNS if row.source(name)}

        self.assertEqual(sourced, set(GOOGLE_EVIDENCE_COLUMNS) | set(INSTAGRAM_EVIDENCE_COLUMNS))


class ScoredLeadRecordTests(unittest.TestCase):
    def setUp(self):
        self.profile = NICHE_PROFILES["bakery"]
        self.lead = qualifying_lead(self.profile)
        self.lead.matched_niches.append("bakery")
        self.scored = ScoredLead(
            lead=self.lead,
            score=score_lead(self.lead, [self.profile]),
            ai_summary="A bakery with no website.",
            outreach_message="A one-page site with your catalogue.",
        )

    def test_the_lead_s_own_fields_reach_the_row(self):
        row = build_row(record_from_scored_lead(self.scored, search_area="Indiranagar"))

        self.assertEqual(row.value("business_name"), self.lead.name)
        self.assertEqual(row.value("business_type"), self.lead.category)
        self.assertEqual(row.value("city"), self.lead.city)
        self.assertEqual(row.value("search_area"), "Indiranagar")
        self.assertEqual(row.value("phone"), self.lead.phone)
        self.assertEqual(row.value("total_score"), self.scored.score.total)

    def test_the_niche_is_labelled_from_the_lead_s_first_match(self):
        row = build_row(record_from_scored_lead(self.scored))

        self.assertEqual(row.value("niche"), self.profile.label)

    def test_the_outreach_message_becomes_the_website_pitch(self):
        row = build_row(record_from_scored_lead(self.scored))

        self.assertEqual(row.value("website_pitch"), self.scored.outreach_message)
        self.assertEqual(row.value("ai_summary"), self.scored.ai_summary)
        self.assertIsNone(row.value("automation_pitch"))

    def test_a_band_is_never_invented_for_a_lead_in_isolation(self):
        # A band is a percentile against a cohort, computed in SQL over every stored
        # business. One lead cannot know its own, and a confident wrong word in front of
        # the operator is worse than a blank.
        row = build_row(record_from_scored_lead(self.scored))

        self.assertIsNone(row.value("audience_band"))
        self.assertIsNone(row.value("banding_method"))

    def test_a_band_supplied_by_the_caller_is_used(self):
        row = build_row(
            record_from_scored_lead(self.scored, audience_band="large", banding_method="relative")
        )

        self.assertEqual(row.value("audience_band"), "large")
        self.assertEqual(row.value("banding_method"), "relative")

    def test_the_listing_url_provenances_the_google_evidence(self):
        row = build_row(
            record_from_scored_lead(
                self.scored,
                evidence=Evidence(reviews=340, rating=4.6),
                peak_hours="Busiest Fri 7-9pm",
            )
        )

        self.assertEqual(row.source("reviews"), self.lead.source_url)
        self.assertEqual(row.source("rating"), self.lead.source_url)
        self.assertIsNone(row.source("runs_ads"))

    def test_fields_the_lead_cannot_know_arrive_as_keywords(self):
        row = build_row(
            record_from_scored_lead(
                self.scored,
                country="India",
                state="Karnataka",
                email="hello@example.invalid",
                facebook_url="https://facebook.com/blush",
                contact=ContactPerson(name="Asha", role="owner", source="review_reply"),
                automation_opportunities=("booking", "reminders"),
                automation_pitch="Automate the order book.",
                verdict=OperatorVerdict(my_verdict="visit", channel="visit"),
            )
        )

        self.assertEqual(row.value("country"), "India")
        self.assertEqual(row.value("state"), "Karnataka")
        self.assertEqual(row.value("email"), "hello@example.invalid")
        self.assertEqual(row.value("contact_name"), "Asha")
        self.assertEqual(row.value("contact_role"), "owner")
        self.assertEqual(row.value("contact_source"), "review_reply")
        self.assertEqual(row.value("automation_opportunities"), "booking; reminders")
        self.assertEqual(row.value("my_verdict"), "visit")


# --- excel: the query-to-record mapping (no database) -----------------------------------


SELECT_LIST = EXPORT_QUERY.split("FROM lead_bands")[0]
QUERY_ALIASES = set(re.findall(r"AS\s+(\w+),?\s*$", SELECT_LIST, flags=re.MULTILINE))


class RecordingRow(dict):
    """A row that remembers which column names were asked for."""

    def __init__(self, values=None):
        super().__init__(values or {})
        self.read: list[str] = []

    def get(self, key, default=None):
        self.read.append(key)
        return super().get(key, default)


FULL_ROW = {
    "business_name": "Blush Salon",
    "niche_id": "bakery",
    "business_type": "beauty_salon",
    "country": "India",
    "state": "Karnataka",
    "city": "Bangalore",
    "search_area": "Indiranagar",
    "address": "100ft Road, Indiranagar",
    "lat": 12.978,
    "lng": 77.641,
    "phone": "+91 90000 00000",
    "email": "hello@example.invalid",
    "website": None,
    "instagram_handle": "@blush.salon",
    "facebook_url": "https://facebook.com/blush",
    "contact_name": "Asha",
    "contact_role": "owner",
    "contact_phone": "+91 90000 00001",
    "contact_email": "asha@example.invalid",
    "contact_source": "review_reply",
    "reviews": 340,
    "rating": "4.6",
    "peak_hours": "Busiest Fri 7-9pm",
    "google_source_url": "https://maps.example/blush",
    "followers": "1200",
    "engagement_rate": "0.042",
    "runs_ads": "true",
    "instagram_source_url": "https://www.instagram.com/blush.salon/",
    "total_score": 71,
    "audience_band": "medium",
    "banding_method": "relative",
    "signals": ["no website", "public phone"],
    "ai_summary": None,
    "website_pitch": "A one-page site with your catalogue.",
    "automation_opportunities": ["booking", "reminders"],
    "automation_pitch": "Automate the order book.",
    "my_verdict": "visit",
    "notes": "Owner in after 4pm.",
    "contacted_on": date(2026, 8, 13),
    "channel": "visit",
    "outcome": "meeting booked",
}


class QueryShapeTests(unittest.TestCase):
    """The SQL and the mapper have to agree, and neither knows about the other."""

    def test_the_select_list_and_the_mapper_name_the_same_columns(self):
        # Both directions matter. A column selected but never read is dead SQL; a column
        # read but never selected is a cell that is blank forever and looks like missing
        # data rather than a broken query.
        row = RecordingRow()
        record_from_row(row)

        self.assertEqual(set(row.read), QUERY_ALIASES)

    def test_the_select_list_covers_every_sheet_column_it_can(self):
        # The columns the query cannot fill are named here so that wiring one up is a
        # visible change rather than a surprise.
        derived = {"niche", "contacted_on"}
        unfillable = set(COLUMNS) - QUERY_ALIASES - derived

        self.assertEqual(unfillable, set())

    def test_the_payload_keys_are_the_ones_the_query_reads(self):
        # Pinned against a payload this test writes itself, because `enrichments.data` is a
        # contract with a fetcher that does not exist yet.
        google = {
            excel.GOOGLE_TYPE_KEY: "beauty_salon",
            excel.GOOGLE_RATING_KEY: "4.6",
            excel.GOOGLE_PEAK_HOURS_KEY: "Busiest Fri 7-9pm",
        }
        instagram = {
            excel.INSTAGRAM_FOLLOWERS_KEY: "1200",
            excel.INSTAGRAM_ENGAGEMENT_KEY: "0.042",
            excel.INSTAGRAM_ADS_KEY: "true",
        }
        for key in list(google) + list(instagram):
            with self.subTest(key=key):
                self.assertIn(f"->>'{key}'", EXPORT_QUERY)

        self.assertEqual(sorted(google), ["peak_hours", "rating", "type"])
        self.assertEqual(sorted(instagram), ["engagement_rate", "followers", "runs_ads"])

    def test_the_query_does_not_order_or_filter(self):
        # Row order is `view.sort_key`'s decision, and the sheet is the whole corpus.
        body = EXPORT_QUERY.split("FROM lead_bands")[1]

        self.assertNotIn("\nORDER BY", body)
        self.assertNotIn("\n WHERE", body)


class RowMappingTests(unittest.TestCase):
    def test_a_full_row_maps_into_every_sheet_column(self):
        row = build_row(record_from_row(FULL_ROW))

        self.assertEqual(row.value("business_name"), "Blush Salon")
        self.assertEqual(row.value("niche"), NICHE_PROFILES["bakery"].label)
        self.assertEqual(row.value("business_type"), "beauty_salon")
        self.assertEqual(row.value("search_area"), "Indiranagar")
        self.assertEqual(row.value("reviews"), 340)
        self.assertAlmostEqual(row.value("rating"), 4.6, places=12)
        self.assertEqual(row.value("followers"), 1200)
        self.assertAlmostEqual(row.value("engagement_rate"), 0.042, places=12)
        self.assertEqual(row.value("runs_ads"), "yes")
        self.assertEqual(row.value("peak_hours"), "Busiest Fri 7-9pm")
        self.assertEqual(row.value("total_score"), 71)
        self.assertEqual(row.value("audience_band"), "medium")
        self.assertEqual(row.value("signals"), "no website; public phone")
        self.assertEqual(row.value("automation_opportunities"), "booking; reminders")
        self.assertEqual(row.value("outcome"), "meeting booked")

    def test_the_source_urls_come_from_the_enrichment_rows(self):
        row = build_row(record_from_row(FULL_ROW))

        self.assertEqual(row.source("reviews"), "https://maps.example/blush")
        self.assertEqual(row.source("followers"), "https://www.instagram.com/blush.salon/")
        self.assertIsNone(row.source("runs_ads"))

    def test_a_row_of_nulls_produces_a_full_width_row_of_blanks(self):
        row = build_row(record_from_row(dict.fromkeys(QUERY_ALIASES)))

        self.assertEqual(len(row.cells), len(COLUMNS))
        for name, value in zip(COLUMNS, row.values, strict=True):
            with self.subTest(column=name):
                self.assertIsNone(value)

    def test_a_missing_key_is_treated_as_a_null_rather_than_raising(self):
        # `record_from_row` reads with `.get`, so a lead that predates a column still
        # exports. The alternative is one KeyError taking the whole sheet down.
        row = build_row(record_from_row({"business_name": "Sparse"}))

        self.assertEqual(row.value("business_name"), "Sparse")
        self.assertIsNone(row.value("reviews"))

    def test_an_unparseable_number_reads_as_unknown_not_zero(self):
        # The same rule the banding view applies to review counts. A zero here would be a
        # statement about the business made on the strength of a parse failure.
        for value in ("unavailable", "", "  ", "1,200", "1.2K", "n/a", None):
            with self.subTest(value=value):
                mapped = record_from_row({"business_name": "X", "reviews": value,
                                          "followers": value, "rating": value,
                                          "engagement_rate": value, "total_score": value})
                built = build_row(mapped)
                self.assertIsNone(built.value("reviews"))
                self.assertIsNone(built.value("followers"))
                self.assertIsNone(built.value("rating"))
                self.assertIsNone(built.value("engagement_rate"))
                self.assertIsNone(built.value("total_score"))

    def test_a_numeric_string_of_zero_is_kept_as_zero(self):
        built = build_row(record_from_row({"business_name": "X", "reviews": "0", "rating": "0"}))

        self.assertEqual(built.value("reviews"), 0)
        self.assertEqual(built.value("rating"), 0.0)

    def test_unrecognised_runs_ads_text_is_unknown_not_false(self):
        cases = [
            ("true", "yes"),
            ("t", "yes"),
            ("1", "yes"),
            ("false", "no"),
            ("0", "no"),
            ("maybe", None),
            ("", None),
            (None, None),
        ]
        for raw, expected in cases:
            with self.subTest(runs_ads=raw):
                built = build_row(record_from_row({"business_name": "X", "runs_ads": raw}))
                self.assertEqual(built.value("runs_ads"), expected)

    def test_signals_written_as_an_object_are_still_shown(self):
        built = build_row(
            record_from_row({"business_name": "X", "signals": {"website": "none", "phone": "yes"}})
        )

        self.assertEqual(built.value("signals"), "phone: yes; website: none")

    def test_signals_written_as_a_bare_string_are_still_shown(self):
        built = build_row(record_from_row({"business_name": "X", "signals": "no website"}))

        self.assertEqual(built.value("signals"), "no website")

    def test_an_empty_business_name_does_not_crash_the_row(self):
        built = build_row(record_from_row({"business_name": None}))

        self.assertIsNone(built.value("business_name"))


class FakeCursor:
    """Just enough DB-API to exercise `fetch_records` without a driver."""

    def __init__(self, names, rows):
        # psycopg hands back 7-tuples; only the first element is ever read.
        self.description = [(name, None, None, None, None, None, None) for name in names]
        self._rows = rows
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql):
        self.executed.append(sql)

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class FetchRecordsTests(unittest.TestCase):
    def test_columns_are_named_from_the_cursor_not_from_their_position(self):
        # Deliberately not the order of the SELECT list: reordering the query must not
        # shift `phone` into `email`.
        cursor = FakeCursor(
            ["email", "business_name", "phone", "total_score"],
            [("hello@example.invalid", "Blush Salon", "+91 90000 00000", 71)],
        )

        records = fetch_records(FakeConnection(cursor))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].business_name, "Blush Salon")
        self.assertEqual(records[0].phone, "+91 90000 00000")
        self.assertEqual(records[0].email, "hello@example.invalid")

    def test_the_export_query_is_the_one_executed(self):
        cursor = FakeCursor(["business_name"], [])

        fetch_records(FakeConnection(cursor))

        self.assertEqual(cursor.executed, [EXPORT_QUERY])

    def test_a_row_narrower_than_the_description_raises(self):
        cursor = FakeCursor(["business_name", "phone"], [("Blush Salon",)])

        with self.assertRaises(ValueError):
            fetch_records(FakeConnection(cursor))

    def test_an_empty_result_set_exports_nothing_rather_than_failing(self):
        self.assertEqual(fetch_records(FakeConnection(FakeCursor(["business_name"], []))), [])


# --- excel: what lands in the file ------------------------------------------------------


SAMPLE_RECORDS = [
    LeadRecord(
        business_name="Blush Salon",
        niche="Beauty Salon",
        city="Bangalore",
        search_area="Indiranagar",
        address="100ft Road, Indiranagar",
        lat=12.978,
        lng=77.641,
        phone="+91 90000 00000",
        instagram_handle="@blush.salon",
        contact=ContactPerson(name="Asha", role="owner"),
        evidence=Evidence(
            reviews=340, rating=4.6, followers=1200, engagement_rate=0.042, runs_ads=True
        ),
        peak_hours="Busiest Fri 7-9pm",
        evidence_sources=evidence_sources(
            google_url="https://maps.example/blush",
            instagram_url="https://www.instagram.com/blush.salon/",
        ),
        total_score=71,
        audience_band="medium",
        banding_method="relative",
        signals=("no website", "public phone"),
        website_pitch="A one-page site with your catalogue.",
    ),
    LeadRecord(business_name="Nothing Known", search_area="Indiranagar"),
    LeadRecord(
        business_name="Whitefield Cafe",
        search_area="Whitefield",
        total_score=90,
        evidence=Evidence(reviews=0, rating=0.0),
        verdict=OperatorVerdict(
            my_verdict="visit",
            notes="Owner in after 4pm.",
            contacted_on=date(2026, 8, 13),
            channel="visit",
            outcome="meeting booked",
        ),
    ),
]


class WorkbookTests(unittest.TestCase):
    """Every assertion here is on a workbook that was written and reopened."""

    def setUp(self):
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.sheet, self.path = written_sheet(SAMPLE_RECORDS, self.directory)

    def test_the_sheet_is_the_only_one_and_is_named_leads(self):
        book = load_workbook(self.path)

        self.assertEqual(book.sheetnames, [SHEET_TITLE])

    def test_the_header_is_the_column_keys_verbatim_and_in_order(self):
        # Not prettified: a later phase reads the operator's edits back by column name.
        header = [cell.value for cell in self.sheet[1]]

        self.assertEqual(header, EXPECTED_COLUMNS)

    def test_the_header_is_bold(self):
        self.assertTrue(cell_at(self.sheet, 1, "business_name").font.bold)

    def test_the_header_row_and_the_name_column_are_frozen(self):
        # B2 freezes row 1 and column A. The operator columns are 39 columns right of the
        # business name, and scrolling to them without the name in view is how a verdict
        # lands on the wrong business.
        self.assertEqual(self.sheet.freeze_panes, "B2")

    def test_the_autofilter_covers_the_header_and_every_data_row(self):
        last = column_letter(COLUMNS[-1])

        self.assertEqual(self.sheet.auto_filter.ref, f"A1:{last}{len(SAMPLE_RECORDS) + 1}")

    def test_every_record_reaches_the_file_at_full_width(self):
        self.assertEqual(self.sheet.max_row, len(SAMPLE_RECORDS) + 1)
        self.assertEqual(self.sheet.max_column, len(COLUMNS))
        for number in range(2, self.sheet.max_row + 1):
            with self.subTest(row=number):
                self.assertEqual(len(self.sheet[number]), len(COLUMNS))

    def test_a_lead_with_no_optional_data_still_occupies_a_full_row(self):
        # The ragged-row case, checked in the file rather than in the row builder.
        rows = {self.sheet.cell(row=n, column=1).value: n for n in range(2, self.sheet.max_row + 1)}
        number = rows["Nothing Known"]

        self.assertEqual(len(self.sheet[number]), len(COLUMNS))
        self.assertEqual(cell_at(self.sheet, number, "outcome").column, len(COLUMNS))

    def test_the_rows_are_in_the_sort_order_not_the_input_order(self):
        names = [self.sheet.cell(row=n, column=1).value for n in range(2, self.sheet.max_row + 1)]

        self.assertEqual(names, ["Blush Salon", "Nothing Known", "Whitefield Cafe"])

    def test_unknown_values_land_as_empty_cells_never_as_zero_or_none(self):
        rows = {self.sheet.cell(row=n, column=1).value: n for n in range(2, self.sheet.max_row + 1)}
        number = rows["Nothing Known"]

        for name in COLUMNS:
            if name in {"business_name", "search_area"}:
                continue
            with self.subTest(column=name):
                self.assertIsNone(cell_at(self.sheet, number, name).value)

    def test_a_real_zero_survives_the_round_trip(self):
        rows = {self.sheet.cell(row=n, column=1).value: n for n in range(2, self.sheet.max_row + 1)}
        number = rows["Whitefield Cafe"]

        self.assertEqual(cell_at(self.sheet, number, "reviews").value, 0)
        self.assertEqual(cell_at(self.sheet, number, "rating").value, 0)

    def test_evidence_cells_carry_their_source_url_as_a_comment(self):
        expected = {
            "reviews": "https://maps.example/blush",
            "rating": "https://maps.example/blush",
            "peak_hours": "https://maps.example/blush",
            "followers": "https://www.instagram.com/blush.salon/",
            "engagement_rate": "https://www.instagram.com/blush.salon/",
        }
        for name, url in expected.items():
            with self.subTest(column=name):
                comment = cell_at(self.sheet, 2, name).comment
                self.assertIsNotNone(comment, f"{name} lost its receipt")
                self.assertEqual(comment.text, url)

    def test_the_comment_says_the_export_wrote_it(self):
        # So the operator can tell a generated note from a colleague's markup.
        self.assertEqual(cell_at(self.sheet, 2, "reviews").comment.author, COMMENT_AUTHOR)

    def test_a_cell_with_no_honest_source_carries_no_comment(self):
        self.assertIsNotNone(cell_at(self.sheet, 2, "runs_ads").value)
        self.assertIsNone(cell_at(self.sheet, 2, "runs_ads").comment)

    def test_only_evidence_cells_carry_comments(self):
        commented = {
            name for name in COLUMNS if cell_at(self.sheet, 2, name).comment is not None
        }

        self.assertEqual(
            commented, set(GOOGLE_EVIDENCE_COLUMNS) | set(INSTAGRAM_EVIDENCE_COLUMNS)
        )

    def test_the_operator_headers_are_a_different_colour_from_the_rest(self):
        operator = cell_at(self.sheet, 1, "my_verdict").fill
        other = cell_at(self.sheet, 1, "business_name").fill

        self.assertEqual(operator.patternType, "solid")
        self.assertEqual(operator.fgColor.rgb, OPERATOR_HEADER_FILL_COLOR)
        self.assertNotEqual(operator.fgColor.rgb, other.fgColor.rgb)

    def test_the_operator_cells_are_the_only_shaded_cells_in_a_row(self):
        # They are the only columns the operator edits and the only ones a later phase
        # reads back, so they have to be findable without counting to column AI.
        shaded = {
            name
            for name in COLUMNS
            if cell_at(self.sheet, 2, name).fill.patternType == "solid"
        }

        self.assertEqual(shaded, set(OPERATOR_COLUMNS))

    def test_every_operator_cell_in_every_row_is_shaded(self):
        for number in range(2, self.sheet.max_row + 1):
            for name in OPERATOR_COLUMNS:
                with self.subTest(row=number, column=name):
                    fill = cell_at(self.sheet, number, name).fill
                    self.assertEqual(fill.patternType, "solid")
                    self.assertEqual(fill.fgColor.rgb, OPERATOR_FILL_COLOR)

    def test_an_operator_column_is_shaded_even_when_it_is_empty(self):
        rows = {self.sheet.cell(row=n, column=1).value: n for n in range(2, self.sheet.max_row + 1)}
        cell = cell_at(self.sheet, rows["Nothing Known"], "notes")

        self.assertIsNone(cell.value)
        self.assertEqual(cell.fill.fgColor.rgb, OPERATOR_FILL_COLOR)

    def test_number_formats_survive_the_write(self):
        for name, expected in NUMBER_FORMATS.items():
            with self.subTest(column=name):
                self.assertEqual(cell_at(self.sheet, 2, name).number_format, expected)

    def test_engagement_rate_is_stored_as_a_fraction_and_only_displayed_as_a_percentage(self):
        cell = cell_at(self.sheet, 2, "engagement_rate")

        self.assertAlmostEqual(cell.value, 0.042, places=12)
        self.assertEqual(cell.number_format, "0.0%")

    def test_a_contacted_date_lands_as_a_date(self):
        rows = {self.sheet.cell(row=n, column=1).value: n for n in range(2, self.sheet.max_row + 1)}
        cell = cell_at(self.sheet, rows["Whitefield Cafe"], "contacted_on")

        self.assertEqual(cell.value, datetime(2026, 8, 13))
        self.assertEqual(cell.number_format, "yyyy-mm-dd")

    def test_prose_columns_are_wrapped_so_a_row_stays_one_row(self):
        for name in ("address", "signals", "website_pitch", "notes"):
            with self.subTest(column=name):
                self.assertTrue(cell_at(self.sheet, 2, name).alignment.wrap_text)

    def test_short_columns_are_not_wrapped(self):
        self.assertFalse(cell_at(self.sheet, 2, "phone").alignment.wrap_text)

    def test_every_column_is_given_a_width(self):
        for name in COLUMNS:
            with self.subTest(column=name):
                width = self.sheet.column_dimensions[column_letter(name)].width
                self.assertGreater(width, 0)


class WorkbookEdgeTests(unittest.TestCase):
    def setUp(self):
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_an_empty_corpus_still_writes_a_usable_header(self):
        sheet, _ = written_sheet([], self.directory)
        last = column_letter(COLUMNS[-1])

        self.assertEqual([cell.value for cell in sheet[1]], EXPECTED_COLUMNS)
        self.assertEqual(sheet.max_row, 1)
        self.assertEqual(sheet.auto_filter.ref, f"A1:{last}1")
        self.assertEqual(sheet.freeze_panes, "B2")

    def test_a_row_of_the_wrong_width_is_refused(self):
        short = ExportRow(cells=(Cell(value="Blush Salon"),))

        with self.assertRaises(ValueError) as caught:
            build_workbook([short])

        self.assertIn(str(len(COLUMNS)), str(caught.exception))

    def test_a_short_row_never_reaches_the_file(self):
        path = self.directory / "leads.xlsx"

        with self.assertRaises(ValueError):
            write_workbook([ExportRow(cells=(Cell(value="x"),))], path)

        self.assertFalse(path.exists())

    def test_the_row_stream_is_consumed_exactly_once(self):
        # A cursor is not rewindable. Iterating twice would silently double the sheet or
        # produce an empty one, depending on the source.
        passes = []

        def counted():
            passes.append(1)
            yield from build_rows(SAMPLE_RECORDS)

        book = build_workbook(counted())

        self.assertEqual(len(passes), 1)
        self.assertEqual(book[SHEET_TITLE].max_row, len(SAMPLE_RECORDS) + 1)

    def test_building_a_workbook_touches_no_file(self):
        build_workbook(build_rows(SAMPLE_RECORDS))

        self.assertEqual(list(self.directory.iterdir()), [])

    def test_a_thousand_rows_all_arrive(self):
        many = [
            record(f"Business {index:04d}", search_area=f"Area {index % 7}", total_score=index)
            for index in range(1000)
        ]

        sheet, _ = written_sheet(many, self.directory)

        self.assertEqual(sheet.max_row, 1001)
        self.assertEqual(sheet.auto_filter.ref, f"A1:{column_letter(COLUMNS[-1])}1001")


class AtomicWriteTests(unittest.TestCase):
    """A half-written sheet replacing a good one costs the operator a week of visits."""

    def setUp(self):
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.path = self.directory / "leads.xlsx"
        export_records([record("Yesterday's Export", search_area="Indiranagar")], self.path)
        self.before = self.path.read_bytes()

    def leftovers(self) -> list[str]:
        return sorted(item.name for item in self.directory.iterdir() if item != self.path)

    def assert_previous_sheet_is_untouched(self):
        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.read_bytes(), self.before)
        sheet = load_workbook(self.path)[SHEET_TITLE]
        self.assertEqual(sheet.cell(row=2, column=1).value, "Yesterday's Export")
        self.assertEqual(self.leftovers(), [])

    def test_a_successful_write_returns_the_path_and_replaces_the_file(self):
        returned = export_records([record("Today's Export")], self.path)

        self.assertEqual(returned, self.path)
        self.assertEqual(load_workbook(self.path)[SHEET_TITLE].cell(row=2, column=1).value,
                         "Today's Export")
        self.assertEqual(self.leftovers(), [])

    def test_missing_parent_directories_are_created(self):
        nested = self.directory / "exports" / "2026-08" / "leads.xlsx"

        export_records([record("Nested")], nested)

        self.assertTrue(nested.is_file())

    def test_a_row_stream_that_dies_partway_leaves_the_previous_sheet_intact(self):
        # The dropped-cursor case: rows 1..n arrive, then the connection goes away.
        def failing_rows():
            yield from build_rows([record("Partial One"), record("Partial Two")])
            raise RuntimeError("connection dropped on row 4000")

        with self.assertRaises(RuntimeError):
            write_workbook(failing_rows(), self.path)

        self.assert_previous_sheet_is_untouched()

    def test_a_failure_while_saving_never_reaches_the_destination(self):
        # The strongest form: the temporary file really does get half a workbook written
        # into it before the failure, so a non-atomic writer would leave that on disk under
        # the operator's filename.
        def broken_save(_self, filename):
            Path(filename).write_bytes(b"PK\x03\x04half a workbook")
            raise OSError(28, "No space left on device")

        with patch.object(Workbook, "save", broken_save):
            with self.assertRaises(OSError):
                export_records([record("Doomed")], self.path)

        self.assert_previous_sheet_is_untouched()

    def test_a_bad_value_on_a_late_row_leaves_the_previous_sheet_intact(self):
        rows = build_rows([record("Fine")])
        rows.append(ExportRow(cells=(Cell(value="too narrow"),)))

        with self.assertRaises(ValueError):
            write_workbook(rows, self.path)

        self.assert_previous_sheet_is_untouched()

    def test_a_refused_replace_leaves_the_old_sheet_readable(self):
        # Windows refuses the replace while the operator has the workbook open in Excel.
        # That is the correct outcome; overwriting a file someone is editing is not.
        with patch.object(excel.os, "replace", side_effect=PermissionError("open in Excel")):
            with self.assertRaises(PermissionError):
                export_records([record("Blocked")], self.path)

        self.assert_previous_sheet_is_untouched()

    def test_the_temporary_file_sits_beside_the_target_not_in_the_system_temp_dir(self):
        # `os.replace` is only atomic within a filesystem. A cross-device move degrades to
        # copy-then-delete, which is exactly the half-written window this avoids.
        seen: list[Path] = []

        def recording_save(_self, filename):
            seen.append(Path(filename))
            raise RuntimeError("stop here")

        with patch.object(Workbook, "save", recording_save):
            with self.assertRaises(RuntimeError):
                export_records([record("Doomed")], self.path)

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].parent, self.path.parent)
        self.assertNotEqual(seen[0], self.path)
        self.assert_previous_sheet_is_untouched()

    def test_writing_a_brand_new_file_leaves_nothing_behind_on_failure(self):
        fresh = self.directory / "first-ever.xlsx"

        def failing_rows():
            yield from build_rows([record("One")])
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            write_workbook(failing_rows(), fresh)

        self.assertFalse(fresh.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_keyboard_interrupt_is_cleaned_up_too(self):
        # `except BaseException`, not `except Exception`: Ctrl-C during a 40,000-row export
        # must not leave a temporary file behind either.
        def interrupted_rows():
            yield from build_rows([record("One")])
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            write_workbook(interrupted_rows(), self.path)

        self.assert_previous_sheet_is_untouched()

    def test_repeated_exports_replace_rather_than_accumulate(self):
        for index in range(3):
            export_records([record(f"Run {index}")], self.path)

        self.assertEqual(self.leftovers(), [])
        self.assertEqual(load_workbook(self.path)[SHEET_TITLE].cell(row=2, column=1).value,
                         "Run 2")


# --- integration: the query and the banding view against a real Postgres ----------------
#
# Skipped unless `LEAD_ENGINE_TEST_DSN` is set, for the reason `tests/test_migrations.py`
# gives: a suite that turns red without Docker gets excluded within a week and then never
# runs at all. Only the two claims that cannot be checked in-process live here -- that
# `EXPORT_QUERY` still names the columns `record_from_row` reads, and that the exporter
# reports the banding rule the cohort size actually put in force.

try:
    import psycopg
    from psycopg.types.json import Jsonb

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)

NOW = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def vector_extension():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


@pytest.fixture
def db(vector_extension):
    """A fully migrated schema that exists only for this test."""
    schema = "test_" + uuid.uuid4().hex
    connection = psycopg.connect(DSN)
    connection.execute(f'CREATE SCHEMA "{schema}"')
    connection.execute(f'SET search_path = "{schema}", public')
    connection.commit()
    try:
        apply_migrations(connection)
        yield connection
    finally:
        connection.rollback()
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        connection.commit()
        connection.close()


def seed_business(
    connection,
    *,
    name="Blush Salon",
    niche="salon",
    city="Bangalore",
    search_area="Indiranagar",
    audience_index=0.5,
    total=71,
    reviews=340,
    signals=("no website",),
):
    business_id = uuid.uuid4()
    connection.execute(
        "INSERT INTO businesses (id, name, niche_id, city, search_area, country, state,"
        " address, lat, lng, phone, email, website, instagram_handle, facebook_url)"
        " VALUES (%s, %s, %s, %s, %s, 'India', 'Karnataka', '100ft Road', 12.978, 77.641,"
        " '+91 90000 00000', 'hello@example.invalid', NULL, '@blush.salon', NULL)",
        (business_id, name, niche, city, search_area),
    )
    if audience_index is not None or total is not None:
        connection.execute(
            "INSERT INTO scores (business_id, scorer_version, signals, evidence,"
            " audience_index, total, scored_at) VALUES (%s, 'google-only', %s, '{}', %s, %s, %s)",
            (business_id, Jsonb(list(signals)), audience_index, total, NOW),
        )
    if reviews is not None:
        connection.execute(
            "INSERT INTO enrichments (business_id, source, status, data, source_url, fetched_at)"
            " VALUES (%s, 'google_maps', 'ok', %s, %s, %s)",
            (
                business_id,
                Jsonb({"reviews": reviews, "rating": "4.6", "type": "beauty_salon",
                       "peak_hours": "Busiest Fri 7-9pm"}),
                "https://maps.example/blush",
                NOW,
            ),
        )
    return business_id


@pytest.mark.integration
@needs_postgres
def test_the_export_query_runs_and_names_the_columns_the_mapper_reads(db):
    seed_business(db)

    with db.cursor() as cursor:
        cursor.execute(EXPORT_QUERY)
        names = {description[0] for description in cursor.description}

    probe = RecordingRow()
    record_from_row(probe)
    assert names == set(probe.read)


@pytest.mark.integration
@needs_postgres
def test_a_seeded_business_survives_the_whole_round_trip(db, tmp_path):
    seed_business(db)

    path = excel.export(db, tmp_path / "leads.xlsx")
    sheet = load_workbook(path)[SHEET_TITLE]

    assert [cell.value for cell in sheet[1]] == EXPECTED_COLUMNS
    assert cell_at(sheet, 2, "business_name").value == "Blush Salon"
    assert cell_at(sheet, 2, "search_area").value == "Indiranagar"
    assert cell_at(sheet, 2, "reviews").value == 340
    assert cell_at(sheet, 2, "total_score").value == 71
    assert cell_at(sheet, 2, "signals").value == "no website"
    assert cell_at(sheet, 2, "reviews").comment.text == "https://maps.example/blush"


@pytest.mark.integration
@needs_postgres
def test_ai_summary_reads_from_the_latest_scores_row_not_a_hardcoded_null(db, tmp_path):
    """Was `NULL::text AS ai_summary` -- nothing wrote one, nothing could read one.
    `Engine.execute_enrichment` now writes it into the latest score's own `evidence` jsonb,
    beside `pitch_angle`; this proves the export query actually reads it back from there."""
    business_id = seed_business(db, name="Enriched Salon")
    db.execute(
        "INSERT INTO scores (business_id, scorer_version, signals, evidence, audience_index,"
        " total, scored_at) VALUES (%s, 'enriched', '{}', %s, 0.5, 84, %s)",
        (
            business_id,
            Jsonb({"pitch_angle": "no website", "ai_summary": "A salon with no website."}),
            NOW + timedelta(minutes=5),
        ),
    )

    path = excel.export(db, tmp_path / "leads.xlsx")
    sheet = load_workbook(path)[SHEET_TITLE]

    # The LATEST score (enriched, scored 5 minutes later) wins, not the google-only one.
    assert cell_at(sheet, 2, "total_score").value == 84
    assert cell_at(sheet, 2, "ai_summary").value == "A salon with no website."


@pytest.mark.integration
@needs_postgres
def test_ai_summary_is_blank_on_a_google_only_score(db, tmp_path):
    """No enrichment pass has run, so there is nothing to summarise -- blank, not a guess."""
    seed_business(db, name="Undiscovered Salon")

    path = excel.export(db, tmp_path / "leads.xlsx")
    sheet = load_workbook(path)[SHEET_TITLE]

    assert cell_at(sheet, 2, "ai_summary").value is None


@pytest.mark.integration
@needs_postgres
def test_banding_method_reads_absolute_below_a_thirty_business_cohort(db):
    for index in range(5):
        seed_business(db, name=f"b{index}", audience_index=index / 5)

    records = fetch_records(db)

    assert len(records) == 5
    assert {item.banding_method for item in records} == {"absolute"}
    # 340 reviews sits in the middle absolute bucket, and every one of them is there.
    assert {item.audience_band for item in records} == {"medium"}


@pytest.mark.integration
@needs_postgres
def test_banding_method_reads_relative_at_thirty(db):
    for index in range(30):
        seed_business(db, name=f"b{index}", audience_index=index / 30)

    records = fetch_records(db)

    assert len(records) == 30
    assert {item.banding_method for item in records} == {"relative"}
    # Relative, so the same review count no longer decides the band.
    assert {item.audience_band for item in records} == {"small", "medium", "large"}


@pytest.mark.integration
@needs_postgres
def test_a_business_the_pipeline_failed_on_still_reaches_the_sheet(db):
    seed_business(db, name="Never Scored", audience_index=None, total=None, reviews=None)

    records = fetch_records(db)
    row = build_row(records[0])

    assert len(records) == 1
    assert row.value("business_name") == "Never Scored"
    assert row.value("audience_band") == "unknown"
    # Unknown, not zero: nothing is known about this one and the sheet has to say so.
    assert row.value("total_score") is None
    assert row.value("reviews") is None


@pytest.mark.integration
@needs_postgres
def test_the_signals_come_from_the_same_score_row_as_the_total(db):
    # `signals` needs a second look at `scores`, which `lead_bands` does not expose. If the
    # two ORDER BYs ever drift, the sheet shows one score version's total beside another's
    # signals -- and both look plausible.
    business_id = seed_business(db, total=40, audience_index=0.2, signals=("stale signal",))
    db.execute(
        "INSERT INTO scores (business_id, scorer_version, signals, evidence, audience_index,"
        " total, scored_at) VALUES (%s, 'instagram', %s, '{}', 0.9, 88, %s)",
        (business_id, Jsonb(["fresh signal"]), NOW),
    )

    row = build_row(fetch_records(db)[0])

    assert row.value("total_score") == 88
    assert row.value("signals") == "fresh signal"


@pytest.mark.integration
@needs_postgres
def test_an_empty_corpus_exports_a_header_only_sheet(db, tmp_path):
    path = excel.export(db, tmp_path / "leads.xlsx")
    sheet = load_workbook(path)[SHEET_TITLE]

    assert [cell.value for cell in sheet[1]] == EXPECTED_COLUMNS
    assert sheet.max_row == 1


if __name__ == "__main__":
    unittest.main()
