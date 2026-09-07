"""Phase 10: the Master/verdict sync and daily snapshots, against a fake Sheets client.

WHAT THIS SUITE DEFENDS

*Operator edits always win.* `sync_master` must never construct a value for any of
`OPERATOR_COLUMNS` -- not even to write back what is already there.
`OperatorColumnsNeverWrittenTests` is the load-bearing class here: it seeds a real,
human-typed verdict on the fake sheet, runs a sync, and asserts the batch payload contains
not one cell address inside that row's operator columns, and that the raw stored value is
byte-for-byte unchanged.

*Exactly one network round trip per sync, however many rows.* `sync_master` must issue at
most one `batch_update_values` call (`BatchingTests`) -- see `sheets.py`'s own docstring on
why: Google's per-user write quota is small enough that a per-row implementation would
exhaust it on a cohort this system's own cost model already targets.

*A brand-new service account cannot own a spreadsheet.* Not simulated here (it is a real,
already-proven fact from a live call this session -- see `sheets.py`'s docstring) but
`GoogleSheetsClient` is shape-tested to confirm it never calls `.create()`, only ever reads
and writes an id handed to it.

NO TEST HERE REACHES THE NETWORK. `GoogleSheetsClientShapeTests` drives a recording stub
standing in for `googleapiclient`'s fluent interface; everything else drives `FakeSheetsClient`.
"""

from __future__ import annotations

import re
import unittest
from datetime import date

from lead_engine.export.sheets import (
    LAST_COLUMN_LETTER,
    MASTER_HEADER,
    MASTER_SHEET,
    PLACE_ID_COLUMN,
    GoogleSheetsClient,
    _column_letter,
    ensure_master_header,
    read_master,
    read_operator_verdicts,
    sync_master,
    write_snapshot,
)
from lead_engine.export.view import COLUMNS, OPERATOR_COLUMNS, LeadRecord


def record(name: str, **fields) -> LeadRecord:
    return LeadRecord(business_name=name, **fields)


_CELL = re.compile(r"^([A-Z]+)(\d+)$")


def _col_to_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


class FakeSheetsClient:
    """An in-memory stand-in for the five operations `sheets.py` actually issues.

    Good enough to prove the module's own logic, not a general Sheets simulator: it
    understands exactly the address shapes `sheets.py` constructs (a whole-tab read, a
    single `A1`-style header check, a single-cell write, and a block write starting at
    `A1`), because those are the only shapes any caller in this codebase ever produces.
    """

    def __init__(self) -> None:
        self.sheets: dict[str, list[list]] = {}
        self.batch_calls: list[dict[str, list[list]]] = []
        self.append_calls: list[tuple[str, list[list]]] = []
        self.add_sheet_calls: list[str] = []

    def get_values(self, range_name: str) -> list[list]:
        sheet, ref = range_name.split("!", 1)
        grid = self.sheets.get(sheet, [])
        if ref == "A1:A1":
            if not grid or not grid[0]:
                return []
            return [[grid[0][0]]]
        # Whole-tab reads: "A1:<LAST_COLUMN_LETTER>".
        return [list(row) for row in grid]

    def batch_update_values(self, updates) -> None:
        self.batch_calls.append(dict(updates))
        for range_name, values in updates.items():
            sheet, ref = range_name.split("!", 1)
            grid = self.sheets.setdefault(sheet, [])
            match = _CELL.match(ref)
            assert match, f"unsupported test address: {ref!r}"
            start_col = _col_to_index(match.group(1))
            start_row = int(match.group(2)) - 1
            for row_offset, row_values in enumerate(values):
                row_index = start_row + row_offset
                while len(grid) <= row_index:
                    grid.append([])
                row = grid[row_index]
                for col_offset, value in enumerate(row_values):
                    col_index = start_col + col_offset
                    while len(row) <= col_index:
                        row.append(None)
                    row[col_index] = value

    def append_values(self, range_name: str, values) -> None:
        self.append_calls.append((range_name, [list(row) for row in values]))
        sheet = range_name.split("!", 1)[0]
        grid = self.sheets.setdefault(sheet, [])
        for row in values:
            grid.append(list(row))

    def list_sheet_titles(self) -> list[str]:
        return list(self.sheets.keys())

    def add_sheet(self, title: str) -> None:
        self.add_sheet_calls.append(title)
        self.sheets.setdefault(title, [])


def seeded(client: FakeSheetsClient, rows: list[list]) -> None:
    """Header plus data rows, written the way a real sync would have left them."""
    client.sheets[MASTER_SHEET] = [list(MASTER_HEADER), *[list(row) for row in rows]]


def bare_row(place_id: str, name: str, **overrides) -> list:
    """A full-width Master row: place_id, then every column blank except what's given."""
    values = {"business_name": name, **overrides}
    return [place_id] + [values.get(column) for column in COLUMNS]


# --- column addressing ---------------------------------------------------------------------


class ColumnLetterTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(_column_letter(0), "A")
        self.assertEqual(_column_letter(25), "Z")
        self.assertEqual(_column_letter(26), "AA")
        self.assertEqual(_column_letter(27), "AB")
        self.assertEqual(_column_letter(51), "AZ")

    def test_last_column_letter_matches_the_actual_header_width(self):
        # MASTER_HEADER is place_id + every view.COLUMNS entry; the last column letter must
        # address exactly that many columns, not one short (a truncated read) or one long
        # (a phantom column nothing ever fills).
        self.assertEqual(_column_letter(len(MASTER_HEADER) - 1), LAST_COLUMN_LETTER)


# --- reading ---------------------------------------------------------------------------------


class ReadMasterTests(unittest.TestCase):
    def test_an_empty_tab_reads_as_no_rows(self):
        client = FakeSheetsClient()
        self.assertEqual(read_master(client), {})

    def test_a_header_only_tab_reads_as_no_rows(self):
        client = FakeSheetsClient()
        seeded(client, [])
        self.assertEqual(read_master(client), {})

    def test_a_row_is_keyed_by_place_id(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "Cake Bee")])
        rows = read_master(client)
        self.assertIn("place-1", rows)
        self.assertEqual(rows["place-1"].values["business_name"], "Cake Bee")

    def test_row_number_is_the_real_1_indexed_sheet_row(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "First"), bare_row("place-2", "Second")])
        rows = read_master(client)
        # Row 1 is the header; data starts at row 2.
        self.assertEqual(rows["place-1"].row_number, 2)
        self.assertEqual(rows["place-2"].row_number, 3)

    def test_a_row_with_a_blank_place_id_is_skipped(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("", "No Join Key")])
        self.assertEqual(read_master(client), {})

    def test_a_reordered_header_is_still_read_correctly_by_name(self):
        # A human dragging a column in the sheet UI must not silently misattribute values --
        # read_master looks columns up by NAME in whatever header row actually exists.
        client = FakeSheetsClient()
        rest = [c for c in COLUMNS if c != "business_name"]
        shuffled_header = [PLACE_ID_COLUMN, "business_name", *rest]
        data_row = ["place-1", "Cake Bee", *([None] * len(rest))]
        client.sheets[MASTER_SHEET] = [shuffled_header, data_row]
        rows = read_master(client)
        self.assertEqual(rows["place-1"].values["business_name"], "Cake Bee")

    def test_operator_verdict_reads_blank_cells_as_none_not_empty_string(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "Cake Bee")])
        verdict = read_master(client)["place-1"].operator_verdict()
        self.assertIsNone(verdict.my_verdict)
        self.assertIsNone(verdict.notes)

    def test_operator_verdict_reads_a_typed_value(self):
        client = FakeSheetsClient()
        seeded(
            client,
            [bare_row("place-1", "Cake Bee", my_verdict="high", notes="called, interested")],
        )
        verdict = read_master(client)["place-1"].operator_verdict()
        self.assertEqual(verdict.my_verdict, "high")
        self.assertEqual(verdict.notes, "called, interested")


class ReadOperatorVerdictsTests(unittest.TestCase):
    def test_every_place_id_maps_to_its_verdict(self):
        client = FakeSheetsClient()
        seeded(
            client,
            [
                bare_row("place-1", "A", my_verdict="high"),
                bare_row("place-2", "B", my_verdict="skip"),
            ],
        )
        verdicts = read_operator_verdicts(client)
        self.assertEqual(verdicts["place-1"].my_verdict, "high")
        self.assertEqual(verdicts["place-2"].my_verdict, "skip")


# --- the invariant: operator columns are never written by a sync -----------------------------


class OperatorColumnsNeverWrittenTests(unittest.TestCase):
    """The single most important guarantee in this module."""

    def test_a_sync_never_addresses_any_operator_column_cell(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "Cake Bee", my_verdict="high", notes="do not touch")])

        sync_master(client, {"place-1": record("Cake Bee Updated", total_score=80)})

        written_columns = set()
        for updates in client.batch_calls:
            for range_name in updates:
                _, ref = range_name.split("!", 1)
                match = _CELL.match(ref)
                if not match:
                    continue
                col_index = _col_to_index(match.group(1))
                if 0 < col_index <= len(MASTER_HEADER):
                    written_columns.add(MASTER_HEADER[col_index])
        for column in OPERATOR_COLUMNS:
            with self.subTest(column=column):
                self.assertNotIn(column, written_columns)

    def test_the_operators_typed_value_survives_a_sync_byte_for_byte(self):
        client = FakeSheetsClient()
        seeded(
            client,
            [bare_row("place-1", "Cake Bee", my_verdict="high", notes="called them Tuesday")],
        )

        sync_master(client, {"place-1": record("Cake Bee", total_score=80)})

        verdict = read_master(client)["place-1"].operator_verdict()
        self.assertEqual(verdict.my_verdict, "high")
        self.assertEqual(verdict.notes, "called them Tuesday")

    def test_an_agent_column_on_the_same_row_is_updated(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "Cake Bee", my_verdict="high")])

        sync_master(client, {"place-1": record("Cake Bee", total_score=91)})

        self.assertEqual(read_master(client)["place-1"].values["total_score"], 91)


# --- new rows ----------------------------------------------------------------------------------


class AppendTests(unittest.TestCase):
    def test_an_unknown_place_id_is_appended_not_updated(self):
        client = FakeSheetsClient()
        seeded(client, [])

        result = sync_master(client, {"place-9": record("New Business")})

        self.assertEqual(result.appended_place_ids, ("place-9",))
        self.assertEqual(result.updated_place_ids, ())
        self.assertEqual(len(client.append_calls), 1)

    def test_an_appended_row_is_full_width_place_id_included(self):
        client = FakeSheetsClient()
        seeded(client, [])
        sync_master(client, {"place-9": record("New Business")})
        _, rows = client.append_calls[0]
        self.assertEqual(len(rows[0]), len(MASTER_HEADER))
        self.assertEqual(rows[0][0], "place-9")

    def test_a_mixed_batch_appends_new_and_updates_existing(self):
        client = FakeSheetsClient()
        seeded(client, [bare_row("place-1", "Existing")])

        result = sync_master(
            client,
            {"place-1": record("Existing", total_score=50), "place-2": record("Brand New")},
        )

        self.assertEqual(result.updated_place_ids, ("place-1",))
        self.assertEqual(result.appended_place_ids, ("place-2",))
        self.assertEqual(result.rows_written, 2)


# --- batching --------------------------------------------------------------------------------


class BatchingTests(unittest.TestCase):
    def test_updating_many_rows_issues_exactly_one_batch_call(self):
        client = FakeSheetsClient()
        seeded(
            client,
            [bare_row(f"place-{i}", f"Business {i}") for i in range(10)],
        )

        sync_master(
            client, {f"place-{i}": record(f"Business {i}", total_score=i) for i in range(10)}
        )

        self.assertEqual(len(client.batch_calls), 1)

    def test_a_sync_with_nothing_to_update_issues_no_batch_call(self):
        client = FakeSheetsClient()
        seeded(client, [])
        sync_master(client, {})
        self.assertEqual(client.batch_calls, [])

    def test_a_pure_append_batch_issues_no_batch_update_call_at_all(self):
        client = FakeSheetsClient()
        seeded(client, [])
        sync_master(client, {"place-1": record("Only New")})
        self.assertEqual(client.batch_calls, [])
        self.assertEqual(len(client.append_calls), 1)


# --- header ------------------------------------------------------------------------------------


class HeaderTests(unittest.TestCase):
    def test_an_empty_tab_gets_the_header_written(self):
        client = FakeSheetsClient()
        ensure_master_header(client)
        self.assertEqual(client.sheets[MASTER_SHEET][0], list(MASTER_HEADER))

    def test_an_existing_header_is_never_overwritten(self):
        client = FakeSheetsClient()
        client.sheets[MASTER_SHEET] = [["place_id", "some_custom_first_column"]]
        ensure_master_header(client)
        first_row = client.sheets[MASTER_SHEET][0]
        self.assertEqual(first_row, ["place_id", "some_custom_first_column"])
        self.assertEqual(client.batch_calls, [])

    def test_a_spreadsheet_with_no_master_tab_at_all_gets_one_created(self):
        # The real shape of a spreadsheet a human just shared for the first time: only
        # Google's default `Sheet1`, no `Master` tab yet at all.
        client = FakeSheetsClient()
        client.sheets["Sheet1"] = []
        ensure_master_header(client)
        self.assertIn(MASTER_SHEET, client.add_sheet_calls)
        self.assertEqual(client.sheets[MASTER_SHEET][0], list(MASTER_HEADER))

    def test_sync_master_creates_the_tab_itself_on_a_cold_start(self):
        # sync_master must not require a caller to remember ensure_master_header first.
        client = FakeSheetsClient()
        client.sheets["Sheet1"] = []
        result = sync_master(client, {"place-1": record("Cake Bee")})
        self.assertEqual(result.appended_place_ids, ("place-1",))
        self.assertEqual(client.sheets[MASTER_SHEET][0], list(MASTER_HEADER))


# --- snapshots -----------------------------------------------------------------------------


class SnapshotTests(unittest.TestCase):
    def test_a_new_tab_is_created_and_titled_by_date(self):
        client = FakeSheetsClient()
        title = write_snapshot(client, date(2026, 9, 6), [("place-1", record("Cake Bee"))])
        self.assertEqual(title, "2026-09-06")
        self.assertIn("2026-09-06", client.add_sheet_calls)

    def test_snapshot_rows_are_place_id_prefixed_and_full_width(self):
        client = FakeSheetsClient()
        write_snapshot(client, date(2026, 9, 6), [("place-1", record("Cake Bee"))])
        grid = client.sheets["2026-09-06"]
        self.assertEqual(grid[0], list(MASTER_HEADER))
        self.assertEqual(grid[1][0], "place-1")
        self.assertEqual(len(grid[1]), len(MASTER_HEADER))

    def test_a_second_snapshot_the_same_day_reuses_the_tab(self):
        client = FakeSheetsClient()
        write_snapshot(client, date(2026, 9, 6), [("place-1", record("First"))])
        write_snapshot(
            client,
            date(2026, 9, 6),
            [("place-1", record("First")), ("place-2", record("Second"))],
        )
        self.assertEqual(client.add_sheet_calls, ["2026-09-06"])

    def test_an_empty_cohort_still_writes_a_header(self):
        client = FakeSheetsClient()
        write_snapshot(client, date(2026, 9, 6), [])
        self.assertEqual(client.sheets["2026-09-06"], [list(MASTER_HEADER)])


# --- the real client, shape-tested against a recording stub ---------------------------------


class RecordingValues:
    def __init__(self, calls: list, get_response: dict) -> None:
        self._calls = calls
        self._get_response = get_response

    def get(self, **kwargs):
        self._calls.append(("values.get", kwargs))
        return _Executable(self._get_response)

    def batchUpdate(self, **kwargs):
        self._calls.append(("values.batchUpdate", kwargs))
        return _Executable({})

    def append(self, **kwargs):
        self._calls.append(("values.append", kwargs))
        return _Executable({})


class RecordingSpreadsheets:
    def __init__(self, calls: list, get_response: dict) -> None:
        self._calls = calls
        self._values = RecordingValues(calls, get_response)

    def values(self):
        return self._values

    def get(self, **kwargs):
        self._calls.append(("spreadsheets.get", kwargs))
        return _Executable({"sheets": [{"properties": {"title": "Master"}}]})

    def batchUpdate(self, **kwargs):
        self._calls.append(("spreadsheets.batchUpdate", kwargs))
        return _Executable({})

    def create(self, **kwargs):  # pragma: no cover - must never be called, see test below
        raise AssertionError("GoogleSheetsClient must never call spreadsheets().create()")


class _Executable:
    def __init__(self, response: dict) -> None:
        self._response = response

    def execute(self):
        return self._response


class RecordingService:
    def __init__(self, calls: list, get_response: dict | None = None) -> None:
        self._spreadsheets = RecordingSpreadsheets(calls, get_response or {})

    def spreadsheets(self):
        return self._spreadsheets


class GoogleSheetsClientShapeTests(unittest.TestCase):
    def test_get_values_requests_the_given_range_on_the_given_spreadsheet(self):
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.get_values("Master!A1:Z")
        self.assertEqual(calls[0][0], "values.get")
        self.assertEqual(calls[0][1]["spreadsheetId"], "sheet-123")
        self.assertEqual(calls[0][1]["range"], "Master!A1:Z")

    def test_batch_update_uses_user_entered_input_and_every_range(self):
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.batch_update_values({"Master!C5": [["x"]], "Master!D5": [["y"]]})
        name, kwargs = calls[0]
        self.assertEqual(name, "values.batchUpdate")
        body = kwargs["body"]
        self.assertEqual(body["valueInputOption"], "USER_ENTERED")
        ranges = {entry["range"] for entry in body["data"]}
        self.assertEqual(ranges, {"Master!C5", "Master!D5"})

    def test_batch_update_with_nothing_to_write_makes_no_call(self):
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.batch_update_values({})
        self.assertEqual(calls, [])

    def test_append_uses_insert_rows(self):
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.append_values("Master!A:A", [["place-1", "Cake Bee"]])
        name, kwargs = calls[0]
        self.assertEqual(name, "values.append")
        self.assertEqual(kwargs["insertDataOption"], "INSERT_ROWS")

    def test_add_sheet_issues_an_addsheet_batch_request(self):
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.add_sheet("2026-09-06")
        name, kwargs = calls[0]
        self.assertEqual(name, "spreadsheets.batchUpdate")
        self.assertEqual(
            kwargs["body"]["requests"][0]["addSheet"]["properties"]["title"], "2026-09-06"
        )

    def test_list_sheet_titles_reads_properties_from_metadata(self):
        calls: list = []
        service = RecordingService(calls)
        client = GoogleSheetsClient(service, "sheet-123")
        titles = client.list_sheet_titles()
        self.assertEqual(titles, ["Master"])

    def test_never_calls_spreadsheets_create(self):
        # The one architectural fact this session proved live: a bare service account
        # cannot own a new file. GoogleSheetsClient must never attempt to find out again.
        calls: list = []
        client = GoogleSheetsClient(RecordingService(calls), "sheet-123")
        client.get_values("Master!A1:Z")
        client.batch_update_values({"Master!A1": [["x"]]})
        client.append_values("Master!A:A", [["x"]])
        client.add_sheet("t")
        client.list_sheet_titles()
        # No AssertionError from RecordingSpreadsheets.create means it was never reached.


if __name__ == "__main__":
    unittest.main()
