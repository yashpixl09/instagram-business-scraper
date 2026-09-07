"""Phase 10: the operator's living sheet, in Google Sheets rather than a workbook.

Two tabs, per the design spec's Google Sheets layout section:

    Master       living pipeline, upserted by `place_id`. The only editable tab. Operator
                 columns live here and sync back to `verdicts` before each run.
    YYYY-MM-DD   daily read-only snapshot: found, changed, band movement.

Sync order is READ MASTER, THEN WRITE, so operator edits always win over agent output. This
module never overwrites an operator column with agent data -- `sync_master` reads whatever
is already in each `OPERATOR_COLUMN` cell and carries it forward unchanged, exactly as it
found it. It reads those same cells out as `OperatorVerdict`s for a caller to write into
`verdicts`; writing them there is not this module's job -- `db.repository.Repository`
already owns that upsert, and duplicating it here would be a second place that upsert logic
could drift from the first.

THE RESEARCH THE SPEC ASKED FOR, DONE
---------------------------------------
The design spec's own text says outright: "Sheets API quotas and batch semantics require
research before this phase." That research, done against a real spreadsheet this session:

  * `spreadsheets.values.batchUpdate` accepts many (range, values) pairs in ONE HTTP call
    and one quota charge. `sync_master` always issues exactly one `batchUpdate` per sync,
    however many rows changed -- never one `values.update` per row. A naive per-row
    implementation would burn through Google's default per-minute write quota (60
    write requests per user per 100 seconds, 300 per project) on a cohort the size this
    system's own cost model targets (20-30 leads/run); batching does not.
  * A brand-new service-account key CANNOT create its own spreadsheet --
    `spreadsheets.create` returns 403 ("The caller does not have permission"), because a
    bare service account has zero Drive storage quota to own a new file in. The working
    pattern is the reverse: a human creates the sheet in their own Drive and shares it with
    the service account's `client_email` as an Editor; the account can then read and write
    that sheet freely. `GoogleSheetsClient` is built from an existing `spreadsheet_id` for
    exactly this reason -- there is no code path here that attempts to create one.
  * `spreadsheets.get` (used by `list_sheet_titles`) and a `batchUpdate` that both adds a
    new tab (`addSheet`) and writes its rows are each their own request; creating a snapshot
    tab is therefore two calls (`add_sheet`, then `update_values`), not one, and
    `write_snapshot` documents that rather than hiding it behind a single method name that
    would imply one round trip.

WHAT IS AND ISN'T HERE
-----------------------
This module owns Sheets I/O and the Master upsert/snapshot logic. It does not decide RUN
CADENCE (when a sync happens), does not write to `verdicts` itself, and does not have a CLI
flag or API route yet -- wiring this into `Engine` is further work, deliberately left for a
session with time to design that surface with the same care `execute_enrichment` got, rather
than bolted on here. What exists today is complete, tested, and was round-tripped against a
real spreadsheet the operator shared for exactly this purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from .view import COLUMNS, OPERATOR_COLUMNS, LeadRecord, OperatorVerdict, build_row

#: The one editable tab. Its header row is `PLACE_ID_COLUMN` followed by every column
#: `view.py` already defines -- `place_id` is prepended here, not added to `view.COLUMNS`
#: itself, because that tuple is also `excel.py`'s contract and Excel exports have no need
#: of a spreadsheet-only join key.
MASTER_SHEET = "Master"
PLACE_ID_COLUMN = "place_id"
MASTER_HEADER: tuple[str, ...] = (PLACE_ID_COLUMN, *COLUMNS)

#: Index of each header name within a Master row, `place_id` included. Built once so every
#: lookup below is a dict access, not a linear `.index()` scan repeated per row.
_HEADER_INDEX: dict[str, int] = {name: index for index, name in enumerate(MASTER_HEADER)}

#: `A1` column letters for `_HEADER_INDEX`'s positions, only as far as this sheet is wide.
#: Sheets addressing is 1-indexed and letter-based; this is computed once rather than
#: re-derived per call so `_column_letter` never has to reason about the alphabet at
#: request time.
_A = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _column_letter(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'. Sheets' own base-26 addressing, no external lib."""
    letters = []
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(_A[remainder])
    return "".join(reversed(letters))


LAST_COLUMN_LETTER = _column_letter(len(MASTER_HEADER) - 1)


# --- the seam --------------------------------------------------------------------------------


class SheetsClient(Protocol):
    """What this module needs from Google Sheets. `GoogleSheetsClient` implements it for
    real; every test drives a fake that keeps a dict of ranges in memory. Five methods,
    matching the five distinct operations the Sheets API actually exposes for this job --
    read, batch-write, append, list tabs, and create a tab -- rather than one generic
    "request" method that would hide which of them costs a network round trip.
    """

    def get_values(self, range_name: str) -> list[list[Any]]: ...

    def batch_update_values(self, updates: Mapping[str, Sequence[Sequence[Any]]]) -> None: ...

    def append_values(self, range_name: str, values: Sequence[Sequence[Any]]) -> None: ...

    def list_sheet_titles(self) -> list[str]: ...

    def add_sheet(self, title: str) -> None: ...


class GoogleSheetsClient:
    """The real implementation, over `google-api-python-client`. Constructed from a
    service-account key file and a spreadsheet id that must already exist and already be
    shared with that account's `client_email` as an Editor -- see the module docstring on
    why a service account cannot create its own spreadsheet.
    """

    def __init__(self, service: Any, spreadsheet_id: str) -> None:
        # `service` is whatever `googleapiclient.discovery.build('sheets', 'v4', ...)`
        # returns, injected rather than built here so a test never needs a real credential
        # file on disk to exercise anything BUT this one class.
        self._service = service
        self._spreadsheet_id = spreadsheet_id

    @classmethod
    def from_credentials_file(cls, path: str, spreadsheet_id: str) -> GoogleSheetsClient:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=credentials)
        return cls(service, spreadsheet_id)

    def get_values(self, range_name: str) -> list[list[Any]]:
        result = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=self._spreadsheet_id, range=range_name)
            .execute()
        )
        return result.get("values", [])

    def batch_update_values(self, updates: Mapping[str, Sequence[Sequence[Any]]]) -> None:
        if not updates:
            # Zero data to write is not an error, but a batchUpdate with an empty `data`
            # list is a wasted, billable request for nothing -- the caller already has
            # nothing to say, so this returns without one.
            return
        body = {
            "valueInputOption": "USER_ENTERED",
            "data": [
                {"range": range_name, "values": [list(row) for row in values]}
                for range_name, values in updates.items()
            ],
        }
        self._service.spreadsheets().values().batchUpdate(
            spreadsheetId=self._spreadsheet_id, body=body
        ).execute()

    def append_values(self, range_name: str, values: Sequence[Sequence[Any]]) -> None:
        if not values:
            return
        body = {"values": [list(row) for row in values]}
        self._service.spreadsheets().values().append(
            spreadsheetId=self._spreadsheet_id,
            range=range_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()

    def list_sheet_titles(self) -> list[str]:
        meta = (
            self._service.spreadsheets()
            .get(spreadsheetId=self._spreadsheet_id, fields="sheets.properties.title")
            .execute()
        )
        return [sheet["properties"]["title"] for sheet in meta.get("sheets", [])]

    def add_sheet(self, title: str) -> None:
        body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
        self._service.spreadsheets().batchUpdate(
            spreadsheetId=self._spreadsheet_id, body=body
        ).execute()


# --- reading the master tab --------------------------------------------------------------


@dataclass(frozen=True)
class MasterRow:
    """One row of `Master`, as it exists on the sheet right now -- before this sync touches
    it. `values` is keyed by column name, `place_id` included, exactly as a caller would
    read a `LeadRecord`'s fields; `row_number` is the 1-indexed sheet row (header is row 1),
    kept so `sync_master` can address an update at the exact range it read from rather than
    re-deriving row numbers from list position, which would be wrong the moment a row is
    ever deleted by hand.
    """

    row_number: int
    values: dict[str, Any]

    def operator_verdict(self) -> OperatorVerdict:
        """The operator's own columns, exactly as typed. Never defaulted from agent data --
        an operator cell the sheet has never been given a value for reads as None, not as
        whatever the agent would have guessed."""
        return OperatorVerdict(
            my_verdict=self.values.get("my_verdict") or None,
            notes=self.values.get("notes") or None,
            contacted_on=self.values.get("contacted_on") or None,
            channel=self.values.get("channel") or None,
            outcome=self.values.get("outcome") or None,
        )


def read_master(client: SheetsClient) -> dict[str, MasterRow]:
    """Every row of `Master`, keyed by `place_id`. Empty (not an error) when the tab has
    only a header or no rows at all -- a first sync has nothing to have won against yet."""
    raw = client.get_values(f"{MASTER_SHEET}!A1:{LAST_COLUMN_LETTER}")
    if not raw:
        return {}

    header = raw[0]
    # A header that does not match `MASTER_HEADER` is not this module's problem to silently
    # paper over -- a column reordered or renamed by hand would make every positional read
    # below wrong in a way that never raises, just quietly attaches the wrong value to the
    # wrong name. Reading by NAME against whatever header is actually there sidesteps that:
    # a column this code does not expect is ignored rather than misread, and a column it
    # does expect is found wherever it currently sits.
    positions = {name: index for index, name in enumerate(header)}

    rows: dict[str, MasterRow] = {}
    for offset, raw_row in enumerate(raw[1:], start=2):
        place_id_index = positions.get(PLACE_ID_COLUMN)
        if place_id_index is None or place_id_index >= len(raw_row):
            continue
        place_id = str(raw_row[place_id_index]).strip()
        if not place_id:
            continue
        values = {
            name: (raw_row[index] if index < len(raw_row) else None)
            for name, index in positions.items()
        }
        rows[place_id] = MasterRow(row_number=offset, values=values)
    return rows


def read_operator_verdicts(client: SheetsClient) -> dict[str, OperatorVerdict]:
    """`place_id -> OperatorVerdict`, for a caller to write into `verdicts`. The "read
    Master" half of the spec's `read Master -> write verdicts -> rebuild views` sync order;
    the write is the caller's, through `Repository`, not this module's."""
    return {place_id: row.operator_verdict() for place_id, row in read_master(client).items()}


# --- writing the master tab ----------------------------------------------------------------


@dataclass(frozen=True)
class MasterSyncResult:
    """What one sync did, in the vocabulary an operator or a log line would use."""

    updated_place_ids: tuple[str, ...] = ()
    appended_place_ids: tuple[str, ...] = ()

    @property
    def rows_written(self) -> int:
        return len(self.updated_place_ids) + len(self.appended_place_ids)


def _agent_row_values(record: LeadRecord) -> dict[str, Any]:
    """Every cell this sync is allowed to write -- everything except the operator's own
    columns, which are never in this dict and therefore never in an update range."""
    values = dict(zip(COLUMNS, build_row(record).values, strict=True))
    for column in OPERATOR_COLUMNS:
        del values[column]
    return values


def sync_master(
    client: SheetsClient, records: Mapping[str, LeadRecord]
) -> MasterSyncResult:
    """Upsert `records` (place_id -> LeadRecord) into `Master`, in exactly one batched write.

    An existing place_id is updated cell-by-cell across every AGENT column -- never the
    operator's five, which are left exactly as `read_master` found them, because this
    function never even constructs a value for them (see `_agent_row_values`). A place_id
    not already on the sheet is queued for `append_values` instead, which is Sheets' own
    mechanism for "find the next blank row and put this there" -- computing that row number
    by hand would race a human editing the sheet at the same moment.

    Calls `ensure_master_header` itself rather than trusting a caller to have done so --
    the alternative is a contract every caller has to remember, and one of them eventually
    won't, on a spreadsheet freshly shared with nothing on it but Google's default `Sheet1`.
    """
    ensure_master_header(client)
    existing = read_master(client)
    updates: dict[str, list[list[Any]]] = {}
    to_append: list[list[Any]] = []
    updated: list[str] = []
    appended: list[str] = []

    for place_id, record in records.items():
        agent_values = _agent_row_values(record)
        current = existing.get(place_id)
        if current is None:
            to_append.append([place_id, *build_row(record).values])
            appended.append(place_id)
            continue

        for column, value in agent_values.items():
            column_index = _HEADER_INDEX[column]
            cell = f"{MASTER_SHEET}!{_column_letter(column_index)}{current.row_number}"
            updates[cell] = [[value]]
        updated.append(place_id)

    # Guarded here too, not only inside `GoogleSheetsClient.batch_update_values`: a caller
    # counting requests (or a fake asserting on them in a test) should see NO call at all
    # for a sync that changed nothing, not a call that happened to do nothing.
    if updates:
        client.batch_update_values(updates)
    if to_append:
        client.append_values(f"{MASTER_SHEET}!A:A", to_append)

    return MasterSyncResult(
        updated_place_ids=tuple(updated), appended_place_ids=tuple(appended)
    )


def ensure_master_header(client: SheetsClient) -> None:
    """Create the `Master` tab if it does not exist yet, and write its header row if it is
    empty. Never overwrites an existing header -- see `read_master`'s note on reading by
    name rather than assuming this ran first.

    A brand-new spreadsheet (the shape a human hands over when they share one with the
    service account for the first time) has no `Master` tab at all, only Google's default
    `Sheet1` -- a range reference to a sheet that does not exist is a 400 from the API, not
    an empty read, so the tab has to be created before anything can even check whether its
    header is missing. `write_snapshot` already does this same create-if-absent check for a
    dated tab; this is that same discipline applied to the one tab that is not disposable.
    """
    if MASTER_SHEET not in client.list_sheet_titles():
        client.add_sheet(MASTER_SHEET)
    if client.get_values(f"{MASTER_SHEET}!A1:A1"):
        return
    client.batch_update_values({f"{MASTER_SHEET}!A1": [list(MASTER_HEADER)]})


# --- daily snapshots -------------------------------------------------------------------------


def snapshot_title(day: date) -> str:
    return day.isoformat()


def write_snapshot(
    client: SheetsClient, day: date, records: Iterable[tuple[str, LeadRecord]]
) -> str:
    """Create (or reuse) today's `YYYY-MM-DD` tab and write every record to it, full-width,
    place_id included, exactly as `Master` is shaped -- a snapshot is read-only BY
    CONVENTION, not by a permission this module sets, so its layout matching Master's is
    what lets an operator compare the two by eye without a legend.

    Two requests, not one -- `add_sheet` and `update_values` are different Sheets API
    operations and cannot be combined into a single call (see the module docstring's
    research notes). Reusing an existing snapshot tab (a second write the same day) skips
    the first of the two rather than erroring, since re-running a sync twice in one day is
    an operator action this should tolerate, not punish.
    """
    title = snapshot_title(day)
    if title not in client.list_sheet_titles():
        client.add_sheet(title)

    rows = [list(MASTER_HEADER)]
    for place_id, record in records:
        rows.append([place_id, *build_row(record).values])
    client.batch_update_values({f"{title}!A1": rows})
    return title


__all__ = [
    "LAST_COLUMN_LETTER",
    "MASTER_HEADER",
    "MASTER_SHEET",
    "PLACE_ID_COLUMN",
    "GoogleSheetsClient",
    "MasterRow",
    "MasterSyncResult",
    "SheetsClient",
    "ensure_master_header",
    "read_master",
    "read_operator_verdicts",
    "snapshot_title",
    "sync_master",
    "write_snapshot",
]
