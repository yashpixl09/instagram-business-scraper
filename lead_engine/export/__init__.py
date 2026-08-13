"""The sales sheet: the one artefact the operator actually works from.

Two modules, split along the line that decides what can be tested cheaply:

  `view`   -- the layout. Which columns exist, in which order, and which row comes first.
              Pure: standard library plus the pure core, no database, no openpyxl. The
              whole sheet's shape is therefore testable in milliseconds with synthetic
              leads, which matters because layout bugs are silent -- a column inserted in
              one place and not another moves every value after it one cell left.
  `excel`  -- the workbook. openpyxl, the query against the `lead_bands` view, cell
              comments carrying each number's source URL, and an atomic write.

`excel` is imported lazily, so `from lead_engine.export import view` -- or any consumer
that only needs the column list, such as the API layer describing the sheet's shape --
stays a pure import and does not require openpyxl to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .view import (
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
    sort_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from .excel import (
        EXPORT_QUERY,
        build_workbook,
        export,
        export_records,
        fetch_records,
        record_from_row,
        write_workbook,
    )

_EXCEL_NAMES = {
    "EXPORT_QUERY",
    "build_workbook",
    "export",
    "export_records",
    "fetch_records",
    "record_from_row",
    "write_workbook",
}

__all__ = [
    "COLUMNS",
    "COLUMN_GROUPS",
    "COLUMN_INDEX",
    "EVIDENCE_COLUMNS",
    "EXPORT_QUERY",
    "GOOGLE_EVIDENCE_COLUMNS",
    "INSTAGRAM_EVIDENCE_COLUMNS",
    "OPERATOR_COLUMNS",
    "Cell",
    "ContactPerson",
    "ExportRow",
    "LeadRecord",
    "OperatorVerdict",
    "build_row",
    "build_rows",
    "build_workbook",
    "evidence_sources",
    "export",
    "export_records",
    "fetch_records",
    "instagram_profile_url",
    "niche_label",
    "record_from_row",
    "record_from_scored_lead",
    "sort_key",
    "write_workbook",
]


def __getattr__(name: str):
    if name in _EXCEL_NAMES:
        from . import excel

        return getattr(excel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
