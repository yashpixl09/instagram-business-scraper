"""The sheet's layout, as data. Pure: no database, no file format, no openpyxl.

This module owns two decisions and nothing else -- *which columns exist, in which order*
and *which row comes first*. Everything about workbooks (fills, comments, freeze panes,
atomic writes) lives in `excel.py`, and everything about where the values came from lives
in the query that feeds this one.

The split is what makes the layout testable. A test for "the operator's columns are the
last five" or "a lead with no phone still produces a full-width row" needs synthetic
`ScoredLead`s and nothing else -- no Postgres, no temp directory, no openpyxl. Layout bugs
are the ones the operator notices (a shifted column silently moves every value one to the
left), so they are the ones worth being able to test exhaustively and instantly.

Emitting empty columns is deliberate
------------------------------------
Half of these columns cannot be populated yet: nothing writes `ai_summary`, contacts are a
later phase, `runs_ads` needs an ad-library lookup that does not exist. They are emitted
anyway, blank. The operator marks up this file by hand and a later phase reads the operator
columns back by name, so a sheet that grows a column every fortnight is a sheet whose
older copies stop being readable. The shape is a contract; only the fill rate changes.

Blank means blank
-----------------
A missing value is `None`, never `""`. In a spreadsheet those are different things: an
empty string is a non-empty cell that defeats COUNTBLANK, "go to special > blanks", and
every filter the operator would use to find the rows still needing work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..models import Evidence
from ..niches import NICHE_PROFILES
from ..scoring import ScoredLead

# The layout. Groups are ordered by how the operator reads a row left to right: who they
# are, where they are, how to reach them, who to ask for, what the evidence says, what the
# agent concluded, what could be automated, and -- last, because they are written rather
# than read -- what the operator found.
COLUMN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", ("business_name", "niche", "business_type")),
    ("geography", ("country", "state", "city", "search_area", "address", "lat", "lng")),
    ("contact", ("phone", "email", "website", "instagram_handle", "facebook_url")),
    (
        "people",
        ("contact_name", "contact_role", "contact_phone", "contact_email", "contact_source"),
    ),
    ("evidence", ("reviews", "rating", "followers", "engagement_rate", "runs_ads", "peak_hours")),
    (
        "agent",
        (
            "total_score",
            "audience_band",
            "banding_method",
            "signals",
            "ai_summary",
            "website_pitch",
        ),
    ),
    ("automation", ("automation_opportunities", "automation_pitch")),
    ("operator", ("my_verdict", "notes", "contacted_on", "channel", "outcome")),
)

COLUMNS: tuple[str, ...] = tuple(name for _, names in COLUMN_GROUPS for name in names)
COLUMN_INDEX: dict[str, int] = {name: index for index, name in enumerate(COLUMNS)}
GROUP_OF: dict[str, str] = {name: group for group, names in COLUMN_GROUPS for name in names}

# Every number in this group is spoken aloud to a business owner ("you have 340 reviews and
# they average 4.6"), so every one of them carries its source URL as a cell comment. See
# `excel.py`.
EVIDENCE_COLUMNS: tuple[str, ...] = dict(COLUMN_GROUPS)["evidence"]

# The only columns the operator writes, and the only ones a later phase reads back.
OPERATOR_COLUMNS: tuple[str, ...] = dict(COLUMN_GROUPS)["operator"]

# Which evidence cell is verifiable at which URL. `runs_ads` is in neither list on purpose:
# it comes from an ad library keyed on a page id this system does not hold, so there is no
# URL that shows it. A cell with no honest source gets no comment rather than a comment
# pointing at a page where the claim cannot be checked -- which is worse than none, because
# the operator would click it, fail to find the number, and stop trusting the others.
GOOGLE_EVIDENCE_COLUMNS: tuple[str, ...] = ("reviews", "rating", "peak_hours")
INSTAGRAM_EVIDENCE_COLUMNS: tuple[str, ...] = ("followers", "engagement_rate")


@dataclass(frozen=True)
class Cell:
    """One cell: what it says, and where that came from.

    `source` is provenance, not decoration. It becomes the comment on evidence cells so a
    number can be checked in one click before it is repeated to a business owner.
    """

    value: Any = None
    source: str | None = None


@dataclass(frozen=True)
class ExportRow:
    """One sheet row, always exactly `len(COLUMNS)` cells wide."""

    cells: tuple[Cell, ...]

    @property
    def values(self) -> tuple[Any, ...]:
        return tuple(cell.value for cell in self.cells)

    def cell(self, column: str) -> Cell:
        return self.cells[COLUMN_INDEX[column]]

    def value(self, column: str) -> Any:
        return self.cell(column).value

    def source(self, column: str) -> str | None:
        return self.cell(column).source


@dataclass(frozen=True)
class ContactPerson:
    """A named human at the business. Every field optional: a review reply gives a name and
    nothing else, an Instagram bio gives an email and nothing else."""

    name: str | None = None
    role: str | None = None
    phone: str | None = None
    email: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class OperatorVerdict:
    """What the operator wrote back. Empty on a freshly exported sheet."""

    my_verdict: str | None = None
    notes: str | None = None
    contacted_on: date | str | None = None
    channel: str | None = None
    outcome: str | None = None


@dataclass(frozen=True)
class LeadRecord:
    """One already-fetched lead, in the shape the sheet needs.

    This is the seam between "how the data is stored" and "how the sheet looks". The
    database query in `excel.py` builds these; so does `record_from_scored_lead` for
    callers holding a freshly scored lead that has not been persisted. Neither producer
    knows anything about column order, and this module knows nothing about either producer.

    Only `business_name` is required. A lead the pipeline failed on -- no score, no
    enrichment, no contact -- is precisely the row worth looking at, so it must be
    representable.
    """

    business_name: str
    niche: str | None = None
    business_type: str | None = None

    country: str | None = None
    state: str | None = None
    city: str | None = None
    search_area: str | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None

    phone: str | None = None
    email: str | None = None
    website: str | None = None
    instagram_handle: str | None = None
    facebook_url: str | None = None

    contact: ContactPerson | None = None

    evidence: Evidence | None = None
    # Google's popular-times summary as prose ("Busiest Fri 7-9pm"). `Evidence` carries the
    # density as a number for scoring; the operator needs the hours, not the mean.
    peak_hours: str | None = None
    # column name -> URL where that value can be verified. Explicit, because provenance is
    # per source: reviews come from a Maps listing and followers from an Instagram profile,
    # and one URL for the whole row would be a lie about at least one of them.
    evidence_sources: Mapping[str, str] = field(default_factory=dict)

    total_score: int | None = None
    audience_band: str | None = None
    banding_method: str | None = None
    signals: Sequence[str] = ()
    ai_summary: str | None = None
    website_pitch: str | None = None

    automation_opportunities: Sequence[str] = ()
    automation_pitch: str | None = None

    verdict: OperatorVerdict | None = None


def niche_label(niche_id: str | None) -> str | None:
    """The registry's human label for a niche id, or the id itself.

    Falling back to the raw id rather than raising matters: a business stored under a niche
    that has since been renamed or retired still has to appear in the sheet. An export that
    aborts on one stale row is worse than one that prints `beauty_salon_v1`.
    """
    if not niche_id:
        return None
    profile = NICHE_PROFILES.get(niche_id)
    return profile.label if profile else niche_id


def instagram_profile_url(handle: str | None) -> str | None:
    """`@blush.salon` -> `https://www.instagram.com/blush.salon/`.

    Not a guess: this is exactly the page a follower count was read from, which is what
    makes it a legitimate source for that cell. A value that is already a URL is normalised
    rather than rebuilt, since some callers store the profile link in the handle column.

    The host check has to cover the `www.` form. It once did not, so a scheme-less
    `www.instagram.com/blush.salon` was treated as a bare handle and became
    `https://www.instagram.com/www.instagram.com/blush.salon/`. That value goes into the cell
    comment on `followers`, so the operator right-clicks a number for its receipt and lands
    on a 404 -- which this module's own docstring calls worse than no comment at all, because
    they stop trusting the other receipts too.
    """
    if not handle:
        return None
    cleaned = handle.strip().lstrip("@").strip("/")
    if not cleaned:
        return None

    # Strip any scheme and leading host before deciding, so every spelling of "this is
    # already an Instagram link" takes the same path: with scheme or without, www. or not.
    without_scheme = cleaned.split("://", 1)[-1]
    host, _, path = without_scheme.partition("/")
    if host.lower() in ("instagram.com", "www.instagram.com"):
        profile = path.strip("/")
        # A host with no path is not a profile; there is nothing to link the number to.
        return f"https://www.instagram.com/{profile}/" if profile else None
    if "://" in cleaned:
        # Some other site's URL. Not ours to rewrite into an Instagram link.
        return handle.strip()
    return f"https://www.instagram.com/{cleaned}/"


def evidence_sources(
    *,
    google_url: str | None = None,
    instagram_url: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the column -> URL map from the two pages evidence is actually read off.

    `extra` wins over both, because a caller that knows the precise URL for one cell knows
    better than this function's grouping.
    """
    sources: dict[str, str] = {}
    if google_url:
        sources.update({column: google_url for column in GOOGLE_EVIDENCE_COLUMNS})
    if instagram_url:
        sources.update({column: instagram_url for column in INSTAGRAM_EVIDENCE_COLUMNS})
    for column, url in (extra or {}).items():
        if url:
            sources[column] = url
    return sources


def record_from_scored_lead(
    scored: ScoredLead,
    *,
    niche: str | None = None,
    country: str | None = None,
    state: str | None = None,
    search_area: str | None = None,
    email: str | None = None,
    instagram_handle: str | None = None,
    facebook_url: str | None = None,
    contact: ContactPerson | None = None,
    evidence: Evidence | None = None,
    peak_hours: str | None = None,
    extra_sources: Mapping[str, str] | None = None,
    audience_band: str | None = None,
    banding_method: str | None = None,
    automation_opportunities: Sequence[str] = (),
    automation_pitch: str | None = None,
    verdict: OperatorVerdict | None = None,
) -> LeadRecord:
    """Turn a scored-but-unpersisted lead into a sheet record.

    `Lead` knows nothing about country, state, search area, or social handles -- those are
    supplied by the run that found it -- so they arrive as keywords. Two mappings are worth
    stating outright rather than leaving to be discovered:

    * `outreach_message` fills `website_pitch`. It is the first offer, the one the score's
      website gap justifies; the automation pitch is written later against detected
      opportunities and is passed in separately.
    * `audience_band` and `banding_method` stay empty unless given. A band is a percentile
      against the cohort, computed in SQL over every stored business (see the `lead_bands`
      view); one lead in isolation cannot know its own band, and inventing one here would
      put a confident, wrong word in front of the operator.
    """
    lead = scored.lead
    niche_id = niche or (lead.matched_niches[0] if lead.matched_niches else None)
    return LeadRecord(
        business_name=lead.name,
        niche=niche_label(niche_id),
        business_type=lead.category,
        country=country,
        state=state,
        city=lead.city,
        search_area=search_area,
        address=lead.address,
        lat=lead.latitude,
        lng=lead.longitude,
        phone=lead.phone,
        email=email,
        website=lead.website,
        instagram_handle=instagram_handle,
        facebook_url=facebook_url,
        contact=contact,
        evidence=evidence,
        peak_hours=peak_hours,
        evidence_sources=evidence_sources(
            google_url=lead.source_url,
            instagram_url=instagram_profile_url(instagram_handle),
            extra=extra_sources,
        ),
        total_score=scored.score.total,
        audience_band=audience_band,
        banding_method=banding_method,
        signals=tuple(scored.score.signals),
        ai_summary=scored.ai_summary,
        website_pitch=scored.outreach_message,
        automation_opportunities=tuple(automation_opportunities),
        automation_pitch=automation_pitch,
        verdict=verdict,
    )


def sort_key(record: LeadRecord) -> tuple[Any, ...]:
    """Search area ascending, then total score descending.

    Not score alone, which is the obvious sort and the wrong one. The operator walks into
    these businesses. A globally ranked list sends them across a 40km city in score order;
    grouping the day into one neighbourhood is worth more than visiting the 3rd-best lead
    before the 5th. Within an area the ranking is the usual one, so the best door in
    Indiranagar is still the first door in Indiranagar.

    Two tie-break rules, both deliberate:

    * Rows with no area sort last. An unplaceable lead cannot be walked to, so it belongs
      at the bottom rather than wedged into whichever neighbourhood sorts before "".
    * Rows with no score sort last within their area, below a score of zero. "Not scored"
      is missing information, not a bad result, and it must not displace a real ranking.

    Area is compared case-folded so that "Indiranagar" and "indiranagar" stay one block
    instead of two separated by half the city.
    """
    area = (record.search_area or "").strip()
    total = record.total_score
    return (
        0 if area else 1,
        area.casefold(),
        0 if total is not None else 1,
        -(total if total is not None else 0),
        # Names last, so a re-export of unchanged data produces an identical file and a
        # diff between two exports is a real change rather than dictionary ordering.
        (record.business_name or "").strip().casefold(),
    )


def _text(value: Any) -> str | None:
    """Trim, and turn whitespace-only into a genuinely empty cell."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _joined(values: Sequence[str] | None) -> str | None:
    """`["no website", "public phone"]` -> `"no website; public phone"`.

    Semicolons rather than commas: several signals and every automation label contain
    commas, and a reader cannot tell a separator from a comma inside an item.
    """
    if not values:
        return None
    items = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(items) or None


def _flag(value: bool | None) -> str | None:
    """True/False/None -> yes/no/blank.

    Blank is load-bearing: "we have never checked whether they run ads" and "they do not
    run ads" are different facts, and FALSE in a cell reads as the second one.
    """
    if value is None:
        return None
    return "yes" if value else "no"


def _row_values(record: LeadRecord) -> dict[str, Any]:
    contact = record.contact or ContactPerson()
    evidence = record.evidence or Evidence()
    verdict = record.verdict or OperatorVerdict()
    return {
        "business_name": _text(record.business_name),
        "niche": _text(record.niche),
        "business_type": _text(record.business_type),
        "country": _text(record.country),
        "state": _text(record.state),
        "city": _text(record.city),
        "search_area": _text(record.search_area),
        "address": _text(record.address),
        "lat": record.lat,
        "lng": record.lng,
        "phone": _text(record.phone),
        "email": _text(record.email),
        "website": _text(record.website),
        "instagram_handle": _text(record.instagram_handle),
        "facebook_url": _text(record.facebook_url),
        "contact_name": _text(contact.name),
        "contact_role": _text(contact.role),
        "contact_phone": _text(contact.phone),
        "contact_email": _text(contact.email),
        "contact_source": _text(contact.source),
        "reviews": evidence.reviews,
        "rating": evidence.rating,
        "followers": evidence.followers,
        # Stored as a fraction (likes/followers). It stays a fraction here and is displayed
        # as a percentage by the cell's number format -- multiplying by 100 into a column
        # headed `engagement_rate` would make the sheet disagree with every other consumer
        # of the same field.
        "engagement_rate": evidence.engagement_rate,
        "runs_ads": _flag(evidence.runs_ads),
        "peak_hours": _text(record.peak_hours),
        "total_score": record.total_score,
        "audience_band": _text(record.audience_band),
        "banding_method": _text(record.banding_method),
        "signals": _joined(record.signals),
        "ai_summary": _text(record.ai_summary),
        "website_pitch": _text(record.website_pitch),
        "automation_opportunities": _joined(record.automation_opportunities),
        "automation_pitch": _text(record.automation_pitch),
        "my_verdict": _text(verdict.my_verdict),
        "notes": _text(verdict.notes),
        "contacted_on": _text(verdict.contacted_on),
        "channel": _text(verdict.channel),
        "outcome": _text(verdict.outcome),
    }


def build_row(record: LeadRecord) -> ExportRow:
    """One record -> one full-width row.

    Width is guaranteed structurally: the row is built by walking `COLUMNS` and looking
    each name up, so a value this module forgot to produce leaves a blank cell in the right
    place rather than shortening the row and shifting every column after it. A value keyed
    to a column that does not exist is the other half of the same mistake and raises here,
    because it means a column was renamed in one place and not the other.
    """
    values = _row_values(record)
    unknown = sorted(set(values) - set(COLUMNS))
    if unknown:
        raise KeyError(f"values for columns that do not exist: {', '.join(unknown)}")
    sources = record.evidence_sources or {}
    return ExportRow(
        cells=tuple(
            Cell(value=values.get(column), source=sources.get(column) or None)
            for column in COLUMNS
        )
    )


def build_rows(records: Iterable[LeadRecord]) -> list[ExportRow]:
    """Every record, sorted for a day of fieldwork. See `sort_key`."""
    return [build_row(record) for record in sorted(records, key=sort_key)]
