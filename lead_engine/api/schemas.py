"""The wire contract: request parsing, response models, and the error envelope.

Pydantic lives here and in `routes.py`, and nowhere else in the project. The repository
hands back frozen dataclasses; those are converted at this boundary with
`model_validate(row, from_attributes=True)` and never carry a pydantic base class down into
the core or the workers.

WHY REQUEST PARSING IS HAND-ROLLED
----------------------------------
Responses are pydantic models. Requests are not, and that is deliberate.

The error envelope this API owes its callers names a *specific code per field* --
`invalid_city`, `invalid_niches`, `invalid_limit`, `invalid_use_ai`, `unsupported_niche`
with the supported list attached. Pydantic answers every one of those with a single
`RequestValidationError` carrying a `loc` tuple, and reconstructing the codes from `loc`
means maintaining a mapping that is longer and more fragile than the checks themselves --
and that silently reports the wrong code the moment a field moves inside the payload.

So `parse_search_request` is the prototype's `lead_finder/api.py:parse_search_request`,
ported rule for rule, including the 160-character city ceiling and the `isinstance(limit,
bool)` rejection (`True` is an `int` in Python and would otherwise be a legal limit of 1).

WHAT IS NEW HERE
----------------
Geography arrives as the `GeoScope` shape:

    {"location": {"country": ..., "state": ..., "city": REQUIRED, "areas": [...]}, ...}

A bare `{"city": "Pune"}` is still accepted, because the prototype's frontend sends it and
breaking it would buy nothing. When a `location` object is present it is authoritative --
its `city` is the city, and a top-level `city` beside it is ignored rather than merged. A
half-specified location is a mistake worth surfacing, not one worth guessing through.

`areas` are capped. Every (area x niche) pair is at least one billed search out of an
allowance of fifty, so a pasted 5,000-element list is not a large request, it is an
expensive typo.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ..geo.scope import GeoScope
from ..niches import UnsupportedNicheError, niche_payload, resolve_niche_ids

#: Length ceiling for any single free-text location component. The prototype's, unchanged.
MAX_TEXT = 160

#: Ranked leads a caller may ask for in one run. The prototype's, unchanged.
MIN_LIMIT, MAX_LIMIT = 1, 500

#: Each area costs at least one billed search per niche, out of fifty that never renew.
MAX_AREAS = 25

#: The registry is 24 profiles; asking for more than that is a malformed request, not a
#: bigger one.
MAX_NICHES = 24

#: Operator verdicts, ported verbatim from the prototype's `OUTREACH_STATUSES`.
VERDICTS: frozenset[str] = frozenset(
    {
        "not_contacted",
        "shortlisted",
        "contacted",
        "replied",
        "follow_up",
        "not_interested",
        "converted",
    }
)

#: How a niche's evidence stands. `starved` means the registry entry is real but nothing has
#: come back for it yet -- which is different from `unverified` (nobody has looked) and must
#: not be flattened into it.
VERIFICATION_STATES: tuple[str, ...] = ("verified", "unverified", "starved")

DEFAULT_VERIFICATION = "unverified"

MAX_NOTES = 5000


class ApiProblem(Exception):
    """One failed request, already classified.

    Carries everything the envelope needs and nothing else. Raised by this module and by the
    service layer; turned into a response by the handlers in `app.py`, which are the only
    place that knows what an HTTP response looks like.
    """

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


# --- the envelope --------------------------------------------------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorEnvelope(BaseModel):
    """`{"error": {"code", "message", "details"}}` -- the only failure shape this API emits.

    Built through a model rather than a dict literal so that a handler cannot ship a
    near-miss: a response body with `detail` instead of `error` is what every FastAPI
    default produces, and a frontend that has to parse two shapes parses neither well.
    """

    error: ErrorBody


def envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return ErrorEnvelope(
        error=ErrorBody(code=code, message=message, details=details or {})
    ).model_dump(mode="json")


# --- requests ------------------------------------------------------------------------------


class SearchSpec(BaseModel):
    """A validated search request, in domain terms.

    Not a wire model: `niche_ids` are resolved registry ids, not the strings the caller
    typed, and `scope` is a `GeoScope`. The CLI builds one of these from argparse and hands
    it to the same service call the route uses, which is what keeps the two surfaces honest
    with each other.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scope: GeoScope
    niche_ids: tuple[str, ...]
    limit: int
    use_ai: bool


class VerdictUpdate(BaseModel):
    """The operator's verdict on one lead. Everything but the verdict is optional."""

    model_config = ConfigDict(frozen=True)

    my_verdict: str
    notes: str = ""
    contacted_on: date | None = None
    channel: str | None = None
    outcome: str | None = None


def _problem(code: str, message: str, details: dict[str, Any] | None = None) -> ApiProblem:
    return ApiProblem(422, code, message, details)


def _text(value: object, code: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_TEXT:
        raise _problem(
            code, f"{field} must be a non-empty string up to {MAX_TEXT} characters."
        )
    return value.strip()


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value.strip()) > MAX_TEXT:
        raise _problem(
            "invalid_request", f"{field} must be a string up to {MAX_TEXT} characters, or null."
        )
    return value.strip() or None


def _areas(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    # A bare string iterates into one-character areas -- eleven of them for "Indiranagar",
    # each a billed search. `GeoScope` raises TypeError on this; catching it here turns it
    # into a 422 with a message instead of a 500.
    if isinstance(value, str) or not isinstance(value, list):
        raise _problem("invalid_request", "areas must be an array of area names.")
    if len(value) > MAX_AREAS:
        raise _problem(
            "invalid_request",
            f"areas must contain at most {MAX_AREAS} names: every area is at least one "
            "billed search per niche.",
        )
    return tuple(_text(area, "invalid_request", "Each area") for area in value)


def parse_search_request(payload: object) -> SearchSpec:
    """Validate a `POST /api/search` body. Raises `ApiProblem` and nothing else.

    Order matters and is the prototype's: city, then niches, then limit, then use_ai. A
    caller fixing one field at a time should meet the same error twice only if they failed
    to fix it.
    """
    if not isinstance(payload, dict):
        raise _problem("invalid_request", "Request body must be an object.")

    location = payload.get("location")
    if location is None:
        # Backward compatibility with the prototype's frontend: a bare city, no envelope.
        location = {"city": payload.get("city")}
    if not isinstance(location, dict):
        raise _problem("invalid_request", "location must be an object.")

    city = _text(location.get("city"), "invalid_city", "City or area")
    country = _optional_text(location.get("country"), "country")
    state = _optional_text(location.get("state"), "state")
    areas = _areas(location.get("areas"))

    niches = payload.get("niches")
    if (
        not isinstance(niches, list)
        or not niches
        or not all(isinstance(item, str) for item in niches)
    ):
        raise _problem(
            "invalid_niches", "Niches must be a non-empty array of supported niche names."
        )
    if len(niches) > MAX_NICHES:
        raise _problem(
            "invalid_niches", f"Niches must contain at most {MAX_NICHES} entries."
        )
    try:
        niche_ids = resolve_niche_ids(list(niches))
    except UnsupportedNicheError as exc:
        # The supported list travels with the rejection. A caller that has to make a second
        # request to find out what it may ask for will hard-code the answer instead.
        raise _problem(
            "unsupported_niche",
            f"Unsupported niche: {exc.value}",
            {"supported_niches": [item["id"] for item in niche_payload()]},
        ) from exc

    limit = payload.get("limit")
    # `isinstance(limit, bool)` first: `True` is an `int` and would pass as a limit of 1.
    if isinstance(limit, bool) or not isinstance(limit, int) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise _problem(
            "invalid_limit", f"Limit must be an integer from {MIN_LIMIT} through {MAX_LIMIT}."
        )

    use_ai = payload.get("use_ai")
    if not isinstance(use_ai, bool):
        raise _problem("invalid_use_ai", "use_ai must be a boolean.")

    try:
        scope = GeoScope(city=city, country=country, state=state, areas=areas)
    except (TypeError, ValueError) as exc:  # pragma: no cover - the checks above pre-empt it
        raise _problem("invalid_city", str(exc)) from exc

    return SearchSpec(scope=scope, niche_ids=tuple(niche_ids), limit=limit, use_ai=use_ai)


def parse_verdict_update(payload: object) -> VerdictUpdate:
    """Validate a `PATCH /api/leads/{id}/status` body.

    `outreach_status` is the prototype's name for the field and is accepted as an alias, for
    the same reason a bare city is.
    """
    if not isinstance(payload, dict):
        raise _problem("invalid_status", "Status update must be an object.")

    verdict = payload.get("my_verdict", payload.get("outreach_status"))
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        raise _problem(
            "invalid_status",
            "my_verdict is required and must be a supported verdict.",
            {"supported_verdicts": sorted(VERDICTS)},
        )

    notes = payload.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str) or len(notes) > MAX_NOTES:
        raise _problem("invalid_notes", f"notes must be a string up to {MAX_NOTES} characters.")

    contacted_on = payload.get("contacted_on")
    if contacted_on is not None:
        if not isinstance(contacted_on, str):
            raise _problem("invalid_status", "contacted_on must be an ISO date (YYYY-MM-DD).")
        try:
            contacted_on = date.fromisoformat(contacted_on)
        except ValueError as exc:
            raise _problem(
                "invalid_status", "contacted_on must be an ISO date (YYYY-MM-DD)."
            ) from exc

    channel = payload.get("channel")
    outcome = payload.get("outcome")
    for name, value in (("channel", channel), ("outcome", outcome)):
        if value is not None and (not isinstance(value, str) or len(value) > MAX_TEXT):
            raise _problem(
                "invalid_status", f"{name} must be a string up to {MAX_TEXT} characters."
            )

    return VerdictUpdate(
        my_verdict=verdict,
        notes=notes,
        contacted_on=contacted_on,
        channel=channel or None,
        outcome=outcome or None,
    )


# --- responses -----------------------------------------------------------------------------

_FROM_ROWS = ConfigDict(from_attributes=True)


class HealthOut(BaseModel):
    status: str


class NicheOut(BaseModel):
    """One registry row, plus how far its evidence has got.

    Everything but `verification` is `niches.niche_payload()` verbatim. `verification` is
    read from `runs.stats` when a run has recorded it and defaults to `unverified`, which is
    the honest answer before anything has looked.
    """

    id: str
    label: str
    queries: list[str]
    include_types: list[str]
    exclude_types: list[str]
    include_suffixes: list[str]
    exclude_suffixes: list[str]
    verification: str = DEFAULT_VERIFICATION


class NicheListOut(BaseModel):
    niches: list[NicheOut]


class BudgetOut(BaseModel):
    """What is left of a billed allowance. Never the key, never a fingerprint of one."""

    provider: str
    configured: bool
    remaining: int | None = None


class ConfigStatusOut(BaseModel):
    """Which providers are wired up -- booleans only.

    No key, no prefix of a key, no length, no fingerprint. A four-character prefix is enough
    to confirm a guess, and this endpoint is reachable from a browser.
    """

    database_configured: bool
    providers: dict[str, bool]
    llm_configured: bool
    search_budget: BudgetOut


class ResolvedLocationOut(BaseModel):
    model_config = _FROM_ROWS

    label: str
    latitude: float
    longitude: float
    radius_meters: int
    precision: str
    gl: str | None = None
    source_query: str


class SearchAcceptedOut(BaseModel):
    """The receipt for an enqueued run. No leads yet -- a worker has not run.

    `planned_searches` is the number of billed searches the enqueued tasks will spend if
    every one of them runs, under the default breadth-first policy of one query variant and
    one page per (area x niche). It is stated here because it is the number worth aborting
    over, and the operator cannot see it anywhere else before the money is gone.
    """

    run_id: UUID
    goal_id: UUID
    status: str
    niches: list[str]
    locations: list[ResolvedLocationOut]
    enqueued_tasks: int
    planned_searches: int
    search_budget_remaining: int | None = None
    message: str


class SearchPlanOut(BaseModel):
    """A dry run of the above: what would be spent, without spending it."""

    niches: list[str]
    locations: list[ResolvedLocationOut]
    planned_searches: int
    search_budget_remaining: int | None = None


class RunOut(BaseModel):
    model_config = _FROM_ROWS

    id: UUID
    goal_id: UUID
    status: str
    trigger: str
    stats: dict[str, Any]
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunListOut(BaseModel):
    runs: list[RunOut]


class LeadOut(BaseModel):
    """One business, with its latest score, its read-time band, and the operator's verdict.

    `audience_index` is `numeric` in Postgres and therefore `Decimal` in the row dataclass;
    it is widened to `float` here because JSON has one number type and a string-encoded
    decimal in the payload is a trap for every client. Nothing downstream re-scores from
    this field -- `scores.audience_index` remains authoritative.
    """

    model_config = _FROM_ROWS

    id: UUID
    place_id: str | None = None
    name: str
    niche_id: str
    country: str | None = None
    state: str | None = None
    city: str
    search_area: str | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    instagram_handle: str | None = None
    facebook_url: str | None = None
    first_seen_at: datetime
    last_seen_at: datetime

    score_total: int | None = None
    audience_index: float | None = None
    audience_band: str | None = None
    banding_method: str | None = None
    cohort_size: int | None = None
    reviews: int | None = None
    scorer_version: str | None = None
    scored_at: datetime | None = None

    my_verdict: str | None = None
    notes: str | None = None
    contacted_on: date | None = None
    channel: str | None = None
    outcome: str | None = None
    verdict_updated_at: datetime | None = None


class LeadListOut(BaseModel):
    run_id: UUID
    count: int
    leads: list[LeadOut]
