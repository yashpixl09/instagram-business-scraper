"""Per-niche verification status: the loud signal that replaces silent emptiness.

None of the 259 type slugs in `niches.py` has ever been seen in a live SearchAPI response.
They were reconstructed from Google's published display labels, and the reconstruction is
sound -- but a single wrong slug in a profile's `include_types` produces a niche that
returns twenty real places and qualifies none of them, forever, while every log line says
the run succeeded and found nothing. That failure is indistinguishable from "there are no
cake shops in Jayanagar", and it costs a credit every time it happens.

So the qualification outcome is accumulated per niche, and a niche that has seen enough
candidates to have qualified something and qualified nothing is declared STARVED:

    unverified   no live response has qualified anything yet. Either nothing has been
                 searched (`last_seen_at IS NULL`) or too little has been seen to judge.
    verified     a run qualified at least one place. Sticky: the slugs have been proven
                 against live data once, and a later neighbourhood with no cake shops in it
                 is not evidence against them.
    starved      0 qualified out of >= 20 candidates. The type slugs are wrong. This is a
                 bug report, addressed to whoever maintains the registry.

WHY THE REJECTIONS ARE SPLIT THREE WAYS
---------------------------------------
`rejected_types` is the correction data -- the only correction data this system will ever
get -- so a rejection is worthless unless it says which registry edit would fix it. Three
reasons, three opposite edits:

    missing_type      the place carried types, none of them qualifying. The recorded slugs
                      are candidates for `include_types`. This is the common starvation
                      shape: a real cake shop typed `dessert_shop` that the profile forgot.
    excluded_type     the place carried a DISQUALIFYING type. The recorded slug is in
                      `exclude_types` (or caught by an exclude suffix) and may be too broad.
                      The opposite edit: loosen an exclusion rather than add an inclusion.
    no_name_evidence  the type gate passed and the strict name gate did not. Points at
                      `qualification_terms`, and at nothing else.

Merging those into one bucket would produce a list of slugs with no indication of whether
to add them, remove them, or leave them alone -- which is data that looks actionable and is
not. They are kept apart in the stored counts by prefixing the key with the reason, so the
whole thing stays one flat jsonb object that Postgres can merge additively across runs.

ACCUMULATION IS THE POINT
-------------------------
One run of one neighbourhood is not enough evidence to condemn a profile. Counts merge
across runs -- `qualified`, `rejected`, and every per-slug tally -- so the correction data
gets stronger the longer the registry stays wrong, and the 20-candidate threshold is
reached by a niche that is quietly failing in five areas as surely as by one failing badly
in one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lead_engine.models import Lead
from lead_engine.niches import (
    DISQUALIFIES,
    QUALIFIES,
    NicheProfile,
    classify_type,
    has_name_evidence,
    slugify_type,
)

from .cells import ConnectionFactory

# --- states ----------------------------------------------------------------------------

UNVERIFIED = "unverified"
VERIFIED = "verified"
STARVED = "starved"

STATES: tuple[str, ...] = (UNVERIFIED, VERIFIED, STARVED)

#: Candidates a niche must have seen before qualifying none of them is treated as evidence
#: about the registry rather than about the neighbourhood. Twenty is one full Google Maps
#: page: a whole response in which nothing matched.
STARVATION_THRESHOLD = 20

# --- rejection reasons -----------------------------------------------------------------

MISSING_TYPE = "missing_type"
EXCLUDED_TYPE = "excluded_type"
NO_NAME_EVIDENCE = "no_name_evidence"

REASONS: tuple[str, ...] = (MISSING_TYPE, EXCLUDED_TYPE, NO_NAME_EVIDENCE)

#: What is recorded when Google returned a place with no type at all. A real and separate
#: signal from "typed, but wrongly": it says the *response parsing* lost the types, not that
#: the registry is missing a slug.
UNTYPED = "_untyped"

#: Separates reason from slug in a stored key. A colon cannot appear in a slug --
#: `slugify_type` emits only `[a-z0-9_]` -- so the key parses back apart unambiguously.
KEY_SEPARATOR = ":"


def rejection_key(reason: str, slug: str) -> str:
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS!r}, got {reason!r}")
    return f"{reason}{KEY_SEPARATOR}{slug}"


def split_rejection_key(key: str) -> tuple[str, str]:
    """`"missing_type:dessert_shop"` -> `("missing_type", "dessert_shop")`."""
    reason, _, slug = str(key).partition(KEY_SEPARATOR)
    return reason, slug


def group_rejected_types(counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Stored flat counts, regrouped by reason for a human to read."""
    grouped: dict[str, dict[str, int]] = {reason: {} for reason in REASONS}
    for key, count in counts.items():
        reason, slug = split_rejection_key(key)
        bucket = grouped.setdefault(reason, {})
        bucket[slug] = bucket.get(slug, 0) + int(count)
    return {reason: slugs for reason, slugs in grouped.items() if slugs}


# --- qualification, with its reason ------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """Why one place did not qualify, and which slugs to blame."""

    reason: str
    slugs: tuple[str, ...]

    def keys(self) -> tuple[str, ...]:
        return tuple(rejection_key(self.reason, slug) for slug in self.slugs)


def slugs_for(lead: Lead) -> tuple[str, ...]:
    """The Google type slugs `matches_niche` will judge, in its own order.

    Deliberately reproduces the slug extraction in `niches.matches_niche` rather than
    calling into it, because that function returns a bool and this needs the slugs. If the
    two ever disagree the tally would blame the wrong slug -- so `tests/test_discovery.py`
    pins them against each other over the whole registry instead of trusting the comment.
    """
    slugs = [slugify_type(lead.category), *(slugify_type(value) for value in lead.raw_categories)]
    return tuple(dict.fromkeys(slug for slug in slugs if slug))


def classify(profile: NicheProfile, lead: Lead) -> Rejection | None:
    """None if the lead qualifies, otherwise why it did not.

    The verdict itself is `niches.matches_niche`'s to make and this must not second-guess
    it; what happens here is that the same rules are re-walked to attribute the failure.
    The precedence is `matches_niche`'s, unchanged: one disqualifying type is fatal before
    either gate, then the type gate, then the strict name gate.
    """
    slugs = slugs_for(lead)
    if not slugs:
        # No type evidence at all. `allow_name_only` profiles can still qualify such a
        # place on its name, so this is only a rejection if the name gate also fails.
        if profile.allow_name_only and profile.strict and has_name_evidence(profile, lead):
            return None
        return Rejection(MISSING_TYPE, (UNTYPED,))

    verdicts = {slug: classify_type(profile, slug) for slug in slugs}

    excluded = tuple(slug for slug, verdict in verdicts.items() if verdict == DISQUALIFIES)
    if excluded:
        return Rejection(EXCLUDED_TYPE, excluded)

    qualifying = tuple(slug for slug, verdict in verdicts.items() if verdict == QUALIFIES)
    if not qualifying:
        # `allow_name_only` is the one way past a failed type gate, and only for a strict
        # profile -- exactly as `matches_niche` reads it.
        if profile.strict and profile.allow_name_only and has_name_evidence(profile, lead):
            return None
        # Nothing disqualified and nothing qualified, so every slug here is NEUTRAL: the
        # precise list of candidates for this profile's `include_types`.
        return Rejection(MISSING_TYPE, slugs)

    if profile.strict and not has_name_evidence(profile, lead):
        # The type gate passed. Only the name gate refused, so the slug is not at fault and
        # recording it under `missing_type` would send the maintainer to the wrong field.
        return Rejection(NO_NAME_EVIDENCE, qualifying)

    return None


@dataclass
class NicheObservation:
    """What one pass learned about one niche. Mergeable, and the unit `record` writes."""

    niche_id: str
    qualified: int = 0
    rejected: int = 0
    rejected_types: dict[str, int] = field(default_factory=dict)

    @property
    def candidates(self) -> int:
        return self.qualified + self.rejected

    def count_qualified(self, n: int = 1) -> None:
        self.qualified += n

    def count_rejection(self, rejection: Rejection) -> None:
        self.rejected += 1
        for key in rejection.keys():
            self.rejected_types[key] = self.rejected_types.get(key, 0) + 1

    def merge(self, other: NicheObservation) -> NicheObservation:
        if other.niche_id != self.niche_id:
            raise ValueError(f"cannot merge {other.niche_id!r} into {self.niche_id!r}")
        merged = dict(self.rejected_types)
        for key, count in other.rejected_types.items():
            merged[key] = merged.get(key, 0) + count
        return NicheObservation(
            niche_id=self.niche_id,
            qualified=self.qualified + other.qualified,
            rejected=self.rejected + other.rejected,
            rejected_types=merged,
        )


def observe(profile: NicheProfile, leads: Iterable[Lead]) -> tuple[list[Lead], NicheObservation]:
    """Split candidates into the ones that qualify and a tally of why the rest did not."""
    observation = NicheObservation(profile.id)
    qualified: list[Lead] = []
    for lead in leads:
        rejection = classify(profile, lead)
        if rejection is None:
            qualified.append(lead)
            observation.count_qualified()
        else:
            observation.count_rejection(rejection)
    return qualified, observation


def derive_state(qualified: int, rejected: int) -> str:
    """The state these accumulated totals imply.

    The pure mirror of the CASE expression in the SQL below. Both exist because the stored
    state has to be computed from post-merge totals inside the upsert, while callers and
    tests need to reason about the rule without a database. They are pinned against each
    other in `tests/test_discovery.py`, which is the only thing that stops them drifting.
    """
    if qualified > 0:
        return VERIFIED
    if rejected >= STARVATION_THRESHOLD:
        return STARVED
    return UNVERIFIED


@dataclass(frozen=True)
class NicheStatus:
    """A row of `niche_status`."""

    niche_id: str
    state: str
    qualified: int
    rejected: int
    rejected_types: dict[str, int]
    last_seen_at: datetime | None

    @property
    def candidates(self) -> int:
        return self.qualified + self.rejected

    @property
    def starved(self) -> bool:
        return self.state == STARVED

    def by_reason(self) -> dict[str, dict[str, int]]:
        return group_rejected_types(self.rejected_types)


# --- SQL -------------------------------------------------------------------------------

_COLUMNS = "niche_id, state, qualified, rejected, rejected_types, last_seen_at"

# The state is derived in SQL, from POST-MERGE totals, in both branches. Deriving it in
# Python and passing it would be wrong on the update path for a reason that is easy to miss:
# a run that qualifies nothing knows only its own numbers, and 0-of-8 twice is 0-of-16 --
# a fact neither run can see on its own. The threshold is interpolated from
# STARVATION_THRESHOLD so the constant has exactly one definition.
_STATE_CASE = f"""
        CASE WHEN {{qualified}} > 0 THEN '{VERIFIED}'
             WHEN {{rejected}} >= {STARVATION_THRESHOLD} THEN '{STARVED}'
             ELSE '{UNVERIFIED}' END
"""

# Counts merge additively, key by key, so a slug seen in three runs carries the sum. Written
# as one statement: read-modify-write from Python would lose a concurrent worker's tally the
# same way a read-then-write budget loses a credit.
_UPSERT = f"""
INSERT INTO niche_status (niche_id, state, qualified, rejected, rejected_types, last_seen_at)
VALUES (%s,
        {_STATE_CASE.format(qualified="%s", rejected="%s")},
        %s, %s, %s::jsonb, %s)
ON CONFLICT (niche_id) DO UPDATE SET
  qualified = niche_status.qualified + excluded.qualified,
  rejected  = niche_status.rejected  + excluded.rejected,
  state = {_STATE_CASE.format(
      qualified="niche_status.qualified + excluded.qualified",
      rejected="niche_status.rejected + excluded.rejected",
  )},
  rejected_types = (
      SELECT coalesce(jsonb_object_agg(slug, tally), '{{}}'::jsonb)
        FROM (SELECT key AS slug, sum(value::bigint) AS tally
                FROM (SELECT * FROM jsonb_each_text(niche_status.rejected_types)
                      UNION ALL
                      SELECT * FROM jsonb_each_text(excluded.rejected_types)) AS pairs
               GROUP BY key) AS merged),
  last_seen_at = excluded.last_seen_at
RETURNING {_COLUMNS}
"""

_SELECT_ONE = f"SELECT {_COLUMNS} FROM niche_status WHERE niche_id = %s"

_SELECT_STARVED = f"SELECT {_COLUMNS} FROM niche_status WHERE state = '{STARVED}' ORDER BY niche_id"

_SELECT_ALL = f"SELECT {_COLUMNS} FROM niche_status ORDER BY niche_id"


def _row(values: Sequence[Any]) -> NicheStatus:
    niche_id, state, qualified, rejected, rejected_types, last_seen_at = values
    # psycopg parses jsonb for us; a stub connection may hand back the string it was given.
    if isinstance(rejected_types, str):
        rejected_types = json.loads(rejected_types)
    return NicheStatus(
        niche_id=niche_id,
        state=state,
        qualified=int(qualified),
        rejected=int(rejected),
        rejected_types={key: int(value) for key, value in (rejected_types or {}).items()},
        last_seen_at=last_seen_at,
    )


class NicheStatusStore:
    """`niche_status`, accumulated across every run. Stateless, commits its own writes."""

    def __init__(self, connect: ConnectionFactory) -> None:
        self._connect = connect

    def record(self, observation: NicheObservation, *, seen_at: datetime) -> NicheStatus:
        """Merge one pass's tally into the niche's running totals.

        `seen_at` is always written, even by a pass that saw zero candidates. That stamp is
        what distinguishes "this niche has never been searched" from "this niche has been
        searched and returned nothing", and those call for opposite investigations.
        """
        payload = json.dumps({key: int(value) for key, value in observation.rejected_types.items()})
        with self._connect() as conn:
            row = conn.execute(
                _UPSERT,
                (
                    observation.niche_id,
                    observation.qualified,
                    observation.rejected,
                    observation.qualified,
                    observation.rejected,
                    payload,
                    seen_at,
                ),
            ).fetchone()
            conn.commit()
        if row is None:  # pragma: no cover - an upsert always returns its row
            raise RuntimeError(f"niche_status upsert returned nothing for {observation.niche_id!r}")
        return _row(row)

    def get(self, niche_id: str) -> NicheStatus | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_ONE, (niche_id,)).fetchone()
            conn.commit()
        return None if row is None else _row(row)

    def starved(self) -> list[NicheStatus]:
        """Every niche whose slugs the evidence says are wrong. Check this after a run."""
        with self._connect() as conn:
            rows = conn.execute(_SELECT_STARVED, ()).fetchall()
            conn.commit()
        return [_row(row) for row in rows]

    def all(self) -> list[NicheStatus]:
        with self._connect() as conn:
            rows = conn.execute(_SELECT_ALL, ()).fetchall()
            conn.commit()
        return [_row(row) for row in rows]
