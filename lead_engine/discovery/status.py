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

WHAT THIS MODULE DOES NOT DO ANY MORE
-------------------------------------
It does not classify. Deciding whether one place qualifies, and which of its type slugs to
blame when it does not, belongs to `providers.searchapi.build_outcome` -- which does it once,
immediately after parsing, for the live client and the fixture provider alike. This module
used to re-walk the same rules against the same profile to produce the same verdict, and two
implementations of one rule is not redundancy: it is a pair that agrees until the day either
side is edited, and then disagrees silently about which niches are broken. What arrives here
now is a finished `SearchOutcome`, and all that happens to it is accumulation.

TWO BUCKETS, BECAUSE THE REPAIRS ARE OPPOSITE
---------------------------------------------
`rejected_types` is the correction data -- the only correction data this system will ever get
-- and it is worthless unless it says which registry edit would fix the niche:

    a slug count      a place carried types and none of them qualified for this profile. The
                      recorded slugs are candidates for `include_types` (or evidence that an
                      exclusion is too broad). The common starvation shape: a real cake shop
                      typed `dessert_shop` that the profile forgot.
    the name gate     the type gate PASSED and the strict name gate did not. No slug is at
                      fault; the fix is a `qualification_terms` edit and nothing else.

A niche starving on the name gate and a niche starving on its taxonomy look identical in a
single "rejected" number and need opposite repairs, so the two are stored apart -- the slug
counts under their own slugs, the name-gate total under one reserved key that no slug can
collide with. Merging them would produce a list that looks actionable and is not.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lead_engine.providers.searchapi import SearchOutcome

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

#: The reserved key inside `rejected_types` that holds the name-gate total. It is not a type
#: slug and cannot be mistaken for one: `slugify_type` emits `[a-z0-9_]` and strips leading
#: underscores, so no Google display label can ever produce a key beginning with `_`.
#:
#: A column of its own would have been the other option, and it was not taken: this number is
#: only ever read beside the slug counts it is being distinguished from, and a migration for
#: one integer buys nothing that a reserved key and two accessors do not.
NAME_GATE_KEY = "_name_gate"


def split_buckets(counts: Mapping[str, int]) -> tuple[dict[str, int], int]:
    """Stored counts, split back into (type slugs, name-gate total)."""
    slugs = {key: int(value) for key, value in counts.items() if not key.startswith("_")}
    return slugs, int(counts.get(NAME_GATE_KEY, 0) or 0)


# --- what one pass learned ----------------------------------------------------------------


@dataclass
class NicheObservation:
    """What one pass learned about one niche. Mergeable, and the unit `record` writes."""

    niche_id: str
    qualified: int = 0
    rejected: int = 0
    #: Google type slug -> times it appeared on a place this niche refused. Never contains
    #: the name-gate count; that is `name_gate_rejected`.
    rejected_types: dict[str, int] = field(default_factory=dict)
    #: Places whose types qualified and whose name did not, for a `strict` profile.
    name_gate_rejected: int = 0

    @classmethod
    def from_outcome(cls, outcome: SearchOutcome, *, dropped: int = 0) -> NicheObservation:
        """Project one finished search onto the niche's running tally.

        `dropped` is how many of `outcome.leads` the caller discarded before counting -- in
        practice the `"Unnamed business"` sentinel, which is a placeholder rather than a
        business and must not be counted as a qualified one. It comes out of `qualified`
        only. `rejected` is `returned - qualified`, which is unchanged by the removal, and a
        NAMELESS place that the provider already rejected is not visible from here at all:
        its slugs are in `rejected_types` and they stay there, since what its types were is
        evidence about the registry whether or not Google could read its sign.
        """
        dropped = max(0, int(dropped))
        return cls(
            niche_id=outcome.niche_id,
            qualified=max(0, int(outcome.qualified) - dropped),
            rejected=max(0, int(outcome.returned) - int(outcome.qualified)),
            rejected_types={str(k): int(v) for k, v in outcome.rejected_types.items()},
            name_gate_rejected=max(0, int(outcome.name_gate_rejected)),
        )

    @property
    def candidates(self) -> int:
        return self.qualified + self.rejected

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
            name_gate_rejected=self.name_gate_rejected + other.name_gate_rejected,
        )

    def stored_counts(self) -> dict[str, int]:
        """Both buckets as the one flat jsonb object the table holds."""
        counts = {key: int(value) for key, value in self.rejected_types.items()}
        if self.name_gate_rejected:
            counts[NAME_GATE_KEY] = int(self.name_gate_rejected)
        return counts


def derive_state(qualified: int, rejected: int) -> str:
    """The state these accumulated totals imply.

    The pure mirror of the CASE expression in the SQL below, and of `niche_state()` in
    migration 0012. All three exist because the stored state has to be computed from
    post-merge totals inside the upsert, while callers and tests need to reason about the
    rule without a database. They are pinned against each other in `tests/test_discovery.py`,
    which is the only thing that stops them drifting.
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
    #: Type slugs only. The name-gate total is stored in the same jsonb and read out here.
    rejected_types: dict[str, int]
    last_seen_at: datetime | None
    name_gate_rejected: int = 0

    @property
    def candidates(self) -> int:
        return self.qualified + self.rejected

    @property
    def starved(self) -> bool:
        return self.state == STARVED

    @property
    def worst_types(self) -> list[tuple[str, int]]:
        """The slugs to read first: biggest offender first, ties broken alphabetically."""
        return sorted(self.rejected_types.items(), key=lambda kv: (-kv[1], kv[0]))


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
    slugs, name_gate = split_buckets(rejected_types or {})
    return NicheStatus(
        niche_id=niche_id,
        state=state,
        qualified=int(qualified),
        rejected=int(rejected),
        rejected_types=slugs,
        last_seen_at=last_seen_at,
        name_gate_rejected=name_gate,
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
        payload = json.dumps(observation.stored_counts())
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
