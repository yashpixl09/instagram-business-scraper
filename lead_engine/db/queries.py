"""Every SQL statement this package runs, as a module-level constant.

SQL lives here and nowhere else. `repository.py` binds parameters and maps rows; it never
concatenates a fragment of a statement, and no other module in the project issues SQL of
its own. That is a rule about *where you look when a query is wrong*: one file, greppable,
diffable, reviewable without reading any Python.

Column lists are derived from the dataclasses in `rows.py` rather than typed out twice.
The f-strings below interpolate `dataclasses.fields(...)` names -- identifiers written by
this repository, never user input -- so the usual objection to building SQL with string
formatting does not apply, and in exchange a column list can never drift from the class it
feeds.

Placeholders are named (`%(run_id)s`) throughout. Positional placeholders in a statement
that mentions the same value three times -- `APPEND_EVENT` does -- are how you end up
passing the arguments in the wrong order at 2am.
"""

from __future__ import annotations

from dataclasses import fields

from .rows import (
    BusinessRow,
    ContactRow,
    EnrichmentRow,
    EventRow,
    GoalRow,
    RunRow,
    ScoreRow,
    TaskRow,
)


def column_list(row_class: type, prefix: str = "") -> str:
    """`"id, run_id, type, ..."` for a row dataclass, optionally table-qualified.

    The prefix exists for `CLAIM_TASKS`: its `UPDATE ... FROM claimed` has two visible
    `id` columns, so an unqualified `RETURNING id` is an ambiguous-column error rather
    than a subtle bug -- but every other column in that statement would silently belong to
    whichever relation happened to have it.
    """
    return ", ".join(prefix + field.name for field in fields(row_class))


GOAL_COLUMNS = column_list(GoalRow)
RUN_COLUMNS = column_list(RunRow)
TASK_COLUMNS = column_list(TaskRow)
EVENT_COLUMNS = column_list(EventRow)
BUSINESS_COLUMNS = column_list(BusinessRow)
ENRICHMENT_COLUMNS = column_list(EnrichmentRow)
CONTACT_COLUMNS = column_list(ContactRow)
SCORE_COLUMNS = column_list(ScoreRow)


# --- the queue ---------------------------------------------------------------------------

# Idempotent by construction. `idem_key` is UNIQUE in the schema, so a re-enqueued task --
# a run resuming after a crash, a scheduler firing twice, an operator re-running a sweep --
# collides and is dropped. DO NOTHING rather than DO UPDATE: the first enqueue's payload is
# the one that matters, and a second one must not resurrect a task that has already been
# claimed, completed, or declared dead.
#
# `RETURNING id` yields exactly one row on insert and none on conflict, which is how the
# caller learns which happened; there is no other reliable signal, since rowcount and
# statusmessage both report 0 for a suppressed conflict.
ENQUEUE_TASK = """
INSERT INTO tasks (run_id, type, idem_key, payload, priority, available_at)
VALUES (%(run_id)s, %(type)s, %(idem_key)s, %(payload)s, %(priority)s,
        coalesce(%(available_at)s::timestamptz, now()))
ON CONFLICT (idem_key) DO NOTHING
RETURNING id
"""

# The dequeue. Two halves, and the split is forced rather than stylistic: `FOR UPDATE SKIP
# LOCKED` is a SELECT-only locking clause, so the rows have to be picked and locked in a
# CTE and updated by id in the outer statement. That is also exactly what makes the queue
# safe under concurrency -- two workers running this at the same instant lock disjoint sets
# of rows, because the second one skips what the first has locked instead of blocking on it
# and then re-reading rows it has already been handed.
#
# `attempts` increments HERE, on the claim, not in `FAIL_TASK`. A worker that is killed
# between claiming and finishing never reports anything, so a counter that only moved on
# failure would leave that task's attempts at zero forever and the reaper would requeue it
# on an infinite loop. Counting claims means every delivery costs an attempt, whatever
# became of the worker, and `max_attempts` is a real ceiling.
CLAIM_TASKS = f"""
WITH claimed AS (
    SELECT id
      FROM tasks
     WHERE status IN ('pending', 'retry')
       AND type = ANY(%(types)s)
       AND available_at <= now()
     ORDER BY priority, id
     FOR UPDATE SKIP LOCKED
     LIMIT %(batch)s
)
UPDATE tasks t
   SET status = 'running',
       locked_by = %(worker_id)s,
       lease_expires = now() + %(lease)s::interval,
       attempts = t.attempts + 1
  FROM claimed c
 WHERE t.id = c.id
RETURNING {column_list(TaskRow, "t.")}
"""

#: Push one task's deadline out from NOW, not from when its batch was claimed.
#:
#: A worker handed five tasks at once holds five leases that all started together. Renewing
#: as each task begins is what makes a serial worker's leases describe the work actually in
#: progress rather than the moment the batch was handed over.
#:
#: Guarded on ownership and status: a worker must not extend a lease on a task the reaper has
#: already taken back and given to someone else. Returning no row is the signal that happened.
RENEW_LEASE = f"""
UPDATE tasks
   SET lease_expires = now() + %(lease)s::interval
 WHERE id = %(task_id)s
   AND locked_by = %(worker_id)s
   AND status = 'running'
RETURNING {column_list(TaskRow)}
"""

#: Only a RUNNING task may be completed.
#:
#: Without the status guard a late duplicate -- a worker whose lease the reaper already
#: reclaimed, finishing anyway -- revives a task that is `dead` and reports it `done`, or
#: overwrites the result of a task some other worker already finished. Returning no row is
#: how the caller learns it no longer owns the task, which is information it needs and
#: cannot get any other way.
COMPLETE_TASK = f"""
UPDATE tasks
   SET status = 'done',
       result = %(result)s,
       error = NULL,
       locked_by = NULL,
       lease_expires = NULL
 WHERE id = %(task_id)s
   AND status = 'running'
RETURNING {TASK_COLUMNS}
"""

# Exponential backoff, capped at an hour: 2s, 4s, 8s ... after the 1st, 2nd, 3rd claim.
# `attempts` has already been incremented by the claim, so the first failure waits 2
# seconds rather than 1 -- deliberate, because the overwhelmingly common first failure is
# a rate-limited or briefly-down upstream, and one second later is too soon to be worth the
# call. The cap keeps a permanently broken dependency from pushing a task's next attempt
# past the end of the run.
_BACKOFF = "now() + (least(pow(2, attempts), 3600) * interval '1 second')"

# `attempts >= max_attempts` reads "this delivery was the last one it was entitled to".
# 'dead' is terminal and deliberately not 'failed': nothing retries it, and the row stays
# with its final error for whoever asks why the run is short.
#: Only a RUNNING task may be failed. Same guard as COMPLETE_TASK, and the worse direction:
#: failing an already-`done` task flips it back to `retry` while KEEPING its stale result, so
#: it is delivered a second time and performed twice -- for a discovery task, a second billed
#: search for a cell already swept. A further failure then marks it `dead`, and the run
#: reports a task buried that in fact succeeded.
FAIL_TASK = f"""
UPDATE tasks
   SET status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'retry' END,
       error = %(error)s,
       locked_by = NULL,
       lease_expires = NULL,
       available_at = {_BACKOFF}
 WHERE id = %(task_id)s
   AND status = 'running'
RETURNING {TASK_COLUMNS}
"""

# The crash path. A worker that dies mid-task reports nothing at all, so the only evidence
# is a lease that stopped being renewed. Same backoff and the same death rule as an
# explicit failure -- a task that kills three workers is not healthier than one that
# returned three errors, and treating it as retryable forever is how one poisoned payload
# takes down a run.
REAP_EXPIRED_LEASES = f"""
UPDATE tasks
   SET status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'retry' END,
       error = 'lease expired',
       locked_by = NULL,
       lease_expires = NULL,
       available_at = {_BACKOFF}
 WHERE status = 'running'
   AND lease_expires < now()
RETURNING {TASK_COLUMNS}
"""

SELECT_TASK = f"""
SELECT {TASK_COLUMNS}
  FROM tasks
 WHERE id = %(task_id)s
"""


# --- control plane -----------------------------------------------------------------------

CREATE_GOAL = f"""
INSERT INTO goals (id, name, spec, schedule, status)
VALUES (%(id)s, %(name)s, %(spec)s, %(schedule)s, %(status)s)
RETURNING {GOAL_COLUMNS}
"""

CREATE_RUN = f"""
INSERT INTO runs (id, goal_id, status, trigger, stats, started_at)
VALUES (%(id)s, %(goal_id)s, %(status)s, %(trigger)s, %(stats)s, now())
RETURNING {RUN_COLUMNS}
"""

# `stats` merges rather than replaces: a run accumulates counters from several stages, and
# the last writer must not erase what the earlier ones recorded.
FINISH_RUN = f"""
UPDATE runs
   SET status = %(status)s,
       stats = stats || coalesce(%(stats)s::jsonb, '{{}}'::jsonb),
       finished_at = now()
 WHERE id = %(id)s
RETURNING {RUN_COLUMNS}
"""

# `events.seq` has no default and the schema's index on (run_id, seq) is not unique, so
# nothing in the database stops two concurrent appends from taking the same number. This
# transaction-scoped advisory lock does. It is keyed on the run, so appends to different
# runs never wait on each other, and it is released by COMMIT or ROLLBACK -- there is no
# unlock statement to forget.
LOCK_RUN_EVENTS = """
SELECT pg_advisory_xact_lock(hashtext('lead_engine_events'), hashtext(%(run_id)s::text))
"""

# INSERT ... SELECT, so the next sequence number is read and written in one statement.
# Casts are explicit because a parameter in a SELECT list does not inherit the target
# column's type the way a parameter in a VALUES list does.
APPEND_EVENT = f"""
INSERT INTO events (run_id, task_id, seq, type, payload)
SELECT %(run_id)s::uuid, %(task_id)s::bigint, coalesce(max(seq), 0) + 1,
       %(type)s::text, %(payload)s::jsonb
  FROM events
 WHERE run_id = %(run_id)s::uuid
RETURNING {EVENT_COLUMNS}
"""


# --- data plane --------------------------------------------------------------------------

# Upsert on `dedupe_key`, which is what `businesses_dedupe_key_uniq` (0008) indexes.
#
# Two things worth knowing before calling this:
#
# * A NULL `dedupe_key` never conflicts -- NULLs are distinct in Postgres -- so a caller
#   that has not derived a key yet inserts a new row every single time. That is the
#   schema's deliberate choice, not an accident here.
# * The conflict target is the dedupe key alone. `place_id` is also UNIQUE, and a row whose
#   place id already belongs to a *different* dedupe key raises UniqueViolation instead of
#   updating. Postgres allows only one inference target per statement, so this cannot be
#   fixed in the SQL; the caller resolves it by deriving the key from the place id when it
#   has one, which `lead_engine.dedupe` does as its first tier.
#
# Optional columns coalesce so that a thin later sighting cannot blank out a fat earlier
# one: a directory listing with no phone number must not erase the phone number Google
# gave us. The NOT NULL identity columns take the new value outright.
UPSERT_BUSINESS = f"""
INSERT INTO businesses (id, place_id, name, niche_id, country, state, city, search_area,
                        address, lat, lng, phone, email, website, instagram_handle,
                        facebook_url, dedupe_key)
VALUES (%(id)s, %(place_id)s, %(name)s, %(niche_id)s, %(country)s, %(state)s, %(city)s,
        %(search_area)s, %(address)s, %(lat)s, %(lng)s, %(phone)s, %(email)s, %(website)s,
        %(instagram_handle)s, %(facebook_url)s, %(dedupe_key)s)
ON CONFLICT (dedupe_key) DO UPDATE
   SET place_id         = coalesce(excluded.place_id, businesses.place_id),
       name             = excluded.name,
       niche_id         = excluded.niche_id,
       country          = coalesce(excluded.country, businesses.country),
       state            = coalesce(excluded.state, businesses.state),
       city             = excluded.city,
       search_area      = coalesce(excluded.search_area, businesses.search_area),
       address          = coalesce(excluded.address, businesses.address),
       lat              = coalesce(excluded.lat, businesses.lat),
       lng              = coalesce(excluded.lng, businesses.lng),
       phone            = coalesce(excluded.phone, businesses.phone),
       email            = coalesce(excluded.email, businesses.email),
       website          = coalesce(excluded.website, businesses.website),
       instagram_handle = coalesce(excluded.instagram_handle, businesses.instagram_handle),
       facebook_url     = coalesce(excluded.facebook_url, businesses.facebook_url),
       last_seen_at     = now()
RETURNING {BUSINESS_COLUMNS}
"""

# Append-only. A second Google fetch for the same business is a new row, because the series
# (180 reviews in March, 340 in August) is itself evidence and an UPDATE would destroy it.
INSERT_ENRICHMENT = f"""
INSERT INTO enrichments (business_id, source, status, data, source_url, run_id)
VALUES (%(business_id)s, %(source)s, %(status)s, %(data)s, %(source_url)s, %(run_id)s)
RETURNING {ENRICHMENT_COLUMNS}
"""

# Append-only, same reasoning as `INSERT_ENRICHMENT`: a page that names a second contact on
# a later fetch adds a row rather than overwriting the first. `found_at` is left to its
# `now()` default, same as `fetched_at` above.
INSERT_CONTACT = f"""
INSERT INTO contacts (business_id, name, role, phone, email, source, source_url, confidence)
VALUES (%(business_id)s, %(name)s, %(role)s, %(phone)s, %(email)s, %(source)s,
        %(source_url)s, %(confidence)s)
RETURNING {CONTACT_COLUMNS}
"""

# Append-only for the same reason, plus one of its own: scores are versioned. The
# google-only score taken at discovery and the richer one taken after Instagram enrichment
# both stay, and `lead_bands` (0009) picks the latest per business.
INSERT_SCORE = f"""
INSERT INTO scores (business_id, scorer_version, total, demand, website_gap, budget,
                    reachability, signals, evidence, audience_index)
VALUES (%(business_id)s, %(scorer_version)s, %(total)s, %(demand)s, %(website_gap)s,
        %(budget)s, %(reachability)s, %(signals)s, %(evidence)s, %(audience_index)s)
RETURNING {SCORE_COLUMNS}
"""
