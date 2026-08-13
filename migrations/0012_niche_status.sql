-- Per-niche verification status: is this niche's type vocabulary actually right?
--
-- `lead_engine/niches.py` says it plainly in its own docstring: the 24 profiles' type
-- vocabulary is authored from Google's published label set and from what these queries are
-- expected to return in an Indian metro. Nobody has checked it against a live response yet.
-- A wrong slug does not raise. It qualifies nothing, and a run reports "0 leads for
-- interior designers in Indiranagar" -- which is indistinguishable from a neighbourhood
-- that has no interior designers, and which nobody investigates because nothing failed.
--
-- This table is what makes that failure loud. It accumulates, per niche, how many
-- candidates a live response offered and how many survived `matches_niche`, plus a count of
-- the type slugs that were REJECTED. Those slugs are the correction data: a niche sitting at
-- 0 qualified with `{"interior_design_studio": 14}` in `rejected_types` is telling the
-- operator exactly which string to add to `include_types`.
--
-- THE THREE STATES
--
--   unverified  no live response has been seen for this niche yet, or too few candidates
--               have been seen to conclude anything. `last_seen_at IS NULL` distinguishes
--               the two: NULL means never searched at all.
--   verified    a run qualified at least one place. The vocabulary works. Sticky, because
--               `qualified` accumulates: one neighbourhood with no cafes must not
--               un-verify a niche that has already proven itself.
--   starved     0 qualified out of >= 20 candidates. THE loud signal. Twenty places came
--               back and not one of them matched, which is not what a real neighbourhood
--               looks like -- it is what a wrong slug looks like.
--
-- Counts merge across runs rather than being overwritten, because the evidence for
-- starvation is cumulative: four runs of five candidates each is the same finding as one
-- run of twenty, and a per-run view would never reach the threshold.

CREATE TABLE niche_status (
  niche_id       text PRIMARY KEY,
  state          text NOT NULL DEFAULT 'unverified',
  qualified      int  NOT NULL DEFAULT 0,
  rejected       int  NOT NULL DEFAULT 0,
  -- {"<google type slug>": <times rejected>}. Merged additively by the writer, so the
  -- correction data accumulates instead of being replaced by the latest run's sample.
  rejected_types jsonb NOT NULL DEFAULT '{}',
  -- NULL until a live provider response has been seen. This is what separates "never
  -- searched" from "searched and inconclusive", which are the same state but very
  -- different findings.
  last_seen_at   timestamptz,

  -- Three states, spelled one way. Free text here would let a typo ('starverd') hide the
  -- one signal this table exists to raise, and a report filtering on 'starved' would show
  -- a clean board while the niche returned nothing for a month.
  CONSTRAINT niche_status_state_known
    CHECK (state IN ('unverified', 'verified', 'starved')),
  CONSTRAINT niche_status_counts_non_negative
    CHECK (qualified >= 0 AND rejected >= 0),
  -- An object, never an array or a scalar. `jsonb_each_text` over a non-object raises at
  -- merge time, which would surface as a failed run in a worker rather than as bad data
  -- refused at the door.
  CONSTRAINT niche_status_rejected_types_is_an_object
    CHECK (jsonb_typeof(rejected_types) = 'object')
);

-- The state rule, as a function rather than as a CASE repeated in two branches of an
-- upsert. It is called from both the INSERT and the ON CONFLICT UPDATE path in
-- `lead_engine/discovery/status.py`, so the threshold below is the single SQL definition of
-- when a niche is starved.
--
-- `lead_engine.discovery.status.derive_state` is the Python twin of this function, needed
-- because callers must be able to interpret a status without a database. Two definitions of
-- one rule is a real risk, and it is answered by a test rather than by a comment:
-- tests/test_discovery.py checks the two agree across a matrix of counts, so drift fails a
-- test instead of quietly disagreeing about which niches are broken.
--
-- Candidates are `qualified + rejected`; the second branch only runs when `qualified` is 0,
-- so it reads exactly as "0 qualified out of 20 or more candidates".
CREATE FUNCTION niche_state(qualified int, rejected int) RETURNS text
  LANGUAGE sql
  IMMUTABLE
  RETURN CASE
           WHEN qualified > 0 THEN 'verified'
           WHEN qualified + rejected >= 20 THEN 'starved'
           ELSE 'unverified'
         END;

-- The operator's question is "which niches are broken", so the index serves that and not
-- the primary key lookup, which is already covered.
CREATE INDEX ON niche_status (state);

COMMENT ON TABLE niche_status IS
  'Per-niche verification of the type vocabulary in lead_engine/niches.py, accumulated '
  'across runs. state=starved means 0 qualified of >= 20 candidates: the niche''s type '
  'slugs are wrong, and rejected_types names the ones it should have matched.';
COMMENT ON COLUMN niche_status.rejected_types IS
  'Google type slug -> times rejected, summed across every run. The correction data for '
  'include_types; the counts are what separate a one-off oddity from the slug this niche '
  'is actually called by.';
