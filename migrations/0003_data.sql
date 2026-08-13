-- DATA plane: the businesses themselves and everything provenanced to them.
--
-- DDL is the spec's `## Schema` block, plus two columns the spec's prose requires but its
-- DDL block omits -- `businesses.dedupe_key` and `scores.audience_index`. Both are flagged
-- below.

CREATE TABLE businesses (
  id               uuid PRIMARY KEY,
  place_id         text UNIQUE,
  name             text NOT NULL,
  niche_id         text NOT NULL,
  country          text,
  state            text,
  city             text NOT NULL,
  search_area      text,
  address          text,
  lat              double precision,
  lng              double precision,
  phone            text,
  email            text,
  website          text,
  instagram_handle text,
  facebook_url     text,
  first_seen_at    timestamptz NOT NULL DEFAULT now(),
  last_seen_at     timestamptz NOT NULL DEFAULT now(),

  -- ADDED (spec prose, not spec DDL).
  --
  -- `place_id` only dedupes what Google gave a place id to. Everything else -- a Firecrawl
  -- hit, a directory listing, the same salon returned under two spellings from two
  -- overlapping search tiles -- arrives without one, and `place_id UNIQUE` lets all of it
  -- in twice.
  --
  -- `lead_engine.dedupe.dedupe_key` computes the key as a four-tier cascade, taking the
  -- first tier that yields anything:
  --
  --     provider|<provider_id>            an authoritative id, when there is one
  --     contact|<name>|<phone-or-host>    same name, same phone or same website host
  --     address|<name>|<address>          same name at the same address
  --     location|<name>|<lat>|<lng>       same name at the same 5dp coordinate
  --
  -- every component lowercased with non-alphanumerics stripped, so "Cafe Noir" and
  -- "cafe-noir" collide as intended.
  --
  -- The database stores and uniquely indexes the key (see 0008); it never computes one.
  -- `dedupe.py` is part of the pure core and this layer must never import it -- the key
  -- arrives already computed on the write path. Nullable because a row can be inserted
  -- before its key is derived; Postgres treats NULLs as distinct, so those rows do not
  -- collide with each other.
  dedupe_key       text
);

CREATE TABLE contacts (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  name        text,
  role        text,                   -- owner|manager|marketing|unknown
  phone       text,
  email       text,
  source      text NOT NULL,          -- review_reply|ig_bio|website|search|directory
  source_url  text,
  confidence  real NOT NULL DEFAULT 0.5,
  found_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON contacts (business_id, confidence DESC);

CREATE TABLE enrichments (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  source      text NOT NULL,
  status      text NOT NULL,
  data        jsonb NOT NULL,
  source_url  text,
  fetched_at  timestamptz NOT NULL DEFAULT now(),
  run_id      uuid
);

-- Append-only, so this index serves both "the current value" (LIMIT 1) and "the series"
-- (a salon moving 180 -> 340 reviews is a buying signal no snapshot reveals).
CREATE INDEX ON enrichments (business_id, source, fetched_at DESC);

CREATE TABLE scores (
  id             bigserial PRIMARY KEY,
  business_id    uuid NOT NULL REFERENCES businesses(id),
  scorer_version text NOT NULL,
  -- No `audience_size` column, deliberately. The spec's DDL declared one NOT NULL, which
  -- contradicts its own mechanism: a band is a percentile against a cohort, so it is
  -- computed at read time by `lead_bands` and changes as the corpus grows. A stored band
  -- is stale the moment the next business is discovered, and a stored band that disagrees
  -- with the view is worse than no band at all. What IS stored is `audience_index` below --
  -- the input, not the derived answer.
  total          int,
  demand         int,
  website_gap    int,
  budget         int,
  reachability   int,
  signals        jsonb NOT NULL,
  evidence       jsonb NOT NULL,
  scored_at      timestamptz NOT NULL DEFAULT now(),

  -- ADDED (spec prose, not spec DDL).
  --
  -- `lead_engine.scoring.audience_index` collapses every audience signal into one
  -- comparable number in [0,1]: reviews .40, followers .30, engagement rate .20,
  -- popular-times density .10, with reviews and followers log-transformed because both
  -- distributions are heavy-tailed. Weights renormalise over whichever components are
  -- present, so a Google-only pass is comparable against an Instagram-enriched one instead
  -- of capping at 0.5.
  --
  -- This is what `lead_bands` (0009) ranks a cohort on. NULL is meaningful and must stay
  -- permitted: it is "nothing is known", which the view bands 'unknown' rather than
  -- guessing. Distinguish it from 0.0, which is evidence of no audience.
  audience_index numeric CHECK (audience_index IS NULL OR audience_index BETWEEN 0 AND 1)
);

CREATE TABLE verdicts (
  business_id  uuid PRIMARY KEY REFERENCES businesses(id),
  my_verdict   text,
  notes        text,
  contacted_on date,
  channel      text,
  outcome      text,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE outreach (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  kind        text NOT NULL,          -- website_pitch | automation_pitch
  channel     text NOT NULL,          -- visit | call | email | dm
  body        text NOT NULL,
  evidence    jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);
