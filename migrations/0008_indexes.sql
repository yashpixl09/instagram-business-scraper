-- Indexes the spec's DDL block does not carry.
--
-- Everything inline with its table is in the file that creates the table. These three are
-- here because they serve access paths described in the spec's prose rather than its
-- schema, and because two of them index columns added in 0003.

-- Dedupe, tier zero. Google's `place_id` covers only what Google gave an id to; everything
-- arriving from Firecrawl, a directory, or a second overlapping search tile has none, and
-- `place_id UNIQUE` waves all of it through. This index is the constraint that actually
-- stops the same salon being inserted twice under two spellings.
--
-- Unique and plain rather than partial: NULLs are distinct in Postgres, so rows whose key
-- has not been derived yet coexist freely, and the same index answers point lookups on the
-- write path.
CREATE UNIQUE INDEX businesses_dedupe_key_uniq ON businesses (dedupe_key);

-- Coverage summaries -- "Indiranagar/salon: 47 found, swept 3d ago" -- and freshness
-- sweeps, both of which slice one niche in one city and sort by staleness.
CREATE INDEX businesses_niche_city_last_seen_idx ON businesses (niche_id, city, last_seen_at);

-- `scores` is versioned and append-only: a google-only row at discovery, a second row
-- after Instagram enrichment, both kept. Every consumer wants the latest per business,
-- which is a DISTINCT ON over exactly this ordering (see `lead_bands` in 0009).
CREATE INDEX scores_business_scored_at_idx ON scores (business_id, scored_at DESC);
