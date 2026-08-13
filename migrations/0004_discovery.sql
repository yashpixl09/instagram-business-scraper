-- Discovery bookkeeping: where a sweep has already been, and what it already paid for.
--
-- `search_cells` is the spec's DDL verbatim. `geo_cache` and `negative_cache` are the two
-- caches the spec's prose specifies without giving DDL.

-- Google Maps returns the same ranked twenty results for the same query point every time,
-- so without a moving cursor the second run filters all twenty away and spends a credit
-- for nothing. A cell is one position in the four-dimensional discovery space
-- (page x tile x query_variant x radius); exhaustion is measured, not predicted.
CREATE TABLE search_cells (
  id            bigserial PRIMARY KEY,
  goal_id       uuid NOT NULL REFERENCES goals(id),
  niche_id      text NOT NULL,
  area          text,
  tile_lat      double precision NOT NULL,
  tile_lng      double precision NOT NULL,
  tile_radius_m int NOT NULL,
  query_variant text NOT NULL,
  next_page     int  NOT NULL DEFAULT 1,
  status        text NOT NULL DEFAULT 'pending',
  new_yield     int  NOT NULL DEFAULT 0,
  total_yield   int  NOT NULL DEFAULT 0,
  last_searched_at timestamptz,
  exhausted_at     timestamptz,
  -- `tile_radius_m` belongs in the key. The spec's DDL omitted it while its prose declares
  -- radius one of the four discovery axes -- so two cells at the same point with different
  -- radii would collide, and "a tighter radius surfaces businesses a wide search buries"
  -- would be unreachable.
  UNIQUE (goal_id, niche_id, tile_lat, tile_lng, tile_radius_m, query_variant)
);

-- Cell selection: highest-yield unexhausted cell first, so sweeps self-organise toward
-- productive ground without manual tuning.
CREATE INDEX ON search_cells (goal_id, status, new_yield DESC);

-- ADDED (spec prose, not spec DDL).
--
-- Resolution composes most-specific-first into one string --
-- "Indiranagar, Bangalore, Karnataka, India" -- geocodes it, and caches on exactly that
-- string. Areas repeat across every run of a goal, and re-geocoding them each time buys
-- nothing but latency and quota.
--
-- Unresolvable input is never cached: it aborts the run. A silent fallback to the city
-- centre would send a day of fieldwork to the wrong neighbourhood undetected, which is the
-- same rule that rejects an unsupported niche rather than substituting a near one.
CREATE TABLE geo_cache (
  query        text PRIMARY KEY,
  formatted    text NOT NULL,
  lat          double precision NOT NULL,
  lng          double precision NOT NULL,
  country_code text,
  resolver     text NOT NULL,
  resolved_at  timestamptz NOT NULL DEFAULT now()
);

-- ADDED (spec prose, not spec DDL).
--
-- Firecrawl does per-business lookups, so its waste is the inverse of a sweep's: it
-- re-pays to rediscover absence. A business with no Instagram handle has no Instagram
-- handle next week either, and looking again every run buys the same empty answer at full
-- price.
--
-- Backoff is 30d -> 90d -> 180d -> never, by consecutive miss count. The fourth miss
-- writes a NULL `retry_after`, which means manual reset only -- that is deliberate, since
-- an automatic retry that keeps finding nothing is a subscription to nothing.
CREATE TABLE negative_cache (
  business_id uuid REFERENCES businesses(id),
  source      text NOT NULL,
  misses      int NOT NULL DEFAULT 1,
  retry_after timestamptz,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (business_id, source)
);
