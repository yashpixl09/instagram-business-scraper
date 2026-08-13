-- lead_bands -- audience size, banded at read time.
--
-- A band is a statement about a business relative to its competitors, not an absolute
-- fact about it. 400 reviews is a large cafe in a residential pocket and a small one on a
-- restaurant strip, and the operator's question -- "is this one of the bigger ones around
-- here?" -- only has an answer inside a cohort. So the band is computed here, per read,
-- and never stored: the same business changes band when its neighbours are discovered,
-- and a stored band would quietly go stale the moment the next sweep landed.
--
-- The cohort is (niche_id, city). Business is local; a salon in Bangalore is not competing
-- with a salon in Delhi, and pooling them would band by metro size rather than by
-- audience.
--
-- --------------------------------------------------------------------------------------
-- Why 30
-- --------------------------------------------------------------------------------------
-- NTILE(3) always fills three tiles. It does not ask whether three tiles mean anything --
-- given three businesses it hands out 'small', 'medium' and 'large' one apiece, and given
-- one it declares that business 'small'. The output is a valid percentile and a false
-- statement: with a handful of rows, "the largest of the ones we happen to have found so
-- far" is being reported to the operator as "large", and the sheet gives no hint that the
-- comparison set was three businesses deep.
--
-- Below 30 in a cohort the ranking is therefore abandoned in favour of the absolute review
-- thresholds from the spec's scoring table (<100 small, 100-500 medium, >500 large). Those
-- are weaker -- they ignore local context, which is the whole point of banding -- but they
-- are honest at any cohort size, and being approximately right beats being precisely
-- arbitrary. 30 is ten per tile: enough that a tile boundary reflects the cohort rather
-- than the discovery order.
--
-- `banding_method` exposes which rule was in force, because a sheet where two rows say
-- 'large' for different reasons is a sheet the operator cannot reason about. It names the
-- cohort's rule, so it is populated even on rows banded 'unknown' -- a reader can tell
-- whether the neighbouring bands came from percentiles or thresholds.
--
-- Cohort size counts only businesses that have an `audience_index`. Forty rows of which
-- five are scored is a cohort of five for ranking purposes, and NTILE over those five is
-- exactly as meaningless as NTILE over five rows would be.

CREATE VIEW lead_bands AS
WITH policy AS (
    -- The one place the cohort threshold is written down.
    SELECT 30::int AS min_cohort
),
latest_score AS (
    -- `scores` is append-only and versioned: a 'google-only' row at discovery, a second
    -- row after Instagram enrichment, both kept. The latest is authoritative. `id DESC`
    -- breaks ties because two rows written in one transaction share `now()` exactly, and
    -- without it the winner would be whichever the planner happened to emit first.
    SELECT DISTINCT ON (s.business_id)
           s.business_id,
           s.audience_index,
           s.scorer_version,
           s.total,
           s.scored_at
    FROM scores s
    ORDER BY s.business_id, s.scored_at DESC, s.id DESC
),
latest_reviews AS (
    -- Review count for the absolute fallback. It lives in the enrichment payload rather
    -- than a column because `enrichments` is a time series -- 180 -> 340 reviews in two
    -- months is a buying signal a snapshot column would erase.
    --
    -- The regex guard is not decoration: `data` is provider JSON, and an unguarded cast
    -- turns one malformed payload into a view that raises for every row in the database.
    -- A missing or unparseable count reads as unknown, never as zero -- zero would band a
    -- business 'small' on the strength of a parse failure.
    SELECT DISTINCT ON (e.business_id)
           e.business_id,
           CASE WHEN e.data->>'reviews' ~ '^[0-9]+$' THEN (e.data->>'reviews')::int END AS reviews
    FROM enrichments e
    WHERE e.source = 'google_maps' AND e.status = 'ok'
    ORDER BY e.business_id, e.fetched_at DESC, e.id DESC
),
cohort AS (
    -- LEFT JOINs throughout: a business with no score row at all, or no enrichment, still
    -- belongs in the sheet. Dropping it would hide the leads the pipeline failed on, which
    -- is precisely the set worth looking at.
    SELECT b.id AS business_id,
           b.name,
           b.niche_id,
           b.city,
           b.search_area,
           ls.audience_index,
           ls.scorer_version,
           ls.total,
           ls.scored_at,
           lr.reviews,
           -- count() of a nullable column counts non-nulls, which is the rankable
           -- population and therefore the cohort size that decides the method.
           count(ls.audience_index) OVER (PARTITION BY b.niche_id, b.city) AS cohort_size
    FROM businesses b
    LEFT JOIN latest_score ls   ON ls.business_id = b.id
    LEFT JOIN latest_reviews lr ON lr.business_id = b.id
),
ranked AS (
    -- Unscored rows are filtered out before NTILE rather than banded afterwards. Left in,
    -- they would sort last (NULLS LAST by default), occupy the top tile, and push genuinely
    -- large businesses down into 'medium' -- the tiles would be counting missing data.
    SELECT c.business_id,
           ntile(3) OVER (
               PARTITION BY c.niche_id, c.city
               ORDER BY c.audience_index, c.business_id
           ) AS tile
    FROM cohort c
    WHERE c.audience_index IS NOT NULL
)
SELECT c.business_id,
       c.name,
       c.niche_id,
       c.city,
       c.search_area,
       c.audience_index,
       c.reviews,
       c.total,
       c.scorer_version,
       c.scored_at,
       c.cohort_size,
       CASE WHEN c.cohort_size >= p.min_cohort THEN 'relative' ELSE 'absolute' END
           AS banding_method,
       CASE
           -- No evidence at all. Say so; do not guess a band from an absent number.
           WHEN c.audience_index IS NULL THEN 'unknown'
           WHEN c.cohort_size >= p.min_cohort THEN
               CASE r.tile WHEN 1 THEN 'small' WHEN 2 THEN 'medium' ELSE 'large' END
           -- Absolute fallback, per the spec's scoring thresholds.
           WHEN c.reviews IS NULL THEN 'unknown'
           WHEN c.reviews < 100 THEN 'small'
           WHEN c.reviews <= 500 THEN 'medium'
           ELSE 'large'
       END AS audience_band
FROM cohort c
CROSS JOIN policy p
LEFT JOIN ranked r ON r.business_id = c.business_id;

COMMENT ON VIEW lead_bands IS
  'Audience bands computed per read against the (niche_id, city) cohort. NTILE(3) at 30+ '
  'rankable businesses, absolute review thresholds below that, ''unknown'' with no '
  'audience_index. Never stored: a band changes when a neighbour is discovered.';
