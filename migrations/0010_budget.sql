-- The lifetime ledger for billed provider searches.
--
-- SearchAPI grants 100 free searches, ONE TIME, non-renewing. There is no monthly reset to
-- forgive a mistake. An exhaustive sweep of one neighbourhood -- 13 tiles x 5 query
-- variants x 3 pages -- is 195 searches: the entire allocation, twice over, for one niche
-- in one city. So the ceiling cannot live in a process-local counter that a restart, a
-- crash loop, or a second worker resets to zero. It lives here.
--
-- One row per provider, not one row per run. `used` only ever goes up.

CREATE TABLE search_budget (
  provider    text PRIMARY KEY,
  limit_total int NOT NULL,
  used        int NOT NULL DEFAULT 0,
  updated_at  timestamptz NOT NULL DEFAULT now(),

  -- Defence in depth. The spend path already refuses to exceed the limit in its WHERE
  -- clause; this makes an overspend unrepresentable even if someone later writes a
  -- read-then-write UPDATE, or fixes a "stuck" worker by hand at 2am.
  CONSTRAINT search_budget_used_non_negative CHECK (used >= 0),
  CONSTRAINT search_budget_limit_non_negative CHECK (limit_total >= 0),
  CONSTRAINT search_budget_within_limit CHECK (used <= limit_total)
);

COMMENT ON TABLE search_budget IS
  'Lifetime, non-renewing ledger of billed provider searches. Never reset by a deploy.';
COMMENT ON COLUMN search_budget.used IS
  'Monotonically increasing. Decrementing it spends money that is already gone.';
