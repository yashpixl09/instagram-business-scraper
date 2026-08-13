-- Widen the search budget key from `provider` to (provider, purpose, key_fingerprint).
--
-- 0010 keyed the ledger on the provider alone, which forced two other real dimensions to be
-- smuggled into that one text column as "searchapi:discover:a3f9c2" -- a composite key
-- pretending to be a string, which nothing can sum across and nothing can validate.
--
-- The three dimensions, and why each is a column:
--
--   provider         which vendor is billing us.
--   purpose          'discover' (new ground) vs 'refresh' (re-checking known ground). The
--                    spec splits the allowance 70/30 so that refresh can never quietly eat
--                    the credits discovery needs. Two budgets, not one budget and a policy.
--   key_fingerprint  which API key the credits belong to. The operator rotates keys: when
--                    the current key's allowance is gone, a new key arrives with its own.
--
-- Rotation is why the fingerprint is part of the KEY rather than a column beside it. A key
-- the ledger has not seen creates a fresh row on the next `ensure()`, and that is the whole
-- rotation mechanism -- no reset command, because a manual step is one someone forgets, and
-- forgetting here means the new key inherits a spent budget and the worker refuses to run.
-- The old row stays, so what each allowance actually bought remains on the record.
--
-- The column stores sha256(key)[:12], never the key. See the CHECK below, which is what
-- makes that a guarantee rather than a convention.

ALTER TABLE search_budget
  ADD COLUMN purpose         text NOT NULL DEFAULT 'discover',
  ADD COLUMN key_fingerprint text NOT NULL DEFAULT '';

ALTER TABLE search_budget DROP CONSTRAINT search_budget_pkey;
ALTER TABLE search_budget
  ADD CONSTRAINT search_budget_pkey PRIMARY KEY (provider, purpose, key_fingerprint);

ALTER TABLE search_budget
  -- Two values, spelled one way. Free text here would let a typo ('refesh') open a silent
  -- third budget that no report sums and no worker ever spends.
  ADD CONSTRAINT search_budget_purpose_known
    CHECK (purpose IN ('discover', 'refresh')),

  -- The column that must never contain a credential. A raw API key does not match this
  -- pattern, so passing one by mistake is rejected by the database rather than stored,
  -- logged, and dumped into whatever reads this table next.
  --
  -- Lowercase-only is deliberate and is about money, not tidiness: the same key hexed in
  -- two cases would be two different primary keys, so a client that upper-cased its digest
  -- would silently mint itself a second, full allowance. That is exactly the rotation
  -- mechanism firing when no rotation happened.
  --
  -- '' means "rotation is not modelled for this provider" and is always legal.
  ADD CONSTRAINT search_budget_fingerprint_is_a_digest
    CHECK (key_fingerprint = '' OR key_fingerprint ~ '^[0-9a-f]{12}$');

COMMENT ON COLUMN search_budget.purpose IS
  'discover | refresh. Separate allowances so refresh cannot starve discovery. Until '
  'discovery has swept the target areas, refresh is worth 0: re-checking known ground '
  'only earns its credits once there is no new ground left.';
COMMENT ON COLUMN search_budget.key_fingerprint IS
  'sha256(api_key) truncated to 12 hex chars, or '''' where rotation is not modelled. '
  'Never the key itself. An unseen fingerprint starts a fresh allowance; the retired '
  'key''s row is kept so its spend history survives the rotation.';
