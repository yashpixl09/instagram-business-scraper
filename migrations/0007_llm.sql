-- LLM router accounting.
--
-- `llm_calls` is the spec's DDL verbatim. `llm_rate_buckets` is the durable half of the
-- router the spec's prose specifies without giving DDL.

CREATE TABLE llm_calls (
  id            bigserial PRIMARY KEY,
  prompt_hash   text NOT NULL,
  provider      text,
  model         text,
  input_tokens  int,
  output_tokens int,
  cost_usd      numeric(10,6),
  latency_ms    int,
  status        text,
  response      text,
  created_at    timestamptz NOT NULL DEFAULT now()
);

-- The response cache, so a re-run after a crash re-spends nothing on completed work.
-- Partial on status='ok': failures must stay retryable, and caching a 429 as the answer to
-- a prompt would make one bad minute permanent.
CREATE UNIQUE INDEX ON llm_calls (prompt_hash) WHERE status = 'ok';

-- ADDED (spec prose, not spec DDL).
--
-- Token buckets and circuit state persist in Postgres because the alternative -- an
-- in-memory limiter -- resets to full on every restart. A process that crashes while rate
-- limited then comes back, believes it has a full bucket, and walks straight into a 429,
-- which trips the breaker, which is how a restart loop turns one throttle into an outage.
--
-- `circuit_state` is closed | open | half_open: three consecutive failures open it,
-- a probe after 60s half-opens it. `opened_at` is what that 60s is measured from.
CREATE TABLE llm_rate_buckets (
  provider             text PRIMARY KEY,
  tokens               numeric NOT NULL,
  refilled_at          timestamptz NOT NULL DEFAULT now(),
  circuit_state        text NOT NULL DEFAULT 'closed',
  consecutive_failures int NOT NULL DEFAULT 0,
  opened_at            timestamptz
);
