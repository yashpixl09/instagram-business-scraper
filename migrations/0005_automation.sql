-- The second offer: which of a qualified business's workflows are worth automating.
--
-- Detection is rule-based, like scoring -- the LLM writes the pitch prose over these rows,
-- it does not decide which opportunities exist. `trigger_signals` records which evidence
-- fired the rule and `evidence` holds the `enrichment_id`s backing each signal, so every
-- claim in a pitch traces to a source row.
--
-- DDL is the spec's `## Schema` block verbatim, moved out of the DATA file because it
-- depends on a catalog (`automations.py`) that lands in a later phase.

CREATE TABLE automation_opportunities (
  id             bigserial PRIMARY KEY,
  business_id    uuid NOT NULL REFERENCES businesses(id),
  opportunity_id text NOT NULL,       -- key into the automations catalog
  confidence     real NOT NULL,
  trigger_signals jsonb NOT NULL,     -- which evidence fired this
  evidence       jsonb NOT NULL,      -- enrichment_ids backing each signal
  detected_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (business_id, opportunity_id)
);
