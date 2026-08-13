-- MEMORY plane: working, episodic, semantic.
--
-- Compaction discards raw tool output and navigation transcripts and writes what it
-- learned here, then reinitialises from `run_summaries` plus `agent_notes`. The discarded
-- detail is still in `events` -- these tables are the summary, not the record.
--
-- DDL is the spec's `## Schema` block verbatim.

CREATE TABLE memories (
  id         bigserial PRIMARY KEY,
  scope      text NOT NULL,           -- global | goal | business
  scope_id   uuid,
  kind       text NOT NULL,           -- fact | lesson | preference | pattern
  content    text NOT NULL,
  embedding  vector(1536),
  confidence real DEFAULT 0.5,
  source_run uuid,
  expires_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- HNSW rather than IVFFlat: no training step, and it does not degrade as rows are added.
-- Production-grade below ~10M vectors against an expected ceiling of ~200k.
CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE run_summaries (
  run_id      uuid PRIMARY KEY,
  narrative   text NOT NULL,
  decisions   jsonb NOT NULL,
  open_issues jsonb NOT NULL,
  embedding   vector(1536)
);

CREATE TABLE agent_notes (
  id         bigserial PRIMARY KEY,
  run_id     uuid NOT NULL,
  key        text NOT NULL,
  value      jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, key)
);
