-- CONTROL plane: goals -> runs -> tasks -> events.
--
-- Event-sourced and resumable. `events` is the log a crashed run replays to skip the
-- steps it already finished, so nothing here is ever updated in place except task
-- leases.
--
-- DDL is the spec's `## Schema` block verbatim.

CREATE TABLE goals (
  id         uuid PRIMARY KEY,
  name       text NOT NULL,
  spec       jsonb NOT NULL,
  schedule   text,
  status     text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE runs (
  id          uuid PRIMARY KEY,
  goal_id     uuid NOT NULL REFERENCES goals(id),
  status      text NOT NULL,
  trigger     text NOT NULL,
  stats       jsonb NOT NULL DEFAULT '{}',
  started_at  timestamptz,
  finished_at timestamptz
);

CREATE TABLE tasks (
  id            bigserial PRIMARY KEY,
  run_id        uuid NOT NULL REFERENCES runs(id),
  type          text NOT NULL,
  idem_key      text NOT NULL UNIQUE,
  payload       jsonb NOT NULL,
  status        text NOT NULL DEFAULT 'pending',
  priority      int  NOT NULL DEFAULT 100,
  available_at  timestamptz NOT NULL DEFAULT now(),
  attempts      int  NOT NULL DEFAULT 0,
  max_attempts  int  NOT NULL DEFAULT 3,
  locked_by     text,
  lease_expires timestamptz,
  result        jsonb,
  error         text
);

-- The dequeue path: `FOR UPDATE SKIP LOCKED` over claimable work only. Partial, because
-- done and failed rows accumulate forever and must not widen the index the hot loop scans.
CREATE INDEX ON tasks (status, type, priority, available_at)
  WHERE status IN ('pending','retry');

CREATE TABLE events (
  id         bigserial PRIMARY KEY,
  run_id     uuid NOT NULL,
  task_id    bigint,
  seq        int  NOT NULL,
  type       text NOT NULL,
  payload    jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Replay order. `run_id` carries no foreign key on purpose: the log is the record of what
-- happened and must survive anything that removes the run it describes.
CREATE INDEX ON events (run_id, seq);
