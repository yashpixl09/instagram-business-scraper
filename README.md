# Lead Engine

Durable agent runtime that finds local businesses without an effective web presence,
qualifies them against real audience evidence, and produces an operator-ready pipeline.

Quality over volume: a small number of high-confidence leads per run, each carrying enough
verified evidence to walk in, call, or DM without further research. Volume accumulates
across runs.

## Status

Building. Phase 0 of 10 complete — the pure core and the niche registry.

Design: [`docs/superpowers/specs/2026-08-13-lead-engine-design.md`](docs/superpowers/specs/2026-08-13-lead-engine-design.md).

| Phase | | |
|---|---|---|
| 0 | Pure core, 24-niche registry | done |
| 1 | Postgres, migrations, task queue, FastAPI | in progress |
| 2 | SearchAPI discovery, geo resolution, search cells | |
| 3 | Persistence, banding, Excel export | **first usable sheet** |
| 4–10 | Enrichment, contacts, LLM router, Instagram, automation pitches, agent loop, Sheets | |

## Setup

Requires Python 3.11+ and Docker.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

cp env.example .env          # then fill in keys
docker compose up -d         # Postgres 17 + pgvector on host port 5433
.venv/Scripts/python.exe -m lead_engine.db.migrate
```

The compose project is named `lead-engine` and binds host port **5433**, deliberately
avoiding 5432 so it cannot collide with a local Postgres install or another project's
container. Its volume is `lead_engine_pgdata`.

## Tests

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q       # pure suite, no infrastructure
.venv/Scripts/python.exe -m ruff check lead_engine tests
```

No test contacts a live external service. Providers sit behind an injectable transport
driven by recorded fixtures, so the suite runs on a train.

Integration tests need Postgres and skip themselves unless `LEAD_ENGINE_TEST_DSN` is set —
the pure suite must never depend on infrastructure. Run them with:

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q -m integration
```

## Design notes

Three rules do most of the work, and each exists because of a specific failure:

**Scoring is deterministic Python; the LLM only writes prose.** Every generated claim
carries the `enrichment_id` it came from, so a sentence that cannot name a source row fails
validation. A plausible invented review count reaching a sales conversation costs more than
a crash.

**No two niches may claim the same Google type.** If type sets are disjoint, substitution
cannot happen by type at all. This held for every type except `caterer`, which two niches
claimed — and that single collision made every caterer look like a cloud kitchen.

**An exclusion is not free.** A disqualifying type is fatal, so two niches excluding each
other's types leave a dual-labelled place matching *nothing* — and Google dual-labels
precisely the businesses worth pitching. `{Massage spa, Ayurvedic clinic}` and
`{Caterer, Banquet hall}` were both invisible until this was fixed.

## Approach

| Stage | Source |
|---|---|
| Discover | SearchAPI Google Maps — website, phone, reviews, rating, `popular_times` |
| Verify no website | SearchAPI `website` field + Firecrawl search |
| Buying intent | Meta Ad Library public page |
| Audience depth | Instagram, via the operator's own browser session |
| Score | Deterministic Python; the LLM never produces a score |
| Output | Excel and Google Sheets, with operator verdicts synced back |

## Architecture

Three planes with separate lifetimes and failure modes:

- **Control** — `goals → runs → tasks → steps`, event-sourced and resumable
- **Data** — typed, provenanced, append-only; every claim traces to a source row
- **Memory** — scratchpad, run summaries, pgvector semantic recall

Postgres is the single source of truth. `SKIP LOCKED` is the queue.

## Boundaries

Instagram enrichment runs through Chrome MCP attached to the operator's existing
authenticated browser profile. The system stores no credentials, automates no login, and
creates no accounts. Access is read-only — no follows, likes, comments, or messages
originate from the agent. Visits are serialized, human-paced, and capped per day; any
block halts the worker rather than retrying into it.
