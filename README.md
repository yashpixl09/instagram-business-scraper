# Lead Engine

Durable agent runtime that finds local businesses without an effective web presence,
qualifies them against real audience evidence, and produces an operator-ready pipeline.

Quality over volume: a small number of high-confidence leads per run, each carrying enough
verified evidence to walk in, call, or DM without further research. Volume accumulates
across runs.

## Status

Design complete. Implementation not started.

See [`docs/superpowers/specs/2026-08-13-lead-engine-design.md`](docs/superpowers/specs/2026-08-13-lead-engine-design.md).

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
