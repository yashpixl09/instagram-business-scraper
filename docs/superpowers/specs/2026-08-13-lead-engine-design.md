# Lead Engine Design

Durable agent runtime that finds local businesses without an effective web presence,
qualifies them against real audience evidence, and produces an operator-ready pipeline.

## Goal

Produce a small number of high-confidence leads per run, each carrying enough verified
evidence that the operator can walk in, call, or DM without further research.

Quality over volume. Twenty leads worth contacting beat five hundred that rot in a sheet.

## Non-Goals

- Bulk lead lists. Volume accumulates across runs, not within one.
- Automated outreach. The system prepares; the operator sends.
- Account creation or login automation on any platform.

## Scope Decisions

| Decision | Rationale |
|---|---|
| Drop Geoapify, use SearchAPI Google Maps | Google's website field is reliable where OSM's is absent; also supplies reviews, rating, histogram, and `popular_times` in the same call |
| Postgres, not SQLite | Agent state, RAG embeddings, and durable queues share one consistency domain |
| No Redis | Postgres `SKIP LOCKED` is sufficient at this volume; a second store adds failure modes without benefit |
| Python + FastAPI | Extends the existing package; replaces the hand-rolled `BaseHTTPRequestHandler` |
| pgvector + HNSW | Production-grade below ~10M vectors. Expected ceiling is ~200k |
| Instagram via operator's own browser session | See Boundaries |

## Boundaries

Instagram enrichment runs through Chrome MCP attached to the operator's existing,
already-authenticated Chrome profile. The system stores no credentials, automates no
login, and creates no accounts. Access is read-only: no follows, likes, comments, or
messages originate from the agent.

Rate discipline is a correctness requirement, not a courtesy: profile visits are
serialized through a single worker, human-paced, and capped per day. Any block or
checkpoint halts the worker and flags the run rather than retrying into it.

Proxy rotation and fingerprint spoofing are out of scope.

## Architecture

Three planes with separate lifetimes and separate failure modes.

```
CONTROL   goals -> runs -> tasks -> steps        event-sourced, resumable
DATA      businesses, enrichments, scores        typed, provenanced, append-only
MEMORY    working, episodic, semantic            scratchpad, summaries, pgvector
```

### Topology

| Service | Role | Concurrency |
|---|---|---|
| `postgres` | pgvector, single source of truth | — |
| `api` | FastAPI | — |
| `worker-discovery` | SearchAPI sweeps | 4 |
| `worker-enrich` | Firecrawl, Ad Library, LLM | 8 |
| `worker-browser` | Chrome MCP → Instagram | 1, hard limit |
| `scheduler` | cron → materializes runs | 1 |

`worker-browser` is pinned to one process. One browser session, one identity, serialized.

## Data Sources

| Source | Provides | Cost |
|---|---|---|
| SearchAPI Google Maps | website, domain, phone, address, type, rating, reviews, `reviews_histogram`, `popular_times`, hours, `place_id`, coordinates | $4/1K searches, 20 results per search |
| Firecrawl search | website confirmation, Instagram handle, owner name | 2 credits / 10 results |
| Firecrawl scrape | site quality grading, Ad Library page | 1 credit / page |
| Chrome MCP → Instagram | followers, following, posts, bio, bio link, verified, recent-post likes and comments | Rate-limited, not metered |
| Meta Ad Library (public page) | whether the business currently buys ads | 1 Firecrawl credit |

Instagram post views and reach are owner-only metrics and are unavailable through any
source. Engagement rate derived from likes and comments replaces them, and is a better
quality signal than raw reach.

The Ad Library API covers only political and social-issue ads globally (plus all ads in
UK/EU). Indian commercial ads are visible on the public page only, hence the scrape path.

## Geographic Model

```python
@dataclass(frozen=True)
class GeoScope:
    city:    str                       # required
    country: str | None = None
    state:   str | None = None
    areas:   tuple[str, ...] = ()
```

`country` and `state` disambiguate; they do not narrow. `country` maps to SearchAPI `gl`.
`city` anchors geocoding. `areas` are the coverage strategy.

One `GeoScope` fans out to N searches — one per area, or one city-wide if no areas given.
Google Maps returns 20 results per call biased to the center point, so a single city-level
search on a 40km metro sees only the core. Area-level searches are how coverage is achieved.

| Precision | Radius | `ll` form |
|---|---|---|
| `area` | 3,000 m | `@12.9784,77.6408,3000` |
| `city` | 15,000 m | `@12.9716,77.5946,15000` |

Resolution composes most-specific-first, geocodes, and caches the result keyed on the
composed string. **Unresolvable locations abort the run.** Silent fallback to a city
center would run a day of outreach against the wrong neighborhood undetected. This mirrors
the existing rule that unsupported niches are rejected rather than substituted.

## Niche Model

Twenty-four profiles. The prototype's eleven — cafe, bakery, cake shop, salon, spa,
boutique, manufacturer, cloud kitchen, home decor, fitness/gym, tutor/class — carry forward
with their calibrated demand and budget bases, qualification terms, strictness and offer
lines intact. Thirteen are added for businesses that plausibly lack a good site, have
budget, and have an obvious automation to pitch: catering, dental clinic, clinic,
veterinary, auto service, preschool, driving school, photographer, event planner, interior
designer, real estate, travel agency, professional services.

SearchAPI returns Google's display labels rather than a hierarchy, so `categories` splits
in two: `queries` drives the search, `include_types` / `exclude_types` /
`include_suffixes` / `exclude_suffixes` decide what qualifies. Slugifying a display label
reconstructs the `type_id` SearchAPI omits.

Three rules govern qualification:

**Precedence** is exact include, exact exclude, suffix exclude, suffix include, neutral.
Exact-include-first is the per-slug exception that carves one type out of a broad suffix
rule without disabling the rule.

**A disqualifier is fatal**, before the strict and name-evidence gates run. `allow_name_only`
— set for cloud kitchen and manufacturer, where Google has no faithful type — relaxes the
type gate but can never bypass an exclusion.

**No two niches may claim the same type.** This is the structural guarantee: if type sets
are disjoint, substitution cannot happen by type at all. It is asserted directly rather
than inferred from the cross product.

Exclusions carry a cost that is easy to miss. Because a disqualifier is fatal, two niches
excluding each other's types leave a dual-labelled place matching *nothing*, and Google
dual-labels precisely the businesses worth pitching — the Kerala-style Ayurveda centre, the
caterer who also runs a banquet hall. So an exclusion is justified only when the type would
otherwise qualify, or when its presence proves a different business rather than a related
service the same business offers. Excluding an already-neutral type buys no precision and
orphans every dual-labelled place.

## Scoring

Scoring is deterministic Python. The LLM never produces a score.

Existing four-bucket model retained: demand /30, website gap /25, budget /25,
reachability /20.

`audience_size` is a separate, objective classification — the agent reports it, the
operator judges potential.

| Band | Threshold |
|---|---|
| `small` | <100 reviews and <2k followers |
| `medium` | 100–500 reviews or 2k–15k followers |
| `large` | >500 reviews or >15k followers |

Modifiers: engagement rate (likes ÷ followers) promotes or demotes one band;
dense `popular_times` promotes. Thresholds are per-niche calibrated and configurable —
they are a starting point pending field validation.

Scoring runs twice per business, because Instagram data arrives after the first pass.
The first pass scores on Google evidence alone and writes a row with
`scorer_version = 'google-only'`. After Instagram enrichment, a second pass writes a new
row incorporating followers and engagement. Both rows persist; the latest by `scored_at`
is authoritative. A business whose Instagram enrichment fails keeps its Google-only score
and is marked as such in the sheet, never left unscored.

## Contacts and Decision-Makers

A lead without a person to talk to is not actionable. Contact discovery is a first-class
stage, not an afterthought, and a business may have several contacts at differing
confidence.

Sources, in descending reliability:

| Source | Signal | Confidence |
|---|---|---|
| `review_reply` | Owner replies to Google reviews, frequently signed | 0.8 |
| `website` | About or Contact page scraped by Firecrawl | 0.8 |
| `ig_bio` | Owner name and email commonly present in Instagram bios | 0.6 |
| `directory` | Justdial, IndiaMART listings carry a named contact person | 0.6 |
| `search` | Firecrawl query for owner, founder, or proprietor | 0.4 |

SearchAPI's `review_results` field is the cheapest source and arrives free with discovery —
it is checked before any paid lookup is issued.

Contacts are extracted under the same rule as every other fact: a validated schema or a
failure. An extractor that cannot find a name records nothing. It never guesses one, since
a fabricated owner name is worse than a blank field the moment the operator opens with it.

Email is captured at both levels — `businesses.email` for a general inbox,
`contacts.email` for a named person — because cold email is a target channel and a generic
address supports a different message than a named one.

## Automation Opportunities

The second offer. Once a business qualifies, the system identifies which of its workflows
are worth automating, so the pitch moves from "you need a website" to a specific,
evidenced proposal.

Detection is rule-based, not generative — the same discipline as scoring. Opportunities
derive from observable gaps crossed with niche, and each fires only when its required
signals are present in `enrichments`.

A static catalog in `automations.py` mirrors the `NicheProfile` pattern already in the
codebase:

```python
@dataclass(frozen=True)
class AutomationOffer:
    id: str
    label: str
    niches: tuple[str, ...]
    required_signals: tuple[str, ...]
    pitch_line: str
    est_hours_saved_weekly: float
```

Representative entries:

| Opportunity | Fires when | Applies to |
|---|---|---|
| Order intake | No website ordering, active social presence | cafe, bakery, cake shop, cloud kitchen |
| Appointment booking + reminders | No booking link, appointment-driven niche | salon, spa, fitness/gym |
| Review response | High review count, owner replies absent or sparse | all |
| Lead capture | Runs Meta ads, no landing page | all |
| Quotation handling | Enquiry-driven, no structured form | manufacturer, home decor |
| Enrolment and batch scheduling | Class-based niche, manual enquiry flow | tutor/class |
| Catalog and WhatsApp checkout | Instagram-only catalog, DM-based ordering | boutique, bakery, home decor |

The LLM writes the pitch prose over detected opportunities; it does not decide which
opportunities exist. `trigger_signals` and `evidence` record why each fired, so every claim
in an automation pitch traces to a source row exactly as scoring claims do.

Only businesses the operator has marked `high` or `medium` in `verdicts` receive automation
pitch generation. This is the one place a tier gate applies, because the pitch is expensive
to generate and worthless before the operator has judged the lead worth pursuing.

## Agent Runtime

### Loop levels

```
Goal    persistent, cron-scheduled
 └ Run   one execution, resumable
    └ Task  leased, retried, idempotent
       └ Step  checkpointed before and after
```

Every step appends to `events` before execution and after completion. Resume replays the
log and skips terminal steps. A crash 40 minutes into a browser session resumes at the
last completed step.

Three recovery modes: **retry** (re-execute, keep history), **replay** (reset history),
**fork** (reset from step N).

### Queue

`FOR UPDATE SKIP LOCKED` with leases and an expiry reaper.

`available_at` serves double duty: exponential backoff and rate gating. The browser worker
sets `available_at = now() + 6s` on dequeue. The daily cap is a scheduler policy writing
future timestamps. No sleep loops, no in-memory limiter to lose on restart.

### Deduplication

Dedup lives at the tool boundary, never in the payload. `discover()` filters against
`businesses.place_id` before returning, so the agent only ever sees businesses it has not
seen. Token cost is zero and constant regardless of corpus size.

Coverage awareness comes from summaries, not lists:
`coverage(geo)` → `"Indiranagar/salon: 47 found, swept 3d ago, 4 new last run"`.

### Search Cells

Dedup prevents duplicate processing, not duplicate fetching. Google Maps returns the same
ranked results for the same query point on every call, so without a moving cursor the
second run filters all twenty results away and spends a credit for nothing.

Page cursors alone are insufficient: Google Maps caps at roughly 100–120 results per query
point. Discovery therefore advances through a four-dimensional space.

| Axis | Mechanism | Yields |
|---|---|---|
| `page` | `page=1,2,3…` | ~100 per query point |
| `tile` | Area subdivided into overlapping 1km circles | Fresh budget per tile |
| `query_variant` | `niches.py` `name_queries` | Different strings surface different businesses |
| `radius` | Tighter radius | Surfaces businesses wide searches bury |

A 3km area hex-packed into ~13 1km tiles raises addressable results from ~100 to ~1,300.

```sql
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
  UNIQUE (goal_id, niche_id, tile_lat, tile_lng, query_variant)
);
CREATE INDEX ON search_cells (goal_id, status, new_yield DESC);
```

Exhaustion is measured rather than predicted:

```
>= 5 new results   advance next_page, keep hot
1-4 new results    advance, deprioritize
0 new results      strike
2 strikes          status = 'exhausted'
```

Discovery selects the highest-yield unexhausted cell, so sweeps self-organize toward
productive ground without manual tuning.

Exhaustion expires after 30 days: the cell resets to `pending` with `next_page = 1`.
Yield will be low, but re-sweeping page 1 is the only path by which newly-opened
businesses enter the pipeline.

### Negative Caching

Firecrawl performs per-business lookups rather than sweeps, so it has no cross-turn
repetition problem. Its waste is the inverse: re-paying to rediscover absence.

Misses back off; successes keep the normal TTL.

| Consecutive misses | Next retry |
|---|---|
| 1 | 30d |
| 2 | 90d |
| 3 | 180d |
| 4+ | never, manual reset only |

### Budget Split

`discover_new` and `refresh` are separate task types with separate budgets, defaulting to
70/30. Without the split, a maturing database lets refresh consume every credit and
discovery stops silently — the pipeline appears busy while finding nothing new.

### Freshness

| Source | TTL |
|---|---|
| `google_maps` | 7d |
| `ad_library` | 14d |
| `instagram` | 30d |
| `firecrawl_web` | 30d |

Re-discovery of a known business with fresh enrichments bumps `last_seen_at` and consumes
no budget. `idem_key` enforces this at the database level.

## Context Engineering

**The agent never holds a lead record.** It holds identifiers plus a one-line descriptor
(~25 tokens each). Full detail arrives through `get_business(id)` only when a task touches
that lead.

**Sub-agent isolation.** A single Instagram profile visit yields 30–50k tokens of DOM and
accessibility markup. Extraction runs in an isolated sub-agent with its own context and a
cheap model, returning ~150 tokens of validated JSON. The markup never reaches the main
loop. The same pattern applies to site grading and the Ad Library check.

**Extractors do not generate prose.** They return a validated schema or fail. Validation
failure retries once, then records `status='error'`. No code path converts a missing value
into an invented one. Prose generation occurs only at the pitch stage, over existing rows,
carrying evidence identifiers.

**Compaction** triggers at ~60% window occupancy or every 25 tasks. Preserves decisions,
open issues, error patterns, and running counts; discards raw tool output, successful
extraction payloads, and navigation transcripts. Writes to `run_summaries` and
reinitializes from summary plus `agent_notes`. Discarded detail remains in `events`.

## LLM Router

```
groq/llama-3.3-70b     30 RPM · 14.4k RPD · free · tool use on all models
  └─429/5xx─> gemini-2.5-flash
      └─429─> nvidia nim
          └─> deterministic template   cannot fail
```

Circuit breaker per provider: `closed → open` after 3 consecutive failures,
`half-open` probe after 60s. Token buckets persist in Postgres so a restart does not reset
rate accounting into an immediate 429.

Routing by task class: structured extraction and classification to Groq; pitch prose to
the best available provider.

Responses cache on `prompt_hash`. A re-run after a crash re-spends nothing on completed
work.

The deterministic terminal fallback is an architectural guarantee, not an error path:
with every provider unavailable, a run still completes with a full sheet.

## Schema

```sql
CREATE EXTENSION vector;

-- CONTROL -----------------------------------------------------------

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
CREATE INDEX ON events (run_id, seq);

-- DATA --------------------------------------------------------------

CREATE TABLE businesses (
  id               uuid PRIMARY KEY,
  place_id         text UNIQUE,
  name             text NOT NULL,
  niche_id         text NOT NULL,
  country          text,
  state            text,
  city             text NOT NULL,
  search_area      text,
  address          text,
  lat              double precision,
  lng              double precision,
  phone            text,
  email            text,
  website          text,
  instagram_handle text,
  facebook_url     text,
  first_seen_at    timestamptz NOT NULL DEFAULT now(),
  last_seen_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE contacts (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  name        text,
  role        text,                   -- owner|manager|marketing|unknown
  phone       text,
  email       text,
  source      text NOT NULL,          -- review_reply|ig_bio|website|search|directory
  source_url  text,
  confidence  real NOT NULL DEFAULT 0.5,
  found_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON contacts (business_id, confidence DESC);

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

CREATE TABLE enrichments (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  source      text NOT NULL,
  status      text NOT NULL,
  data        jsonb NOT NULL,
  source_url  text,
  fetched_at  timestamptz NOT NULL DEFAULT now(),
  run_id      uuid
);
CREATE INDEX ON enrichments (business_id, source, fetched_at DESC);

CREATE TABLE scores (
  id             bigserial PRIMARY KEY,
  business_id    uuid NOT NULL REFERENCES businesses(id),
  scorer_version text NOT NULL,
  audience_size  text NOT NULL,
  total          int,
  demand         int,
  website_gap    int,
  budget         int,
  reachability   int,
  signals        jsonb NOT NULL,
  evidence       jsonb NOT NULL,
  scored_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE verdicts (
  business_id  uuid PRIMARY KEY REFERENCES businesses(id),
  my_verdict   text,
  notes        text,
  contacted_on date,
  channel      text,
  outcome      text,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE outreach (
  id          bigserial PRIMARY KEY,
  business_id uuid NOT NULL REFERENCES businesses(id),
  kind        text NOT NULL,          -- website_pitch | automation_pitch
  channel     text NOT NULL,          -- visit | call | email | dm
  body        text NOT NULL,
  evidence    jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- MEMORY ------------------------------------------------------------

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
CREATE UNIQUE INDEX ON llm_calls (prompt_hash) WHERE status = 'ok';
```

### Design notes

`enrichments` is append-only, making review and follower counts time series. A salon
moving 180 → 340 reviews in two months is growing — a buying signal no snapshot reveals.

`scores.evidence` maps every claim to the `enrichment_id` it derives from. Generated
claims that cannot name a source row fail validation. This makes the existing
"do not invent review counts" prompt instruction structurally enforceable.

`verdicts` has no agent write path. Operator judgments survive every run.

`contacts` and `automation_opportunities` both carry `evidence` or `source_url`, extending
traceability to the two fields most likely to embarrass the operator if fabricated: a
person's name and a claim about how their business runs.

## Outreach Generation

Four channels, four artifacts. A walk-in and a cold email are not the same message, and a
single body text reused across channels reads as a template in at least three of them.

| Channel | Artifact | Shape |
|---|---|---|
| `visit` | Talking points, not a script | Three openers keyed to observed evidence, plus the strongest single number to lead with |
| `call` | Opener plus objection handling | 20-second opener, two likely objections with responses |
| `email` | Subject and body | Subject references a specific observation; body under 120 words |
| `dm` | Single message | Under 60 words, no greeting block, one question |

Every artifact is generated over validated rows and carries `evidence`. Generation is
grounded on concrete observations — review count, peak hours, absent website, active ad
spend — because a pitch that cites a real number is the entire advantage over a generic
approach, and a pitch that cites a wrong number destroys the meeting.

Two pitch kinds share this structure: `website_pitch` for the initial offer and
`automation_pitch` for qualified leads, drawing on `automation_opportunities`.

Generation falls back to deterministic templates when every LLM provider is unavailable.
The templates are weaker but factually grounded, since they interpolate the same validated
rows.

## Outputs

One view builder, two writers.

```
businesses + enrichments + scores + verdicts
   └─> view builder ─┬─> XLSX (openpyxl)
                     └─> Google Sheets (upsert)
```

Sheet columns:

| Group | Columns |
|---|---|
| Identity | `business_name`, `niche`, `business_type` |
| Geography | `country`, `state`, `city`, `search_area`, `address`, `lat`, `lng` |
| Contact | `phone`, `email`, `website`, `instagram_handle`, `facebook_url` |
| People | `contact_name`, `contact_role`, `contact_phone`, `contact_email`, `contact_source` |
| Evidence | `reviews`, `rating`, `followers`, `engagement_rate`, `runs_ads`, `peak_hours` |
| Agent output | `total_score`, `audience_size`, `signals`, `ai_summary`, `website_pitch` |
| Automation | `automation_opportunities`, `automation_pitch` |
| Operator | `my_verdict`, `notes`, `contacted_on`, `channel`, `outcome` |

Every evidence column carries its source URL in a cell comment, so a number can be checked
in one click before it is spoken aloud to a business owner.

Sort by `search_area` — it groups a day of fieldwork into one neighborhood rather than a
list scattered across a 40km city.

### Google Sheets layout

- **`Master`** — living pipeline, upserted by `place_id`. The only editable tab. Operator
  columns live here and sync back to `verdicts` before each run.
- **`YYYY-MM-DD`** — daily read-only snapshots: found, changed, band movement.

Sync order is read Master → write `verdicts` → rebuild views, so operator edits always win
over agent output. Splitting editable from snapshot avoids the question of which tab holds
a given note.

Sheets API quotas and batch semantics require research before this phase.

## Testing

No test may contact a live external service. Every provider sits behind an interface with
a fake implementation driven by recorded fixtures. A test suite that depends on SearchAPI
being reachable fails on a train, and a test suite that depends on Instagram being
reachable is a liability.

| Layer | Approach |
|---|---|
| Scoring, banding, niche matching | Pure unit tests. The existing suite carries over |
| Geographic resolution | Fixture-driven; asserts unresolvable input aborts rather than falls back |
| Search cells | Asserts cursor advance, yield-measured exhaustion, and 30-day reset |
| Providers | Recorded JSON fixtures. Malformed and empty payloads are cases, not accidents |
| Queue | Integration tests against real Postgres: lease expiry, concurrent `SKIP LOCKED` claims, idempotency under duplicate enqueue |
| Resume | Kill a run mid-flight, restart, assert completed steps are skipped and no work repeats |
| LLM router | Fake providers returning 429 and 5xx; asserts circuit opens, chain advances, deterministic fallback terminates |
| Extractors | Saved HTML fixtures. Asserts malformed input yields `status='error'`, never a fabricated value |
| Browser worker | Fixture HTML only. Never live, in any test, under any flag |

The negative cases carry the weight. The failure this system is most exposed to is not a
crash but a plausible fabricated number reaching a sales conversation, so every extractor
has an explicit test that bad input produces an error rather than a guess.

Verification commands stay as they are today:

```powershell
python -m pytest -q
python -m compileall -q lead_finder
```

## Cost Model

At 20–30 leads per run:

| Item | Per run | Monthly (daily runs) |
|---|---|---|
| SearchAPI | 1–2 searches | <$0.25 |
| Firecrawl | ~120 credits | $16 (Hobby, 5,000 credits) |
| LLM | Free tier | $0 |
| **Total** | | **~$16** |

Small batches remove the enrichment backlog entirely: 20–30 Instagram visits fall inside
the daily ceiling, so every lead is fully enriched within its own run. The durable queue
remains as the crash-recovery mechanism, without backlog pressure.

## Sequencing

Schema is built complete from the start — migrations are expensive, tables are cheap.
Workers land incrementally, each shipping usable output.

0. Port from the prototype: `niches.py`, `scoring.py`, `dedupe.py`, `models.py` and their
   tests. These are tested and correct; only the provider, storage, and API layers are
   being replaced
1. Postgres, migrations, FastAPI skeleton, task queue with leases
2. Discovery via SearchAPI, geographic resolution, search cells, dedup at tool boundary
3. Deterministic scoring + Excel export — **first usable sheet**
4. Firecrawl enrichment: website verification, site grading, Ad Library
5. Contact discovery: review replies, website About pages, directories
6. LLM router with circuit breaker + per-channel outreach generation
7. Chrome MCP Instagram worker, isolated sub-agent, rate-gated
8. Automation opportunity detection + automation pitch
9. Full agent loop: events, compaction, memory, resume
10. Google Sheets adapter with verdict sync

Phases 0–3 produce a sheet the operator can work from. Everything after deepens leads that
are already actionable, so no phase leaves the system unusable.

## Deferred

Multi-agent orchestration with a master agent over shared memory. The substrate exists —
sub-agent isolation, `memories.scope`, the shared event log, task-type dispatch — so this
is additive rather than a rewrite. Revisit once the single-agent pipeline runs in production.

## Open Questions

- Audience band thresholds need field validation against real Bangalore results. The
  numbers in this spec are a starting point, not a finding.
- Niche set and area list for the first runs are not yet fixed.
- Perplexity Sonar (~$0.01/lookup) as a contact-discovery fallback is deferred until the
  free sources — review replies, website About pages, directories — have a measured hit
  rate. Paying for a lookup that the free path already resolves is the wrong default.
- Google Sheets API quotas and batch semantics require research before phase 10.
- `est_hours_saved_weekly` in the automations catalog is currently an estimate per
  opportunity type. It should become niche-specific once real conversations produce
  evidence.
