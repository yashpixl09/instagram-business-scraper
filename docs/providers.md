# Providers

What the live vendor docs actually say, recorded so nobody has to re-derive it from a
half-remembered SDK. Each section names the date the docs were read; if you are reading
this months later, re-check before trusting a number.

---

## TinyFish — Search and Fetch

**Docs read: 2026-08-13**, from <https://docs.tinyfish.ai> (`/search-api`,
`/search-api/reference`, `/fetch-api`, `/fetch-api/reference`, `/authentication`,
`/error-codes`, `/rate-limits`) and the vendor's own cookbook,
<https://github.com/tinyfish-io/tinyfish-cookbook>.

Client: `lead_engine/providers/tinyfish.py`. Tests: `tests/test_tinyfish.py`.

### Why this is the primary enrichment path

Search and Fetch are free on every plan and consume **zero credits** — they are capped by
request rate alone. The 500 one-time signup credits apply only to the Agent and Browser
APIs, which this project does not use. That makes TinyFish the default enrichment route and
leaves metered Firecrawl calls for jobs TinyFish genuinely cannot do.

### Endpoints

| | Search | Fetch |
|---|---|---|
| Method + URL | `GET https://api.search.tinyfish.ai` | `POST https://api.fetch.tinyfish.ai` |
| Auth | `X-API-Key: <raw key>` | `X-API-Key: <raw key>` |
| Payload | query string | JSON body |
| Credits | none | none |

Auth is the **raw key in an `X-API-Key` header** — not `Authorization: Bearer`. The docs'
own curl example is `-H "X-API-Key: $TINYFISH_API_KEY"`. Keys come from
<https://agent.tinyfish.ai/api-keys>; ours lives in `TINYFISH_API_KEY`.

Note the two APIs sit on **separate hosts**, not on paths under one host. The
`/rate-limits` page refers to them as `/v1/search` and `/v1/fetch`, which is the only place
that phrasing appears; every example everywhere else uses the subdomains above, so that is
what the client uses.

### Rate limits — per API key, per minute

| Plan | Search (queries/min) | Fetch (URLs/min) |
|---|---|---|
| Free / Pay As You Go | 30 | 150 |
| Starter | 60 | 300 |
| Pro | 120 | 600 |

Fetch is metered in **URLs, not requests** — one request carrying 10 URLs spends 10.
A 429 may carry `Retry-After` and `X-RateLimit-Limit`; responses also carry `X-Request-ID`,
worth quoting to support.

The client enforces both caps itself with a token bucket (`TinyFishClient.search_limiter`,
`.fetch_limiter`), refilling continuously rather than on a minute boundary: a burst of 30
searches goes out instantly, the 31st waits two seconds. Raise the caps for a paid plan
through the constructor (`searches_per_minute=`, `fetch_urls_per_minute=`) rather than
editing the module constants.

### Search request

`query` is the only required parameter. Documented optionals:

`purpose` (≤2000 chars, explains intent and improves ranking), `location`, `language`,
`include_domains` / `exclude_domains` (comma-separated), `recency_minutes` (1–5,256,000),
`after_date` / `before_date` (`YYYY-MM-DD`), `domain_type` (`web` | `news` |
`research_paper`), `pub_year_min` / `pub_year_max` (research papers only), `page` (0–10).

Constraints the docs call out: `recency_minutes` cannot be combined with the date filters;
`after_date` must be ≤ `before_date`; date and recency filters are unsupported for
`domain_type=research_paper`.

> **There is no result-count parameter.** No `limit`, `num_results`, `max_results`, or
> `count` appears anywhere in the reference, and the default page size is not documented
> either. `TinyFishClient.search(query, limit=...)` therefore truncates client-side —
> `limit` is an upper bound on one page, never a promise of that many rows. Pass `page=`
> to go further. A test asserts no invented count parameter is sent, because an
> unrecognised one would be ignored silently and look like it worked.

### Search response

```json
{"query": "...", "results": [{"position": 1, "site_name": "...", "title": "...",
 "snippet": "...", "url": "...", "date": "..."}], "total_results": 12, "page": 0}
```

News results add `publisher`; research-paper results add `authors`, `venue`, `year`,
`cited_by_count`, `pdf_url`. The client maps rows onto `SearchResult` and requires only
`url` — a result we cannot fetch is not a result — treating the rest as optional metadata.

### Fetch request

```json
{"urls": ["https://..."], "format": "markdown"}
```

`urls` takes **1–10 URLs per request** and per-URL failures do not block the others.
Other documented fields: `format` (`markdown` default, `html`, `json`), `purpose`, `links`,
`image_links`, `ttl` (cache freshness in seconds; omit to accept any cached entry, `0`
forces a live fetch), `per_url_timeout_ms`, `if_none_match` / `if_modified_since` (single
URL only), `include_etag_and_last_modified`, `include_selectors` (≤20 CSS selectors) and
`exclude_selectors` (excludes applied first, then includes scope what remains).

The client fixes `format` to `markdown` and does not expose it: under `format=json` the
`text` field becomes an object rather than a string, and everything downstream of this
client wants text.

### Fetch response

```json
{"results": [{"url": "...", "final_url": "...", "title": "...", "description": "...",
  "language": "...", "author": "...", "published_date": "...", "format": "markdown",
  "text": "...", "links": [], "image_links": [], "not_modified": false, "etag": "...",
  "last_modified": "...", "unmatched_selectors": [], "latency_ms": 812}],
 "errors": [{"url": "...", "error": "...", "status": 404,
  "unmatched_selectors": [], "candidate_selectors": []}]}
```

HTML, JSON and PDF are supported; binary files come back as entries in `errors[]`.

> **Per-URL failures arrive inside a `200`.** The `status` on an `errors[]` entry describes
> the **site being scraped**, not TinyFish. The client therefore does *not* route it
> through the provider status ladder: a shop whose website answers 403 would otherwise be
> reported as `provider_auth_failed` — "TinyFish rejected the configured key" — and would
> trip the provider circuit breaker over something entirely outside TinyFish. Every per-URL
> failure maps to `provider_bad_response`. `fetch_many()` simply omits the URLs that
> failed; `fetch()` raises, because it has nothing to return.

### Error codes

Provider-level failures use `{"error": {"code": "...", "message": "..."}}`. Documented
statuses: `400 INVALID_INPUT`, `401 MISSING_API_KEY|INVALID_API_KEY|UNAUTHORIZED`,
`402 INSUFFICIENT_CREDITS`, `403 FORBIDDEN`, `404 NOT_FOUND`, `409 RETRY_REQUIRED`,
`422` (invalid CSS selector, Fetch only), `429 RATE_LIMIT_EXCEEDED`, `500 INTERNAL_ERROR`.

How the client maps them onto `lead_engine.providers.errors`:

| Upstream | Code | API status | Retryable |
|---|---|---|---|
| 429 | `provider_rate_limited` | 429 | yes |
| 401, 403 | `provider_auth_failed` | 503 | no |
| 5xx | `provider_unavailable` | 503 | yes |
| other 4xx (400, 402, 404, 409, 422) | `provider_bad_response` | 502 | yes |
| unparseable / unexpected body | `provider_bad_response` | 502 | yes |
| timeout, connection refused, DNS | `provider_unavailable` | 503 | yes |

**Open question for whoever owns the taxonomy.** The 5xx row diverges from
`errors.from_status`, which sends everything that is not 429 or 401/403 — 5xx included — to
`provider_bad_response`. This client follows its own brief instead, because a TinyFish 500
is upstream being down rather than upstream sending unparseable JSON. Both codes are
retryable, so nothing retries differently; the visible difference is the API status, 503
rather than 502. `test_the_brief_mapping_is_pinned_where_it_diverges_from_errors_from_status`
pins the current choice so the divergence is on the record. Reconciling it is a one-line
change in `_status_error`.

### Message discipline

No error from this client carries the response body, the requested URL, or the API key.
Two habits keep it that way, and both are load-bearing:

1. Messages are static prose with at most an HTTP status interpolated. `errors._safe_message`
   silently substitutes the generic default for any message `redact()` would alter, so an
   interpolated body would not leak — it would *vanish*, taking the only diagnostic with it.
2. Transport failures are re-raised `from None`. httpx exceptions carry `.request.url`, and
   for providers that authenticate by query string that is a key in a logfile. The cost is
   real: the original httpx exception is gone from the traceback.

`tests/test_tinyfish.py` plants a body sentinel that `redact()` would *not* catch, so the
leak assertions test this client rather than the shared safety net.

### Verification

`tests/test_tinyfish.py` runs entirely on `httpx.MockTransport` and a fake clock: no
sockets, no real seconds. It was checked by mutation — 23 deliberate defects introduced
into the client one at a time, 22 of which the suite caught, including a wrong auth header,
a removed rate limiter, a body interpolated into an error message, and each status
mis-mapped. The survivor replaced the "JSON body is not an object" guard with `payload = {}`,
which reaches the very next guard and raises the same `provider_bad_response`; it is an
equivalent mutant, not a gap.
