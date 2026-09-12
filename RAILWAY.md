# Deploying Lead Engine to Railway

This guide walks you through deploying Lead Engine on [Railway](https://railway.app/).

## Overview

Lead Engine requires:
1. **Python FastAPI Service** (Web Server + API)
2. **PostgreSQL Database** (with `pgvector` extension)

The project is pre-configured with `railway.json`, `Dockerfile`, and automatic database migration support.

---

## Step 1: Deploy PostgreSQL Database on Railway

1. Go to your [Railway Dashboard](https://railway.app/dashboard).
2. Click **+ New Project** (or open an existing project).
3. Select **Provision PostgreSQL**.
4. Railway will create a PostgreSQL database instance and automatically set the `DATABASE_URL` variable.

> **Note on `pgvector`**: Railway's default PostgreSQL instance includes vector support.

---

## Step 2: Deploy Lead Engine Service from GitHub

1. In your Railway project, click **+ New** -> **GitHub Repo**.
2. Select **`instagram-business-scraper`** (or your repository name).
3. Select the branch you want to deploy (`phase-0-core` or `main`).
4. Railway will automatically detect the `Dockerfile` / `railway.json` and start the build.

---

## Step 3: Connect Database to Service

1. Click on your **Lead Engine** service in Railway.
2. Go to **Variables** tab.
3. Click **Add Reference Variable** or set:
   - `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (or reference the Postgres service's `DATABASE_URL`).

Lead Engine is pre-configured to detect `DATABASE_URL` automatically and execute all SQL migrations on startup (`python -m lead_engine.db.migrate`).

---

## Step 4: Configure Application Environment Variables

Under the **Variables** tab of your Lead Engine service, add any API keys you wish to use:

| Variable | Description |
|---|---|
| `SEARCHAPI_KEY` | (Optional) SearchAPI key for Google Maps discovery |
| `LEAD_ENGINE_SEARCH_BUDGET` | Budget allocation for SearchAPI key (default: 50) |
| `TINYFISH_API_KEY` | (Optional) TinyFish key for web enrichment |
| `FIRECRAWL_API_KEY` | (Optional) Firecrawl fallback key |
| `GROQ_API_KEY` | (Optional) Groq LLM API key |
| `GEMINI_API_KEY` | (Optional) Gemini LLM API key |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.5-flash`) |
| `NVIDIA_API_KEY` | (Optional) NVIDIA NIM API key |

---

## Step 5: Public Networking & Domain

1. Go to the **Settings** tab of the Lead Engine service.
2. Under **Networking**, click **Generate Domain** (e.g. `lead-engine-production.up.railway.app`).
3. Open `https://<your-domain>.up.railway.app/` in your browser.
4. You will be redirected to `/app/` where the interactive web control panel is hosted!
5. Health checks are served at `/health`.

---

## Deploying via Railway CLI (Alternative)

If you prefer using the command line:

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```
