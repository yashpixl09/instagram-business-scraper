# Lead Engine 🚀

A durable agent runtime that finds local businesses without an effective web presence, qualifies them against real audience evidence, enriches social & Instagram data via human-paced browser sessions, and produces an operator-ready sales pipeline with AI-generated outreach copy.

---

## 🌟 Features

- **Google Maps Discovery**: Finds local businesses by city and niche using SearchAPI.
- **Website Absence Verification**: Detects businesses operating without a site or with poor landing pages (via TinyFish & Firecrawl).
- **Deterministic Lead Scoring**: 0–100 score + tier banding (High, Medium, Low) calculated in Python (no LLM hallucinations in scores).
- **AI Sales Copy Generation**: Tailored outreach & automation pitches generated via LLM Router (Groq, Gemini, NVIDIA NIM).
- **Instagram Browser Enrichment**: Human-paced, read-only Instagram profile signal extraction using Chrome MCP attached to your logged-in Chrome profile.
- **Web Dashboard UI**: Clean, interactive Web Control Panel hosted directly on FastAPI (`/app/`).
- **Google Sheets & Excel Export**: Bi-directional sync with Google Sheets & custom formatted Excel exports.

---

## 📋 Prerequisites

Before running Lead Engine, ensure you have installed:
1. **Python 3.11 or higher** ([python.org](https://www.python.org/))
2. **Docker Desktop** ([docker.com](https://www.docker.com/)) — *Required for local Postgres database*
3. **Google Chrome Browser** — *Logged into your Instagram account (for Instagram profile enrichment)*

---

## 🛠️ Quick Start Guide

### Step 1: Open Terminal in Project Folder

Navigate to the unzipped project folder:
```powershell
cd instagram-business-scraper
```

### Step 2: Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Step 3: Set Up Environment Variables

Copy the template environment file to `.env`:
```powershell
cp env.example .env
```

Open `.env` in a text editor and fill in your API keys (e.g. `SEARCHAPI_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `TINYFISH_API_KEY`).

### Step 4: Start PostgreSQL Database

Start the local PostgreSQL 17 + pgvector container:
```powershell
docker compose up -d
```
*(Runs on host port `5433` to prevent collision with local Postgres installs).*

### Step 5: Run Database Migrations

Apply all SQL schema migrations:
```powershell
python -m lead_engine.db.migrate
```

### Step 6: Launch Web Server & Dashboard UI

Start the Uvicorn web server:
```powershell
uvicorn --factory lead_engine.api.app:create_app --reload --port 8000
```

Open your browser and navigate to:
👉 **[http://localhost:8000/app/](http://localhost:8000/app/)**

---

## 📷 Running Instagram Profile Enrichment

Instagram profile signal extraction (`worker-browser`) runs on your local machine using Chrome attached to your logged-in Instagram browser session:

1. Open Google Chrome on your computer and make sure you are signed into **Instagram**.
2. Run the local worker process:
   ```powershell
   python -m lead_engine.workers.runner
   ```
3. The worker connects to the task queue, safely navigates to Instagram profiles at human-paced rates (30s+ intervals), extracts profile signals (followers, bio, link), and records findings into Postgres.

---

## ☁️ Cloud Deployment (Railway)

Lead Engine is pre-configured with a multi-stage `Dockerfile`, `railway.json`, and automatic migration runners.

To deploy to Railway:
1. Connect your repository to Railway.
2. Provision a **PostgreSQL** database.
3. Link `DATABASE_URL` to your web service.
4. Railway will automatically build the container, run database migrations, and host your live dashboard at `https://your-app.up.railway.app/app/`.

---

## 🧪 Running Tests

To verify your installation and run the unit test suite (1100+ tests):
```powershell
pytest tests/ -q
```
