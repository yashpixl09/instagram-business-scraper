// Lead Engine control panel. Vanilla JS, no build step -- this is served directly by the
// same FastAPI process as the API it calls, so "run the server" is the whole deployment.

const state = {
  niches: [],
  selectedNiches: new Set(),
  runs: [],
  leads: [],
};

// --- small helpers -------------------------------------------------------------------

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = data && data.error ? data.error.message : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data;
}

function toast(message, kind) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast" + (kind ? " " + kind : "");
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 6000);
}

function fmt(value, fallback) {
  return value === null || value === undefined || value === "" ? (fallback ?? "—") : value;
}

// --- tabs ------------------------------------------------------------------------------

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "leads") refreshRunSelects().then(loadLeads);
      if (btn.dataset.tab === "enrich") refreshRunSelects();
    });
  });
}

// --- dashboard ---------------------------------------------------------------------

async function loadDashboard() {
  try {
    const status = await api("GET", "/api/config/status");
    document.getElementById("stat-db").textContent = status.database_configured ? "connected" : "not set";
    document.getElementById("stat-db").className = "card-value " + (status.database_configured ? "good" : "bad");

    const remaining = status.search_budget.remaining;
    document.getElementById("stat-budget").textContent = remaining === null ? "unknown" : remaining;

    document.getElementById("stat-llm").textContent = status.llm_configured ? "yes" : "no (deterministic only)";
    document.getElementById("stat-llm").className = "card-value " + (status.llm_configured ? "good" : "");

    document.getElementById("stat-sheets").textContent = status.sheets_configured ? "yes" : "no";
    document.getElementById("stat-sheets").className = "card-value " + (status.sheets_configured ? "good" : "bad");

    const providerList = document.getElementById("provider-list");
    providerList.innerHTML = "";
    Object.entries(status.providers).forEach(([name, on]) => {
      const chip = document.createElement("span");
      chip.className = "chip " + (on ? "on" : "off");
      chip.textContent = name + (on ? " ✓" : " ✗");
      providerList.appendChild(chip);
    });
  } catch (err) {
    toast("Could not load status: " + err.message, "error");
  }

  try {
    const { runs } = await api("GET", "/api/runs?limit=15");
    state.runs = runs;
    const tbody = document.querySelector("#runs-table tbody");
    tbody.innerHTML = "";
    runs.forEach((run) => {
      const tr = document.createElement("tr");
      const stats = run.stats || {};
      tr.innerHTML = `
        <td>${fmt(run.started_at, "").toString().replace("T", " ").slice(0, 19)}</td>
        <td>${run.status}</td>
        <td>${fmt(stats.new_businesses, "-")}</td>
        <td>${fmt(stats.searches_spent, "-")}</td>
        <td>${fmt(stats.scored, "-")}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    toast("Could not load runs: " + err.message, "error");
  }
}

// --- new run -------------------------------------------------------------------------

async function loadNiches() {
  try {
    const { niches } = await api("GET", "/api/niches");
    state.niches = niches;
    const box = document.getElementById("niche-checkboxes");
    box.innerHTML = "";
    niches.forEach((niche) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = niche.label;
      chip.dataset.id = niche.id;
      chip.addEventListener("click", () => {
        if (state.selectedNiches.has(niche.id)) {
          state.selectedNiches.delete(niche.id);
          chip.classList.remove("selected");
        } else {
          state.selectedNiches.add(niche.id);
          chip.classList.add("selected");
        }
      });
      box.appendChild(chip);
    });
  } catch (err) {
    toast("Could not load niches: " + err.message, "error");
  }
}

function readRunForm() {
  const form = document.getElementById("run-form");
  const data = new FormData(form);
  const areas = (data.get("areas") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (state.selectedNiches.size === 0) {
    throw new Error("Pick at least one niche.");
  }
  return {
    location: {
      city: data.get("city"),
      state: data.get("state") || null,
      country: data.get("country") || null,
      areas,
    },
    niches: Array.from(state.selectedNiches),
    limit: Number(data.get("limit")) || 10,
    use_ai: form.elements["use_ai"].checked,
  };
}

function showResult(elId, obj) {
  const el = document.getElementById(elId);
  el.hidden = false;
  el.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

async function previewCost() {
  try {
    const spec = readRunForm();
    const plan = await api("POST", "/api/search/plan", spec);
    showResult(
      "run-result",
      `Planned searches: ${plan.planned_searches}\n` +
        `Search budget remaining: ${plan.search_budget_remaining ?? "unknown"}\n` +
        `Locations: ${plan.locations.map((l) => l.label).join(", ")}`
    );
  } catch (err) {
    toast(err.message, "error");
  }
}

async function runNow() {
  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const spec = readRunForm();
    const run = await api("POST", "/api/search/run", spec);
    const stats = run.stats || {};
    let summary =
      `Run ${run.id}\nStatus: ${run.status}\n` +
      `New businesses: ${fmt(stats.new_businesses)}\n` +
      `Searches spent: ${fmt(stats.searches_spent)}\n` +
      `Budget remaining: ${fmt(stats.search_budget_remaining, "unknown")}`;
    showResult("run-result", summary);
    toast("Run complete.", "success");

    const form = document.getElementById("run-form");
    if (form.elements["enrich"].checked && stats.new_businesses > 0) {
      showResult("run-result", summary + "\n\nEnriching…");
      const enrichReport = await api("POST", "/api/enrich", {
        run_id: run.id,
        use_ai: form.elements["use_ai"].checked,
      });
      summary +=
        `\n\nEnriched: ${enrichReport.enriched}, rescored: ${enrichReport.scored}, ` +
        `automation offers: ${enrichReport.offers_detected}, outreach drafted: ${enrichReport.outreach_written}`;
      showResult("run-result", summary);
    }

    if (form.elements["sync_sheets"].checked) {
      showResult("run-result", summary + "\n\nSyncing to Google Sheets…");
      const sync = await api("POST", "/api/sync-sheets", {});
      summary +=
        `\n\nSheets sync: ${sync.rows_updated} updated, ${sync.rows_appended} appended, ` +
        `${sync.verdicts_read} verdicts read back`;
      showResult("run-result", summary);
    }

    loadDashboard();
  } catch (err) {
    toast(err.message, "error");
    showResult("run-result", "Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run now";
  }
}

// --- leads --------------------------------------------------------------------------

async function refreshRunSelects() {
  try {
    const { runs } = await api("GET", "/api/runs?limit=50");
    state.runs = runs;
    ["leads-run-select", "enrich-run-select"].forEach((id) => {
      const select = document.getElementById(id);
      const current = select.value;
      select.innerHTML = "";
      runs.forEach((run) => {
        const opt = document.createElement("option");
        opt.value = run.id;
        const when = fmt(run.started_at, "").toString().replace("T", " ").slice(0, 19);
        opt.textContent = `${when} — ${run.status} (${(run.stats || {}).new_businesses ?? 0} new)`;
        select.appendChild(opt);
      });
      if (current) select.value = current;
    });
  } catch (err) {
    toast("Could not load runs: " + err.message, "error");
  }
}

function bandClass(band) {
  return "band-" + (band || "unknown");
}

async function loadLeads() {
  const runId = document.getElementById("leads-run-select").value;
  if (!runId) return;
  try {
    const { leads } = await api("GET", `/api/leads?run_id=${runId}&limit=200`);
    state.leads = leads;
    const tbody = document.querySelector("#leads-table tbody");
    tbody.innerHTML = "";
    leads.forEach((lead) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${lead.name}</td>
        <td>${fmt(lead.niche_id)}</td>
        <td>${fmt(lead.score_total)}</td>
        <td class="${bandClass(lead.audience_band)}">${fmt(lead.audience_band)}</td>
        <td>${fmt(lead.my_verdict, "-")}</td>
      `;
      tr.addEventListener("click", () => showLeadDetail(lead.id));
      tbody.appendChild(tr);
    });
    if (!leads.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="muted">No leads in this run.</td></tr>';
    }
  } catch (err) {
    toast("Could not load leads: " + err.message, "error");
  }
}

async function showLeadDetail(leadId) {
  const panel = document.getElementById("lead-detail");
  panel.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const lead = await api("GET", `/api/leads/${leadId}`);
    const signals = (lead.signals || []).join(", ");
    const offers = (lead.automation_opportunities || []).join(", ");
    const reachLabels = { mobile: "📱 Mobile", instagram: "📷 Instagram DM", phone: "☎️ Phone (type unknown)", email: "✉️ Email" };
    panel.innerHTML = `
      <h3>${lead.name}</h3>
      <p class="muted">${fmt(lead.niche_id)} · ${fmt(lead.city)}${lead.search_area ? " / " + lead.search_area : ""}</p>

      <div class="field"><div class="field-label">Best way to reach out</div>
        <div class="field-value">${lead.best_reach_channel ? `<strong>${reachLabels[lead.best_reach_channel] || lead.best_reach_channel}</strong>: ${lead.best_reach_value}\n${lead.best_reach_note || ""}` : "nothing found yet"}</div></div>

      <div class="field"><div class="field-label">Score / Band</div>
        <div class="field-value">${fmt(lead.score_total)} / <span class="${bandClass(lead.audience_band)}">${fmt(lead.audience_band)}</span></div></div>

      <div class="field"><div class="field-label">Contact</div>
        <div class="field-value">${fmt(lead.phone)}${lead.website ? "\n" + lead.website : ""}${lead.contact_name ? "\n" + lead.contact_name + (lead.contact_role ? " (" + lead.contact_role + ")" : "") : ""}</div></div>

      <div class="field"><div class="field-label">Signals</div><div class="field-value">${fmt(signals, "none yet")}</div></div>

      <div class="field"><div class="field-label">AI summary</div><div class="field-value">${fmt(lead.ai_summary, "not enriched yet")}</div></div>

      <div class="field"><div class="field-label">Website pitch</div><div class="field-value">${fmt(lead.website_pitch, "not drafted yet")}</div></div>

      <div class="field"><div class="field-label">Automation opportunities</div><div class="field-value">${fmt(offers, "none (needs a high/medium verdict)")}</div></div>
      ${lead.automation_pitch ? `<div class="field"><div class="field-label">Automation pitch</div><div class="field-value">${lead.automation_pitch}</div></div>` : ""}

      <div class="field"><div class="field-label">Verdict</div>
        <select id="verdict-select">
          ${["", "high", "medium", "low", "skip"].map((v) => `<option value="${v}" ${lead.my_verdict === v ? "selected" : ""}>${v || "(none)"}</option>`).join("")}
        </select>
      </div>
      <div class="field"><div class="field-label">Notes</div>
        <textarea id="verdict-notes">${fmt(lead.notes, "")}</textarea>
      </div>
      <button id="btn-save-verdict" class="btn-primary" type="button">Save verdict</button>
    `;
    document.getElementById("btn-save-verdict").addEventListener("click", () => saveVerdict(leadId));
  } catch (err) {
    panel.innerHTML = `<p class="muted">Could not load: ${err.message}</p>`;
  }
}

async function saveVerdict(leadId) {
  const my_verdict = document.getElementById("verdict-select").value || null;
  const notes = document.getElementById("verdict-notes").value;
  if (!my_verdict && !notes) {
    toast("Pick a verdict or write a note first.", "error");
    return;
  }
  try {
    await fetch(`/api/leads/${leadId}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ my_verdict, notes }),
    });
    toast("Verdict saved.", "success");
    loadLeads();
  } catch (err) {
    toast("Could not save: " + err.message, "error");
  }
}

// --- enrich --------------------------------------------------------------------------

async function enrichRun() {
  const runId = document.getElementById("enrich-run-select").value;
  if (!runId) { toast("Pick a run first.", "error"); return; }
  const btn = document.getElementById("btn-enrich");
  btn.disabled = true;
  btn.textContent = "Enriching…";
  try {
    const report = await api("POST", "/api/enrich", {
      run_id: runId,
      use_ai: document.getElementById("enrich-use-ai").checked,
    });
    showResult(
      "enrich-result",
      `Enriched: ${report.enriched}\nRescored: ${report.scored}\n` +
        `Automation offers detected: ${report.offers_detected}\nOutreach drafted: ${report.outreach_written}\n\n` +
        `Provider usage: ${JSON.stringify(report.usage, null, 2)}`
    );
    toast("Enrichment complete.", "success");
  } catch (err) {
    showResult("enrich-result", "Error: " + err.message);
    toast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Enrich this run";
  }
}

// --- sync ---------------------------------------------------------------------------

async function syncSheets() {
  const btn = document.getElementById("btn-sync");
  btn.disabled = true;
  btn.textContent = "Syncing…";
  try {
    const result = await api("POST", "/api/sync-sheets", {});
    showResult(
      "sync-result",
      `Operator verdicts read from the sheet: ${result.verdicts_read}\n` +
        `Master rows updated: ${result.rows_updated}\nMaster rows appended: ${result.rows_appended}`
    );
    toast("Sheets sync complete.", "success");
  } catch (err) {
    showResult("sync-result", "Error: " + err.message);
    toast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Sync now";
  }
}

// --- wire up -------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadDashboard();
  loadNiches();
  document.getElementById("btn-preview").addEventListener("click", previewCost);
  document.getElementById("btn-run").addEventListener("click", runNow);
  document.getElementById("btn-refresh-leads").addEventListener("click", loadLeads);
  document.getElementById("leads-run-select").addEventListener("change", loadLeads);
  document.getElementById("btn-enrich").addEventListener("click", enrichRun);
  document.getElementById("btn-sync").addEventListener("click", syncSheets);
});
