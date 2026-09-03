import * as api from "./api.js";
import { createCharts, updateCharts, assignColor, computeSeries, MAX_OVERLAY_RUNS } from "./charts.js";

const charts = createCharts();

let capabilities = null;
let scope = "per_packet";
let liveRunId = null;
let pollTimer = null;
let pollIntervalMs = 2000;
const selectedHistoryIds = new Set();
const binsCache = new Map(); // runId -> bins[]
let latestRunsList = [];

const $ = (id) => document.getElementById(id);

// ── Per-method "Advanced" toggle defaults ────────────────────────────────────────────
// Mirrors Server/utils/helper_function.py's detect_cfo_sto sentinel table exactly (see
// that function's docstring): the classical/no-hint down-chirp scan can legitimately
// land on either of the preamble's two down-chirps (a ~50/50, SNR-independent coin flip
// between two structurally-identical signals), and sync_payload_offset_adjust is what
// disambiguates that — required (True) without a hint, unnecessary and slightly risky
// (True) with one, where the down-chirp search is already deterministic. Applying ONE
// fixed default regardless of method silently costs ~1 payload symbol on about half of
// all conventional-path packets (the scan picks the wrong down-chirp, the payload crop
// lands a full symbol late, and demod's own off-by-one handling quietly reports 9
// comparable symbols instead of 10 rather than an outright failure) without ever
// showing up in the CFO/STO charts, since those estimates stay correct either way.
const METHOD_DEFAULTS = {
  conventional: {
    precorrect_cfo_frac: false, use_fractional_sto: false,
    verify_network_id: true, sync_payload_offset_adjust: true, sto_method: "v1",
  },
  calora: {
    precorrect_cfo_frac: true, use_fractional_sto: true,
    verify_network_id: false, sync_payload_offset_adjust: false, sto_method: "v3",
  },
};
// Sourced only from user interaction with these specific controls (see wireTouchTracking
// below) — never touched by applyMethodDefaults itself, so switching methods repeatedly
// doesn't self-poison this set.
const touchedAdvanced = new Set();

function applyMethodDefaults(method) {
  const defaults = METHOD_DEFAULTS[method] || METHOD_DEFAULTS.conventional;
  for (const [id, value] of Object.entries(defaults)) {
    if (touchedAdvanced.has(id)) continue; // respect a deliberate user override
    const el = $(id);
    if (el.type === "checkbox") el.checked = value;
    else el.value = value;
  }
}

function wireTouchTracking() {
  for (const id of Object.keys(METHOD_DEFAULTS.conventional)) {
    $(id).addEventListener("change", () => touchedAdvanced.add(id));
  }
}

// ── Capabilities: checkpoint dropdown + NeLoRa gate ─────────────────────────────────
async function loadCapabilities() {
  capabilities = await api.getCapabilities();

  const sel = $("calora_checkpoint");
  sel.innerHTML = "";
  for (const c of capabilities.checkpoints) {
    if (!c.selectable && c.filename !== capabilities.checkpoints[0]?.filename) {
      // still show disabled options, per plan: visible-but-disabled, never hidden
    }
    const opt = document.createElement("option");
    opt.value = c.filename;
    let text = c.label ? `${c.label} — ${c.filename}` : `${c.filename} (undocumented)`;
    if (!c.selectable) text += " [unavailable]";
    opt.textContent = text;
    opt.disabled = !c.selectable;
    if (c.is_current_server_default) opt.selected = true;
    sel.appendChild(opt);
  }

  const neloraOpt = $("nelora-option");
  neloraOpt.disabled = !capabilities.nelora_available;
  neloraOpt.textContent = capabilities.nelora_available
    ? "NeLoRa (neural)"
    : "NeLoRa (neural) — not yet validated";
}

$("preamble_detection_method").addEventListener("change", (e) => {
  $("calora-checkpoint-row").hidden = e.target.value !== "calora";
  applyMethodDefaults(e.target.value);
});

// ── Packets-per-bin live hint ────────────────────────────────────────────────────────
function updatePacketsPerBinHint() {
  const n = Number($("n_packets").value) || 0;
  const lo = Number($("snr_lo").value), hi = Number($("snr_hi").value);
  const nBins = Math.max(1, hi - lo + 1);
  $("packets-per-bin-hint").textContent = hi >= lo
    ? `~${(n / nBins).toFixed(1)} packets/SNR bin on average (${nBins} bins)`
    : "SNR range: lo must be <= hi";
}
["n_packets", "snr_lo", "snr_hi"].forEach((id) =>
  $(id).addEventListener("input", updatePacketsPerBinHint));

// ── Config collection ────────────────────────────────────────────────────────────────
function collectConfig() {
  const method = $("preamble_detection_method").value;
  return {
    client: {
      sf: 7,
      bw: Number($("bw").value),
      fs: Number($("fs").value),
      n_packets: Number($("n_packets").value),
      payload_len: 10,
      cfo_hz_range: [Number($("cfo_lo").value), Number($("cfo_hi").value)],
      sto_samp_range: [Number($("sto_lo").value), Number($("sto_hi").value)],
      snr_db_range: [Number($("snr_lo").value), Number($("snr_hi").value)],
      seed: Number($("seed").value),
    },
    server: {
      preamble_detection_method: method,
      calora_checkpoint: method === "calora" ? $("calora_checkpoint").value : null,
      symbol_decoding_method: $("symbol_decoding_method").value,
      toggles: {
        apply_cfo_correction: $("apply_cfo_correction").checked,
        apply_sto_correction: $("apply_sto_correction").checked,
        precorrect_cfo_frac: $("precorrect_cfo_frac").checked,
        use_fractional_sto: $("use_fractional_sto").checked,
        verify_network_id: $("verify_network_id").checked,
        sync_payload_offset_adjust: $("sync_payload_offset_adjust").checked,
        sto_method: $("sto_method").value,
        downchirp_search_radius: Number($("downchirp_search_radius").value),
      },
    },
  };
}

function fillFormFromConfig(clientConfig, serverConfig) {
  $("bw").value = clientConfig.bw;
  $("fs").value = clientConfig.fs;
  $("n_packets").value = clientConfig.n_packets;
  $("cfo_lo").value = clientConfig.cfo_hz_range[0];
  $("cfo_hi").value = clientConfig.cfo_hz_range[1];
  $("sto_lo").value = clientConfig.sto_samp_range[0];
  $("sto_hi").value = clientConfig.sto_samp_range[1];
  $("snr_lo").value = clientConfig.snr_db_range[0];
  $("snr_hi").value = clientConfig.snr_db_range[1];
  $("seed").value = clientConfig.seed;

  $("preamble_detection_method").value = serverConfig.preamble_detection_method;
  $("calora-checkpoint-row").hidden = serverConfig.preamble_detection_method !== "calora";
  if (serverConfig.calora_checkpoint) $("calora_checkpoint").value = serverConfig.calora_checkpoint;
  $("symbol_decoding_method").value = serverConfig.symbol_decoding_method;

  // Fall back to the SAME run's method-appropriate defaults (never a hardcoded CALoRa-
  // shaped guess) for any field the loaded config happens to be missing — then pin every
  // advanced toggle as "touched" so this run's exact settings survive if the user then
  // flips preamble_detection_method to compare one run against a same-everything-else
  // variant, which is the whole point of "Use config" (see METHOD_DEFAULTS' docstring).
  const t = serverConfig.toggles || {};
  const fallback = METHOD_DEFAULTS[serverConfig.preamble_detection_method] || METHOD_DEFAULTS.conventional;
  $("apply_cfo_correction").checked = t.apply_cfo_correction !== false;
  $("apply_sto_correction").checked = t.apply_sto_correction !== false;
  $("precorrect_cfo_frac").checked = t.precorrect_cfo_frac ?? fallback.precorrect_cfo_frac;
  $("use_fractional_sto").checked = t.use_fractional_sto ?? fallback.use_fractional_sto;
  $("verify_network_id").checked = t.verify_network_id ?? fallback.verify_network_id;
  $("sync_payload_offset_adjust").checked = t.sync_payload_offset_adjust ?? fallback.sync_payload_offset_adjust;
  $("sto_method").value = t.sto_method || fallback.sto_method;
  $("downchirp_search_radius").value = t.downchirp_search_radius ?? fallback.downchirp_search_radius;
  Object.keys(METHOD_DEFAULTS.conventional).forEach((id) => touchedAdvanced.add(id));
  updatePacketsPerBinHint();
}

// ── Run lifecycle ─────────────────────────────────────────────────────────────────────
$("config-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("form-error").hidden = true;
  const config = collectConfig();
  try {
    const res = await api.startRun(config);
    liveRunId = res.run_id;
    assignColor(liveRunId);
    selectedHistoryIds.add(liveRunId);
    setFormEnabled(false);
    $("progress-meter").hidden = false;
    startPolling();
  } catch (err) {
    $("form-error").textContent = err.message;
    $("form-error").hidden = false;
  }
});

$("cancel-btn").addEventListener("click", async () => {
  if (liveRunId) await api.cancelRun(liveRunId).catch(() => {});
});

function setFormEnabled(enabled) {
  $("start-btn").hidden = !enabled;
  $("cancel-btn").hidden = enabled;
  Array.from($("config-form").elements).forEach((el) => {
    if (el.id !== "sf") el.disabled = !enabled;
  });
}

function startPolling() {
  stopPolling();
  poll();
  pollTimer = setInterval(poll, pollIntervalMs);
  document.addEventListener("visibilitychange", onVisibilityChange);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  document.removeEventListener("visibilitychange", onVisibilityChange);
}

function onVisibilityChange() {
  if (!liveRunId) return;
  pollIntervalMs = document.hidden ? 10000 : 2000;
  startPolling();
}

async function poll() {
  if (!liveRunId) return;
  let status;
  try {
    status = await api.getRunStatus(liveRunId);
  } catch {
    return; // transient network hiccup — try again next tick
  }

  binsCache.set(liveRunId, status.bins);
  renderProgress(status);
  renderChartsNow();

  if (["completed", "failed", "cancelled"].includes(status.status)) {
    stopPolling();
    setFormEnabled(true);
    $("progress-meter").hidden = true;
    liveRunId = null;
    await refreshHistory();
  }
}

function renderProgress(status) {
  const { packets_done, packets_total } = status.progress;
  const pct = packets_total ? (packets_done / packets_total) * 100 : 0;
  $("meter-fill").style.width = `${pct}%`;
  $("progress-text").textContent = `${packets_done} / ${packets_total} packets — ${status.status}`;
}

// ── History ───────────────────────────────────────────────────────────────────────────
function statusBadge(status) {
  return `<span class="status-badge ${status}">${status}</span>`;
}

function configSummary(clientConfig, serverConfig) {
  const method = serverConfig.preamble_detection_method === "calora"
    ? `CALoRa (${(serverConfig.calora_checkpoint || "?").replace(".pth", "")})`
    : "Conventional";
  const demod = serverConfig.symbol_decoding_method === "nelora" ? "NeLoRa" : "conv demod";
  const t = serverConfig.toggles || {};
  const corr = [t.apply_cfo_correction !== false ? "CFO" : null,
                t.apply_sto_correction !== false ? "STO" : null].filter(Boolean).join("+") || "no correction";
  return `${method} + ${demod} · ${corr} · SF${clientConfig.sf} ${clientConfig.bw / 1000}kHz@${clientConfig.fs / 1000}kHz · `
       + `SNR ${clientConfig.snr_db_range[0]}..${clientConfig.snr_db_range[1]}`;
}

async function refreshHistory() {
  const { runs } = await api.listRuns();
  latestRunsList = runs;

  // Auto-resume: if a run is live and we don't already know about it, adopt it.
  const runningRun = runs.find((r) => r.status === "running" || r.status === "queued");
  if (runningRun && !liveRunId) {
    liveRunId = runningRun.run_id;
    assignColor(liveRunId);
    selectedHistoryIds.add(liveRunId);
    setFormEnabled(false);
    $("progress-meter").hidden = false;
    startPolling();
  }

  renderHistoryTable(runs);
  renderChartsNow();
}

function renderHistoryTable(runs) {
  const tbody = $("history-body");
  tbody.innerHTML = "";
  for (const r of runs) {
    const tr = document.createElement("tr");
    const checked = selectedHistoryIds.has(r.run_id);
    const isLive = r.run_id === liveRunId;

    tr.innerHTML = `
      <td><input type="checkbox" class="row-select" ${checked ? "checked" : ""}
                 ${!checked && selectedHistoryIds.size >= MAX_OVERLAY_RUNS ? "disabled" : ""}></td>
      <td class="label-cell"><input type="text" value="${r.label.replace(/"/g, "&quot;")}"></td>
      <td>${r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
      <td>${configSummary(r.client_config, r.server_config)}</td>
      <td>${statusBadge(r.status)}${isLive ? " (live)" : ""}</td>
      <td class="row-actions">
        <button type="button" class="duplicate-btn">Use config</button>
        <button type="button" class="delete-btn">Delete</button>
      </td>`;

    tr.querySelector(".row-select").addEventListener("change", (e) => {
      if (e.target.checked) {
        if (selectedHistoryIds.size >= MAX_OVERLAY_RUNS) { e.target.checked = false; return; }
        selectedHistoryIds.add(r.run_id);
        assignColor(r.run_id);
        ensureBinsLoaded(r.run_id).then(renderChartsNow);
      } else {
        selectedHistoryIds.delete(r.run_id);
        renderChartsNow();
      }
      renderHistoryTable(latestRunsList); // refresh disabled-state of other checkboxes
    });

    tr.querySelector(".label-cell input").addEventListener("change", (e) => {
      api.renameRun(r.run_id, e.target.value).catch(() => {});
    });

    tr.querySelector(".duplicate-btn").addEventListener("click", () => {
      fillFormFromConfig(r.client_config, r.server_config);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    tr.querySelector(".delete-btn").addEventListener("click", async () => {
      await api.deleteRun(r.run_id).catch(() => {});
      selectedHistoryIds.delete(r.run_id);
      binsCache.delete(r.run_id);
      refreshHistory();
    });

    tbody.appendChild(tr);
  }
}

async function ensureBinsLoaded(runId) {
  if (binsCache.has(runId)) return;
  const detail = await api.getRunDetail(runId).catch(() => null);
  binsCache.set(runId, detail ? detail.bins : []);
}

// ── Chart + diagnostics + warning-banner rendering ──────────────────────────────────
function currentRunsToShow() {
  const runs = [];
  for (const id of selectedHistoryIds) {
    const meta = latestRunsList.find((r) => r.run_id === id);
    const bins = binsCache.get(id) || [];
    const isLive = id === liveRunId;
    let label = meta ? meta.label : id.slice(0, 8);
    if (isLive) {
      const total = bins.reduce((a, b) => a + b.n_total, 0);
      label += ` (running)`;
    }
    runs.push({ runId: id, label, color: assignColor(id), bins, dashed: isLive });
  }
  return runs;
}

function renderChartsNow() {
  const runs = currentRunsToShow();
  updateCharts(charts, runs, scope);
  renderDiagnostics(runs);
  renderWarningBanner(runs);
  renderRawDataTable(runs);
}

// ── Raw data table ────────────────────────────────────────────────────────────────────
// Tracks the dropdown's own choice across re-renders so a poll tick (which calls
// renderChartsNow every ~2s) doesn't reset what the user is looking at.
let rawDataSelectedRunId = null;

function renderRawDataTable(runs) {
  const select = $("raw-data-run-select");
  const previousChoice = rawDataSelectedRunId;

  select.innerHTML = "";
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.runId;
    opt.textContent = r.label;
    select.appendChild(opt);
  }

  const stillVisible = runs.some((r) => r.runId === previousChoice);
  rawDataSelectedRunId = stillVisible ? previousChoice : (runs[0]?.runId ?? null);
  if (rawDataSelectedRunId) select.value = rawDataSelectedRunId;

  const download = $("raw-data-download");
  const copyBtn = $("raw-data-copy-btn");
  const tbody = $("raw-data-body");
  tbody.innerHTML = "";

  const run = runs.find((r) => r.runId === rawDataSelectedRunId);
  if (!run) {
    select.hidden = true;
    download.hidden = true;
    copyBtn.hidden = true;
    tbody.innerHTML = `<tr><td colspan="12" id="raw-data-empty">No run selected — check a run below to see its numbers here.</td></tr>`;
    return;
  }
  select.hidden = download.hidden = copyBtn.hidden = false;
  download.href = `/api/runs/${run.runId}/export.csv`;

  const series = computeSeries(run.bins, scope);
  const sortedBins = [...run.bins].sort((a, b) => a.snr_db - b.snr_db);
  for (const b of sortedBins) {
    const cfoPct = series.cfo[b.snr_db];
    const stoPct = series.sto[b.snr_db];
    const symPct = series.symbol[b.snr_db];
    const symDenom = scope === "conditioned" ? b.symbol_total : b.symbol_total_sent;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${b.snr_db}</td>
      <td>${b.n_total}</td>
      <td class="${b.preamble_missed ? "fail-cell" : ""}">${b.preamble_missed}</td>
      <td class="${b.preamble_mislocated ? "fail-cell" : ""}">${b.preamble_mislocated ?? "—"}</td>
      <td class="${b.offset_fail ? "fail-cell" : ""}">${b.offset_fail}</td>
      <td class="${b.demod_fail ? "fail-cell" : ""}">${b.demod_fail}</td>
      <td>${cfoPct === null ? "—" : cfoPct.toFixed(1) + "%"}</td>
      <td>${b.cfo_mae_hz === null ? "—" : b.cfo_mae_hz.toFixed(2)}</td>
      <td>${stoPct === null ? "—" : stoPct.toFixed(1) + "%"}</td>
      <td>${b.sto_mae_samples === null ? "—" : b.sto_mae_samples.toFixed(3)}</td>
      <td>${b.symbol_correct}/${symDenom}</td>
      <td>${symPct === null ? "—" : symPct.toFixed(1) + "%"}</td>`;
    tbody.appendChild(tr);
  }
  if (sortedBins.length === 0) {
    tbody.innerHTML = `<tr><td colspan="12" id="raw-data-empty">No packets landed in any SNR bin yet.</td></tr>`;
  }
}

$("raw-data-run-select").addEventListener("change", (e) => {
  rawDataSelectedRunId = e.target.value;
  renderChartsNow();
});

$("raw-data-copy-btn").addEventListener("click", async () => {
  const table = $("raw-data-table");
  const rows = [...table.querySelectorAll("tr")].map((tr) =>
    [...tr.children].map((cell) => cell.textContent.trim()).join("\t"));
  try {
    await navigator.clipboard.writeText(rows.join("\n"));
    const btn = $("raw-data-copy-btn");
    const original = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = original; }, 1200);
  } catch {
    /* clipboard permission denied — the download link is the fallback */
  }
});

function renderDiagnostics(runs) {
  const tbody = $("diagnostics-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of runs) {
    const n = r.bins.reduce((a, b) => a + b.n_total, 0);
    // Mislocated counts as a preamble failure here too, so "Reached" really is the
    // number of packets the offset estimator got a usable starting point for.
    const missed = r.bins.reduce((a, b) => a + b.preamble_missed + (b.preamble_mislocated ?? 0), 0);
    const offFail = r.bins.reduce((a, b) => a + b.offset_fail, 0);
    const reached = n - missed - offFail;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.label}</td><td>${n}</td>`
      + `<td>${n ? (missed / n * 100).toFixed(1) : 0}%</td>`
      + `<td>${n ? (offFail / n * 100).toFixed(1) : 0}%</td>`
      + `<td>${reached}</td>`;
    tbody.appendChild(tr);
  }
}

function renderWarningBanner(runs) {
  const banner = $("warning-banner");
  if (scope !== "conditioned") { banner.hidden = true; return; }
  const bad = runs
    .map((r) => {
      const n = r.bins.reduce((a, b) => a + b.n_total, 0);
      const missed = r.bins.reduce((a, b) => a + b.preamble_missed + (b.preamble_mislocated ?? 0), 0);
      return { label: r.label, rate: n ? missed / n : 0 };
    })
    .filter((x) => x.rate > 0.15);

  if (bad.length === 0) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.textContent = bad
    .map((x) => `Conditioned numbers may be misleading for "${x.label}": only ${((1 - x.rate) * 100).toFixed(1)}% of packets reached the offset stage.`)
    .join(" ");
}

document.querySelectorAll(".scope-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".scope-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    scope = btn.dataset.scope;
    renderChartsNow();
  });
});

// ── Boot ──────────────────────────────────────────────────────────────────────────────
(async function init() {
  wireTouchTracking();
  applyMethodDefaults($("preamble_detection_method").value); // matches the <select>'s own default
  updatePacketsPerBinHint();
  await loadCapabilities().catch((e) => console.error("capabilities failed", e));
  await refreshHistory().catch((e) => console.error("history failed", e));
})();
