import * as api from "./api.js";
import {
  createCharts,
  updateCharts,
  assignColor,
  assignMarker,
  getRunStyle,
  resolveSelectedStyles,
  setRunColor,
  setRunMarker,
  resetAllRunStyles,
  pruneRunStyles,
  computeSeries,
  MAX_OVERLAY_RUNS,
  exportSweepChartPng,
  createSingleFrameCharts,
  updateSingleFrameCharts,
  exportSingleFrameChartPng,
  createSyncExperimentCharts,
  updateSyncExperimentCharts,
  exportSyncChartPng,
} from "./charts.js";

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
    precorrect_cfo_frac: false,
    use_fractional_sto: true,
    verify_network_id: true, sync_payload_offset_adjust: true, sto_method: "v3",
  },
  calora: {
    precorrect_cfo_frac: true,
    use_fractional_sto: true,
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
  if (neloraOpt) {
    neloraOpt.disabled = !capabilities.nelora_available;
    neloraOpt.textContent = capabilities.nelora_available
      ? "NeLoRa (neural)"
      : "NeLoRa (neural) — not yet validated";
  }

  const loraseekOpt = $("loraseek-option");
  if (loraseekOpt) {
    loraseekOpt.disabled = !capabilities.loraseek_available;
    loraseekOpt.textContent = capabilities.loraseek_available
      ? "LoRaSeek (neural)"
      : "LoRaSeek (neural) — checkpoint unavailable";
  }

  const syncLsOpt = $("sync-loraseek-option");
  if (syncLsOpt) {
    syncLsOpt.disabled = !capabilities.loraseek_available;
  }
  const syncGlOpt = $("sync-gloriphy-option");
  if (syncGlOpt) {
    syncGlOpt.disabled = !capabilities.gloriphy_available;
  }

  const initialSf = Number($("sf").value) || 7;
  updateNeloraCheckpointsForSf(initialSf);
  updateGloriphyCheckpointsForSf(initialSf);
  updateLoraseekCheckpointsForSf(initialSf);
  syncSfCompatibility(initialSf);
}

function updateNeloraCheckpointsForSf(sf) {
  const nsel = $("nelora_checkpoint");
  const neloraOpt = $("nelora-option");
  if (!nsel || !capabilities) return;

  nsel.innerHTML = "";
  const matching = (capabilities.nelora_checkpoints || []).filter((ck) => ck.sf === sf);
  for (const ck of matching) {
    const opt = document.createElement("option");
    opt.value = ck.id;
    opt.textContent = ck.label;
    if (ck.is_default) opt.selected = true;
    nsel.appendChild(opt);
  }

  if (neloraOpt) {
    if (!capabilities.nelora_available) {
      neloraOpt.disabled = true;
      neloraOpt.textContent = "NeLoRa (neural) — checkpoints unavailable";
    } else if (matching.length === 0) {
      neloraOpt.disabled = true;
      neloraOpt.textContent = `NeLoRa (neural) — SF${sf} checkpoint not ready`;
    } else {
      neloraOpt.disabled = false;
      neloraOpt.textContent = "NeLoRa (neural)";
    }
  }
}

function updateLoraseekCheckpointsForSf(sf) {
  const lsel = $("loraseek_checkpoint");
  const loraseekOpt = $("loraseek-option");
  if (!lsel || !capabilities) return;

  lsel.innerHTML = "";
  const matching = (capabilities.loraseek_checkpoints || []).filter((ck) => ck.sf === sf);
  for (const ck of matching) {
    const opt = document.createElement("option");
    opt.value = ck.filename;
    opt.textContent = ck.label;
    if (ck.is_default) opt.selected = true;
    lsel.appendChild(opt);
  }

  if (loraseekOpt) {
    if (!capabilities.loraseek_available) {
      loraseekOpt.disabled = true;
      loraseekOpt.textContent = "LoRaSeek (neural) — checkpoints unavailable";
    } else if (matching.length === 0) {
      loraseekOpt.disabled = true;
      loraseekOpt.textContent = `LoRaSeek (neural) — SF${sf} checkpoint not ready`;
    } else {
      loraseekOpt.disabled = false;
      loraseekOpt.textContent = "LoRaSeek (neural)";
    }
  }
}

function updateGloriphyCheckpointsForSf(sf) {
  const gsel = $("gloriphy_checkpoint");
  const gloriphyOpt = $("gloriphy-option");
  if (!gsel || !capabilities) return;

  gsel.innerHTML = "";
  const matching = (capabilities.gloriphy_checkpoints || []).filter((ck) => ck.sf === sf);
  for (const ck of matching) {
    const opt = document.createElement("option");
    opt.value = ck.id;
    opt.textContent = ck.label;
    if (ck.is_default) opt.selected = true;
    gsel.appendChild(opt);
  }

  if (gloriphyOpt) {
    if (!capabilities.gloriphy_available) {
      gloriphyOpt.disabled = true;
      gloriphyOpt.textContent = "GLoRiPHY (neural) — checkpoints unavailable";
    } else if (matching.length === 0) {
      gloriphyOpt.disabled = true;
      gloriphyOpt.textContent = `GLoRiPHY (neural) — SF${sf} checkpoint not ready`;
    } else {
      gloriphyOpt.disabled = false;
      gloriphyOpt.textContent = "GLoRiPHY (neural)";
    }
  }
}

function syncSfCompatibility(sf) {
  const caloraOpt = $("preamble_detection_method")?.querySelector('option[value="calora"]');
  if (caloraOpt) {
    caloraOpt.disabled = (sf !== 7);
    if (sf !== 7 && $("preamble_detection_method").value === "calora") {
      $("preamble_detection_method").value = "conventional";
      $("calora-checkpoint-row").hidden = true;
      applyMethodDefaults("conventional");
    }
  }

  const loraseekOpt = $("loraseek-option");
  if (loraseekOpt && capabilities?.loraseek_available) {
    loraseekOpt.disabled = (sf !== 7 && sf !== 8);
    if (sf !== 7 && sf !== 8 && $("symbol_decoding_method").value === "loraseek") {
      $("symbol_decoding_method").value = "conventional";
      $("loraseek-checkpoint-row").hidden = true;
    }
  }

  const neloraOpt = $("nelora-option");
  if (neloraOpt && capabilities?.nelora_available) {
    neloraOpt.disabled = (sf !== 7 && sf !== 8);
    if (sf !== 7 && sf !== 8 && $("symbol_decoding_method").value === "nelora") {
      $("symbol_decoding_method").value = "conventional";
      $("nelora-checkpoint-row").hidden = true;
    }
  }

  const currentDemod = $("symbol_decoding_method").value;
  if (currentDemod === "gloriphy") {
    const matching = (capabilities?.gloriphy_checkpoints || []).filter((ck) => ck.sf === sf);
    if (matching.length === 0) {
      $("symbol_decoding_method").value = "conventional";
      $("gloriphy-checkpoint-row").hidden = true;
    }
  }

  const syncSel = $("sync_denoiser");
  const syncLsOpt = $("sync-loraseek-option");
  if (syncLsOpt && capabilities?.loraseek_available) {
    const lsMatch = (capabilities?.loraseek_checkpoints || []).some((ck) => ck.sf === sf);
    syncLsOpt.disabled = !lsMatch;
    if (!lsMatch && syncSel && syncSel.value === "loraseek") {
      syncSel.value = "none";
    }
  }
  const syncGlOpt = $("sync-gloriphy-option");
  if (syncGlOpt && capabilities?.gloriphy_available) {
    const glMatch = (capabilities?.gloriphy_checkpoints || []).some((ck) => ck.sf === sf);
    syncGlOpt.disabled = !glMatch;
    if (!glMatch && syncSel && syncSel.value === "gloriphy") {
      syncSel.value = "none";
    }
  }
}

$("preamble_detection_method").addEventListener("change", (e) => {
  $("calora-checkpoint-row").hidden = e.target.value !== "calora";
  applyMethodDefaults(e.target.value);
});

$("symbol_decoding_method").addEventListener("change", (e) => {
  $("nelora-checkpoint-row").hidden = e.target.value !== "nelora";
  $("loraseek-checkpoint-row").hidden = e.target.value !== "loraseek";
  $("gloriphy-checkpoint-row").hidden = e.target.value !== "gloriphy";
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
      sf: Number($("sf").value) || 7,
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
      sync_denoiser: $("sync_denoiser")?.value || "none",
      symbol_decoding_method: $("symbol_decoding_method").value,
      nelora_checkpoint: $("symbol_decoding_method").value === "nelora" ? $("nelora_checkpoint")?.value : null,
      loraseek_checkpoint: $("symbol_decoding_method").value === "loraseek" ? $("loraseek_checkpoint").value : null,
      gloriphy_checkpoint: $("symbol_decoding_method").value === "gloriphy" ? $("gloriphy_checkpoint").value : null,
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
  if (clientConfig.sf) {
    $("sf").value = clientConfig.sf;
    updateNeloraCheckpointsForSf(clientConfig.sf);
    updateGloriphyCheckpointsForSf(clientConfig.sf);
    updateLoraseekCheckpointsForSf(clientConfig.sf);
    syncSfCompatibility(clientConfig.sf);
  }
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
  if ($("sync_denoiser")) $("sync_denoiser").value = serverConfig.sync_denoiser || "none";
  $("symbol_decoding_method").value = serverConfig.symbol_decoding_method;
  $("nelora-checkpoint-row").hidden = serverConfig.symbol_decoding_method !== "nelora";
  $("loraseek-checkpoint-row").hidden = serverConfig.symbol_decoding_method !== "loraseek";
  $("gloriphy-checkpoint-row").hidden = serverConfig.symbol_decoding_method !== "gloriphy";
  if (serverConfig.nelora_checkpoint && $("nelora_checkpoint")) $("nelora_checkpoint").value = serverConfig.nelora_checkpoint;
  if (serverConfig.loraseek_checkpoint) $("loraseek_checkpoint").value = serverConfig.loraseek_checkpoint;
  if (serverConfig.gloriphy_checkpoint) $("gloriphy_checkpoint").value = serverConfig.gloriphy_checkpoint;

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

export function datasetImpairmentInfo(clientConfig) {
  if (!clientConfig) return "Data: Clean";
  const cfo = clientConfig.cfo_hz_range || [0, 0];
  const sto = clientConfig.sto_samp_range || [0, 0];
  const hasCfo = (cfo[0] !== 0 || cfo[1] !== 0);
  const hasSto = (sto[0] !== 0 || sto[1] !== 0);
  if (hasCfo && hasSto) return "Data: Both CFO & STO";
  if (hasCfo) return "Data: CFO only";
  if (hasSto) return "Data: STO only";
  return "Data: Clean";
}

function configSummary(clientConfig, serverConfig) {
  if (!clientConfig) clientConfig = {};
  if (!serverConfig) serverConfig = {};

  const method = serverConfig.preamble_detection_method === "calora"
    ? `CALoRa (${(serverConfig.calora_checkpoint || "?").replace(".pth", "")})`
    : "Conventional";
  const syncDen = serverConfig.sync_denoiser || "none";
  const syncStr = syncDen === "loraseek" ? " + LoRaSeek Sync"
    : syncDen === "gloriphy" ? " + GLoRiPHY Sync" : "";
  let demod = "conv demod";
  if (serverConfig.symbol_decoding_method === "nelora") {
    const ckpt = serverConfig.nelora_checkpoint ? ` (${serverConfig.nelora_checkpoint})` : "";
    demod = `NeLoRa${ckpt}`;
  } else if (serverConfig.symbol_decoding_method === "loraseek") {
    const ckpt = (serverConfig.loraseek_checkpoint || "90000").replace(".pkl", "").replace("_LoRaSeek", "");
    demod = `LoRaSeek (${ckpt})`;
  } else if (serverConfig.symbol_decoding_method === "gloriphy") {
    const ckpt = (serverConfig.gloriphy_checkpoint || "default").split("/").pop().replace("best_conformer_jgj.pkl", "").replace(/^_|_$/g, "");
    demod = `GLoRiPHY (${ckpt})`;
  }
  const t = serverConfig.toggles || {};
  const corr = [t.apply_cfo_correction !== false ? "CFO" : null,
                t.apply_sto_correction !== false ? "STO" : null].filter(Boolean).join("+") || "no correction";
  const dataTag = datasetImpairmentInfo(clientConfig);
  const bwStr = clientConfig.bw ? ` ${clientConfig.bw / 1000}kHz` : "";
  const fsStr = clientConfig.fs ? `@${clientConfig.fs / 1000}kHz` : "";
  const snrStr = Array.isArray(clientConfig.snr_db_range) && clientConfig.snr_db_range.length >= 2
    ? ` · SNR ${clientConfig.snr_db_range[0]}..${clientConfig.snr_db_range[1]}`
    : "";
  return `${method}${syncStr} [${dataTag}] + ${demod} · ${corr} · SF${clientConfig.sf ?? 7}${bwStr}${fsStr}${snrStr}`;
}

async function refreshHistory() {
  const { runs } = await api.listRuns();
  latestRunsList = runs;
  pruneRunStyles(runs.map((r) => r.run_id));

  // Auto-resume: if a run is live and we don't already know about it, adopt it.
  const runningRun = runs.find((r) => r.status === "running" || r.status === "queued");
  if (runningRun && !liveRunId) {
    liveRunId = runningRun.run_id;
    selectedHistoryIds.add(liveRunId);
    setFormEnabled(false);
    $("progress-meter").hidden = false;
    startPolling();
  }

  renderHistoryTable(runs);
  renderChartsNow();
}

function getActiveResolvedStyles() {
  const selectedItems = [...selectedHistoryIds].map((id) => ({
    runId: id,
    meta: latestRunsList.find((r) => r.run_id === id) || null,
  }));
  return resolveSelectedStyles(selectedItems);
}

function renderHistoryTable(runs) {
  const tbody = $("history-body");
  tbody.innerHTML = "";
  const activeStyles = getActiveResolvedStyles();

  for (const r of runs) {
    const tr = document.createElement("tr");
    const checked = selectedHistoryIds.has(r.run_id);
    const isLive = r.run_id === liveRunId;
    const style = activeStyles.get(r.run_id) || getRunStyle(r.run_id, r);

    tr.innerHTML = `
      <td><input type="checkbox" class="row-select" ${checked ? "checked" : ""}
                 ${!checked && selectedHistoryIds.size >= MAX_OVERLAY_RUNS ? "disabled" : ""}></td>
      <td class="style-cell">
        <div class="style-controls">
          <input type="color" class="run-color-input" list="journal-color-presets"
                 value="${style.color}" title="Click to change line color">
          <select class="run-marker-select" title="Line marker shape">
            <option value="circle" ${style.marker === "circle" ? "selected" : ""}>●</option>
            <option value="rect" ${style.marker === "rect" ? "selected" : ""}>■</option>
            <option value="triangle" ${style.marker === "triangle" ? "selected" : ""}>▲</option>
            <option value="rectRot" ${style.marker === "rectRot" ? "selected" : ""}>◆</option>
          </select>
        </div>
      </td>
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
        ensureBinsLoaded(r.run_id).then(renderChartsNow);
      } else {
        selectedHistoryIds.delete(r.run_id);
        renderChartsNow();
      }
      renderHistoryTable(latestRunsList); // refresh disabled-state & active style badges
    });

    const colorInput = tr.querySelector(".run-color-input");
    const onColorChange = (e) => {
      setRunColor(r.run_id, e.target.value);
      renderChartsNow();
    };
    colorInput.addEventListener("input", onColorChange);
    colorInput.addEventListener("change", onColorChange);

    tr.querySelector(".run-marker-select").addEventListener("change", (e) => {
      setRunMarker(r.run_id, e.target.value);
      renderChartsNow();
    });

    tr.querySelector(".label-cell input").addEventListener("change", (e) => {
      r.label = e.target.value;
      api.renameRun(r.run_id, e.target.value).catch(() => {});
      renderChartsNow();
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

// Build a clean journal-ready legend label for Preamble & Demod Sweep runs, e.g.:
//   "[Data CFO & TO]; CALoRa Preamble detection + LoRaSeek demod, use CFO+STO correction, SF7 125kHz @1MHz"
// Preserves user-edited custom labels while simplifying all auto-generated legacy/new labels.
function formatSweepLegendLabel(meta, fallbackId) {
  if (!meta) return fallbackId.slice(0, 8);
  const rawLabel = (meta.label || "").trim();
  const isLegacyAuto =
    rawLabel === "" ||
    rawLabel.includes("[Data:") ||
    rawLabel.includes("125kHz@1000kHz") ||
    rawLabel.includes("conv demod") ||
    (rawLabel.includes("demod,") && rawLabel.includes("SNR "));

  if (!isLegacyAuto) {
    return rawLabel;
  }

  const cli = meta.client_config || {};
  const srv = meta.server_config || {};

  const cfo = cli.cfo_hz_range || [0, 0];
  const sto = cli.sto_samp_range || [0, 0];
  const hasCfo = (cfo[0] !== 0 || cfo[1] !== 0);
  const hasSto = (sto[0] !== 0 || sto[1] !== 0);
  let dataTag = "Data Clean";
  if (hasCfo && hasSto) dataTag = "Data CFO & TO";
  else if (hasCfo) dataTag = "Data CFO";
  else if (hasSto) dataTag = "Data TO";

  const preMethod = srv.preamble_detection_method === "calora"
    ? "CALoRa Preamble detection"
    : "Conventional Preamble detection";

  const syncDen = srv.sync_denoiser || "none";
  const syncPart = syncDen === "loraseek" ? " + LoRaSeek Sync"
    : syncDen === "gloriphy" ? " + GLoRiPHY Sync" : "";

  const demodMap = {
    nelora: "NeLoRa demod",
    loraseek: "LoRaSeek demod",
    gloriphy: "GLoRiPHY demod",
    conventional: "Conventional demod",
  };
  const demodLabel = demodMap[srv.symbol_decoding_method] || `${srv.symbol_decoding_method || "Conventional"} demod`;

  const t = srv.toggles || {};
  const corrBits = [];
  if (t.apply_cfo_correction !== false) corrBits.push("CFO");
  if (t.apply_sto_correction !== false) corrBits.push("STO");
  const corrLabel = corrBits.length ? `use ${corrBits.join("+")} correction` : "no correction";

  const sf = cli.sf ?? 7;
  const bw = cli.bw ?? 125000;
  const fs = cli.fs ?? 1000000;
  const bwStr = `${bw / 1000}kHz`;
  const fsStr = fs >= 1000000 ? `${fs / 1000000}MHz` : `${fs / 1000}kHz`;

  return `[${dataTag}]; ${preMethod}${syncPart} + ${demodLabel}, ${corrLabel}, SF${sf} ${bwStr} @${fsStr}`;
}

// ── Chart + diagnostics + warning-banner rendering ──────────────────────────────────
function currentRunsToShow() {
  const runs = [];
  const activeStyles = getActiveResolvedStyles();
  for (const id of selectedHistoryIds) {
    const meta = latestRunsList.find((r) => r.run_id === id);
    const bins = binsCache.get(id) || [];
    const isLive = id === liveRunId;
    let label = formatSweepLegendLabel(meta, id);
    if (isLive) {
      label += " (running)";
    }
    const style = activeStyles.get(id) || getRunStyle(id, meta);
    runs.push({
      runId: id,
      label,
      color: style.color,
      marker: style.marker,
      bins,
      dashed: isLive,
    });
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
      <td>${cfoPct === null ? "—" : cfoPct.toFixed(1) + "%"}</td>
      <td>${b.cfo_mae_hz === null ? "—" : b.cfo_mae_hz.toFixed(2)}</td>
      <td>${stoPct === null ? "—" : stoPct.toFixed(1) + "%"}</td>
      <td>${b.sto_mae_samples === null ? "—" : b.sto_mae_samples.toFixed(3)}</td>
      <td>${b.symbol_correct}/${symDenom}</td>
      <td>${symPct === null ? "—" : symPct.toFixed(1) + "%"}</td>`;
    tbody.appendChild(tr);
  }
  if (sortedBins.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" id="raw-data-empty">No packets landed in any SNR bin yet.</td></tr>`;
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
    const reached = n - missed;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.label}</td><td>${n}</td>`
      + `<td>${n ? (missed / n * 100).toFixed(1) : 0}%</td>`
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

document.querySelectorAll(".scope-btn[data-scope]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".scope-btn[data-scope]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    scope = btn.dataset.scope;
    renderChartsNow();
  });
});

function setupSweepExportControls() {
  $("reset-run-colors-btn")?.addEventListener("click", () => {
    resetAllRunStyles();
    renderHistoryTable(latestRunsList);
    renderChartsNow();
  });

  document.querySelectorAll(".sweep-export-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.chart;
      if (key) exportSweepChartPng(charts, key);
    });
  });

  document.querySelectorAll(".sweep-layout-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sweep-layout-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const grid = $("sweep-chart-grid");
      if (grid) {
        grid.classList.toggle("single-col", btn.dataset.layout === "1col");
        // Resize charts smoothly after grid reflow
        requestAnimationFrame(() => {
          Object.values(charts).forEach((c) => c?.resize());
        });
      }
    });
  });
}

// ── Second Page: Navigation & Single-Frame Symbol Benchmark ──────────────────────────
let singleFrameCharts = null;
let lastSingleFrameRes = null;

function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.dataset.target;
      document.querySelectorAll(".page-container").forEach((page) => {
        page.hidden = (page.id !== targetId);
      });
      if (targetId === "page-sweep") {
        requestAnimationFrame(() => {
          Object.values(charts).forEach((c) => c?.resize());
        });
      }
      if (targetId === "page-single-frame" && singleFrameCharts) {
        requestAnimationFrame(() => {
          singleFrameCharts.acc?.resize();
          singleFrameCharts.latency?.resize();
        });
      }
      if (targetId === "page-sync-exp") {
        if (!syncExpCharts) {
          syncExpCharts = createSyncExperimentCharts();
        }
        ensureSyncExperimentLoaded();
        requestAnimationFrame(() => {
          Object.values(syncExpCharts || {}).forEach((c) => c?.resize());
        });
      }
    });
  });
}

function setupSingleFrameControls() {
  $("single-apply-cfo")?.addEventListener("change", (e) => {
    $("single-cfo-row").hidden = !e.target.checked;
  });
  $("single-apply-sto")?.addEventListener("change", (e) => {
    $("single-sto-row").hidden = !e.target.checked;
  });

  document.querySelectorAll(".single-export-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.chart;
      if (singleFrameCharts && key) {
        exportSingleFrameChartPng(singleFrameCharts, key, lastSingleFrameRes);
      }
    });
  });

  $("single-frame-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("single-run-btn");
    const errorEl = $("single-error");
    errorEl.hidden = true;
    btn.disabled = true;
    btn.textContent = "Evaluating Models...";

    const nSymbols = Number($("single-n-symbols").value) || 10;
    if (nSymbols < 1 || nSymbols > 1000) {
      errorEl.textContent = "Payload length must be between 1 and 1000 symbols.";
      errorEl.hidden = false;
      btn.disabled = false;
      btn.textContent = "Evaluate Models";
      return;
    }

    const rawSeed = $("single-seed").value.trim();
    const parsedSeed = rawSeed === "" ? -1 : Number(rawSeed);

    const payload = {
      sf: Number($("single-sf").value) || 7,
      bw: 125000,
      fs: 1000000,
      n_symbols: nSymbols,
      snr_db: Number($("single-snr").value) ?? 10,
      apply_cfo: $("single-apply-cfo").checked,
      cfo_hz: Number($("single-cfo-hz").value) || 0,
      apply_sto: $("single-apply-sto").checked,
      sto_samples: Number($("single-sto-samp").value) || 0,
      apply_compensation: $("single-apply-compensation")?.checked ?? true,
      seed: Number.isNaN(parsedSeed) ? -1 : parsedSeed,
    };

    try {
      const res = await api.runSingleFrameBenchmark(payload);
      renderSingleFrameResults(res);
    } catch (err) {
      errorEl.textContent = err.message || "Failed to run single-frame benchmark.";
      errorEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Evaluate Models";
    }
  });
}

function renderSingleFrameResults(res) {
  $("single-results-placeholder").hidden = true;
  $("single-results-container").hidden = false;

  // Update summary chips
  $("summary-chip-sf").textContent = `SF${res.sf}`;
  $("summary-chip-snr").textContent = `SNR: ${res.snr_db} dB`;

  if (res.apply_cfo) {
    if (res.apply_compensation && res.sync_success) {
      $("summary-chip-cfo").textContent = `CFO: ${res.cfo_applied_hz} Hz (Est: ${res.cfo_estimated_hz} Hz, Res: ${res.residual_cfo_hz} Hz)`;
    } else if (res.apply_compensation && !res.sync_success) {
      $("summary-chip-cfo").textContent = `CFO: ${res.cfo_applied_hz} Hz (Sync Failed)`;
    } else {
      $("summary-chip-cfo").textContent = `CFO: ${res.cfo_applied_hz} Hz (Raw / Uncompensated)`;
    }
  } else {
    $("summary-chip-cfo").textContent = "CFO: None";
  }

  if (res.apply_sto) {
    if (res.apply_compensation && res.sync_success) {
      $("summary-chip-sto").textContent = `STO: ${res.sto_applied_samp} samp (Est: ${res.sto_estimated_samp} samp)`;
    } else {
      $("summary-chip-sto").textContent = `STO: ${res.sto_applied_samp} samp (Raw)`;
    }
  } else {
    $("summary-chip-sto").textContent = "STO: None";
  }

  const syncChip = $("summary-chip-sync");
  if (syncChip) {
    if (!res.apply_cfo && !res.apply_sto) {
      syncChip.textContent = "Sync: Nominal";
    } else if (res.apply_compensation) {
      syncChip.textContent = res.sync_success ? "Sync: Locked & Compensated" : "Sync: Failed (Preamble Lost)";
    } else {
      syncChip.textContent = "Sync: Bypassed (Raw Channel)";
    }
  }

  $("summary-chip-symbols").textContent = `${res.n_symbols} symbols`;

  // Render Table
  const tbody = $("single-models-body");
  tbody.innerHTML = "";
  const order = [
    "basic_single_band",
    "dechirp_downsampled",
    "conventional",
    "nelora",
    "calora",
    "loraseek",
    "gloriphy",
  ];

  for (const key of order) {
    const m = res.models[key];
    if (!m) continue;

    const tr = document.createElement("tr");

    let accBadgeHtml;
    let matchedHtml;
    let timeHtml;

    if (m.available && m.accuracy_pct !== undefined) {
      const accClass = m.accuracy_pct === 100 ? "perfect" : (m.accuracy_pct >= 80 ? "good" : "poor");
      accBadgeHtml = `<span class="acc-badge ${accClass}">${m.accuracy_pct.toFixed(1)}%</span>`;
      matchedHtml = `<strong>${m.matched}</strong> / ${m.total}`;
      timeHtml = `${m.execution_time_ms.toFixed(1)} ms`;
    } else {
      accBadgeHtml = `<span class="acc-badge na">N/A</span>`;
      matchedHtml = `<span style="color:var(--text-dim);">-</span>`;
      timeHtml = `<span style="color:var(--text-dim);">-</span>`;
    }

    const statusBadge = m.available
      ? `<span class="status-badge completed">Ready</span>`
      : `<span class="status-badge cancelled" title="${m.status || ''}">Unavailable</span>`;

    const REF_MAP = {
      conventional: { num: 1, title: "Reference [1]: From Demodulation to Decoding (Dual-Band Dechirping) — Xu et al." },
      nelora:       { num: 2, title: "Reference [2]: NELoRa — Li et al., ACM SenSys '21" },
      loraseek:     { num: 3, title: "Reference [3]: LoRaSeek — ACM MobiCom '25" },
      gloriphy:     { num: 4, title: "Reference [4]: GLoRiPHY — ACM SenSys '24" },
      calora:       { num: 5, title: "Reference [5]: CALoRa (Chirp-Aware Self-Attention) — IEEE" },
    };
    const refInfo = REF_MAP[key];
    const refMark = refInfo
      ? `<a href="#ref-item-${refInfo.num}" class="ref-cite-mark" title="${refInfo.title}">[${refInfo.num}]</a>`
      : "";

    tr.innerHTML = `
      <td><span class="model-pill ${key}">${m.name}</span>${refMark}</td>
      <td>${statusBadge}</td>
      <td>${accBadgeHtml}</td>
      <td>${matchedHtml}</td>
      <td>${timeHtml}</td>
      <td style="color:var(--text-dim); font-size:11px;">${m.notes || m.status}</td>
    `;
    tbody.appendChild(tr);
  }

  // Update Charts (create lazily AFTER #single-results-container is visible so
  // canvases measure real non-zero layout dimensions, then resize + draw)
  lastSingleFrameRes = res;
  requestAnimationFrame(() => {
    if (!singleFrameCharts) {
      singleFrameCharts = createSingleFrameCharts();
    }
    if (singleFrameCharts) {
      singleFrameCharts.acc?.resize();
      singleFrameCharts.latency?.resize();
      updateSingleFrameCharts(singleFrameCharts, res.models, res);
    }
  });

  // Render Symbol-by-Symbol Breakdown Table
  const gt = res.ground_truth_symbols || [];
  const totalSyms = res.n_symbols || gt.length;
  const thead = $("single-symbols-head");
  const sbody = $("single-symbols-body");
  const hintEl = $("single-symbols-hint");

  const displayCount = Math.min(gt.length, 64);
  if (hintEl) {
    if (totalSyms > 64) {
      hintEl.textContent = `Showing first 64 of ${totalSyms} symbols (${totalSyms - 64} omitted from table view). Green indicates exact match with Ground Truth; Red indicates error. Overall metrics above evaluate all ${totalSyms} symbols.`;
    } else {
      hintEl.textContent = "Green indicates exact match with Ground Truth; Red indicates decoding error.";
    }
  }

  thead.innerHTML = `
    <tr>
      <th>Symbol #</th>
      <th>Ground Truth</th>
      <th>FFT (Decimation 1/8)</th>
      <th>FFT (Downsample Polyphase)</th>
      <th>FFT Without Downsample<a href="#ref-item-1" class="ref-cite-mark" title="Reference [1]: Xu et al.">[1]</a></th>
      <th>NeLoRa<a href="#ref-item-2" class="ref-cite-mark" title="Reference [2]: Li et al., ACM SenSys '21">[2]</a></th>
      <th>CALoRa<a href="#ref-item-5" class="ref-cite-mark" title="Reference [5]: IEEE">[5]</a></th>
      <th>LoRaSeek<a href="#ref-item-3" class="ref-cite-mark" title="Reference [3]: ACM MobiCom '25">[3]</a></th>
      <th>GLoRiPhy<a href="#ref-item-4" class="ref-cite-mark" title="Reference [4]: ACM SenSys '24">[4]</a></th>
    </tr>
  `;

  const rows = [];
  for (let i = 0; i < displayCount; i++) {
    const gtVal = gt[i];

    let rowHtml = `<tr>
      <td style="font-weight:600; color:var(--text-dim);">#${i + 1}</td>
      <td class="symbol-gt-cell">${gtVal}</td>
    `;

    for (const key of order) {
      const m = res.models[key];
      if (m && m.available && m.symbols && m.symbols[i] !== undefined) {
        const pred = m.symbols[i];
        const matchClass = (pred === gtVal) ? "symbol-match-ok" : "symbol-match-err";
        rowHtml += `<td class="${matchClass}">${pred}</td>`;
      } else {
        rowHtml += `<td class="symbol-match-na">—</td>`;
      }
    }

    rowHtml += `</tr>`;
    rows.push(rowHtml);
  }

  if (totalSyms > 64) {
    rows.push(`<tr>
      <td colspan="9" style="text-align: center; color: var(--text-dim); font-style: italic; padding: 10px; background: #fafaf9;">
        ... ${totalSyms - 64} additional symbols omitted (table capped at 64 samples). All ${totalSyms} symbols were benchmarked in the metrics above.
      </td>
    </tr>`);
  }

  sbody.innerHTML = rows.join("");
}

// ── Third Page: Denoised Up/Down-Chirp Synchronization Experiment ──────────────────────
let syncExpCharts = null;
let syncSavedResults = {};
let syncCurrentData = null;
let syncLoadedOnce = false;

async function ensureSyncExperimentLoaded() {
  if (syncLoadedOnce && syncCurrentData) {
    renderSyncExperiment(syncCurrentData);
    return;
  }
  try {
    const resp = await api.getSyncExperimentResults();
    syncSavedResults = resp.results || {};
    syncLoadedOnce = true;
    const initial = syncSavedResults.sf7 || syncSavedResults.sf8 || null;
    if (initial) {
      selectSavedSyncSf(initial.sf);
    } else if ($("sync-saved-meta")) {
      $("sync-saved-meta").textContent = "No saved results yet — click 'Run Live Sync Experiment' below.";
    }
  } catch (err) {
    console.error("Failed to load sync experiment results:", err);
  }
}

function selectSavedSyncSf(sf) {
  const key = `sf${sf}`;
  document.querySelectorAll(".sync-preset-btn").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.sf) === Number(sf));
  });
  const data = syncSavedResults[key];
  if (data) {
    syncCurrentData = data;
    renderSyncExperiment(data);
    if ($("sync-saved-meta")) {
      $("sync-saved-meta").textContent = `Showing saved SF${data.sf} sweep (${data.n_trials} trials/SNR, runtime ${data.elapsed_sec}s).`;
    }
  } else if ($("sync-saved-meta")) {
    $("sync-saved-meta").textContent = `No saved SF${sf} sweep found yet — use the form below to run one.`;
  }
}

function renderSyncExperiment(data) {
  if (!data) return;
  if (!syncExpCharts) {
    syncExpCharts = createSyncExperimentCharts();
  }

  // Summary chips
  $("sync-chip-sf").textContent = `SF${data.sf}`;
  $("sync-chip-trials").textContent = `N=${data.n_trials} trials/SNR`;
  $("sync-chip-cfo").textContent = `CFO: [${data.cfo_range_hz[0]}, ${data.cfo_range_hz[1]}] Hz`;
  $("sync-chip-sto").textContent = `STO: [${data.sto_range_samp[0]}, ${data.sto_range_samp[1]}] samp`;
  $("sync-chip-tol").textContent = `Tol: ±${data.cfo_tol_hz} Hz / ±${data.sto_tol_samp} samp`;

  // KPI Highlight cards (derived directly from data.series so they always populate
  // regardless of backend highlights schema)
  const snrs = data.snr_list || [];
  const series = data.series || {};
  const hl = data.highlights || {};
  const targetSnr = hl.ref_snr_db !== undefined ? hl.ref_snr_db : (Number(data.sf) === 7 ? -18 : -20);
  let refIdx = snrs.indexOf(targetSnr);
  if (refIdx < 0) refIdx = Math.floor(snrs.length / 2);
  const refSnr = snrs[refIdx];
  const snrTag = refSnr !== undefined ? `@ ${refSnr} dB` : "";

  const getVal = (mKey, metricKey) => {
    const v = series[mKey]?.[metricKey]?.[refIdx];
    return typeof v === "number" && !Number.isNaN(v) ? v : null;
  };
  const fmtPct = (v) => (v === null ? "—" : `${v.toFixed(1)}%`);
  const fmtGain = (v, base) => {
    if (v === null || base === null) return "";
    const g = v - base;
    const sign = g >= 0 ? "+" : "";
    return `<span class="sync-gain-badge">${sign}${g.toFixed(1)}%</span>`;
  };
  const fmtSignedDiff = (v, base) => {
    if (v === null || base === null) return "—";
    const g = v - base;
    return `${g >= 0 ? "+" : ""}${g.toFixed(1)}%`;
  };

  const convCfo = getVal("conv_single", "cfo_acc");
  const convSto = getVal("conv_single", "sto_acc");
  const convPay = getVal("conv_single", "payload_acc");

  const glCfo = getVal("gloriphy_single", "cfo_acc");
  const glSto = getVal("gloriphy_single", "sto_acc");
  const glPay = getVal("gloriphy_single", "payload_acc");

  const lsCfo = getVal("loraseek_single", "cfo_acc");
  const lsSto = getVal("loraseek_single", "sto_acc");
  const lsPay = getVal("loraseek_single", "payload_acc");

  $("kpi-base-title").textContent = `Conventional ${snrTag}`;
  $("kpi-base-val").innerHTML = `CFO ${fmtPct(convCfo)} · STO ${fmtPct(convSto)}`;
  $("kpi-base-sub").textContent = `Downstream Payload Acc: ${fmtPct(convPay)}`;

  $("kpi-single-title").textContent = `GLoRiPHY Denoised ${snrTag}`;
  $("kpi-single-val").innerHTML = `CFO ${fmtPct(glCfo)}${fmtGain(glCfo, convCfo)}`;
  $("kpi-single-sub").innerHTML = `STO ${fmtPct(glSto)} (${fmtSignedDiff(glSto, convSto)}) · Payload: ${fmtPct(glPay)}`;

  $("kpi-multi-title").textContent = `LoRaSeek Denoised ${snrTag}`;
  $("kpi-multi-val").innerHTML = `CFO ${fmtPct(lsCfo)}${fmtGain(lsCfo, convCfo)}`;
  $("kpi-multi-sub").innerHTML = `STO ${fmtPct(lsSto)} (${fmtSignedDiff(lsSto, convSto)}) · Payload: ${fmtPct(lsPay)}`;

  // Update all 5 Chart.js graphs
  updateSyncExperimentCharts(syncExpCharts, data);

  // Render numerical breakdown table
  renderSyncTable(data);
}

function renderSyncTable(data) {
  if (!data) return;
  const metric = $("sync-table-metric")?.value || "cfo_acc";
  const series = data.series || {};
  const meta = data.methods_meta || {};
  const snrs = data.snr_list || [];
  const allowedOrder = ["conv_single", "gloriphy_single", "loraseek_single"];
  const order = allowedOrder.filter((m) => series[m]);
  const fallbackColor = {
    conv_single: "#334155",
    gloriphy_single: "#e11d48",
    loraseek_single: "#2563eb",
  };
  const fallbackShort = {
    conv_single: "Conventional",
    gloriphy_single: "GLoRiPHY Denoised",
    loraseek_single: "LoRaSeek Denoised",
  };

  const lowerIsBetter = (metric === "cfo_mae" || metric === "sto_mae" || metric === "avg_ms");
  const unit =
    metric === "cfo_mae" ? " Hz"
    : metric === "sto_mae" ? " samp"
    : metric === "avg_ms" ? " ms"
    : "%";

  const thead = $("sync-table-head");
  const tbody = $("sync-table-body");
  if (!thead || !tbody) return;

  thead.innerHTML = `<tr>
    <th>SNR (dB)</th>
    ${order
      .map((m) => {
        const color = fallbackColor[m] || meta[m]?.color || "#475569";
        const label = fallbackShort[m] || meta[m]?.short || m;
        return `<th><span class="sync-method-dot" style="background:${color}"></span>${label}</th>`;
      })
      .join("")}
  </tr>`;

  const rowsHtml = snrs.map((snr, idx) => {
    const vals = order.map((m) => series[m]?.[metric]?.[idx]);
    const numericVals = vals.filter((v) => typeof v === "number" && !Number.isNaN(v));
    const bestVal = numericVals.length
      ? (lowerIsBetter ? Math.min(...numericVals) : Math.max(...numericVals))
      : null;

    const cells = vals
      .map((v) => {
        if (v === undefined || v === null) return `<td>—</td>`;
        const isBest = bestVal !== null && Math.abs(v - bestVal) < 1e-6;
        const formatted = (metric === "cfo_mae" && v >= 100) ? v.toFixed(0) : v.toFixed(1);
        return `<td class="${isBest ? "sync-best-cell" : ""}">${formatted}${unit}</td>`;
      })
      .join("");

    return `<tr><td>${snr} dB</td>${cells}</tr>`;
  });

  tbody.innerHTML = rowsHtml.join("");
}

function setupSyncExperimentControls() {
  document.querySelectorAll(".sync-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sf = Number(btn.dataset.sf);
      selectSavedSyncSf(sf);
    });
  });

  document.querySelectorAll(".sync-export-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const chartKey = btn.dataset.chart;
      const sf = syncCurrentData?.sf || 7;
      if (syncExpCharts && chartKey) {
        exportSyncChartPng(syncExpCharts, chartKey, sf);
      }
    });
  });

  $("sync-table-metric")?.addEventListener("change", () => {
    if (syncCurrentData) renderSyncTable(syncCurrentData);
  });

  $("sync-sf")?.addEventListener("change", (e) => {
    const sf = Number(e.target.value);
    if (sf === 8) {
      $("sync-snr-min").value = -26;
      $("sync-snr-max").value = -12;
    } else {
      $("sync-snr-min").value = -24;
      $("sync-snr-max").value = -10;
    }
  });

  $("sync-exp-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("sync-run-btn");
    const errEl = $("sync-error");
    errEl.hidden = true;
    btn.disabled = true;
    btn.textContent = "Running Sync Experiment...";

    const rawSeed = $("sync-seed").value.trim();
    const parsedSeed = rawSeed === "" ? -1 : Number(rawSeed);

    const shouldSave = Boolean($("sync-save-preset")?.checked);

    const payload = {
      sf: Number($("sync-sf").value) || 7,
      snr_min: Number($("sync-snr-min").value),
      snr_max: Number($("sync-snr-max").value),
      snr_step: Number($("sync-snr-step").value) || 2,
      n_trials: Number($("sync-n-trials").value) || 40,
      cfo_lo: Number($("sync-cfo-lo").value),
      cfo_hi: Number($("sync-cfo-hi").value),
      sto_lo: Number($("sync-sto-lo").value),
      sto_hi: Number($("sync-sto-hi").value),
      seed: Number.isNaN(parsedSeed) ? -1 : parsedSeed,
      save: shouldSave,
    };

    try {
      const res = await api.runSyncExperiment(payload);
      if (shouldSave) {
        syncSavedResults[`sf${res.sf}`] = res;
        selectSavedSyncSf(res.sf);
      } else {
        syncCurrentData = res;
        document.querySelectorAll(".sync-preset-btn").forEach((b) => b.classList.remove("active"));
        renderSyncExperiment(res);
        if ($("sync-saved-meta")) {
          $("sync-saved-meta").textContent = `Showing live custom SF${res.sf} sweep (${res.n_trials} trials/SNR, ${res.elapsed_sec}s). Click SF7/SF8 above to return to saved N=100 presets.`;
        }
      }
    } catch (err) {
      errEl.textContent = err.message || "Failed to run denoised synchronization experiment.";
      errEl.hidden = false;
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Live Sync Experiment";
    }
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────────────────
(async function init() {
  wireTouchTracking();
  setupNavigation();
  setupSweepExportControls();
  setupSingleFrameControls();
  setupSyncExperimentControls();
  applyMethodDefaults($("preamble_detection_method").value); // matches the <select>'s own default
  updatePacketsPerBinHint();
  await loadCapabilities().catch((e) => console.error("capabilities failed", e));
  await refreshHistory().catch((e) => console.error("history failed", e));
})();
