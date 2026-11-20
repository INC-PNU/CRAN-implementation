// Chart.js setup + publication/journal-grade rendering across all 3 dashboard tabs.
// Charts are created once and updated in place via chart.update() so transitions stay smooth.

const JOURNAL_FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";

export const PALETTE = [
  "#2563eb", // Royal Blue
  "#e11d48", // Crimson Rose
  "#059669", // Emerald Green
  "#d97706", // Amber Gold
  "#7c3aed", // Deep Violet
  "#0891b2", // Teal Cyan
  "#334155", // Charcoal Slate
  "#c026d3", // Fuchsia
];
const MARKER_SHAPES = [
  "circle",
  "rect",
  "triangle",
  "rectRot",
  "circle",
  "rect",
  "triangle",
  "rectRot",
];
export const MAX_OVERLAY_RUNS = PALETTE.length;

const STYLE_KEY = "lora_dashboard_run_styles_v2";
const LEGACY_COLOR_KEY = "lora_dashboard_run_colors";

function loadStyleOverrides() {
  try {
    return JSON.parse(localStorage.getItem(STYLE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveStyleOverrides(map) {
  try {
    localStorage.setItem(STYLE_KEY, JSON.stringify(map));
  } catch {
    /* private browsing / storage disabled — overrides just won't persist */
  }
}

// Consistent semantic defaults based on Preamble Method × Sync Denoiser × Channel Impairment:
//   Normal sync:
//     Conventional + Clean   -> Charcoal Slate (#334155), ●
//     Conventional + CFO/TO  -> Crimson Rose   (#e11d48), ■
//     CALoRa       + Clean   -> Royal Blue     (#2563eb), ▲
//     CALoRa       + CFO/TO  -> Emerald Green  (#059669), ◆
//   Denoised sync:
//     LoRaSeek Sync          -> Deep Violet    (#7c3aed), ▲
//     GLoRiPHY Sync          -> Amber Gold     (#d97706), ◆
export function semanticStyleForRun(meta) {
  const cli = meta?.client_config || {};
  const srv = meta?.server_config || {};
  const cfo = cli.cfo_hz_range || [0, 0];
  const sto = cli.sto_samp_range || [0, 0];
  const impaired = cfo[0] !== 0 || cfo[1] !== 0 || sto[0] !== 0 || sto[1] !== 0;
  const isCalora = srv.preamble_detection_method === "calora";
  const syncDen = srv.sync_denoiser || "none";

  if (syncDen === "loraseek") return { color: "#7c3aed", marker: "triangle" };
  if (syncDen === "gloriphy") return { color: "#d97706", marker: "rectRot" };
  if (!isCalora && !impaired) return { color: "#334155", marker: "circle" };
  if (!isCalora && impaired)  return { color: "#e11d48", marker: "rect" };
  if (isCalora && !impaired)  return { color: "#2563eb", marker: "triangle" };
  return { color: "#059669", marker: "rectRot" };
}

export function getRunStyle(runId, meta = null) {
  const overrides = loadStyleOverrides();
  const custom = overrides[runId] || {};
  const base = semanticStyleForRun(meta);
  return {
    color: custom.color || base.color,
    marker: custom.marker || base.marker,
    isCustom: Boolean(custom.color || custom.marker),
  };
}

// Resolve styles for currently selected runs so two overlaid curves in the same semantic
// bucket automatically receive distinct colors/markers unless explicitly overridden.
export function resolveSelectedStyles(selectedItems) {
  const overrides = loadStyleOverrides();
  const resolved = new Map();
  const usedColors = new Set();

  // First reserve all explicitly chosen user colors
  for (const { runId } of selectedItems) {
    const customColor = overrides[runId]?.color;
    if (customColor) usedColors.add(customColor.toLowerCase());
  }

  let fallbackIdx = 0;
  for (const { runId, meta } of selectedItems) {
    const custom = overrides[runId] || {};
    const base = semanticStyleForRun(meta);

    let color = custom.color || base.color;
    let marker = custom.marker || base.marker;

    if (!custom.color) {
      if (usedColors.has(color.toLowerCase())) {
        // Pick first unused publication palette color to prevent accidental duplicates
        const alt = PALETTE.find((c) => !usedColors.has(c.toLowerCase()));
        color = alt || PALETTE[fallbackIdx % PALETTE.length];
        if (!custom.marker) {
          marker = MARKER_SHAPES[fallbackIdx % MARKER_SHAPES.length];
        }
        fallbackIdx++;
      }
      usedColors.add(color.toLowerCase());
    }

    resolved.set(runId, { color, marker, isCustom: Boolean(custom.color || custom.marker) });
  }
  return resolved;
}

export function setRunColor(runId, color) {
  const overrides = loadStyleOverrides();
  overrides[runId] = { ...(overrides[runId] || {}), color };
  saveStyleOverrides(overrides);
}

export function setRunMarker(runId, marker) {
  const overrides = loadStyleOverrides();
  overrides[runId] = { ...(overrides[runId] || {}), marker };
  saveStyleOverrides(overrides);
}

export function resetAllRunStyles() {
  try {
    localStorage.removeItem(STYLE_KEY);
    localStorage.removeItem(LEGACY_COLOR_KEY);
  } catch {
    /* ignore */
  }
}

export function pruneRunStyles(validRunIds) {
  const valid = new Set(validRunIds);
  const overrides = loadStyleOverrides();
  let changed = false;
  for (const rid of Object.keys(overrides)) {
    if (!valid.has(rid)) {
      delete overrides[rid];
      changed = true;
    }
  }
  if (changed) saveStyleOverrides(overrides);
}

export function assignColor(runId, meta = null) {
  return getRunStyle(runId, meta).color;
}

export function assignMarker(runId, meta = null) {
  return getRunStyle(runId, meta).marker;
}

function drawJournalMarker(ctx, shape, cx, cy, r, fillColor) {
  ctx.save();
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  if (shape === "rect") {
    const s = r * 1.72;
    ctx.rect(cx - s / 2, cy - s / 2, s, s);
  } else if (shape === "triangle") {
    const h = r * 1.15;
    ctx.moveTo(cx, cy - h * 1.15);
    ctx.lineTo(cx + h * 1.05, cy + h * 0.85);
    ctx.lineTo(cx - h * 1.05, cy + h * 0.85);
    ctx.closePath();
  } else if (shape === "rectRot") {
    const d = r * 1.28;
    ctx.moveTo(cx, cy - d);
    ctx.lineTo(cx + d, cy);
    ctx.lineTo(cx, cy + d);
    ctx.lineTo(cx - d, cy);
    ctx.closePath();
  } else {
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

// Unified publication-grade Chart.js plugin:
// 1. Solid pure-white canvas background (clean snapshots & HD PNG downloads).
// 2. Crisp 4-sided IEEE/ACM dark frame around the plot area (drawn behind markers).
// 3. Optional crisp numeric data labels directly above bar-chart columns.
// 4. Vector line+marker legend swatches (—●—, —■—, —▲—, —◆—) matching line curves 1-to-1.
const journalPlotPlugin = {
  id: "journalPlotPlugin",
  beforeDraw(chart) {
    const { ctx, width, height } = chart;
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.restore();
  },
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    ctx.save();
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.3;
    ctx.strokeRect(
      chartArea.left,
      chartArea.top,
      chartArea.right - chartArea.left,
      chartArea.bottom - chartArea.top
    );
    ctx.restore();
  },
  afterDatasetsDraw(chart, _args, pluginOptions) {
    const cfg = pluginOptions || chart.config?.options?.plugins?.journalPlotPlugin;
    if (!cfg || !cfg.barLabels || chart.config.type !== "bar" || !chart.chartArea) return;
    const { ctx } = chart;
    const ds = chart.data.datasets[0];
    const meta = chart.getDatasetMeta(0);
    if (!ds || !meta || !meta.data) return;

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.font = `700 14.5px ${JOURNAL_FONT}`;

    meta.data.forEach((bar, i) => {
      if (!bar || !Number.isFinite(bar.x) || !Number.isFinite(bar.y)) return;
      const val = ds.data[i];
      const available = ds._available ? ds._available[i] !== false : true;
      let text;
      if (!available || val === null || val === undefined) {
        text = "N/A";
        ctx.fillStyle = "#94a3b8";
      } else if (cfg.barLabelFormat === "percent") {
        text = `${Number(val).toFixed(1)}%`;
        ctx.fillStyle = "#0f172a";
      } else if (cfg.barLabelFormat === "latency") {
        const num = Number(val);
        text = num >= 1000 ? `${(num / 1000).toFixed(2)}s` : `${num.toFixed(1)}`;
        ctx.fillStyle = "#0f172a";
      } else {
        text = String(val);
        ctx.fillStyle = "#0f172a";
      }
      const yPos = Math.max(bar.y - 6, chart.chartArea.top + 18);
      ctx.fillText(text, bar.x, yPos);
    });
    ctx.restore();
  },
  afterDraw(chart) {
    if (chart.config.type !== "line") return;
    const legend = chart.legend;
    if (!legend || !legend.legendHitBoxes || !legend.legendItems) return;
    const { ctx } = chart;
    const boxW = legend.options?.labels?.boxWidth ?? 40;

    legend.legendItems.forEach((item, idx) => {
      const hitBox = legend.legendHitBoxes[idx];
      const ds = chart.data.datasets[item.datasetIndex];
      if (!hitBox || !ds) return;

      const meta = chart.getDatasetMeta(item.datasetIndex);
      const isHidden = meta.hidden === true;
      const cx = hitBox.left + boxW / 2;
      const cy = hitBox.top + hitBox.height / 2;
      const color = ds.borderColor || "#334155";
      const shape = ds.pointStyle || "circle";
      const dash = Array.isArray(ds.borderDash) ? ds.borderDash : [];

      ctx.save();
      ctx.globalAlpha = isHidden ? 0.3 : 1.0;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3.0;
      ctx.lineCap = "round";
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(hitBox.left + 2, cy);
      ctx.lineTo(hitBox.left + boxW - 2, cy);
      ctx.stroke();

      ctx.setLineDash([]);
      drawJournalMarker(ctx, shape, cx, cy, 5.2, color);
      ctx.restore();
    });
  },
};

function transparentLegendLabels(chart) {
  const items = Chart.defaults.plugins.legend.labels.generateLabels(chart);
  return items.map((item) => ({
    ...item,
    fillStyle: "rgba(0,0,0,0)",
    strokeStyle: "rgba(0,0,0,0)",
    lineWidth: 0,
  }));
}

function downloadChartPng(chart, filename) {
  if (!chart) return;
  const url = chart.toBase64Image("image/png", 1.0);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 1: PREAMBLE & DEMOD SWEEP CHARTS
// ═══════════════════════════════════════════════════════════════════════════════

function pct(numerator, denominator) {
  return denominator ? (numerator / denominator) * 100 : null;
}

// Raw bins[] -> per-metric {snr: value} for the given scope. Both scopes are always
// derivable from the same raw counts, so switching scope needs zero network requests.
export function computeSeries(bins, scope) {
  const out = { preamble: {}, cfo: {}, sto: {}, symbol: {}, n: {} };
  for (const b of bins) {
    const mislocated = b.preamble_mislocated ?? 0;
    const located = b.n_total - b.preamble_missed - mislocated;
    const reached = located - (b.offset_fail || 0);
    out.preamble[b.snr_db] = pct(located, b.n_total);
    out.n[b.snr_db] = b.n_total;
    if (scope === "conditioned") {
      out.cfo[b.snr_db] = pct(b.cfo_correct, reached);
      out.sto[b.snr_db] = pct(b.sto_correct, reached);
      out.symbol[b.snr_db] = pct(b.symbol_correct, b.symbol_total);
    } else {
      out.cfo[b.snr_db] = pct(b.cfo_correct, b.n_total);
      out.sto[b.snr_db] = pct(b.sto_correct, b.n_total);
      out.symbol[b.snr_db] = pct(b.symbol_correct, b.symbol_total_sent);
    }
  }
  return out;
}

function lineDataset(run, values, snrLabels) {
  const marker = run.marker || assignMarker(run.runId);
  return {
    label: run.label,
    data: snrLabels.map((s) => (values[s] === undefined ? null : values[s])),
    borderColor: run.color,
    backgroundColor: run.color,
    borderWidth: 3.0,
    borderDash: run.dashed ? [8, 5] : [],
    spanGaps: true,
    tension: 0.14,
    clip: false,
    pointRadius: 5.5,
    pointHoverRadius: 7.2,
    pointStyle: marker,
    pointBackgroundColor: run.color,
    pointBorderColor: "#ffffff",
    pointBorderWidth: 1.6,
    fill: false,
  };
}

function makeSweepLineChart(canvasId, titleText, yLabel) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [] },
    plugins: [journalPlotPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.45,
      devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2.5),
      animation: { duration: 220 },
      layout: {
        padding: { top: 10, right: 20, bottom: 8, left: 8 },
      },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          border: { display: true, color: "#0f172a", width: 1.4 },
          grid: {
            color: "rgba(15, 23, 42, 0.10)",
            tickLength: 6,
            tickColor: "#0f172a",
          },
          ticks: {
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
            maxRotation: 0,
            padding: 7,
          },
          title: {
            display: true,
            text: "SNR (dB)",
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
            padding: { top: 8 },
          },
        },
        y: {
          min: 0,
          max: 100,
          border: { display: true, color: "#0f172a", width: 1.4 },
          grid: {
            color: "rgba(15, 23, 42, 0.12)",
            tickLength: 6,
            tickColor: "#0f172a",
          },
          ticks: {
            stepSize: 20,
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
            padding: 7,
          },
          title: {
            display: true,
            text: yLabel,
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
            padding: { bottom: 10 },
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: titleText,
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 19, weight: "700" },
          padding: { top: 2, bottom: 14 },
        },
        legend: {
          position: "bottom",
          align: "start",
          labels: {
            boxWidth: 40,
            boxHeight: 16,
            padding: 14,
            color: "#0f172a",
            font: (c) => ({
              family: JOURNAL_FONT,
              size: c.chart.width >= 780 ? 15.5 : (c.chart.width >= 660 ? 14.5 : 13.5),
              weight: "600",
            }),
            generateLabels: transparentLegendLabels,
          },
        },
        tooltip: {
          callbacks: {
            title(items) {
              if (!items.length) return "";
              return `SNR: ${items[0].label} dB`;
            },
            label(item) {
              const v = item.raw;
              if (v === null || v === undefined) return `${item.dataset.label}: N/A`;
              return `${item.dataset.label}: ${Number(v).toFixed(1)}%`;
            },
            afterLabel(item) {
              const n = item.chart._nCounts?.[item.datasetIndex]?.[item.dataIndex];
              return n !== undefined ? `  (N = ${n} packets)` : "";
            },
          },
        },
      },
    },
  });
}

export function createCharts() {
  return {
    preamble: makeSweepLineChart("chart-preamble", "Preamble Detection Rate vs. SNR", "Detection Rate (%)"),
    cfo: makeSweepLineChart("chart-cfo", "CFO Estimation Success Rate vs. SNR", "CFO Success Rate (%)"),
    sto: makeSweepLineChart("chart-sto", "STO Estimation Success Rate vs. SNR", "STO Success Rate (%)"),
    symbol: makeSweepLineChart("chart-symbol", "Symbol Decoding Accuracy vs. SNR", "Symbol Accuracy (%)"),
    nbin: new Chart(document.getElementById("chart-nbin").getContext("2d"), {
      type: "bar",
      data: { labels: [], datasets: [] },
      plugins: [journalPlotPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 3.4,
        devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2.5),
        animation: { duration: 220 },
        layout: {
          padding: { top: 8, right: 16, bottom: 6, left: 6 },
        },
        scales: {
          x: {
            border: { display: true, color: "#0f172a", width: 1.4 },
            grid: { display: false, tickLength: 6, tickColor: "#0f172a" },
            ticks: {
              color: "#0f172a",
              font: { family: JOURNAL_FONT, size: 15, weight: "600" },
              maxRotation: 0,
            },
            title: {
              display: true,
              text: "SNR (dB)",
              color: "#0f172a",
              font: { family: JOURNAL_FONT, size: 17, weight: "700" },
              padding: { top: 8 },
            },
          },
          y: {
            beginAtZero: true,
            border: { display: true, color: "#0f172a", width: 1.4 },
            grid: {
              color: "rgba(15, 23, 42, 0.12)",
              tickLength: 6,
              tickColor: "#0f172a",
            },
            ticks: {
              color: "#0f172a",
              font: { family: JOURNAL_FONT, size: 15, weight: "600" },
              padding: 7,
            },
            title: {
              display: true,
              text: "Packets (N)",
              color: "#0f172a",
              font: { family: JOURNAL_FONT, size: 17, weight: "700" },
              padding: { bottom: 8 },
            },
          },
        },
        plugins: {
          title: {
            display: true,
            text: "Packets Evaluated per SNR Bin",
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 18.5, weight: "700" },
            padding: { top: 2, bottom: 12 },
          },
          legend: {
            position: "bottom",
            align: "start",
            labels: {
              boxWidth: 18,
              boxHeight: 14,
              padding: 14,
              color: "#0f172a",
              font: { family: JOURNAL_FONT, size: 14.5, weight: "600" },
            },
          },
        },
      },
    }),
  };
}

// runs: array of {runId, label, color, marker, bins, dashed}
export function updateCharts(charts, runs, scope) {
  const snrSet = new Set();
  const seriesPerRun = runs.map((r) => ({ run: r, series: computeSeries(r.bins, scope) }));
  for (const { series } of seriesPerRun) {
    Object.keys(series.preamble).forEach((s) => snrSet.add(Number(s)));
  }
  const snrLabels = [...snrSet].sort((a, b) => a - b);

  for (const key of ["preamble", "cfo", "sto", "symbol"]) {
    const chart = charts[key];
    chart.data.labels = snrLabels.map((s) => String(s));
    chart.data.datasets = seriesPerRun.map(({ run, series }) => lineDataset(run, series[key], snrLabels));
    chart._nCounts = seriesPerRun.map(({ series }) => snrLabels.map((s) => series.n[s]));
    chart.update();
  }

  charts.nbin.data.labels = snrLabels.map((s) => String(s));
  charts.nbin.data.datasets = seriesPerRun.map(({ run, series }) => ({
    label: run.label,
    data: snrLabels.map((s) => series.n[s] ?? 0),
    backgroundColor: run.color,
    borderColor: "#0f172a",
    borderWidth: 0.8,
    borderRadius: 2,
  }));
  charts.nbin.update();
}

export function exportSweepChartPng(charts, chartKey) {
  downloadChartPng(charts?.[chartKey], `sweep_${chartKey}.png`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 2: SINGLE-FRAME SYMBOL BENCHMARK BAR CHARTS
// ═══════════════════════════════════════════════════════════════════════════════

export function createSingleFrameCharts() {
  const modelColors = {
    basic_single_band: "#64748b",
    dechirp_downsampled: "#334155",
    conventional: "#059669",
    nelora: "#d97706",
    calora: "#2563eb",
    loraseek: "#7c3aed",
    gloriphy: "#e11d48",
  };

  const accEl = document.getElementById("chart-single-acc");
  const latEl = document.getElementById("chart-single-latency");
  if (!accEl || !latEl) return null;

  const makeBarX = () => ({
    border: { display: true, color: "#0f172a", width: 1.4 },
    grid: { display: false, tickLength: 6, tickColor: "#0f172a" },
    ticks: {
      color: "#0f172a",
      font: { family: JOURNAL_FONT, size: 14, weight: "700" },
      maxRotation: 0,
      minRotation: 0,
      autoSkip: false,
      padding: 7,
    },
  });

  const accChart = new Chart(accEl.getContext("2d"), {
    type: "bar",
    data: { labels: [], datasets: [] },
    plugins: [journalPlotPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.5,
      devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2.5),
      animation: { duration: 250 },
      layout: {
        padding: { top: 10, right: 18, bottom: 8, left: 8 },
      },
      scales: {
        x: makeBarX(),
        y: {
          min: 0,
          max: 114,
          border: { display: true, color: "#0f172a", width: 1.4 },
          grid: {
            color: (c) => (c.tick?.value <= 100 ? "rgba(15, 23, 42, 0.12)" : "transparent"),
            tickLength: 6,
            tickColor: "#0f172a",
          },
          ticks: {
            stepSize: 20,
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
            padding: 7,
            callback: (v) => (Number(v) <= 100 ? String(v) : ""),
          },
          title: {
            display: true,
            text: "Symbol Accuracy (%)",
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
            padding: { bottom: 10 },
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Symbol Decoding Accuracy by Method",
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 18.5, weight: "700" },
          padding: { top: 2, bottom: 14 },
        },
        legend: { display: false },
        journalPlotPlugin: {
          barLabels: true,
          barLabelFormat: "percent",
        },
        tooltip: {
          callbacks: {
            title: (items) => (items.length ? items[0].label.replace(/,/g, " ") : ""),
            label: (item) => `Accuracy: ${Number(item.raw).toFixed(1)}%`,
          },
        },
      },
    },
  });

  const latChart = new Chart(latEl.getContext("2d"), {
    type: "bar",
    data: { labels: [], datasets: [] },
    plugins: [journalPlotPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.5,
      devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2.5),
      animation: { duration: 250 },
      layout: {
        padding: { top: 10, right: 18, bottom: 8, left: 8 },
      },
      scales: {
        x: makeBarX(),
        y: {
          beginAtZero: true,
          grace: "20%",
          border: { display: true, color: "#0f172a", width: 1.4 },
          grid: {
            color: "rgba(15, 23, 42, 0.12)",
            tickLength: 6,
            tickColor: "#0f172a",
          },
          ticks: {
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
            padding: 7,
          },
          title: {
            display: true,
            text: "Execution Latency (ms)",
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
            padding: { bottom: 10 },
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Execution Latency by Method",
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 18.5, weight: "700" },
          padding: { top: 2, bottom: 14 },
        },
        legend: { display: false },
        journalPlotPlugin: {
          barLabels: true,
          barLabelFormat: "latency",
        },
        tooltip: {
          callbacks: {
            title: (items) => (items.length ? items[0].label.replace(/,/g, " ") : ""),
            label: (item) => `Latency: ${Number(item.raw).toFixed(1)} ms`,
          },
        },
      },
    },
  });

  return { acc: accChart, latency: latChart, colors: modelColors };
}

export function updateSingleFrameCharts(charts, modelsData, meta = null) {
  if (!charts) return;
  const labels = [];
  const accValues = [];
  const latValues = [];
  const bgColors = [];
  const availFlags = [];

  const order = [
    "basic_single_band",
    "dechirp_downsampled",
    "conventional",
    "nelora",
    "calora",
    "loraseek",
    "gloriphy",
  ];
  const multiLineNames = {
    basic_single_band: ["FFT", "(Decim. 1/8)"],
    dechirp_downsampled: ["FFT", "(Polyphase)"],
    conventional: ["Dual-Band", "FFT [1]"],
    nelora: ["NeLoRa", "[2]"],
    calora: ["CALoRa", "[5]"],
    loraseek: ["LoRaSeek", "[3]"],
    gloriphy: ["GLoRiPHY", "[4]"],
  };
  for (const key of order) {
    const m = modelsData[key];
    if (!m) continue;
    labels.push(multiLineNames[key] || m.name);
    const isAvail = Boolean(m.available && m.accuracy_pct !== undefined);
    availFlags.push(isAvail);
    accValues.push(isAvail ? m.accuracy_pct : 0);
    latValues.push(isAvail && m.execution_time_ms !== undefined ? m.execution_time_ms : 0);
    bgColors.push(isAvail ? charts.colors[key] : "#cbd5e1");
  }

  const ctxSuffix = meta
    ? ` (SF${meta.sf}, SNR ${meta.snr_db} dB, N=${meta.n_symbols})`
    : "";

  charts.acc.options.plugins.title.text = `Symbol Decoding Accuracy${ctxSuffix}`;
  charts.acc.data.labels = labels;
  charts.acc.data.datasets = [
    {
      data: accValues,
      backgroundColor: bgColors,
      borderColor: "#0f172a",
      borderWidth: 1,
      borderRadius: 3,
      maxBarThickness: 48,
      _available: availFlags,
    },
  ];
  charts.acc.update();

  charts.latency.options.plugins.title.text = `Execution Latency${ctxSuffix}`;
  charts.latency.data.labels = labels;
  charts.latency.data.datasets = [
    {
      data: latValues,
      backgroundColor: bgColors,
      borderColor: "#0f172a",
      borderWidth: 1,
      borderRadius: 3,
      maxBarThickness: 48,
      _available: availFlags,
    },
  ];
  charts.latency.update();
}

export function exportSingleFrameChartPng(charts, chartKey, meta = null) {
  const sf = meta?.sf ?? 7;
  const snr = meta?.snr_db !== undefined ? `_snr${meta.snr_db}` : "";
  downloadChartPng(charts?.[chartKey], `single_frame_sf${sf}${snr}_${chartKey}.png`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 3: DENOISED UP/DOWN-CHIRP SYNCHRONIZATION EXPERIMENT CHARTS
// ═══════════════════════════════════════════════════════════════════════════════

const SYNC_SINGLE_STYLES = {
  conv_single: {
    label: "Conventional",
    color: "#334155",
    pointStyle: "circle",
  },
  gloriphy_single: {
    label: "GLoRiPHY Denoised",
    color: "#e11d48",
    pointStyle: "rect",
  },
  loraseek_single: {
    label: "LoRaSeek Denoised",
    color: "#2563eb",
    pointStyle: "triangle",
  },
};

function formatJournalLogTick(val, allowHalf = false) {
  if (val <= 0) return "";
  if (allowHalf && Math.abs(val - 0.5) < 1e-6) return "0.5";
  const exp = Math.log10(val);
  if (Math.abs(exp - Math.round(exp)) < 1e-5) {
    const rounded = Math.round(val);
    return rounded >= 1000 ? rounded.toLocaleString("en-US") : String(rounded);
  }
  return "";
}

function makeSyncLineChart(
  canvasId,
  titleText,
  yLabel,
  { min = 0, max = 100, isLog = false, logMin = 1, allowHalfTick = false, unit = "%", aspectRatio = 1.6 } = {}
) {
  const el = document.getElementById(canvasId);
  if (!el) return null;

  const yScale = isLog
    ? {
        type: "logarithmic",
        min: logMin,
        border: { display: true, color: "#0f172a", width: 1.4 },
        grid: {
          color: (ctx) => {
            const v = ctx.tick?.value;
            if (!v || v <= 0) return "rgba(15, 23, 42, 0.06)";
            const exp = Math.log10(v);
            return Math.abs(exp - Math.round(exp)) < 1e-5
              ? "rgba(15, 23, 42, 0.16)"
              : "rgba(15, 23, 42, 0.05)";
          },
          tickLength: 6,
          tickColor: "#0f172a",
        },
        ticks: {
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
          padding: 7,
          autoSkip: false,
          callback: (val) => formatJournalLogTick(Number(val), allowHalfTick),
        },
        title: {
          display: true,
          text: yLabel,
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
          padding: { bottom: 10 },
        },
      }
    : {
        beginAtZero: true,
        ...(min !== undefined ? { min } : {}),
        ...(max !== undefined ? { max } : {}),
        border: { display: true, color: "#0f172a", width: 1.4 },
        grid: {
          color: "rgba(15, 23, 42, 0.12)",
          tickLength: 6,
          tickColor: "#0f172a",
        },
        ticks: {
          stepSize: 20,
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
          padding: 7,
        },
        title: {
          display: true,
          text: yLabel,
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
          padding: { bottom: 10 },
        },
      };

  return new Chart(el.getContext("2d"), {
    type: "line",
    data: { labels: [], datasets: [] },
    plugins: [journalPlotPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio,
      devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2.5),
      animation: { duration: 220 },
      layout: {
        padding: { top: 10, right: 20, bottom: 8, left: 8 },
      },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          border: { display: true, color: "#0f172a", width: 1.4 },
          grid: {
            color: "rgba(15, 23, 42, 0.10)",
            tickLength: 6,
            tickColor: "#0f172a",
          },
          ticks: {
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 15.5, weight: "600" },
            maxRotation: 0,
            autoSkip: false,
            padding: 7,
          },
          title: {
            display: true,
            text: "SNR (dB)",
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 17.5, weight: "700" },
            padding: { top: 8 },
          },
        },
        y: yScale,
      },
      plugins: {
        title: {
          display: true,
          text: titleText,
          color: "#0f172a",
          font: { family: JOURNAL_FONT, size: 19, weight: "700" },
          padding: { top: 2, bottom: 14 },
        },
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 40,
            boxHeight: 16,
            padding: 22,
            color: "#0f172a",
            font: { family: JOURNAL_FONT, size: 16, weight: "600" },
            generateLabels: transparentLegendLabels,
          },
        },
        tooltip: {
          callbacks: {
            title(items) {
              if (!items.length) return "";
              return `SNR: ${items[0].label} dB`;
            },
            label(item) {
              const v = item.raw;
              if (v === null || v === undefined) return `${item.dataset.label}: N/A`;
              return `${item.dataset.label}: ${Number(v).toFixed(1)}${unit}`;
            },
          },
        },
      },
    },
  });
}

export function createSyncExperimentCharts() {
  return {
    cfoAcc: makeSyncLineChart(
      "chart-sync-cfo-acc",
      "CFO Estimation Accuracy vs. SNR",
      "CFO Accuracy (%)",
      { min: 0, max: 100, unit: "%", aspectRatio: 1.55 }
    ),
    stoAcc: makeSyncLineChart(
      "chart-sync-sto-acc",
      "STO Estimation Accuracy vs. SNR",
      "STO Accuracy (%)",
      { min: 0, max: 100, unit: "%", aspectRatio: 1.55 }
    ),
    cfoMae: makeSyncLineChart(
      "chart-sync-cfo-mae",
      "CFO Mean Absolute Error vs. SNR",
      "CFO MAE (Hz)",
      { isLog: true, logMin: 5, allowHalfTick: false, unit: " Hz", aspectRatio: 1.55 }
    ),
    stoMae: makeSyncLineChart(
      "chart-sync-sto-mae",
      "STO Mean Absolute Error vs. SNR",
      "STO MAE (samples)",
      { isLog: true, logMin: 0.4, allowHalfTick: true, unit: " samp", aspectRatio: 1.55 }
    ),
    payloadAcc: makeSyncLineChart(
      "chart-sync-payload-acc",
      "Compensated Payload Symbol Accuracy vs. SNR",
      "Payload Accuracy (%)",
      { min: 0, max: 100, unit: "%", aspectRatio: 2.1 }
    ),
  };
}

export function updateSyncExperimentCharts(charts, expData) {
  if (!charts || !expData) return;
  const snrs = expData.snr_list || [];
  const series = expData.series || {};
  const sfTag = expData.sf ? ` (SF${expData.sf})` : "";
  const allowedOrder = ["conv_single", "gloriphy_single", "loraseek_single"];
  const order = allowedOrder.filter((mKey) => series[mKey]);

  const buildDatasets = (metricKey, logFloor = null) =>
    order.map((mKey) => {
      const style = SYNC_SINGLE_STYLES[mKey] || { label: mKey, color: "#334155", pointStyle: "circle" };
      const rawVals = series[mKey]?.[metricKey] || [];
      const data = logFloor !== null
        ? rawVals.map((v) => (v === null || v === undefined ? null : Math.max(Number(v), logFloor)))
        : rawVals;
      return {
        label: style.label,
        data,
        borderColor: style.color,
        backgroundColor: style.color,
        borderWidth: 3.0,
        borderDash: [],
        tension: 0.14,
        clip: false,
        pointRadius: 5.5,
        pointHoverRadius: 7.2,
        pointStyle: style.pointStyle,
        pointBackgroundColor: style.color,
        pointBorderColor: "#ffffff",
        pointBorderWidth: 1.6,
        fill: false,
      };
    });

  const configs = [
    ["cfoAcc", "cfo_acc", null, `CFO Estimation Accuracy vs. SNR${sfTag}`],
    ["stoAcc", "sto_acc", null, `STO Estimation Accuracy vs. SNR${sfTag}`],
    ["cfoMae", "cfo_mae", 5, `CFO Mean Absolute Error vs. SNR${sfTag}`],
    ["stoMae", "sto_mae", 0.4, `STO Mean Absolute Error vs. SNR${sfTag}`],
    ["payloadAcc", "payload_acc", null, `Compensated Payload Symbol Accuracy vs. SNR${sfTag}`],
  ];

  for (const [chartKey, metricKey, logFloor, titleText] of configs) {
    const chart = charts[chartKey];
    if (!chart) continue;
    chart.options.plugins.title.text = titleText;
    chart.data.labels = snrs.map((s) => String(s));
    chart.data.datasets = buildDatasets(metricKey, logFloor);
    chart.update();
  }
}

export function exportSyncChartPng(charts, chartKey, sf = 7) {
  downloadChartPng(charts?.[chartKey], `sync_sf${sf}_${chartKey}.png`);
}
