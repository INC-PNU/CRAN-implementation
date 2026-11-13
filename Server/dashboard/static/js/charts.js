// Chart.js setup + rendering. Charts are created once and updated in place via
// chart.update() on every poll (never destroyed/recreated) so Chart.js's own animation
// carries old->new positions smoothly instead of a jarring redraw.

export const PALETTE = [
  "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
  "#e87ba4", "#008300", "#4a3aa7", "#d03b3b",
];
export const MAX_OVERLAY_RUNS = PALETTE.length;

const COLOR_KEY = "lora_dashboard_run_colors";

function loadColorMap() {
  try {
    return JSON.parse(localStorage.getItem(COLOR_KEY)) || {};
  } catch {
    return {};
  }
}

function saveColorMap(map) {
  try {
    localStorage.setItem(COLOR_KEY, JSON.stringify(map));
  } catch {
    /* private browsing / storage disabled — color assignment just won't persist */
  }
}

// Assigned by first-selected order, persisted, never recycled on deselect — so a run
// keeps its color across the session and unchecking one run never repaints the others.
export function assignColor(runId) {
  const map = loadColorMap();
  if (map[runId] !== undefined) return PALETTE[map[runId] % PALETTE.length];
  const used = new Set(Object.values(map));
  let slot = 0;
  while (used.has(slot) && slot < PALETTE.length) slot++;
  if (slot >= PALETTE.length) slot = Object.keys(map).length % PALETTE.length;
  map[runId] = slot;
  saveColorMap(map);
  return PALETTE[slot];
}

function pct(numerator, denominator) {
  return denominator ? (numerator / denominator) * 100 : null;
}

// Raw bins[] -> per-metric {snr: value} for the given scope. Both scopes are always
// derivable from the same raw counts, so switching scope needs zero network requests.
export function computeSeries(bins, scope) {
  const out = { preamble: {}, cfo: {}, sto: {}, symbol: {}, n: {} };
  for (const b of bins) {
    // ?? 0: manifests written before preamble_mislocated existed lack the field, and
    // read back as the old (inflated) behaviour rather than NaN.
    const mislocated = b.preamble_mislocated ?? 0;
    // "Detected" requires landing within tolerance of the true preamble, not merely
    // answering yes. CALoRa's detector cannot answer no at all (argmax over sliding
    // windows + a confidence gate its probability floor never trips), so without this
    // term its rate read ~100% down to -40 dB.
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
  return {
    label: run.label,
    data: snrLabels.map((s) => (values[s] === undefined ? null : values[s])),
    borderColor: run.color,
    backgroundColor: run.color,
    borderDash: run.dashed ? [6, 4] : [],
    spanGaps: true,
    tension: 0.15,
    pointRadius: 3,
  };
}

function makeLineChart(canvasId, yLabel) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      animation: { duration: 250 },
      scales: {
        x: { title: { display: true, text: "SNR (dB)" } },
        y: { title: { display: true, text: yLabel }, min: 0, max: 100 },
      },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            afterLabel(item) {
              const n = item.chart._nCounts?.[item.datasetIndex]?.[item.dataIndex];
              return n !== undefined ? `N=${n}` : "";
            },
          },
        },
      },
    },
  });
}

export function createCharts() {
  return {
    preamble: makeLineChart("chart-preamble", "Detection rate (%)"),
    cfo: makeLineChart("chart-cfo", "Success rate (%)"),
    sto: makeLineChart("chart-sto", "Success rate (%)"),
    symbol: makeLineChart("chart-symbol", "Accuracy (%)"),
    nbin: new Chart(document.getElementById("chart-nbin").getContext("2d"), {
      type: "bar",
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        animation: { duration: 250 },
        scales: {
          x: { title: { display: true, text: "SNR (dB)" } },
          y: { title: { display: true, text: "Packets" }, beginAtZero: true },
        },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
      },
    }),
  };
}

// runs: array of {runId, label, color, bins, dashed}
export function updateCharts(charts, runs, scope) {
  const snrSet = new Set();
  const seriesPerRun = runs.map((r) => ({ run: r, series: computeSeries(r.bins, scope) }));
  for (const { series } of seriesPerRun) {
    Object.keys(series.preamble).forEach((s) => snrSet.add(Number(s)));
  }
  const snrLabels = [...snrSet].sort((a, b) => a - b);

  for (const key of ["preamble", "cfo", "sto", "symbol"]) {
    const chart = charts[key];
    chart.data.labels = snrLabels;
    chart.data.datasets = seriesPerRun.map(({ run, series }) => lineDataset(run, series[key], snrLabels));
    chart._nCounts = seriesPerRun.map(({ series }) => snrLabels.map((s) => series.n[s]));
    chart.update();
  }

  charts.nbin.data.labels = snrLabels;
  charts.nbin.data.datasets = seriesPerRun.map(({ run, series }) => ({
    label: run.label,
    data: snrLabels.map((s) => series.n[s] ?? 0),
    backgroundColor: run.color,
  }));
  charts.nbin.update();
}

export function createSingleFrameCharts() {
  const modelColors = {
    basic_single_band: "#94a3b8",
    dechirp_downsampled: "#64748b",
    conventional: "#10b981",
    nelora: "#f59e0b",
    calora: "#3b82f6",
    loraseek: "#8b5cf6",
    gloriphy: "#ec4899",
  };

  const accEl = document.getElementById("chart-single-acc");
  const latEl = document.getElementById("chart-single-latency");
  if (!accEl || !latEl) return null;

  const accChart = new Chart(accEl.getContext("2d"), {
    type: "bar",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      animation: { duration: 300 },
      scales: {
        y: {
          min: 0,
          max: 100,
          title: { display: true, text: "Accuracy (%)" },
          ticks: { callback: (v) => v + "%" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => `Accuracy: ${item.raw}%`,
          },
        },
      },
    },
  });

  const latChart = new Chart(latEl.getContext("2d"), {
    type: "bar",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      animation: { duration: 300 },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Latency (ms)" },
          ticks: { callback: (v) => v + " ms" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => `Latency: ${item.raw} ms`,
          },
        },
      },
    },
  });

  return { acc: accChart, latency: latChart, colors: modelColors };
}

export function updateSingleFrameCharts(charts, modelsData) {
  if (!charts) return;
  const labels = [];
  const accValues = [];
  const latValues = [];
  const bgColors = [];

  const order = [
    "basic_single_band",
    "dechirp_downsampled",
    "conventional",
    "nelora",
    "calora",
    "loraseek",
    "gloriphy",
  ];
  const shortNames = {
    basic_single_band: "FFT (Decimation 1/8)",
    dechirp_downsampled: "FFT (Downsample Polyphase)",
    conventional: "FFT (Without Downsample) [1]",
    nelora: "NeLoRa",
    calora: "CALoRa",
    loraseek: "LoRaSeek",
    gloriphy: "GLoRiPhy",
  };
  for (const key of order) {
    const m = modelsData[key];
    if (!m) continue;
    labels.push(shortNames[key] || m.name);
    accValues.push(m.available && m.accuracy_pct !== undefined ? m.accuracy_pct : 0);
    latValues.push(m.available && m.execution_time_ms !== undefined ? m.execution_time_ms : 0);
    bgColors.push(m.available ? charts.colors[key] : "#cbd5e1");
  }

  charts.acc.data.labels = labels;
  charts.acc.data.datasets = [
    {
      data: accValues,
      backgroundColor: bgColors,
      borderRadius: 4,
    },
  ];
  charts.acc.update();

  charts.latency.data.labels = labels;
  charts.latency.data.datasets = [
    {
      data: latValues,
      backgroundColor: bgColors,
      borderRadius: 4,
    },
  ];
  charts.latency.update();
}
