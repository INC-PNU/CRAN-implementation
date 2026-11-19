// Thin fetch() wrappers around the dashboard REST API. Every call is same-origin
// (this app serves both the API and the static files), so no CORS handling needed.

async function asJson(resp) {
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(body.detail || body.error || `HTTP ${resp.status}`);
    err.status = resp.status;
    err.body = body;
    throw err;
  }
  return body;
}

export function getCapabilities() {
  return fetch("/api/capabilities").then(asJson);
}

export function startRun(config) {
  return fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  }).then(asJson);
}

export function getRunStatus(runId) {
  return fetch(`/api/runs/${runId}/status`).then(asJson);
}

export function listRuns() {
  return fetch("/api/runs").then(asJson);
}

export function getRunDetail(runId) {
  return fetch(`/api/runs/${runId}`).then(asJson);
}

export function renameRun(runId, label) {
  return fetch(`/api/runs/${runId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  }).then(asJson);
}

export function cancelRun(runId) {
  return fetch(`/api/runs/${runId}/cancel`, { method: "POST" }).then(asJson);
}

export function deleteRun(runId) {
  return fetch(`/api/runs/${runId}`, { method: "DELETE" }).then(asJson);
}

export function runSingleFrameBenchmark(params) {
  return fetch("/api/benchmark/single-frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then(asJson);
}

export function getSyncExperimentResults() {
  return fetch("/api/benchmark/sync-experiment").then(asJson);
}

export function runSyncExperiment(params) {
  return fetch("/api/benchmark/sync-experiment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  }).then(asJson);
}
