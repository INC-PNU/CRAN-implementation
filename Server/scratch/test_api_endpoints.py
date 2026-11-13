import sys
import json
from pathlib import Path

server_dir = Path("Server").resolve()
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from dashboard_app import app

client = app.test_client()

# Test 1: GET /api/runs (Page 1)
res_runs = client.get("/api/runs")
print("GET /api/runs status:", res_runs.status_code)
runs_data = json.loads(res_runs.data)
print(f"Loaded {len(runs_data.get('runs', []))} past runs.")

# Test 2: POST /api/benchmark/single-frame (Page 2)
payload = {
    "sf": 7,
    "bw": 125000,
    "fs": 1000000,
    "n_symbols": 8,
    "snr_db": 10.0,
    "apply_cfo": False,
    "cfo_hz": 0.0,
    "apply_sto": False,
    "sto_samples": 0,
    "apply_compensation": True,
    "seed": 42
}
res_bench = client.post(
    "/api/benchmark/single-frame",
    data=json.dumps(payload),
    content_type="application/json"
)
print("POST /api/benchmark/single-frame status:", res_bench.status_code)
bench_data = json.loads(res_bench.data)
models = bench_data.get("models", {})
print("Returned models in benchmark:")
for k, v in models.items():
    print(f"  - {k}: {v.get('name')} | Acc: {v.get('accuracy_pct')}%")
