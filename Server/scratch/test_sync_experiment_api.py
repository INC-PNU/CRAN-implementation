import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_DIR = os.path.join(ROOT, "Server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from dashboard_app import app

client = app.test_client()

print("=== 1. Testing GET / ===")
r = client.get("/")
assert r.status_code == 200, f"GET / failed: {r.status_code}"
html = r.get_data(as_text=True)
required_ids = [
    'id="tab-sync-exp"',
    'id="page-sync-exp"',
    'id="chart-sync-cfo-acc"',
    'id="chart-sync-sto-acc"',
    'id="chart-sync-cfo-mae"',
    'id="chart-sync-sto-mae"',
    'id="chart-sync-payload-acc"',
    'id="sync-table-body"',
    'id="sync-run-btn"',
]
for req in required_ids:
    assert req in html, f"Missing {req} in index.html!"
print("GET / passed: all required DOM IDs present.")

print("\n=== 2. Testing GET /api/benchmark/sync-experiment ===")
r = client.get("/api/benchmark/sync-experiment")
assert r.status_code == 200, f"GET /api/benchmark/sync-experiment failed: {r.status_code}"
data = r.get_json()
assert "results" in data, f"Missing 'results' key: {list(data.keys())}"
assert "sf7" in data["results"] and "sf8" in data["results"], f"Missing sf7/sf8: {list(data['results'].keys())}"

for key in ["sf7", "sf8"]:
    ds = data["results"][key]
    snrs = ds["snr_list"]
    order = ds["methods_order"]
    assert order == ["conv_single", "gloriphy_single", "loraseek_single"], f"Unexpected methods_order: {order}"
    series = ds["series"]
    assert set(series.keys()) == {"conv_single", "gloriphy_single", "loraseek_single"}
    print(f"\n--- Saved dataset: {key.upper()} (N={ds['n_trials']}, SNRs={snrs}, elapsed={ds.get('elapsed_sec')}s) ---")
    print("Highlights:", ds.get("highlights"))
    header = (
        f"{'SNR':>5} | {'Method':<16} | {'CFO Acc':>7} | {'STO Acc':>7} | "
        f"{'CFO MAE(Hz)':>11} | {'STO MAE(s)':>10} | {'Pay Acc':>7} | {'Avg ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for i, snr in enumerate(snrs):
        for mid in order:
            s = series[mid]
            assert len(s["cfo_acc"]) == len(snrs)
            print(
                f"{snr:>5} | {mid:<16} | {s['cfo_acc'][i]:>6.1f}% | {s['sto_acc'][i]:>6.1f}% | "
                f"{s['cfo_mae'][i]:>11.1f} | {s['sto_mae'][i]:>10.2f} | "
                f"{s['payload_acc'][i]:>6.1f}% | {s['avg_ms'][i]:>7.1f}"
            )
        print("-" * len(header))

orig_sf7_n = data["results"]["sf7"]["n_trials"]
orig_sf8_n = data["results"]["sf8"]["n_trials"]

print("\n=== 3. Testing POST /api/benchmark/sync-experiment (save=False, fast smoke) ===")
payload = {
    "sf": 7,
    "snr_min": -20,
    "snr_max": -18,
    "snr_step": 2,
    "n_trials": 5,
    "seed": 42,
    "save": False,
}
r = client.post("/api/benchmark/sync-experiment", json=payload)
assert r.status_code == 200, f"POST failed: {r.status_code} {r.get_data(as_text=True)}"
res = r.get_json()
assert res["sf"] == 7
assert res["n_trials"] == 5
assert res["snr_list"] == [-20, -18]
assert set(res["series"].keys()) == {
    "conv_single",
    "gloriphy_single",
    "loraseek_single",
}
print(f"POST smoke test passed in {res.get('elapsed_sec')}s.")

# Test invalid parameters return 400
r_bad = client.post("/api/benchmark/sync-experiment", json={"sf": 9, "n_trials": 10})
assert r_bad.status_code == 400, f"Expected 400 for sf=9, got {r_bad.status_code}"
print("Invalid parameter guard passed (sf=9 -> 400).")

# Ensure saved JSON was NOT overwritten
r2 = client.get("/api/benchmark/sync-experiment")
d2 = r2.get_json()
assert d2["results"]["sf7"]["n_trials"] == orig_sf7_n
assert d2["results"]["sf8"]["n_trials"] == orig_sf8_n
print(f"Verified saved SF7 (N={orig_sf7_n}) & SF8 (N={orig_sf8_n}) datasets were NOT overwritten.")
print("\nALL SYNC EXPERIMENT API CHECKS PASSED!")
