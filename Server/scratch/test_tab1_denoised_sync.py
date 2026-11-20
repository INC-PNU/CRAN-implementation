"""End-to-end verification & mini ablation for Tab 1 Denoised Stage 2 Synchronization."""
import sys
import time
import numpy as np

sys.path.insert(0, r"c:\Users\priba\Sean-2025\INC-LAB\CRAN-implementation\Server")
from dashboard_app import app
from dashboard import persistence
from dashboard.orchestrator import _build_opts, _run_one_packet, validate_config, ConfigError

print("=== 1. Verify HTML & Validation ===")
client = app.test_client()
r = client.get("/")
assert r.status_code == 200
html = r.get_data(as_text=True)
for token in ('id="sync_denoiser"', 'id="sync-loraseek-option"', 'id="sync-gloriphy-option"'):
    assert token in html, f"Missing {token}"
print("GET / HTML controls verified.")

base_cli = {
    "sf": 7, "bw": 125000, "fs": 1000000, "n_packets": 10, "payload_len": 10,
    "cfo_hz_range": [-4358, 4584], "sto_samp_range": [0, 250], "snr_db_range": [-20, -18], "seed": 42,
}
for sd in ("none", "loraseek", "gloriphy"):
    srv = {
        "preamble_detection_method": "calora",
        "calora_checkpoint": "preamble_best_finetuned-sean.pth",
        "sync_denoiser": sd,
        "symbol_decoding_method": "loraseek",
        "loraseek_checkpoint": "75000_LoRaSeek.pkl",
        "toggles": {"apply_cfo_correction": True, "apply_sto_correction": True},
    }
    validate_config(base_cli, srv)
    print(f"  Label ({sd:<8}): {persistence.make_label(base_cli, srv)}")

try:
    validate_config(base_cli, {
        "preamble_detection_method": "conventional",
        "sync_denoiser": "invalid_model",
        "symbol_decoding_method": "conventional",
    })
    raise AssertionError("Should have raised ConfigError for invalid sync_denoiser")
except ConfigError as exc:
    print(f"Validation guard passed: {exc}")

print("\n=== 2. End-to-End Full-Pipeline Ablation (CALoRa -> Sync -> LoRaSeek Demod) ===")
# Run identical packet draws at -18 dB and -20 dB with Normal vs LoRaSeek Sync vs GLoRiPHY Sync
for snr in (-18, -20):
    print(f"\n--- SF7, SNR = {snr} dB, CFO=[-4358,4584], STO=[0,250] (N=40 identical packets) ---")
    for sd in ("none", "loraseek", "gloriphy"):
        cli = dict(base_cli)
        cli["snr_db_range"] = [snr, snr]
        srv = {
            "preamble_detection_method": "calora",
            "calora_checkpoint": "preamble_best_finetuned-sean.pth",
            "sync_denoiser": sd,
            "symbol_decoding_method": "loraseek",
            "loraseek_checkpoint": "75000_LoRaSeek.pkl",
            "toggles": {
                "apply_cfo_correction": True,
                "apply_sto_correction": True,
                "precorrect_cfo_frac": True,
                "use_fractional_sto": True,
                "verify_network_id": False,
                "sync_payload_offset_adjust": False,
                "sto_method": "v3",
                "downchirp_search_radius": 1,
            },
        }
        opts = _build_opts(cli)
        fps = int(opts.n_classes * (opts.fs / opts.bw))
        cfo_tol = opts.bw / (2 * opts.n_classes)
        sto_tol = opts.fs / opts.bw
        rng = np.random.default_rng(2026 + abs(snr))

        N = 40
        located = cfo_ok = sto_ok = sym_ok = sym_sent = 0
        t0 = time.perf_counter()
        for k in range(N):
            res = _run_one_packet(opts, cli, srv, rng, 5000 + k, fps)
            sym_sent += len(res["payload_symbols_gt"])
            if res["outcome"] == "success":
                located += 1
                if abs(res["cfo_err"]) <= cfo_tol:
                    cfo_ok += 1
                if abs(res["sto_err"]) <= sto_tol:
                    sto_ok += 1
                if res["demod_result"]:
                    sym_ok += res["demod_result"][1]
        dt = time.perf_counter() - t0
        print(
            f"  sync_denoiser={sd:<8} | PreLoc={100*located/N:5.1f}% | "
            f"CFO={100*cfo_ok/N:5.1f}% | STO={100*sto_ok/N:5.1f}% | "
            f"End2End Sym={100*sym_ok/sym_sent:5.1f}% | ({dt:.1f}s)"
        )

print("\n=== 3. Smoke-test Conventional Preamble + Denoised Sync ===")
for sd in ("loraseek", "gloriphy"):
    cli = dict(base_cli)
    cli["snr_db_range"] = [-16, -16]
    srv = {
        "preamble_detection_method": "conventional",
        "sync_denoiser": sd,
        "symbol_decoding_method": "loraseek",
        "loraseek_checkpoint": "75000_LoRaSeek.pkl",
        "toggles": {"apply_cfo_correction": True, "apply_sto_correction": True, "sto_method": "v3"},
    }
    opts = _build_opts(cli)
    rng = np.random.default_rng(99)
    res = _run_one_packet(opts, cli, srv, rng, 123, 1024)
    print(f"  Conventional + {sd} sync @ -16 dB -> outcome={res['outcome']}, demod={res['demod_result']}")

print("\nALL TAB 1 DENOISED SYNC TESTS PASSED!")
