import sys
import numpy as np

server_dir = r"c:\Users\priba\Sean-2025\INC-LAB\CRAN-implementation\Server"
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

from dashboard import orchestrator

client_config = {
    "sf": 7, "bw": 125000, "fs": 1000000,
    "n_packets": 40,
    "cfo_hz_range": [-4334, 3344],
    "sto_samp_range": [0, 0],
    "payload_len": 10,
    "snr_db_range": [-10, -10],
}

def test_config(name, toggles):
    server_config = {
        "preamble_detection_method": "calora",
        "calora_checkpoint": "premamble_best_finetuned_cfo_mix.pth",
        "symbol_decoding_method": "conventional",
        "toggles": toggles
    }
    opts = orchestrator._build_opts(client_config)
    fps = int(opts.n_classes * (opts.fs / opts.bw))
    rng = np.random.default_rng(12345)

    comp_tot = 0
    corr_tot = 0
    imperfect = 0
    for i in range(40):
        r = orchestrator._run_one_packet(opts, client_config, server_config, rng, None, fps)
        if r["outcome"] == "success":
            comp, corr = r["demod_result"]
            comp_tot += comp
            corr_tot += corr
            if corr < comp:
                imperfect += 1
    pct = (corr_tot / comp_tot * 100) if comp_tot else 0
    print(f"{name:<45}: {corr_tot}/{comp_tot} = {pct:6.2f}% (imperfect pkts: {imperfect})")

print("=== Comparing Toggles on Exact 40 Packets ===")
test_config("1. Run 2a57ecf (precorrect=False, sync_adj=True)", {
    "apply_cfo_correction": True, "apply_sto_correction": True,
    "precorrect_cfo_frac": False, "use_fractional_sto": True,
    "verify_network_id": False, "sync_payload_offset_adjust": True,
    "sto_method": "v3", "downchirp_search_radius": 1
})

test_config("2. Turn ON precorrect_cfo_frac=True", {
    "apply_cfo_correction": True, "apply_sto_correction": True,
    "precorrect_cfo_frac": True, "use_fractional_sto": True,
    "verify_network_id": False, "sync_payload_offset_adjust": True,
    "sto_method": "v3", "downchirp_search_radius": 1
})

test_config("3. precorrect=True AND sync_adj=False (main3 default)", {
    "apply_cfo_correction": True, "apply_sto_correction": True,
    "precorrect_cfo_frac": True, "use_fractional_sto": True,
    "verify_network_id": False, "sync_payload_offset_adjust": False,
    "sto_method": "v3", "downchirp_search_radius": 1
})

test_config("4. precorrect=False AND sync_adj=False", {
    "apply_cfo_correction": True, "apply_sto_correction": True,
    "precorrect_cfo_frac": False, "use_fractional_sto": True,
    "verify_network_id": False, "sync_payload_offset_adjust": False,
    "sto_method": "v3", "downchirp_search_radius": 1
})
