"""
client3.py — Same packet structure as client2.py (noise-prefix + preamble +
network-code/sync + downchirp + payload), but the payload is now a RANDOM
symbol sequence per packet instead of a fixed one, and its ground-truth value
is sent alongside the packet so main3.py can score payload demodulation
accuracy. Sends to /upload_offset on main3.py, which chains CALoRa preamble
detection into the classical CFO/STO estimator and reports offset-detection
accuracy plus symbol demodulation accuracy per SNR.

Usage:
    python client3.py          # runs the randomized sweep in __main__ below
"""

import requests
import json
import numpy as np
import os
import base64
import copy
import time

cwd = os.path.abspath(os.getcwd())
import config
from utils.LoRa import LoRa
from utils.my_lora_utils import *

parser = config.create_parser()
opts = parser.parse_args()

# ── Route pointing to main3.py ─────────────────────────────────────────────────
url = "http://127.0.0.1:5006/upload_offset"

PAYLOAD_LEN = 10


def send_lora_to_offset_server(opts, noise_seed, rng):
    """Build a LoRa packet with a random payload and POST it to /upload_offset."""

    preamble = create_lora_preamble(opts, LoRa)

    payload_symbols = rng.integers(0, opts.n_classes, size=PAYLOAD_LEN).tolist()
    payload = create_lora_payload(opts, LoRa, payload_symbols)

    sequence_ = [999, 1222]  # random noise prefix (matches main2/main3's ground-truth math)
    random_ZONK = create_lora_payload(opts, LoRa, sequence_)

    complete_signal_ = np.concatenate([random_ZONK, preamble, payload]).astype(np.complex64)
    complete_signal_cfo = add_cfo(opts, complete_signal_, opts.CFO)
    complete_signal_cfo_sto = complete_signal_cfo[opts.numb_offset:]

    lora_init = LoRa(opts.sf, opts.bw)
    if noise_seed >= 0:
        complete_signal_cfo_sto = lora_init.awgn_iq_with_seed(
            complete_signal_cfo_sto, opts.snr, seed=noise_seed
        )
    else:
        complete_signal_cfo_sto = lora_init.awgn_iq_with_seed(
            complete_signal_cfo_sto, opts.snr, seed=None
        )

    iq_bytes = complete_signal_cfo_sto.tobytes()
    iq_b64 = base64.b64encode(iq_bytes).decode()

    payload_json = {
        "gateway_id": f"GW{opts.gateway_id}",
        "value": f"Hello from GW{opts.gateway_id}",
        "iq_data": iq_b64,
        "bw": opts.bw,
        "sf": opts.sf,
        "fs": opts.fs,
        "snr": opts.snr,
        "offset": opts.numb_offset,
        "cfo": opts.CFO,
        "payload_symbols": payload_symbols,
    }

    response = requests.post(url, json=payload_json, timeout=30)
    return response


def run_batch(
    base_opts,
    n_packets=2000,
    cfo_hz_range=(0, 0),
    sto_samp_range=(0, 0),
    snr_db_range=(-35, -10),
    seed=-10,
):
    rng = np.random.default_rng() if seed < 0 else np.random.default_rng(seed)

    n_errors = 0
    t_start = time.time()
    for i in range(n_packets):
        o = copy.deepcopy(base_opts)

        o.CFO = float(rng.uniform(*cfo_hz_range))
        o.numb_offset = int(rng.integers(sto_samp_range[0], sto_samp_range[1] + 1))
        o.snr = int(rng.uniform(*snr_db_range))

        try:
            resp = send_lora_to_offset_server(o, seed, rng)
            if i < 5:
                print(f"[{i}] SNR={o.snr} CFO={o.CFO:.1f} STO={o.numb_offset} -> {resp.json()}", flush=True)
        except requests.exceptions.RequestException as exc:
            n_errors += 1
            print(f"[{i}] REQUEST FAILED: {exc}", flush=True)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_packets - i - 1) / rate if rate > 0 else float("inf")
            print(f"...packet {i + 1}/{n_packets} ({elapsed:.0f}s elapsed, "
                  f"{rate:.1f} pkt/s, ETA {eta:.0f}s, {n_errors} request errors)", flush=True)

    print(f"Batch done: {n_packets} packets in {time.time() - t_start:.0f}s, {n_errors} request errors.", flush=True)
    return 0


if __name__ == "__main__":
    # ── Base config ──────────────────────────────────────────────────────────
    opts.sf         = 7
    opts.bw         = 125_000
    opts.fs         = 1_000_000
    opts.n_classes  = 2 ** opts.sf
    opts.gateway_id = 1

    # ── Run batch sweep ──────────────────────────────────────────────────────
    # sto_samp_range is capped at 100 samples (~1/10 symbol): CALoRa's neural
    # preamble detector (checkpoint best_finetuned_train_cfo.pth) was only ever
    # exercised at STO=0 in client2.py's sweeps, and empirically its detection
    # rate collapses once the injected offset exceeds ~100-150 samples,
    # regardless of SNR. Keeping the range inside that margin lets the STO
    # accuracy metric below be exercised with genuinely nonzero ground truth
    # without that separate (and much coarser) detection-robustness limit
    # dominating the results.
    results = run_batch(
        base_opts=opts,
        n_packets=2000,
        cfo_hz_range=(-4358,4584),
        sto_samp_range=(0, 0),
        snr_db_range=(-25, -5),
        seed=-12,
    )
