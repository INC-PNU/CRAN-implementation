"""
client2.py — Same as client.py but sends to the /upload_calora endpoint on port 5001,
             which uses the CALoRa PreambleDetector model instead of classical DSP.

Usage:
    python client2.py          # runs the batch test (same params as client.py)
"""

import requests
import json
import numpy as np
import os
import torch
import base64
import copy

cwd = os.path.abspath(os.getcwd())
import config
from utils.LoRa import LoRa
from utils.my_lora_utils import *

parser = config.create_parser()
opts = parser.parse_args()

# ── Route pointing to main2.py ─────────────────────────────────────────────────
url = "http://127.0.0.1:5005/upload_calora"


def send_lora_to_calora_server(opts, noise_seed):
    """Build a LoRa packet (identical to client.py) and POST it to /upload_calora."""

    preamble        = create_lora_preamble(opts, LoRa)
    sequence        = [56, 22, 5, 32, 56, 12]
    payload         = create_lora_payload(opts, LoRa, sequence)

    sequence_       = [999, 1222]                           # random noise prefix
    random_ZONK     = create_lora_payload(opts, LoRa, sequence_)

    complete_signal_  = np.concatenate([random_ZONK, preamble, payload]).astype(np.complex64)
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
    iq_b64   = base64.b64encode(iq_bytes).decode()

    payload_json = {
        "gateway_id": f"GW{opts.gateway_id}",
        "value":      f"Hello from GW{opts.gateway_id}",
        "iq_data":    iq_b64,
        "bw":         opts.bw,
        "sf":         opts.sf,
        "fs":         opts.fs,
        "snr":        opts.snr,
        "offset":     opts.numb_offset,
        "cfo":        opts.CFO,
    }

    response = requests.post(url, json=payload_json)
    return response


def run_batch(
    base_opts,
    n_packets=10,
    cfo_hz_range=(0, 0),
    sto_samp_range=(0, 0),
    snr_db_range=(-15, 10),
    seed=1234,
):
    rng = np.random.default_rng() if seed < 0 else np.random.default_rng(seed)

    for i in range(n_packets):
        o = copy.deepcopy(base_opts)

        o.CFO         = float(rng.uniform(*cfo_hz_range))
        o.numb_offset = int(rng.integers(sto_samp_range[0], sto_samp_range[1] + 1))
        o.snr         = int(rng.uniform(*snr_db_range))

        send_lora_to_calora_server(o, seed)

    return 0


# ── Base config ────────────────────────────────────────────────────────────────
opts.sf         = 7
opts.bw         = 125_000
opts.fs         = 1_000_000
opts.n_classes  = 2 ** opts.sf
opts.gateway_id = 1

# ── Run batch test ─────────────────────────────────────────────────────────────
results = run_batch(
    base_opts=opts,
    n_packets=1000,
    cfo_hz_range=(0, 0),
    sto_samp_range=(0, 0),
    snr_db_range=(-25, 10),
    seed=-11,
)
