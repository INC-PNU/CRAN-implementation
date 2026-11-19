import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.LoRa import LoRa
from utils.my_lora_utils import create_lora_preamble, create_lora_payload
from utils import gloriphy_inference

SF = 8
BW = 125_000
FS = 1_000_000
N_SYMBOLS = 1000
SNR_LIST = [-21, -20, -19, -18, -17, -16, -15]
SEED = 42

VARIANTS = [
    "sf8/awgn_filter",
    "sf8/awgn_filter_retrained",
    "sf8/denoisegen_simulation",
    "sf8/denoisegen_real_world",
]

SHORT_NAMES = {
    "sf8/awgn_filter": "awgn_filter",
    "sf8/awgn_filter_retrained": "awgn_retrained",
    "sf8/denoisegen_simulation": "denoisegen_sim",
    "sf8/denoisegen_real_world": "denoisegen_real",
}


def eval_variant_chunked(payload_iq, preamble_iq, sfd_iq, ckpt_id, chunk_syms=100):
    nsamp = (2 ** SF) * (FS // BW)
    preds = []
    for i in range(0, N_SYMBOLS, chunk_syms):
        sub_iq = payload_iq[i * nsamp : (i + chunk_syms) * nsamp]
        p = gloriphy_inference.demodulate_payload_gloriphy(
            sub_iq,
            SF,
            BW,
            FS,
            checkpoint_id=ckpt_id,
            preamble_iq=preamble_iq,
            sfd_iq=sfd_iq,
        )
        preds.extend(p)
    return preds[:N_SYMBOLS]


def main():
    n_classes = 2 ** SF
    fps = int(n_classes * (FS / BW))

    opts = type("", (), {})()
    opts.sf = SF
    opts.bw = BW
    opts.fs = FS
    opts.n_classes = n_classes
    opts.no_of_preamble = 8
    opts.sync_sym = [8, 8]

    # Pre-warm models
    for v in VARIANTS:
        info = gloriphy_inference.checkpoint_info(v, SF)
        gloriphy_inference._load(info)

    print(f"GLoRiPHY SF8 4-Checkpoint Comparison (N={N_SYMBOLS} symbols/SNR, seed={SEED})\n")
    header = f"{'SNR (dB)':>8} | " + " | ".join(f"{SHORT_NAMES[v]:>16}" for v in VARIANTS)
    print(header)
    print("-" * len(header), flush=True)

    results = {v: {} for v in VARIANTS}
    timings = {v: [] for v in VARIANTS}

    lora = LoRa(SF, BW)

    for snr_db in SNR_LIST:
        rng = np.random.default_rng(SEED + abs(snr_db) * 100)
        gt_symbols = rng.integers(0, n_classes, size=N_SYMBOLS).tolist()

        preamble = create_lora_preamble(opts, LoRa)
        payload = create_lora_payload(opts, LoRa, gt_symbols)
        sig = np.concatenate([preamble, payload]).astype(np.complex64)

        iq = lora.awgn_iq_with_seed(sig, snr_db, seed=SEED + abs(snr_db) * 100)

        payload_start = int(12.25 * fps)
        payload_iq = iq[payload_start : payload_start + N_SYMBOLS * fps]
        preamble_iq = iq[0 : int(8 * fps)]
        sfd_iq = iq[int(10 * fps) : payload_start]

        row_cells = []
        for v in VARIANTS:
            t0 = time.perf_counter()
            preds = eval_variant_chunked(payload_iq, preamble_iq, sfd_iq, v, chunk_syms=100)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings[v].append(elapsed_ms)

            matched = int(np.sum(np.array(preds) == np.array(gt_symbols)))
            acc = (matched / N_SYMBOLS) * 100.0
            results[v][snr_db] = (acc, matched)
            row_cells.append(f"{acc:6.1f}% ({matched:4d})")

        print(f"{snr_db:8d} | " + " | ".join(f"{c:>16}" for c in row_cells), flush=True)

    print("-" * len(header))
    mean_cells = [
        f"{np.mean([results[v][s][0] for s in SNR_LIST]):6.2f}% avg"
        for v in VARIANTS
    ]
    print(f"{'Mean Acc':>8} | " + " | ".join(f"{c:>16}" for c in mean_cells))

    time_cells = [f"{np.mean(timings[v]):7.1f} ms" for v in VARIANTS]
    print(f"{'Avg Time':>8} | " + " | ".join(f"{c:>16}" for c in time_cells), flush=True)


if __name__ == "__main__":
    main()
