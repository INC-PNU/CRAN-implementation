"""
main3.py — Flask server that chains CALoRa preamble detection (main2.py) into the
classical CFO/STO estimator (detect_cfo_sto in utils/helper_function.py), then
demodulates the payload and scores two things per SNR:

  1. Offset-detection accuracy: how close the estimated CFO (Hz) and STO (samples)
     are to the ground-truth values the client injected, both as mean-absolute-error
     and as a tolerance-based success rate.
  2. Payload symbol demodulation accuracy: the estimated CFO/STO are used to crop and
     correct the payload, which is then demodulated and compared against the random
     ground-truth symbol sequence the client sent alongside the packet.

Route:  POST /upload_offset
        JSON payload: gateway_id, iq_data, bw, sf, fs, snr, offset, cfo, preamble,
                       payload_symbols (list[int] — ground-truth payload sent by client3.py)

This file does not modify or import anything that would change main2.py/client2.py's
behavior — it only imports main2's already-loaded CALoRa model/detector functions for
reuse. detect_cfo_sto's new `preamble_hint_symbol` parameter defaults to None, so
main.py/main2.py's own calls (which never pass it) are unaffected.
"""

from flask import Flask, request, jsonify
import sys
import os
import time
import threading
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── project utils ─────────────────────────────────────────────────────────────
from utils.helper_function import read_base64_convert_to_np, detect_cfo_sto
from utils.my_lora_utils import calculate_symbol_alliqfile_with_down_sampling
from utils.LoRa import LoRa

# ── Reuse CALoRa model + preamble detector already defined in main2.py ────────
# Importing main2 triggers its one-time model load (guarded by `if __name__ ==
# "__main__"` around app.run(), so the Flask server in main2.py never starts).
import main2

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

CALORA_THRESH = main2.CALORA_THRESH

# Tolerance for the offset "success rate" metric.
CFO_TOL_HZ = None       # computed per-request from bw/n_classes (half a frequency bin)
STO_TOL_SAMPLES = None  # computed per-request from fs/bw (~1 chip)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth STO helper
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_symmetric(x, n):
    """Wrap x into (-n/2, n/2] modulo n."""
    x_mod = x % n
    if x_mod > n / 2:
        x_mod -= n
    return x_mod


def true_fine_sto(numb_offset_samples: int, frame_per_symbol: int) -> float:
    """
    The client crops `numb_offset_samples` off the front of the TX buffer before
    noise is added. Relative to the receiver's fixed frame grid (which assumes no
    crop), that makes the real chirp edges appear `numb_offset_samples` EARLIER
    (mod one symbol) — i.e. a negative fine-STO in the same sign convention that
    detect_cfo_sto's cross-correlation lag (`lag_samples`) already uses downstream
    (`index*framePerSymbol + sto` as the payload crop start).
    """
    return _wrap_symmetric(-numb_offset_samples, frame_per_symbol)


# ─────────────────────────────────────────────────────────────────────────────
# Global stats & watchdog
# ─────────────────────────────────────────────────────────────────────────────
ended = False
index = 0


def create_stats():
    return {
        "n_total": 0,
        "preamble_missed": 0,
        "offset_fail": 0,
        "demod_fail": 0,
        "cfo_abs_err": [],
        "sto_abs_err": [],
        "cfo_correct": 0,
        "sto_correct": 0,
        # "symbol_total"/"symbol_correct": conditioned on reaching the demod stage
        # (excludes packets that never got that far, e.g. preamble/offset failures).
        "symbol_total": 0,
        "symbol_correct": 0,
        # "symbol_total_sent": every symbol actually transmitted this SNR bin,
        # counted for ALL packets regardless of outcome — so SymbolAccuracyPerPacket
        # below reflects end-to-end accuracy against everything that was sent, not
        # just the subset that survived to demodulation.
        "symbol_total_sent": 0,
    }


GLOBAL_STATS = defaultdict(create_stats)


class Watchdog:
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.last_update = time.time()

    def update(self):
        self.last_update = time.time()

    def is_timeout(self):
        return time.time() - self.last_update > self.timeout


wd = Watchdog(5)


def watchdog_loop():
    global ended
    while True:
        if wd.is_timeout() and not ended:
            ended = True
            print("\nTimeout! No upload_offset received.")
            wd.update()
            print("----------------- SUMMARY (Offset + Demod Evaluation) -----------------")
            header = (
                f"{'SNR':>5} | {'N':>5} | {'PreMiss':>7} | {'OffFail':>7} | "
                f"{'CFO_MAE':>9} | {'CFO_ok%':>7} | {'CFO_okPkt%':>10} | "
                f"{'STO_MAE':>8} | {'STO_ok%':>7} | {'STO_okPkt%':>10} | "
                f"{'SymAcc%':>7} | {'SymAccPkt%':>10}"
            )
            print(header)
            print("-" * len(header))

            csv_rows = []
            for snr in sorted(GLOBAL_STATS):
                s = GLOBAL_STATS[snr]
                n_total = s["n_total"]
                cfo_mae = float(np.mean(s["cfo_abs_err"])) if s["cfo_abs_err"] else float("nan")
                sto_mae = float(np.mean(s["sto_abs_err"])) if s["sto_abs_err"] else float("nan")
                n_offset_eval = len(s["cfo_abs_err"])
                # Conditioned on offset estimation having produced a result at all
                # (excludes preamble-miss/offset-fail packets, where cfo_est/sto_est
                # come back as None, from the denominator).
                cfo_ok_pct = (s["cfo_correct"] / n_offset_eval * 100) if n_offset_eval else 0.0
                sto_ok_pct = (s["sto_correct"] / n_offset_eval * 100) if n_offset_eval else 0.0
                # Unconditioned: correct vs every packet sent this SNR bin — a preamble
                # miss or a failed (None) offset estimate counts as wrong here, so this
                # is the true end-to-end offset-detection accuracy, not just accuracy
                # among the subset that happened to produce an estimate.
                cfo_ok_per_packet_pct = (s["cfo_correct"] / n_total * 100) if n_total else 0.0
                sto_ok_per_packet_pct = (s["sto_correct"] / n_total * 100) if n_total else 0.0
                # Conditioned on reaching the demod stage (excludes preamble/offset failures).
                sym_acc_pct = (s["symbol_correct"] / s["symbol_total"] * 100) if s["symbol_total"] else 0.0
                # Unconditioned: correct symbols vs every symbol actually sent this SNR bin,
                # across ALL packets (preamble/offset/demod failures count as 0 correct).
                sym_acc_per_packet_pct = (
                    s["symbol_correct"] / s["symbol_total_sent"] * 100
                ) if s["symbol_total_sent"] else 0.0

                csv_rows.append((snr, n_total, s["preamble_missed"], s["offset_fail"],
                                  cfo_mae, cfo_ok_pct, cfo_ok_per_packet_pct,
                                  sto_mae, sto_ok_pct, sto_ok_per_packet_pct,
                                  sym_acc_pct, sym_acc_per_packet_pct))

                print(
                    f"{snr:>5} | {n_total:>5} | {s['preamble_missed']:>7} | {s['offset_fail']:>7} | "
                    f"{cfo_mae:>9.2f} | {cfo_ok_pct:>6.2f}% | {cfo_ok_per_packet_pct:>9.2f}% | "
                    f"{sto_mae:>8.3f} | {sto_ok_pct:>6.2f}% | {sto_ok_per_packet_pct:>9.2f}% | "
                    f"{sym_acc_pct:>6.2f}% | {sym_acc_per_packet_pct:>9.2f}%"
                )

            import csv
            from datetime import datetime

            results_dir = Path(__file__).resolve().parent / "results3"
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = results_dir / f"offset_demod_results_{timestamp}.csv"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "SNR", "N", "PreambleMissed", "OffsetFail",
                    "CFO_MAE_Hz", "CFO_SuccessRate", "CFO_SuccessRatePerPacket",
                    "STO_MAE_samples", "STO_SuccessRate", "STO_SuccessRatePerPacket",
                    "SymbolAccuracy", "SymbolAccuracyPerPacket",
                ])
                for row in csv_rows:
                    (snr_val, n_total, pm, of, cfo_mae, cfo_ok, cfo_ok_pp,
                     sto_mae, sto_ok, sto_ok_pp, sym_acc, sym_acc_pp) = row
                    writer.writerow([snr_val, n_total, pm, of,
                                      round(cfo_mae, 4), round(cfo_ok, 4), round(cfo_ok_pp, 4),
                                      round(sto_mae, 4), round(sto_ok, 4), round(sto_ok_pp, 4),
                                      round(sym_acc, 4), round(sym_acc_pp, 4)])

            print(f"\nResults saved to: {csv_path}")
        time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Payload symbol comparison (mirrors main.py's length handling for the demod
# output, generalized to any ground-truth payload length instead of a hardcoded 10)
# ─────────────────────────────────────────────────────────────────────────────
def score_payload_symbols(demodulated, ground_truth):
    n_gt = len(ground_truth)
    gt = np.array(ground_truth)
    demod = np.array(demodulated)

    if len(demod) == n_gt + 1:
        demod = demod[1:]
    elif len(demod) == n_gt - 1:
        gt = gt[1:]
    elif len(demod) != n_gt:
        return None  # unrecoverable length mismatch

    n_compare = min(len(demod), len(gt))
    matches = int(np.sum(demod[:n_compare] == gt[:n_compare]))
    return n_compare, matches


# ─────────────────────────────────────────────────────────────────────────────
# Route: POST /upload_offset
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/upload_offset", methods=["POST"])
def upload_offset():
    global ended, index

    index += 1
    if index % 50 == 0:
        print("Packet no:", index)
    ended = False
    wd.update()  # reset the idle timer as soon as a request arrives, before any
                 # processing that could throw — otherwise an exception mid-request
                 # would skip every later wd.update() call and the watchdog could
                 # fire "Timeout!" even while requests are actively coming in.

    data = request.get_json()

    gateway_id      = data.get("gateway_id")
    b64_lora_signal = data.get("iq_data")
    bw              = data.get("bw", 125_000)
    sf              = data.get("sf", 7)
    fs              = data.get("fs", 1_000_000)
    snr             = data.get("snr")
    no_of_preamble  = data.get("preamble", 8)
    true_offset     = data.get("offset", 0)
    true_cfo        = data.get("cfo", 0.0)
    payload_symbols_gt = data.get("payload_symbols", [])

    np_lora_signal = read_base64_convert_to_np(b64_lora_signal)

    opts                = type("", (), {})()
    opts.sf             = sf
    opts.bw             = bw
    opts.fs             = fs
    opts.n_classes      = 2 ** sf
    opts.gateway_id     = gateway_id
    opts.no_of_preamble = no_of_preamble
    opts.sync_sym       = [8, 8]

    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))

    s = GLOBAL_STATS[snr]
    s["n_total"] += 1
    s["symbol_total_sent"] += len(payload_symbols_gt)

    # ── Stage 1: CALoRa neural preamble detection (reused from main2.py) ────────
    is_detected, t_start, t_end, mean_prob = main2.detect_preamble_calora(opts, np_lora_signal)
    if not is_detected:
        s["preamble_missed"] += 1
        wd.update()
        return jsonify({"status": "fail", "stage": "preamble_not_detected", "mean_prob": round(mean_prob, 4)}), 200

    # Convert the CALoRa spectrogram column back into a symbol-frame index, so
    # detect_cfo_sto can resume its downchirp/CFO/STO search from there directly.
    win_len = opts.n_classes // 2
    hop = win_len // 2
    cols_per_symbol = framePerSymbol // hop + 1
    hint_symbol_index = max(0, t_start // cols_per_symbol)
    print(t_start, t_start // cols_per_symbol)
    # ── Stage 2: classical CFO/STO estimation, hinted by CALoRa's location ──────
    index_payload, cfo_est, sto_est, preamble_found = detect_cfo_sto(
        opts, LoRa, np_lora_signal, preamble_hint_symbol=hint_symbol_index
    )
    print(cfo_est, sto_est, preamble_found)

    if index_payload in (-1, -2) or index_payload is None:
        s["offset_fail"] += 1
        wd.update()
        return jsonify({"status": "fail", "stage": "offset_estimation_failed"}), 200

    # ── Offset-detection accuracy ────────────────────────────────────────────────
    sto_true = true_fine_sto(true_offset, framePerSymbol)
    cfo_err = float(cfo_est - true_cfo)
    sto_err = float(sto_est - sto_true)

    cfo_tol_hz = opts.bw / (2 * opts.n_classes)      # half a frequency bin
    sto_tol_samples = opts.fs / opts.bw               # ~1 chip

    s["cfo_abs_err"].append(abs(cfo_err))
    s["sto_abs_err"].append(abs(sto_err))
    if abs(cfo_err) <= cfo_tol_hz:
        s["cfo_correct"] += 1
    if abs(sto_err) <= sto_tol_samples:
        s["sto_correct"] += 1

    # ── Stage 3: crop + correct + demodulate the payload ────────────────────────
    payload_start = int(index_payload * framePerSymbol) + int(sto_est)
    payload_signal = np_lora_signal[payload_start:]

    n = len(payload_signal)
    t_axis = np.arange(n) / opts.fs
    payload_corrected = payload_signal * np.exp(-1j * 2 * np.pi * cfo_est * t_axis)

    demod_symbols = calculate_symbol_alliqfile_with_down_sampling(
        payload_corrected, opts.sf, opts.bw, opts.fs, show=False
    )

    demod_result = score_payload_symbols(demod_symbols, payload_symbols_gt)
    if demod_result is None:
        s["demod_fail"] += 1
    else:
        n_compare, matches = demod_result
        s["symbol_total"] += n_compare
        s["symbol_correct"] += matches

    if index <= 5:
        print(
            f"[dbg #{index}] SNR={snr} true_cfo={true_cfo:.1f} est_cfo={cfo_est:.1f} "
            f"(err={cfo_err:+.1f}Hz) | true_sto={sto_true:.2f} est_sto={sto_est:.2f} "
            f"(err={sto_err:+.2f}) | demod={demod_symbols} gt={payload_symbols_gt}"
        )

    wd.update()
    return jsonify({
        "status": "success",
        "cfo_est": cfo_est,
        "sto_est": sto_est,
        "cfo_err": cfo_err,
        "sto_err": sto_err,
        "demod_symbols": [int(x) for x in demod_symbols],
        "payload_symbols_gt": payload_symbols_gt,
        "demod_matches": (None if demod_result is None else demod_result[1]),
        "demod_compared": (None if demod_result is None else demod_result[0]),
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=watchdog_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5006, debug=False, threaded=False)
