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
reuse. detect_cfo_sto's `preamble_hint_symbol`/`downchirp_hint_symbol` parameters both
default to None, so main.py/main2.py's own calls (which never pass them) are unaffected.

CFO/STO are solved in closed form from the classic up/down chirp pair: CALoRa's t_end
locates the up-chirp run, +2 sync symbols locates the first down-chirp, and the up-chirp
is taken a fixed 5 frames before it. Dechirping each against the opposite reference gives
two tones whose sum yields CFO_int and whose difference yields STO_int.
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


# ═════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT TOGGLES — edit, restart this server, then run client3.py
#  Every combination writes its own CSV (the mode tag goes in the filename), so
#  runs never overwrite each other and stay traceable.
# ═════════════════════════════════════════════════════════════════════════════

# ── Demodulation stage (this file) ───────────────────────────────────────────
# Multiply the payload by exp(-j2.pi.cfo_est.t) before demodulating.
# OFF -> every symbol comes out shifted by round(CFO / (bw/n_classes)), the same
#        constant for the whole packet (cyclic, mod n_classes). Symbol accuracy
#        collapses to roughly the fraction of packets whose CFO happens to fall
#        inside half a bin.
APPLY_CFO_CORRECTION = False

# Add sto_est to the payload crop point.
# OFF -> the payload is cropped on the nominal frame grid instead, so the whole
#        injected timing offset stays uncompensated.
APPLY_STO_CORRECTION = True

# ── Estimation stage (utils/helper_function.py, CALoRa path only) ────────────
# VERSI 1.1: remove the fractional CFO from the signal before the FFT peaks are
# read. OFF -> the fractional CFO leaks into the STO estimate as up to ±0.5 chip,
# and pushes the integer CFO onto the wrong bin whenever the fraction nears 0.5.
PRECORRECT_CFO_FRAC = True

# Add shift_sto_index, the sub-chip STO refinement.
# OFF -> STO is quantised to whole chips (fs/bw = 8 samples at SF7/1 MHz).
USE_FRACTIONAL_STO = True

# Decode the 2 network-id/sync symbols and reject the packet if neither matches.
# ON -> reproduces the conventional guard. It only rejects; it cannot improve an
#       estimate, and a sync decode needs far more SNR than locating a down-chirp.
VERIFY_NETWORK_ID = False

# How many frames either side of the CALoRa hint to search for the down-chirp.
DOWNCHIRP_SEARCH_RADIUS = 1


def mode_tag():
    """Compact, filename-safe summary of the toggles above."""
    return (f"cfo{int(APPLY_CFO_CORRECTION)}"
            f"_sto{int(APPLY_STO_CORRECTION)}"
            f"_frac{int(PRECORRECT_CFO_FRAC)}"
            f"_fsto{int(USE_FRACTIONAL_STO)}"
            f"_nid{int(VERIFY_NETWORK_ID)}"
            f"_r{DOWNCHIRP_SEARCH_RADIUS}")


def print_modes():
    print("=" * 62)
    print("  ACTIVE MODES".ljust(46) + mode_tag())
    print("=" * 62)
    for name, value in (
        ("APPLY_CFO_CORRECTION", APPLY_CFO_CORRECTION),
        ("APPLY_STO_CORRECTION", APPLY_STO_CORRECTION),
        ("PRECORRECT_CFO_FRAC", PRECORRECT_CFO_FRAC),
        ("USE_FRACTIONAL_STO", USE_FRACTIONAL_STO),
        ("VERIFY_NETWORK_ID", VERIFY_NETWORK_ID),
    ):
        print(f"  {'ON ' if value else 'off'}  {name}")
    print(f"       DOWNCHIRP_SEARCH_RADIUS = {DOWNCHIRP_SEARCH_RADIUS}")
    print("=" * 62)

# Tolerance for the offset "success rate" metric.
CFO_TOL_HZ = None       # computed per-request from bw/n_classes (half a frequency bin)
STO_TOL_SAMPLES = None  # computed per-request from fs/bw (~1 chip)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth STO helper
# ─────────────────────────────────────────────────────────────────────────────
# client3.py prefixes every packet with 2 frames of noise (its `sequence_ = [999, 1222]`,
# symbols out of range so create_lora_payload emits random samples) before the preamble.
N_NOISE_FRAMES = 2


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
            print_modes()
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
            csv_path = results_dir / f"offset_demod_results_{timestamp}_{mode_tag()}.csv"

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

    # Convert CALoRa's spectrogram columns back into a symbol-frame index, so
    # detect_cfo_sto can jump straight to the down-chirp instead of scanning for it.
    #
    # CALoRa is trained to label only the 8 up-chirps (its Ls candidates are 262/264/266
    # columns ~ 8 symbols), so t_end is the LAST column of the up-chirp run and t_end+1 is
    # the first column of the network-id/sync pair. Rounding that to a whole symbol boundary
    # gives the sync frame; the 2 sync symbols then put the first down-chirp 2 frames later.
    win_len = opts.n_classes // 2
    hop = win_len // 2
    cols_per_symbol = framePerSymbol // hop + 1
    sync_symbol_index = int(round((t_end + 1) / cols_per_symbol))
    downchirp_hint_index = max(0, sync_symbol_index + 2)

    # ── Stage 2: classical CFO/STO estimation, hinted by CALoRa's location ──────
    index_payload, cfo_est, sto_est, preamble_found = detect_cfo_sto(
        opts, LoRa, np_lora_signal,
        downchirp_hint_symbol=downchirp_hint_index,
        precorrect_cfo_frac=PRECORRECT_CFO_FRAC,
        use_fractional_sto=USE_FRACTIONAL_STO,
        verify_network_id=VERIFY_NETWORK_ID,
        downchirp_search_radius=DOWNCHIRP_SEARCH_RADIUS,
    )

    sto_true = true_fine_sto(true_offset, framePerSymbol)

    # ── Debug trace for the first few packets ────────────────────────────────────
    # Printed BEFORE the failure check below, so a packet that fails offset estimation
    # still shows what it should have found. Ground truth follows client3.py's layout:
    # 2 noise frames, then 8 up-chirps + 2 network-id symbols + 2.25 down-chirps, then
    # the payload — with `true_offset` samples cropped off the front afterwards.
    if index <= 5 or (index > 1000 and snr == -7):
        gt_preamble_sample = N_NOISE_FRAMES * framePerSymbol - true_offset
        gt_downchirp_frame = N_NOISE_FRAMES + no_of_preamble + 2      # past the 2 sync symbols
        gt_payload_frame = gt_downchirp_frame + 2.25                  # past the 2.25 down-chirps
        # Column index the same way iq_to_network_input builds it: every symbol chunk is
        # STFT'd on its own into cols_per_symbol columns, so a sample maps to
        # chunk*cols_per_symbol + (offset within chunk)/hop.
        gt_preamble_col = ((gt_preamble_sample // framePerSymbol) * cols_per_symbol
                           + (gt_preamble_sample % framePerSymbol) / hop)

        print(f"\n[dbg #{index}] SNR={snr} dB " + "-" * 52)
        print(f"   ground truth : cfo={true_cfo:.1f} Hz | sto={sto_true:.1f} samples "
              f"(client cropped {true_offset})")
        print(f"   preamble at  : sample={gt_preamble_sample} | frame={N_NOISE_FRAMES} "
              f"| spec column={gt_preamble_col:.1f}")
        print(f"   CALoRa       : t_start={t_start} (gt {gt_preamble_col:.1f}) | t_end={t_end} "
              f"| mean_prob={mean_prob:.4f}")
        print(f"   down-chirp   : frame={downchirp_hint_index} (gt {gt_downchirp_frame})")
        if cfo_est is None:
            print(f"   estimate     : FAILED at offset estimation (index_payload={index_payload})")
        else:
            print(f"   estimate     : cfo={cfo_est:.1f} Hz (err={cfo_est - true_cfo:+.1f}) "
                  f"| sto={sto_est:.1f} samples (err={sto_est - sto_true:+.1f})")
            print(f"   payload at   : frame={index_payload:.2f} (gt {gt_payload_frame:.2f})")

    if index_payload in (-1, -2) or index_payload is None:
        s["offset_fail"] += 1
        wd.update()
        return jsonify({"status": "fail", "stage": "offset_estimation_failed"}), 200

    # ── Offset-detection accuracy ────────────────────────────────────────────────
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
    # The estimates above are always computed and always scored; the toggles only
    # decide whether they are actually APPLIED here, so the offset metrics stay
    # comparable across modes while the demod metric shows what compensation buys.
    payload_start = int(index_payload * framePerSymbol)
    if APPLY_STO_CORRECTION:
        payload_start += int(sto_est)
    payload_signal = np_lora_signal[payload_start:]

    if APPLY_CFO_CORRECTION:
        t_axis = np.arange(len(payload_signal)) / opts.fs
        payload_corrected = payload_signal * np.exp(-1j * 2 * np.pi * cfo_est * t_axis)
    else:
        payload_corrected = payload_signal

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

    if index <= 5  or (index > 1000 and snr == -7):
        # cfo/sto and their errors are already in the debug block above — only the
        # demodulated payload is left to report here.
        n_ok = "length mismatch" if demod_result is None else f"{demod_result[1]}/{demod_result[0]} correct"
        print(f"   demod        : {[int(x) for x in demod_symbols]}")
        print(f"   payload gt   : {payload_symbols_gt}  -> {n_ok}")

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
    print_modes()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5006, debug=False, threaded=False)
