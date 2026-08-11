"""
main2.py — Flask server for LoRa preamble detection via CALoRa (PreambleDetector).

Route:  POST /upload_calora
        Same JSON payload as /upload (gateway_id, iq_data, bw, sf, fs, snr, offset, cfo)

Evaluates preamble detection accuracy (TP, FP, FN, TN, AvgProb, DetRate) per SNR
matching the reference evaluation logic.
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
import torch
import torch.nn.functional as F

# ── project utils ─────────────────────────────────────────────────────────────
from utils.helper_function import read_base64_convert_to_np
from utils.nelora_utils import spec_to_network_input

# ── CALoRa model ───────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "calora" / "calora_model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from models.calora.calora_model.model_calora import PreambleDetector, best_interval_from_p, CALoRa  # noqa: E402

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CHECKPOINT_MAP = {
    7: Path(__file__).resolve().parent
        / "models" / "calora" / "checkpoints" / "sf7" / "ori_checkpoints"
        / "best_finetuned_train_cfo.pth",
}

_MODEL_CACHE: dict = {}

def _load_model(sf: int):
    """Load (or return cached) PreambleDetector for a given SF."""
    if sf in _MODEL_CACHE:
        return _MODEL_CACHE[sf]

    ckpt_path = _CHECKPOINT_MAP.get(sf)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"[CALoRa] No checkpoint found for SF{sf} at: {ckpt_path}")
        return None

    use_se = (sf >= 8)
    Ls = (262, 264, 266)

    model = PreambleDetector(C=64, Ls=Ls, use_se_tcn=use_se)
    print(ckpt_path)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # if isinstance(state, dict) and "model_state_dict" in state:
    #     state = state["model_state_dict"]
    # elif isinstance(state, dict) and "state_dict" in state:
    #     state = state["state_dict"]

    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
  
    load_res = model.load_state_dict(state, strict=False)
    print(f"[PreambleDetector] Load Result: {load_res}")
    
    model.to(DEVICE)
    model.eval()

    _MODEL_CACHE[sf] = model
    print(f"[CALoRa] PreambleDetector loaded for SF{sf} from {ckpt_path.name}")
    return model

_load_model(7)


_CALORA_DENOISE_MODEL = {}

def load_calora_denoiser(sf: int, weights_dir: str = None, calora_dir: str = None, device: torch.device = None):
    if device is None:
        device = DEVICE
    if weights_dir is None:
        # Defaulting to standard path in project structure
        weights_dir = str(Path(__file__).resolve().parent / "models" / "calora" / "checkpoints" / f"sf{sf}" / "ori_checkpoints")
    if calora_dir is None:
        calora_dir = str(Path(__file__).resolve().parent / "models" / "calora" / "calora_model")

    if sf in _CALORA_DENOISE_MODEL:
        return _CALORA_DENOISE_MODEL[sf]
    
    weight_path = os.path.join(weights_dir, f"chirp_restorer_sf{sf}.pth")
    if not os.path.isfile(weight_path):
        print(f"[CALoRa Denoiser] Warning: Weight file not found: {weight_path}")
        return None

    if calora_dir not in sys.path:
        sys.path.insert(0, calora_dir)
        
    model = CALoRa().to(device)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    load_res = model.load_state_dict(state, strict=False)
    print(f"[CALoRa Denoiser] Load Result: {load_res}")
    model.eval()
    print(f"  Loaded chirp restorer SF{sf} from {weight_path}")
    
    _CALORA_DENOISE_MODEL[sf] = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# IQ → network input tensor
# ─────────────────────────────────────────────────────────────────────────────
def iq_to_network_input(iq: np.ndarray, sf: int, bw: int, fs: int, num_symbols: int = 20) -> torch.Tensor:
    """
    Compute STFT symbol-by-symbol (for num_symbols=20).
    Each symbol (1024 samples for SF7 @ 1MHz/125kHz) yields 33 time columns.
    20 symbols x 33 columns = 660 total time columns.
    """
    n_classes = 2 ** sf
    osr       = fs // bw
    nsamp     = n_classes * osr      # samples per symbol (e.g. 1024 for SF7)
    nfft      = nsamp
    win_len   = n_classes // 2        # 64
    hop       = win_len // 2          # 32

    class _SpecOpts:
        pass
    spec_opts = _SpecOpts()
    spec_opts.freq_size       = n_classes
    spec_opts.normalization   = True
    spec_opts.x_image_channel = 2      # real + imag (B, 2, H, W)

    window = torch.hann_window(win_len)
    mag_chunks = []

    calora_denoiser = load_calora_denoiser(sf)

    # Loop through up to 20 symbols
    for s_idx in range(num_symbols):
        start = s_idx * nsamp
        end   = start + nsamp

        if start >= len(iq):
            # Zero-pad if signal is shorter than 20 symbols
            chunk = np.zeros(nsamp, dtype=np.complex64)
        elif end > len(iq):
            chunk = np.pad(iq[start:].astype(np.complex64), (0, end - len(iq)), mode='constant')
        else:
            chunk = iq[start:end].astype(np.complex64)

        x_chunk = torch.from_numpy(chunk)
        Z_chunk = torch.stft(
            x_chunk,
            n_fft=nfft,
            hop_length=hop,
            win_length=win_len,
            # window=window,
            return_complex=True,
            pad_mode="constant",
        )  # (nfft, 33)
        Z_batch = Z_chunk.unsqueeze(0)

        ri_chunk = spec_to_network_input(Z_batch, spec_opts)  # (1, 2, H, 33)
        
        if calora_denoiser is not None:
            with torch.no_grad():
                ri_chunk = calora_denoiser(ri_chunk.to(DEVICE)).cpu()
        
        #mag_chunk = torch.sqrt(ri_chunk[:, 0:1, :, :] ** 2 + ri_chunk[:, 1:2, :, :] ** 2)
        mag_chunk = torch.abs(ri_chunk[:, 0:1, :, :]) + torch.abs(ri_chunk[:, 1:2, :, :])
       
        mag_chunks.append(mag_chunk)

    # Concatenate all 20 symbol spectrograms along the time axis (dim 3) -> (1, 1, 128, 660)
    mag = torch.cat(mag_chunks, dim=3)
   
    mu = float(mag.mean())
    sigma = float(mag.std() + 1e-6)
    mag = (mag - mu) / sigma
   
    print("CALoRa Tensor Shape:", mag.shape)
    return mag.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib Debug Helper
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save debug images to disk
import matplotlib.pyplot as plt

def plot_debug_spectrogram(
    iq_signal: np.ndarray,
    spec_tensor: torch.Tensor,
    opts,
    t_start: int,
    t_end: int,
    true_start_col: int,
    mean_prob: float,
    snr: int,
    save_path: str = "debug_calora_spectrogram.png"
):
    """
    Plot and save debugging figures:
    1. Raw IQ Spectrogram via plt.specgram
    2. CALoRa Input Tensor (128 x 660) with t_start and true_start_col lines
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    plt.style.use('dark_background')
    print(iq_signal.shape)
    print(spec_tensor.shape)
    # 1. Raw IQ Signal Spectrogram
    Pxx, freqs, bins, im1 = ax1.specgram(
        iq_signal,
        NFFT=256,
        Fs=opts.fs,
        noverlap=128,
        cmap='jet'
    )
    ax1.set_title(f"Raw IQ Spectrogram (SNR={snr} dB, Length={len(iq_signal)} samples)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")

    # 2. CALoRa Input Tensor Matrix (128 x 660)
    mag_np = spec_tensor[0, 0].cpu().numpy()  # (128, 660)
    im2 = ax2.imshow(
        mag_np,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    # Draw vertical lines for Ground Truth vs Predicted Start
    ax2.axvline(x=true_start_col, color='cyan', linestyle='--', linewidth=2, label=f'True Start Col ({true_start_col})')
    ax2.axvline(x=t_start, color='red', linestyle='-', linewidth=2, label=f'Pred Start Col ({t_start})')
    ax2.axvline(x=t_end, color='orange', linestyle=':', linewidth=2, label=f'Pred End Col ({t_end})')

    ax2.set_title(f"CALoRa Input Tensor [128 x 660] (Mean Prob={mean_prob:.4f})")
    ax2.set_xlabel("Spectrogram Columns")
    ax2.set_ylabel("Frequency Bins")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  📸 Saved debug spectrogram image to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CALoRa preamble detection
# ─────────────────────────────────────────────────────────────────────────────
CALORA_THRESH = 0.5            # Probability threshold (mean_prob > thresh)
LOCATION_TOL_COLS = 37         # Tolerance in spectrogram columns (~1 LoRa symbol)


def detect_preamble_calora(opts, iq: np.ndarray):
    """
    Run PreambleDetector on the full IQ signal.

    Returns:
        is_detected (bool) : True if mean_prob > thresh
        t_start (int)     : Predicted preamble start column
        t_end (int)       : Predicted preamble end column
        mean_prob (float) : Mean probability over predicted preamble window
    """
    model = _load_model(opts.sf)
    if model is None:
        return False, 0, 0, 0.0

    try:
        spec = iq_to_network_input(iq, opts.sf, opts.bw, opts.fs)
    except Exception as exc:
        print(f"  [CALoRa] spectrogram failed: {exc}")
        return False, 0, 0, 0.0

    with torch.no_grad():
       
        _logits, p, _s = model(spec)

    p_np = p[0].cpu().numpy()

    Ls = tuple(int(l) for l in model.Ls.cpu().tolist())
    try:
        L_hat, t_start, t_end = best_interval_from_p(p_np, Ls)
    except Exception as exc:
        print(f"  [CALoRa] best_interval failed: {exc}")
        return False, 0, 0, 0.0
    
    mean_prob = float(np.mean(p_np[t_start : t_end + 1]))
    is_detected = (mean_prob > CALORA_THRESH)

    return is_detected, t_start, t_end, mean_prob


# ─────────────────────────────────────────────────────────────────────────────
# Global stats & watchdog
# ─────────────────────────────────────────────────────────────────────────────
ended = False
index = 0


def create_stats():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "probs": [],
    }


GLOBAL_STATS = defaultdict(create_stats)


class Watchdog:
    def __init__(self, timeout=5):
        self.timeout     = timeout
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
            print("\n⚠️  Timeout! No upload_calora received.")
            wd.update()
            print("----------------- SUMMARY (CALoRa Evaluation) -----------------")
            print(
                f"{'SNR':>5} | {'N':>5} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'AvgProb':>8} | {'DetRate':>8}"
            )
            print("-" * 62)
            tot_tp = tot_fp = tot_fn = tot_tn = 0

            # Collect rows for CSV output
            csv_rows = []

            for snr in sorted(GLOBAL_STATS):
                s        = GLOBAL_STATS[snr]
                tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
                n_total  = tp + fp + fn + tn
                avg_prob = np.mean(s["probs"]) if s["probs"] else 0.0
                det_rate = (tp / n_total * 100) if n_total > 0 else 0.0

                tot_tp += tp
                tot_fp += fp
                tot_fn += fn
                tot_tn += tn

                csv_rows.append((snr, det_rate))

                print(
                    f"{snr:>5} | {n_total:>5} | {tp:>5} | {fp:>5} | {fn:>5} | {avg_prob:>8.4f} | {det_rate:>7.2f}%"
                )

            tot_all = tot_tp + tot_fp + tot_fn + tot_tn
            if tot_all > 0:
                overall_det_rate = (tot_tp / tot_all) * 100
                print("-" * 62)
                print(f"Overall TP={tot_tp} FP={tot_fp} FN={tot_fn} TN={tot_tn} | DetRate: {overall_det_rate:.2f}%")

            # ── Write SNR & DetRate to CSV ────────────────────────────────
            import csv
            from datetime import datetime

            results_dir = Path(__file__).resolve().parent / "results"
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path  = results_dir / f"calora_results_{timestamp}.csv"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["SNR", "DetRate"])
                for snr_val, det_val in csv_rows:
                    writer.writerow([snr_val, round(det_val, 4)])

            print(f"\n📄 Results saved to: {csv_path}")
        time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Route: POST /upload_calora
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/upload_calora", methods=["POST"])
def upload_calora():
    global ended, index

    index += 1
    if index % 50 == 0:
        print("Packet no:", index)
    ended = False

    data = request.get_json()

    gateway_id      = data.get("gateway_id")
    b64_lora_signal = data.get("iq_data")
    bw              = data.get("bw",      125_000)
    sf              = data.get("sf",      7)
    fs              = data.get("fs",      1_000_000)
    snr             = data.get("snr")
    no_of_preamble  = data.get("preamble", 8)
    offset          = data.get("offset",  0)
    cfo = data.get("cfo",0)
    print("cfo :", cfo, " offset : ", offset, " SNR : ", snr)
    offset_in_spec = round(offset/1024 * 33)
    print(offset_in_spec)
    # ── Decode IQ ───────────────────────────────────────────────────────────
    np_lora_signal = read_base64_convert_to_np(b64_lora_signal)

    # ── Build opts ──────────────────────────────────────────────────────────
    opts                = type("", (), {})()
    opts.sf             = sf
    opts.bw             = bw
    opts.fs             = fs
    opts.n_classes      = 2 ** sf
    opts.gateway_id     = gateway_id
    opts.no_of_preamble = no_of_preamble

    # ── Calculate Ground Truth Preamble Start Column ────────────────────────
    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))
    random_zonk_samples = 2 * framePerSymbol
    true_start_samples  = max(0, random_zonk_samples)
    
    win_len = opts.n_classes // 2
    hop     = win_len // 2
    true_start_col = ((true_start_samples // hop) + 1) - offset_in_spec # true_start_samples // hop
    
    # ── CALoRa preamble detection ────────────────────────────────────────────
    spec_tensor = iq_to_network_input(np_lora_signal, opts.sf, opts.bw, opts.fs)
    is_detected, t_start, t_end, mean_prob = detect_preamble_calora(opts, np_lora_signal)
    
    # Save a debugging plot for packet #1
    if index == 10 or index == 3:
       
        plot_debug_spectrogram(
            iq_signal=np_lora_signal,
            spec_tensor=spec_tensor,
            opts=opts,
            t_start=t_start,
            t_end=t_end,
            true_start_col=true_start_col,
            mean_prob=mean_prob,
            snr=snr,
            save_path= "debug_calora_spectrogram" + str(index) + " " + str(mean_prob) + ".png"
        )
    
    bool_pred  = mean_prob > CALORA_THRESH
    # bool_exist = abs(t_start - true_start_col) <= LOCATION_TOL_COLS
    bool_exist = abs(t_start - true_start_col) <= LOCATION_TOL_COLS

    print("t_start:", t_start, "| true_start_col:", true_start_col, "| mean_prob:", round(mean_prob, 4), "| pred:", bool_pred, "| exist:", bool_exist)

    GLOBAL_STATS[snr]["probs"].append(mean_prob)

    if bool_pred and bool_exist:
        GLOBAL_STATS[snr]["tp"] += 1   # prob > thresh AND correct location
    elif bool_pred and not bool_exist:
        GLOBAL_STATS[snr]["fn"] += 1   # prob > thresh BUT wrong location
    elif not bool_pred and bool_exist:
        GLOBAL_STATS[snr]["fp"] += 1   # prob <= thresh AND correct location (missed)
    else:
        GLOBAL_STATS[snr]["tn"] += 1   # prob <= thresh AND wrong location

    wd.update()
    return jsonify({
        "status":     "success" if is_detected else "fail",
        "mean_prob":  round(mean_prob, 4),
        "t_start":    t_start,
        "bool_pred":  int(bool_pred),
        "bool_exist": int(bool_exist),
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=watchdog_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5005, debug=False, threaded=False)
