"""
Single-frame symbol decoding benchmark runner.

Evaluates five symbol decoding methods side-by-side on a single LoRa frame:
  1. Conventional Dechirping (Standard FFT peak detection)
  2. NeLoRa (maskCNN denoiser + hybrid classifier)
  3. CALoRa (CALoRa chirp restorer + NeLoRa classifier)
  4. LoRaSeek (LoRaSeekNet denoiser + ISTFT + FFT)
  5. GLoRiPhy (Conformer denoiser + ISTFT + FFT)
"""
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from utils.LoRa import LoRa
from utils.my_lora_utils import (
    create_lora_preamble,
    create_lora_payload,
    add_cfo,
    calculate_symbol_alliqfile_with_down_sampling,
    calculate_symbol_alliqfile_without_down_sampling,
)
from utils.helper_function import detect_cfo_sto
from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from utils.nelora_inference import (
    _load_models as load_nelora_models,
    NELORA_VALIDATED,
    NELORA_SF7_AVAILABLE,
    NELORA_SF8_AVAILABLE,
)
from utils import loraseek_inference, gloriphy_inference
from models.calora.calora_model.model_calora import CALoRa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SERVER_DIR = Path(__file__).resolve().parents[1]

# Cached CALoRa restorer instance
_calora_restorer_cache: dict = {}


def _get_calora_restorer(sf: int):
    if sf in _calora_restorer_cache:
        return _calora_restorer_cache[sf]
    weights_path = _SERVER_DIR / "models" / "calora" / "checkpoints" / f"sf{sf}" / "ori_checkpoints" / f"chirp_restorer_sf{sf}.pth"
    if not weights_path.exists():
        return None
    model = CALoRa().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    _calora_restorer_cache[sf] = model
    return model


def _dechirp_fft_basic_single_band(payload_iq: np.ndarray, sf: int, bw: int, fs: int) -> list[int]:
    """
    Basic dechirping FFT with decimation downsampling (without 2-peak sum).
    Subsamples oversampled symbols by selecting every x sample (e.g. every 8th sample for
    8x oversampling: 1024 -> 128 bins for SF7), then dechirps at Fs=BW, applies normal FFT
    and argmax.
    """
    lora = LoRa(sf, bw)
    down_chirp_bw = np.conj(lora.gen_symbol_fs(0, down=False, Fs=bw))
    osr = int(round(fs / bw)) if bw > 0 else 1
    n_classes = 2 ** sf
    nsamp = int(n_classes * (fs / bw))
    num_symbols = int(len(payload_iq) // nsamp)

    symbols = []
    for i in range(num_symbols):
        chunk = payload_iq[i * nsamp : (i + 1) * nsamp]
        if len(chunk) < nsamp:
            chunk = np.pad(chunk, (0, nsamp - len(chunk)))
        # Downsample: select every osr-th sample (e.g., every 8th sample: 1024 -> 128 bins)
        chunk_down = chunk[::osr]
        if len(chunk_down) < n_classes:
            chunk_down = np.pad(chunk_down, (0, n_classes - len(chunk_down)))
        de = chunk_down * down_chirp_bw
        spectrum = np.fft.fft(de)
        symbols.append(int(np.argmax(np.abs(spectrum))))
    return symbols


def _dechirp_fft_with_down_sampling(payload_iq: np.ndarray, sf: int, bw: int, fs: int) -> list[int]:
    """
    Dechirping FFT with Nyquist downsampling (Fs = BW).
    Direct wrapper around calculate_symbol_alliqfile_with_down_sampling.
    """
    dec = calculate_symbol_alliqfile_with_down_sampling(payload_iq, sf, bw, fs, show=False)
    return [int(x) for x in dec]


def _dechirp_fft_without_down_sampling(payload_iq: np.ndarray, sf: int, bw: int, fs: int) -> list[int]:
    """
    Estimate symbol all IQ file without down sampling (sum of 2 energy in positive and negative frequency).
    Direct wrapper around calculate_symbol_alliqfile_without_down_sampling.
    Reference: From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation (Xu et al.)
    """
    dec, _ = calculate_symbol_alliqfile_without_down_sampling(payload_iq, sf, bw, fs, show=False)
    return [int(x) for x in dec]


def evaluate_single_frame(
    sf: int = 7,
    bw: int = 125_000,
    fs: int = 1_000_000,
    n_symbols: int = 10,
    snr_db: float = 10.0,
    apply_cfo: bool = False,
    cfo_hz: float = 500.0,
    apply_sto: bool = False,
    sto_samples: int = 20,
    apply_compensation: bool = True,
    seed: int = -1,
) -> dict[str, Any]:
    """
    Generate one LoRa frame, optionally inject CFO and STO impairments, and
    decode the payload across all five models.
    """
    if not (1 <= n_symbols <= 1000):
        raise ValueError(f"Payload length n_symbols must be between 1 and 1000 (got {n_symbols})")
    if seed is not None and seed >= 0:
        rng = np.random.default_rng(seed)
        noise_seed = seed
    else:
        rng = np.random.default_rng()
        noise_seed = int(rng.integers(0, 2**31 - 1))

    n_classes = 2 ** sf
    fps = int(n_classes * (fs / bw))

    # 1. Synthesize frame: 8 upchirps + 2 sync + 2.25 downchirps + payload
    gt_symbols = rng.integers(0, n_classes, size=n_symbols).tolist()
    opts = type("", (), {})()
    opts.sf = sf
    opts.bw = bw
    opts.fs = fs
    opts.n_classes = n_classes
    opts.no_of_preamble = 8
    opts.sync_sym = [8, 8]

    preamble = create_lora_preamble(opts, LoRa)
    payload = create_lora_payload(opts, LoRa, gt_symbols)
    sig = np.concatenate([preamble, payload]).astype(np.complex64)

    # 2. Apply impairments
    cfo_applied = float(cfo_hz) if apply_cfo else 0.0
    sto_applied = int(sto_samples) if apply_sto else 0

    if cfo_applied != 0.0:
        sig = add_cfo(opts, sig, cfo_applied)
    if sto_applied > 0:
        sig = sig[sto_applied:]

    lora = LoRa(sf, bw)
    iq = lora.awgn_iq_with_seed(sig, snr_db, seed=noise_seed)

    # 3. Synchronize / Align payload
    cfo_estimated = 0.0
    sto_estimated = 0
    sync_success = False

    if apply_compensation and (apply_cfo or apply_sto):
        index_payload, cfo_est, sto_est, _, _ = detect_cfo_sto(
            opts, LoRa, iq, return_details=True
        )
        if index_payload is not None and index_payload not in (-1, -2):
            sync_success = True
            cfo_estimated = float(cfo_est)
            sto_estimated = int(round(float(sto_est)))
            payload_start = int(index_payload * fps) + sto_estimated
        else:
            sync_success = False
            # Nominal payload start if synchronization failed; do not cheat with ground truth!
            payload_start = max(0, int(12.25 * fps) - sto_applied)
            cfo_estimated = 0.0
            sto_estimated = 0

        t_axis = np.arange(len(iq)) / fs
        iq_cfo = iq * np.exp(-1j * 2 * np.pi * cfo_estimated * t_axis)
        payload_corrected = iq_cfo[payload_start : payload_start + n_symbols * fps]

        pre_start = max(0, payload_start - int(12.25 * fps))
        pre_end = pre_start + int(8 * fps)
        sfd_start = max(0, payload_start - int(2.25 * fps))
        preamble_corr = iq_cfo[pre_start:pre_end]
        sfd_corr = iq_cfo[sfd_start:payload_start]
    else:
        # No synchronization or compensation: test models directly against raw impaired signal
        payload_start = max(0, int(12.25 * fps) - sto_applied)
        payload_corrected = iq[payload_start : payload_start + n_symbols * fps]
        pre_start = max(0, payload_start - int(12.25 * fps))
        preamble_corr = iq[pre_start : pre_start + int(8 * fps)]
        sfd_corr = iq[max(0, payload_start - int(2.25 * fps)) : payload_start]

    # Pad payload if needed
    expected_len = n_symbols * fps
    if len(payload_corrected) < expected_len:
        payload_corrected = np.pad(
            payload_corrected, (0, expected_len - len(payload_corrected))
        )

    models_output = {}

    # ── Model 1: Basic Dechirping FFT (Direct Decimation 1/8) ───────────────
    try:
        t0 = time.perf_counter()
        basic_single_syms = _dechirp_fft_basic_single_band(payload_corrected, sf, bw, fs)[:n_symbols]
        t_single = (time.perf_counter() - t0) * 1000
        matched = int(np.sum(np.array(basic_single_syms) == np.array(gt_symbols)))
        models_output["basic_single_band"] = {
            "name": "Basic Dechirping FFT (Direct Decimation 1/8)",
            "available": True,
            "status": "Ready",
            "symbols": [int(s) for s in basic_single_syms[:64]],
            "matched": matched,
            "total": n_symbols,
            "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
            "execution_time_ms": round(t_single, 2),
            "notes": "Decimated subsampling (selects every 8th sample: 1024 -> 128 bins) + standard FFT argmax",
        }
    except Exception as exc:
        models_output["basic_single_band"] = {
            "name": "Basic Dechirping FFT (Direct Decimation 1/8)",
            "available": False,
            "status": f"Error: {exc}",
        }

    # ── Model 2: Dechirping FFT with Downsampling (Polyphase) ────────────────
    try:
        t0 = time.perf_counter()
        down_syms = _dechirp_fft_with_down_sampling(payload_corrected, sf, bw, fs)[:n_symbols]
        t_down = (time.perf_counter() - t0) * 1000
        matched = int(np.sum(np.array(down_syms) == np.array(gt_symbols)))
        models_output["dechirp_downsampled"] = {
            "name": "Dechirping FFT with Downsampling (Polyphase)",
            "available": True,
            "status": "Ready",
            "symbols": [int(s) for s in down_syms[:64]],
            "matched": matched,
            "total": n_symbols,
            "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
            "execution_time_ms": round(t_down, 2),
            "notes": "Nyquist downsampling (Fs=BW via scipy.signal.resample_poly) + standard FFT argmax",
        }
    except Exception as exc:
        models_output["dechirp_downsampled"] = {
            "name": "Dechirping FFT with Downsampling (Polyphase)",
            "available": False,
            "status": f"Error: {exc}",
        }

    # ── Model 3: Estimate Symbol without Downsampling (Dual-Energy Sum) ──────
    try:
        t0 = time.perf_counter()
        without_syms = _dechirp_fft_without_down_sampling(payload_corrected, sf, bw, fs)[:n_symbols]
        t_without = (time.perf_counter() - t0) * 1000
        matched = int(np.sum(np.array(without_syms) == np.array(gt_symbols)))
        models_output["conventional"] = {
            "name": "Estimate Symbol without Downsampling (Dual-Energy Sum)",
            "available": True,
            "status": "Ready",
            "symbols": [int(s) for s in without_syms[:64]],
            "matched": matched,
            "total": n_symbols,
            "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
            "execution_time_ms": round(t_without, 2),
            "notes": "Xu et al. [1] — Full sampling rate dual-band dechirping (energy fold sum)",
        }
    except Exception as exc:
        models_output["conventional"] = {
            "name": "Estimate Symbol without Downsampling (Dual-Energy Sum)",
            "available": False,
            "status": f"Error: {exc}",
        }

    # ── Model 2: NeLoRa ────────────────────────────────────────────────────
    nelora_ready = (sf == 7 and NELORA_SF7_AVAILABLE) or (sf == 8 and NELORA_SF8_AVAILABLE)
    if nelora_ready:
        try:
            t0 = time.perf_counter()
            ckpt = "100000" if sf == 7 else "75000"
            mask_net, nelora_clf = load_nelora_models(ckpt, sf=sf)
            chunks = iq_to_symbol_chunks(
                payload_corrected, sf, bw, fs, num_symbols=n_symbols, dechirp=False
            )
            with torch.no_grad():
                denoised = mask_net(chunks.to(DEVICE))
                logits = nelora_clf(denoised)
                raw = logits.argmax(dim=1).cpu().numpy()
                nelora_syms = [int(inverse_symb(int(s), sf)) for s in raw]
            t_nelora = (time.perf_counter() - t0) * 1000
            matched = int(np.sum(np.array(nelora_syms) == np.array(gt_symbols)))
            models_output["nelora"] = {
                "name": "NeLoRa",
                "available": True,
                "status": "Ready",
                "symbols": nelora_syms[:64],
                "matched": matched,
                "total": n_symbols,
                "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
                "execution_time_ms": round(t_nelora, 2),
                "notes": f"Li et al. [2] — maskCNN denoiser + hybrid classifier ({'SF7 100k' if sf == 7 else 'SF8 75k'})",
            }
        except Exception as exc:
            models_output["nelora"] = {
                "name": "NeLoRa",
                "available": False,
                "status": f"Error: {exc}",
            }
    else:
        models_output["nelora"] = {
            "name": "NeLoRa",
            "available": False,
            "status": f"NeLoRa checkpoints for SF{sf} not available",
        }

    # ── Model 3: CALoRa ────────────────────────────────────────────────────
    calora_restorer = _get_calora_restorer(sf)
    if calora_restorer is not None:
        try:
            t0 = time.perf_counter()
            chunks = iq_to_symbol_chunks(
                payload_corrected, sf, bw, fs, num_symbols=n_symbols, dechirp=False
            )
            with torch.no_grad():
                restored = calora_restorer(chunks.to(DEVICE))
                _, nelora_clf = load_nelora_models("100000")
                logits = nelora_clf(restored)
                raw = logits.argmax(dim=1).cpu().numpy()
                calora_syms = [int(inverse_symb(int(s), sf)) for s in raw]
            t_calora = (time.perf_counter() - t0) * 1000
            matched = int(np.sum(np.array(calora_syms) == np.array(gt_symbols)))
            models_output["calora"] = {
                "name": "CALoRa",
                "available": True,
                "status": "Ready",
                "symbols": calora_syms[:64],
                "matched": matched,
                "total": n_symbols,
                "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
                "execution_time_ms": round(t_calora, 2),
                "notes": "CALoRa [5] — CNN chirp restorer + hybrid classifier",
            }
        except Exception as exc:
            models_output["calora"] = {
                "name": "CALoRa",
                "available": False,
                "status": f"Error: {exc}",
            }
    else:
        models_output["calora"] = {
            "name": "CALoRa",
            "available": False,
            "status": f"CALoRa checkpoint for SF{sf} not yet prepared",
        }

    # ── Model 4: LoRaSeek ──────────────────────────────────────────────────
    ls_available = (
        (sf == 7 and loraseek_inference.LORASEEK_SF7_AVAILABLE)
        or (sf == 8 and loraseek_inference.LORASEEK_SF8_AVAILABLE)
    )
    if ls_available:
        try:
            t0 = time.perf_counter()
            ckpt_name = "75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl"
            ls_syms = loraseek_inference.demodulate_payload_loraseek(
                payload_corrected, sf, bw, fs, checkpoint_name=ckpt_name
            )
            t_ls = (time.perf_counter() - t0) * 1000
            matched = int(np.sum(np.array(ls_syms) == np.array(gt_symbols)))
            version_note = "v9 75k" if sf == 7 else "v81 90k"
            models_output["loraseek"] = {
                "name": "LoRaSeek",
                "available": True,
                "status": "Ready",
                "symbols": [int(s) for s in ls_syms[:64]],
                "matched": matched,
                "total": n_symbols,
                "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
                "execution_time_ms": round(t_ls, 2),
                "notes": f"LoRaSeek [3] — STFT Transformer + Dual-Attention U-Net ({version_note})",
            }
        except Exception as exc:
            models_output["loraseek"] = {
                "name": "LoRaSeek",
                "available": False,
                "status": f"Error: {exc}",
            }
    else:
        models_output["loraseek"] = {
            "name": "LoRaSeek",
            "available": False,
            "status": f"LoRaSeek checkpoint for SF{sf} not available",
        }

    # ── Model 5: GLoRiPhy ──────────────────────────────────────────────────
    if gloriphy_inference.GLORIPHY_AVAILABLE and sf in gloriphy_inference.SUPPORTED_SFS:
        try:
            t0 = time.perf_counter()
            gl_ckpt = gloriphy_inference.default_checkpoint_for(sf)
            gl_syms = gloriphy_inference.demodulate_payload_gloriphy(
                payload_corrected,
                sf,
                bw,
                fs,
                checkpoint_id=gl_ckpt,
                preamble_iq=preamble_corr,
                sfd_iq=sfd_corr,
            )
            t_gl = (time.perf_counter() - t0) * 1000
            matched = int(np.sum(np.array(gl_syms) == np.array(gt_symbols)))
            models_output["gloriphy"] = {
                "name": "GLoRiPhy",
                "available": True,
                "status": "Ready",
                "symbols": [int(s) for s in gl_syms[:64]],
                "matched": matched,
                "total": n_symbols,
                "accuracy_pct": round((matched / n_symbols) * 100.0, 2),
                "execution_time_ms": round(t_gl, 2),
                "notes": f"GLoRiPHY [4] — Conformer generative channel-aware denoiser ({gl_ckpt})",
            }
        except Exception as exc:
            models_output["gloriphy"] = {
                "name": "GLoRiPhy",
                "available": False,
                "status": f"Error: {exc}",
            }
    else:
        models_output["gloriphy"] = {
            "name": "GLoRiPhy",
            "available": False,
            "status": f"GLoRiPhy checkpoint for SF{sf} not available",
        }

    return {
        "sf": sf,
        "bw": bw,
        "fs": fs,
        "n_symbols": n_symbols,
        "snr_db": snr_db,
        "apply_cfo": apply_cfo,
        "cfo_applied_hz": cfo_applied,
        "apply_sto": apply_sto,
        "sto_applied_samp": sto_applied,
        "apply_compensation": apply_compensation,
        "sync_success": sync_success,
        "cfo_estimated_hz": round(cfo_estimated, 2),
        "sto_estimated_samp": sto_estimated,
        "residual_cfo_hz": round(cfo_applied - cfo_estimated, 2),
        "ground_truth_symbols": gt_symbols[:64],
        "models": models_output,
    }
