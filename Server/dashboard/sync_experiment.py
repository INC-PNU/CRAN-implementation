"""
Denoised Up-Chirp / Down-Chirp Synchronization Benchmark Experiment.

Evaluates the hypothesis:
  "Denoising noisy LoRa preamble up-chirps and SFD down-chirps using neural
   denoisers (GLoRiPHY or LoRaSeek) BEFORE conventional up/down dechirping
   significantly improves CFO and STO estimation in the ultra-low SNR regime."

Key signal-processing insight:
  Both GLoRiPHY (ConformerGen_Core) and LoRaSeek (LoRaSeekNet) were trained
  exclusively on LoRa UP-chirps. To denoise an SFD DOWN-chirp without retraining,
  we exploit complex-conjugate chirp symmetry:
      r_down*(t) = exp(+j * pi * (BW/T) * (t - dtau)^2 - j * 2*pi * df * t) + n*(t)
  which transforms a down-chirp into an exact up-chirp with identical timing offset
  and negated frequency offset. Passing conj(down_frame) through the neural denoiser
  and taking conj(...) of the reconstructed waveform yields a denoised down-chirp
  preserving both CFO (+df) and STO (dtau).

Methods compared (all using 1 preamble up-chirp + 1 SFD down-chirp):
  1. conv_single      : Pure Conventional Sync (1 up-chirp + 1 down-chirp; helper_function.py v3)
  2. gloriphy_single  : GLoRiPHY Denoise (1 up + 1 conj down) -> Conventional Dechirp
  3. loraseek_single  : LoRaSeek Denoise (1 up + 1 conj down) -> Conventional Dechirp
"""
import json
import sys
import threading
import time
from datetime import datetime, timezone
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
)
from utils.helper_function import (
    wrap_symmetric,
    to_nearest_N_center,
    round_left_left_zero,
    round_right_right_zero,
    predict_te,
)
from utils import gloriphy_inference, loraseek_inference
from models.gloriphy.gloriphy_model.stft_utils import spec_to_network_input

RESULTS_FILE = Path(__file__).resolve().parent / "sync_experiment_results.json"
_file_lock = threading.Lock()

METHOD_META = {
    "conv_single": {
        "label": "Conventional",
        "short": "Conventional",
        "color": "#334155",
        "dashed": False,
        "point_style": "circle",
        "group": "conventional",
        "description": "Baseline helper_function.py v3: single up-chirp + single down-chirp dual-band dechirping.",
    },
    "gloriphy_single": {
        "label": "GLoRiPHY Denoised",
        "short": "GLoRiPHY Denoised",
        "color": "#e11d48",
        "dashed": False,
        "point_style": "rect",
        "group": "gloriphy",
        "description": "Denoise up-chirp & conjugated down-chirp with GLoRiPHY, then conventional dechirp.",
    },
    "loraseek_single": {
        "label": "LoRaSeek Denoised",
        "short": "LoRaSeek Denoised",
        "color": "#2563eb",
        "dashed": False,
        "point_style": "triangle",
        "group": "loraseek",
        "description": "Denoise up-chirp & conjugated down-chirp with LoRaSeek, then conventional dechirp.",
    },
}

METHOD_ORDER = [
    "conv_single",
    "gloriphy_single",
    "loraseek_single",
]


def _denoise_waveforms_gloriphy(frames_np: np.ndarray, sf: int = 7, batch_size: int = 64) -> np.ndarray:
    """Denoise batch of up-chirp IQ frames [N, nsamp] via GLoRiPHY + full-rate Hamming ISTFT."""
    info = gloriphy_inference.checkpoint_info(None, sf)
    loaded = gloriphy_inference._load(info)
    opts = loaded.opts
    device = gloriphy_inference.DEVICE

    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(frames_np), batch_size):
            x_t = torch.from_numpy(frames_np[i : i + batch_size].astype(np.complex64)).to(device)
            stft_in = gloriphy_inference._stft(x_t, opts)
            net_in = spec_to_network_input(stft_in, opts)
            processed_spec, _, _ = loaded.model(net_in)

            proc_c = torch.view_as_complex(processed_spec.permute(0, 2, 3, 1).contiguous())
            B, _, T = proc_c.shape
            trim_size = opts.freq_size // 2
            full_spec = torch.zeros(
                (B, opts.demod.d_num_samples, T), dtype=torch.complex64, device=device
            )
            full_spec[:, -trim_size:, :] = proc_c[:, :trim_size, :]
            full_spec[:, :trim_size, :] = proc_c[:, trim_size:, :]

            recon = torch.istft(
                full_spec,
                n_fft=opts.stft_nfft,
                hop_length=opts.hop_length,
                win_length=opts.stft_window,
                window=opts.hamming_window,
                return_complex=True,
            )
            out_chunks.append(recon.cpu().numpy())

    return np.concatenate(out_chunks, axis=0)


def _denoise_waveforms_loraseek(frames_np: np.ndarray, sf: int = 7, batch_size: int = 64) -> np.ndarray:
    """Denoise batch of up-chirp IQ frames [N, nsamp] via LoRaSeek + full-rate Hann ISTFT."""
    ckpt = "75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl"
    model = loraseek_inference._load_loraseek_model(ckpt, sf=sf)
    opts = loraseek_inference._LoRaSeekOptsSF8() if sf == 8 else loraseek_inference._LoRaSeekOptsSF7()
    device = loraseek_inference.DEVICE
    window = torch.hann_window(opts.stft_window, device=device)
    trim_size = opts.freq_size // 2
    nsamp = opts.stft_nfft

    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(frames_np), batch_size):
            x_t = torch.from_numpy(frames_np[i : i + batch_size].astype(np.complex64)).to(device)
            raw_spec = torch.stft(
                input=x_t,
                n_fft=opts.stft_nfft,
                hop_length=opts.stft_overlap,
                win_length=opts.stft_window,
                window=window,
                pad_mode="constant",
                return_complex=True,
            )
            y = torch.cat((raw_spec[:, -trim_size:, :], raw_spec[:, 0:trim_size, :]), dim=1)
            y_abs_max = torch.clamp(torch.amax(torch.abs(y), dim=(1, 2), keepdim=True), min=1e-8)
            y_norm = y / y_abs_max
            net_in = torch.view_as_real(y_norm).permute(0, 3, 1, 2).to(dtype=torch.float32)[:, :, :, :32]

            denoised_net = model(net_in)
            spec_complex = torch.complex(denoised_net[:, 0], denoised_net[:, 1])

            B, _, T = spec_complex.shape
            full_spec = torch.zeros((B, opts.stft_nfft, T), dtype=torch.complex64, device=device)
            full_spec[:, -trim_size:, :] = spec_complex[:, :trim_size, :]
            full_spec[:, :trim_size, :] = spec_complex[:, trim_size:, :]

            recon = torch.istft(
                full_spec,
                n_fft=opts.stft_nfft,
                hop_length=opts.stft_overlap,
                win_length=opts.stft_window,
                window=window,
                onesided=False,
                return_complex=True,
                center=True,
            ).cpu().numpy()

            if recon.shape[-1] < nsamp:
                recon = np.pad(recon, ((0, 0), (0, nsamp - recon.shape[-1])))
            else:
                recon = recon[:, :nsamp]
            out_chunks.append(recon)

    return np.concatenate(out_chunks, axis=0)


def _dechirp_dual_band_spectrum(frame: np.ndarray, ref_chirp: np.ndarray, n_classes: int) -> np.ndarray:
    """Dual-band folded dechirped FFT magnitude spectrum (matches helper_function.py lines 406-412)."""
    dechirped = frame * ref_chirp
    psd = np.abs(np.fft.fftshift(np.fft.fft(dechirped)))
    center = len(psd) // 2
    return psd[center : center + n_classes] + psd[center - n_classes : center]


def _estimate_cfo_frac_raw(frames_up: np.ndarray, down_ref: np.ndarray, bw: int, n_classes: int) -> float:
    """Consecutive up-chirp phase difference CFO_frac estimator (helper_function.py lines 115-134)."""
    z_sum = 0j
    for k in range(len(frames_up) - 1):
        d1 = frames_up[k] * down_ref
        d2 = frames_up[k + 1] * down_ref
        z_sum += np.vdot(d1, d2)
    phase_diff = np.angle(z_sum)
    return float((phase_diff * bw) / (2.0 * np.pi * n_classes))


def _solve_cfo_sto_from_spectra(
    combine_up: np.ndarray,
    combine_down: np.ndarray,
    cfo_frac: float,
    bw: int,
    fs: int,
    n_classes: int,
    use_frac_sto: bool = True,
) -> tuple[float, float]:
    """Closed-form up/down chirp CFO & STO solver (exact helper_function.py v3 math)."""
    over_sample = fs / bw
    symbol_up = int(np.argmax(combine_up))
    symbol_down = int(np.argmax(combine_down))

    shift_sto_index = 0.0
    if use_frac_sto:
        next_idx = (symbol_up + 1) % n_classes
        prev_idx = (symbol_up - 1 + n_classes) % n_classes
        prev_val = float(combine_up[prev_idx])
        next_val = float(combine_up[next_idx])
        peak_val = max(float(combine_up[symbol_up]), 1e-12)
        if next_val > prev_val:
            sto_off = 4.0 * np.tanh(2.5 * (next_val - prev_val) / peak_val)
            shift_sto_index = float(np.round(predict_te(sto_off * -1.0) * over_sample) * -1.0)
        else:
            sto_off = 4.0 * np.tanh(2.5 * (prev_val - next_val) / peak_val)
            shift_sto_index = float(np.round(predict_te(sto_off * -1.0) * over_sample))

    next_idx_d = (symbol_down + 1) % n_classes
    prev_idx_d = (symbol_down - 1 + n_classes) % n_classes
    type_type = 1 if combine_down[next_idx_d] > combine_down[prev_idx_d] else 2

    cfo_bins = to_nearest_N_center((symbol_up + symbol_down) / 2.0, n_classes)
    cfo_bins = round_left_left_zero(cfo_bins) if type_type == 1 else round_right_right_zero(cfo_bins)

    cfo_int_hz = (cfo_bins * bw) / n_classes
    cfo_total = float(cfo_int_hz + cfo_frac)

    sto_int_chips = wrap_symmetric(cfo_bins + n_classes - symbol_up, n_classes)
    sto_total = float(sto_int_chips * over_sample + shift_sto_index)
    return cfo_total, sto_total


def _eval_payload_after_sync(
    iq: np.ndarray,
    gt_payload: list[int],
    cfo_est: float,
    sto_est: float,
    sf: int,
    bw: int,
    fs: int,
    fps: int,
) -> float:
    """Apply estimated CFO & STO compensation to the packet and demodulate payload via polyphase FFT."""
    n_pay = len(gt_payload)
    payload_start = int(round(12.25 * fps)) + int(round(sto_est))
    if payload_start < 0 or payload_start >= len(iq):
        return 0.0
    t_all = np.arange(len(iq)) / fs
    iq_corr = iq * np.exp(-1j * 2.0 * np.pi * cfo_est * t_all)
    pay_iq = iq_corr[payload_start : payload_start + n_pay * fps]
    if len(pay_iq) < n_pay * fps:
        pay_iq = np.pad(pay_iq, (0, n_pay * fps - len(pay_iq)))
    dec = calculate_symbol_alliqfile_with_down_sampling(pay_iq, sf, bw, fs, show=False)
    n_cmp = min(len(dec), n_pay)
    if n_cmp == 0:
        return 0.0
    return float(np.sum(np.array(dec[:n_cmp]) == np.array(gt_payload[:n_cmp])) / n_pay) * 100.0


def run_sync_experiment(
    sf: int = 7,
    snr_min: int = -24,
    snr_max: int = -10,
    snr_step: int = 2,
    n_trials: int = 60,
    cfo_lo: float = -4358.0,
    cfo_hi: float = 4584.0,
    sto_lo: int = 0,
    sto_hi: int = 50,
    payload_len: int = 10,
    seed: int = 2026,
) -> dict[str, Any]:
    """
    Run a Monte-Carlo comparison of Pure Conventional vs. GLoRiPHY-Denoised vs.
    LoRaSeek-Denoised Up/Down-Chirp CFO & STO synchronization across SNR bins.
    """
    if sf not in (7, 8):
        raise ValueError(f"Sync experiment supports SF7 and SF8 (got sf={sf})")
    if snr_min > snr_max or snr_step <= 0:
        raise ValueError("Invalid SNR range/step")
    if not (5 <= n_trials <= 500):
        raise ValueError("n_trials must be between 5 and 500")

    bw = 125_000
    fs = 1_000_000
    n_classes = 2 ** sf
    fps = int(n_classes * (fs / bw))
    cfo_tol_hz = bw / (2.0 * n_classes)
    sto_tol_samp = fs / bw

    opts = type("", (), {})()
    opts.sf = sf
    opts.bw = bw
    opts.fs = fs
    opts.n_classes = n_classes
    opts.no_of_preamble = 8
    opts.sync_sym = [8, 8]

    lora = LoRa(sf, bw)
    up_ref = lora.gen_symbol_fs(0, down=False, Fs=fs)
    down_ref = np.conj(up_ref)
    t_sym = np.arange(fps) / fs

    # Check model availability
    gl_ok = gloriphy_inference.GLORIPHY_AVAILABLE and (sf in gloriphy_inference.SUPPORTED_SFS)
    ls_ok = (sf == 7 and loraseek_inference.LORASEEK_SF7_AVAILABLE) or (
        sf == 8 and loraseek_inference.LORASEEK_SF8_AVAILABLE
    )

    active_methods = ["conv_single"]
    if gl_ok:
        gloriphy_inference._load(gloriphy_inference.checkpoint_info(None, sf))
        active_methods.append("gloriphy_single")
    if ls_ok:
        ckpt = "75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl"
        loraseek_inference._load_loraseek_model(ckpt, sf=sf)
        active_methods.append("loraseek_single")

    snr_list = list(range(int(snr_min), int(snr_max) + 1, int(snr_step)))
    if snr_list[-1] != int(snr_max):
        snr_list.append(int(snr_max))

    rng = np.random.default_rng(seed if (seed is not None and seed >= 0) else None)
    base_noise_seed = int(seed) if (seed is not None and seed >= 0) else int(rng.integers(0, 2**20))

    series = {
        m: {
            "cfo_acc": [],
            "sto_acc": [],
            "cfo_mae": [],
            "sto_mae": [],
            "payload_acc": [],
            "avg_ms": [],
        }
        for m in active_methods
    }

    t_exp_start = time.perf_counter()

    for snr in snr_list:
        trials = []
        pack2_list = []

        for tr in range(n_trials):
            cfo_true = float(rng.uniform(cfo_lo, cfo_hi))
            numb_offset = int(rng.integers(sto_lo, sto_hi + 1))
            sto_true = float(wrap_symmetric(-numb_offset, fps))
            gt_payload = rng.integers(0, n_classes, size=payload_len).tolist()

            preamble = create_lora_preamble(opts, LoRa)
            payload = create_lora_payload(opts, LoRa, gt_payload)
            sig = np.concatenate([preamble, payload]).astype(np.complex64)
            sig = add_cfo(opts, sig, cfo_true)
            sig = sig[numb_offset:]

            noise_seed = abs(base_noise_seed * 131 + snr * 997 + tr + 17) % (2**31 - 1)
            iq = lora.awgn_iq_with_seed(sig, float(snr), seed=noise_seed)

            raw_ups_for_frac = np.stack([iq[k * fps : (k + 1) * fps] for k in range(0, 6)], axis=0)
            t_f0 = time.perf_counter()
            cfo_frac_raw = _estimate_cfo_frac_raw(raw_ups_for_frac, down_ref, bw, n_classes)
            frac_ms = (time.perf_counter() - t_f0) * 1000.0

            corr = np.exp(-1j * 2.0 * np.pi * cfo_frac_raw * t_sym)
            # Single preamble up-chirp (frame 5) and single SFD down-chirp (frame 10)
            up_corr = iq[5 * fps : 6 * fps] * corr
            down_corr = iq[10 * fps : 11 * fps] * corr

            trials.append({
                "iq": iq,
                "gt_payload": gt_payload,
                "cfo_true": cfo_true,
                "sto_true": sto_true,
                "cfo_frac_raw": cfo_frac_raw,
                "frac_ms": frac_ms,
                "up_corr": up_corr,
                "down_corr": down_corr,
            })
            # Pack 2 frames: 1 up-chirp (frame 5) + 1 conjugated down-chirp (frame 10)
            pack2_list.append(np.stack([up_corr, np.conj(down_corr)], axis=0))

        big_pack = np.concatenate(pack2_list, axis=0)  # [n_trials * 2, fps]

        # Denoise batches and record per-frame inference time
        den_gl_all = None
        gl_ms_per_frame = 0.0
        if gl_ok:
            tg0 = time.perf_counter()
            den_gl_all = _denoise_waveforms_gloriphy(big_pack, sf=sf)
            gl_ms_per_frame = ((time.perf_counter() - tg0) * 1000.0) / len(big_pack)

        den_ls_all = None
        ls_ms_per_frame = 0.0
        if ls_ok:
            tl0 = time.perf_counter()
            den_ls_all = _denoise_waveforms_loraseek(big_pack, sf=sf)
            ls_ms_per_frame = ((time.perf_counter() - tl0) * 1000.0) / len(big_pack)

        # Accumulators per SNR
        acc_cfo_ok = {m: 0 for m in active_methods}
        acc_sto_ok = {m: 0 for m in active_methods}
        acc_cfo_err = {m: [] for m in active_methods}
        acc_sto_err = {m: [] for m in active_methods}
        acc_pay = {m: [] for m in active_methods}
        acc_ms = {m: [] for m in active_methods}

        for tr, td in enumerate(trials):
            cfo_true = td["cfo_true"]
            sto_true = td["sto_true"]
            cfo_frac_raw = td["cfo_frac_raw"]

            def score(mname: str, cfo_e: float, sto_e: float, lat_ms: float):
                ce = abs(cfo_e - cfo_true)
                se = abs(sto_e - sto_true)
                acc_cfo_err[mname].append(ce)
                acc_sto_err[mname].append(se)
                if ce <= cfo_tol_hz:
                    acc_cfo_ok[mname] += 1
                if se <= sto_tol_samp:
                    acc_sto_ok[mname] += 1
                acc_pay[mname].append(
                    _eval_payload_after_sync(td["iq"], td["gt_payload"], cfo_e, sto_e, sf, bw, fs, fps)
                )
                acc_ms[mname].append(lat_ms)

            # 1. Pure Conventional (1 up + 1 down)
            tc0 = time.perf_counter()
            cu_s = _dechirp_dual_band_spectrum(td["up_corr"], down_ref, n_classes)
            cd_s = _dechirp_dual_band_spectrum(td["down_corr"], up_ref, n_classes)
            cfo_cs, sto_cs = _solve_cfo_sto_from_spectra(cu_s, cd_s, cfo_frac_raw, bw, fs, n_classes)
            score("conv_single", cfo_cs, sto_cs, td["frac_ms"] + (time.perf_counter() - tc0) * 1000.0)

            # 2. GLoRiPHY Denoised (1 up + 1 conj down)
            if gl_ok and den_gl_all is not None:
                gl_up = den_gl_all[tr * 2]
                gl_down = np.conj(den_gl_all[tr * 2 + 1])

                tg1 = time.perf_counter()
                gu_s = _dechirp_dual_band_spectrum(gl_up, down_ref, n_classes)
                gd_s = _dechirp_dual_band_spectrum(gl_down, up_ref, n_classes)
                cfo_gs, sto_gs = _solve_cfo_sto_from_spectra(gu_s, gd_s, cfo_frac_raw, bw, fs, n_classes)
                score("gloriphy_single", cfo_gs, sto_gs, td["frac_ms"] + 2.0 * gl_ms_per_frame + (time.perf_counter() - tg1) * 1000.0)

            # 3. LoRaSeek Denoised (1 up + 1 conj down)
            if ls_ok and den_ls_all is not None:
                ls_up = den_ls_all[tr * 2]
                ls_down = np.conj(den_ls_all[tr * 2 + 1])

                tl1 = time.perf_counter()
                lu_s = _dechirp_dual_band_spectrum(ls_up, down_ref, n_classes)
                ld_s = _dechirp_dual_band_spectrum(ls_down, up_ref, n_classes)
                cfo_ls, sto_ls = _solve_cfo_sto_from_spectra(lu_s, ld_s, cfo_frac_raw, bw, fs, n_classes)
                score("loraseek_single", cfo_ls, sto_ls, td["frac_ms"] + 2.0 * ls_ms_per_frame + (time.perf_counter() - tl1) * 1000.0)

        for m in active_methods:
            series[m]["cfo_acc"].append(round((acc_cfo_ok[m] / n_trials) * 100.0, 2))
            series[m]["sto_acc"].append(round((acc_sto_ok[m] / n_trials) * 100.0, 2))
            series[m]["cfo_mae"].append(round(float(np.mean(acc_cfo_err[m])), 2))
            series[m]["sto_mae"].append(round(float(np.mean(acc_sto_err[m])), 2))
            series[m]["payload_acc"].append(round(float(np.mean(acc_pay[m])), 2))
            series[m]["avg_ms"].append(round(float(np.mean(acc_ms[m])), 2))

    # Build ultra-low SNR highlight KPIs (pick reference SNR where conv_single is struggling: -18 dB for SF7, -20 dB for SF8)
    target_snr = -18 if sf == 7 else -20
    if target_snr in snr_list:
        ref_idx = snr_list.index(target_snr)
    else:
        ref_idx = len(snr_list) // 2

    ref_snr = snr_list[ref_idx]
    conv_cfo_ref = series["conv_single"]["cfo_acc"][ref_idx]
    conv_sto_ref = series["conv_single"]["sto_acc"][ref_idx]
    conv_pay_ref = series["conv_single"]["payload_acc"][ref_idx]

    highlights: dict[str, Any] = {
        "ref_snr_db": ref_snr,
        "conv_single_cfo_acc": conv_cfo_ref,
        "conv_single_sto_acc": conv_sto_ref,
        "conv_single_pay_acc": conv_pay_ref,
    }
    if "gloriphy_single" in series:
        highlights.update({
            "gloriphy_cfo_acc": series["gloriphy_single"]["cfo_acc"][ref_idx],
            "gloriphy_sto_acc": series["gloriphy_single"]["sto_acc"][ref_idx],
            "gloriphy_pay_acc": series["gloriphy_single"]["payload_acc"][ref_idx],
            "gloriphy_cfo_gain": round(series["gloriphy_single"]["cfo_acc"][ref_idx] - conv_cfo_ref, 1),
            "gloriphy_sto_gain": round(series["gloriphy_single"]["sto_acc"][ref_idx] - conv_sto_ref, 1),
        })
    if "loraseek_single" in series:
        highlights.update({
            "loraseek_cfo_acc": series["loraseek_single"]["cfo_acc"][ref_idx],
            "loraseek_sto_acc": series["loraseek_single"]["sto_acc"][ref_idx],
            "loraseek_pay_acc": series["loraseek_single"]["payload_acc"][ref_idx],
            "loraseek_cfo_gain": round(series["loraseek_single"]["cfo_acc"][ref_idx] - conv_cfo_ref, 1),
            "loraseek_sto_gain": round(series["loraseek_single"]["sto_acc"][ref_idx] - conv_sto_ref, 1),
        })

    return {
        "sf": sf,
        "bw": bw,
        "fs": fs,
        "n_trials": n_trials,
        "payload_len": payload_len,
        "cfo_range_hz": [cfo_lo, cfo_hi],
        "sto_range_samp": [sto_lo, sto_hi],
        "cfo_tol_hz": round(cfo_tol_hz, 2),
        "sto_tol_samp": round(sto_tol_samp, 2),
        "seed": seed,
        "snr_list": snr_list,
        "methods_order": active_methods,
        "methods_meta": {m: METHOD_META[m] for m in active_methods},
        "series": series,
        "highlights": highlights,
        "elapsed_sec": round(time.perf_counter() - t_exp_start, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_saved_results() -> dict[str, Any]:
    """Load precomputed sync experiment results (keyed by 'sf7', 'sf8')."""
    with _file_lock:
        if not RESULTS_FILE.exists():
            return {}
        try:
            return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}


def save_results(sf: int, result: dict[str, Any]) -> None:
    """Persist sync experiment results under key 'sf<N>'."""
    with _file_lock:
        existing = {}
        if RESULTS_FILE.exists():
            try:
                existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing[f"sf{sf}"] = result
        RESULTS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print("Running full SF7 & SF8 Denoised Synchronization Experiments (N=100 trials/SNR)...", flush=True)
    r7 = run_sync_experiment(sf=7, snr_min=-24, snr_max=-10, snr_step=2, n_trials=100, seed=2026)
    save_results(7, r7)
    print(f"SF7 complete in {r7['elapsed_sec']}s -> highlights: {r7['highlights']}", flush=True)

    r8 = run_sync_experiment(sf=8, snr_min=-26, snr_max=-12, snr_step=2, n_trials=100, seed=2026)
    save_results(8, r8)
    print(f"SF8 complete in {r8['elapsed_sec']}s -> highlights: {r8['highlights']}", flush=True)
    print(f"Saved results to {RESULTS_FILE}", flush=True)
