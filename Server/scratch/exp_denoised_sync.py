"""
Early Scientific Experiment:
Compare CFO & STO estimation accuracy across Ultra-Low SNR (-24 dB to -8 dB):
  1. Pure Conventional Sync (Raw up-chirp & down-chirp dechirping, exact helper_function.py v3 math)
  2. Conventional Multi-Chirp Sync (Avg FFT over 6 up-chirps & 2 down-chirps — classical ceiling baseline)
  3. GLoRiPHY-Denoised Sync (Denoise 1 up-chirp + 1 conj down-chirp with GLoRiPHY, raw CFO_frac)
  4. GLoRiPHY-Denoised Multi-Chirp Sync (Denoise 6 up-chirps + 2 conj down-chirps with GLoRiPHY)
  5. LoRaSeek-Denoised Sync (Denoise 1 up-chirp + 1 conj down-chirp with LoRaSeek, raw CFO_frac)
  6. LoRaSeek-Denoised Full Sync (Denoise preamble up-chirps for CFO_frac + up/down chirps with LoRaSeek)
  7. LoRaSeek-Denoised Multi-Chirp Sync (Denoise 6 up-chirps + 2 conj down-chirps with LoRaSeek)

Note: To isolate CFO & STO synchronization accuracy cleanly (assuming preamble frame is located,
e.g. via CALoRa or known frame boundary), we pass the exact frame window index (fdown = 10, fup = 5)
so every method operates on the exact same noisy packet IQ.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.LoRa import LoRa
from utils.my_lora_utils import create_lora_preamble, create_lora_payload, add_cfo
from utils.helper_function import (
    estimate_cfo_frac,
    wrap_symmetric,
    to_nearest_N_center,
    round_left_left_zero,
    round_right_right_zero,
    predict_te,
)
from utils import gloriphy_inference, loraseek_inference
from models.gloriphy.gloriphy_model.stft_utils import spec_to_network_input


def denoise_gloriphy(frames_np: np.ndarray, sf: int = 7) -> np.ndarray:
    info = gloriphy_inference.checkpoint_info(None, sf)
    loaded = gloriphy_inference._load(info)
    opts = loaded.opts
    device = gloriphy_inference.DEVICE

    x_t = torch.from_numpy(frames_np.astype(np.complex64)).to(device)
    with torch.no_grad():
        stft_in = gloriphy_inference._stft(x_t, opts)
        net_in = spec_to_network_input(stft_in, opts)
        processed_spec, _, _ = loaded.model(net_in)

        proc_c = torch.view_as_complex(processed_spec.permute(0, 2, 3, 1).contiguous())
        B, _, T = proc_c.shape
        trim_size = opts.freq_size // 2
        full_spec = torch.zeros((B, opts.demod.d_num_samples, T), dtype=torch.complex64, device=device)
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
    return recon.cpu().numpy()


def denoise_loraseek(frames_np: np.ndarray, sf: int = 7) -> np.ndarray:
    ckpt = "75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl"
    model = loraseek_inference._load_loraseek_model(ckpt, sf=sf)
    opts = loraseek_inference._LoRaSeekOptsSF8() if sf == 8 else loraseek_inference._LoRaSeekOptsSF7()
    device = loraseek_inference.DEVICE

    x_t = torch.from_numpy(frames_np.astype(np.complex64)).to(device)
    window = torch.hann_window(opts.stft_window, device=device)
    with torch.no_grad():
        raw_spec = torch.stft(
            input=x_t,
            n_fft=opts.stft_nfft,
            hop_length=opts.stft_overlap,
            win_length=opts.stft_window,
            window=window,
            pad_mode="constant",
            return_complex=True,
        )
        trim_size = opts.freq_size // 2
        y = torch.cat((raw_spec[:, -trim_size:, :], raw_spec[:, 0:trim_size, :]), dim=1)
        y_abs_max = torch.clamp(torch.amax(torch.abs(y), dim=(1, 2), keepdim=True), min=1e-8)
        y_norm = y / y_abs_max
        net_in = torch.view_as_real(y_norm).permute(0, 3, 1, 2).to(dtype=torch.float32)[:, :, :, :32]

        # Mini-batch if large
        if net_in.shape[0] <= 64:
            denoised_net = model(net_in)
        else:
            denoised_net = torch.cat([model(net_in[i:i+64]) for i in range(0, net_in.shape[0], 64)], dim=0)

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

    nsamp = opts.stft_nfft
    if recon.shape[-1] < nsamp:
        recon = np.pad(recon, ((0, 0), (0, nsamp - recon.shape[-1])))
    else:
        recon = recon[:, :nsamp]
    return recon


def dechirp_spectrum(frame: np.ndarray, ref_chirp: np.ndarray, n_classes: int) -> np.ndarray:
    dechirped = frame * ref_chirp
    psd = np.abs(np.fft.fftshift(np.fft.fft(dechirped)))
    center = len(psd) // 2
    return psd[center : center + n_classes] + psd[center - n_classes : center]


def estimate_cfo_frac_from_frames(frames_up: np.ndarray, down_ref: np.ndarray, bw: int, n_classes: int) -> float:
    """Consecutive up-chirp phase difference CFO_frac estimator (exact helper_function.py formula)."""
    z_sum = 0j
    for k in range(len(frames_up) - 1):
        d1 = frames_up[k] * down_ref
        d2 = frames_up[k + 1] * down_ref
        z_sum += np.vdot(d1, d2)
    phase_diff = np.angle(z_sum)
    return float((phase_diff * bw) / (2.0 * np.pi * n_classes))


def solve_cfo_sto_from_spectra(combine_up, combine_down, cfo_frac, opts, use_frac_sto=True):
    n_classes = opts.n_classes
    over_sample = opts.fs / opts.bw

    symbol_up = int(np.argmax(combine_up))
    symbol_down = int(np.argmax(combine_down))

    shift_sto_index = 0.0
    if use_frac_sto:
        next_idx = (symbol_up + 1) % n_classes
        prev_idx = (symbol_up - 1 + n_classes) % n_classes
        prev_val = combine_up[prev_idx]
        next_val = combine_up[next_idx]
        peak_val = max(float(combine_up[symbol_up]), 1e-12)
        def te_nl(te, A=4.0, B=2.5):
            return A * np.tanh(B * te)
        if next_val > prev_val:
            sto_off = te_nl((next_val - prev_val) / peak_val)
            shift_sto_index = float(np.round(predict_te(sto_off * -1) * over_sample) * -1)
        else:
            sto_off = te_nl((prev_val - next_val) / peak_val)
            shift_sto_index = float(np.round(predict_te(sto_off * -1) * over_sample))

    next_idx_d = (symbol_down + 1) % n_classes
    prev_idx_d = (symbol_down - 1 + n_classes) % n_classes
    type_type = 1 if combine_down[next_idx_d] > combine_down[prev_idx_d] else 2

    cfo_bins = to_nearest_N_center((symbol_up + symbol_down) / 2.0, n_classes)
    cfo_bins = round_left_left_zero(cfo_bins) if type_type == 1 else round_right_right_zero(cfo_bins)

    cfo_int_hz = (cfo_bins * opts.bw) / n_classes
    cfo_total = cfo_int_hz + cfo_frac

    sto_int_chips = wrap_symmetric(cfo_bins + n_classes - symbol_up, n_classes)
    sto_total = sto_int_chips * over_sample + shift_sto_index
    return cfo_total, sto_total


def run_experiment(sf=7, n_trials=40, snr_list=(-24, -22, -20, -18, -16, -14, -12, -10)):
    bw = 125_000
    fs = 1_000_000
    n_classes = 2 ** sf
    fps = int(n_classes * (fs / bw))
    cfo_tol = bw / (2.0 * n_classes)   # 488.28 Hz for SF7
    sto_tol = fs / bw                  # 8.0 samples

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
    t = np.arange(fps) / fs

    # Pre-warm models
    gloriphy_inference._load(gloriphy_inference.checkpoint_info(None, sf))
    loraseek_inference._load_loraseek_model("75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl", sf=sf)

    methods = [
        "conv_single",
        "conv_multi",
        "gloriphy_single",
        "gloriphy_multi",
        "loraseek_single",
        "loraseek_full",
        "loraseek_multi",
    ]
    stats = {m: {s: {"cfo_err": [], "sto_err": [], "cfo_ok": 0, "sto_ok": 0} for s in snr_list} for m in methods}

    rng = np.random.default_rng(2026)

    for snr in snr_list:
        t0 = time.perf_counter()
        # Generate all n_trials packets for this SNR and batch neural calls for speed!
        trials_data = []
        all_gl_frames = []   # 8 frames per trial: 6 up (frames 1..6) + 2 conj(down) (frames 10..11)
        all_ls_raw_up = []   # 6 raw up frames per trial BEFORE frac CFO precorrection (for loraseek_full CFO_frac)

        for tr in range(n_trials):
            cfo_true = float(rng.uniform(-4358.0, 4584.0))
            numb_offset = int(rng.integers(0, 51))
            sto_true = float(wrap_symmetric(-numb_offset, fps))

            preamble = create_lora_preamble(opts, LoRa)
            payload = create_lora_payload(opts, LoRa, [0, 1])
            sig = np.concatenate([preamble, payload]).astype(np.complex64)
            sig = add_cfo(opts, sig, cfo_true)
            sig = sig[numb_offset:]
            iq = lora.awgn_iq_with_seed(sig, float(snr), seed=abs(snr) * 10000 + tr + 1)

            # Extract raw frames: preamble up-chirps 1..6 (avoid frame 0 partial edge), down-chirps 10..11
            raw_ups_uncorr = np.stack([iq[k * fps : (k + 1) * fps] for k in range(1, 7)], axis=0)
            raw_downs_uncorr = np.stack([iq[k * fps : (k + 1) * fps] for k in (10, 11)], axis=0)

            # Raw CFO_frac from preamble frames 0..5 (matching helper_function.py lines 123-134: range(0, 5) -> 5 pairs)
            raw_ups_for_frac = np.stack([iq[k * fps : (k + 1) * fps] for k in range(0, 6)], axis=0)
            cfo_frac_raw = estimate_cfo_frac_from_frames(raw_ups_for_frac, down_ref, bw, n_classes)

            # Precorrect fractional CFO on up & down frames
            corr = np.exp(-1j * 2 * np.pi * cfo_frac_raw * t)
            ups_corr = raw_ups_uncorr * corr[None, :]
            downs_corr = raw_downs_uncorr * corr[None, :]

            trials_data.append({
                "cfo_true": cfo_true,
                "sto_true": sto_true,
                "cfo_frac_raw": cfo_frac_raw,
                "ups_corr": ups_corr,       # index 4 corresponds to frame 5 (fup_chosen = fdown - 5)
                "downs_corr": downs_corr,   # index 0 corresponds to frame 10 (fdown = 10)
                "raw_ups_uncorr": raw_ups_uncorr,
                "raw_downs_uncorr": raw_downs_uncorr,
            })

            # Pack 8 frames: 6 precorrected up-chirps + 2 conjugated precorrected down-chirps
            pack8 = np.concatenate([ups_corr, np.conj(downs_corr)], axis=0)
            all_gl_frames.append(pack8)
            all_ls_raw_up.append(raw_ups_for_frac)

        # Batched neural denoising across all trials at this SNR
        big_pack = np.concatenate(all_gl_frames, axis=0)  # [n_trials * 8, fps]
        den_gl_all = denoise_gloriphy(big_pack, sf)
        den_ls_all = denoise_loraseek(big_pack, sf)

        # Also denoise uncorrected up-chirps with LoRaSeek for loraseek_full CFO_frac
        big_ls_frac = np.concatenate(all_ls_raw_up, axis=0)  # [n_trials * 6, fps]
        den_ls_frac_all = denoise_loraseek(big_ls_frac, sf)

        for tr in range(n_trials):
            td = trials_data[tr]
            cfo_true = td["cfo_true"]
            sto_true = td["sto_true"]
            cfo_frac_raw = td["cfo_frac_raw"]

            def record(mname, cfo_e, sto_e):
                ce = abs(cfo_e - cfo_true)
                se = abs(sto_e - sto_true)
                stats[mname][snr]["cfo_err"].append(ce)
                stats[mname][snr]["sto_err"].append(se)
                if ce <= cfo_tol:
                    stats[mname][snr]["cfo_ok"] += 1
                if se <= sto_tol:
                    stats[mname][snr]["sto_ok"] += 1

            # 1. conv_single (frame 5 up, frame 10 down)
            cu_s = dechirp_spectrum(td["ups_corr"][4], down_ref, n_classes)
            cd_s = dechirp_spectrum(td["downs_corr"][0], up_ref, n_classes)
            record("conv_single", *solve_cfo_sto_from_spectra(cu_s, cd_s, cfo_frac_raw, opts))

            # 2. conv_multi (non-coherent sum of 6 up-chirp spectra and 2 down-chirp spectra)
            cu_m = sum(dechirp_spectrum(td["ups_corr"][k], down_ref, n_classes) for k in range(6))
            cd_m = sum(dechirp_spectrum(td["downs_corr"][k], up_ref, n_classes) for k in range(2))
            record("conv_multi", *solve_cfo_sto_from_spectra(cu_m, cd_m, cfo_frac_raw, opts))

            # Slice denoised 8-frame block for this trial
            gl_8 = den_gl_all[tr * 8 : (tr + 1) * 8]
            gl_ups = gl_8[:6]
            gl_downs = np.conj(gl_8[6:8])

            # 3. gloriphy_single
            gu_s = dechirp_spectrum(gl_ups[4], down_ref, n_classes)
            gd_s = dechirp_spectrum(gl_downs[0], up_ref, n_classes)
            record("gloriphy_single", *solve_cfo_sto_from_spectra(gu_s, gd_s, cfo_frac_raw, opts))

            # 4. gloriphy_multi
            gu_m = sum(dechirp_spectrum(gl_ups[k], down_ref, n_classes) for k in range(6))
            gd_m = sum(dechirp_spectrum(gl_downs[k], up_ref, n_classes) for k in range(2))
            record("gloriphy_multi", *solve_cfo_sto_from_spectra(gu_m, gd_m, cfo_frac_raw, opts))

            # Slice LoRaSeek denoised 8-frame block
            ls_8 = den_ls_all[tr * 8 : (tr + 1) * 8]
            ls_ups = ls_8[:6]
            ls_downs = np.conj(ls_8[6:8])

            # 5. loraseek_single (raw CFO_frac)
            lu_s = dechirp_spectrum(ls_ups[4], down_ref, n_classes)
            ld_s = dechirp_spectrum(ls_downs[0], up_ref, n_classes)
            record("loraseek_single", *solve_cfo_sto_from_spectra(lu_s, ld_s, cfo_frac_raw, opts))

            # 6. loraseek_full (denoised CFO_frac from LoRaSeek + denoised single up/down)
            ls_frac_frames = den_ls_frac_all[tr * 6 : (tr + 1) * 6]
            cfo_frac_ls = estimate_cfo_frac_from_frames(ls_frac_frames, down_ref, bw, n_classes)
            record("loraseek_full", *solve_cfo_sto_from_spectra(lu_s, ld_s, cfo_frac_ls, opts))

            # 7. loraseek_multi (raw CFO_frac + 6 up / 2 down denoised spectra)
            lu_m = sum(dechirp_spectrum(ls_ups[k], down_ref, n_classes) for k in range(6))
            ld_m = sum(dechirp_spectrum(ls_downs[k], up_ref, n_classes) for k in range(2))
            record("loraseek_multi", *solve_cfo_sto_from_spectra(lu_m, ld_m, cfo_frac_raw, opts))

        dt = time.perf_counter() - t0
        print(f"SNR={snr:3d} dB done in {dt:.1f}s", flush=True)

    print(f"\n=== SF{sf} CFO ACCURACY (% within ±{cfo_tol:.1f} Hz) [N={n_trials}/SNR] ===")
    hdr = f"{'SNR':>5} | " + " | ".join(f"{m:>15}" for m in methods)
    print(hdr)
    print("-" * len(hdr))
    for s in snr_list:
        print(f"{s:5d} | " + " | ".join(f"{stats[m][s]['cfo_ok']/n_trials*100:14.1f}%" for m in methods))

    print(f"\n=== SF{sf} STO ACCURACY (% within ±{sto_tol:.1f} samples) [N={n_trials}/SNR] ===")
    print(hdr)
    print("-" * len(hdr))
    for s in snr_list:
        print(f"{s:5d} | " + " | ".join(f"{stats[m][s]['sto_ok']/n_trials*100:14.1f}%" for m in methods))

    print(f"\n=== SF{sf} CFO MAE (Hz) ===")
    print(hdr)
    print("-" * len(hdr))
    for s in snr_list:
        print(f"{s:5d} | " + " | ".join(f"{np.mean(stats[m][s]['cfo_err']):15.1f}" for m in methods))

    print(f"\n=== SF{sf} STO MAE (samples) ===")
    print(hdr)
    print("-" * len(hdr))
    for s in snr_list:
        print(f"{s:5d} | " + " | ".join(f"{np.mean(stats[m][s]['sto_err']):15.2f}" for m in methods))


if __name__ == "__main__":
    run_experiment(sf=7, n_trials=40)
