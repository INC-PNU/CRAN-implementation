"""
Physics & signal-processing probe:
Inspect how GLoRiPHY and LoRaSeek behave on:
  1. Up-chirp frames with CFO (integer + fractional) and STO (integer + fractional)
  2. Conjugated down-chirp frames (conj -> denoise -> conj) with CFO and STO
  3. Phase preservation across consecutive preamble frames (for CFO_frac)
  4. Sub-bin energy lean preservation (for STO_frac)
  5. Ultra-low SNR performance (-24 dB to -8 dB)
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.LoRa import LoRa
from utils.my_lora_utils import create_lora_preamble, create_lora_payload, add_cfo
from utils.helper_function import estimate_cfo_frac, wrap_symmetric, to_nearest_N_center, round_left_left_zero, round_right_right_zero, predict_te
from utils import gloriphy_inference, loraseek_inference
from models.gloriphy.gloriphy_model.stft_utils import spec_to_network_input


def denoise_waveforms_gloriphy(frames_np: np.ndarray, sf: int = 7) -> np.ndarray:
    """
    Denoise a batch of up-chirp IQ frames [B, nsamp] using GLoRiPHY ConformerGen_Core
    and reconstruct full-rate time-domain waveforms [B, nsamp] via Hamming ISTFT.
    """
    info = gloriphy_inference.checkpoint_info(None, sf)
    loaded = gloriphy_inference._load(info)
    opts = loaded.opts
    device = gloriphy_inference.DEVICE

    x_t = torch.from_numpy(frames_np.astype(np.complex64)).to(device)
    with torch.no_grad():
        stft_in = gloriphy_inference._stft(x_t, opts)
        net_in = spec_to_network_input(stft_in, opts)
        processed_spec, _, _ = loaded.model(net_in)

        # Reconstruct full-rate complex waveform via ISTFT (same as stft_utils.reconstruct_from_stft_batched)
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


def denoise_waveforms_loraseek(frames_np: np.ndarray, sf: int = 7, full_rate: bool = True) -> np.ndarray:
    """
    Denoise a batch of up-chirp IQ frames [B, nsamp] using LoRaSeekNet.
    If full_rate=True: zero-pad central 2^SF bins into n_fft=nsamp and ISTFT at Fs=1 MHz (length nsamp).
    If full_rate=False: Nyquist ISTFT at Fs=BW (length 2^SF).
    """
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

        denoised_net = model(net_in)
        spec_complex = torch.complex(denoised_net[:, 0], denoised_net[:, 1])  # [B, freq_size, 32]

        if full_rate:
            B, _, T = spec_complex.shape
            # Pad to T=33 (repeat last or zero) so ISTFT matches nsamp, or use length=opts.stft_nfft
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
        else:
            win_i = torch.hann_window(opts.istft_window, device=device)
            spec_shifted = torch.fft.ifftshift(spec_complex, dim=1)
            recon = torch.istft(
                spec_shifted,
                n_fft=opts.istft_nfft,
                hop_length=opts.istft_overlap,
                win_length=opts.istft_window,
                window=win_i,
                onesided=False,
                return_complex=True,
                center=True,
            ).cpu().numpy()
            n_cls = 2 ** sf
            if recon.shape[-1] < n_cls:
                recon = np.pad(recon, ((0, 0), (0, n_cls - recon.shape[-1])))
            else:
                recon = recon[:, :n_cls]
            return recon


def dechirp_dual_band(frame, ref_chirp, n_classes):
    dechirped = frame * ref_chirp
    if len(frame) == n_classes:
        combine = np.abs(np.fft.fft(dechirped))
    else:
        psd = np.abs(np.fft.fftshift(np.fft.fft(dechirped)))
        center = len(psd) // 2
        combine = psd[center : center + n_classes] + psd[center - n_classes : center]
    return combine, int(np.argmax(combine))


def estimate_cfo_sto_from_pair(up_frame, down_frame, cfo_frac, opts, up_ref, down_ref, use_frac_sto=True):
    n_classes = opts.n_classes
    over_sample = opts.fs / opts.bw

    combine_up, symbol_up = dechirp_dual_band(up_frame, down_ref, n_classes)
    combine_down, symbol_down = dechirp_dual_band(down_frame, up_ref, n_classes)

    # Fractional STO from up-chirp sidebins (same as helper_function.py lines 418-440)
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

    # Down-chirp rounding tie-break (same as helper_function.py lines 455-486)
    next_idx_d = (symbol_down + 1) % n_classes
    prev_idx_d = (symbol_down - 1 + n_classes) % n_classes
    type_type = 1 if combine_down[next_idx_d] > combine_down[prev_idx_d] else 2

    cfo_bins = to_nearest_N_center((symbol_up + symbol_down) / 2.0, n_classes)
    cfo_bins = round_left_left_zero(cfo_bins) if type_type == 1 else round_right_right_zero(cfo_bins)

    cfo_int_hz = (cfo_bins * opts.bw) / n_classes
    cfo_total = cfo_int_hz + cfo_frac

    sto_int_chips = wrap_symmetric(cfo_bins + n_classes - symbol_up, n_classes)
    sto_int_samp = sto_int_chips * over_sample
    sto_total = sto_int_samp + shift_sto_index

    return {
        "symbol_up": symbol_up,
        "symbol_down": symbol_down,
        "cfo_int_hz": cfo_int_hz,
        "cfo_frac_hz": cfo_frac,
        "cfo_total_hz": cfo_total,
        "sto_int_samp": sto_int_samp,
        "sto_frac_samp": shift_sto_index,
        "sto_total_samp": sto_total,
    }


def main():
    sf = 7
    bw = 125_000
    fs = 1_000_000
    n_classes = 2 ** sf
    fps = int(n_classes * (fs / bw))

    opts = type("", (), {})()
    opts.sf = sf
    opts.bw = bw
    opts.fs = fs
    opts.n_classes = n_classes
    opts.no_of_preamble = 8
    opts.sync_sym = [8, 8]

    lora = LoRa(sf, bw)
    up_ref_fs = lora.gen_symbol_fs(0, down=False, Fs=fs)
    down_ref_fs = np.conj(up_ref_fs)
    up_ref_bw = lora.gen_symbol_fs(0, down=False, Fs=bw)
    down_ref_bw = np.conj(up_ref_bw)

    # Test clean (+30 dB) signal with BOTH fractional+integer CFO and STO
    cfo_true = 2345.6   # ~2.40 bins (integer 2 bins = 1953.125 Hz, frac = +392.5 Hz)
    numb_offset = 27    # sto_true = -27 samples (-3.375 chips)
    sto_true = wrap_symmetric(-numb_offset, fps)

    preamble = create_lora_preamble(opts, LoRa)
    payload = create_lora_payload(opts, LoRa, [10, 20, 30, 40])
    sig = np.concatenate([preamble, payload]).astype(np.complex64)
    sig = add_cfo(opts, sig, cfo_true)
    sig = sig[numb_offset:]
    iq = lora.awgn_iq_with_seed(sig, 30.0, seed=7)

    # Known frame positions: up-chirps at 0..7, sync at 8..9, down-chirps at 10..11
    fdown = 10
    fup = fdown - 5  # frame 5
    t = np.arange(fps) / fs

    # 1. Raw fractional CFO
    cfo_frac_raw = estimate_cfo_frac(opts, iq, fdown, down_ref_fs)
    corr_frac = np.exp(-1j * 2 * np.pi * cfo_frac_raw * t)

    up_raw = iq[fup * fps : (fup + 1) * fps] * corr_frac
    down_raw = iq[fdown * fps : (fdown + 1) * fps] * corr_frac

    # Check phase preservation of GLoRiPHY & LoRaSeek on 2 consecutive up-chirps
    pair_up = np.stack([iq[2 * fps : 3 * fps], iq[3 * fps : 4 * fps]], axis=0)
    raw_phase = np.angle(np.vdot(pair_up[0] * down_ref_fs, pair_up[1] * down_ref_fs))
    gl_pair = denoise_waveforms_gloriphy(pair_up, sf)
    gl_phase = np.angle(np.vdot(gl_pair[0] * down_ref_fs, gl_pair[1] * down_ref_fs))
    ls_pair = denoise_waveforms_loraseek(pair_up, sf, full_rate=True)
    ls_phase = np.angle(np.vdot(ls_pair[0] * down_ref_fs, ls_pair[1] * down_ref_fs))

    print(f"=== CLEAN (+30 dB) PHYSICS CHECK ===")
    print(f"GT: cfo_true = {cfo_true:.2f} Hz, sto_true = {sto_true} samp")
    print(f"Inter-symbol phase diff (for CFO_frac): raw={raw_phase:.4f} rad, GLoRiPHY={gl_phase:.4f} rad, LoRaSeek={ls_phase:.4f} rad")

    # Conventional raw
    res_conv = estimate_cfo_sto_from_pair(up_raw, down_raw, cfo_frac_raw, opts, up_ref_fs, down_ref_fs)
    print("\n[Conventional Raw]")
    print(" ", res_conv, f"| cfo_err={res_conv['cfo_total_hz']-cfo_true:+.2f} Hz, sto_err={res_conv['sto_total_samp']-sto_true:+.2f} samp")

    # GLoRiPHY: denoise up_raw and conj(down_raw) -> conj back
    batch_gl = np.stack([up_raw, np.conj(down_raw)], axis=0)
    den_gl = denoise_waveforms_gloriphy(batch_gl, sf)
    up_gl, down_gl = den_gl[0], np.conj(den_gl[1])
    res_gl = estimate_cfo_sto_from_pair(up_gl, down_gl, cfo_frac_raw, opts, up_ref_fs, down_ref_fs)
    print("\n[GLoRiPHY Denoised (conj trick for down-chirp)]")
    print(" ", res_gl, f"| cfo_err={res_gl['cfo_total_hz']-cfo_true:+.2f} Hz, sto_err={res_gl['sto_total_samp']-sto_true:+.2f} samp")

    # LoRaSeek full-rate
    den_ls_full = denoise_waveforms_loraseek(batch_gl, sf, full_rate=True)
    up_ls_f, down_ls_f = den_ls_full[0], np.conj(den_ls_full[1])
    res_ls_f = estimate_cfo_sto_from_pair(up_ls_f, down_ls_f, cfo_frac_raw, opts, up_ref_fs, down_ref_fs)
    print("\n[LoRaSeek Denoised Full-Rate ISTFT (conj trick)]")
    print(" ", res_ls_f, f"| cfo_err={res_ls_f['cfo_total_hz']-cfo_true:+.2f} Hz, sto_err={res_ls_f['sto_total_samp']-sto_true:+.2f} samp")

    # LoRaSeek Nyquist-rate
    den_ls_nyq = denoise_waveforms_loraseek(batch_gl, sf, full_rate=False)
    up_ls_n, down_ls_n = den_ls_nyq[0], np.conj(den_ls_nyq[1])
    res_ls_n = estimate_cfo_sto_from_pair(up_ls_n, down_ls_n, cfo_frac_raw, opts, up_ref_bw, down_ref_bw)
    print("\n[LoRaSeek Denoised Nyquist ISTFT (conj trick)]")
    print(" ", res_ls_n, f"| cfo_err={res_ls_n['cfo_total_hz']-cfo_true:+.2f} Hz, sto_err={res_ls_n['sto_total_samp']-sto_true:+.2f} samp")


if __name__ == "__main__":
    main()
