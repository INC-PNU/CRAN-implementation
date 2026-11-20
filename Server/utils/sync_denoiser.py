"""
Shared full-rate neural waveform denoising helpers for Stage 2 LoRa synchronization
(CFO & STO estimation via up-chirp / down-chirp dechirping).

Used by:
  - Server/utils/helper_function.py (detect_cfo_sto when sync_denoiser != "none")
  - Server/dashboard/orchestrator.py (Tab 1 full-pipeline sweep)
  - Server/dashboard/sync_experiment.py (Tab 3 denoised sync experiment)

Key physics:
  Both LoRaSeek and GLoRiPHY are trained exclusively on LoRa UP-chirps.
  For SFD DOWN-chirps, complex conjugation turns a down-chirp into an exact up-chirp
  with identical timing offset and negated CFO. Denoising conj(down_frame) and taking
  conj(...) of the output yields a clean down-chirp preserving both +CFO and +STO.
"""
from __future__ import annotations

import numpy as np
import torch

from utils import gloriphy_inference, loraseek_inference
from models.gloriphy.gloriphy_model.stft_utils import spec_to_network_input

VALID_SYNC_DENOISERS = ("none", "loraseek", "gloriphy")

_hann_cache: dict[tuple[int, str], torch.Tensor] = {}


def _get_hann_window(win_len: int, device: torch.device) -> torch.Tensor:
    key = (win_len, str(device))
    w = _hann_cache.get(key)
    if w is None:
        w = torch.hann_window(win_len, device=device)
        _hann_cache[key] = w
    return w


def is_sync_denoiser_available(method: str, sf: int = 7) -> bool:
    if method == "none":
        return True
    if method == "loraseek":
        return (sf == 7 and loraseek_inference.LORASEEK_SF7_AVAILABLE) or (
            sf == 8 and loraseek_inference.LORASEEK_SF8_AVAILABLE
        )
    if method == "gloriphy":
        return gloriphy_inference.GLORIPHY_AVAILABLE and (sf in gloriphy_inference.SUPPORTED_SFS)
    return False


def denoise_upchirp_waveforms(
    frames_np: np.ndarray,
    method: str,
    sf: int = 7,
    batch_size: int = 64,
) -> np.ndarray:
    """Denoise a batch of full-rate up-chirp IQ frames [N, nsamp] -> [N, nsamp]."""
    if method in (None, "", "none"):
        return frames_np

    squeeze = False
    if frames_np.ndim == 1:
        frames_np = frames_np[None, :]
        squeeze = True

    if method == "loraseek":
        out = _denoise_loraseek(frames_np, sf=sf, batch_size=batch_size)
    elif method == "gloriphy":
        out = _denoise_gloriphy(frames_np, sf=sf, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown sync_denoiser: {method!r} (expected one of {VALID_SYNC_DENOISERS})")

    return out[0] if squeeze else out


def denoise_downchirp_waveforms(
    frames_np: np.ndarray,
    method: str,
    sf: int = 7,
    batch_size: int = 64,
) -> np.ndarray:
    """Denoise SFD down-chirp IQ frame(s) via complex-conjugate up-chirp symmetry."""
    if method in (None, "", "none"):
        return frames_np
    return np.conj(denoise_upchirp_waveforms(np.conj(frames_np), method=method, sf=sf, batch_size=batch_size))


def _denoise_gloriphy(frames_np: np.ndarray, sf: int = 7, batch_size: int = 64) -> np.ndarray:
    info = gloriphy_inference.checkpoint_info(None, sf)
    loaded = gloriphy_inference._load(info)
    opts = loaded.opts
    device = gloriphy_inference.DEVICE
    nsamp = frames_np.shape[-1]

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
            ).cpu().numpy()

            if recon.shape[-1] < nsamp:
                recon = np.pad(recon, ((0, 0), (0, nsamp - recon.shape[-1])))
            else:
                recon = recon[:, :nsamp]
            out_chunks.append(recon)

    return np.concatenate(out_chunks, axis=0)


def _denoise_loraseek(frames_np: np.ndarray, sf: int = 7, batch_size: int = 64) -> np.ndarray:
    ckpt = "75000_LoRaSeek.pkl" if sf == 7 else "90000_LoRaSeek.pkl"
    model = loraseek_inference._load_loraseek_model(ckpt, sf=sf)
    opts = loraseek_inference._LoRaSeekOptsSF8() if sf == 8 else loraseek_inference._LoRaSeekOptsSF7()
    device = loraseek_inference.DEVICE
    window = _get_hann_window(opts.stft_window, device)
    trim_size = opts.freq_size // 2
    nsamp = frames_np.shape[-1]

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
