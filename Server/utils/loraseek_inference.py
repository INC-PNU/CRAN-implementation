"""
LoRaSeek demodulation wrapper — SF7 only (trained with checkpoint 90000_LoRaSeek.pkl
located at Server/models/loraseek/checkpoints/sf7/v9_checkpoints/).

LoRaSeek operates as a neural denoiser in the STFT frequency domain:
  1. Segments payload IQ into one-symbol chunks (1024 samples for SF7 @ 1MHz / 125kHz BW).
  2. Computes Hann-windowed STFT for each symbol chunk.
  3. Formats normalized central frequency bins into tensor [B, 2, 128, 32].
  4. Runs forward pass through LoRaSeekNet (Transformer + Dual Attention U-Net).
  5. Reconstructs denoised time-domain complex IQ via ISTFT.
  6. Standard dechirping + FFT peak detection on the reconstructed clean signal.
"""
import threading
from pathlib import Path

import numpy as np
import torch

from utils.LoRa import LoRa
from models.loraseek.loraseek_model.model_LoRaSeek import LoRaSeekNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CHECKPOINTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "loraseek"
    / "checkpoints"
    / "sf7"
    / "v9_checkpoints"
)

AVAILABLE_CHECKPOINTS = {
    "90000_LoRaSeek.pkl": "90000_LoRaSeek.pkl",
    "75000_LoRaSeek.pkl": "75000_LoRaSeek.pkl",
    "90000": "90000_LoRaSeek.pkl",
    "75000": "75000_LoRaSeek.pkl",
}
DEFAULT_CHECKPOINT = "90000_LoRaSeek.pkl"

LORASEEK_AVAILABLE = (_CHECKPOINTS_DIR / DEFAULT_CHECKPOINT).exists()


def list_loraseek_checkpoints() -> list[dict]:
    """Returns available LoRaSeek checkpoint options for the dashboard."""
    checkpoints = [
        {
            "filename": "90000_LoRaSeek.pkl",
            "label": "90000 iterations (default)",
            "is_default": True,
            "exists": (_CHECKPOINTS_DIR / "90000_LoRaSeek.pkl").exists(),
        },
        {
            "filename": "75000_LoRaSeek.pkl",
            "label": "75000 iterations (best model)",
            "is_default": False,
            "exists": (_CHECKPOINTS_DIR / "75000_LoRaSeek.pkl").exists(),
        },
    ]
    return checkpoints


class _LoRaSeekOpts:
    x_image_channel = 2
    y_image_channel = 2
    base_channel = 16
    channel_attention_reduction = 8
    num_of_transformers = 4
    num_of_heads = 2
    conv_kernel_size = 3
    conv_padding_size = 1
    sf = 7
    bw = 125_000
    fs = 1_000_000
    n_classes = 128
    freq_size = 128
    t_ = 33
    normalization = 1
    stft_nfft = 1024
    stft_window = 64
    stft_overlap = 32
    istft_nfft = 128
    istft_window = 16
    istft_overlap = 4


_lock = threading.Lock()
_loraseek_models: dict = {}


def _resolve_checkpoint_filename(checkpoint_name: str | None) -> str:
    if not checkpoint_name:
        return DEFAULT_CHECKPOINT
    return AVAILABLE_CHECKPOINTS.get(checkpoint_name, checkpoint_name)


def _load_loraseek_model(checkpoint_name: str | None = None):
    """Lazy, thread-safe model loading cached by checkpoint name."""
    canonical_name = _resolve_checkpoint_filename(checkpoint_name)
    if canonical_name in _loraseek_models:
        return _loraseek_models[canonical_name]

    with _lock:
        if canonical_name in _loraseek_models:
            return _loraseek_models[canonical_name]

        ckpt_path = _CHECKPOINTS_DIR / canonical_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LoRaSeek checkpoint not found: {ckpt_path}")

        opts = _LoRaSeekOpts()
        model = LoRaSeekNet(opts)
        weights = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(weights, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in weights:
                    weights = weights[key]
                    break
        model.load_state_dict(weights, strict=True)
        model.to(DEVICE).eval()
        _loraseek_models[canonical_name] = model
        return model


def demodulate_payload_loraseek(
    payload_iq: np.ndarray,
    sf: int,
    bw: int,
    fs: int,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
) -> list[int]:
    """
    Demodulate LoRa payload IQ samples using LoRaSeek neural denoising followed by
    standard dechirping.
    Same contract as calculate_symbol_alliqfile_with_down_sampling and
    demodulate_payload_nelora.
    """
    if not LORASEEK_AVAILABLE:
        raise RuntimeError("LoRaSeek model checkpoint is not available on disk.")
    if sf != 7:
        raise ValueError(f"LoRaSeek is SF7-only in this version, got sf={sf}")

    model = _load_loraseek_model(checkpoint_name)
    opts = _LoRaSeekOpts()

    nsamp = (2 ** sf) * (fs // bw)
    num_symbols = int(round(len(payload_iq) / nsamp))
    if num_symbols == 0:
        return []

    # 1. Slice payload into symbol chunks of nsamp
    chunks = []
    for i in range(num_symbols):
        st = i * nsamp
        en = st + nsamp
        if en <= len(payload_iq):
            chunks.append(payload_iq[st:en])
        else:
            chunks.append(np.pad(payload_iq[st:].astype(np.complex64), (0, en - len(payload_iq))))

    batch_np = np.stack(chunks, axis=0)
    batch_t = torch.from_numpy(batch_np).to(dtype=torch.complex64, device=DEVICE)

    # 2. Compute Hann-windowed STFT
    window = torch.hann_window(opts.stft_window, device=DEVICE)
    raw_spec = torch.stft(
        input=batch_t,
        n_fft=opts.stft_nfft,
        hop_length=opts.stft_overlap,
        win_length=opts.stft_window,
        window=window,
        pad_mode="constant",
        return_complex=True,
    )

    # 3. Trim central bandwidth and normalize
    trim_size = opts.freq_size // 2
    y = torch.cat((raw_spec[:, -trim_size:, :], raw_spec[:, 0:trim_size, :]), dim=1)
    y_abs = torch.abs(y)
    y_abs_max = torch.amax(y_abs, dim=(1, 2), keepdim=True)
    y_abs_max = torch.clamp(y_abs_max, min=1e-8)
    y_norm = torch.div(y, y_abs_max)

    # Format into [B, 2, 128, 32]
    net_in = torch.view_as_real(y_norm).permute(0, 3, 1, 2)
    net_in_32 = net_in[:, :, :, :32]

    # 4. Neural denoising forward pass
    with torch.no_grad():
        denoised_net = model(net_in_32)

    # 5. Inverse STFT at Nyquist rate (istft_nfft=128, istft_window=16, istft_overlap=4)
    # The LoRaSeek model denoises the cropped central 128-bin spectrum.
    # Reconstructing directly with istft_nfft=128 yields clean time-domain symbols
    # sampled at Fs = BW = 125 kHz (128 samples per symbol).
    spec_complex = torch.complex(denoised_net[:, 0], denoised_net[:, 1])
    window_istft = torch.hann_window(opts.istft_window, device=DEVICE)
    spec_shifted = torch.fft.ifftshift(spec_complex, dim=1)

    recon_128 = torch.istft(
        spec_shifted,
        n_fft=opts.istft_nfft,
        hop_length=opts.istft_overlap,
        win_length=opts.istft_window,
        window=window_istft,
        onesided=False,
        return_complex=True,
        center=True,
    ).cpu().numpy()

    n_classes = 2 ** sf
    if recon_128.shape[-1] > n_classes:
        recon_128 = recon_128[:, :n_classes]
    elif recon_128.shape[-1] < n_classes:
        recon_128 = np.pad(recon_128, ((0, 0), (0, n_classes - recon_128.shape[-1])))

    # 6. Dechirping at Fs = BW (128 samples, standard downchirp)
    lora_gen = LoRa(sf, bw)
    ref_down = lora_gen.gen_symbol_fs(0, down=True, Fs=bw)
    dechirped = recon_128 * ref_down
    spectrum = np.fft.fft(dechirped, axis=-1)
    symbols = np.argmax(np.abs(spectrum), axis=-1)

    return [int(s) for s in symbols]
