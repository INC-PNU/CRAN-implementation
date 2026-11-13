"""
LoRaSeek demodulation wrapper supporting both SF7 and SF8:
  - SF7: trained with checkpoints located at Server/models/loraseek/checkpoints/sf7/v9_checkpoints/
         (model: models.loraseek.loraseek_model.model_LoRaSeek.LoRaSeekNet)
  - SF8: trained with checkpoints located at Server/models/loraseek/checkpoints/sf8/v81_checkpoints/
         (model: models.loraseek.loraseek_model.model_LoRaSeeksf8.LoRaSeekNet)

LoRaSeek operates as a neural denoiser in the STFT frequency domain:
  1. Segments payload IQ into one-symbol chunks (1024 samples for SF7, 2048 for SF8 @ 1MHz / 125kHz BW).
  2. Computes Hann-windowed STFT for each symbol chunk.
  3. Formats normalized central frequency bins into tensor [B, 2, 2^SF, 32].
  4. Runs forward pass through LoRaSeekNet (Transformer + Dual Attention U-Net).
  5. Reconstructs denoised time-domain complex IQ via Nyquist-rate ISTFT.
  6. Standard dechirping + FFT peak detection on the reconstructed clean signal.
"""
import threading
from pathlib import Path

import numpy as np
import torch

from utils.LoRa import LoRa
from models.loraseek.loraseek_model.model_LoRaSeek import LoRaSeekNet as LoRaSeekNetSF7
from models.loraseek.loraseek_model.model_LoRaSeeksf8 import LoRaSeekNet as LoRaSeekNetSF8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CHECKPOINTS_DIR_SF7 = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "loraseek"
    / "checkpoints"
    / "sf7"
    / "v9_checkpoints"
)

_CHECKPOINTS_DIR_SF8 = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "loraseek"
    / "checkpoints"
    / "sf8"
    / "v81_checkpoints"
)

SUPPORTED_SFS = [7, 8]

DEFAULT_CHECKPOINTS = {
    7: "90000_LoRaSeek.pkl",
    8: "90000_LoRaSeek.pkl",
}
DEFAULT_CHECKPOINT = "90000_LoRaSeek.pkl"

AVAILABLE_CHECKPOINTS = {
    "90000_LoRaSeek.pkl": "90000_LoRaSeek.pkl",
    "75000_LoRaSeek.pkl": "75000_LoRaSeek.pkl",
    "90000": "90000_LoRaSeek.pkl",
    "75000": "75000_LoRaSeek.pkl",
}

LORASEEK_SF7_AVAILABLE = (_CHECKPOINTS_DIR_SF7 / "90000_LoRaSeek.pkl").exists()
LORASEEK_SF8_AVAILABLE = (_CHECKPOINTS_DIR_SF8 / "90000_LoRaSeek.pkl").exists()
LORASEEK_AVAILABLE = LORASEEK_SF7_AVAILABLE or LORASEEK_SF8_AVAILABLE


def list_loraseek_checkpoints(sf: int | None = None) -> list[dict]:
    """Returns available LoRaSeek checkpoint options for the dashboard."""
    checkpoints = []

    if sf is None or sf == 7:
        checkpoints.extend([
            {
                "filename": "90000_LoRaSeek.pkl",
                "label": "SF7 · 90000 iterations (v9 default)",
                "sf": 7,
                "is_default": (sf == 7 or sf is None),
                "exists": (_CHECKPOINTS_DIR_SF7 / "90000_LoRaSeek.pkl").exists(),
            },
            {
                "filename": "75000_LoRaSeek.pkl",
                "label": "SF7 · 75000 iterations (v9 best model)",
                "sf": 7,
                "is_default": False,
                "exists": (_CHECKPOINTS_DIR_SF7 / "75000_LoRaSeek.pkl").exists(),
            },
        ])

    if sf is None or sf == 8:
        checkpoints.extend([
            {
                "filename": "90000_LoRaSeek.pkl",
                "label": "SF8 · 90000 iterations (v81 default)",
                "sf": 8,
                "is_default": (sf == 8),
                "exists": (_CHECKPOINTS_DIR_SF8 / "90000_LoRaSeek.pkl").exists(),
            },
        ])

    return checkpoints


class _LoRaSeekOptsSF7:
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


class _LoRaSeekOptsSF8:
    x_image_channel = 2
    y_image_channel = 2
    base_channel = 16
    channel_attention_reduction = 8
    num_of_transformers = 4
    num_of_heads = 2
    conv_kernel_size = 3
    conv_padding_size = 1
    sf = 8
    bw = 125_000
    fs = 1_000_000
    n_classes = 256
    freq_size = 256
    t_ = 33
    normalization = 1
    stft_nfft = 2048
    stft_window = 128
    stft_overlap = 64
    istft_nfft = 256
    istft_window = 32
    istft_overlap = 8


_lock = threading.Lock()
_loraseek_models: dict = {}


def _resolve_checkpoint_filename(checkpoint_name: str | None, sf: int = 7) -> str:
    if not checkpoint_name:
        return DEFAULT_CHECKPOINTS.get(sf, DEFAULT_CHECKPOINT)
    return AVAILABLE_CHECKPOINTS.get(checkpoint_name, checkpoint_name)


def _load_loraseek_model(checkpoint_name: str | None = None, sf: int = 7):
    """Lazy, thread-safe model loading cached by (sf, checkpoint_name)."""
    canonical_name = _resolve_checkpoint_filename(checkpoint_name, sf=sf)
    cache_key = (sf, canonical_name)

    if cache_key in _loraseek_models:
        return _loraseek_models[cache_key]

    with _lock:
        if cache_key in _loraseek_models:
            return _loraseek_models[cache_key]

        if sf == 7:
            ckpt_path = _CHECKPOINTS_DIR_SF7 / canonical_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"LoRaSeek SF7 checkpoint not found: {ckpt_path}")
            opts = _LoRaSeekOptsSF7()
            model = LoRaSeekNetSF7(opts)
        elif sf == 8:
            ckpt_path = _CHECKPOINTS_DIR_SF8 / canonical_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"LoRaSeek SF8 checkpoint not found: {ckpt_path}")
            opts = _LoRaSeekOptsSF8()
            model = LoRaSeekNetSF8(opts)
        else:
            raise ValueError(f"LoRaSeek supports SF7 and SF8 only, got sf={sf}")

        weights = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(weights, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in weights:
                    weights = weights[key]
                    break
        model.load_state_dict(weights, strict=True)
        model.to(DEVICE).eval()
        _loraseek_models[cache_key] = model
        return model


def demodulate_payload_loraseek(
    payload_iq: np.ndarray,
    sf: int,
    bw: int,
    fs: int,
    checkpoint_name: str | None = None,
) -> list[int]:
    """
    Demodulate LoRa payload IQ samples using LoRaSeek neural denoising followed by
    standard dechirping.
    Supports SF7 (v9 checkpoints) and SF8 (v81 checkpoints).
    """
    if sf not in SUPPORTED_SFS:
        raise ValueError(f"LoRaSeek supports SF7 and SF8, got sf={sf}")

    if sf == 7 and not LORASEEK_SF7_AVAILABLE:
        raise RuntimeError("LoRaSeek SF7 model checkpoint is not available on disk.")
    if sf == 8 and not LORASEEK_SF8_AVAILABLE:
        raise RuntimeError("LoRaSeek SF8 model checkpoint is not available on disk.")

    model = _load_loraseek_model(checkpoint_name, sf=sf)
    opts = _LoRaSeekOptsSF8() if sf == 8 else _LoRaSeekOptsSF7()

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

    # Format into [B, 2, freq_size, 32] (Float32)
    net_in = torch.view_as_real(y_norm).permute(0, 3, 1, 2).to(dtype=torch.float32)
    net_in_32 = net_in[:, :, :, :32]

    # 4. Neural denoising forward pass
    with torch.no_grad():
        denoised_net = model(net_in_32)

    # 5. Inverse STFT at Nyquist rate (istft_nfft=2^sf, istft_window, istft_overlap)
    # The LoRaSeek model denoises the cropped central 2^sf-bin spectrum.
    # Reconstructing directly with istft_nfft=2^sf yields clean time-domain symbols
    # sampled at Fs = BW = 125 kHz (2^sf samples per symbol).
    spec_complex = torch.complex(denoised_net[:, 0], denoised_net[:, 1])
    window_istft = torch.hann_window(opts.istft_window, device=DEVICE)
    spec_shifted = torch.fft.ifftshift(spec_complex, dim=1)

    recon = torch.istft(
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
    if recon.shape[-1] > n_classes:
        recon = recon[:, :n_classes]
    elif recon.shape[-1] < n_classes:
        recon = np.pad(recon, ((0, 0), (0, n_classes - recon.shape[-1])))

    # 6. Dechirping at Fs = BW (2^sf samples, standard downchirp)
    lora_gen = LoRa(sf, bw)
    ref_down = lora_gen.gen_symbol_fs(0, down=True, Fs=bw)
    dechirped = recon * ref_down
    spectrum = np.fft.fft(dechirped, axis=-1)
    symbols = np.argmax(np.abs(spectrum), axis=-1)

    return [int(s) for s in symbols]
