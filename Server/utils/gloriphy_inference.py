"""
GLoRiPHY demodulation wrapper (Sabharwal et al., "Enhancing LoRa Reception with
Generative Models: Channel-Aware Denoising of LoRaPHY Signals", SenSys '24).

Source studied: FOR_STUDY_PURPOSE/GloriPhy/GLoRiPHY_source. Model code is vendored into
Server/models/gloriphy/gloriphy_model/, checkpoints live under
Server/models/gloriphy/checkpoints/sf<N>/<variant_dir>/best_conformer_jgj.pkl.

Two GLoRiPHY variants are supported, chosen by the checkpoint's directory name:

  * awgn_filter*   -> ConformerGen_Core  ("AWGN filter", trained with --train_denoiseGenCore)
       Input : one payload symbol's STFT.
  * denoisegen*    -> ConformerGen2      (full channel-aware GLoRiPHY, --train_denoiseGen)
       Input : the received preamble (8 up-chirps) + SFD (2.25 down-chirps) -> channel
               estimate, plus one payload symbol's STFT. The frozen AWGN filter is part
               of the checkpoint (model.conformer_symbol).

Per-symbol pipeline (identical to GLoRiPHY_source/train.py testing loops):
  1. Slice the payload into 2^SF * (fs/bw) sample chunks (fs/bw must be 8, as trained).
  2. Hamming STFT: n_fft = 8*2^SF, win = 2^SF/2, hop = win/2  -> [B, 8*2^SF, 33].
  3. Keep the central 2^SF bins, max-normalise, real/imag channels -> [B, 2, 2^SF, 33].
  4. Conformer denoises the spectrogram.
  5. ISTFT -> dechirp -> 16x zero-padded FFT peak -> symbol index (`pred_ind`, the
     quantity GLoRiPHY's own accuracy metric uses; the auxiliary classification head is
     NOT used — it is a training regulariser only and is far less accurate).

Symbol convention: GLoRiPHY's label for mod.gen_symbol(N - k) is k, and
utils.LoRa.gen_symbol_fs(k) is mod.gen_symbol(N - k) up to a constant phase, so the
returned indices are directly comparable with the dashboard ground truth (verified at
SF8: 100% accuracy on clean symbols).

There is NO SF7 checkpoint in the GLoRiPHY release. Training one (GPU strongly advised):
    python GLoRiPHY_source/main.py --root_path GLoRiPHY_source --train_denoiseGenCore \
        --sf 7 --transformer_layers 2 --transformer_encoder_dim 256 --num_epochs 700 \
        --attention_dropout 0.15 --ff_dropout 0.1 --conv_dropout 0.1 --inp_dropout 0.1 \
        --checkpoint_dir checkpoints/AWGN_Filter_new/simulation/SF7 --sched_factor 0.4
then copy best_conformer_jgj.pkl to Server/models/gloriphy/checkpoints/sf7/awgn_filter/.
It is picked up automatically (discovery is by directory scan).
"""
import math
import re
import threading
import types
from pathlib import Path

import numpy as np
import torch

from models.gloriphy.gloriphy_model.model import ConformerGen_Core, ConformerGen2
from models.gloriphy.gloriphy_model.stft_utils import (
    GLoRiPHYDemod, spec_to_network_input, spec_to_network_input_complex, get_channel_estimate)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS_ROOT = Path(__file__).resolve().parents[1] / "models" / "gloriphy" / "checkpoints"
CHECKPOINT_FILENAME = "best_conformer_jgj.pkl"
REQUIRED_FS_BW_RATIO = 8  # every released checkpoint was trained at fs = 8 * bw

VARIANT_AWGN = "awgn_filter"
VARIANT_FULL = "full"

_VARIANT_LABELS = {
    VARIANT_AWGN: "AWGN filter",
    VARIANT_FULL: "full, channel-aware",
}


def _variant_for_dir(dirname: str) -> str | None:
    d = dirname.lower()
    if d.startswith("awgn_filter"):
        return VARIANT_AWGN
    if d.startswith("denoisegen"):
        return VARIANT_FULL
    return None


def _discover() -> list[dict]:
    out = []
    if not CHECKPOINTS_ROOT.exists():
        return out
    for sf_dir in sorted(CHECKPOINTS_ROOT.iterdir()):
        m = re.fullmatch(r"sf(\d+)", sf_dir.name.lower())
        if not (sf_dir.is_dir() and m):
            continue
        sf = int(m.group(1))
        for vdir in sorted(sf_dir.iterdir()):
            variant = _variant_for_dir(vdir.name)
            ckpt = vdir / CHECKPOINT_FILENAME
            if variant is None or not ckpt.exists():
                continue
            prefix = "awgn_filter" if variant == VARIANT_AWGN else "denoisegen"
            suffix = vdir.name[len(prefix):].strip("_")
            tag = f" — {suffix.replace('_', ' ')}" if suffix else ""
            out.append({
                "id": f"sf{sf}/{vdir.name}",
                "sf": sf,
                "variant": variant,
                "needs_preamble": variant == VARIANT_FULL,
                "label": f"SF{sf} · {_VARIANT_LABELS[variant]}{tag}",
                "path": str(ckpt),
            })
    # Default per SF: the official AWGN filter (dir named exactly "awgn_filter") if
    # present, else the first AWGN filter, else the first checkpoint of that SF.
    by_sf: dict = {}
    for c in out:
        by_sf.setdefault(c["sf"], []).append(c)
    for sf, items in by_sf.items():
        pick = (next((c for c in items if c["id"].endswith("/awgn_filter")), None)
                or next((c for c in items if c["variant"] == VARIANT_AWGN), None)
                or items[0])
        for c in items:
            c["is_default"] = c is pick
    return out


_CHECKPOINTS = _discover()
AVAILABLE_CHECKPOINTS = {c["id"]: c for c in _CHECKPOINTS}
SUPPORTED_SFS = sorted({c["sf"] for c in _CHECKPOINTS})
GLORIPHY_AVAILABLE = bool(_CHECKPOINTS)


def list_gloriphy_checkpoints() -> list[dict]:
    """Checkpoint options for the dashboard (path omitted — server-side detail)."""
    return [{k: v for k, v in c.items() if k != "path"} for c in _CHECKPOINTS]


def default_checkpoint_for(sf: int) -> str | None:
    return next((c["id"] for c in _CHECKPOINTS if c["sf"] == sf and c.get("is_default")), None)


def checkpoint_info(checkpoint_id: str | None, sf: int) -> dict:
    """Resolves (and validates) a checkpoint id for a given SF. Raises ValueError."""
    ckpt_id = checkpoint_id or default_checkpoint_for(sf)
    if ckpt_id is None:
        raise ValueError(f"no GLoRiPHY checkpoint available for SF{sf} "
                         f"(available SFs: {SUPPORTED_SFS or 'none'})")
    info = AVAILABLE_CHECKPOINTS.get(ckpt_id)
    if info is None:
        raise ValueError(f"unknown gloriphy_checkpoint: {ckpt_id!r}")
    if info["sf"] != sf:
        raise ValueError(f"gloriphy_checkpoint {ckpt_id!r} is for SF{info['sf']}, but the run is SF{sf}")
    return info


# ── Model construction ──────────────────────────────────────────────────────────────

def _build_opts(sf: int) -> types.SimpleNamespace:
    """Mirrors GLoRiPHY_source/main.py's derived STFT parameters."""
    demod = GLoRiPHYDemod(sf, fs_bw_ratio=REQUIRED_FS_BW_RATIO, fft_factor=16)
    opts = types.SimpleNamespace()
    opts.sf = sf
    opts.demod = demod
    opts.stft_nfft = demod.d_num_samples
    opts.stft_window = demod.d_num_symbols // 2
    opts.hop_length = opts.stft_window // 2
    opts.freq_size = demod.d_num_symbols
    opts.n_classes = demod.d_num_symbols
    opts.normalization = True
    opts.x_image_channel = 2
    opts.use_symbol_index = False
    opts.device = DEVICE
    opts.hamming_window = torch.hamming_window(opts.stft_window).to(DEVICE)
    opts.downchirp = torch.tensor(demod.d_downchirp[:demod.d_num_samples],
                                  dtype=torch.complex64).unsqueeze(0).to(DEVICE)
    return opts


def _arch_from_state_dict(sd: dict, layer_prefix: str) -> tuple[int, int, int]:
    """(encoder_dim, num_layers, num_heads) read off the weights, so checkpoints trained
    with non-default --transformer_encoder_dim / --transformer_layers load correctly."""
    ln_key = f"{layer_prefix}.0.sequential.4.weight"           # ConformerBlock's final LayerNorm
    u_key = f"{layer_prefix}.0.sequential.1.module.attention.u_bias"
    encoder_dim = int(sd[ln_key].shape[0])
    num_heads = int(sd[u_key].shape[0])
    idx = {int(k[len(layer_prefix) + 1:].split(".")[0]) for k in sd if k.startswith(layer_prefix + ".")}
    return encoder_dim, max(idx) + 1, num_heads


def _conformer_for(sf: int, encoder_dim: int) -> str:
    # train.py::create_model: emb 512 -> sf-1 encoder/decoder, 1024 -> sf-2.
    shift = int(round(math.log2(encoder_dim / 256))) if encoder_dim >= 256 else 0
    return f"sf_{sf - shift}"


_lock = threading.Lock()
_models: dict = {}


class _Loaded:
    def __init__(self, model, opts, variant):
        self.model = model
        self.opts = opts
        self.variant = variant


def _load(info: dict) -> _Loaded:
    key = info["id"]
    if key in _models:
        return _models[key]
    with _lock:
        if key in _models:
            return _models[key]
        sf = info["sf"]
        opts = _build_opts(sf)
        sd = torch.load(info["path"], map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for k in ("model", "model_state_dict", "state_dict"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break

        if info["variant"] == VARIANT_AWGN:
            dim, layers, heads = _arch_from_state_dict(sd, "encoder.layers")
            model = ConformerGen_Core(opts, conformer_for=_conformer_for(sf, dim), encoder_dim=dim,
                                      num_attention_heads=heads, num_encoder_layers=layers)
        else:
            dim, layers, heads = _arch_from_state_dict(sd, "conformer_mask.layers")
            model = ConformerGen2(opts, conformer_for=_conformer_for(sf, dim), encoder_dim=dim,
                                  num_attention_heads=heads, num_encoder_layers=layers)
            # Reference (ideal) preamble+SFD STFT used by get_channel_estimate. GLoRiPHY reads
            # it from packet_7_GT_7_.bin; that file's preamble/SFD are exactly
            # Utils.m::gen_packet's ideal chirps, which reference_preamble_sfd() reproduces.
            pre, sfd = opts.demod.reference_preamble_sfd()
            opts.preamble_conj = torch.conj(_preamble_sfd_stft(
                torch.from_numpy(pre).to(torch.complex64).unsqueeze(0),
                torch.from_numpy(sfd).to(torch.complex64).unsqueeze(0), opts))[0]

        model.load_state_dict(sd, strict=True)
        model.to(DEVICE).eval()
        loaded = _Loaded(model, opts, info["variant"])
        _models[key] = loaded
        return loaded


# ── Signal processing ───────────────────────────────────────────────────────────────

def _stft(x: torch.Tensor, opts) -> torch.Tensor:
    return torch.stft(x, n_fft=opts.stft_nfft, hop_length=opts.hop_length,
                      win_length=opts.stft_window, window=opts.hamming_window,
                      pad_mode="constant", return_complex=True)


def _preamble_sfd_stft(preamble: torch.Tensor, sfd: torch.Tensor, opts) -> torch.Tensor:
    p = spec_to_network_input_complex(_stft(preamble.to(DEVICE), opts), opts)
    s = spec_to_network_input_complex(_stft(sfd.to(DEVICE), opts), opts)
    return torch.cat((p, s), dim=-1)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def demodulate_payload_gloriphy(
    payload_iq: np.ndarray,
    sf: int,
    bw: int,
    fs: int,
    checkpoint_id: str | None = None,
    preamble_iq: np.ndarray | None = None,
    sfd_iq: np.ndarray | None = None,
) -> list[int]:
    """
    Demodulate a CFO/STO-corrected LoRa payload with GLoRiPHY. Same contract as
    calculate_symbol_alliqfile_with_down_sampling / demodulate_payload_loraseek.

    preamble_iq (8 symbols) and sfd_iq (2.25 symbols), taken from the same corrected
    signal, are required for the full (denoisegen) variant and ignored otherwise.
    """
    if fs != REQUIRED_FS_BW_RATIO * bw:
        raise ValueError(f"GLoRiPHY checkpoints require fs = {REQUIRED_FS_BW_RATIO} x bw "
                         f"(got fs={fs}, bw={bw})")
    info = checkpoint_info(checkpoint_id, sf)
    loaded = _load(info)
    opts = loaded.opts
    nsamp = opts.demod.d_num_samples

    num_symbols = int(round(len(payload_iq) / nsamp))
    if num_symbols == 0:
        return []
    chunks = np.stack([_fit(payload_iq[i * nsamp:(i + 1) * nsamp], nsamp) for i in range(num_symbols)])
    symbols_t = torch.from_numpy(chunks).to(device=DEVICE, dtype=torch.complex64)

    with torch.no_grad():
        net_in = spec_to_network_input(_stft(symbols_t, opts), opts)
        if loaded.variant == VARIANT_AWGN:
            _, _, pred_ind = loaded.model(net_in)
        else:
            if preamble_iq is None or sfd_iq is None:
                raise ValueError("the full GLoRiPHY variant needs preamble_iq and sfd_iq")
            pre = torch.from_numpy(_fit(preamble_iq, 8 * nsamp)).unsqueeze(0)
            sfd = torch.from_numpy(_fit(sfd_iq, int(2.25 * nsamp))).unsqueeze(0)
            chan = get_channel_estimate(_preamble_sfd_stft(pre, sfd, opts), opts)
            chan = chan.expand(num_symbols, -1, -1, -1).contiguous()
            _, _, pred_ind = loaded.model(chan, net_in)

    return [int(s) for s in pred_ind.long().cpu().numpy()]
