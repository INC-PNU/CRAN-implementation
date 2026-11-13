"""
NeLoRa demodulation wrapper — SF7 and SF8 support.
- SF7: checkpoints in Server/models/nelora/checkpoints/sf7/ori_checkpoints/
- SF8: checkpoints in Server/models/nelora/checkpoints/sf8/sf8_checkpoints/

NeLoRa is NOT a preamble locator like CALoRa. It is an end-to-end per-symbol classifier
(maskCNNModel denoises a one-symbol spectrogram chunk, classificationHybridModel
classifies the result to one of 2**sf logits) that replaces dechirp-FFT-argmax
demodulation — it is orthogonal to whichever technique located the preamble.

NELORA_VALIDATED gates whether this module may be used at all. Two things about its
expected input cannot be determined by reading model_NeLoRa.py alone (no config, no
training/eval script shipped with the checkpoints):
  1. whether it expects the RAW (undechirped) per-symbol STFT or a dechirped one, and
  2. whether its raw argmax needs inverse_symb() (Server/utils/nelora_utils.py:139-142,
     otherwise orphaned/unused) applied before it equals the transmitted symbol.
validate_nelora.py settles both empirically against known-symbol synthetic test vectors.
"""
import threading
from pathlib import Path

import numpy as np
import torch

from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from models.nelora.nelora_model.model_NeLoRa import maskCNNModel, classificationHybridModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute paths to checkpoint directories
_BASE_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "nelora" / "checkpoints"
_CHECKPOINT_DIR_SF7 = _BASE_CHECKPOINT_DIR / "sf7" / "ori_checkpoints"
_CHECKPOINT_DIR_SF8 = _BASE_CHECKPOINT_DIR / "sf8" / "sf8_checkpoints"

_SF7_CHECKPOINTS = {
    "100000": {
        "mask": "100000_maskCNN.pkl",
        "classifier": "100000_C_XtoY.pkl",
        "label": "SF7 · 100000 iterations (default)",
    }
}

_SF8_CHECKPOINTS = {
    "75000": {
        "mask": "75000_maskCNN.pkl",
        "classifier": "75000_C_XtoY.pkl",
        "label": "SF8 · 75000 iterations (recommended: best low-SNR)",
    },
    "80000": {
        "mask": "80000_maskCNN.pkl",
        "classifier": "80000_C_XtoY.pkl",
        "label": "SF8 · 80000 iterations (high accuracy)",
    },
    "85000": {
        "mask": "85000_maskCNN.pkl",
        "classifier": "85000_C_XtoY.pkl",
        "label": "SF8 · 85000 iterations",
    },
    "90000": {
        "mask": "90000_maskCNN.pkl",
        "classifier": "90000_C_XtoY.pkl",
        "label": "SF8 · 90000 iterations",
    },
    "95000": {
        "mask": "95000_maskCNN.pkl",
        "classifier": "95000_C_XtoY.pkl",
        "label": "SF8 · 95000 iterations",
    },
    "100000": {
        "mask": "100000_maskCNN.pkl",
        "classifier": "100000_C_XtoY.pkl",
        "label": "SF8 · 100000 iterations",
    },
    "105000": {
        "mask": "105000_maskCNN.pkl",
        "classifier": "105000_C_XtoY.pkl",
        "label": "SF8 · 105000 iterations",
    },
    "110000": {
        "mask": "110000_maskCNN.pkl",
        "classifier": "110000_C_XtoY.pkl",
        "label": "SF8 · 110000 iterations",
    },
    "115000": {
        "mask": "115000_maskCNN.pkl",
        "classifier": "115000_C_XtoY.pkl",
        "label": "SF8 · 115000 iterations",
    },
}

DEFAULT_CHECKPOINT_BY_SF = {7: "100000", 8: "75000"}
DEFAULT_CHECKPOINT = "100000"
SUPPORTED_SFS = (7, 8)

AVAILABLE_CHECKPOINTS = {
    # SF7
    "100000": "100000",
    # SF8
    "75000": "75000",
    "80000": "80000",
    "85000": "85000",
    "90000": "90000",
    "95000": "95000",
    "105000": "105000",
    "110000": "110000",
    "115000": "115000",
}

NELORA_SF7_AVAILABLE = (_CHECKPOINT_DIR_SF7 / "100000_maskCNN.pkl").exists()
NELORA_SF8_AVAILABLE = (_CHECKPOINT_DIR_SF8 / "75000_maskCNN.pkl").exists()
NELORA_AVAILABLE = NELORA_SF7_AVAILABLE or NELORA_SF8_AVAILABLE

# ── Validated via validate_nelora.py & SF8 empirical spike ────
# input='raw', inverse_symb=True scored 100% on clean and high SNR for both SF7 and SF8.
NELORA_VALIDATED = True
NELORA_INPUT_MODE = "raw"
NELORA_APPLY_INVERSE_SYMB = True


# Opts for maskCNNModel
class _MaskOptsSF7:
    x_image_channel = 2
    y_image_channel = 2
    freq_size = 128       # 2**7
    conv_dim_lstm = 1024  # 8 * freq_size
    lstm_dim = 400
    fc1_dim = 600


class _MaskOptsSF8:
    x_image_channel = 2
    y_image_channel = 2
    freq_size = 256       # 2**8
    conv_dim_lstm = 2048  # 8 * freq_size
    lstm_dim = 400
    fc1_dim = 600


# Backward compatibility alias
_MaskOpts = _MaskOptsSF7

_lock = threading.Lock()
_mask_cache: dict = {}
_classifier_cache: dict = {}


def _unwrap_state_dict(state):
    """Mirrors defensive checkpoint unwrapping — these .pkl files accept a plain
    state_dict or one nested under a common key."""
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in state:
                return state[key]
    return state


def list_nelora_checkpoints(sf: int | None = None) -> list[dict]:
    """Returns available NeLoRa checkpoint options for the dashboard."""
    checkpoints = []
    if sf is None or sf == 7:
        for cid, info in _SF7_CHECKPOINTS.items():
            checkpoints.append({
                "id": cid,
                "label": info.get("label", f"SF7 · {cid} iterations"),
                "sf": 7,
                "is_default": (cid == DEFAULT_CHECKPOINT_BY_SF[7]),
            })
    if sf is None or sf == 8:
        for cid, info in _SF8_CHECKPOINTS.items():
            checkpoints.append({
                "id": cid,
                "label": info.get("label", f"SF8 · {cid} iterations"),
                "sf": 8,
                "is_default": (cid == DEFAULT_CHECKPOINT_BY_SF[8]),
            })
    return checkpoints


def _load_models(checkpoint_id: str | None = None, sf: int = 7):
    """Lazy, cached for process lifetime, keyed by (sf, checkpoint_id)."""
    if sf not in (7, 8):
        raise ValueError(f"NeLoRa only supports SF7 and SF8, got sf={sf}")

    if not checkpoint_id:
        checkpoint_id = DEFAULT_CHECKPOINT_BY_SF.get(sf, "100000")

    cache_key = (sf, checkpoint_id)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key], _classifier_cache[cache_key]

    with _lock:
        if cache_key in _mask_cache:
            return _mask_cache[cache_key], _classifier_cache[cache_key]

        if sf == 7:
            if checkpoint_id not in _SF7_CHECKPOINTS:
                raise ValueError(f"Unknown SF7 NeLoRa checkpoint: {checkpoint_id}")
            files = _SF7_CHECKPOINTS[checkpoint_id]
            ckpt_dir = _CHECKPOINT_DIR_SF7
            mask_opts = _MaskOptsSF7()
            classifier = classificationHybridModel(conv_dim_in=2, conv_dim_out=128, conv_dim_lstm=1024)
        else:  # sf == 8
            if checkpoint_id not in _SF8_CHECKPOINTS:
                raise ValueError(f"Unknown SF8 NeLoRa checkpoint: {checkpoint_id}")
            files = _SF8_CHECKPOINTS[checkpoint_id]
            ckpt_dir = _CHECKPOINT_DIR_SF8
            mask_opts = _MaskOptsSF8()
            classifier = classificationHybridModel(conv_dim_in=2, conv_dim_out=256, conv_dim_lstm=2048)

        mask_model = maskCNNModel(mask_opts)
        mask_state = torch.load(ckpt_dir / files["mask"], map_location=DEVICE)
        mask_model.load_state_dict(_unwrap_state_dict(mask_state), strict=False)
        mask_model.to(DEVICE).eval()

        classifier_state = torch.load(ckpt_dir / files["classifier"], map_location=DEVICE)
        classifier.load_state_dict(_unwrap_state_dict(classifier_state), strict=False)
        classifier.to(DEVICE).eval()

        _mask_cache[cache_key] = mask_model
        _classifier_cache[cache_key] = classifier
        return mask_model, classifier


def _classify_chunks(chunks: torch.Tensor, checkpoint_id: str | None = None, sf: int = 7) -> np.ndarray:
    """chunks: (N, 2, freq_size, cols_per_symbol) -> (N,) int array of raw argmax indices,
    BEFORE any inverse_symb correction."""
    mask_model, classifier = _load_models(checkpoint_id, sf=sf)
    with torch.no_grad():
        denoised = mask_model(chunks.to(DEVICE))
        logits = classifier(denoised)
    return logits.argmax(dim=1).cpu().numpy()


def demodulate_payload_nelora(payload_iq: np.ndarray, sf: int, bw: int, fs: int,
                              checkpoint_id: str | None = None) -> list[int]:
    """Same call contract as calculate_symbol_alliqfile_with_down_sampling(data, sf, bw,
    fs, show=False) so the orchestrator can swap them by config alone."""
    if not NELORA_VALIDATED:
        raise RuntimeError(
            "NeLoRa has not been validated yet — run dashboard/validate_nelora.py first "
            "and set NELORA_VALIDATED/NELORA_INPUT_MODE/NELORA_APPLY_INVERSE_SYMB from its "
            "result before calling demodulate_payload_nelora().")
    if sf not in (7, 8):
        raise ValueError(f"NeLoRa supports SF7 and SF8, got sf={sf}")

    n_classes = 2 ** sf
    nsamp = n_classes * (fs // bw)
    num_symbols = int(round(len(payload_iq) / nsamp))

    chunks = iq_to_symbol_chunks(payload_iq, sf, bw, fs, num_symbols=num_symbols,
                                 dechirp=(NELORA_INPUT_MODE == "dechirped"))
    raw = _classify_chunks(chunks, checkpoint_id=checkpoint_id, sf=sf)

    if NELORA_APPLY_INVERSE_SYMB:
        return [int(inverse_symb(int(s), sf)) for s in raw]
    return [int(s) for s in raw]
