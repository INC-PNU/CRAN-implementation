"""
NeLoRa demodulation wrapper — SF7 only for this version (the only spreading factor with
a checkpoint pair on disk: Server/models/nelora/checkpoints/sf7/ori_checkpoints/).

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
Until it has been run and both constants below are set from its result, NELORA_VALIDATED
stays False and callers (dashboard/orchestrator.py, dashboard/routes.py) must refuse any
request for demod_method="nelora".
"""
import threading
from pathlib import Path

import numpy as np
import torch

from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from models.nelora.nelora_model.model_NeLoRa import maskCNNModel, classificationHybridModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute path, independent of CWD (matches main2.py's Path(__file__)-based convention) —
# this file lives in Server/utils/, so parents[1] is Server/.
_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "nelora" / "checkpoints" / "sf7" / "ori_checkpoints"
_SF7_CHECKPOINTS = {"100000": {"mask": "100000_maskCNN.pkl", "classifier": "100000_C_XtoY.pkl"}}

# ── Set by validate_nelora.py — see that script's docstring for how to (re-)run it ────
# Validated 2026-09-03 via `python -m dashboard.validate_nelora`: input='raw',
# inverse_symb=True scored 100.00% (exhaustive, all 128 SF7 symbols) at noiseless, 0dB,
# and -5dB SNR, while the other 3 combinations (raw+no-inverse, dechirped+either) sat at
# 0.78-1.56% — at or within noise of the 1/128=0.78% chance level. Unambiguous pass.
NELORA_VALIDATED = True
NELORA_INPUT_MODE = "raw"
NELORA_APPLY_INVERSE_SYMB = True

# SF7-only opts for maskCNNModel — verified against the actual checkpoint tensor shapes
# (fc2.weight, dense.weight, lstm.weight_ih_l0 all confirmed to match these exactly).
# SF8 would need freq_size=256, conv_dim_lstm=2048 for both models — deferred, not this version.
class _MaskOpts:
    x_image_channel = 2
    y_image_channel = 2
    freq_size = 128       # 2**7
    conv_dim_lstm = 1024  # 8 * freq_size
    lstm_dim = 400
    fc1_dim = 600


_lock = threading.Lock()
_mask_cache: dict = {}
_classifier_cache: dict = {}


def _unwrap_state_dict(state):
    """Mirrors main2.py's defensive checkpoint unwrapping — these .pkl files have no
    documented format, so accept a plain state_dict or one nested under a common key."""
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in state:
                return state[key]
    return state


def list_nelora_checkpoints(sf: int = 7) -> list[str]:
    if sf != 7:
        return []
    return list(_SF7_CHECKPOINTS.keys())


def _load_models(checkpoint_id: str):
    """Lazy, cached for process lifetime, keyed by checkpoint_id (SF7-only this version —
    add an sf axis to the cache key if SF8 support is ever added)."""
    if checkpoint_id in _mask_cache:
        return _mask_cache[checkpoint_id], _classifier_cache[checkpoint_id]

    with _lock:
        if checkpoint_id in _mask_cache:
            return _mask_cache[checkpoint_id], _classifier_cache[checkpoint_id]

        files = _SF7_CHECKPOINTS[checkpoint_id]

        mask_model = maskCNNModel(_MaskOpts())
        mask_state = torch.load(_CHECKPOINT_DIR / files["mask"], map_location=DEVICE)
        mask_model.load_state_dict(_unwrap_state_dict(mask_state), strict=False)
        mask_model.to(DEVICE).eval()

        classifier = classificationHybridModel()  # SF7 defaults match the checkpoint exactly
        classifier_state = torch.load(_CHECKPOINT_DIR / files["classifier"], map_location=DEVICE)
        classifier.load_state_dict(_unwrap_state_dict(classifier_state), strict=False)
        classifier.to(DEVICE).eval()

        _mask_cache[checkpoint_id] = mask_model
        _classifier_cache[checkpoint_id] = classifier
        return mask_model, classifier


def _classify_chunks(chunks: torch.Tensor, checkpoint_id: str) -> np.ndarray:
    """chunks: (N, 2, freq_size, cols_per_symbol) -> (N,) int array of raw argmax indices,
    BEFORE any inverse_symb correction (validate_nelora.py needs the raw form to test
    both hypotheses; demodulate_payload_nelora applies the resolved correction itself)."""
    mask_model, classifier = _load_models(checkpoint_id)
    with torch.no_grad():
        denoised = mask_model(chunks.to(DEVICE))
        logits = classifier(denoised)
    return logits.argmax(dim=1).cpu().numpy()


def demodulate_payload_nelora(payload_iq: np.ndarray, sf: int, bw: int, fs: int,
                              checkpoint_id: str = "100000") -> list[int]:
    """Same call contract as calculate_symbol_alliqfile_with_down_sampling(data, sf, bw,
    fs, show=False) so the orchestrator can swap them by config alone."""
    if not NELORA_VALIDATED:
        raise RuntimeError(
            "NeLoRa has not been validated yet — run dashboard/validate_nelora.py first "
            "and set NELORA_VALIDATED/NELORA_INPUT_MODE/NELORA_APPLY_INVERSE_SYMB from its "
            "result before calling demodulate_payload_nelora().")
    if sf != 7:
        raise ValueError(f"NeLoRa is SF7-only in this version, got sf={sf}")

    n_classes = 2 ** sf
    nsamp = n_classes * (fs // bw)
    num_symbols = int(round(len(payload_iq) / nsamp))

    chunks = iq_to_symbol_chunks(payload_iq, sf, bw, fs, num_symbols=num_symbols,
                                 dechirp=(NELORA_INPUT_MODE == "dechirped"))
    raw = _classify_chunks(chunks, checkpoint_id)

    if NELORA_APPLY_INVERSE_SYMB:
        return [int(inverse_symb(int(s), sf)) for s in raw]
    return [int(s) for s in raw]
