"""
CALoRa preamble-location bridge for the dashboard — lets a run pick ANY SF7 checkpoint
in Server/models/calora/checkpoints/sf7/ori_checkpoints/, not just the one hardcoded in
main2.py's _CHECKPOINT_MAP.

Deliberately reimplements the ~10 lines of main2.detect_preamble_calora() that take an
explicit model, rather than monkeypatching main2._CHECKPOINT_MAP/_MODEL_CACHE: those are
shared mutable module-level state, and mutating them would let two concurrent dashboard
runs with different checkpoint selections race on the same dict, and would risk affecting
main2.py if it were ever run standalone in the same process. Only main2's PUBLIC names
(PreambleDetector, iq_to_network_input, best_interval_from_p, CALORA_THRESH) are reused.

`main2` is imported lazily, inside functions, not at module top — main2.py calls
_load_model(7) unconditionally at module scope (its own line, outside any __main__
guard), which eagerly loads a checkpoint the instant it's imported. Deferring the import
means starting the dashboard app costs nothing until a run actually requests CALoRa.
"""
import threading
from pathlib import Path

import numpy as np
import torch

_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "calora" / "checkpoints" / "sf7" / "ori_checkpoints"

# Best-effort labels from Server/models/calora/checkpoints/sf7/ori_checkpoints/readme.md,
# which only documents 4 concepts for 7 files. "sean"/"sf7"/"chirp_restorer" map cleanly
# by substring; "orii" ("default training with CFO timedom, STO specdom, per symbol aug")
# doesn't substring-match anything, so its filename mapping below is an educated guess,
# not a confirmed fact — worth a human check with whoever wrote the readme before this
# ships for thesis-quality comparisons. 3 files have no readme entry at all.
_LABELS = {
    "chirp_restorer_sf7.pth": ("chirp_restorer", "Denoiser component, not a preamble-detection checkpoint.", True),
    "preamble_best_finetuned-sean.pth": ("sean", "CFO time-domain / STO time-domain, per-packet augmentation.", True),
    "preamble_best_finetuned-sf7.pth": ("sf7", "Newer augmented weights.", True),
    "preamble_best_finetuned.pth": ("orii (best guess, unverified)",
                                    "CFO time-domain / STO spec-domain, per-symbol augmentation — "
                                    "filename-to-label mapping is a guess, not confirmed.", True),
    "best_finetuned_train_cfo.pth": (None, None, False),
    "best_finetuned_train_clean.pth": (None, None, False),
    "premamble_best_finetuned_cfo_mix.pth": (None, None, False),
}
_CURRENT_SERVER_DEFAULT = "premamble_best_finetuned_cfo_mix.pth"
_DENOISER_FILENAME = "chirp_restorer_sf7.pth"  # never a detection-checkpoint choice

_lock = threading.Lock()
_model_cache: dict = {}
_probe_cache: dict = {}


def _instantiate():
    """SF7-only instantiation, matching main2.py's own PreambleDetector(...) call exactly
    (use_se = (sf>=8) -> False for sf=7)."""
    import main2  # lazy — see module docstring
    return main2.PreambleDetector(C=64, Ls=(262, 264, 266), use_se_tcn=False)


def _probe(filename: str) -> dict:
    """Load a checkpoint once and report load_state_dict's IncompatibleKeys, so a
    checkpoint that doesn't actually match this architecture (different Ls/use_se_tcn at
    training time) is caught here instead of silently producing garbage — main2.py's own
    loader uses strict=False, which accepts a partial/mismatched load without erroring."""
    if filename in _probe_cache:
        return _probe_cache[filename]

    path = _CHECKPOINT_DIR / filename
    result = {"loadable": False, "missing_keys": [], "unexpected_keys": []}
    try:
        model = _instantiate()
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in state:
                    state = state[key]
                    break
        res = model.load_state_dict(state, strict=False)
        result["loadable"] = True
        result["missing_keys"] = list(res.missing_keys)
        result["unexpected_keys"] = list(res.unexpected_keys)
    except Exception as exc:
        result["error"] = str(exc)

    _probe_cache[filename] = result
    return result


def list_calora_checkpoints() -> list[dict]:
    """One entry per file in the SF7 checkpoint directory. `selectable` is False for the
    denoiser (never a detection choice) and for any detector checkpoint whose probe
    reports missing/unexpected keys (probed once, lazily, on first call to this
    function — subsequent calls reuse the cached result)."""
    out = []
    for path in sorted(_CHECKPOINT_DIR.glob("*.pth")):
        filename = path.name
        label, description, documented = _LABELS.get(filename, (None, None, False))
        is_denoiser = (filename == _DENOISER_FILENAME)

        if is_denoiser:
            selectable = False
            load_warnings = None
        else:
            probe = _probe(filename)
            clean = probe["loadable"] and not probe["missing_keys"] and not probe["unexpected_keys"]
            selectable = clean
            load_warnings = None if clean else (
                probe.get("error") or
                f"missing_keys={probe['missing_keys']} unexpected_keys={probe['unexpected_keys']}"
            )

        out.append({
            "filename": filename,
            "label": label,
            "description": description,
            "documented": documented,
            "selectable": selectable,
            "is_current_server_default": (filename == _CURRENT_SERVER_DEFAULT),
            "load_warnings": load_warnings,
        })
    return out


def load_calora_detector(checkpoint_filename: str):
    """Lazy, cached for process lifetime, keyed by filename (SF7-only this version)."""
    if checkpoint_filename in _model_cache:
        return _model_cache[checkpoint_filename]

    with _lock:
        if checkpoint_filename in _model_cache:
            return _model_cache[checkpoint_filename]

        import main2  # lazy — see module docstring
        model = _instantiate()
        state = torch.load(_CHECKPOINT_DIR / checkpoint_filename, map_location=main2.DEVICE)
        if isinstance(state, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state, strict=False)
        model.to(main2.DEVICE)
        model.eval()

        _model_cache[checkpoint_filename] = model
        return model


def detect_preamble_calora_with_model(opts, iq: np.ndarray, model):
    """Reimplements main2.detect_preamble_calora()'s body against an explicit model,
    instead of main2's own internal _load_model(opts.sf) lookup. Same return contract:
    (is_detected, t_start, t_end, mean_prob)."""
    import main2  # lazy — see module docstring

    try:
        spec = main2.iq_to_network_input(iq, opts.sf, opts.bw, opts.fs)
    except Exception as exc:
        print(f"  [CALoRa bridge] spectrogram failed: {exc}")
        return False, 0, 0, 0.0

    with torch.no_grad():
        _logits, p, _s = model(spec)

    p_np = p[0].cpu().numpy()
    Ls = tuple(int(l) for l in model.Ls.cpu().tolist())
    try:
        _L_hat, t_start, t_end = main2.best_interval_from_p(p_np, Ls)
    except Exception as exc:
        print(f"  [CALoRa bridge] best_interval failed: {exc}")
        return False, 0, 0, 0.0

    mean_prob = float(np.mean(p_np[t_start:t_end + 1]))
    is_detected = mean_prob > main2.CALORA_THRESH
    return is_detected, t_start, t_end, mean_prob
