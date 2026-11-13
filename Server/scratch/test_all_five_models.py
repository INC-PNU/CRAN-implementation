"""Unified test of all 5 symbol decoding models on a single frame."""
import sys, time
from pathlib import Path
ROOT = Path(r"c:\Users\priba\Sean-2025\INC-LAB\CRAN-implementation\Server")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from utils.LoRa import LoRa
from utils.my_lora_utils import create_lora_preamble, create_lora_payload, add_cfo, calculate_symbol_alliqfile_with_down_sampling
from utils.helper_function import detect_cfo_sto
from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from utils.nelora_inference import _load_models as load_nelora_models
from models.calora.calora_model.model_calora import CALoRa
from utils import loraseek_inference, gloriphy_inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_single_frame_benchmark(sf=7, bw=125000, fs=1000000, n_symbols=10, snr_db=10, apply_cfo=False, cfo_hz=500.0, apply_sto=False, sto_samples=20, seed=0):
    rng = np.random.default_rng(seed)
    n_classes = 2 ** sf
    fps = int(n_classes * (fs / bw))
    gt_symbols = rng.integers(0, n_classes, size=n_symbols).tolist()

    opts = type("", (), {})()
    opts.sf = sf; opts.bw = bw; opts.fs = fs; opts.n_classes = n_classes; opts.no_of_preamble = 8; opts.sync_sym = [8, 8]

    preamble = create_lora_preamble(opts, LoRa)
    payload = create_lora_payload(opts, LoRa, gt_symbols)
    sig = np.concatenate([preamble, payload]).astype(np.complex64)

    # Apply impairments
    cfo_applied = cfo_hz if apply_cfo else 0.0
    sto_applied = sto_samples if apply_sto else 0
    if cfo_applied != 0.0:
        sig = add_cfo(opts, sig, cfo_applied)
    if sto_applied > 0:
        sig = sig[sto_applied:]

    lora = LoRa(sf, bw)
    iq = lora.awgn_iq_with_seed(sig, snr_db, seed=seed)

    # If impairments present, estimate & correct via preamble
    if apply_cfo or apply_sto:
        index_payload, cfo_est, sto_est, _, _ = detect_cfo_sto(opts, LoRa, iq, return_details=True)
        payload_start = int(index_payload * fps) + int(sto_est)
        t_axis = np.arange(len(iq)) / fs
        iq_cfo = iq * np.exp(-1j * 2 * np.pi * cfo_est * t_axis)
        payload_corrected = iq_cfo[payload_start:payload_start + n_symbols * fps]
        pre_start = max(0, payload_start - int(12.25 * fps))
        pre_end = pre_start + int(8 * fps)
        sfd_start = max(0, payload_start - int(2.25 * fps))
        preamble_corr = iq_cfo[pre_start:pre_end]
        sfd_corr = iq_cfo[sfd_start:payload_start]
    else:
        payload_start = int(12.25 * fps)
        payload_corrected = iq[payload_start:payload_start + n_symbols * fps]
        preamble_corr = iq[:int(8 * fps)]
        sfd_corr = iq[int(10 * fps):int(12.25 * fps)]

    results = {}

    # 1. Conventional Dechirping
    t0 = time.perf_counter()
    conv_preds = calculate_symbol_alliqfile_with_down_sampling(payload_corrected, sf, bw, fs, show=False)[:n_symbols]
    t_conv = (time.perf_counter() - t0) * 1000
    results["conventional"] = {
        "name": "Conventional Dechirping", "available": True,
        "symbols": [int(s) for s in conv_preds],
        "accuracy_pct": round(float(np.mean(np.array(conv_preds) == np.array(gt_symbols))) * 100, 2),
        "execution_time_ms": round(t_conv, 2)
    }

    # 2. NeLoRa
    t0 = time.perf_counter()
    if sf == 7:
        mask_net, nelora_clf = load_nelora_models("100000")
        chunks = iq_to_symbol_chunks(payload_corrected, sf, bw, fs, num_symbols=n_symbols, dechirp=False)
        with torch.no_grad():
            denoised = mask_net(chunks.to(DEVICE))
            logits = nelora_clf(denoised)
            raw = logits.argmax(dim=1).cpu().numpy()
            nelora_preds = [int(inverse_symb(int(s), sf)) for s in raw]
        t_nelora = (time.perf_counter() - t0) * 1000
        results["nelora"] = {
            "name": "NeLoRa", "available": True, "symbols": nelora_preds,
            "accuracy_pct": round(float(np.mean(np.array(nelora_preds) == np.array(gt_symbols))) * 100, 2),
            "execution_time_ms": round(t_nelora, 2)
        }
    else:
        results["nelora"] = {"name": "NeLoRa", "available": False, "error": "NeLoRa is currently configured for SF7"}

    # 3. CALoRa
    t0 = time.perf_counter()
    calora_weights = ROOT / "models" / "calora" / "checkpoints" / f"sf{sf}" / "ori_checkpoints" / f"chirp_restorer_sf{sf}.pth"
    if sf == 7 and calora_weights.exists():
        calora_restorer = CALoRa().to(DEVICE)
        st = torch.load(calora_weights, map_location=DEVICE)
        if isinstance(st, dict) and "model" in st: st = st["model"]
        calora_restorer.load_state_dict(st, strict=False)
        calora_restorer.eval()
        chunks = iq_to_symbol_chunks(payload_corrected, sf, bw, fs, num_symbols=n_symbols, dechirp=False)
        with torch.no_grad():
            restored = calora_restorer(chunks.to(DEVICE))
            _, nelora_clf = load_nelora_models("100000")
            logits = nelora_clf(restored)
            raw = logits.argmax(dim=1).cpu().numpy()
            calora_preds = [int(inverse_symb(int(s), sf)) for s in raw]
        t_calora = (time.perf_counter() - t0) * 1000
        results["calora"] = {
            "name": "CALoRa", "available": True, "symbols": calora_preds,
            "accuracy_pct": round(float(np.mean(np.array(calora_preds) == np.array(gt_symbols))) * 100, 2),
            "execution_time_ms": round(t_calora, 2)
        }
    else:
        results["calora"] = {"name": "CALoRa", "available": False, "error": f"CALoRa checkpoint for SF{sf} not yet prepared"}

    # 4. LoRaSeek
    t0 = time.perf_counter()
    if sf == 7 and loraseek_inference.LORASEEK_AVAILABLE:
        ls_preds = loraseek_inference.demodulate_payload_loraseek(payload_corrected, sf, bw, fs, checkpoint_name="75000_LoRaSeek.pkl")
        t_ls = (time.perf_counter() - t0) * 1000
        results["loraseek"] = {
            "name": "LoRaSeek", "available": True, "symbols": [int(s) for s in ls_preds],
            "accuracy_pct": round(float(np.mean(np.array(ls_preds) == np.array(gt_symbols))) * 100, 2),
            "execution_time_ms": round(t_ls, 2)
        }
    else:
        results["loraseek"] = {"name": "LoRaSeek", "available": False, "error": f"LoRaSeek checkpoint for SF{sf} not available"}

    # 5. GLoRiPhy
    t0 = time.perf_counter()
    if gloriphy_inference.GLORIPHY_AVAILABLE and sf in gloriphy_inference.SUPPORTED_SFS:
        gl_ckpt = gloriphy_inference.default_checkpoint_for(sf)
        gl_preds = gloriphy_inference.demodulate_payload_gloriphy(
            payload_corrected, sf, bw, fs, checkpoint_id=gl_ckpt,
            preamble_iq=preamble_corr, sfd_iq=sfd_corr
        )
        t_gl = (time.perf_counter() - t0) * 1000
        results["gloriphy"] = {
            "name": "GLoRiPhy", "available": True, "symbols": [int(s) for s in gl_preds],
            "accuracy_pct": round(float(np.mean(np.array(gl_preds) == np.array(gt_symbols))) * 100, 2),
            "execution_time_ms": round(t_gl, 2)
        }
    else:
        results["gloriphy"] = {"name": "GLoRiPhy", "available": False, "error": f"GLoRiPhy checkpoint for SF{sf} not available"}

    return {"gt_symbols": gt_symbols, "results": results}

print("=== SF7 Test (clean) ===")
res7 = run_single_frame_benchmark(sf=7, snr_db=10)
print("GT:", res7["gt_symbols"])
for k, v in res7["results"].items():
    print(f"  {k:15s}: available={v['available']} acc={v.get('accuracy_pct')}% time={v.get('execution_time_ms')}ms")

print("\n=== SF8 Test (clean) ===")
res8 = run_single_frame_benchmark(sf=8, snr_db=10)
print("GT:", res8["gt_symbols"])
for k, v in res8["results"].items():
    print(f"  {k:15s}: available={v['available']} acc={v.get('accuracy_pct')}% time={v.get('execution_time_ms')}ms error={v.get('error')}")
