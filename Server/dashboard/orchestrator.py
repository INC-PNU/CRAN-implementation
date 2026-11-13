"""
In-process sweep orchestration. Packets are synthesized and estimated directly —
no HTTP/base64 round-trip through main.py/main2.py/main3.py's Flask servers. This is
what lets a single unified detect_cfo_sto call (Server/utils/helper_function.py) drive
BOTH preamble-detection methods, with the SAME toggle able to be set identically on both
for a true single-variable comparison — but the DEFAULT value of each toggle must still
match detect_cfo_sto's own per-method sentinel table below, not one fixed value applied
to both methods.

That distinction matters concretely: the classical/no-hint down-chirp scan can legitimately
land on either of the preamble's two down-chirps (Server/utils/helper_function.py has an
explicit "Choose second downchirp, might have better signal" branch) -- a roughly 50/50,
SNR-independent coin flip driven by which one's noise-perturbed amplitude measures higher,
since the two are structurally identical. sync_payload_offset_adjust exists to disambiguate
this by decoding the sync symbols; its correct default is True without a hint (the original
no-hint code always ran this) and False with one (the CALoRa path's down-chirp search is a
deterministic pair-scoring method that always resolves to the FIRST down-chirp, so the
decode is unnecessary there and actively risks a wrong-but-plausible result if the sync
symbols themselves fail to decode at marginal SNR -- the same reasoning that led to removing
mandatory sync verification from the CALoRa path entirely). Defaulting this (and the other
toggles below) to one fixed value regardless of method silently corrupts whichever method
that value wasn't tuned for -- e.g. sync_payload_offset_adjust=False on the conventional
path leaves every "picked the second down-chirp" packet cropped a full symbol late, which
demod_result's off-by-one handling then quietly reports as 9 comparable symbols instead of
10, not as an outright failure.

SF7-only this version (matches: CALoRa has no SF8/9 checkpoints; NeLoRa validation and
opts table are SF7-only too).
"""
import threading
import time
import uuid

import numpy as np

from utils.LoRa import LoRa
from utils.helper_function import detect_cfo_sto, wrap_symmetric
from utils.my_lora_utils import (create_lora_preamble, create_lora_payload, add_cfo,
                                 calculate_symbol_alliqfile_with_down_sampling)
from utils import nelora_inference, loraseek_inference, gloriphy_inference
from dashboard import calora_bridge, persistence

MAX_CONCURRENT_RUNS = 3

# Two toggles (apply_cfo_correction/apply_sto_correction) are NOT method-dependent -- they
# gate whether the orchestrator applies the estimate downstream, unrelated to how the
# estimate itself was produced. Everything else here mirrors detect_cfo_sto's own sentinel
# resolution (helper_function.py's detect_cfo_sto docstring) exactly, so a run that never
# touches the "Advanced" toggles gets that function's proven-safe default for whichever
# preamble_detection_method it picked, not a one-size-fits-all guess.
CONVENTIONAL_DEFAULT_TOGGLES = {
    "apply_cfo_correction": True,
    "apply_sto_correction": True,
    "precorrect_cfo_frac": False,
    "use_fractional_sto": True,
    "verify_network_id": True,
    "sync_payload_offset_adjust": True,
    "sto_method": "v3",
    "downchirp_search_radius": 1,
}
CALORA_DEFAULT_TOGGLES = {
    "apply_cfo_correction": True,
    "apply_sto_correction": True,
    "precorrect_cfo_frac": True,
    "use_fractional_sto": True,
    "verify_network_id": False,
    "sync_payload_offset_adjust": False,
    "sto_method": "v3",
    "downchirp_search_radius": 1,
}


def default_toggles_for(preamble_detection_method: str) -> dict:
    return CALORA_DEFAULT_TOGGLES if preamble_detection_method == "calora" else CONVENTIONAL_DEFAULT_TOGGLES


# How far the located payload start may sit from ground truth and still count as located
# for both CALoRa and conventional paths (in frames).
PREAMBLE_LOCATION_TOL_FRAMES = 1.0
CALORA_LOCATION_TOL_COLS = 33  # Kept for backward compatibility if imported externally


class TooManyRunsError(Exception):
    pass


class ConfigError(ValueError):
    pass


def validate_config(client_config: dict, server_config: dict) -> None:
    """Raises ConfigError with a human-readable message on anything invalid. Called by
    dashboard/routes.py before starting a run, so a bad config becomes an HTTP 400
    instead of a mid-sweep crash."""
    sf = client_config.get("sf", 7)
    if sf != 7:
        raise ConfigError(f"SF{sf} is not enabled yet — keeping the benchmark strictly at SF7 "
                          "so CALoRa preamble detection and symbol decoders compare on the exact same dataset.")
    for key in ("cfo_hz_range", "sto_samp_range", "snr_db_range"):
        lo, hi = client_config.get(key, [0, 0])
        if lo > hi:
            raise ConfigError(f"{key}: lo ({lo}) must be <= hi ({hi})")
    if client_config.get("n_packets", 0) <= 0:
        raise ConfigError("n_packets must be positive")

    method = server_config.get("preamble_detection_method", "conventional")
    if method not in ("conventional", "calora"):
        raise ConfigError(f"unknown preamble_detection_method: {method!r}")
    if method == "calora":
        ckpt = server_config.get("calora_checkpoint")
        checkpoints = {c["filename"]: c for c in calora_bridge.list_calora_checkpoints()}
        if ckpt not in checkpoints:
            raise ConfigError(f"unknown calora_checkpoint: {ckpt!r}")
        if not checkpoints[ckpt]["selectable"]:
            raise ConfigError(f"calora_checkpoint {ckpt!r} failed its compatibility probe: "
                              f"{checkpoints[ckpt]['load_warnings']}")

    demod = server_config.get("symbol_decoding_method", "conventional")
    if demod not in ("conventional", "nelora", "loraseek", "gloriphy"):
        raise ConfigError(f"unknown symbol_decoding_method: {demod!r}")
    if demod == "nelora":
        if not nelora_inference.NELORA_VALIDATED:
            raise ConfigError("symbol_decoding_method='nelora' is not available: "
                              "NeLoRa has not passed dashboard/validate_nelora.py yet.")
    if demod == "loraseek":
        if not loraseek_inference.LORASEEK_AVAILABLE:
            raise ConfigError("symbol_decoding_method='loraseek' is not available: "
                              "LoRaSeek checkpoints not found on disk.")
        ckpt = server_config.get("loraseek_checkpoint")
        if ckpt and ckpt not in loraseek_inference.AVAILABLE_CHECKPOINTS:
            raise ConfigError(f"unknown loraseek_checkpoint: {ckpt!r}")
    if demod == "gloriphy":
        if not gloriphy_inference.GLORIPHY_AVAILABLE:
            raise ConfigError("symbol_decoding_method='gloriphy' is not available: "
                              "GLoRiPHY checkpoints not found on disk.")
        if sf not in gloriphy_inference.SUPPORTED_SFS:
            raise ConfigError(f"symbol_decoding_method='gloriphy' does not have a checkpoint for SF{sf}. "
                              f"Available SFs on disk: {gloriphy_inference.SUPPORTED_SFS}.")
        ckpt = server_config.get("gloriphy_checkpoint")
        if ckpt and ckpt not in gloriphy_inference.AVAILABLE_CHECKPOINTS:
            raise ConfigError(f"unknown gloriphy_checkpoint: {ckpt!r}")


def _score_payload_symbols(demodulated, ground_truth):
    """Ported verbatim from main3.py's score_payload_symbols."""
    n_gt = len(ground_truth)
    gt = np.array(ground_truth)
    demod = np.array(demodulated)

    if len(demod) == n_gt + 1:
        demod = demod[1:]
    elif len(demod) == n_gt - 1:
        gt = gt[1:]
    elif len(demod) != n_gt:
        return None

    n_compare = min(len(demod), len(gt))
    matches = int(np.sum(demod[:n_compare] == gt[:n_compare]))
    return n_compare, matches


def _new_bin():
    return {
        "n_total": 0, "preamble_missed": 0, "preamble_mislocated": 0,
        "cfo_correct": 0, "sto_correct": 0,
        "cfo_abs_err_sum": 0.0, "cfo_abs_err_count": 0,
        "sto_abs_err_sum": 0.0, "sto_abs_err_count": 0,
        "symbol_total": 0, "symbol_correct": 0, "symbol_total_sent": 0,
    }


def _publish_bins(acc: dict) -> list:
    """acc: {snr_int: bin_accumulator} -> the manifest/API bins[] shape (raw counts plus
    a running mean reduced from the sum/count accumulators — never the raw error list)."""
    out = []
    for snr in sorted(acc):
        b = acc[snr]
        out.append({
            "snr_db": snr,
            "n_total": b["n_total"],
            "preamble_missed": b["preamble_missed"],
            "preamble_mislocated": b["preamble_mislocated"],
            "cfo_correct": b["cfo_correct"],
            "sto_correct": b["sto_correct"],
            "cfo_mae_hz": (b["cfo_abs_err_sum"] / b["cfo_abs_err_count"]) if b["cfo_abs_err_count"] else None,
            "sto_mae_samples": (b["sto_abs_err_sum"] / b["sto_abs_err_count"]) if b["sto_abs_err_count"] else None,
            "symbol_total": b["symbol_total"],
            "symbol_correct": b["symbol_correct"],
            "symbol_total_sent": b["symbol_total_sent"],
        })
    return out


def _build_opts(client_config: dict):
    opts = type("", (), {})()
    opts.sf = client_config["sf"]
    opts.bw = client_config["bw"]
    opts.fs = client_config["fs"]
    opts.n_classes = 2 ** opts.sf
    opts.gateway_id = "dashboard"
    opts.no_of_preamble = client_config.get("no_of_preamble", 8)
    opts.sync_sym = client_config.get("sync_sym", [8, 8])
    return opts


def _run_one_packet(opts, client_config, server_config, rng, noise_seed, framePerSymbol):
    toggles = {**default_toggles_for(server_config["preamble_detection_method"]),
              **server_config.get("toggles", {})}
    payload_len = client_config.get("payload_len", 10)

    cfo_true = float(rng.uniform(*client_config["cfo_hz_range"]))
    lo, hi = client_config["sto_samp_range"]
    numb_offset = int(rng.integers(lo, hi + 1))
    snr_lo, snr_hi = client_config["snr_db_range"]
    snr = int(rng.uniform(snr_lo, snr_hi))
    payload_symbols_gt = rng.integers(0, opts.n_classes, size=payload_len).tolist()

    preamble = create_lora_preamble(opts, LoRa)
    payload = create_lora_payload(opts, LoRa, payload_symbols_gt)
    zonk = create_lora_payload(opts, LoRa, [999, 1222])  # out-of-range -> random noise, matches client3.py
    sig = np.concatenate([zonk, preamble, payload]).astype(np.complex64)
    sig = add_cfo(opts, sig, cfo_true)
    sig = sig[numb_offset:]
    lora_init = LoRa(opts.sf, opts.bw)
    iq = lora_init.awgn_iq_with_seed(sig, snr, seed=noise_seed)

    # Ground-truth payload start, in frames: 2 zonk + 8 up-chirps + 2 sync + 2.25 down,
    # shifted left by the STO crop. Derived from opts rather than hardcoded to 14.25 so
    # it stays correct if no_of_preamble/sync_sym change.
    gt_payload_frame = (2 + opts.no_of_preamble + len(opts.sync_sym) + 2.25
                        - numb_offset / framePerSymbol)
    sto_true = wrap_symmetric(-numb_offset, framePerSymbol)

    result = {
        "snr": snr,
        "payload_symbols_gt": payload_symbols_gt,
        "gt_payload_frame": float(gt_payload_frame),
        "gt_payload_sample": int(gt_payload_frame * framePerSymbol),
        "cfo_true": float(cfo_true),
        "sto_true": int(sto_true),
        "numb_offset": int(numb_offset),
        "cfo_int_hz": None,
        "cfo_frac_hz": None,
        "sto_int_samp": None,
        "sto_frac_samp": None,
        "cfo_est": None,
        "sto_est": None,
        "index_payload": None,
        "payload_start": None,
        "payload_start_frame": None,
    }

    details = None
    if server_config["preamble_detection_method"] == "calora":
        model = calora_bridge.load_calora_detector(server_config["calora_checkpoint"])
        is_detected, t_start, t_end, _mean_prob = calora_bridge.detect_preamble_calora_with_model(opts, iq, model)

        if not is_detected:
            result["outcome"] = "preamble_missed"
            return result

        win_len = opts.n_classes // 2
        hop = win_len // 2
        cols_per_symbol = framePerSymbol // hop + 1
        sync_symbol_index = int(round((t_end + 1) / cols_per_symbol))
        downchirp_hint_index = max(0, sync_symbol_index + 2)

        index_payload, cfo_est, sto_est, _found, details = detect_cfo_sto(
            opts, LoRa, iq,
            downchirp_hint_symbol=downchirp_hint_index,
            precorrect_cfo_frac=toggles["precorrect_cfo_frac"],
            use_fractional_sto=toggles.get("use_fractional_sto", True),
            verify_network_id=toggles["verify_network_id"],
            sync_payload_offset_adjust=toggles["sync_payload_offset_adjust"],
            sto_method=toggles["sto_method"],
            downchirp_search_radius=toggles["downchirp_search_radius"],
            return_details=True,
        )
        if index_payload in (-1, -2) or index_payload is None:
            result["outcome"] = "preamble_mislocated"
            result["index_payload"] = float(index_payload) if index_payload is not None else None
            return result
    else:
        index_payload, cfo_est, sto_est, _found, details = detect_cfo_sto(
            opts, LoRa, iq,
            precorrect_cfo_frac=toggles["precorrect_cfo_frac"],
            use_fractional_sto=toggles.get("use_fractional_sto", True),
            verify_network_id=toggles["verify_network_id"],
            sync_payload_offset_adjust=toggles["sync_payload_offset_adjust"],
            sto_method=toggles["sto_method"],
            return_details=True,
        )
        if index_payload == -1:
            result["outcome"] = "preamble_missed"
            result["index_payload"] = float(index_payload) if index_payload is not None else None
            return result
        if index_payload == -2 or index_payload is None:
            result["outcome"] = "preamble_mislocated"
            result["index_payload"] = float(index_payload) if index_payload is not None else None
            return result

    result["index_payload"] = float(index_payload)
    result["cfo_est"] = float(cfo_est)
    result["sto_est"] = float(sto_est)
    if details:
        result["cfo_int_hz"] = details.get("cfo_int_hz")
        result["cfo_frac_hz"] = details.get("cfo_frac_hz")
        result["sto_int_samp"] = details.get("sto_int_samp")
        result["sto_frac_samp"] = details.get("sto_frac_samp")

    if abs(float(index_payload) - gt_payload_frame) > PREAMBLE_LOCATION_TOL_FRAMES:
        result["outcome"] = "preamble_mislocated"
        return result

    result["cfo_err"] = float(cfo_est - cfo_true)
    result["sto_err"] = float(sto_est - sto_true)

    payload_start = int(index_payload * framePerSymbol)
    if toggles["apply_sto_correction"]:
        payload_start += int(sto_est)
    result["payload_start"] = int(payload_start)
    result["payload_start_frame"] = float(payload_start / framePerSymbol)

    payload_signal = iq[payload_start:]

    if toggles["apply_cfo_correction"]:
        t_axis = np.arange(len(payload_signal)) / opts.fs
        payload_corrected = payload_signal * np.exp(-1j * 2 * np.pi * cfo_est * t_axis)
    else:
        payload_corrected = payload_signal

    if server_config["symbol_decoding_method"] == "loraseek":
        ckpt = server_config.get("loraseek_checkpoint", loraseek_inference.DEFAULT_CHECKPOINT)
        demod_symbols = loraseek_inference.demodulate_payload_loraseek(
            payload_corrected, opts.sf, opts.bw, opts.fs, checkpoint_name=ckpt)
    elif server_config["symbol_decoding_method"] == "nelora":
        demod_symbols = nelora_inference.demodulate_payload_nelora(
            payload_corrected, opts.sf, opts.bw, opts.fs,
            checkpoint_id=server_config.get("nelora_checkpoint", "100000"))
    elif server_config["symbol_decoding_method"] == "gloriphy":
        ckpt = server_config.get("gloriphy_checkpoint")
        pre_start = max(0, payload_start - int(12.25 * framePerSymbol))
        pre_end = pre_start + int(8 * framePerSymbol)
        sfd_start = max(0, payload_start - int(2.25 * framePerSymbol))
        if toggles["apply_cfo_correction"]:
            t_all = np.arange(len(iq)) / opts.fs
            iq_cfo = iq * np.exp(-1j * 2 * np.pi * cfo_est * t_all)
            preamble_corr = iq_cfo[pre_start:pre_end]
            sfd_corr = iq_cfo[sfd_start:payload_start]
        else:
            preamble_corr = iq[pre_start:pre_end]
            sfd_corr = iq[sfd_start:payload_start]
        demod_symbols = gloriphy_inference.demodulate_payload_gloriphy(
            payload_corrected, opts.sf, opts.bw, opts.fs, checkpoint_id=ckpt,
            preamble_iq=preamble_corr, sfd_iq=sfd_corr)
    else:
        demod_symbols = calculate_symbol_alliqfile_with_down_sampling(
            payload_corrected, opts.sf, opts.bw, opts.fs, show=False)

    result["outcome"] = "success"
    result["demod_result"] = _score_payload_symbols(demod_symbols, payload_symbols_gt)
    return result


def run_experiment(run_id: str, client_config: dict, server_config: dict,
                   publish_cb, cancel_event: threading.Event) -> None:
    """The background-thread worker. publish_cb(status, progress, bins) is called
    periodically (not necessarily every packet) so the caller can push a fresh snapshot
    into RunManager's in-memory state and, less often, to disk."""
    opts = _build_opts(client_config)
    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))
    n_packets = client_config["n_packets"]

    seed = client_config.get("seed", -1)
    if seed is not None and seed >= 0:
        rng = np.random.default_rng(seed)
        noise_seed = seed  # fixed base seed for every packet's AWGN draw, matching client3.py's convention
    else:
        rng = np.random.default_rng()
        noise_seed = None

    acc: dict = {}
    publish_cb("running", {"packets_done": 0, "packets_total": n_packets}, [])

    try:
        for i in range(n_packets):
            if cancel_event.is_set():
                publish_cb("cancelled", {"packets_done": i, "packets_total": n_packets}, _publish_bins(acc))
                return

            r = _run_one_packet(opts, client_config, server_config, rng, noise_seed, framePerSymbol)

            if (i + 1) % 50 == 0 or i == 0:
                print("=" * 75)
                print(f"[Packet {i+1}/{n_packets}] (Run: {run_id[:8]})")
                print(f"  * SNR                       : {r.get('snr')} dB")
                print(f"  * GT Payload Location       : frame {r.get('gt_payload_frame'):.3f} (sample {r.get('gt_payload_sample')})")
                print(f"  * GT CFO                    : {r.get('cfo_true'):.2f} Hz")
                print(f"  * GT STO                    : {r.get('sto_true')} samples (numb_offset: {r.get('numb_offset')})")
                if r.get("outcome") in ("preamble_missed", "preamble_mislocated"):
                    print(f"  * Detection Outcome         : {r.get('outcome')} (detected frame: {r.get('index_payload')})")
                else:
                    cfo_i = r.get('cfo_int_hz')
                    cfo_f = r.get('cfo_frac_hz')
                    cfo_tot = r.get('cfo_est')
                    sto_i = r.get('sto_int_samp')
                    sto_f = r.get('sto_frac_samp')
                    sto_tot = r.get('sto_est')
                    p_start = r.get('payload_start')
                    p_frame = r.get('payload_start_frame')
                    p_det = r.get('index_payload')

                    cfo_i_str = f"{cfo_i:.2f} Hz" if cfo_i is not None else "N/A"
                    cfo_f_str = f"{cfo_f:.2f} Hz" if cfo_f is not None else "N/A"
                    cfo_tot_str = f"{cfo_tot:.2f} Hz" if cfo_tot is not None else "N/A"
                    sto_i_str = f"{sto_i:.2f} samp" if sto_i is not None else "N/A"
                    sto_f_str = f"{sto_f:.2f} samp" if sto_f is not None else "N/A"
                    sto_tot_str = f"{sto_tot:.2f} samp" if sto_tot is not None else "N/A"

                    print(f"  * CFO Estimation            : Total {cfo_tot_str} (Int: {cfo_i_str}, Frac: {cfo_f_str})")
                    print(f"  * STO Estimation            : Total {sto_tot_str} (Int: {sto_i_str}, Frac: {sto_f_str})")
                    print(f"  * Payload Start (corrected) : sample {p_start} (frame {p_frame:.3f}, detected frame: {p_det:.3f})")
                print("=" * 75, flush=True)

            snr = r["snr"]
            b = acc.setdefault(snr, _new_bin())
            b["n_total"] += 1
            b["symbol_total_sent"] += len(r["payload_symbols_gt"])

            if r["outcome"] == "preamble_missed":
                b["preamble_missed"] += 1
            elif r["outcome"] == "preamble_mislocated":
                b["preamble_mislocated"] += 1
            else:
                cfo_tol = opts.bw / (2 * opts.n_classes)
                sto_tol = opts.fs / opts.bw
                b["cfo_abs_err_sum"] += abs(r["cfo_err"])
                b["cfo_abs_err_count"] += 1
                b["sto_abs_err_sum"] += abs(r["sto_err"])
                b["sto_abs_err_count"] += 1
                if abs(r["cfo_err"]) <= cfo_tol:
                    b["cfo_correct"] += 1
                if abs(r["sto_err"]) <= sto_tol:
                    b["sto_correct"] += 1

                demod_result = r["demod_result"]
                if demod_result is not None:
                    n_compare, matches = demod_result
                    b["symbol_total"] += n_compare
                    b["symbol_correct"] += matches

            if (i + 1) % 10 == 0 or (i + 1) == n_packets:
                publish_cb("running", {"packets_done": i + 1, "packets_total": n_packets}, _publish_bins(acc))

        publish_cb("completed", {"packets_done": n_packets, "packets_total": n_packets}, _publish_bins(acc))
    except Exception as exc:
        # len(acc) is the SNR-bin count, not a packet count — reporting it here made a
        # sweep that had actually finished all its packets show up as "30/4000".
        done = sum(b["n_total"] for b in acc.values())
        publish_cb("failed", {"packets_done": done, "packets_total": n_packets}, _publish_bins(acc), error=str(exc))
        raise


class RunManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._runs: dict = {}  # run_id -> live in-memory state, replaced wholesale per update
        self._semaphore = threading.Semaphore(MAX_CONCURRENT_RUNS)

    def start_run(self, client_config: dict, server_config: dict, label: str | None = None) -> str:
        validate_config(client_config, server_config)

        if not self._semaphore.acquire(blocking=False):
            raise TooManyRunsError(f"at most {MAX_CONCURRENT_RUNS} runs may be active at once")

        run_id = persistence.new_run_id()
        manifest = persistence.create_manifest(run_id, client_config, server_config, label)
        cancel_event = threading.Event()

        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id, "label": manifest["label"], "status": "queued",
                "error": None, "progress": manifest["progress"], "bins": [],
                "client_config": client_config, "server_config": server_config,
                "cancel_event": cancel_event,
            }

        def _publish(status, progress, bins, error=None):
            with self._lock:
                prev = self._runs.get(run_id, {})
                self._runs[run_id] = {**prev, "status": status, "progress": progress,
                                      "bins": bins, "error": error}
            if status in ("completed", "failed", "cancelled") or progress["packets_done"] % 50 == 0:
                manifest["status"] = status
                manifest["error"] = error
                manifest["progress"] = progress
                manifest["bins"] = bins
                if status == "running" and manifest["started_at"] is None:
                    from datetime import datetime, timezone
                    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
                if status in ("completed", "failed", "cancelled"):
                    from datetime import datetime, timezone
                    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                persistence.write_manifest(run_id, manifest)
                if status == "completed":
                    persistence.export_per_snr_csv(run_id)

        def _worker():
            try:
                run_experiment(run_id, client_config, server_config, _publish, cancel_event)
            finally:
                self._semaphore.release()

        thread = threading.Thread(target=_worker, daemon=True, name=f"run-{run_id[:8]}")
        with self._lock:
            self._runs[run_id]["thread"] = thread
        thread.start()
        return run_id

    def get_status(self, run_id: str) -> dict | None:
        with self._lock:
            live = self._runs.get(run_id)
        if live is not None:
            return {k: v for k, v in live.items() if k not in ("cancel_event", "thread")}
        manifest = persistence.read_manifest(run_id)
        if manifest is None:
            return None
        return {"run_id": run_id, "label": manifest["label"], "status": manifest["status"],
               "error": manifest["error"], "progress": manifest["progress"], "bins": manifest["bins"]}

    def list_runs(self) -> list[dict]:
        with self._lock:
            live_ids = set(self._runs.keys())
            live_summaries = [
                {"run_id": rid, "label": s["label"], "status": s["status"],
                 "client_config": s["client_config"], "server_config": s["server_config"]}
                for rid, s in self._runs.items() if s["status"] in ("queued", "running")
            ]
        disk = [m for m in persistence.list_manifests() if m["run_id"] not in live_ids
               or m["status"] not in ("queued", "running")]
        disk_summaries = [
            {"run_id": m["run_id"], "label": m["label"], "status": m["status"],
             "created_at": m["created_at"], "finished_at": m["finished_at"],
             "client_config": m["client_config"], "server_config": m["server_config"]}
            for m in disk
        ]
        return live_summaries + disk_summaries

    def cancel_run(self, run_id: str) -> bool:
        with self._lock:
            live = self._runs.get(run_id)
            if live is None:
                return False
            live["cancel_event"].set()
            return True

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            self._runs.pop(run_id, None)
        return persistence.delete_run(run_id)


MANAGER = RunManager()
