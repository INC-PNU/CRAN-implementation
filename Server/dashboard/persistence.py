"""
Run persistence — Server/dashboard_runs/<run_id>/manifest.json is the source of truth;
per_snr.csv is a regenerable convenience export (percentage columns computed from the
manifest's raw bins) for pandas/Excel, never read back by this module.

Deliberately a NEW, clean, versioned schema, not a reuse of the 4 incompatible CSV
schemas already scattered across Server/results/, results2/, results3/, temp_results/ —
those stay untouched for the existing notebooks. Listing scans manifest.json files
directly; no separate index database, matching this project's scale (one local user,
dozens-hundreds of runs).
"""
import csv
import io
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

RUNS_DIR = Path(__file__).resolve().parents[1] / "dashboard_runs"

SCHEMA_VERSION = 1

# Windows-only hazard, but it bites this app hard. CPython opens files without
# FILE_SHARE_DELETE, so while any thread has manifest.json open for reading,
# os.replace() onto it fails with PermissionError [WinError 5], and the reader can
# itself fail with [Errno 13]. The dashboard polls GET /api/runs every 2 s, and
# list_manifests() reads *every* run's manifest on each poll, while the worker thread
# writes the live run's manifest — so the two genuinely collide. Before this, one such
# collision raised straight out of _publish and killed the run thread mid-sweep.
#
# Every reader and writer of these files lives in the one Flask process, so the fix is
# to serialize them here: a reader never holds a handle while a replace runs. Retrying
# alone is NOT sufficient — measured with readers spinning flat out, 7 of 9 writes still
# exhausted the retry budget, because a saturating reader keeps a handle open almost
# continuously. The lock removes that contention outright; _with_retry then only has to
# cover what the lock cannot see (antivirus, Explorer preview, a second dashboard
# process), which is brief and rare.
_FILE_LOCK = threading.RLock()

_RETRY_DELAYS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.15, 0.15, 0.15, 0.15)


def _with_retry(op):
    """Run op(), retrying transient Windows sharing failures. Re-raises if they persist
    past ~0.9 s total, which would mean a real permissions problem, not contention."""
    for delay in _RETRY_DELAYS:
        try:
            return op()
        except PermissionError:
            time.sleep(delay)
    return op()


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write via a uniquely-named sibling tmp + atomic replace. The tmp name carries a
    random suffix so two concurrent writers never share one (a fixed 'foo.tmp' lets one
    truncate the other's half-written file, then one replace hits a vanished source)."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding=encoding)
    try:
        _with_retry(lambda: tmp.replace(path))  # atomic on POSIX and Windows (same volume)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass  # best-effort cleanup; the original error is the one worth reporting
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def make_label(client_config: dict, server_config: dict) -> str:
    """Auto-generated summary label, e.g. 'CALoRa (sean) + full correction, SF7
    125kHz@1MHz, SNR -25..-5'. Overridable by the caller (POST /api/runs's "label")."""
    method = server_config.get("preamble_detection_method", "conventional")
    if method == "calora":
        ckpt = server_config.get("calora_checkpoint") or "?"
        method_label = f"CALoRa ({ckpt.replace('.pth', '')})"
    else:
        method_label = "Conventional"

    demod = server_config.get("symbol_decoding_method", "conventional")
    demod_label = "NeLoRa demod" if demod == "nelora" else "conv demod"

    toggles = server_config.get("toggles", {})
    corr_bits = []
    if toggles.get("apply_cfo_correction", True):
        corr_bits.append("CFO")
    if toggles.get("apply_sto_correction", True):
        corr_bits.append("STO")
    corr_label = "+".join(corr_bits) + " correction" if corr_bits else "no correction"

    sf = client_config.get("sf", 7)
    bw = client_config.get("bw", 125_000)
    fs = client_config.get("fs", 1_000_000)
    snr_lo, snr_hi = client_config.get("snr_db_range", [None, None])

    cfo_range = client_config.get("cfo_hz_range", [0, 0])
    sto_range = client_config.get("sto_samp_range", [0, 0])
    has_cfo = (cfo_range[0] != 0 or cfo_range[1] != 0)
    has_sto = (sto_range[0] != 0 or sto_range[1] != 0)
    if has_cfo and has_sto:
        data_tag = "Data: Both CFO & STO"
    elif has_cfo:
        data_tag = "Data: CFO only"
    elif has_sto:
        data_tag = "Data: STO only"
    else:
        data_tag = "Data: Clean"

    return (f"{method_label} [{data_tag}] + {demod_label}, {corr_label}, "
            f"SF{sf} {bw//1000}kHz@{fs//1000}kHz, SNR {snr_lo}..{snr_hi}")


def create_manifest(run_id: str, client_config: dict, server_config: dict, label: str | None) -> dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "label": label or make_label(client_config, server_config),
        "status": "queued",
        "error": None,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "client_config": client_config,
        "server_config": server_config,
        "progress": {"packets_done": 0, "packets_total": client_config.get("n_packets", 0)},
        "bins": [],
    }
    write_manifest(run_id, manifest)
    return manifest


def write_manifest(run_id: str, manifest: dict) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2)
    with _FILE_LOCK:
        _atomic_write_text(d / "manifest.json", payload)


def read_manifest(run_id: str) -> dict | None:
    path = run_dir(run_id) / "manifest.json"

    def _read():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # Also covers the run being deleted between check and read — no TOCTOU gap.
            return None

    # Held only for the duration of one small read, so a poll that walks every run
    # (list_manifests) yields the lock back to the worker between manifests.
    with _FILE_LOCK:
        return _with_retry(_read)


def list_manifests() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        manifest = read_manifest(d.name)
        if manifest is not None:
            out.append(manifest)
    out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return out


def delete_run(run_id: str) -> bool:
    d = run_dir(run_id)
    # Same lock as the readers: rmtree racing a poll's read_manifest would otherwise
    # fail on Windows with the directory "in use".
    with _FILE_LOCK:
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True


def _bin_percentages(b: dict) -> dict:
    # .get(): manifests written before preamble_mislocated existed have no such key. They
    # read back as 0, i.e. exactly the old behaviour, so old runs still render — but their
    # preamble rate remains the inflated pre-fix number and is NOT comparable with runs
    # recorded after this change.
    mislocated = b.get("preamble_mislocated", 0)
    located = b["n_total"] - b["preamble_missed"] - mislocated
    reached = located - b["offset_fail"]
    return {
        "cfo_success_rate_conditioned_pct": (b["cfo_correct"] / reached * 100) if reached else 0.0,
        "cfo_success_rate_per_packet_pct": (b["cfo_correct"] / b["n_total"] * 100) if b["n_total"] else 0.0,
        "sto_success_rate_conditioned_pct": (b["sto_correct"] / reached * 100) if reached else 0.0,
        "sto_success_rate_per_packet_pct": (b["sto_correct"] / b["n_total"] * 100) if b["n_total"] else 0.0,
        "symbol_accuracy_conditioned_pct": (b["symbol_correct"] / b["symbol_total"] * 100) if b["symbol_total"] else 0.0,
        "symbol_accuracy_per_packet_pct": (b["symbol_correct"] / b["symbol_total_sent"] * 100) if b["symbol_total_sent"] else 0.0,
        # "Detected" now requires landing in the right place, not merely answering yes.
        "preamble_detection_rate_pct": (located / b["n_total"] * 100) if b["n_total"] else 0.0,
        # The old definition, kept so a pre-fix run can still be reproduced from a CSV.
        "preamble_answered_yes_pct": ((b["n_total"] - b["preamble_missed"]) / b["n_total"] * 100) if b["n_total"] else 0.0,
    }


def export_per_snr_csv(run_id: str) -> Path:
    """Regenerates per_snr.csv from manifest.json's bins, with percentage columns added
    for pandas/Excel convenience. Export-only — never read back by this module."""
    manifest = read_manifest(run_id)
    if manifest is None:
        raise FileNotFoundError(run_id)

    path = run_dir(run_id) / "per_snr.csv"
    raw_cols = ["snr_db", "n_total", "preamble_missed", "preamble_mislocated",
                "offset_fail", "demod_fail",
                "cfo_correct", "sto_correct", "cfo_mae_hz", "sto_mae_samples",
                "symbol_total", "symbol_correct", "symbol_total_sent"]
    pct_cols = ["preamble_detection_rate_pct", "preamble_answered_yes_pct",
                "cfo_success_rate_conditioned_pct", "cfo_success_rate_per_packet_pct",
                "sto_success_rate_conditioned_pct", "sto_success_rate_per_packet_pct",
                "symbol_accuracy_conditioned_pct", "symbol_accuracy_per_packet_pct"]

    # Built in memory then written atomically: GET /api/runs/<id>/export.csv regenerates
    # this file and hands it to send_file(), which streams it — so a run completing at
    # that moment must not be rewriting the same path underneath the stream.
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(raw_cols + pct_cols)
    for b in sorted(manifest["bins"], key=lambda x: x["snr_db"]):
        pct = _bin_percentages(b)
        writer.writerow([b.get(c) for c in raw_cols] + [round(pct[c], 4) for c in pct_cols])

    with _FILE_LOCK:
        _atomic_write_text(path, buf.getvalue())
    return path
