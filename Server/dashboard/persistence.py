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
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timezone

RUNS_DIR = Path(__file__).resolve().parents[1] / "dashboard_runs"

SCHEMA_VERSION = 1


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

    return (f"{method_label} + {demod_label}, {corr_label}, "
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
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(d / "manifest.json")  # atomic on both POSIX and Windows (same volume)


def read_manifest(run_id: str) -> dict | None:
    path = run_dir(run_id) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


def _bin_percentages(b: dict) -> dict:
    reached = b["n_total"] - b["preamble_missed"] - b["offset_fail"]
    return {
        "cfo_success_rate_conditioned_pct": (b["cfo_correct"] / reached * 100) if reached else 0.0,
        "cfo_success_rate_per_packet_pct": (b["cfo_correct"] / b["n_total"] * 100) if b["n_total"] else 0.0,
        "sto_success_rate_conditioned_pct": (b["sto_correct"] / reached * 100) if reached else 0.0,
        "sto_success_rate_per_packet_pct": (b["sto_correct"] / b["n_total"] * 100) if b["n_total"] else 0.0,
        "symbol_accuracy_conditioned_pct": (b["symbol_correct"] / b["symbol_total"] * 100) if b["symbol_total"] else 0.0,
        "symbol_accuracy_per_packet_pct": (b["symbol_correct"] / b["symbol_total_sent"] * 100) if b["symbol_total_sent"] else 0.0,
        "preamble_detection_rate_pct": ((b["n_total"] - b["preamble_missed"]) / b["n_total"] * 100) if b["n_total"] else 0.0,
    }


def export_per_snr_csv(run_id: str) -> Path:
    """Regenerates per_snr.csv from manifest.json's bins, with percentage columns added
    for pandas/Excel convenience. Export-only — never read back by this module."""
    manifest = read_manifest(run_id)
    if manifest is None:
        raise FileNotFoundError(run_id)

    path = run_dir(run_id) / "per_snr.csv"
    raw_cols = ["snr_db", "n_total", "preamble_missed", "offset_fail", "demod_fail",
                "cfo_correct", "sto_correct", "cfo_mae_hz", "sto_mae_samples",
                "symbol_total", "symbol_correct", "symbol_total_sent"]
    pct_cols = ["preamble_detection_rate_pct",
                "cfo_success_rate_conditioned_pct", "cfo_success_rate_per_packet_pct",
                "sto_success_rate_conditioned_pct", "sto_success_rate_per_packet_pct",
                "symbol_accuracy_conditioned_pct", "symbol_accuracy_per_packet_pct"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(raw_cols + pct_cols)
        for b in sorted(manifest["bins"], key=lambda x: x["snr_db"]):
            pct = _bin_percentages(b)
            writer.writerow([b.get(c) for c in raw_cols] + [round(pct[c], 4) for c in pct_cols])
    return path
