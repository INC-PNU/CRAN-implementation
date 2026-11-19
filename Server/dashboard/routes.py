"""Flask routes — REST API contract for the playground dashboard. Registered onto the
app created in dashboard_app.py (kept separate so dashboard_app.py stays a thin entry
point, matching this project's existing main.py/main2.py/main3.py convention)."""
from flask import Blueprint, request, jsonify, render_template, send_file

from dashboard import calora_bridge, persistence
from dashboard.orchestrator import MANAGER, ConfigError, TooManyRunsError
from dashboard.symbol_benchmark import evaluate_single_frame
from dashboard.sync_experiment import (
    run_sync_experiment,
    load_saved_results as load_sync_experiment_results,
    save_results as save_sync_experiment_results,
)
from utils import nelora_inference, loraseek_inference, gloriphy_inference

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/capabilities")
def capabilities():
    return jsonify({
        "nelora_available": nelora_inference.NELORA_VALIDATED,
        "nelora_checkpoints": nelora_inference.list_nelora_checkpoints(),
        "loraseek_available": loraseek_inference.LORASEEK_AVAILABLE,
        "loraseek_checkpoints": loraseek_inference.list_loraseek_checkpoints(),
        "gloriphy_available": gloriphy_inference.GLORIPHY_AVAILABLE,
        "gloriphy_supported_sfs": gloriphy_inference.SUPPORTED_SFS,
        "gloriphy_checkpoints": gloriphy_inference.list_gloriphy_checkpoints(),
        "checkpoints": calora_bridge.list_calora_checkpoints(),
    })


@bp.route("/api/benchmark/single-frame", methods=["POST"])
def benchmark_single_frame():
    data = request.get_json(force=True) or {}
    try:
        sf = int(data.get("sf", 7))
        bw = int(data.get("bw", 125_000))
        fs = int(data.get("fs", 1_000_000))
        n_symbols = int(data.get("n_symbols", 10))
        if not (1 <= n_symbols <= 1000):
            return jsonify({
                "error": "invalid_parameters",
                "detail": f"Payload length (n_symbols) must be between 1 and 1000 (got {n_symbols})"
            }), 400
        snr_db = float(data.get("snr_db", 10.0))
        apply_cfo = bool(data.get("apply_cfo", False))
        cfo_hz = float(data.get("cfo_hz", 500.0))
        apply_sto = bool(data.get("apply_sto", False))
        sto_samples = int(data.get("sto_samples", 20))
        apply_compensation = bool(data.get("apply_compensation", True))
        seed = int(data.get("seed", -1))

        result = evaluate_single_frame(
            sf=sf, bw=bw, fs=fs, n_symbols=n_symbols, snr_db=snr_db,
            apply_cfo=apply_cfo, cfo_hz=cfo_hz,
            apply_sto=apply_sto, sto_samples=sto_samples,
            apply_compensation=apply_compensation,
            seed=seed
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": "benchmark_failed", "detail": str(exc)}), 500


@bp.route("/api/benchmark/sync-experiment", methods=["GET"])
def get_sync_experiment():
    """Return precomputed denoised-synchronization experiment results (SF7 & SF8)."""
    return jsonify({"results": load_sync_experiment_results()}), 200


@bp.route("/api/benchmark/sync-experiment", methods=["POST"])
def post_sync_experiment():
    """Run a live Monte-Carlo denoised-synchronization comparison sweep."""
    data = request.get_json(force=True) or {}
    try:
        sf = int(data.get("sf", 7))
        snr_min = int(data.get("snr_min", -24 if sf == 7 else -26))
        snr_max = int(data.get("snr_max", -10 if sf == 7 else -12))
        snr_step = int(data.get("snr_step", 2))
        n_trials = int(data.get("n_trials", 40))
        cfo_lo = float(data.get("cfo_lo", -4358.0))
        cfo_hi = float(data.get("cfo_hi", 4584.0))
        sto_lo = int(data.get("sto_lo", 0))
        sto_hi = int(data.get("sto_hi", 50))
        seed = int(data.get("seed", 2026))
        should_save = bool(data.get("save", True))

        if sf not in (7, 8):
            return jsonify({"error": "invalid_parameters", "detail": f"SF must be 7 or 8 (got {sf})"}), 400
        if snr_min > snr_max or snr_step <= 0:
            return jsonify({"error": "invalid_parameters", "detail": "Invalid SNR range or step"}), 400
        if not (5 <= n_trials <= 500):
            return jsonify({"error": "invalid_parameters", "detail": "n_trials must be between 5 and 500"}), 400
        if cfo_lo > cfo_hi or sto_lo > sto_hi:
            return jsonify({"error": "invalid_parameters", "detail": "Range lo must be <= hi"}), 400

        result = run_sync_experiment(
            sf=sf,
            snr_min=snr_min,
            snr_max=snr_max,
            snr_step=snr_step,
            n_trials=n_trials,
            cfo_lo=cfo_lo,
            cfo_hi=cfo_hi,
            sto_lo=sto_lo,
            sto_hi=sto_hi,
            seed=seed,
        )
        if should_save:
            save_sync_experiment_results(sf, result)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": "sync_experiment_failed", "detail": str(exc)}), 500


@bp.route("/api/runs", methods=["POST"])
def create_run():
    data = request.get_json(force=True) or {}
    client_config = data.get("client", {})
    server_config = data.get("server", {})
    label = data.get("label")

    try:
        run_id = MANAGER.start_run(client_config, server_config, label)
    except ConfigError as exc:
        return jsonify({"error": "invalid_config", "detail": str(exc)}), 400
    except TooManyRunsError as exc:
        return jsonify({"error": "too_many_runs", "detail": str(exc)}), 409

    return jsonify({"run_id": run_id, "status": "queued"}), 201


@bp.route("/api/runs", methods=["GET"])
def list_runs():
    return jsonify({"runs": MANAGER.list_runs()})


@bp.route("/api/runs/<run_id>/status", methods=["GET"])
def run_status(run_id):
    status = MANAGER.get_status(run_id)
    if status is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(status)


@bp.route("/api/runs/<run_id>", methods=["GET"])
def run_detail(run_id):
    manifest = persistence.read_manifest(run_id)
    if manifest is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(manifest)


@bp.route("/api/runs/<run_id>/export.csv", methods=["GET"])
def export_run_csv(run_id):
    manifest = persistence.read_manifest(run_id)
    if manifest is None:
        return jsonify({"error": "not_found"}), 404
    # Regenerated fresh every request (cheap: one run's bins are at most ~60 rows), so a
    # download taken mid-sweep reflects progress as of the last periodic manifest write,
    # not a stale file left over from a previous export.
    path = persistence.export_per_snr_csv(run_id)
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in manifest["label"])[:60]
    return send_file(path, mimetype="text/csv", as_attachment=True,
                     download_name=f"{safe_label}_{run_id[:8]}.csv")


@bp.route("/api/runs/<run_id>", methods=["PATCH"])
def rename_run(run_id):
    manifest = persistence.read_manifest(run_id)
    if manifest is None:
        return jsonify({"error": "not_found"}), 404
    data = request.get_json(force=True) or {}
    if "label" in data:
        manifest["label"] = data["label"]
        persistence.write_manifest(run_id, manifest)
    return jsonify(manifest)


@bp.route("/api/runs/<run_id>/cancel", methods=["POST"])
def cancel_run(run_id):
    ok = MANAGER.cancel_run(run_id)
    if not ok:
        return jsonify({"error": "not_found_or_not_running"}), 404
    return jsonify({"status": "cancelled"})


@bp.route("/api/runs/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    MANAGER.cancel_run(run_id)  # no-op if not running
    ok = MANAGER.delete_run(run_id)
    if not ok:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"deleted": True})
