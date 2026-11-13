"""Flask routes — REST API contract for the playground dashboard. Registered onto the
app created in dashboard_app.py (kept separate so dashboard_app.py stays a thin entry
point, matching this project's existing main.py/main2.py/main3.py convention)."""
from flask import Blueprint, request, jsonify, render_template, send_file

from dashboard import calora_bridge, persistence
from dashboard.orchestrator import MANAGER, ConfigError, TooManyRunsError
from dashboard.symbol_benchmark import evaluate_single_frame
from utils import nelora_inference, loraseek_inference, gloriphy_inference

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/capabilities")
def capabilities():
    return jsonify({
        "nelora_available": nelora_inference.NELORA_VALIDATED,
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
