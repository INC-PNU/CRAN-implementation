"""Flask routes — REST API contract for the playground dashboard. Registered onto the
app created in dashboard_app.py (kept separate so dashboard_app.py stays a thin entry
point, matching this project's existing main.py/main2.py/main3.py convention)."""
from flask import Blueprint, request, jsonify, render_template, send_file

from dashboard import calora_bridge, persistence
from dashboard.orchestrator import MANAGER, ConfigError, TooManyRunsError
from utils import nelora_inference

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/capabilities")
def capabilities():
    return jsonify({
        "nelora_available": nelora_inference.NELORA_VALIDATED,
        "checkpoints": calora_bridge.list_calora_checkpoints(),
    })


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
