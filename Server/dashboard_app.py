"""
dashboard_app.py — Flask entry point for the LoRa playground dashboard.

Runs entirely in-process: packet generation and estimation happen directly via
dashboard/orchestrator.py, with no HTTP/base64 round-trip through main.py/main2.py/
main3.py's own servers (those keep running standalone, completely unmodified, on their
own ports 5000/5005/5006 — this app uses a different port and imports only shared,
read-only utility functions from them).

threaded=True is required (unlike main.py/2/3.py's threaded=False): the frontend polls
GET /api/runs/<id>/status every ~2s while a sweep runs in a background thread, and a
single-threaded Flask dev server would block that poll behind the sweep itself.
"""
from pathlib import Path

from flask import Flask

from dashboard.routes import bp

_HERE = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(_HERE / "dashboard" / "templates"),
    static_folder=str(_HERE / "dashboard" / "static"),
)
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5010, debug=False, threaded=True)
