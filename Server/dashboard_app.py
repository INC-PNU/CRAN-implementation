"""
dashboard_app.py — Flask entry point for the LoRa playground dashboard.

Runs entirely in-process: packet generation and estimation happen directly via
dashboard/orchestrator.py, with no HTTP/base64 round-trip through main.py/main2.py/
main3.py's own servers (those keep running standalone, completely unmodified, on their
own ports 5000/5005/5006 — this app uses a different port and imports only shared,
read-only utility functions from them).

Supported symbol decoding benchmarks:
  - Conventional dechirping (classical)
  - NeLoRa (neural)
  - LoRaSeek (neural denoiser)
  - GLoRiPHY (neural conformer: AWGN-filter and full channel-aware denoiser)

threaded=True is required (unlike main.py/2/3.py's threaded=False): the frontend polls
GET /api/runs/<id>/status every ~2s while a sweep runs in a background thread, and a
single-threaded Flask dev server would block that poll behind the sweep itself.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from flask import Flask

from dashboard.routes import bp
from utils import gloriphy_inference, loraseek_inference, nelora_inference

app = Flask(
    __name__,
    template_folder=str(_HERE / "dashboard" / "templates"),
    static_folder=str(_HERE / "dashboard" / "static"),
)
app.register_blueprint(bp)

if __name__ == "__main__":
    print("[Dashboard] Symbol decoding benchmarks initialized:")
    print("  * Conventional dechirping : Available")
    print(f"  * NeLoRa                 : {'Available' if nelora_inference.NELORA_VALIDATED else 'Gated (not validated)'}")
    print(f"  * LoRaSeek               : {'Available' if loraseek_inference.LORASEEK_AVAILABLE else 'Unavailable'}")
    print(f"  * GLoRiPHY               : {'Available' if gloriphy_inference.GLORIPHY_AVAILABLE else 'Unavailable'} "
          f"(Supported SFs: {gloriphy_inference.SUPPORTED_SFS})")
    app.run(host="127.0.0.1", port=5010, debug=False, threaded=True)
