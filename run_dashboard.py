"""
Convenience launcher for the LoRa C-RAN Analytics & Demodulation Benchmark Dashboard.
Runs Server/dashboard_app.py from the repository root.
"""
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SERVER_DIR = REPO_ROOT / "Server"
APP_PATH = SERVER_DIR / "dashboard_app.py"


def main():
    if not APP_PATH.exists():
        print(f"Error: Could not find {APP_PATH}")
        sys.exit(1)

    # Determine python executable: prefer repo's venv if available
    venv_py = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = SERVER_DIR / "venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    cmd = [py_exec, str(APP_PATH)] + sys.argv[1:]
    print(f"Starting LoRa C-RAN Dashboard from {SERVER_DIR}...")
    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SERVER_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, cwd=str(SERVER_DIR), env=env)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")


if __name__ == "__main__":
    main()
