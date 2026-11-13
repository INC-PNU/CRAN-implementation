"""
Convenience launcher for the LoRa C-RAN Client transmitter.
Runs Client/client3.py (default), client2.py, or client.py from the repository root.

Usage:
    python run_client.py [--ver 1|2|3] [other client args...]
    
    Default: launches Client/client3.py (Transmitter for main3.py protocol).
"""
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CLIENT_DIR = REPO_ROOT / "Client"


def main():
    args = sys.argv[1:]
    ver = "3"
    clean_args = []
    
    i = 0
    while i < len(args):
        if args[i] in ("--ver", "-v", "--version") and i + 1 < len(args):
            ver = args[i + 1]
            i += 2
        elif args[i].startswith("--ver="):
            ver = args[i].split("=", 1)[1]
            i += 1
        else:
            clean_args.append(args[i])
            i += 1

    target_script = CLIENT_DIR / (f"client{ver}.py" if ver in ("2", "3") else "client.py")
    if not target_script.exists():
        print(f"Error: Target script {target_script} not found.")
        sys.exit(1)

    # Determine python executable: prefer repo's venv if available
    venv_py = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = CLIENT_DIR.parent / "Server" / "venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    cmd = [py_exec, str(target_script)] + clean_args
    print(f"Launching C-RAN Client ({target_script.name}) from {CLIENT_DIR}...")
    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CLIENT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, cwd=str(CLIENT_DIR), env=env)
    except KeyboardInterrupt:
        print("\nClient stopped by user.")


if __name__ == "__main__":
    main()
