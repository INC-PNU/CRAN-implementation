"""
Convenience launcher for the LoRa C-RAN Server receiver.
Runs Server/main3.py (default), main2.py, or main.py from the repository root.

Usage:
    python run_server.py [--ver 1|2|3] [other main args...]
    
    Default: launches Server/main3.py (CALoRa detection + conventional offset demodulation).
"""
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SERVER_DIR = REPO_ROOT / "Server"


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

    target_script = SERVER_DIR / (f"main{ver}.py" if ver in ("2", "3") else "main.py")
    if not target_script.exists():
        print(f"Error: Target script {target_script} not found.")
        sys.exit(1)

    # Determine python executable: prefer repo's venv if available
    venv_py = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = SERVER_DIR / "venv" / "Scripts" / "python.exe"
    py_exec = str(venv_py) if venv_py.exists() else sys.executable

    cmd = [py_exec, str(target_script)] + clean_args
    print(f"Launching C-RAN Server ({target_script.name}) from {SERVER_DIR}...")
    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SERVER_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, cwd=str(SERVER_DIR), env=env)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")


if __name__ == "__main__":
    main()
