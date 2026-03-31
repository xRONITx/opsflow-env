from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    run([sys.executable, "-m", "py_compile", "app.py", "inference.py", "server/app.py", "opsflow_env/env.py"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    run(["openenv", "validate"])
    print("\nLocal smoke checks passed.")


if __name__ == "__main__":
    main()
