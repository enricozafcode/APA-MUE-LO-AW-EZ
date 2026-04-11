#!/usr/bin/env python3
"""Create .venv and install requirements.txt (Windows / macOS / Linux)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(py: str | Path, args: list[str], *, cwd: Path | str | None = None) -> None:
    cmd = [str(py), *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _ensure_venv(venv_dir: Path) -> None:
    """Attempts to create a venv, falling back to downloading virtualenv if missing."""
    if venv_dir.is_dir() and _venv_python(venv_dir).is_file():
        print(f"Using existing virtual environment at {venv_dir}")
        return

    print(f"Creating virtual environment at {venv_dir} ...")
    
    # 1. Try the standard built-in venv
    try:
        import venv
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        if _venv_python(venv_dir).is_file():
            return
    except Exception as e:
        print(f"Built-in 'venv' failed or is unavailable: {e}")

    # 2. Fallback: Download standalone virtualenv if venv is missing (common on Ubuntu/Debian)
    print("Falling back to downloading standalone 'virtualenv' ...")
    url = "https://bootstrap.pypa.io/virtualenv.pyz"
    
    with tempfile.TemporaryDirectory() as td:
        vz_path = Path(td) / "virtualenv.pyz"
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, vz_path)
            # Run the downloaded zipapp to create the environment
            _run(sys.executable, [str(vz_path), str(venv_dir)])
        except Exception as e:
            print(f"Fatal error: Could not bootstrap virtual environment. {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv"),
        help="Virtual environment directory (default: .venv).",
    )
    args = p.parse_args()

    venv_dir = (root / args.venv).resolve() if not args.venv.is_absolute() else args.venv
    py = _venv_python(venv_dir)
    requirements = root / "requirements.txt"

    if not requirements.is_file():
        print(f"Missing {requirements}", file=sys.stderr)
        sys.exit(1)

    # Automatically create the environment, downloading tools if needed
    _ensure_venv(venv_dir)

    if not py.is_file():
        print(f"Expected interpreter not found: {py}", file=sys.stderr)
        sys.exit(1)

    print("Upgrading pip ...")
    _run(py, ["-m", "pip", "install", "--upgrade", "pip"], cwd=root)

    print("Installing dependencies from requirements.txt (this may take a while) ...")
    _run(py, ["-m", "pip", "install", "-r", str(requirements)], cwd=root)

    print("\nSetup finished. Place competition files under `data/` (see README).")
    print("Activate the environment:")
    if sys.platform == "win32":
        print(f"  {venv_dir}\\Scripts\\activate")
    else:
        print(f"  source {venv_dir / 'bin' / 'activate'}")


if __name__ == "__main__":
    main()