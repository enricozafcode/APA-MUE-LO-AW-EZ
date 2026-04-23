"""Filesystem locations shared across the project."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def birdclef_data_dir() -> Path:
    """
    Root folder for BirdCLEF files (train.csv, train_audio/, etc.).

    Override with BIRDCLEF_DATA_DIR in .env if the dataset lives elsewhere.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_root() / ".env", override=False)
    except ImportError:
        pass
    raw = (os.environ.get("BIRDCLEF_DATA_DIR") or "data").strip()
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (repo_root() / p).resolve()


def get_experiments_dir() -> Path:
    d = repo_root() / "logs" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_next_experiment_dir() -> Path:
    """Creates and returns the next sequentially numbered experiment directory."""
    experiments_dir = get_experiments_dir()
    existing = sorted(experiments_dir.glob("exp_*"))
    next_num = len(existing) + 1
    exp_dir = experiments_dir / f"exp_{next_num:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
