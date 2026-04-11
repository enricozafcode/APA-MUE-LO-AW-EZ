"""Filesystem locations shared by notebooks and training code."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def birdclef_data_dir() -> Path:
    """
    Root folder for BirdCLEF files (e.g. ``train_metadata.csv``, audio).

    Team workflow: copy or extract the competition into the repo's ``data/`` directory
    (tracked in git as an empty folder via ``data/.gitkeep``; file contents stay local).

    Override with ``BIRDCLEF_DATA_DIR`` in ``.env`` (absolute path) if the dataset lives elsewhere.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root() / ".env", override=False)
    except ImportError:
        pass
    raw = (os.environ.get("BIRDCLEF_DATA_DIR") or "data").strip()
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (repo_root() / p).resolve()
