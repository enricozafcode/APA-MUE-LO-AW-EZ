"""Loads BirdCLEF metadata and produces a lightweight summary for the LLM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_birdclef_summary(data_dir: Path, top_n: int = 10) -> dict:
    """
    Reads train.csv and returns a structured metadata summary.

    The summary is intentionally small — it is embedded directly into LLM prompts,
    so we only include what the model needs to propose a sensible experiment.
    """
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")

    df = pd.read_csv(train_csv)

    label_col = "primary_label" if "primary_label" in df.columns else df.columns[0]
    species_counts = df[label_col].value_counts()

    return {
        "total_samples": int(len(df)),
        "num_species": int(species_counts.nunique()),
        "top_species_by_count": {k: int(v) for k, v in species_counts.head(top_n).items()},
        "columns": list(df.columns),
        "data_dir": str(data_dir),
        "train_csv": str(train_csv),
        "audio_dir": str(data_dir / "train_audio"),
    }
