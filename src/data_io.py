from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__:
    from .paths import birdclef_data_dir
else:  # pragma: no cover - fallback for direct script execution
    from paths import birdclef_data_dir


@dataclass(frozen=True)
class BirdCLEFDataPaths:
    data_root: Path
    train_csv: Path
    taxonomy_csv: Path
    sample_submission_csv: Path
    train_soundscapes_labels_csv: Path
    train_audio_dir: Path
    train_soundscapes_dir: Path


def resolve_birdclef_paths() -> BirdCLEFDataPaths:
    root = birdclef_data_dir()
    return BirdCLEFDataPaths(
        data_root=root,
        train_csv=root / "train.csv",
        taxonomy_csv=root / "taxonomy.csv",
        sample_submission_csv=root / "sample_submission.csv",
        train_soundscapes_labels_csv=root / "train_soundscapes_labels.csv",
        train_audio_dir=root / "train_audio",
        train_soundscapes_dir=root / "train_soundscapes",
    )


def validate_required_files(paths: BirdCLEFDataPaths) -> list[str]:
    """Returns a list of missing required files for MVP experiments."""
    required = {
        "train_csv": paths.train_csv,
        "sample_submission_csv": paths.sample_submission_csv,
        "train_audio_dir": paths.train_audio_dir,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return missing


def load_core_tables(paths: BirdCLEFDataPaths) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if paths.train_csv.exists():
        tables["train"] = pd.read_csv(paths.train_csv)
    if paths.taxonomy_csv.exists():
        tables["taxonomy"] = pd.read_csv(paths.taxonomy_csv)
    if paths.sample_submission_csv.exists():
        tables["sample_submission"] = pd.read_csv(paths.sample_submission_csv)
    if paths.train_soundscapes_labels_csv.exists():
        tables["train_soundscapes_labels"] = pd.read_csv(paths.train_soundscapes_labels_csv)
    return tables


def summarize_tables(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name, df in tables.items():
        summary[name] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
    return summary


def species_columns_from_sample_submission(sample_submission_df: pd.DataFrame) -> list[str]:
    return [column for column in sample_submission_df.columns if column != "row_id"]
