"""
Precomputed mel tensors for labeled train_soundscapes evaluation.

Soundscape eval loads the same OGG windows and computes identical mels on every
CNN experiment. Caching them once per (n_mels, n_frames, sr, hop_length) removes
repeated librosa I/O from the hot path. Eval uses clean mels (no training aug).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if __package__:
    from .data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission
else:
    from data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR = PROJECT_ROOT / "logs" / "meta_agent" / "cnn_cache"


def soundscape_mel_cache_path(
    cache_dir: Path,
    *,
    n_mels: int,
    n_frames: int,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Path:
    cache_dir = Path(cache_dir)
    name = (
        f"soundscape_mels_{int(n_mels)}x{int(n_frames)}"
        f"_sr{int(sample_rate)}_hop{int(hop_length)}.npz"
    )
    return cache_dir / name


def _segment_to_mel(
    seg: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    import librosa
    import tensorflow as tf

    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_r = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_r.astype(np.float32)


def build_soundscape_mel_cache(
    cache_dir: Path,
    *,
    n_mels: int = 64,
    n_frames: int = 128,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Path:
    """Build (X_mels, row_ids) for all labeled soundscape windows."""
    import librosa

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = soundscape_mel_cache_path(
        cache_dir,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    labels = tables.get("train_soundscapes_labels")
    if labels is None or labels.empty:
        raise RuntimeError("train_soundscapes_labels.csv missing or empty.")

    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)

    def _tok_labels(v):
        import pandas as pd

        if pd.isna(v) or v == "":
            return set()
        return {t.strip() for t in str(v).split(";") if t.strip()}

    def _merge_label_sets(s):
        out = set()
        for v in s:
            out |= _tok_labels(v)
        return out

    grp = (
        labels.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(_merge_label_sets)
        .reset_index()
    )
    import pandas as pd

    grp["end_sec"] = pd.to_timedelta(grp["end"]).dt.total_seconds().astype(int)
    grp["row_id"] = (
        grp["filename"].str.replace(".ogg", "", regex=False) + "_" + grp["end_sec"].astype(str)
    )

    required_stems = {rid.rsplit("_", 1)[0] for rid in grp["row_id"]}
    ogg_files = [
        paths.train_soundscapes_dir / f"{stem}.ogg"
        for stem in sorted(required_stems)
        if (paths.train_soundscapes_dir / f"{stem}.ogg").exists()
    ]
    if not ogg_files:
        raise RuntimeError(f"No labeled train soundscapes in {paths.train_soundscapes_dir}")

    window_samples = int(round(sample_rate * clip_seconds))
    target_windows = int(round(60.0 / clip_seconds))
    if target_windows <= 0:
        raise RuntimeError(f"Invalid clip_seconds={clip_seconds}")
    target_samples = window_samples * target_windows

    labeled_row_ids = set(grp["row_id"])

    X_parts: list[np.ndarray] = []
    row_ids: list[str] = []

    for fi, fpath in enumerate(ogg_files, start=1):
        if fi % 10 == 0:
            print(f"  [soundscape cache] {fi}/{len(ogg_files)} files", flush=True)
        name = fpath.stem
        y_full, _ = librosa.load(str(fpath), sr=sample_rate, mono=True)
        if len(y_full) > target_samples:
            y_full = y_full[:target_samples]
        elif len(y_full) < target_samples:
            y_full = np.pad(y_full, (0, target_samples - len(y_full)))

        for wi in range(target_windows):
            end_sec = int(round((wi + 1) * clip_seconds))
            row_id = f"{name}_{end_sec}"
            if row_id not in labeled_row_ids:
                continue
            st = wi * window_samples
            seg = y_full[st : st + window_samples]
            mel = _segment_to_mel(
                seg,
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_frames=n_frames,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            X_parts.append(mel)
            row_ids.append(row_id)

    if not X_parts:
        raise RuntimeError("No labeled soundscape windows matched cache build.")

    X_mels = np.stack(X_parts, axis=0).astype(np.float32)
    manifest = {
        "n_mels": n_mels,
        "n_frames": n_frames,
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_windows": int(X_mels.shape[0]),
        "n_files": len(ogg_files),
    }
    np.savez_compressed(
        str(out_path),
        X_mels=X_mels,
        row_ids=np.array(row_ids, dtype=object),
        manifest=json.dumps(manifest),
    )
    print(
        f"  [soundscape cache] saved {out_path.name} "
        f"windows={X_mels.shape[0]} shape={X_mels.shape[1:]}"
    )
    return out_path


def ensure_soundscape_mel_cache(
    cache_dir: Path | None = None,
    *,
    n_mels: int = 64,
    n_frames: int = 128,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    force: bool = False,
) -> Path:
    cache_dir = Path(cache_dir or DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR)
    path = soundscape_mel_cache_path(
        cache_dir,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    if path.exists() and not force:
        return path
    return build_soundscape_mel_cache(
        cache_dir,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
    )
