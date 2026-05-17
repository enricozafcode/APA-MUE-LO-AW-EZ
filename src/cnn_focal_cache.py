"""
Precomputed focal training mels for CNN stage 1a/1b (fixed aug) and 1c (per preset).

Builds once per (aug_preset, max_samples, n_mels, n_frames) so architecture search
experiments train on identical tensors. Uses seed=42 for clip selection and audio aug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .augmentation import AudioAugmenter, get_audio_embedding_aug, load_random_soundscape_noise, mix_snr
    from .cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    from .data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission
else:
    from augmentation import AudioAugmenter, get_audio_embedding_aug, load_random_soundscape_noise, mix_snr
    from cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    from data_io import load_core_tables, resolve_birdclef_paths, species_columns_from_sample_submission

DEFAULT_FOCAL_MEL_CACHE_DIR = DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
FOCAL_CLIP_SEED = 42
FOCAL_AUG_SEED = 42


def focal_train_cache_path(
    cache_dir: Path,
    *,
    aug_preset: str,
    max_samples: int | None,
    n_mels: int,
    n_frames: int,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
) -> Path:
    ms_key = "all" if max_samples is None else int(max_samples)
    name = (
        f"focal_train_{str(aug_preset).strip().lower()}_{ms_key}"
        f"_{int(n_mels)}x{int(n_frames)}_sr{int(sample_rate)}.npz"
    )
    return Path(cache_dir) / name


def select_focal_training_clips(
    train_df,
    paths,
    species_cols: list[str],
    *,
    max_samples: int | None,
    seed: int = FOCAL_CLIP_SEED,
) -> tuple[list[tuple[str, Path]], int]:
    """Match CNN harness stratified sampler (seed=42, min 2 per species when possible)."""
    sp2i = {s: i for i, s in enumerate(species_cols)}
    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"

    candidates: list[tuple[str, Path]] = []
    for row in train_df.itertuples(index=False):
        label = str(getattr(row, lcol))
        rel = getattr(row, fcol)
        if label not in sp2i:
            continue
        ap = paths.train_audio_dir / str(rel)
        if ap.exists():
            candidates.append((label, ap))

    if not candidates:
        raise RuntimeError("No candidate audio files found after path/label filtering.")

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Path]] = {}
    for label, ap in candidates:
        by_label.setdefault(label, []).append(ap)
    for paths_list in by_label.values():
        rng.shuffle(paths_list)

    budget = len(candidates) if max_samples is None else min(int(max_samples), len(candidates))
    selected: list[tuple[str, Path]] = []
    leftovers: list[tuple[str, Path]] = []
    min_per_species = 2
    for label, paths_list in by_label.items():
        take = min(len(paths_list), min_per_species)
        for ap in paths_list[:take]:
            if len(selected) < budget:
                selected.append((label, ap))
        for ap in paths_list[take:]:
            leftovers.append((label, ap))

    if len(selected) < budget:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: budget - len(selected)])

    represented = len({label for label, _ in selected})
    return selected, represented


def wav_to_mel(
    wav: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    import librosa
    import tensorflow as tf

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_r = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_r.astype(np.float32)


def load_focal_mel(
    audio_path: Path,
    *,
    sample_rate: int,
    clip_seconds: float,
    n_mels: int,
    n_frames: int,
    aug_preset: str | None,
    paths,
    rng: np.random.Generator,
) -> np.ndarray:
    import librosa

    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    if aug_preset:
        embed_aug = get_audio_embedding_aug(str(aug_preset))
        audio_aug = AudioAugmenter(embed_aug.get("audio", {}))
        wav = audio_aug.apply(wav.astype(np.float32), sample_rate, rng)
        if embed_aug.get("use_snr_mixing"):
            mix_prob = float(embed_aug.get("mix_prob", 0.35))
            if rng.random() < mix_prob:
                ss_dir = paths.train_soundscapes_dir
                pool = sorted(ss_dir.glob("*.ogg")) if ss_dir.exists() else []
                noise = load_random_soundscape_noise(
                    rng, pool, sr=sample_rate, clip_sec=float(clip_seconds)
                )
                if noise is not None:
                    snr_db = float(
                        rng.uniform(
                            float(embed_aug.get("snr_min_db", 0.0)),
                            float(embed_aug.get("snr_max_db", 15.0)),
                        )
                    )
                    wav = mix_snr(wav, noise, snr_db)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

    return wav_to_mel(
        wav,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_frames=n_frames,
    )


def build_focal_train_cache(
    cache_dir: Path,
    *,
    aug_preset: str,
    max_samples: int | None,
    n_mels: int = 64,
    n_frames: int = 128,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
    clip_seed: int = FOCAL_CLIP_SEED,
    aug_seed: int = FOCAL_AUG_SEED,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = focal_train_cache_path(
        cache_dir,
        aug_preset=aug_preset,
        max_samples=max_samples,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
    )

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    train_df = tables["train"]
    species_cols = species_columns_from_sample_submission(tables["sample_submission"])
    sp2i = {s: i for i, s in enumerate(species_cols)}

    selected, represented = select_focal_training_clips(
        train_df, paths, species_cols, max_samples=max_samples, seed=clip_seed
    )
    load_rng = np.random.default_rng(aug_seed)

    X_items: list[np.ndarray] = []
    y_items: list[np.ndarray] = []
    clip_records: list[dict[str, str]] = []

    for i, (label, ap) in enumerate(selected, start=1):
        try:
            mel = load_focal_mel(
                ap,
                sample_rate=sample_rate,
                clip_seconds=clip_seconds,
                n_mels=n_mels,
                n_frames=n_frames,
                aug_preset=aug_preset,
                paths=paths,
                rng=load_rng,
            )
        except Exception:
            continue
        yv = np.zeros(len(species_cols), dtype=np.float32)
        yv[sp2i[label]] = 1.0
        X_items.append(mel)
        y_items.append(yv)
        clip_records.append({"label": label, "path": str(ap)})
        if i % 500 == 0:
            print(f"  [focal cache] loaded {i}/{len(selected)} clips...", flush=True)

    if not X_items:
        raise RuntimeError("No focal training clips could be loaded for cache.")

    X_train = np.stack(X_items, dtype=np.float32)
    y_train = np.stack(y_items, dtype=np.float32)
    manifest: dict[str, Any] = {
        "aug_preset": aug_preset,
        "max_samples": max_samples,
        "n_mels": n_mels,
        "n_frames": n_frames,
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "clip_seed": clip_seed,
        "aug_seed": aug_seed,
        "n_clips": int(X_train.shape[0]),
        "represented_species": represented,
    }
    np.savez_compressed(
        str(out_path),
        X_train=X_train,
        y_train=y_train,
        clip_records=np.array(clip_records, dtype=object),
        manifest=json.dumps(manifest),
    )
    print(
        f"  [focal cache] saved {out_path.name} "
        f"clips={X_train.shape[0]} species={represented} shape={X_train.shape[1:]}"
    )
    return out_path


def ensure_focal_train_cache(
    cache_dir: Path | None = None,
    *,
    aug_preset: str,
    max_samples: int | None,
    n_mels: int = 64,
    n_frames: int = 128,
    sample_rate: int = 32000,
    clip_seconds: float = 5.0,
    force: bool = False,
) -> Path:
    cache_dir = Path(cache_dir or DEFAULT_FOCAL_MEL_CACHE_DIR)
    path = focal_train_cache_path(
        cache_dir,
        aug_preset=aug_preset,
        max_samples=max_samples,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
    )
    if path.exists() and not force:
        return path
    return build_focal_train_cache(
        cache_dir,
        aug_preset=aug_preset,
        max_samples=max_samples,
        n_mels=n_mels,
        n_frames=n_frames,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
    )
