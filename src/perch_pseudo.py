"""
Stage 1e — pseudo-label unlabeled train_soundscapes and fine-tune the final Perch head.

Embeds 5s windows (no augmentation), filters by top1/runner-up thresholds, stores soft labels,
then refines the stage-1d head on focal supervised embeddings + pseudo windows.

Pseudo-label generation follows the BirdNET / Perch notebook pattern:
  - ONNX session first (CPU), then TensorFlow head on CPU only (no Metal + ORT deadlock)
  - Per-file progress heartbeats (not silent 16-file batches)
  - One embed batch per file where possible; ``model(x, training=False)`` not ``predict()``
  - Partial NPZ checkpoints every N files
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SR = 32_000
CLIP_SEC = 5.0
PERCH_SAMPLES = int(SR * CLIP_SEC)
FILE_SEC = 60.0
N_WINDOWS_PER_FILE = int(FILE_SEC // CLIP_SEC)


def labeled_soundscape_window_keys(labels_csv: Path) -> set[tuple[str, int]]:
    """(file_stem, end_sec) keys for windows that already have competition labels."""
    if not labels_csv.exists():
        return set()

    def _tok(val: object) -> set[str]:
        if pd.isna(val) or val == "":
            return set()
        return {t.strip() for t in str(val).split(";") if t.strip()}

    lab = pd.read_csv(labels_csv)
    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[_tok(v) for v in s]))
        .reset_index()
    )
    grp["end_sec"] = pd.to_timedelta(grp["end"]).dt.total_seconds().astype(int)
    return {
        (str(row["filename"]).replace(".ogg", ""), int(row["end_sec"]))
        for _, row in grp.iterrows()
    }


def unlabeled_soundscape_files(
    soundscapes_dir: Path,
    labels_csv: Path,
) -> list[Path]:
    """`.ogg` files with at least one window not present in the labels CSV."""
    labeled = labeled_soundscape_window_keys(labels_csv)
    if not labeled:
        return sorted(soundscapes_dir.glob("*.ogg"))

    out: list[Path] = []
    import librosa

    window_sec = int(CLIP_SEC)
    for fp in sorted(soundscapes_dir.glob("*.ogg")):
        stem = fp.stem
        try:
            dur = librosa.get_duration(path=str(fp))
        except Exception:
            continue
        has_unlabeled = False
        for end_sec in range(window_sec, int(dur) + 1, window_sec):
            if (stem, end_sec) not in labeled:
                has_unlabeled = True
                break
        if has_unlabeled:
            out.append(fp)
    return out


def soundscape_windows_for_file(
    fp: Path,
    *,
    labeled_keys: set[tuple[str, int]],
    sr: int = SR,
    clip_sec: float = CLIP_SEC,
    file_sec: float = FILE_SEC,
) -> list[tuple[np.ndarray, str, int]]:
    """
    Load up to ``file_sec`` of audio once, emit non-overlapping ``clip_sec`` windows.
    Skips windows already in ``labeled_keys``. Returns (waveform, row_id, end_sec).
    """
    import librosa

    stem = fp.stem
    step = int(sr * clip_sec)
    n_full = int(file_sec * sr)
    try:
        y, _ = librosa.load(str(fp), sr=sr, mono=True, duration=file_sec)
    except Exception:
        return []

    y = y.astype(np.float32)
    if len(y) < n_full:
        y = np.pad(y, (0, n_full - len(y)))

    rows: list[tuple[np.ndarray, str, int]] = []
    for w in range(N_WINDOWS_PER_FILE):
        start = w * step
        end_sec = int((start + step) / sr)
        if (stem, end_sec) in labeled_keys:
            continue
        chunk = y[start : start + step]
        if len(chunk) < step:
            chunk = np.pad(chunk, (0, step - len(chunk)))
        rows.append((chunk, f"{stem}_{end_sec}", end_sec))
    return rows


def _filter_and_soft_label(
    probs: np.ndarray,
    *,
    top1_threshold: float,
    runnerup_max: float,
    pseudo_label_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask and soft label matrix for accepted windows."""
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top1_vals = top2[:, 1]
    top2_vals = top2[:, 0]
    mask = (top1_vals >= top1_threshold) & (top2_vals < runnerup_max)
    if not np.any(mask):
        return mask, np.zeros((0, probs.shape[1]), dtype=np.float32)
    y_soft = probs[mask].astype(np.float32) * float(pseudo_label_weight)
    y_soft[y_soft < 1e-4] = 0.0
    return mask, y_soft


def _find_head_code_path(teacher_head_path: Path) -> Path | None:
    head_dir = teacher_head_path.parent
    project_root = head_dir
    for _ in range(6):
        if (project_root / "submission").is_dir():
            break
        if project_root.parent == project_root:
            break
        project_root = project_root.parent
    for candidate in (
        head_dir / "best_head_code.py",
        project_root / "submission" / "perch_best_head_code.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_pseudo_teacher_head(teacher_head_path: Path) -> Any:
    """
    Load the 1d head on CPU via build_head + weights (notebook style).

    Avoids ``load_model`` + ``predict()`` which can hang when ONNX is loaded on macOS.
    """
    from perch_agent import configure_tensorflow_cpu_only

    configure_tensorflow_cpu_only()
    import tensorflow as tf

    head_dir = teacher_head_path.parent
    weights_path = head_dir / "final_head.weights.h5"
    if not weights_path.is_file():
        alt = teacher_head_path.with_name("final_head.weights.h5")
        if alt.is_file():
            weights_path = alt

    species_path = head_dir / "species_cols.json"
    if not species_path.is_file():
        raise FileNotFoundError(f"species_cols.json not found in {head_dir}")
    n_classes = len(json.loads(species_path.read_text(encoding="utf-8")))
    emb_dim = 1536

    code_path = _find_head_code_path(teacher_head_path)
    if code_path is not None and weights_path.is_file():
        print(f"  [1e] Teacher: {code_path.name} + {weights_path.name} (CPU)", flush=True)
        ns: dict[str, Any] = {"tf": tf}
        exec(code_path.read_text(encoding="utf-8"), ns)
        build_head = ns["build_head"]
        teacher = build_head(emb_dim, n_classes)
        teacher.load_weights(str(weights_path))
        return teacher

    if teacher_head_path.is_file():
        print(f"  [1e] Teacher: load_model({teacher_head_path.name}) (CPU fallback)", flush=True)
        return tf.keras.models.load_model(str(teacher_head_path), compile=False)

    raise FileNotFoundError(
        f"No teacher head at {teacher_head_path} and no weights at {weights_path}"
    )


def _head_probs(teacher: Any, embs: np.ndarray) -> np.ndarray:
    """Forward pass without ``Model.predict`` (notebook uses ``model(x, training=False)``)."""
    import tensorflow as tf

    out = teacher(embs, training=False)
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def _partial_save(
    out_path: Path,
    kept_x: list[np.ndarray],
    kept_y: list[np.ndarray],
    kept_ids: list[str],
) -> None:
    if not kept_x:
        return
    partial = out_path.with_name(out_path.stem + "_partial.npz")
    np.savez_compressed(
        str(partial),
        X_pseudo=np.concatenate(kept_x, axis=0).astype(np.float32),
        y_pseudo=np.concatenate(kept_y, axis=0).astype(np.float32),
        row_ids=np.array(kept_ids, dtype=object),
    )
    print(f"    [partial save: {sum(x.shape[0] for x in kept_x)} pseudo windows] → {partial.name}", flush=True)


def write_empty_pseudo_label_cache(
    out_path: Path,
    *,
    embed_dim: int,
    n_classes: int,
    top1_threshold: float,
    runnerup_max: float,
    pseudo_label_weight: float,
    n_windows_seen: int = 0,
    n_files: int = 0,
    n_files_failed: int = 0,
    x_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Write a valid empty pseudo NPZ so stage 1e can fine-tune on supervised data only."""
    if x_shape is None:
        x_shape = (0, int(embed_dim))
    X_pseudo = np.zeros(x_shape, dtype=np.float32)
    y_pseudo = np.zeros((0, int(n_classes)), dtype=np.float32)
    row_ids = np.array([], dtype=object)
    sample_weight = np.zeros(0, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        X_pseudo=X_pseudo,
        y_pseudo=y_pseudo,
        row_ids=row_ids,
        sample_weight=sample_weight,
        top1_threshold=np.float32(top1_threshold),
        runnerup_max=np.float32(runnerup_max),
        pseudo_label_weight=np.float32(pseudo_label_weight),
        n_windows_seen=np.int64(n_windows_seen),
        n_accepted=np.int64(0),
    )
    stats = {
        "n_files": int(n_files),
        "n_files_failed": int(n_files_failed),
        "n_windows_seen": int(n_windows_seen),
        "n_accepted": 0,
        "accept_rate": 0.0,
        "out_path": str(out_path),
        "empty_pseudo": True,
    }
    print(
        f"  [1e] No pseudo windows passed thresholds — wrote empty cache "
        f"({n_windows_seen} windows scanned) → {out_path.name}. "
        f"Stage 1e will fine-tune on supervised (+ val) only.",
        flush=True,
    )
    return stats


def build_pseudo_label_cache(
    *,
    config: dict,
    teacher_head_path: Path,
    out_path: Path,
    soundscapes_dir: Path,
    labels_csv: Path,
    top1_threshold: float = 0.55,
    runnerup_max: float = 0.35,
    pseudo_label_weight: float = 0.8,
    file_batch_size: int = 16,
    embed_batch_size: int = 16,
    max_files: int | None = None,
    heartbeat_every: int = 5,
    partial_save_every: int = 500,
) -> dict[str, Any]:
    """
    Embed unlabeled soundscape windows, score with the stage-1d head, apply thresholds.
    Saves ``X_pseudo``, ``y_pseudo``, ``row_ids``, ``sample_weight`` to ``out_path``.
    """
    from perch_agent import (
        _ensure_deps,
        _find_or_download_onnx,
        _find_or_download_perch_labels,
        _load_onnx_session,
        _perch_embed_batch,
    )

    _ensure_deps()
    perch_cfg = config.get("perch", {})
    stage_1e = (config.get("meta_agent") or {}).get("stage_1e") or {}
    heartbeat_every = int(stage_1e.get("heartbeat_every", heartbeat_every))
    partial_save_every = int(stage_1e.get("partial_save_every", partial_save_every))
    embed_batch_size = max(1, int(embed_batch_size))

    if not teacher_head_path.exists():
        raise FileNotFoundError(f"Teacher head not found: {teacher_head_path}")

    labeled_keys = labeled_soundscape_window_keys(labels_csv)
    noise_pool = unlabeled_soundscape_files(soundscapes_dir, labels_csv)
    if max_files is not None:
        noise_pool = noise_pool[: int(max_files)]

    print(
        f"\n  [1e] Pseudo-label pool: {len(noise_pool)} soundscape files "
        f"(top1≥{top1_threshold}, runner-up<{runnerup_max}, soft weight={pseudo_label_weight})",
        flush=True,
    )

    onnx_path = _find_or_download_onnx(
        perch_cfg.get("onnx_dataset", "rishikeshjani/perch-onnx-for-birdclef-2026")
    )
    _find_or_download_perch_labels(
        perch_cfg.get(
            "perch_labels_model",
            "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu",
        )
    )

    print("  [1e] Step 1: ONNX session (CPU)…", flush=True)
    t0 = time.time()
    sess, inp_name, emb_idx, logit_idx = _load_onnx_session(onnx_path)
    print(f"  [1e] Step 1 done ({time.time() - t0:.1f}s)", flush=True)

    print("  [1e] Step 2: teacher head on CPU (after ONNX — avoids Mac Metal hang)…", flush=True)
    t0 = time.time()
    teacher = _load_pseudo_teacher_head(teacher_head_path)
    print(f"  [1e] Step 2 done ({time.time() - t0:.1f}s)", flush=True)

    print("  [1e] Step 3: warmup ONNX + head forward…", flush=True)
    t0 = time.time()
    warm = np.zeros(PERCH_SAMPLES, dtype=np.float32)
    warm_emb, _ = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, [warm])
    _ = _head_probs(teacher, warm_emb)
    print(f"  [1e] Step 3 done ({time.time() - t0:.1f}s) — starting per-file pass", flush=True)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_ids: list[str] = []
    n_windows_seen = 0
    n_accepted = 0
    n_failed = 0
    t_start = time.time()

    for fi, fp in enumerate(noise_pool):
        file_t0 = time.time()
        try:
            windows = soundscape_windows_for_file(fp, labeled_keys=labeled_keys)
            if not windows:
                continue

            wavs = [w[0] for w in windows]
            ids = [w[1] for w in windows]
            embs_list: list[np.ndarray] = []
            for start in range(0, len(wavs), embed_batch_size):
                chunk = wavs[start : start + embed_batch_size]
                emb, _ = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, chunk)
                embs_list.append(emb)
            embs = np.concatenate(embs_list, axis=0).astype(np.float32)
            probs = _head_probs(teacher, embs)
            n_windows_seen += len(windows)

            mask, y_soft = _filter_and_soft_label(
                probs,
                top1_threshold=top1_threshold,
                runnerup_max=runnerup_max,
                pseudo_label_weight=pseudo_label_weight,
            )
            if np.any(mask):
                kept_x.append(embs[mask])
                kept_y.append(y_soft)
                kept_ids.extend([ids[j] for j, ok in enumerate(mask) if ok])
                n_accepted += int(mask.sum())

        except Exception as exc:
            n_failed += 1
            print(f"  [1e] skip {fp.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        file_dt = time.time() - file_t0
        if (
            (fi + 1) % heartbeat_every == 0
            or file_dt > 30
            or (fi + 1) == len(noise_pool)
        ):
            elapsed = time.time() - t_start
            rate = (fi + 1) / max(elapsed, 1e-3)
            eta_min = (len(noise_pool) - fi - 1) / max(rate, 1e-3) / 60
            kr = 100.0 * n_accepted / max(n_windows_seen, 1)
            print(
                f"  [1e] [{fi + 1}/{len(noise_pool)}] "
                f"kept={n_accepted}/{n_windows_seen} ({kr:.1f}%) "
                f"| failed_files={n_failed} | {rate:.2f} f/s | ETA {eta_min:.1f}m "
                f"| last={file_dt:.1f}s",
                flush=True,
            )

        if (fi + 1) % partial_save_every == 0 and kept_x:
            try:
                _partial_save(out_path, kept_x, kept_y, kept_ids)
            except Exception as exc:
                print(f"    [partial save failed: {exc}]", flush=True)

        if (fi + 1) % 100 == 0:
            gc.collect()

    if not kept_x:
        n_classes = int(_head_probs(teacher, warm_emb).shape[1])
        emb_dim = int(warm_emb.shape[1])
        return write_empty_pseudo_label_cache(
            out_path,
            embed_dim=emb_dim,
            n_classes=n_classes,
            top1_threshold=top1_threshold,
            runnerup_max=runnerup_max,
            pseudo_label_weight=pseudo_label_weight,
            n_windows_seen=n_windows_seen,
            n_files=len(noise_pool),
            n_files_failed=n_failed,
        )

    X_pseudo = np.concatenate(kept_x, axis=0).astype(np.float32)
    y_pseudo = np.concatenate(kept_y, axis=0).astype(np.float32)
    row_ids = np.array(kept_ids, dtype=object)
    sample_weight = np.full(len(X_pseudo), float(pseudo_label_weight), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        X_pseudo=X_pseudo,
        y_pseudo=y_pseudo,
        row_ids=row_ids,
        sample_weight=sample_weight,
        top1_threshold=np.float32(top1_threshold),
        runnerup_max=np.float32(runnerup_max),
        pseudo_label_weight=np.float32(pseudo_label_weight),
        n_windows_seen=np.int64(n_windows_seen),
        n_accepted=np.int64(len(X_pseudo)),
    )

    stats = {
        "n_files": len(noise_pool),
        "n_files_failed": n_failed,
        "n_windows_seen": int(n_windows_seen),
        "n_accepted": int(len(X_pseudo)),
        "accept_rate": float(len(X_pseudo) / max(n_windows_seen, 1)),
        "out_path": str(out_path),
    }
    print(
        f"  [1e] Pseudo cache: {stats['n_accepted']}/{stats['n_windows_seen']} windows "
        f"({stats['accept_rate']:.1%}) → {out_path.name}",
        flush=True,
    )
    return stats
