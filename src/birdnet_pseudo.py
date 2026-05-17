"""
Stage 1e BirdNET — pseudo-label unlabeled train_soundscapes and fine-tune the final head.

Embeds 5s windows with BirdNET v2.4 (no augmentation), filters by top1/runner-up thresholds,
then refines the stage-1d head on focal supervised embeddings + pseudo windows.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .perch_pseudo import (
        _filter_and_soft_label,
        _partial_save,
        soundscape_windows_for_file,
        unlabeled_soundscape_files,
    )
except ImportError:
    from perch_pseudo import (
        _filter_and_soft_label,
        _partial_save,
        soundscape_windows_for_file,
        unlabeled_soundscape_files,
    )

BIRDNET_EMB_DIM = 1024
SR = 32_000


def _configure_tensorflow_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


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
        project_root / "submission" / "birdnet_best_head_code.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _n_classes_from_train_cache(train_cache: Path | None) -> int:
    if train_cache is not None and train_cache.exists():
        d = np.load(str(train_cache), allow_pickle=True)
        if "y" in d.files:
            return int(d["y"].shape[1])
        if "y_train" in d.files:
            return int(d["y_train"].shape[1])
    from birdnet_agent import _build_species_map

    cols, _ = _build_species_map()
    return len(cols)


def _load_pseudo_teacher_head(
    teacher_head_path: Path,
    *,
    train_cache: Path | None = None,
) -> Any:
    """Load stage-1d head on CPU via build_head + weights."""
    _configure_tensorflow_cpu_only()
    import tensorflow as tf

    head_dir = teacher_head_path.parent
    weights_path = head_dir / "final_head.weights.h5"
    if not weights_path.is_file():
        alt = teacher_head_path.with_name("final_head.weights.h5")
        if alt.is_file():
            weights_path = alt

    n_classes = _n_classes_from_train_cache(train_cache)
    code_path = _find_head_code_path(teacher_head_path)
    if code_path is not None and weights_path.is_file():
        print(f"  [BirdNET 1e] Teacher: {code_path.name} + {weights_path.name} (CPU)", flush=True)
        ns: dict[str, Any] = {"tf": tf}
        exec(code_path.read_text(encoding="utf-8"), ns)
        build_head = ns["build_head"]
        teacher = build_head(BIRDNET_EMB_DIM, n_classes)
        teacher.load_weights(str(weights_path))
        return teacher

    if teacher_head_path.is_file():
        print(
            f"  [BirdNET 1e] Teacher: load_model({teacher_head_path.name}) (CPU fallback)",
            flush=True,
        )
        return tf.keras.models.load_model(str(teacher_head_path), compile=False)

    raise FileNotFoundError(
        f"No teacher head at {teacher_head_path} and no weights at {weights_path}"
    )


def _head_probs(teacher: Any, embs: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    out = teacher(embs, training=False)
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def build_birdnet_pseudo_label_cache(
    *,
    config: dict,
    teacher_head_path: Path,
    out_path: Path,
    soundscapes_dir: Path,
    labels_csv: Path,
    top1_threshold: float = 0.55,
    runnerup_max: float = 0.35,
    pseudo_label_weight: float = 0.8,
    embed_batch_size: int = 32,
    train_cache: Path | None = None,
    max_files: int | None = None,
    heartbeat_every: int = 5,
    partial_save_every: int = 500,
) -> dict[str, Any]:
    """
    Embed unlabeled soundscape windows with BirdNET, score with the stage-1d head,
    apply thresholds. Saves ``X_pseudo``, ``y_pseudo``, ``row_ids`` to ``out_path``.
    """
    import birdnet_agent as bn

    stage_1e = (config.get("meta_agent") or {}).get("birdnet_stage_1e") or {}
    stage_1e = stage_1e or (config.get("meta_agent") or {}).get("stage_1e") or {}
    heartbeat_every = int(stage_1e.get("heartbeat_every", heartbeat_every))
    partial_save_every = int(stage_1e.get("partial_save_every", partial_save_every))
    embed_batch_size = max(1, int(embed_batch_size))

    if not teacher_head_path.exists():
        raise FileNotFoundError(f"Teacher head not found: {teacher_head_path}")

    from perch_pseudo import labeled_soundscape_window_keys

    labeled_keys = labeled_soundscape_window_keys(labels_csv)
    noise_pool = unlabeled_soundscape_files(soundscapes_dir, labels_csv)
    if max_files is not None:
        noise_pool = noise_pool[: int(max_files)]

    print(
        f"\n  [BirdNET 1e] Pseudo-label pool: {len(noise_pool)} soundscape files "
        f"(top1≥{top1_threshold}, runner-up<{runnerup_max}, soft weight={pseudo_label_weight})",
        flush=True,
    )

    print("  [BirdNET 1e] Step 1: BirdNET encoder…", flush=True)
    t0 = time.time()
    bn.init_birdnet()
    print(f"  [BirdNET 1e] Step 1 done ({time.time() - t0:.1f}s)", flush=True)

    print("  [BirdNET 1e] Step 2: teacher head on CPU…", flush=True)
    t0 = time.time()
    teacher = _load_pseudo_teacher_head(teacher_head_path, train_cache=train_cache)
    print(f"  [BirdNET 1e] Step 2 done ({time.time() - t0:.1f}s)", flush=True)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_ids: list[str] = []
    n_windows_seen = 0
    n_accepted = 0
    n_failed = 0
    t_start = time.time()

    batch_size = int(
        (config.get("birdnet") or {}).get("embed_batch_size", embed_batch_size)
    )

    with bn._bird_model.encode_session(
        batch_size=batch_size,
        prefetch_ratio=2,
        n_workers=4,
        n_producers=2,
    ) as session:
        bn._bird_session = session

        print("  [BirdNET 1e] Step 3: warmup embed + head…", flush=True)
        t0 = time.time()
        warm = np.zeros(int(SR * 5.0), dtype=np.float32)
        warm_emb = bn._embed_batch([warm], SR)
        _ = _head_probs(teacher, warm_emb)
        print(f"  [BirdNET 1e] Step 3 done ({time.time() - t0:.1f}s) — per-file pass", flush=True)

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
                    embs_list.append(bn._embed_batch(chunk, SR))
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
                print(f"  [BirdNET 1e] skip {fp.name}: {type(exc).__name__}: {exc}", flush=True)
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
                    f"  [BirdNET 1e] [{fi + 1}/{len(noise_pool)}] "
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

        bn._bird_session = None

    if not kept_x:
        from perch_pseudo import write_empty_pseudo_label_cache

        if train_cache is not None and Path(train_cache).exists():
            d_tr = np.load(str(train_cache), allow_pickle=True)
            y_tr = d_tr["y_train"] if "y_train" in d_tr.files else d_tr["y"]
            x_tr = d_tr["X_train"] if "X_train" in d_tr.files else d_tr["X"]
            n_classes = int(y_tr.shape[1])
            emb_dim = int(x_tr.shape[1])
        else:
            bn.init_birdnet()
            with bn._bird_model.encode_session(
                batch_size=batch_size, prefetch_ratio=2, n_workers=4, n_producers=2
            ) as session:
                bn._bird_session = session
                warm_emb = bn._embed_batch([np.zeros(int(SR * 5.0), dtype=np.float32)], SR)
                bn._bird_session = None
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
        f"  [BirdNET 1e] Pseudo cache: {stats['n_accepted']}/{stats['n_windows_seen']} windows "
        f"({stats['accept_rate']:.1%}) → {out_path.name}",
        flush=True,
    )
    return stats
