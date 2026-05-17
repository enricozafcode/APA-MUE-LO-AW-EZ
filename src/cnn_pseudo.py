"""
Stage 1e CNN — pseudo-label unlabeled train_soundscapes and fine-tune the final model.

Pseudo-label scan follows the Perch/BirdNET pattern (see ``perch_pseudo.py``):
  - ``configure_tensorflow_cpu_only()`` before ``import tensorflow`` (avoids Mac Metal hang)
  - ``model(x, training=False)`` batch forward — never ``Model.predict()``
  - Per-file heartbeats so the terminal never looks frozen
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .perch_pseudo import (
        labeled_soundscape_window_keys,
        unlabeled_soundscape_files,
        soundscape_windows_for_file,
    )
except ImportError:
    from perch_pseudo import (
        labeled_soundscape_window_keys,
        unlabeled_soundscape_files,
        soundscape_windows_for_file,
    )

SR = 32_000
CLIP_SEC = 5.0


def _configure_tensorflow_cpu_only() -> None:
    """CPU-only TF (Perch 1e pattern — Metal + predict can hang on macOS)."""
    try:
        from .perch_agent import configure_tensorflow_cpu_only
    except ImportError:
        from perch_agent import configure_tensorflow_cpu_only

    configure_tensorflow_cpu_only()


def _cnn_forward_probs(model, batch: np.ndarray) -> np.ndarray:
    """Forward pass without ``Model.predict`` (same as Perch ``_head_probs``)."""
    out = model(batch, training=False)
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def _load_species_cols(data_dir: Path) -> list[str]:
    sub = data_dir / "sample_submission.csv"
    df = pd.read_csv(sub)
    return [c for c in df.columns if c != "row_id"]


def build_cnn_pseudo_cache(
    *,
    config: dict,
    teacher_model_path: Path,
    slot_code_path: Path,
    out_path: Path,
    soundscapes_dir: Path | None = None,
    labels_csv: Path | None = None,
    top1_threshold: float = 0.55,
    runnerup_max: float = 0.35,
    pseudo_label_weight: float = 0.8,
    max_files: int | None = None,
    max_windows: int | None = None,
    heartbeat_every: int = 10,
) -> dict:
    """Predict on unlabeled windows; save mel tensors + soft multi-hot labels."""
    import time

    import librosa

    _configure_tensorflow_cpu_only()
    import tensorflow as tf

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(config.get("data_dir", root / "data"))
    soundscapes_dir = soundscapes_dir or data_dir / "train_soundscapes"
    labels_csv = labels_csv or data_dir / "train_soundscapes_labels.csv"
    species_cols = _load_species_cols(data_dir)
    n_classes = len(species_cols)
    pseudo_cfg = config.get("cnn_pseudo_refine") or {}
    sw_pseudo = float(pseudo_cfg.get("sample_weight_pseudo", 0.5))
    heartbeat_every = max(1, int(heartbeat_every))

    slot_code = slot_code_path.read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(slot_code, "<slot>", "exec"), ns)  # noqa: S102
    cfg = ns["get_training_config"]()
    n_mels = int(cfg.get("n_mels", 64))
    n_frames = int(cfg.get("n_frames", 128))

    print(
        f"  [pseudo] loading teacher on CPU → {Path(teacher_model_path).name}",
        flush=True,
    )
    model = tf.keras.models.load_model(str(teacher_model_path), compile=False)
    in_shape = getattr(model, "input_shape", None)
    if in_shape and len(in_shape) >= 3:
        n_mels = int(in_shape[1] or n_mels)
        n_frames = int(in_shape[2] or n_frames)

    print("  [pseudo] warmup forward (CPU, no predict())…", flush=True)
    t_warm = time.time()
    warm = np.zeros((1, n_mels, n_frames, 1), dtype=np.float32)
    _cnn_forward_probs(model, warm)
    print(f"  [pseudo] warmup done ({time.time() - t_warm:.1f}s)", flush=True)

    labeled_keys = labeled_soundscape_window_keys(labels_csv)
    print(
        f"  [pseudo] finding unlabeled soundscapes (cap={max_files})…",
        flush=True,
    )
    files = unlabeled_soundscape_files(
        soundscapes_dir, labels_csv, max_files=max_files
    )

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    w_list: list[float] = []
    row_ids: list[str] = []

    def _wav_to_mel(wav: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SR, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        with tf.device("/CPU:0"):
            mel_resized = tf.image.resize(
                mel_db[..., np.newaxis], (n_mels, n_frames)
            ).numpy()
        return mel_resized.astype(np.float32)

    print(
        f"  [pseudo] scanning {len(files)} file(s) "
        f"(top1>={top1_threshold} top2<{runnerup_max} mels={n_mels}x{n_frames})",
        flush=True,
    )
    accepted = 0
    scanned = 0
    for fi, fp in enumerate(files, 1):
        print(f"  [pseudo] file {fi}/{len(files)}: {fp.name} — loading windows…", flush=True)
        windows = soundscape_windows_for_file(fp, labeled_keys=labeled_keys)
        if not windows:
            print(
                f"  [pseudo] file {fi}/{len(files)}: no unlabeled windows "
                f"(accepted={accepted} scanned={scanned})",
                flush=True,
            )
            continue
        print(
            f"  [pseudo] file {fi}/{len(files)}: {len(windows)} window(s) — mels + predict…",
            flush=True,
        )
        mels: list[np.ndarray] = []
        meta: list[tuple[str, int]] = []
        for wi, (wav, row_id, _end) in enumerate(windows, 1):
            if max_windows is not None and scanned >= int(max_windows):
                break
            scanned += 1
            mels.append(_wav_to_mel(wav))
            meta.append((row_id, wi))
            if wi == 1 or wi == len(windows) or wi % heartbeat_every == 0:
                print(
                    f"  [pseudo]   mel {wi}/{len(windows)} "
                    f"(total scanned={scanned} accepted={accepted})",
                    flush=True,
                )
        if not mels:
            continue
        batch = np.stack(mels, axis=0).astype(np.float32)
        preds = _cnn_forward_probs(model, batch)
        for mel, (row_id, _wi), pred in zip(mels, meta, preds):
            order = np.argsort(pred)[::-1]
            top1 = float(pred[order[0]])
            top2 = float(pred[order[1]]) if len(order) > 1 else 0.0
            if top1 < top1_threshold or top2 >= runnerup_max:
                continue
            y_soft = np.zeros(n_classes, dtype=np.float32)
            y_soft[order[0]] = top1 * pseudo_label_weight
            X_list.append(mel)
            y_list.append(y_soft)
            w_list.append(sw_pseudo)
            row_ids.append(row_id)
            accepted += 1
            if accepted <= 15 or accepted % 25 == 0:
                sp = species_cols[order[0]] if order[0] < len(species_cols) else "?"
                print(
                    f"  [pseudo] ACCEPT #{accepted}: {row_id} "
                    f"species={sp} top1={top1:.3f} top2={top2:.3f}",
                    flush=True,
                )
        print(
            f"  [pseudo] file {fi}/{len(files)} done | accepted={accepted} scanned={scanned}",
            flush=True,
        )
        if max_windows is not None and scanned >= int(max_windows):
            print(f"  [pseudo] reached max_windows={max_windows} — stopping scan", flush=True)
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not X_list:
        X_empty = np.zeros((0, n_mels, n_frames, 1), dtype=np.float32)
        y_empty = np.zeros((0, n_classes), dtype=np.float32)
        np.savez_compressed(
            str(out_path),
            X_pseudo=X_empty,
            y_pseudo=y_empty,
            sample_weight=np.zeros(0, dtype=np.float32),
            row_id=np.array([], dtype=object),
            n_accepted=np.int32(0),
            n_scanned=np.int32(scanned),
            species_cols=np.array(species_cols, dtype=object),
        )
        stats = {
            "n_accepted": 0,
            "n_scanned": scanned,
            "out_path": str(out_path),
            "empty_pseudo": True,
        }
        print(
            f"  [CNN 1e] No pseudo windows passed thresholds ({scanned} scanned) — "
            f"wrote empty cache → {out_path.name}. Fine-tune will use supervised mels only.",
            flush=True,
        )
        return stats

    np.savez_compressed(
        str(out_path),
        X_pseudo=np.stack(X_list, axis=0),
        y_pseudo=np.stack(y_list, axis=0),
        sample_weight=np.array(w_list, dtype=np.float32),
        row_id=np.array(row_ids, dtype=object),
        n_accepted=np.int32(accepted),
        n_scanned=np.int32(scanned),
        species_cols=np.array(species_cols, dtype=object),
    )
    rate = 100.0 * accepted / max(scanned, 1)
    stats = {"n_accepted": accepted, "n_scanned": scanned, "out_path": str(out_path)}
    print(
        f"  [pseudo] Saved {accepted}/{scanned} windows ({rate:.1f}% accept) → {out_path}",
        flush=True,
    )
    return stats


def pseudo_npz_is_empty(npz_path: Path) -> bool:
    """True when cache is missing or has zero pseudo windows."""
    if not npz_path.exists():
        return True
    try:
        d = np.load(str(npz_path), allow_pickle=True)
        X_ps = d["X_pseudo"]
        return int(X_ps.shape[0]) == 0 if X_ps.size else True
    except (KeyError, OSError, ValueError):
        return True


def build_cnn_pseudo_refine_script(
    *,
    slot_code_path: Path,
    teacher_model_path: Path,
    pseudo_npz: Path,
    sample_weight_supervised: float = 1.0,
    sample_weight_pseudo: float = 0.5,
    epochs: int = 15,
    learning_rate: float = 2e-4,
    val_split: float = 0.1,
    model_save_path: str,
    focal_cache_dir: Path | None = None,
    max_supervised_samples: int | None = None,
    aug_dict: dict | None = None,
) -> str:
    """Fine-tune CNN on focal mels + pseudo mels with warm-start from stage-1d teacher.

    The generated script:
      * Tries to reuse the focal_train_*.npz cache that stage 1d already
        built — full re-loading of ~20K WAVs takes ~15 minutes per run
        and during that window the parent terminal sees nothing.
      * Falls back to per-file librosa decoding with a 1E_LOAD heartbeat
        every 500 files so it never looks frozen.
      * Uses Keras verbose=2 (one line per epoch) plus an explicit
        per-epoch heartbeat callback. verbose=1 emits a `\\r`-updating
        progress bar that buffers as a single line when piped, hiding
        intra-epoch progress entirely.
    """
    slot_code = slot_code_path.read_text(encoding="utf-8")
    focal_cache_str = str(focal_cache_dir) if focal_cache_dir is not None else ""
    _aug_repr = repr(dict(aug_dict or {}))
    init_block = f"""
    _teacher = Path(r"{teacher_model_path}")
    model = build_model(input_shape, num_classes)
    if _teacher.exists():
        try:
            model.load_weights(str(_teacher))
            print(f"  [1e] Warm-start weights from {{_teacher.name}}", flush=True)
        except Exception as _lw_exc:
            _loaded = tf.keras.models.load_model(str(_teacher), compile=False)
            model.set_weights(_loaded.get_weights())
            del _loaded
            print(
                f"  [1e] Warm-start via set_weights from {{_teacher.name}} "
                f"(load_weights: {{_lw_exc}})",
                flush=True,
            )
    else:
        print("  [1e] Teacher missing — random init", flush=True)
"""

    project_root = Path(__file__).resolve().parents[1]
    return f'''
from __future__ import annotations
import os, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(r"{project_root}")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("BIRDCLEF_DATA_DIR", str(_PROJECT_ROOT / "data"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from src.perch_agent import configure_tensorflow_cpu_only
configure_tensorflow_cpu_only()

import numpy as np
import tensorflow as tf

from src.data_io import (
    load_core_tables, resolve_birdclef_paths,
    species_columns_from_sample_submission, validate_required_files,
)
def _wav_to_mel(wav, sample_rate, n_mels, n_frames):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_resized.astype(np.float32)

# --- Locked architecture ---
{slot_code.strip()}

_PSEUDO_NPZ = Path(r"{pseudo_npz}")
_SAVE = Path(r"{model_save_path}")
_FT_EPOCHS = {epochs}
_FT_LR = {learning_rate}
_VAL_SPLIT = {val_split}
_SW_SUP = {sample_weight_supervised}
_SW_PS = {sample_weight_pseudo}
_FOCAL_CACHE_DIR = Path(r"{focal_cache_str}") if r"{focal_cache_str}" else None
_MAX_SUP = {max_supervised_samples!r}  # None = use all supervised focal mels from 1d
_META_AUG = {_aug_repr}  # same aug knobs as stage 1d final train

def _apply_training_augmentation(X, prob, noise_std, tmask, fmask, seed=42):
    if prob <= 0.0:
        return X, 0
    if noise_std <= 0.0 and tmask <= 0 and fmask <= 0:
        return X, 0
    rng_aug = np.random.default_rng(seed)
    X_aug = np.array(X, copy=True)
    changed = 0
    n, n_m, n_f, _ = X_aug.shape
    tmask = max(0, min(int(tmask), max(0, n_f - 1)))
    fmask = max(0, min(int(fmask), max(0, n_m - 1)))
    for i in range(n):
        if rng_aug.random() > prob:
            continue
        if noise_std > 0:
            X_aug[i] += rng_aug.normal(0.0, noise_std, size=X_aug[i].shape).astype(np.float32)
        if tmask > 0:
            w = int(rng_aug.integers(1, tmask + 1))
            t0 = int(rng_aug.integers(0, max(1, n_f - w + 1)))
            X_aug[i, :, t0:t0 + w, :] = 0.0
        if fmask > 0:
            h = int(rng_aug.integers(1, fmask + 1))
            f0 = int(rng_aug.integers(0, max(1, n_m - h + 1)))
            X_aug[i, f0:f0 + h, :, :] = 0.0
        changed += 1
    return X_aug.astype(np.float32), changed

def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()
    cfg["max_samples"] = None
    cfg["epochs"] = _FT_EPOCHS
    cfg["learning_rate"] = _FT_LR
    cfg["val_split"] = _VAL_SPLIT
    sample_rate = int(cfg.get("sample_rate", 32000))
    clip_seconds = float(cfg.get("clip_seconds", 5.0))
    n_mels = int(cfg.get("n_mels", 64))
    n_frames = int(cfg.get("n_frames", 128))
    batch_size = int(cfg.get("batch_size", 32))
    aug_preset = cfg.get("aug_preset")
    print(
        f"[1e] CONFIG sample_rate={{sample_rate}} clip_sec={{clip_seconds}} "
        f"n_mels={{n_mels}} n_frames={{n_frames}} batch_size={{batch_size}} "
        f"epochs={{_FT_EPOCHS}} lr={{_FT_LR}} val_split={{_VAL_SPLIT}} "
        f"aug_preset={{aug_preset}}",
        flush=True,
    )

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    train_df = tables["train"]
    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)
    n_classes = len(species_cols)
    sp2i = {{s: i for i, s in enumerate(species_cols)}}
    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"

    X_list, y_list, sw_list = [], [], []
    cache_hit = False
    if _MAX_SUP is not None:
        print(
            f"[1e] TEST MODE: capping supervised samples at max_supervised_samples={{_MAX_SUP}}",
            flush=True,
        )
    if _FOCAL_CACHE_DIR is not None and aug_preset:
        # Reuse the focal cache stage 1d already built. The cache key is
        # `focal_train_{{preset}}_{{ms_key}}_{{n_mels}}x{{n_frames}}_sr{{sr}}.npz` —
        # ms_key="all" matches the FINAL-RUN harness (max_samples=None).
        _ms_keys = ["all", "None"]
        if _MAX_SUP is not None:
            _ms_keys.append(str(int(_MAX_SUP)))
        for _ms_key in _ms_keys:
            candidate = _FOCAL_CACHE_DIR / f"focal_train_{{aug_preset}}_{{_ms_key}}_{{n_mels}}x{{n_frames}}_sr{{sample_rate}}.npz"
            if candidate.exists():
                print(f"[1e] Reusing 1d focal cache: {{candidate.name}}", flush=True)
                _cd = np.load(str(candidate), allow_pickle=True)
                X_focal = _cd["X_train"].astype(np.float32)
                y_focal = _cd["y_train"].astype(np.float32)
                n_cached = len(X_focal)
                # Deterministic subset for `max_supervised_samples` — same
                # seed every time so repeat runs see the same clips.
                if _MAX_SUP is not None and n_cached > int(_MAX_SUP):
                    _rng_sub = np.random.default_rng(42)
                    _sel = _rng_sub.choice(n_cached, size=int(_MAX_SUP), replace=False)
                    _sel.sort()
                    X_focal = X_focal[_sel]
                    y_focal = y_focal[_sel]
                    print(
                        f"[1e] Subset focal cache: {{n_cached}} -> {{len(X_focal)}}",
                        flush=True,
                    )
                for i in range(len(X_focal)):
                    X_list.append(X_focal[i])
                    y_list.append(y_focal[i])
                    sw_list.append(_SW_SUP)
                print(f"[1e] Loaded focal clips from cache: {{len(X_focal)}}", flush=True)
                cache_hit = True
                break

    if not cache_hit:
        rows = list(train_df.itertuples(index=False))
        if _MAX_SUP is not None and len(rows) > int(_MAX_SUP):
            # Shuffle deterministically then truncate — gives a representative
            # subset rather than the first N rows (which are usually one
            # species in disk order).
            _rng_sub = np.random.default_rng(42)
            _idx = np.arange(len(rows))
            _rng_sub.shuffle(_idx)
            rows = [rows[i] for i in _idx[: int(_MAX_SUP)]]
            print(
                f"[1e] Subset train rows (no cache): {{len(_idx)}} -> {{len(rows)}}",
                flush=True,
            )
        total = len(rows)
        print(f"[1e] No focal cache — decoding {{total}} train files from disk", flush=True)
        t0 = time.time()
        loaded = 0
        skipped = 0
        import librosa
        for ri, row in enumerate(rows, 1):
            label = str(getattr(row, lcol))
            if label not in sp2i:
                skipped += 1
            else:
                ap = paths.train_audio_dir / str(getattr(row, fcol))
                if not ap.exists():
                    skipped += 1
                else:
                    try:
                        wav, _ = librosa.load(str(ap), sr=sample_rate, mono=True, duration=clip_seconds)
                        tl = int(sample_rate * clip_seconds)
                        if len(wav) < tl:
                            wav = np.pad(wav, (0, tl - len(wav)))
                        else:
                            wav = wav[:tl]
                        mel = _wav_to_mel(wav, sample_rate, n_mels, n_frames)
                        y = np.zeros(n_classes, dtype=np.float32)
                        y[sp2i[label]] = 1.0
                        X_list.append(mel)
                        y_list.append(y)
                        sw_list.append(_SW_SUP)
                        loaded += 1
                    except Exception:
                        skipped += 1
            if ri % 500 == 0 or ri == total:
                elapsed = time.time() - t0
                rate = loaded / max(elapsed, 1e-6)
                eta = (total - ri) / max(rate, 1e-6) if rate > 0 else 0.0
                print(
                    f"1E_LOAD: row={{ri}}/{{total}} loaded={{loaded}} "
                    f"skipped={{skipped}} rate={{rate:.1f}}/s eta={{eta:.0f}}s",
                    flush=True,
                )

    d = np.load(str(_PSEUDO_NPZ), allow_pickle=True)
    X_ps = d["X_pseudo"].astype(np.float32)
    y_ps = d["y_pseudo"].astype(np.float32)
    n_ps = int(X_ps.shape[0]) if X_ps.size else 0
    for i in range(n_ps):
        X_list.append(X_ps[i])
        y_list.append(y_ps[i])
        sw_list.append(_SW_PS)

    if not X_list:
        raise RuntimeError("1e: no supervised or pseudo samples available — nothing to fine-tune.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    sw = np.array(sw_list, dtype=np.float32)
    n_sup = len(X_list) - n_ps
    print(
        f"[1e] DATASET supervised={{n_sup}} pseudo={{n_ps}} total={{len(X)}} "
        f"shape={{X.shape}}",
        flush=True,
    )

    if n_sup > 0:
        _ap = float(_META_AUG.get("aug_prob", cfg.get("aug_prob", 1.0)))
        _ans = float(_META_AUG.get("aug_noise_std", cfg.get("aug_noise_std", 0.0)))
        _atm = int(_META_AUG.get("aug_time_mask", cfg.get("aug_time_mask", 0)))
        _afm = int(_META_AUG.get("aug_freq_mask", cfg.get("aug_freq_mask", 0)))
        X_sup, aug_changed = _apply_training_augmentation(
            X[:n_sup], _ap, _ans, _atm, _afm, seed=43
        )
        X[:n_sup] = X_sup
        print(
            f"[1e] AUG_APPLIED supervised={{aug_changed}}/{{n_sup}} "
            f"prob={{_ap}} noise={{_ans}} tmask={{_atm}} fmask={{_afm}}",
            flush=True,
        )

    input_shape = (n_mels, n_frames, 1)
    num_classes = n_classes
{init_block}
    opt = tf.keras.optimizers.Adam(learning_rate=_FT_LR)
    # Match stage-1d harness: loss only, no val/AUC (val+AUC can hang on macOS CPU).
    model.compile(optimizer=opt, loss="binary_crossentropy")
    try:
        model.summary(print_fn=lambda s: print(s, flush=True))
    except Exception:
        pass

    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    _fit_bs = max(1, min(int(batch_size), n))
    _steps = max(1, (n + _fit_bs - 1) // _fit_bs)
    print(
        f"[1e] TRAIN_START samples={{n}} batch_size={{_fit_bs}} "
        f"steps_per_epoch={{_steps}} epochs={{_FT_EPOCHS}} (no val — 1d-style)",
        flush=True,
    )

    # model.fit() can hang on macOS (graph compile). Manual eager batches match 1d.
    try:
        tf.config.run_functions_eagerly(True)
    except (RuntimeError, ValueError, AttributeError):
        pass

    def _batch_loss(raw):
        if isinstance(raw, dict):
            return float(raw.get("loss", raw.get(list(raw.keys())[0], 0.0)))
        if isinstance(raw, (list, tuple)):
            return float(raw[0])
        return float(raw)

    train_start_ts = time.time()
    _warm = idx[: min(_fit_bs, n)]
    print("1E_WARMUP: one train_on_batch…", flush=True)
    _wl = model.train_on_batch(X[_warm], y[_warm], sample_weight=sw[_warm])
    print(f"1E_WARMUP_DONE loss={{_batch_loss(_wl):.6f}}", flush=True)

    for epoch in range(_FT_EPOCHS):
        epoch_start_ts = time.time()
        print(f"1E_EPOCH_BEGIN: epoch={{epoch + 1}}/{{_FT_EPOCHS}}", flush=True)
        order = idx.copy()
        rng.shuffle(order)
        batch_losses: list[float] = []
        for step, start in enumerate(range(0, n, _fit_bs)):
            bi = order[start : start + _fit_bs]
            raw = model.train_on_batch(X[bi], y[bi], sample_weight=sw[bi])
            batch_losses.append(_batch_loss(raw))
            if step == 0 or (step + 1) == _steps:
                print(
                    f"1E_TRAIN_BATCH: batch={{step + 1}}/{{_steps}} "
                    f"loss={{batch_losses[-1]:.6f}}",
                    flush=True,
                )
        mean_loss = sum(batch_losses) / max(len(batch_losses), 1)
        elapsed_total = time.time() - train_start_ts
        epoch_elapsed = time.time() - epoch_start_ts
        done = epoch + 1
        avg = elapsed_total / max(done, 1)
        eta = max(0.0, avg * max(_FT_EPOCHS - done, 0))
        print(
            f"1E_HEARTBEAT: epoch={{done}}/{{_FT_EPOCHS}} "
            f"epoch_time_s={{epoch_elapsed:.1f}} "
            f"elapsed_s={{elapsed_total:.1f}} eta_s={{eta:.1f}} "
            f"loss={{mean_loss:.5f}}",
            flush=True,
        )
    print("[1e] TRAIN_DONE", flush=True)
    _SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(_SAVE)
    print(f"[1e] MODEL_SAVED: {{_SAVE}}", flush=True)
    print("PSEUDO_REFINE_DONE", flush=True)

if __name__ == "__main__":
    main()
'''
