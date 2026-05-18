"""
Autonomous BirdCLEF Research Agent — Structured Search (Integrated)
====================================================================
Phase 0: Baseline creation (generate + verify initial working model)
Phase 1: Linear search (coordinate descent — one dimension at a time, coarse then zoom)
Phase 2: Random search (random combos → LLM analysis → focused exploitation → tweaks)
Phase 2.5: Medium-scale validation
Phase 2.6: Reality-check gate
Transfer Exploration: LLM-guided pretrained backbone search (MobileNetV2, etc.)
Phase 3: Final training on best overall config + Kaggle notebook

This file integrates the previously-separate `agent_transfer` module directly,
so no external import of `agent_transfer` is required.
"""

from __future__ import annotations

import sys
import inspect
import json
import math
import random
import re
import ast

# Force UTF-8 stdout/stderr on Windows (handles unicode chars like → ★ ─ in print statements)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path
from datetime import datetime

import numpy as np

if __package__:
    from .code_executor import CodeExecutor
    from .evaluator import Evaluator
    from .llm_client import LLMClient
    from .eda_agent import run_eda_phase
else:
    from code_executor import CodeExecutor
    from evaluator import Evaluator
    from llm_client import LLMClient
    from eda_agent import run_eda_phase

try:
    from .soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict
    from .cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    from .cnn_focal_cache import DEFAULT_FOCAL_MEL_CACHE_DIR
except ImportError:
    from soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict
    from cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    from cnn_focal_cache import DEFAULT_FOCAL_MEL_CACHE_DIR


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS & SEARCH SPACE
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "depth": 3,
    "filters_base": 32,
    "filter_pattern": "doubling",
    "dropout": 0.0,
    "batch_norm": True,
    "pooling_type": "global_avg",
    "residuals": None,  # None = auto (enable for depth>=6, disable otherwise)
    "weight_decay": 1e-4,
    "classifier_hidden_units": 256,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "n_mels": 64,
    "n_frames": 128,
    "val_split": 0.2,
    "aug_prob": 0.0,
    "aug_noise_std": 0.0,
    "aug_time_mask": 0,
    "aug_freq_mask": 0,
}

# Ordered by impact tier (Tier 1 searched first)
SEARCH_DIMENSIONS = [
    {"name": "depth",          "coarse": [1, 2, 3, 4, 6, 8, 12, 15, 20], "type": "int",         "zoom": True},
    {"name": "filters_base",   "coarse": [8, 16, 32, 64, 128],           "type": "int",         "zoom": True},
    {"name": "learning_rate",  "coarse": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4], "type": "log_float", "zoom": True},
    {"name": "weight_decay",   "coarse": [0.0, 1e-4, 1e-3, 1e-2],        "type": "log_float", "zoom": False},
    {"name": "classifier_hidden_units", "coarse": [0, 128, 256, 512],    "type": "int",       "zoom": False},
    {"name": "residuals",      "coarse": [False, True],                   "type": "bool",        "zoom": False},
]


# ═══════════════════════════════════════════════════════════════════════════
# HARNESS (fixed data-loading / training / evaluation wrapper)
# ═══════════════════════════════════════════════════════════════════════════

HARNESS_PREFIX = """
from __future__ import annotations
import os, sys, inspect, time
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_ROOT = None
for _cand in [_SCRIPT_PATH.parent] + list(_SCRIPT_PATH.parents):
    if (_cand / "src").exists() and (_cand / "configs").exists():
        _PROJECT_ROOT = _cand
        break
if _PROJECT_ROOT is None:
    # Fallback for unexpected layouts.
    _PROJECT_ROOT = _SCRIPT_PATH.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("BIRDCLEF_DATA_DIR", str(_PROJECT_ROOT / "data"))

from src.data_io import (
    load_core_tables,
    resolve_birdclef_paths,
    species_columns_from_sample_submission,
    validate_required_files,
)

def _wav_to_mel(wav, sample_rate, n_mels, n_frames):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = tf.image.resize(mel_db[..., np.newaxis], (n_mels, n_frames)).numpy()
    return mel_resized.astype(np.float32)


def _load_focal_mel(audio_path, sample_rate, clip_seconds, n_mels, n_frames, cfg, paths, rng):
    # Focal clip -> optional audio aug + soundscape SNR mix -> mel (training only).
    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    preset = cfg.get("aug_preset")
    if preset:
        from src.augmentation import (
            AudioAugmenter,
            get_audio_embedding_aug,
            load_random_soundscape_noise,
            mix_snr,
        )
        embed_aug = get_audio_embedding_aug(str(preset), cache_dir=_FOCAL_MEL_CACHE_DIR)
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
                    snr_db = float(rng.uniform(
                        float(embed_aug.get("snr_min_db", 0.0)),
                        float(embed_aug.get("snr_max_db", 15.0)),
                    ))
                    wav = mix_snr(wav, noise, snr_db)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]

    return _wav_to_mel(wav, sample_rate, n_mels, n_frames)


def _default_load_mel(audio_path, sample_rate, clip_seconds, n_mels, n_frames):
    target_len = int(sample_rate * clip_seconds)
    wav, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True, duration=clip_seconds)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]
    return _wav_to_mel(wav, sample_rate, n_mels, n_frames)

_FOCAL_MEL_CACHE_DIR = None
""".strip()


def _make_harness_suffix(*, is_final=False, model_save_path=""):
    if is_final:
        # Respect cfg['max_samples'] when set (e.g. cnn_stage_1d.max_samples=500
        # during development). Defaults to None → use ALL data, which is the
        # original FINAL-RUN intent. Previously this line hardcoded `None`,
        # silently ignoring stage_1d.max_samples and loading every focal clip
        # with heavy augmentation — the most common cause of 1d "getting stuck".
        max_line = (
            "    _ms = cfg.get('max_samples', None)\n"
            "    max_samples = None if _ms is None else int(_ms)  # FINAL RUN: cfg-controlled (default ALL data)"
        )
        val_line = (
            "    _vs = cfg.get('val_split', 0.1)\n"
            "    val_split = float(_vs) if _vs is not None else 0.1  # FINAL RUN: checkpoint on validation"
        )
        ckpt_line = "    checkpoint_best = True"
        # NOTE: `save_block` is NOT itself an f-string — it's a regular Python
        # string spliced verbatim into the outer assembled script. The braces
        # below must therefore be SINGLE `{...}` so they form valid f-string
        # substitutions *inside the generated script*. Doubled `{{...}}` here
        # used to print literal text like "final_save_target={_mp}" at runtime.
        save_block = (
            f'\n    _mp = Path(r"{model_save_path}")\n'
            "    _mp.parent.mkdir(parents=True, exist_ok=True)\n"
            '    print(f"PHASE3_DEBUG: final_save_target={_mp}")\n'
            "    if checkpoint_best and has_validation and (_ckpt is not None) and _ckpt.exists():\n"
            '        print(f"PHASE3_DEBUG: restoring_best_checkpoint=True path={_ckpt}")\n'
            "        model = tf.keras.models.load_model(_ckpt)\n"
            '        print(f"MODEL_RESTORED_FROM_BEST: {_ckpt}")\n'
            "    else:\n"
            "        print(\n"
            '            f"PHASE3_DEBUG: restoring_best_checkpoint=False "\n'
            '            f"checkpoint_best={checkpoint_best} has_validation={has_validation} "\n'
            '            f"ckpt_exists={(_ckpt is not None and _ckpt.exists())}"\n'
            "        )\n"
            "    model.save(_mp)\n"
            '    print(f"MODEL_SAVED: {_mp}")\n'
        )
    else:
        max_line = '    max_samples = cfg.get("max_samples", 1500)'
        val_line = (
            "    _vs = cfg.get('val_split', 0.2)\n"
            "    val_split = float(_vs) if _vs is not None else 0.2"
        )
        ckpt_line = "    checkpoint_best = bool(cfg.get('use_best_checkpoint', True))"
        save_block = ""

    return f"""
def main():
    global model, X_train, y_train, X_val, y_val
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()
    cfg.setdefault("optimizer", "adam")
{max_line}
{val_line}
{ckpt_line}
    print(
        f"PHASE3_DEBUG: val_split_requested={{val_split}} checkpoint_best={{checkpoint_best}}",
        flush=True,
    )
    sample_rate   = cfg.get("sample_rate", 32000)
    clip_seconds  = cfg.get("clip_seconds", 5.0)
    n_mels        = int(cfg.get("n_mels", 64))
    n_frames      = int(cfg.get("n_frames", 128))
    epochs        = int(cfg.get("epochs", 3))
    batch_size    = int(cfg.get("batch_size", 32))
    learning_rate = cfg.get("learning_rate", 1e-3)
    aug_prob      = float(cfg.get("aug_prob", 0.0))
    aug_noise_std = float(cfg.get("aug_noise_std", 0.0))
    aug_time_mask = int(cfg.get("aug_time_mask", 0))
    aug_freq_mask = int(cfg.get("aug_freq_mask", 0))
    aug_preset    = cfg.get("aug_preset")
    print(
        f"CONFIG_ACTIVE: max_samples={{max_samples}} epochs={{epochs}} "
        f"n_mels={{n_mels}} n_frames={{n_frames}} batch_size={{batch_size}} "
        f"aug_preset={{aug_preset}}",
        flush=True,
    )

    optimizer_name = cfg.get("optimizer", "adam")
    if optimizer_name == "sgd_momentum":
        _opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        _opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    paths = resolve_birdclef_paths()
    missing = validate_required_files(paths)
    if missing:
        raise FileNotFoundError(f"Missing files: {{missing}}")

    tables = load_core_tables(paths)
    train_df = tables["train"]
    sample_sub = tables["sample_submission"]
    species_cols = species_columns_from_sample_submission(sample_sub)
    sp2i = {{s: i for i, s in enumerate(species_cols)}}

    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"
    mel_fn = _default_load_mel
    if "build_features" in globals() and callable(build_features):
        try:
            sig = inspect.signature(build_features)
            if len(sig.parameters) == 5:
                mel_fn = build_features
        except Exception:
            pass

    _focal_cache_file = None
    if aug_preset and _FOCAL_MEL_CACHE_DIR:
        _ms_key = "all" if max_samples is None else int(max_samples)
        _focal_cache_file = (
            Path(_FOCAL_MEL_CACHE_DIR)
            / f"focal_train_{{aug_preset}}_{{_ms_key}}_{{n_mels}}x{{n_frames}}_sr{{int(sample_rate)}}.npz"
        )

    if _focal_cache_file is not None and _focal_cache_file.exists():
        print(f"FOCAL_TRAIN_CACHE: loading {{_focal_cache_file.name}}")
        _fcd = np.load(str(_focal_cache_file), allow_pickle=True)
        X_all = _fcd["X_train"].astype(np.float32)
        y_all = _fcd["y_train"].astype(np.float32)
        try:
            import json as _json
            _manifest = _json.loads(str(_fcd["manifest"]))
            represented_species = int(_manifest.get("represented_species", 0))
        except Exception:
            represented_species = 0
        print(
            f"  Focal cache clips={{X_all.shape[0]}} represented_species={{represented_species}}"
        )
    else:
        # Locked clip list shared across CNN phases (1a–1d); same seed/max_samples → same files.
        import json as _json
        _clip_seed = int(cfg.get("focal_clip_seed", 42))
        _ms_key = "all" if max_samples is None else int(max_samples)
        _manifest_file = None
        if _FOCAL_MEL_CACHE_DIR:
            _manifest_file = (
                Path(_FOCAL_MEL_CACHE_DIR)
                / f"focal_train_clip_manifest_{{_ms_key}}_sr{{int(sample_rate)}}_seed{{_clip_seed}}.jsonl"
            )
        selected = []
        if _manifest_file is not None and _manifest_file.exists():
            print(f"FOCAL_CLIP_MANIFEST: loading {{_manifest_file.name}}")
            with _manifest_file.open(encoding="utf-8") as _mf:
                for _line in _mf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _row = _json.loads(_line)
                    selected.append((str(_row["label"]), Path(_row["path"])))
            represented_species = len(set(_l for _l, _ in selected))
            print(
                f"  Manifest clips={{len(selected)}} represented_species={{represented_species}}"
            )
        else:
            candidates = []
            for row in train_df.itertuples(index=False):
                label = str(getattr(row, lcol))
                rel = getattr(row, fcol)
                if label not in sp2i:
                    continue
                ap = paths.train_audio_dir / str(rel)
                if not ap.exists():
                    continue
                candidates.append((label, ap))

            if not candidates:
                raise RuntimeError("No candidate audio files found after path/label filtering.")

            rng = np.random.default_rng(_clip_seed)
            by_label = dict()
            for label, ap in candidates:
                by_label.setdefault(label, []).append(ap)
            for paths_list in by_label.values():
                rng.shuffle(paths_list)

            budget = len(candidates) if (max_samples is None) else min(int(max_samples), len(candidates))
            selected = []
            leftovers = []
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

            represented_species = len(set(label for label, _ap in selected))
            print(
                "  Candidate files=", len(candidates),
                "selected=", len(selected),
                "budget=", budget,
                "represented_species=", represented_species,
                "clip_seed=", _clip_seed,
            )

        # Decode selected audio (deterministic aug seed=42) or custom build_features.
        load_rng = np.random.default_rng(42)
        X_items, y_items = [], []
        for i, (label, ap) in enumerate(selected, start=1):
            try:
                if aug_preset:
                    mel = _load_focal_mel(ap, sample_rate, clip_seconds, n_mels, n_frames, cfg, paths, load_rng)
                else:
                    mel = mel_fn(ap, sample_rate, clip_seconds, n_mels, n_frames)
            except Exception:
                continue
            yv = np.zeros(len(species_cols), dtype=np.float32)
            yv[sp2i[label]] = 1.0
            X_items.append(mel)
            y_items.append(yv)
            if len(X_items) % 500 == 0:
                print(f"  Loaded {{len(X_items)}}/{{len(selected)}} selected samples...")

        if not X_items:
            raise RuntimeError("No samples loaded.")
        X_all = np.stack(X_items, dtype=np.float32)
        y_all = np.stack(y_items, dtype=np.float32)

        if _focal_cache_file is not None and aug_preset:
            _focal_cache_file.parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            _manifest = {{
                "aug_preset": aug_preset,
                "max_samples": max_samples,
                "n_mels": n_mels,
                "n_frames": n_frames,
                "represented_species": represented_species,
            }}
            np.savez_compressed(
                str(_focal_cache_file),
                X_train=X_all,
                y_train=y_all,
                manifest=_json.dumps(_manifest),
            )
            print(f"FOCAL_TRAIN_CACHE: saved {{_focal_cache_file.name}} clips={{len(X_items)}}")

    # 4) Train on all focal clips. Selection/evaluation is done externally
    # on labeled train_soundscapes via _append_eval_wrapper.
    split_mode = "soundscape_eval_only"
    val_split = 0.0
    X_train, y_train = X_all, y_all
    X_val, y_val = X_all, y_all

    print(
        f"DATA: X_train={{X_train.shape}}, y_train={{y_train.shape}}, "
        f"X_val={{X_val.shape}}, y_val={{y_val.shape}}, val_split={{val_split}}, split_mode={{split_mode}}"
    )
    print(
        f"AUG_CFG: preset={{aug_preset}} prob={{aug_prob}} noise_std={{aug_noise_std}} "
        f"time_mask={{aug_time_mask}} freq_mask={{aug_freq_mask}}"
    )
    print(
        f"PHASE3_DEBUG: split_stats train_n={{len(X_train)}} val_n={{len(X_val)}} "
        f"unique_species_selected={{represented_species}}"
    )

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

    X_train_fit, aug_changed = _apply_training_augmentation(
        X_train, aug_prob, aug_noise_std, aug_time_mask, aug_freq_mask, seed=42
    )
    print(f"AUG_APPLIED: changed={{aug_changed}}/{{len(X_train)}}", flush=True)

    model = build_model(X_train.shape[1:], len(species_cols))
    model.compile(optimizer=_opt, loss="binary_crossentropy")
    model.summary()
    has_validation = False
    print(f"PHASE3_DEBUG: has_validation={{has_validation}}", flush=True)
    _n_train = len(X_train_fit)
    _fit_bs = max(1, min(int(batch_size), _n_train))
    _steps_per_epoch = max(1, (_n_train + _fit_bs - 1) // _fit_bs)
    print(
        f"TRAIN_START: samples={{_n_train}} batch_size={{_fit_bs}} "
        f"steps_per_epoch={{_steps_per_epoch}} epochs={{epochs}}",
        flush=True,
    )
    fit_kwargs = {{}}
    callbacks = []
    train_start_ts = time.time()
    epoch_start_ts = [None]
    best_val_loss = [None]
    best_epoch = [None]

    def _on_epoch_begin(epoch, logs=None):
        epoch_start_ts[0] = time.time()

    def _on_epoch_end(epoch, logs=None):
        elapsed_total = time.time() - train_start_ts
        epoch_elapsed = (time.time() - epoch_start_ts[0]) if epoch_start_ts[0] else 0.0
        done = epoch + 1
        avg_per_epoch = elapsed_total / max(done, 1)
        eta = max(0.0, avg_per_epoch * max(epochs - done, 0))
        _loss = logs.get("loss") if logs else None
        _loss_s = f" loss={{float(_loss):.6f}}" if _loss is not None else ""
        print(
            f"TRAIN_HEARTBEAT: epoch={{done}}/{{epochs}} "
            f"epoch_time_s={{epoch_elapsed:.1f}} "
            f"elapsed_s={{elapsed_total:.1f}} eta_s={{eta:.1f}}{{_loss_s}}",
            flush=True,
        )
        if logs is not None and ("val_loss" in logs) and (logs["val_loss"] is not None):
            cur_val = float(logs["val_loss"])
            if best_val_loss[0] is None or cur_val < best_val_loss[0]:
                best_val_loss[0] = cur_val
                best_epoch[0] = done

    def _on_batch_end(batch, logs=None):
        if logs is None:
            return
        _bl = logs.get("loss")
        if _bl is not None and (batch == 0 or (batch + 1) % max(1, _steps_per_epoch) == 0):
            print(
                f"TRAIN_BATCH: batch={{batch + 1}} loss={{float(_bl):.6f}}",
                flush=True,
            )

    callbacks.append(tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=_on_epoch_begin,
        on_epoch_end=_on_epoch_end,
        on_train_batch_end=_on_batch_end,
    ))
    _ckpt = None
    if checkpoint_best and has_validation:
        _ckpt = Path("logs") / f"best_{Path(__file__).stem}.keras"
        print(f"PHASE3_DEBUG: checkpoint_enabled=True checkpoint_path={{_ckpt}}")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(_ckpt),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
    else:
        print("PHASE3_DEBUG: checkpoint_enabled=False")
    history = model.fit(
        X_train_fit, y_train,
        epochs=epochs,
        batch_size=_fit_bs,
        verbose=2,
        callbacks=callbacks,
        **fit_kwargs,
    )
    print("TRAIN_DONE", flush=True)
    if best_epoch[0] is not None:
        print(f"BEST_EPOCH_BY_VAL_LOSS: epoch={{best_epoch[0]}}/{{epochs}} val_loss={{best_val_loss[0]:.6f}}")
    else:
        print("BEST_EPOCH_BY_VAL_LOSS: unavailable (no validation metrics).")
    print(f"TRAIN_LOSS: {{history.history['loss'][-1]:.6f}}")
{save_block}
    y_pred = model.predict(X_train, verbose=0).astype(np.float32)
    mean_pred = y_pred.mean(axis=0)
    sub_df = sample_sub.copy()
    for ic, col in enumerate(species_cols):
        sub_df[col] = float(mean_pred[ic])
    sub_path = Path("submission/submission_generated.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    print(f"SUBMISSION: {{sub_path}}")

if __name__ == "__main__":
    main()
"""


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRIC CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_filters_list(depth, filters_base, filter_pattern):
    filters = []
    for i in range(depth):
        if filter_pattern == "constant":
            f = filters_base
        elif filter_pattern == "doubling":
            f = filters_base * (2 ** min(i, 4))
        else:
            f = filters_base
        filters.append(min(f, 512))
    return filters


def _compute_max_pools(depth, n_mels, n_frames):
    min_spatial = min(n_mels, n_frames)
    if min_spatial < 4:
        return 0
    return min(depth, max(1, int(math.log2(min_spatial)) - 1))


def generate_slot_code(params):
    """Generate get_training_config() + build_model() from a parameter dict."""
    d = params
    filters_list = _compute_filters_list(d["depth"], d["filters_base"], d["filter_pattern"])
    max_pools = _compute_max_pools(d["depth"], d["n_mels"], d["n_frames"])
    auto_residuals = d["depth"] >= 6
    force_residuals = d.get("residuals", None)
    use_residuals = auto_residuals if force_residuals is None else bool(force_residuals)

    return f'''def get_training_config():
    return {{
        "max_samples": {d.get("max_samples", 1500)},
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": {d["n_mels"]},
        "n_frames": {d["n_frames"]},
        "epochs": {d.get("epochs", 3)},
        "batch_size": {d["batch_size"]},
        "learning_rate": {d["learning_rate"]},
        "optimizer": "{d["optimizer"]}",
        "val_split": {d.get("val_split", 0.2)},
        "weight_decay": {d.get("weight_decay", 0.0)},
        "classifier_hidden_units": {d.get("classifier_hidden_units", 0)},
        "pooling_type": "{d.get("pooling_type", "global_avg")}",
        "aug_prob": {d.get("aug_prob", 0.0)},
        "aug_noise_std": {d.get("aug_noise_std", 0.0)},
        "aug_time_mask": {d.get("aug_time_mask", 0)},
        "aug_freq_mask": {d.get("aug_freq_mask", 0)},
        "aug_preset": {repr(d.get("aug_preset"))},
    }}


def build_model(input_shape, num_classes):
    import tensorflow as tf
    filters_list = {filters_list}
    max_pools = {max_pools}
    use_batch_norm = {d["batch_norm"]}
    use_residuals = {use_residuals}
    dropout_rate = {d["dropout"]}
    weight_decay = {d.get("weight_decay", 0.0)}
    classifier_hidden_units = {d.get("classifier_hidden_units", 0)}
    pooling_type = "{d.get("pooling_type", "global_avg")}"
    reg = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate(filters_list):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=reg)(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if i < max_pools:
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        if use_residuals and i > 0:
            sc_filters = tf.keras.backend.int_shape(x)[-1]
            shortcut = tf.keras.layers.Conv2D(sc_filters, (1, 1), padding="same")(shortcut)
            if i < max_pools:
                shortcut = tf.keras.layers.MaxPooling2D((2, 2))(shortcut)
            if tf.keras.backend.int_shape(x) == tf.keras.backend.int_shape(shortcut):
                x = tf.keras.layers.Add()([x, shortcut])
    if pooling_type == "global_avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling_type == "global_max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.Flatten()(x)
    if classifier_hidden_units and classifier_hidden_units > 0:
        x = tf.keras.layers.Dense(classifier_hidden_units, activation="relu", kernel_regularizer=reg)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=reg)(x)
    model = tf.keras.Model(inputs, x)
    return model
'''


def describe_params(params):
    parts = [
        f"depth={params.get('depth','?')}",
        f"filt={params.get('filters_base','?')}-{params.get('filter_pattern','?')}",
        f"lr={params.get('learning_rate','?')}",
        f"wd={params.get('weight_decay','?')}",
        f"hid={params.get('classifier_hidden_units','?')}",
        f"res={params.get('residuals','auto')}",
        str(params.get("optimizer", "?")),
        f"pool={params.get('pooling_type','?')}",
        f"drop={params.get('dropout','?')}",
        f"BN={'Y' if params.get('batch_norm') else 'N'}",
        f"mels={params.get('n_mels','?')}",
        f"frames={params.get('n_frames','?')}",
        f"bs={params.get('batch_size','?')}",
        f"aug_preset={params.get('aug_preset')}",
        f"aug_p={params.get('aug_prob', 0.0)}",
        f"aug_n={params.get('aug_noise_std', 0.0)}",
        f"aug_tm={params.get('aug_time_mask', 0)}",
        f"aug_fm={params.get('aug_freq_mask', 0)}",
    ]
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _truncate(text, max_chars=4000):
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."

def _save_json(data, path):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def _load_json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_saved_search_state(logs_dir):
    linear_path = logs_dir / "linear.json"
    linear_best_path = logs_dir / "linear_best_params.json"
    random_path = logs_dir / "random_results.json"
    missing = [str(p) for p in (linear_path, linear_best_path, random_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot resume from medium: missing saved search files: "
            + ", ".join(missing)
        )

    lin_results = _load_json_file(linear_path)
    lin_best = _load_json_file(linear_best_path)
    rnd_results = _load_json_file(random_path)
    if not isinstance(lin_results, list) or not isinstance(rnd_results, list) or not isinstance(lin_best, dict):
        raise ValueError("Saved search files have an unexpected format; refusing to resume from medium.")

    rnd_ok = [r for r in rnd_results if r.get("success") and _result_rank_value(r) >= 0]
    best_rnd = max(rnd_ok, key=_result_rank_value) if rnd_ok else None
    return lin_results, lin_best, rnd_results, best_rnd

def _fmt_score(v):
    return f"{v:.6f}" if isinstance(v, (int, float)) else "N/A"


def _ranking_metric_from_config(config: dict | None) -> str:
    if not config:
        return PRIMARY_META_METRIC
    return str(config.get("meta_agent", {}).get("primary_metric", PRIMARY_META_METRIC))


def _ranking_value_from_metrics(metrics: dict | None) -> float | None:
    if not metrics or metrics.get("status") != "success":
        return None
    if metrics.get("ranking_value") is not None:
        return float(metrics["ranking_value"])
    key = metrics.get("ranking_metric", PRIMARY_META_METRIC)
    if key == PRIMARY_META_METRIC:
        v = metrics.get("macro_average_precision")
        if v is not None:
            return float(v)
    v = metrics.get("macro_roc_auc")
    return float(v) if v is not None else None


def _fmt_experiment_metrics(metrics: dict | None) -> str:
    return format_metrics_dict(metrics, ranking_metric=PRIMARY_META_METRIC)


def _search_result_entry(
    run_id: str,
    search_type: str,
    params: dict,
    metrics: dict | None,
    attempts: int,
    description: str,
    slot_code: str,
) -> dict:
    rv = _ranking_value_from_metrics(metrics)
    return {
        "run_id": run_id,
        "search_type": search_type,
        "params": params,
        "success": rv is not None,
        "attempts": attempts,
        "description": description,
        "slot_code": slot_code,
        "ranking_metric": (metrics or {}).get("ranking_metric", PRIMARY_META_METRIC),
        "ranking_value": rv,
        "macro_roc_auc": (metrics or {}).get("macro_roc_auc"),
        "macro_average_precision": (metrics or {}).get("macro_average_precision"),
        "competition_macro_auc_v2": (metrics or {}).get("competition_macro_auc_v2"),
        "median_per_class_auc": (metrics or {}).get("median_per_class_auc"),
    }


def _result_rank_value(entry: dict) -> float:
    v = entry.get("ranking_value")
    if v is not None:
        return float(v)
    if entry.get("macro_average_precision") is not None:
        return float(entry["macro_average_precision"])
    if entry.get("macro_roc_auc") is not None:
        return float(entry["macro_roc_auc"])
    return -1.0


def _coverage_adjusted_score_agent(macro_auc: float, scored_columns: int, total_classes: int = 234) -> float:
    denom = max(int(total_classes), 1)
    coverage = max(0.0, min(1.0, float(scored_columns) / float(denom)))
    return float(macro_auc) * coverage


def _phase1_selection_score(macro_auc: float | None, num_scored: int | None, total: int = 234) -> float:
    if macro_auc is None:
        return -1.0
    if num_scored is None:
        return float(macro_auc)
    return _coverage_adjusted_score_agent(float(macro_auc), int(num_scored), total_classes=total)


def _best_params_from_transfer_slot(slot_code: str) -> dict:
    ns: dict = {}
    exec(slot_code, ns)
    tc = ns.get("get_training_config", lambda: {})() or {}
    bp = dict(DEFAULT_PARAMS)
    if isinstance(tc, dict):
        for k in (
            "n_mels",
            "n_frames",
            "batch_size",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "val_split",
        ):
            if k in tc:
                bp[k] = tc[k]
    return bp


def _final_config_override_block(base_cfg):
    safe_cfg = dict(base_cfg)
    return (
        "\n\n# --- FINAL OVERRIDE: force final training config ---\n"
        "def get_training_config():\n"
        f"    return {repr(safe_cfg)}\n"
    )

def assemble_script(
    slot_code,
    *,
    is_final: bool = False,
    model_save_path: str = "",
    focal_cache_dir: Path | str | None = None,
):
    focal_line = ""
    if focal_cache_dir is not None:
        focal_line = f'\n_FOCAL_MEL_CACHE_DIR = r"{Path(focal_cache_dir).resolve()}"\n'
    suffix = _make_harness_suffix(is_final=is_final, model_save_path=model_save_path)
    return f"{HARNESS_PREFIX}{focal_line}\n\n# --- GENERATED MODEL CODE ---\n{slot_code.strip()}\n\n{suffix}\n"

def _append_eval_wrapper(script, run_id, eval_dir, mel_cache_dir=None):
    yt = eval_dir / f"y_true_{run_id}.npy"
    yp = eval_dir / f"y_pred_{run_id}.npy"
    cache_dir = Path(mel_cache_dir) if mel_cache_dir is not None else DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    cache_dir_str = str(cache_dir.resolve())
    return script + f"""

import numpy as _np
import pandas as _pd
import tensorflow as _tf

_cfg = get_training_config()
_sr = int(_cfg.get("sample_rate", 32000))
_clip_seconds = float(_cfg.get("clip_seconds", 5.0))
_n_mels = int(_cfg.get("n_mels", 64))
_n_frames = int(_cfg.get("n_frames", 128))
_n_fft = int(_cfg.get("n_fft", 1024))
_hop_length = int(_cfg.get("hop_length", 512))

_paths = resolve_birdclef_paths()
_tables = load_core_tables(_paths)
if "train_soundscapes_labels" not in _tables:
    raise RuntimeError("Missing train_soundscapes_labels.csv for soundscape evaluation.")
if "sample_submission" not in _tables:
    raise RuntimeError("Missing sample_submission.csv for soundscape evaluation.")

_sample_sub = _tables["sample_submission"]
_species_cols = species_columns_from_sample_submission(_sample_sub)
_sp2i = {{s: i for i, s in enumerate(_species_cols)}}

_labels = _tables["train_soundscapes_labels"].copy()
if _labels.empty:
    raise RuntimeError("train_soundscapes_labels.csv is empty.")

def _tok_labels(_v):
    if _pd.isna(_v) or _v == "":
        return set()
    return {{_t.strip() for _t in str(_v).split(";") if _t.strip()}}

def _merge_label_sets(_s):
    _out = set()
    for _v in _s:
        _out |= _tok_labels(_v)
    return _out

_grp = (
    _labels.groupby(["filename", "start", "end"], sort=False)["primary_label"]
    .agg(_merge_label_sets)
    .reset_index()
)
_grp["end_sec"] = _pd.to_timedelta(_grp["end"]).dt.total_seconds().astype(int)
_grp["row_id"] = _grp["filename"].str.replace(".ogg", "", regex=False) + "_" + _grp["end_sec"].astype(str)

_rows = []
for _r in _grp.itertuples(index=False):
    _vec = _np.zeros(len(_species_cols), dtype=_np.float32)
    for _code in _r.primary_label:
        _j = _sp2i.get(_code)
        if _j is not None:
            _vec[_j] = 1.0
    _rows.append((_r.row_id, _vec))

_y_true_df = _pd.DataFrame(
    [_v for _rid, _v in _rows],
    index=[_rid for _rid, _v in _rows],
    columns=_species_cols,
).sort_index()
if not _y_true_df.index.is_unique:
    _y_true_df = _y_true_df.groupby(level=0).max()
_y_true_df.index.name = "row_id"

_required_stems = {{_rid.rsplit("_", 1)[0] for _rid in _y_true_df.index}}
_ogg_files = [
    _paths.train_soundscapes_dir / f"{{_stem}}.ogg"
    for _stem in sorted(_required_stems)
    if (_paths.train_soundscapes_dir / f"{{_stem}}.ogg").exists()
]
if not _ogg_files:
    raise RuntimeError(f"No labeled train soundscapes found in {{_paths.train_soundscapes_dir}}.")

_window_samples = int(round(_sr * _clip_seconds))
_target_windows = int(round(60.0 / _clip_seconds))
if _target_windows <= 0:
    raise RuntimeError(f"Invalid clip_seconds={{_clip_seconds}} for soundscape evaluation.")
_target_samples = _window_samples * _target_windows

_cache_path = (
    __import__("pathlib").Path(r"{cache_dir_str}")
    / f"soundscape_mels_{{_n_mels}}x{{_n_frames}}_sr{{_sr}}_hop{{_hop_length}}.npz"
)
_pred_rows = []
print("SOUNDSCAPE_EVAL_START", flush=True)
if _cache_path.exists():
    print(f"SOUNDSCAPE_EVAL_CACHE: loading {{_cache_path.name}}", flush=True)
    _cd = _np.load(str(_cache_path), allow_pickle=True)
    _X_all = _cd["X_mels"].astype(_np.float32)
    _cached_ids = [str(x) for x in _cd["row_ids"]]
    _bs = max(1, int(_cfg.get("batch_size", 32)))
    _pred_chunks = []
    for _si in range(0, len(_X_all), _bs):
        _pred_chunks.append(model.predict(_X_all[_si : _si + _bs], verbose=0).astype(_np.float32))
    _preds_all = _np.concatenate(_pred_chunks, axis=0) if _pred_chunks else _np.zeros((0, len(_species_cols)), dtype=_np.float32)
    for _rid, _pv in zip(_cached_ids, _preds_all):
        _row = {{"row_id": _rid}}
        for _col, _p in zip(_species_cols, _pv):
            _row[_col] = float(_p)
        _pred_rows.append(_row)
else:
    _cache_X = []
    _cache_ids = []
    for _fi, _fpath in enumerate(_ogg_files, start=1):
        if (_fi % 10) == 0:
            print(f"SOUNDSCAPE_EVAL_PROGRESS: {{_fi}}/{{len(_ogg_files)}} files")
        _name = _fpath.stem
        _y_full, _ = librosa.load(str(_fpath), sr=_sr, mono=True)
        if len(_y_full) > _target_samples:
            _y_full = _y_full[:_target_samples]
        elif len(_y_full) < _target_samples:
            _y_full = _np.pad(_y_full, (0, _target_samples - len(_y_full)))

        _batch = _np.zeros((_target_windows, _n_mels, _n_frames, 1), dtype=_np.float32)
        for _wi in range(_target_windows):
            _st = _wi * _window_samples
            _seg = _y_full[_st : _st + _window_samples]
            _mel = librosa.feature.melspectrogram(
                y=_seg,
                sr=_sr,
                n_mels=_n_mels,
                n_fft=_n_fft,
                hop_length=_hop_length,
                power=2.0,
            )
            _mel_db = librosa.power_to_db(_mel, ref=_np.max)
            _mel_r = _tf.image.resize(_mel_db[..., _np.newaxis], (_n_mels, _n_frames)).numpy().astype(_np.float32)
            _batch[_wi] = _mel_r
            _end_sec = int(round((_wi + 1) * _clip_seconds))
            _rid = f"{{_name}}_{{_end_sec}}"
            if _rid in _y_true_df.index:
                _cache_X.append(_mel_r)
                _cache_ids.append(_rid)

        _preds = model.predict(_batch, verbose=0).astype(_np.float32)
        for _wi in range(_target_windows):
            _end_sec = int(round((_wi + 1) * _clip_seconds))
            _row = {{"row_id": f"{{_name}}_{{_end_sec}}" }}
            for _col, _p in zip(_species_cols, _preds[_wi]):
                _row[_col] = float(_p)
            _pred_rows.append(_row)

    if _cache_X:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(
            str(_cache_path),
            X_mels=_np.stack(_cache_X, axis=0).astype(_np.float32),
            row_ids=_np.array(_cache_ids, dtype=object),
        )
        print(f"SOUNDSCAPE_EVAL_CACHE: saved {{_cache_path.name}} windows={{len(_cache_ids)}}")

_pred_df = _pd.DataFrame(_pred_rows)
_pred_renamed = _pred_df.rename(columns={{_c: f"{{_c}}_pred" for _c in _species_cols}})
_merged = _y_true_df.reset_index().merge(_pred_renamed, on="row_id", how="inner")
if _merged.empty:
    raise RuntimeError("No row_id overlap between train_soundscapes_labels and predictions.")

_pred_cols = [f"{{_c}}_pred" for _c in _species_cols]
_yt = _merged[_species_cols].to_numpy(dtype=_np.float32)
_yp = _merged[_pred_cols].to_numpy(dtype=_np.float32)
if _yt.shape != _yp.shape:
    raise RuntimeError(f"Shape mismatch: y_true={{_yt.shape}} vs y_pred={{_yp.shape}}")

print(
    f"SOUNDSCAPE_EVAL_READY: windows={{len(_merged)}} species={{len(_species_cols)}} "
    f"pos_species={{int((_yt.sum(axis=0) > 0).sum())}}"
)
_np.save(r\"{yt}\", _yt)
_np.save(r\"{yp}\", _yp)
print("EVAL_ARTIFACTS_SAVED")
"""

def extract_python_code(response):
    if not response or not response.strip():
        return ""
    blocks = re.findall(r"```python\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[0].strip()
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
    return blocks[0].strip() if blocks else ""

def validate_slot_code(code):
    try:
        tree = ast.parse(code)
    except Exception as e:
        return [str(e)]
    if "build_model" not in code:
        return ["Missing build_model"]
    if "get_training_config" not in code:
        return ["Missing get_training_config"]
    allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
               ast.Assign, ast.AnnAssign, ast.ClassDef)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if not isinstance(node, allowed):
            return ["Top-level executable code not allowed."]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_features":
            if len(node.args.args) != 5:
                return ["build_features must have 5 params."]
    return []

GENERATION_SYSTEM_PROMPT = (
    "You are an ML research assistant for BirdCLEF experiments.\n"
    "Return ONLY one ```python``` code block and nothing else.\n"
    "Rules:\n"
    "- Define exactly get_training_config() and build_model(input_shape, num_classes)\n"
    "- Optional build_features(audio_path, sample_rate, clip_seconds, n_mels, n_frames)\n"
    "- Do NOT define main(), _ORIG_GET_TRAINING_CONFIG, or _META_OVERRIDES\n"
    "- No executable top-level statements except imports/assignments/function defs\n"
    "- Final layer must be Dense(num_classes, activation='sigmoid')\n"
    "- Compile with binary_crossentropy\n"
    "- Keras 3: use Input(shape=...) with Sequential; for residuals use Functional API "
    "(Input → conv branches → Add), never model.add(layers.Add()([shortcut, model])).\n"
)

SAFE_BASELINE_SLOT_CODE = """def get_training_config():
    return {
        "max_samples": 1500,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "val_split": 0.2,
    }


def build_model(input_shape, num_classes):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])
    return model
"""


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER (with LLM auto-fix on failure)
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    slot_code,
    run_id,
    code_dir,
    eval_dir,
    executor,
    evaluator,
    mel_cache_dir=None,
    focal_cache_dir=None,
    *,
    is_final: bool = False,
):
    """Run a single experiment. Returns (metrics_or_None, exec_result)."""
    if focal_cache_dir is None and not is_final:
        focal_cache_dir = DEFAULT_FOCAL_MEL_CACHE_DIR
    script = assemble_script(
        slot_code,
        is_final=is_final,
        focal_cache_dir=focal_cache_dir,
    )
    script = _append_eval_wrapper(script, run_id, eval_dir, mel_cache_dir=mel_cache_dir)
    script_path = code_dir / f"{run_id}.py"
    script_path.write_text(script, encoding="utf-8")
    (code_dir / f"{run_id}_slot.py").write_text(slot_code, encoding="utf-8")

    timeout_s = getattr(executor, "timeout_seconds", None)
    print(
        f"  [CNN Run] {run_id} → training + soundscape eval "
        f"(timeout={timeout_s}s, log stream on)",
        flush=True,
    )
    t0 = time.time()
    result = executor.run_file(script_path, stream_output=True, label=run_id)
    elapsed = time.time() - t0
    if getattr(result, "timed_out", False):
        print(f"  [CNN Run] {run_id} TIMED OUT after {elapsed:.1f}s", flush=True)
    elif result.success:
        print(f"  [CNN Run] {run_id} finished OK in {elapsed:.1f}s", flush=True)
    else:
        print(f"  [CNN Run] {run_id} failed in {elapsed:.1f}s (exit {result.return_code})", flush=True)
    if not result.success:
        return None, result

    yt = eval_dir / f"y_true_{run_id}.npy"
    yp = eval_dir / f"y_pred_{run_id}.npy"
    if yt.exists() and yp.exists():
        ev = evaluator.evaluate_from_files(yt, yp)
        return ev.metrics, result
    return None, result

def _clean_error_text(stderr: str) -> str:
    if not stderr:
        return ""
    lines = stderr.splitlines()
    filtered = [ln for ln in lines if "NotOpenSSLWarning" not in ln and "urllib3/__init__.py:35" not in ln]
    text = "\n".join(filtered).strip()
    return text if text else stderr.strip()


def _extract_best_epoch(stdout: str) -> int | None:
    if not stdout:
        return None
    m = re.search(r"BEST_EPOCH_BY_VAL_LOSS:\s*epoch=(\d+)/\d+", stdout)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def run_experiment_until_success(
    slot_code,
    run_id,
    code_dir,
    eval_dir,
    executor,
    evaluator,
    llm,
    temperature,
    max_attempts=5,
    use_llm_fixes=True,
    mel_cache_dir=None,
    focal_cache_dir=None,
    reapply_overrides=None,
):
    """Run experiment, auto-fix with LLM on failure.

    ``reapply_overrides`` (optional) is a callable ``str -> str`` that
    re-injects the meta-agent's locked training overrides (locked
    augmentation, mel shape, max_samples, etc.) into a fresh slot. When the
    LLM returns a fix it usually emits only ``get_training_config`` +
    ``build_model`` with no ``_META_OVERRIDES`` block, which silently
    overrode locked augmentation on every retry. Always passing the LLM
    fix through this callback before using it keeps the lock in place.
    """
    current_slot = slot_code
    for attempt in range(1, max_attempts + 1):
        metrics, result = run_experiment(
            current_slot,
            f"{run_id}_a{attempt}",
            code_dir,
            eval_dir,
            executor,
            evaluator,
            mel_cache_dir=mel_cache_dir,
            focal_cache_dir=focal_cache_dir,
        )
        if metrics and metrics.get("status") == "success":
            best_epoch = _extract_best_epoch(result.stdout or "")
            if best_epoch is not None:
                metrics = dict(metrics)
                metrics["best_epoch"] = best_epoch
            if attempt > 1:
                print(f"    Fixed after {attempt} attempts")
            return current_slot, metrics, result, attempt

        if getattr(result, "timed_out", False):
            error_text = result.stderr or f"Timed out after {getattr(executor, 'timeout_seconds', '?')}s"
        elif not result.success:
            error_text = _clean_error_text(result.stderr or result.stdout)
        else:
            error_text = "No valid metrics produced (check EVAL_ARTIFACTS_SAVED in log)."
        err_path = code_dir / f"{run_id}_a{attempt}_stderr.log"
        err_path.write_text((result.stderr or "") + "\n--- stdout ---\n" + (result.stdout or ""), encoding="utf-8")
        if attempt == max_attempts:
            print(f"    Failed all {max_attempts} attempts. Error: {_truncate(error_text, 200)}")
            return current_slot, None, result, attempt

        if not use_llm_fixes:
            print(f"    Attempt {attempt} failed (LLM fixes disabled for this run).")
            return current_slot, None, result, attempt

        llm_timeout = getattr(llm, "timeout_seconds", 600)
        print(
            f"    Attempt {attempt} failed — requesting LLM fix "
            f"(coder timeout={llm_timeout:.0f}s)…",
            flush=True,
        )
        if result.stdout and "TRAIN_HEARTBEAT" not in result.stdout and "TRAIN_START" not in result.stdout:
            print("    (no TRAIN_* lines in log — failure likely before/during setup)", flush=True)
        fix_prompt = (
            f"The model code failed.\nError:\n{_truncate(error_text, 3000)}\n\n"
            f"Current code:\n```python\n{current_slot}\n```\n\n"
            "Fix build_model() and/or get_training_config(). Return ONLY one ```python``` block "
            "with those two functions. Do NOT include _ORIG_GET_TRAINING_CONFIG, "
            "_META_OVERRIDES, harness code, or main(). "
            "For residual blocks use Functional API, not Add() on a Sequential model."
        )
        t_fix = time.time()
        resp = llm.generate_from_messages(
            messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                      {"role": "user", "content": fix_prompt}],
            temperature=temperature,
        )
        print(f"    [LLM fix] done in {time.time() - t_fix:.1f}s", flush=True)
        candidate = extract_python_code(resp)
        issues = validate_slot_code(candidate) if candidate else ["No code found."]
        if not issues:
            # The LLM almost always returns a bare get_training_config + build_model
            # with no _META_OVERRIDES block. If we just took the candidate as-is,
            # locked augmentation / mel shape / max_samples would silently revert
            # to whatever the LLM chose — that's the "stage 1b changes aug" bug.
            if reapply_overrides is not None:
                try:
                    candidate = reapply_overrides(candidate)
                except Exception as exc:
                    print(f"    [Warning] reapply_overrides failed: {exc!r} — using raw LLM fix")
            current_slot = candidate
        else:
            print(f"    LLM fix invalid: {issues}")

    return current_slot, None, result, max_attempts


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 0: BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def _run_baseline(executor, evaluator, llm, temperature, dirs, search_cfg):
    print("\n" + "=" * 60)
    print("  PHASE 0: BASELINE VERIFICATION")
    print("=" * 60)

    params = dict(DEFAULT_PARAMS)
    params["max_samples"] = search_cfg["cheap"]["max_samples"]
    params["epochs"] = search_cfg["cheap"]["epochs"]
    params["val_split"] = search_cfg["cheap"].get("val_split", 0.2)
    slot_code = generate_slot_code(params)

    # Stage A: deterministic generated baseline, no LLM fixes.
    slot_code, metrics, result, attempts = run_experiment_until_success(
        slot_code, "baseline_gen", dirs["baseline_codes"], dirs["eval"],
        executor, evaluator, llm, temperature, max_attempts=1, use_llm_fixes=False,
    )

    # Stage B: hardcoded safe baseline, no LLM fixes.
    if not (metrics and metrics.get("status") == "success"):
        print("  Generated baseline failed. Trying safe baseline template...")
        slot_code, metrics, result, attempts_b = run_experiment_until_success(
            SAFE_BASELINE_SLOT_CODE, "baseline_safe", dirs["baseline_codes"], dirs["eval"],
            executor, evaluator, llm, temperature, max_attempts=1, use_llm_fixes=False,
        )
        attempts += attempts_b

    # Stage C: only then allow iterative LLM fixes.
    if not (metrics and metrics.get("status") == "success"):
        print("  Safe baseline failed. Falling back to LLM auto-fix loop...")
        slot_code, metrics, result, attempts_c = run_experiment_until_success(
            slot_code, "baseline_fix", dirs["baseline_codes"], dirs["eval"],
            executor, evaluator, llm, temperature, max_attempts=5, use_llm_fixes=True,
        )
        attempts += attempts_c

    auc = metrics["macro_roc_auc"] if metrics and metrics.get("status") == "success" else None
    entry = {"params": params, "score": auc, "status": "success" if auc else "failed", "attempts": attempts,
             "description": describe_params(params)}
    _save_json(entry, dirs["logs"] / "baseline_result.json")
    print(f"\n  Baseline AUC = {_fmt_score(auc)} (attempts={attempts})")
    return slot_code, entry


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: LINEAR SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def _compute_zoom_values(dim, winner):
    coarse = dim["coarse"]
    dt = dim["type"]
    if dt in ("bool", "categorical") or not dim.get("zoom"):
        return []
    idx = coarse.index(winner) if winner in coarse else 0
    lo = coarse[max(0, idx - 1)]
    hi = coarse[min(len(coarse) - 1, idx + 1)]
    if lo == hi:
        return []
    if dt == "int":
        step = max(1, (int(hi) - int(lo)) // 5)
        return [v for v in range(int(lo), int(hi) + 1, step) if v != winner and v not in coarse][:5]
    elif dt == "float":
        return [round(v, 3) for v in np.linspace(lo, hi, 6) if abs(v - winner) > 0.01][:4]
    elif dt == "log_float":
        cands = np.logspace(np.log10(max(lo, 1e-8)), np.log10(max(hi, 1e-8)), 6)
        return [round(v, 7) for v in cands if abs(v - winner) / max(winner, 1e-10) > 0.05][:4]
    return []


def _run_linear_search(baseline_slot, executor, evaluator, llm, temperature, dirs, search_cfg):
    print("\n" + "=" * 60)
    print("  PHASE 1: LINEAR SEARCH (coordinate descent)")
    print("=" * 60)

    budget = search_cfg.get("linear_budget", 75)
    s, e = search_cfg["cheap"]["max_samples"], search_cfg["cheap"]["epochs"]
    vs = search_cfg["cheap"].get("val_split", 0.2)
    current_best = dict(DEFAULT_PARAMS)
    all_results = []
    rc = 0
    cf = 0

    for dim in SEARCH_DIMENSIONS:
        dn = dim["name"]
        if rc >= budget:
            break
        print(f"\n  ── Sweeping: {dn} ({dim['coarse']}) ──")
        dim_results = []

        for val in dim["coarse"]:
            if rc >= budget:
                break
            rc += 1
            p = dict(current_best); p[dn] = val; p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
            slot = generate_slot_code(p)
            rid = f"L{rc:03d}_{dn}"
            slot, metrics, result, att = run_experiment_until_success(
                slot, rid, dirs["linear_codes"], dirs["eval"], executor, evaluator, llm, temperature)
            entry = _search_result_entry(
                rid, "coarse",
                {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
                metrics, att, describe_params(p), slot,
            )
            entry["dimension"] = dn
            entry["value"] = val
            all_results.append(entry)
            dim_results.append(entry)
            print(f"    {dn}={val} → {_fmt_experiment_metrics(metrics)} (att={att})")
            if not entry["success"]:
                cf += 1
                if cf >= 5: print(f"\n  ⚠ {cf} consecutive failures on {dn}={val}")
            else:
                cf = 0

        ok = [r for r in dim_results if r["success"]]
        if not ok:
            print(f"  No successful runs for {dn}. Keeping default={current_best[dn]}")
            continue
        coarse_best = max(ok, key=_result_rank_value)
        cbv = coarse_best["value"]
        print(
            f"  Coarse winner: {dn}={cbv} "
            f"(AP={coarse_best.get('macro_average_precision', 'N/A')}, "
            f"AUC={coarse_best.get('competition_macro_auc_v2') or coarse_best.get('macro_roc_auc', 'N/A')})"
        )

        zoom_vals = _compute_zoom_values(dim, cbv)
        if zoom_vals:
            print(f"  Zooming: {zoom_vals}")
            for val in zoom_vals:
                if rc >= budget:
                    break
                rc += 1
                p = dict(current_best); p[dn] = val; p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
                slot = generate_slot_code(p)
                rid = f"L{rc:03d}_{dn}_zoom"
                slot, metrics, result, att = run_experiment_until_success(
                    slot, rid, dirs["linear_codes"], dirs["eval"], executor, evaluator, llm, temperature)
                auc = metrics["macro_roc_auc"] if metrics and metrics.get("status") == "success" else None
                entry = {"run_id": rid, "dimension": dn, "value": val, "search_type": "zoom",
                         "params": {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
                         "macro_roc_auc": auc, "success": auc is not None, "attempts": att,
                         "best_epoch": metrics.get("best_epoch") if metrics else None,
                         "description": describe_params(p)}
                all_results.append(entry); dim_results.append(entry)
                print(f"    {dn}={val} (zoom) → {_fmt_score(auc)}")
                cf = 0 if entry["success"] else cf + 1

        all_ok = [r for r in dim_results if r["success"]]
        if all_ok:
            winner = max(all_ok, key=lambda r: r["macro_roc_auc"])
            current_best[dn] = winner["value"]
            print(f"  ★ Locked {dn}={winner['value']} (AUC={winner['macro_roc_auc']:.6f})")

    _save_json(all_results, dirs["logs"] / "linear.json")
    _save_json(current_best, dirs["logs"] / "linear_best_params.json")
    tok = sum(1 for r in all_results if r["success"])
    print(f"\n  Linear done: {rc} iters, {tok} successful")
    print(f"  Best: {describe_params(current_best)}")
    return all_results, current_best


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: RANDOM SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def _locked_cnn_aug(config: dict) -> dict:
    """Fixed spectrogram aug knobs when meta-agent locks a baseline preset."""
    return dict(config.get("cnn_augmentation") or {})


def _apply_locked_aug(params: dict, config: dict) -> dict:
    if not config.get("lock_augmentation"):
        return params
    locked = _locked_cnn_aug(config)
    if not locked:
        return params
    out = dict(params)
    out.update(locked)
    return out


def _sample_random_params(rng, config: dict | None = None):
    config = config or {}
    p = {
        "depth": int(rng.choice([1, 2, 3, 4, 6, 8, 12, 15, 20])),
        "filters_base": int(rng.choice([8, 16, 32, 64, 128])),
        "filter_pattern": "doubling",
        "learning_rate": float(rng.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4])),
        "optimizer": "adam",
        "dropout": 0.0,
        "batch_norm": True,
        "residuals": bool(rng.choice([True, False])),
        "weight_decay": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
        "classifier_hidden_units": int(rng.choice([0, 128, 256, 512])),
        "pooling_type": "global_avg",
        "batch_size": 32,
        "n_mels": 64,
        "n_frames": 128,
        "aug_prob": 0.0,
        "aug_noise_std": 0.0,
        "aug_time_mask": 0,
        "aug_freq_mask": 0,
    }
    return _apply_locked_aug(p, config)

def _tweak_params(base, rng, config: dict | None = None):
    config = config or {}
    p = dict(base)
    dims = rng.choice(["depth", "filters_base", "learning_rate", "weight_decay", "classifier_hidden_units", "residuals"],
                       size=int(rng.choice([1,2])), replace=False)
    for d in dims:
        if d == "depth":
            p["depth"] = int(rng.choice([1, 2, 3, 4, 6, 8, 12, 15, 20]))
        elif d == "filters_base":
            p["filters_base"] = int(rng.choice([8, 16, 32, 64, 128]))
        elif d == "learning_rate":
            p["learning_rate"] = float(rng.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
        elif d == "weight_decay":
            p["weight_decay"] = float(rng.choice([0.0, 1e-4, 1e-3, 1e-2]))
        elif d == "classifier_hidden_units":
            p["classifier_hidden_units"] = int(rng.choice([0, 128, 256, 512]))
        elif d == "residuals":
            p["residuals"] = bool(rng.choice([True, False]))
    return _apply_locked_aug(p, config)

def _tweak_augmentation_params(base, rng):
    p = dict(base)
    dims = rng.choice(
        ["aug_prob", "aug_noise_std", "aug_time_mask", "aug_freq_mask"],
        size=int(rng.choice([1, 2])),
        replace=False,
    )
    for d in dims:
        if d == "aug_prob":
            p["aug_prob"] = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
        elif d == "aug_noise_std":
            p["aug_noise_std"] = float(rng.choice([0.0, 0.003, 0.007, 0.015]))
        elif d == "aug_time_mask":
            p["aug_time_mask"] = int(rng.choice([0, 8, 16, 24]))
        elif d == "aug_freq_mask":
            p["aug_freq_mask"] = int(rng.choice([0, 4, 8, 12]))
    return p

def _researcher_analysis(llm_researcher, results, temperature=0.6):
    """Researcher LLM (e.g. DeepSeek R1) analyzes experiment history.
    Produces a short text research direction — no code, just reasoning.
    Returns empty string if not enough data or researcher not configured.
    """
    if llm_researcher is None:
        return ""
    ok = [r for r in results if r.get("success") and r.get("macro_roc_auc")]
    if len(ok) < 3:
        return ""
    ok.sort(key=lambda r: r["macro_roc_auc"], reverse=True)
    lines = []
    for i, r in enumerate(ok[:15]):
        lines.append(
            f"{i+1}. AUC={r['macro_roc_auc']:.6f} | {r.get('description', 'unknown')}"
        )
    prompt = (
        "You are a deep learning research expert analyzing BirdCLEF 2026 audio classification "
        "experiments (234 species, macro ROC-AUC metric, mel-spectrogram input).\n\n"
        "IMPORTANT CONTEXT — scale sensitivity:\n"
        "All experiments so far were run on a small search budget (max 2000 samples, often fewer due to label filtering). "
        "The final model will be trained on ~35,000 samples. "
        "This means:\n"
        "- Augmentation results are UNRELIABLE at small scale. Even if augmentation "
        "showed little benefit here, it almost certainly helps at full scale. "
        "Always recommend strong augmentation for the final run regardless of what the "
        "search results suggest.\n"
        "- Architecture choices (depth, filters, residuals) transfer more reliably from "
        "small to large scale.\n\n"
        "Experiment results so far (best first):\n"
        + "\n".join(lines)
        + "\n\n"
        "Provide a concise research direction (3-5 sentences):\n"
        "1. What pattern do you see in what works vs. what does not?\n"
        "2. What ONE specific architectural or hyperparameter change should be tried next and why?\n"
        "3. Is there a clear ceiling being hit that suggests a fundamentally different approach?\n"
        "4. What augmentation strategy should be used in the final run (remember: small-scale "
        "augmentation results are not reliable)?\n\n"
        "Think step by step. No code — just your reasoning."
    )
    print("  [Researcher] Analyzing experiment history...")
    hint = llm_researcher.generate_from_messages(
        messages=[
            {"role": "system", "content": "You are a deep learning research expert. Think step by step. Be concise and specific."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    if hint and not hint.startswith("Error"):
        print(f"  [Researcher] Direction: {hint[:200]}...")
    return hint or ""


def _analyze_random_results(results, llm, temperature):
    ok = [r for r in results if r.get("success") and _result_rank_value(r) >= 0]
    if len(ok) < 3:
        return None
    ok.sort(key=_result_rank_value, reverse=True)
    lines = ["rank | macro_AP (ranking) | macro_AUC | description", "-" * 80]
    for i, r in enumerate(ok[:20]):
        ap = r.get("macro_average_precision")
        auc = r.get("competition_macro_auc_v2") or r.get("macro_roc_auc")
        ap_s = f"{ap:.5f}" if ap is not None else "N/A"
        auc_s = f"{auc:.5f}" if auc is not None else "N/A"
        lines.append(f"{i+1:4d} | {ap_s:>18s} | {auc_s:>9s} | {r['description']}")
    prompt = (
        "Results from random architecture search for bird audio classification.\n\n"
        + "\n".join(lines) + "\n\n"
        "Return ONLY a JSON object with narrowed ranges for focused search:\n"
        '{"depth":[min,max],"filters_base":[min,max],"learning_rate":[min,max],'
        '"weight_decay":[min,max],"classifier_hidden_units":[min,max],"residuals":true/false/null,'
        '"analysis":"brief summary"}'
    )
    resp = llm.generate_from_messages(
        messages=[{"role":"system","content":"Return ONLY valid JSON."},{"role":"user","content":prompt}],
        temperature=temperature)
    try:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m: return json.loads(m.group())
    except: pass
    return None

def _sample_narrowed(ranges, rng):
    p = _sample_random_params(rng)
    if not ranges: return p
    def _pick(key, choices):
        if key in ranges and isinstance(ranges[key], list) and len(ranges[key]) == 2:
            f = [v for v in choices if ranges[key][0] <= v <= ranges[key][1]]
            if f: return int(rng.choice(f))
        return p[key]
    p["depth"] = _pick("depth", [1, 2, 3, 4, 6, 8, 12, 15, 20])
    p["filters_base"] = _pick("filters_base", [8, 16, 32, 64, 128])
    p["classifier_hidden_units"] = _pick("classifier_hidden_units", [0, 128, 256, 512])
    if "learning_rate" in ranges and isinstance(ranges["learning_rate"], list) and len(ranges["learning_rate"])==2:
        p["learning_rate"] = float(10**rng.uniform(np.log10(max(ranges["learning_rate"][0],1e-6)),
                                                    np.log10(min(ranges["learning_rate"][1],1.0))))
    if "weight_decay" in ranges and isinstance(ranges["weight_decay"], list) and len(ranges["weight_decay"]) == 2:
        wd_min = max(ranges["weight_decay"][0], 1e-8)
        wd_max = max(ranges["weight_decay"][1], wd_min)
        p["weight_decay"] = float(10 ** rng.uniform(np.log10(wd_min), np.log10(wd_max)))
    for key in ("residuals",):
        if key in ranges and ranges[key] is not None: p[key] = bool(ranges[key])
    return p


def _best_successful(results, top_k=4):
    ok = [r for r in results if r.get("success") and _result_rank_value(r) >= 0]
    ok.sort(key=_result_rank_value, reverse=True)
    return ok[:top_k]


def _is_parametric_result(entry):
    p = entry.get("params", {})
    return isinstance(p, dict) and "depth" in p and "filters_base" in p and "learning_rate" in p


def _best_parametric_successful(results, top_k=4):
    ok = [r for r in _best_successful(results, len(results) or top_k) if _is_parametric_result(r)]
    return ok[:top_k]


def _save_phase2_results(logs_dir, all_results, top_k):
    _save_json(all_results, logs_dir / "random_results.json")
    _save_json(_best_successful(all_results, top_k), logs_dir / "final_results.json")


def _generate_ai_free_slot(llm, temperature, best_results, fallback_slot, cheap_cfg):
    top_lines = []
    for i, r in enumerate(best_results[:4], start=1):
        m = {
            "status": "success",
            "macro_average_precision": r.get("macro_average_precision"),
            "macro_roc_auc": r.get("macro_roc_auc"),
            "competition_macro_auc_v2": r.get("competition_macro_auc_v2"),
            "median_per_class_auc": r.get("median_per_class_auc"),
        }
        top_lines.append(f"{i}. {_fmt_experiment_metrics(m)} | {r['description']}")
    top_summary = "\n".join(top_lines) if top_lines else "No successful prior runs yet."

    prompt = (
        "You are improving a BirdCLEF audio classification model.\n"
        "Create a stronger architecture than prior runs while keeping training cheap for search.\n\n"
        f"Top prior results:\n{top_summary}\n\n"
        f"Current baseline slot code:\n```python\n{fallback_slot}\n```\n\n"
        "Return ONLY Python code with get_training_config() and build_model(input_shape, num_classes).\n"
        "Constraints:\n"
        f"- max_samples must be {cheap_cfg['max_samples']}\n"
        f"- epochs must be {cheap_cfg['epochs']}\n"
        f"- val_split must be {cheap_cfg.get('val_split', 0.2)}\n"
        "- Keep final layer as Dense(num_classes, activation='sigmoid').\n"
        "- No top-level executable statements.\n"
    )
    resp = llm.generate_from_messages(
        messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    candidate = extract_python_code(resp)
    issues = validate_slot_code(candidate) if candidate else ["No code found."]
    if issues:
        print(f"    AI-free generation invalid, using fallback slot. Issues: {issues}")
        return fallback_slot
    return candidate


def _summarize_architectures(results, limit=12):
    rows = []
    seen = set()
    ordered = sorted(
        [r for r in results if r.get("description")],
        key=lambda r: (r.get("macro_roc_auc") or -1.0),
        reverse=True,
    )
    for r in ordered:
        desc = r.get("description", "")
        key = (r.get("search_type"), desc)
        if key in seen:
            continue
        seen.add(key)
        auc = r.get("macro_roc_auc")
        auc_txt = f"{auc:.6f}" if isinstance(auc, (int, float)) else "N/A"
        rows.append(f"- {r.get('search_type','unknown')}: AUC={auc_txt} | {desc}")
        if len(rows) >= limit:
            break
    return "\n".join(rows) if rows else "- No prior architecture summaries yet."


def _has_auc_plateau(results, repeats=3, decimals=6):
    aucs = [
        round(float(r["macro_roc_auc"]), decimals)
        for r in results
        if isinstance(r.get("macro_roc_auc"), (int, float))
    ]
    if len(aucs) < repeats:
        return False, None
    tail = aucs[-repeats:]
    if len(set(tail)) == 1:
        return True, tail[-1]
    return False, None


def _generate_ai_free_slot_with_context(
    llm, temperature, best_results, fallback_slot, cheap_cfg, tried_summary, force_novelty=False,
    researcher_hint="",
):
    top_lines = []
    for i, r in enumerate(best_results[:4], start=1):
        top_lines.append(f"{i}. AUC={r['macro_roc_auc']:.6f} | {r['description']}")
    top_summary = "\n".join(top_lines) if top_lines else "No successful prior runs yet."
    novelty_block = (
        "IMPORTANT: Recent AI-free attempts plateaued at the same AUC multiple times.\n"
        "You MUST use a clearly different modeling direction than prior attempts.\n"
        "Examples of acceptable direction changes: optimizer family change, non-trivial depth/pooling redesign, "
        "different regularization strategy, or architectural block pattern changes.\n"
        "Do not return a near-duplicate of previously tried architectures.\n\n"
    ) if force_novelty else ""

    researcher_block = (
        f"Research direction from expert analysis:\n{researcher_hint}\n\n"
    ) if researcher_hint else ""

    prompt = (
        "You are improving a BirdCLEF audio classification model.\n"
        "Create a stronger architecture than prior runs while keeping training cheap for search.\n\n"
        + researcher_block
        + f"Top prior results:\n{top_summary}\n\n"
        f"Previously tried architecture summaries:\n{tried_summary}\n\n"
        + novelty_block
        + f"Current baseline slot code:\n```python\n{fallback_slot}\n```\n\n"
        "Return ONLY Python code with get_training_config() and build_model(input_shape, num_classes).\n"
        "Constraints:\n"
        f"- max_samples must be {cheap_cfg['max_samples']}\n"
        f"- epochs must be {cheap_cfg['epochs']}\n"
        f"- val_split must be {cheap_cfg.get('val_split', 0.2)}\n"
        "- Keep final layer as Dense(num_classes, activation='sigmoid').\n"
        "- No top-level executable statements.\n"
    )
    resp = llm.generate_from_messages(
        messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    candidate = extract_python_code(resp)
    issues = validate_slot_code(candidate) if candidate else ["No code found."]
    if issues:
        print(f"    AI-free generation invalid, using fallback slot. Issues: {issues}")
        return fallback_slot
    return candidate


def _generate_medium_slot(llm, temperature, best_results, fallback_slot, medium_cfg, medium_history_summary=""):
    top_lines = []
    for i, r in enumerate(best_results[:4], start=1):
        top_lines.append(f"{i}. AUC={r['macro_roc_auc']:.6f} | {r['description']}")
    top_summary = "\n".join(top_lines) if top_lines else "No successful prior runs yet."

    history_block = (
        f"Medium-stage history so far (attempt summaries):\n{medium_history_summary}\n\n"
        if medium_history_summary else
        "Medium-stage history so far: none (first generated attempt).\n\n"
    )

    prompt = (
        "You are improving a BirdCLEF audio classification model for medium-scale validation.\n"
        "Problem at hand:\n"
        "- Multi-label bird species classification from 5s audio clips.\n"
        "- Input is mel spectrograms; output is sigmoid probabilities for all species.\n"
        "- Metric is macro ROC-AUC.\n"
        "- Goal: maximize generalization under this medium training budget before final full training.\n\n"
        "Use prior best models as inspiration and propose a stronger architecture.\n"
        "Really try to find a high-performing direction, not small cosmetic edits.\n\n"
        f"Top prior models:\n{top_summary}\n\n"
        + history_block +
        f"Reference slot code:\n```python\n{fallback_slot}\n```\n\n"
        "Return ONLY Python code with get_training_config() and build_model(input_shape, num_classes).\n"
        "Constraints:\n"
        f"- max_samples must be {medium_cfg['max_samples']}\n"
        f"- epochs must be {medium_cfg['epochs']}\n"
        f"- val_split must be {medium_cfg.get('val_split', 0.2)}\n"
        "- Keep final layer as Dense(num_classes, activation='sigmoid').\n"
        "- No top-level executable statements.\n"
    )
    resp = llm.generate_from_messages(
        messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    candidate = extract_python_code(resp)
    issues = validate_slot_code(candidate) if candidate else ["No code found."]
    if issues:
        print(f"    Medium LLM generation invalid, using fallback slot. Issues: {issues}")
        return fallback_slot
    return candidate


def _run_medium_stage(previous_results, executor, evaluator, llm, temperature, dirs, search_cfg):
    medium_cfg = search_cfg.get("medium", {})
    ms_cfg = search_cfg.get("medium_stage", {})
    if not ms_cfg.get("enabled", True):
        return [], None

    total_runs = int(ms_cfg.get("total_runs", 10))
    promoted_runs = int(ms_cfg.get("promoted_runs", 4))
    medium_max_attempts = max(1, int(ms_cfg.get("max_attempts_per_model", 1)))
    promoted_runs = max(0, min(promoted_runs, total_runs))
    generated_runs = max(0, total_runs - promoted_runs)
    s = int(medium_cfg.get("max_samples", 10000))
    e = int(medium_cfg.get("epochs", 15))
    vs = float(medium_cfg.get("val_split", 0.2))

    print("\n" + "=" * 60)
    print("  PHASE 2.5: MEDIUM SCALE VALIDATION")
    print("=" * 60)
    print(f"  Runs: {total_runs} total ({promoted_runs} promoted + {generated_runs} LLM-generated)")
    print(f"  Config: max_samples={s}, epochs={e}, val_split={vs}")
    print(f"  Attempt policy: max_attempts_per_model={medium_max_attempts}")
    print("  Checkpoint policy: restore best validation epoch (not last epoch).")

    rng = np.random.default_rng(search_cfg.get("random_seed", 42) + 99)
    all_results = []

    top_overall = [r for r in _best_successful(previous_results, max(promoted_runs, 1) * 3) if r.get("slot_code")]
    top_overall = top_overall[:max(promoted_runs, 1)]
    if promoted_runs > 0:
        print(f"\n  ── M1: {promoted_runs} promoted models (overall top, scaled as-is) ──")
    for i in range(promoted_runs):
        if not top_overall:
            break
        base = top_overall[i % len(top_overall)]
        p = dict(DEFAULT_PARAMS)
        p.update(base.get("params", {}))
        p["max_samples"] = s
        p["epochs"] = e
        p["val_split"] = vs
        slot = base["slot_code"] + _final_config_override_block({
            "max_samples": s,
            "sample_rate": 32000,
            "clip_seconds": 5.0,
            "n_mels": p.get("n_mels", 64),
            "n_frames": p.get("n_frames", 128),
            "epochs": e,
            "batch_size": p.get("batch_size", 32),
            "learning_rate": p.get("learning_rate", 1e-3),
            "optimizer": p.get("optimizer", "adam"),
            "val_split": vs,
            "weight_decay": p.get("weight_decay", 0.0),
            "classifier_hidden_units": p.get("classifier_hidden_units", 0),
            "pooling_type": p.get("pooling_type", "global_avg"),
            "use_best_checkpoint": True,
        })
        rid = f"M{i+1:03d}_promoted"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature,
            max_attempts=medium_max_attempts, use_llm_fixes=medium_max_attempts > 1
        )
        auc = metrics["macro_roc_auc"] if metrics and metrics.get("status") == "success" else None
        entry = {
            "run_id": rid,
            "search_type": "medium_promoted",
            "params": {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            "macro_roc_auc": auc,
            "num_scored_columns": int(metrics["num_scored_columns"])
            if metrics and metrics.get("status") == "success" and metrics.get("num_scored_columns") is not None
            else None,
            "success": auc is not None,
            "attempts": att,
            "best_epoch": metrics.get("best_epoch") if metrics else None,
            "description": describe_params(p),
            "slot_code": slot,
        }
        all_results.append(entry)
        print(f"    {rid}: {_fmt_score(auc)} | {entry['description']}")

    if generated_runs > 0:
        print(f"\n  ── M2: {generated_runs} LLM-generated medium models (successful target) ──")
    medium_history = []
    gen_success = 0
    gen_candidates = 0
    max_gen_candidates = max(generated_runs * 5, generated_runs)
    while gen_success < generated_runs and gen_candidates < max_gen_candidates:
        gen_candidates += 1
        ref_pool = _best_successful(previous_results + all_results, 4)
        ref_params_pool = _best_parametric_successful(previous_results + all_results, 1)
        if ref_params_pool:
            seed_p = dict(ref_params_pool[0]["params"])
        else:
            seed_p = _sample_random_params(rng)
        seed_p["max_samples"] = s
        seed_p["epochs"] = e
        seed_p["val_split"] = vs
        seed_slot = generate_slot_code(seed_p)
        medium_history_summary = _summarize_architectures(medium_history, limit=10)
        slot = _generate_medium_slot(
            llm,
            temperature,
            ref_pool,
            seed_slot,
            {"max_samples": s, "epochs": e, "val_split": vs},
            medium_history_summary=medium_history_summary,
        )
        rid = f"M{promoted_runs + gen_candidates:03d}_llm"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature,
            max_attempts=medium_max_attempts, use_llm_fixes=medium_max_attempts > 1
        )
        auc = metrics["macro_roc_auc"] if metrics and metrics.get("status") == "success" else None
        success = auc is not None
        if success:
            gen_success += 1
        entry = {
            "run_id": rid,
            "search_type": "medium_llm",
            "params": {k: v for k, v in seed_p.items() if k not in ("max_samples", "epochs")},
            "macro_roc_auc": auc,
            "num_scored_columns": int(metrics["num_scored_columns"])
            if metrics and metrics.get("status") == "success" and metrics.get("num_scored_columns") is not None
            else None,
            "success": success,
            "counts_toward_budget": success,
            "attempts": att,
            "best_epoch": metrics.get("best_epoch") if metrics else None,
            "description": f"Medium LLM generation candidate #{gen_candidates}",
            "slot_code": slot,
        }
        all_results.append(entry)
        medium_history.append(entry)
        if success:
            print(f"    {rid}: {_fmt_score(auc)} | accepted ({gen_success}/{generated_runs})")
        else:
            print(f"    {rid}: failed after {att} attempts, does not count toward medium LLM budget. Moving on.")
    if gen_success < generated_runs:
        print(f"  Warning: medium LLM generated target not fully reached ({gen_success}/{generated_runs}).")

    _save_json(all_results, dirs["logs"] / "medium_results.json")
    _save_json(_best_successful(all_results, 4), dirs["logs"] / "medium_top_results.json")
    all_ok = [r for r in all_results if r.get("success") and r.get("macro_roc_auc")]
    best = max(all_ok, key=lambda r: r["macro_roc_auc"]) if all_ok else None
    print(f"\n  Medium stage done: {len(all_results)} iters, {len(all_ok)} successful")
    if best:
        print(f"  Best medium: AUC={best['macro_roc_auc']:.6f} | {best['description']}")
    return all_results, best


def _run_reality_check_gate(candidates, executor, evaluator, dirs, search_cfg):
    gate_cfg = search_cfg.get("reality_gate", {})
    if not gate_cfg.get("enabled", True):
        return [], None

    top_k = int(gate_cfg.get("top_k", 5))
    use_all = bool(gate_cfg.get("use_all_candidates", False))
    s = int(gate_cfg.get("max_samples", search_cfg.get("medium", {}).get("max_samples", 10000)))
    e = int(gate_cfg.get("epochs", search_cfg.get("medium", {}).get("epochs", 15)))
    vs = float(gate_cfg.get("val_split", 0.2))
    split_mode = str(gate_cfg.get("split_mode", "group_holdout"))
    eval_candidates = [c for c in candidates if c.get("slot_code")]
    eval_candidates = sorted(eval_candidates, key=lambda r: (r.get("macro_roc_auc") or -1.0), reverse=True)
    if not use_all:
        eval_candidates = eval_candidates[:top_k]

    print("\n" + "=" * 60)
    print("  PHASE 2.6: REALITY-CHECK GATE")
    print("=" * 60)
    print(f"  Candidates: {len(eval_candidates)} ({'all' if use_all else f'top_k={top_k}'})")
    print(f"  Gate config: samples={s}, epochs={e}, val_split={vs}, split_mode={split_mode}")

    results = []
    for i, c in enumerate(eval_candidates, start=1):
        base_params = dict(DEFAULT_PARAMS)
        base_params.update(c.get("params", {}))
        gate_cfg_override = {
            "max_samples": s,
            "sample_rate": 32000,
            "clip_seconds": 5.0,
            "n_mels": base_params.get("n_mels", 64),
            "n_frames": base_params.get("n_frames", 128),
            "epochs": e,
            "batch_size": base_params.get("batch_size", 32),
            "learning_rate": base_params.get("learning_rate", 1e-3),
            "optimizer": base_params.get("optimizer", "adam"),
            "val_split": vs,
            "weight_decay": base_params.get("weight_decay", 0.0),
            "classifier_hidden_units": base_params.get("classifier_hidden_units", 0),
            "pooling_type": base_params.get("pooling_type", "global_avg"),
            "use_best_checkpoint": True,
            "split_mode": split_mode,
        }
        slot = c["slot_code"] + _final_config_override_block(gate_cfg_override)
        rid = f"G{i:03d}_{c.get('run_id','cand')}"
        slot, metrics, _result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator,
            llm=None, temperature=0.0, max_attempts=1, use_llm_fixes=False
        )
        auc = metrics["macro_roc_auc"] if metrics and metrics.get("status") == "success" else None
        entry = {
            "run_id": rid,
            "source_run_id": c.get("run_id"),
            "search_type": "reality_gate",
            "params": c.get("params"),
            "macro_roc_auc": auc,
            "num_scored_columns": int(metrics["num_scored_columns"])
            if metrics and metrics.get("status") == "success" and metrics.get("num_scored_columns") is not None
            else None,
            "success": auc is not None,
            "attempts": att,
            "best_epoch": metrics.get("best_epoch") if metrics else None,
            "description": c.get("description", ""),
            "slot_code": c.get("slot_code"),
        }
        results.append(entry)
        print(f"    {rid}: {_fmt_score(auc)} | source={c.get('run_id')}")

    _save_json(results, dirs["logs"] / "reality_check_results.json")
    ok = [r for r in results if r.get("success") and isinstance(r.get("macro_roc_auc"), (int, float))]
    best = max(ok, key=lambda r: r["macro_roc_auc"]) if ok else None
    if best:
        print(f"  Gate winner: AUC={best['macro_roc_auc']:.6f} | source={best.get('source_run_id')}")
    else:
        print("  Gate winner: none (all failed).")
    return results, best


def _run_random_search(baseline_slot, executor, evaluator, llm, temperature, dirs, search_cfg,
                       llm_researcher=None, researcher_temp=0.6, config=None):
    config = config or {}
    print("\n" + "=" * 60)
    print("  PHASE 2: RANDOM SEARCH")
    print(f"  Ranking: {_ranking_metric_from_config(config)} on labeled train_soundscapes")
    print("=" * 60)

    budget = search_cfg.get("random_budget", 50)
    s, e = search_cfg["cheap"]["max_samples"], search_cfg["cheap"]["epochs"]
    vs = search_cfg["cheap"].get("val_split", 0.2)
    phase2_cfg = search_cfg.get("phase2", {})
    explore_count = int(phase2_cfg.get("random_experiments", budget))
    focused_count = int(phase2_cfg.get("focused_experiments", 50))
    tweak_count = int(phase2_cfg.get("tweak_experiments", 50))
    aug_tweak_count = 0 if config.get("lock_augmentation") else int(
        phase2_cfg.get("augmentation_tweak_experiments", 5)
    )
    ai_free_count = int(phase2_cfg.get("ai_free_experiments", 50))
    final_tweak_count = int(phase2_cfg.get("final_tweak_experiments", 15))
    top_keep = int(phase2_cfg.get("top_results_keep", 4))
    rng = np.random.default_rng(search_cfg.get("random_seed", 42))
    all_results = []
    rc = 0
    current_best_slot = baseline_slot

    # Sub-phase A: Pure random
    print(f"\n  ── A: {explore_count} random experiments ──")
    for _ in range(explore_count):
        rc += 1
        p = _sample_random_params(rng, config); p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
        slot = generate_slot_code(p)
        rid = f"R{rc:03d}_explore"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature)
        entry = _search_result_entry(
            rid, "explore",
            {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            metrics, att, describe_params(p), slot,
        )
        all_results.append(entry)
        if entry["success"]:
            current_best_slot = slot
        print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | {describe_params(p)}")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    # Sub-phase C-aug: augmentation tweaks of top configs
    print(f"\n  ── C-aug: {aug_tweak_count} augmentation tweaks of top configs ──")
    for i in range(aug_tweak_count):
        ok = _best_parametric_successful(all_results, 5)
        if not ok:
            break
        rc += 1
        p = _tweak_augmentation_params(ok[i % len(ok)]["params"], rng)
        p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
        slot = generate_slot_code(p)
        rid = f"R{rc:03d}_tweak_aug"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature)
        entry = _search_result_entry(
            rid, "tweak_aug",
            {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            metrics, att, describe_params(p), slot,
        )
        all_results.append(entry)
        if entry["success"]:
            current_best_slot = slot
        print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | {describe_params(p)}")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    # Researcher analysis (dual-LLM: DeepSeek reasons, then coder acts)
    researcher_hint = _researcher_analysis(llm_researcher, all_results, researcher_temp)

    # LLM analysis
    print(f"\n  ── Analyzing results with LLM ──")
    narrowed = _analyze_random_results(all_results, llm, temperature)
    if narrowed:
        print(f"  Analysis: {narrowed.get('analysis','')[:300]}")
        _save_json(narrowed, dirs["logs"] / "random_llm_analysis.json")
    else:
        print("  LLM analysis failed. Using top-K exploitation.")

    # Sub-phase B: Focused
    print(f"\n  ── B: {focused_count} focused experiments ──")
    for _ in range(focused_count):
        rc += 1
        p = _sample_narrowed(narrowed, rng); p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
        slot = generate_slot_code(p)
        rid = f"R{rc:03d}_focused"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature)
        entry = _search_result_entry(
            rid, "focused",
            {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            metrics, att, describe_params(p), slot,
        )
        all_results.append(entry)
        if entry["success"]:
            current_best_slot = slot
        print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | {describe_params(p)}")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    # Sub-phase C: broad tweak pass
    print(f"\n  ── C: {tweak_count} tweaks of top configs ──")
    for i in range(tweak_count):
        ok = _best_parametric_successful(all_results, 5)
        if not ok:
            p = _sample_random_params(rng, config)
        else:
            p = _tweak_params(ok[i % len(ok)]["params"], rng, config)
        rc += 1
        p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
        slot = generate_slot_code(p)
        rid = f"R{rc:03d}_tweak"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature)
        entry = _search_result_entry(
            rid, "tweak",
            {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            metrics, att, describe_params(p), slot,
        )
        all_results.append(entry)
        if entry["success"]:
            current_best_slot = slot
        print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | {describe_params(p)}")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    # Sub-phase D: AI free experiments
    print(f"\n  ── D: {ai_free_count} AI freely experiments (successful target) ──")
    ai_free_history = []
    ai_success = 0
    ai_candidates = 0
    max_ai_candidates = max(ai_free_count * 5, ai_free_count)
    while ai_success < ai_free_count and ai_candidates < max_ai_candidates:
        ai_candidates += 1
        rc += 1
        top_now = _best_successful(all_results, top_keep)
        top_param_now = _best_parametric_successful(all_results, 1)
        seed_slot = current_best_slot
        tried_summary = _summarize_architectures(all_results, limit=12)
        plateau, auc_plateau = _has_auc_plateau(ai_free_history, repeats=3, decimals=6)
        if plateau:
            print(f"    AI-free plateau detected at AUC={auc_plateau:.6f}; forcing novel direction.")
        ai_slot = _generate_ai_free_slot_with_context(
            llm=llm,
            temperature=temperature,
            best_results=top_now,
            fallback_slot=seed_slot,
            cheap_cfg={"max_samples": s, "epochs": e, "val_split": vs},
            tried_summary=tried_summary,
            force_novelty=plateau,
            researcher_hint=researcher_hint,
        )
        rid = f"R{rc:03d}_ai_free"
        ai_slot, metrics, result, att = run_experiment_until_success(
            ai_slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature, max_attempts=7)
        entry = _search_result_entry(
            rid,
            "ai_free",
            top_param_now[0]["params"] if top_param_now else dict(DEFAULT_PARAMS),
            metrics,
            att,
            f"AI free generation candidate #{ai_candidates}",
            ai_slot,
        )
        entry["counts_toward_budget"] = entry["success"]
        success = entry["success"]
        if success:
            ai_success += 1
        all_results.append(entry)
        ai_free_history.append(entry)
        if success:
            current_best_slot = ai_slot
            print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | accepted ({ai_success}/{ai_free_count})")
        else:
            print(f"    R{rc:03d}: failed after {att} attempts, does not count toward AI-free budget. Moving on.")
    if ai_success < ai_free_count:
        print(f"  Warning: AI-free target not fully reached ({ai_success}/{ai_free_count}).")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    # Final tightening tweaks
    print(f"\n  ── C: {final_tweak_count} tweaks of top configs ──")
    for i in range(final_tweak_count):
        ok = _best_parametric_successful(all_results, 5)
        if not ok:
            break
        rc += 1
        p = _tweak_params(ok[i % len(ok)]["params"], rng, config)
        p["max_samples"] = s; p["epochs"] = e; p["val_split"] = vs
        slot = generate_slot_code(p)
        rid = f"R{rc:03d}_tweak_final"
        slot, metrics, result, att = run_experiment_until_success(
            slot, rid, dirs["random_codes"], dirs["eval"], executor, evaluator, llm, temperature)
        entry = _search_result_entry(
            rid, "tweak_final",
            {k: v for k, v in p.items() if k not in ("max_samples", "epochs")},
            metrics, att, describe_params(p), slot,
        )
        all_results.append(entry)
        if entry["success"]:
            current_best_slot = slot
        print(f"    R{rc:03d}: {_fmt_experiment_metrics(metrics)} | {describe_params(p)}")
    _save_phase2_results(dirs["logs"], all_results, top_keep)

    all_ok = [r for r in all_results if r.get("success") and _result_rank_value(r) >= 0]
    best = max(all_ok, key=_result_rank_value) if all_ok else None
    print(f"\n  Random done: {rc} iters, {len(all_ok)} successful")
    if best:
        best_m = {
            "status": "success",
            "macro_average_precision": best.get("macro_average_precision"),
            "macro_roc_auc": best.get("macro_roc_auc"),
            "competition_macro_auc_v2": best.get("competition_macro_auc_v2"),
            "median_per_class_auc": best.get("median_per_class_auc"),
            "ranking_metric": best.get("ranking_metric"),
            "ranking_value": best.get("ranking_value"),
        }
        print(f"  Best: {_fmt_experiment_metrics(best_m)} | {best['description']}")
    return all_results, best


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFER EXPLORATION (formerly agent_transfer.py)
# ═══════════════════════════════════════════════════════════════════════════
#
# This section is the inlined equivalent of the previous `agent_transfer`
# module: an LLM-driven exploration loop that uses pretrained ImageNet
# backbones (MobileNetV2, etc.) for the BirdCLEF task. Slot code generated
# here also defines build_model() / get_training_config(), and is wrapped by
# the same harness used by the parametric pipeline above.
#
# Public entry point: `run_transfer_exploration_phase(...)` returns a dict
# with at least `effective_slot_code` and `transfer_effective_score`.
# ═══════════════════════════════════════════════════════════════════════════

# --- Default starter slot for transfer iterations (CNN baseline, will be
#     replaced by LLM with pretrained backbones once stagnation is detected). ---

TRANSFER_DEFAULT_SLOT_CODE = '''EXPERIMENT_META = {
    "model_type": "cnn",
    "architecture": "3x Conv2D + GAP + Dropout",
    "change": "baseline",
    "key_params": {"lr": 1e-3, "batch_size": 32, "epochs": 5},
}


def get_training_config():
    return {
        "max_samples": 1500,
        "sample_rate": 32000,
        "clip_seconds": 5.0,
        "n_mels": 64,
        "n_frames": 128,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "val_split": 0.2,
    }


def build_model(input_shape, num_classes):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model
'''


TRANSFER_SYSTEM_PROMPT = (
    "You are an ML research assistant. You write Python functions to define "
    "neural network architectures for bird species audio classification.\n\n"
    "RULES:\n"
    "- Return ONLY a ```python``` code block, nothing else\n"
    "- Define exactly: get_training_config(), build_model(input_shape, num_classes), and EXPERIMENT_META\n"
    "- Optionally define: build_features(audio_path, sample_rate, clip_seconds, n_mels, n_frames)\n"
    "- Do NOT define main(), do NOT load data, do NOT write submission logic\n"
    "- build_model must return a COMPILED Keras model\n"
    "- Last layer must be Dense(num_classes, activation='sigmoid')\n"
    "- Loss must be 'binary_crossentropy'\n"
    "- input_shape is (n_mels, n_frames, 1) — a mel spectrogram image\n"
    "- num_classes is 234 (one per bird species)\n\n"
    "TRANSFER LEARNING:\n"
    "- You may use MobileNetV2 as a pretrained backbone instead of a custom CNN\n"
    "- Only switch to pretrained when explicitly told CNN results are stagnating\n"
    "- For the first pretrained run: freeze the backbone (base_model.trainable = False), train only the head\n"
    "- Only unfreeze (base_model.trainable = True) if a frozen pretrained run already improved results\n"
    "- Spectrograms are 1-channel: resize to (96, 96) and repeat to 3 channels before passing to MobileNetV2\n"
    "- Example pretrained build_model:\n"
    "    base = tf.keras.applications.MobileNetV2(input_shape=(96,96,3), include_top=False, weights='imagenet')\n"
    "    base.trainable = False\n"
    "    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)\n"
    "    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)\n"
    "    return tf.keras.Model(inputs=base.input, outputs=out)\n\n"
    "EXPERIMENT_META must be a dict with these keys:\n"
    "    model_type: 'cnn' | 'pretrained'\n"
    "    architecture: short description, e.g. '3x Conv2D + GAP' or 'MobileNetV2 frozen'\n"
    "    change: what you changed vs the previous run, or 'baseline' for the first run\n"
    "    key_params: dict with lr, batch_size, epochs, and any other relevant params\n"
)

TRANSFER_SEED_PROMPT = (
    "Write get_training_config() and build_model() for BirdCLEF 2026.\n\n"
    "Task: multi-label bird species classification from mel spectrograms.\n"
    "Input: mel spectrogram of shape (n_mels, n_frames, 1), default (64, 128, 1).\n"
    "Output: 234 species probabilities (sigmoid).\n"
    "Loss: binary_crossentropy.\n\n"
    "Data info:\n"
    "- ~24,000 training audio clips (.ogg) in data/train_audio/\n"
    "- train.csv columns: primary_label, filename, secondary_labels, latitude, longitude, "
    "scientific_name, common_name, etc.\n"
    "- 234 target species defined by sample_submission.csv\n"
    "- Audio is 5-second clips converted to mel spectrograms\n\n"
    "Start with a simple CNN baseline (Conv2D blocks + GlobalAveragePooling).\n\n"
    "Here is the starting code to modify:\n"
    f"```python\n{TRANSFER_DEFAULT_SLOT_CODE}```"
)


def _transfer_extract_python_code(response: str) -> str:
    """Same as extract_python_code but kept local to make this section self-contained."""
    if not response or not response.strip():
        return ""
    blocks = re.findall(r"```python\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[0].strip()
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return ""


def _transfer_validate_slot_code(code: str) -> list:
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    if not re.search(r"def\s+build_model\s*\(", code):
        issues.append("Missing: def build_model(input_shape, num_classes)")
    if not re.search(r"def\s+get_training_config\s*\(", code):
        issues.append("Missing: def get_training_config()")
    if not re.search(r"EXPERIMENT_META\s*=\s*\{", code):
        issues.append("Missing: EXPERIMENT_META = {...}")
    if re.search(r"def\s+main\s*\(", code):
        issues.append("Do NOT define main()")
    return issues


def _transfer_extract_experiment_meta(slot_code: str) -> dict:
    ns: dict = {}
    try:
        exec(slot_code, ns)
        meta = ns.get("EXPERIMENT_META", {})
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass
    return {}


def _transfer_build_feedback_prompt(*, stdout, stderr, is_error, metrics, current_code, cnn_stagnating=False):
    if is_error:
        return (
            "The last run FAILED with this error:\n"
            f"```\n{_truncate(stderr, 3000)}\n```\n\n"
            "Fix the bug. Return the corrected get_training_config() and build_model().\n"
            f"Current code:\n```python\n{current_code}\n```"
        )
    metrics_str = ""
    if metrics:
        metrics_str = (
            f"macro_roc_auc = {metrics.get('macro_roc_auc', 'N/A')}\n"
            f"scored_species = {metrics.get('num_scored_columns', '?')}\n"
            f"training_samples = {metrics.get('num_samples', '?')}\n"
        )
    if cnn_stagnating:
        improvement_hint = (
            "CNN results have been flat for the last 3 runs — no meaningful improvement.\n"
            "Switch to transfer learning: use MobileNetV2 as a frozen pretrained backbone.\n"
            "Follow the pretrained example in the system prompt exactly.\n"
            "Set model_type='pretrained' and architecture='MobileNetV2 frozen' in EXPERIMENT_META.\n"
        )
    else:
        improvement_hint = (
            "Propose ONE specific improvement to increase macro_roc_auc. Options:\n"
            "- More/fewer Conv2D layers or filters\n"
            "- Add BatchNormalization\n"
            "- Change dropout, learning rate, epochs, batch size\n"
            "- Increase max_samples\n"
            "- Change n_mels, n_frames\n"
            "- Add data augmentation in build_features()\n"
        )
    return (
        "The last experiment completed successfully.\n\n"
        f"Results:\n{metrics_str}\n"
        f"Training output:\n{_truncate(stdout, 2000)}\n\n"
        + improvement_hint
        + f"\nReturn the complete updated code.\nCurrent code:\n```python\n{current_code}\n```"
    )


def _transfer_trim_and_append(messages, role, content):
    messages.append({"role": role, "content": content})
    if len(messages) > 6:
        messages = messages[:2] + messages[-4:]
    return messages


def _generate_transfer_kaggle_notebook(slot_code: str, output_path: Path) -> None:
    """Render a Kaggle inference notebook for a transfer-learning slot.

    Kept identical to the original agent_transfer._generate_kaggle_notebook
    so submissions remain reproducible.
    """
    ns: dict = {}
    try:
        exec(slot_code, ns)
        cfg = ns.get("get_training_config", lambda: {})()
    except Exception:
        cfg = {}

    sr = cfg.get("sample_rate", 32000)
    cs = cfg.get("clip_seconds", 5.0)
    nm = cfg.get("n_mels", 64)
    nf = cfg.get("n_frames", 128)

    nb = {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# BirdCLEF+ 2026 — Inference Notebook\n",
                "Auto-generated by the autonomous research agent (transfer exploration).\n",
                "Upload model.keras as a Kaggle dataset and reference it below."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import os, numpy as np, pandas as pd, librosa, tensorflow as tf\n",
                "from pathlib import Path\n",
                "\n",
                f"SR = {sr}\n",
                f"CLIP_SECONDS = {cs}\n",
                f"N_MELS = {nm}\n",
                f"N_FRAMES = {nf}\n",
                "N_FFT = 1024\n",
                "HOP_LENGTH = 512\n",
                "\n",
                "COMP_DIR = '/kaggle/input/birdclef-2026'\n",
                "MODEL_DIR = '/kaggle/input/birdclef-model'  # your uploaded model dataset\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model.keras'))\n",
                "sample_sub = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))\n",
                "species_cols = [c for c in sample_sub.columns if c != 'row_id']\n",
                "print(f'Model loaded. Species: {len(species_cols)}')\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "test_dir = Path(COMP_DIR) / 'test_soundscapes'\n",
                "test_files = sorted(test_dir.glob('*.ogg')) if test_dir.exists() else []\n",
                "print(f'Test soundscapes: {len(test_files)}')\n",
                "\n",
                "all_rows = []\n",
                "seg_samples = SR * CLIP_SECONDS\n",
                "\n",
                "for fpath in test_files:\n",
                "    name = fpath.stem\n",
                "    y_full, _ = librosa.load(str(fpath), sr=SR, mono=True)\n",
                "    for start in range(0, len(y_full), seg_samples):\n",
                "        seg = y_full[start:start+seg_samples]\n",
                "        if len(seg) < seg_samples:\n",
                "            seg = np.pad(seg, (0, seg_samples - len(seg)))\n",
                "        end_sec = (start // seg_samples + 1) * CLIP_SECONDS\n",
                "        mel = librosa.feature.melspectrogram(\n",
                "            y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)\n",
                "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
                "        mel_r = tf.image.resize(mel_db[..., np.newaxis], (N_MELS, N_FRAMES)).numpy().astype(np.float32)\n",
                "        preds = model.predict(mel_r[np.newaxis, ...], verbose=0)[0]\n",
                "        row = {'row_id': f'{name}_{end_sec}'}\n",
                "        for col, p in zip(species_cols, preds):\n",
                "            row[col] = float(p)\n",
                "        all_rows.append(row)\n",
                "\n",
                "print(f'Prediction rows: {len(all_rows)}')\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "sub = pd.DataFrame(all_rows) if all_rows else sample_sub.copy()\n",
                "sub = sample_sub[['row_id']].merge(sub, on='row_id', how='left')\n",
                "sub[species_cols] = sub[species_cols].fillna(0.0).clip(0.0, 1.0)\n",
                "sub.to_csv('/kaggle/working/submission.csv', index=False)\n",
                "print(f'Submission saved: {sub.shape}')\n",
            ]},
        ]
    }
    output_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def run_transfer_exploration_phase(
    config: dict,
    *,
    logs_dir: Path,
    eval_dir: Path,
    llm: LLMClient,
    executor: CodeExecutor,
    evaluator: Evaluator,
    interactive_pick_final: bool = False,
) -> dict:
    """LLM-driven transfer learning exploration loop.

    Runs up to ``config['search']['transfer_exploration']['max_iterations']``
    iterations. Each iteration asks the LLM for a fresh
    ``build_model``/``get_training_config`` pair, executes it through the
    standard harness, evaluates ROC-AUC on the validation split, and feeds
    metrics back to the LLM. Returns the best slot code along with a
    best macro ROC-AUC score that the global selector can compare against
    the parametric CNN pipeline's score.
    """
    transfer_dir = logs_dir / "transfer_codes"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    tex = config.get("search", {}).get("transfer_exploration", {})
    max_iters = int(tex.get("max_iterations", 10))
    temp = config.get("llm", {}).get("temperature", 0.2)
    max_failures = int(tex.get("max_failures_before_stop", 5))

    print(f"  Transfer iterations: {max_iters}")
    print("  Strategy: LLM proposes architectures (CNN → pretrained on stagnation)")

    current_slot = TRANSFER_DEFAULT_SLOT_CODE
    best_slot = TRANSFER_DEFAULT_SLOT_CODE
    best_metrics = None
    best_auc = 0.0
    best_num_scored = 0
    metrics_history: list = []

    messages = [
        {"role": "system", "content": TRANSFER_SYSTEM_PROMPT},
        {"role": "user", "content": TRANSFER_SEED_PROMPT},
    ]
    consec_fail = 0

    for it in range(1, max_iters + 1):
        print(f"\n  ── Transfer iter {it}/{max_iters} ──")

        # 1) Ask the LLM for a slot.
        t0 = time.time()
        resp = llm.generate_from_messages(messages=messages, temperature=temp)
        print(f"    LLM responded in {time.time()-t0:.1f}s")

        slot = _transfer_extract_python_code(resp)
        if not slot:
            messages = _transfer_trim_and_append(messages, "assistant", resp)
            messages = _transfer_trim_and_append(
                messages, "user",
                "No ```python``` block found. Return ONLY code with get_training_config(), build_model(), and EXPERIMENT_META.",
            )
            consec_fail += 1
            metrics_history.append({"iteration": it, "status": "no_code"})
            if consec_fail >= max_failures:
                break
            continue

        issues = _transfer_validate_slot_code(slot)
        if issues:
            print(f"    Validation: {issues}")
            messages = _transfer_trim_and_append(messages, "assistant", resp)
            messages = _transfer_trim_and_append(
                messages, "user",
                f"Errors: {'; '.join(issues)}. Fix and return corrected code.",
            )
            consec_fail += 1
            metrics_history.append({"iteration": it, "status": "validation_failed"})
            if consec_fail >= max_failures:
                break
            continue

        current_slot = slot
        exp_meta = _transfer_extract_experiment_meta(slot)

        # 2) Save and execute through the standard harness.
        run_id = f"T{it:03d}"
        (transfer_dir / f"{run_id}_slot.py").write_text(slot, encoding="utf-8")
        full_script = assemble_script(slot)
        full_script = _append_eval_wrapper(
            full_script, run_id, eval_dir, mel_cache_dir=DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
        )
        script_path = transfer_dir / f"{run_id}.py"
        script_path.write_text(full_script, encoding="utf-8")

        if exp_meta:
            print(
                f"    [{exp_meta.get('model_type', '?')}] "
                f"{exp_meta.get('architecture', '')} — {exp_meta.get('change', '')}"
            )

        t0 = time.time()
        result = executor.run_file(script_path)
        dt = time.time() - t0
        print(f"    {'SUCCESS' if result.success else 'FAILED'} ({dt:.1f}s)")

        # 3) Evaluate.
        metrics_obj = None
        if result.success:
            consec_fail = 0
            yt = eval_dir / f"y_true_{run_id}.npy"
            yp = eval_dir / f"y_pred_{run_id}.npy"
            if yt.exists() and yp.exists():
                ev = evaluator.evaluate_from_files(yt, yp)
                if ev.metrics.get("status") == "success":
                    metrics_obj = ev.metrics
                    auc = float(metrics_obj["macro_roc_auc"])
                    n_scored = int(metrics_obj.get("num_scored_columns", 0))
                    print(f"    macro_roc_auc = {auc:.6f} | scored = {n_scored}")
                    if auc > best_auc:
                        best_auc = auc
                        best_slot = slot
                        best_metrics = metrics_obj
                        best_num_scored = n_scored
                        print(f"    ★ NEW BEST ({best_auc:.6f})")
                        (logs_dir / "best_transfer_slot.py").write_text(slot, encoding="utf-8")
            metrics_history.append({
                "iteration": it,
                "status": "success",
                "time": round(dt, 1),
                "macro_roc_auc": metrics_obj.get("macro_roc_auc") if metrics_obj else None,
                "num_scored_columns": metrics_obj.get("num_scored_columns") if metrics_obj else None,
                "experiment_meta": exp_meta,
            })
        else:
            consec_fail += 1
            print(f"    stderr: {_truncate(result.stderr, 200)}")
            metrics_history.append({
                "iteration": it,
                "status": "failed",
                "time": round(dt, 1),
                "experiment_meta": exp_meta,
            })
            if consec_fail >= max_failures:
                break

        # 4) Stagnation check over last 3 successful CNN runs.
        recent_aucs = [
            m["macro_roc_auc"] for m in metrics_history[-3:]
            if m["status"] == "success" and m.get("macro_roc_auc") is not None
            and m.get("experiment_meta", {}).get("model_type") == "cnn"
        ]
        cnn_stagnating = (
            len(recent_aucs) >= 3 and (max(recent_aucs) - min(recent_aucs)) < 0.02
        )

        # 5) Build the next feedback prompt.
        feedback = _transfer_build_feedback_prompt(
            stdout=result.stdout,
            stderr=result.stderr,
            is_error=not result.success,
            metrics=metrics_obj,
            current_code=current_slot,
            cnn_stagnating=cnn_stagnating,
        )
        messages = _transfer_trim_and_append(messages, "assistant", resp)
        messages = _transfer_trim_and_append(messages, "user", feedback)

    # Persist exploration log for global selection.
    _save_json(metrics_history, logs_dir / "transfer_metrics_history.json")

    effective_slot = best_slot if best_auc > 0 else None
    best_transfer_auc = best_auc if effective_slot is not None else None

    return {
        "effective_slot_code": effective_slot,
        "best_macro_roc_auc": best_transfer_auc,
        "best_num_scored_columns": best_num_scored if effective_slot is not None else None,
        "transfer_best_auc": best_transfer_auc,
        "metrics_history": metrics_history,
        "best_metrics": best_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INSIGHTS & NOTEBOOK GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _generate_insights(llm, results, phase, dirs, temperature):
    ok = [r for r in results if r.get("success") and r.get("macro_roc_auc")]
    if not ok:
        (dirs["logs"] / f"{phase}_final_result_insights.txt").write_text(
            f"No successful experiments in {phase} phase.\n", encoding="utf-8")
        return
    ok.sort(key=lambda r: r["macro_roc_auc"], reverse=True)
    lines = []
    for i, r in enumerate(ok):
        l = f"{i+1}. AUC={r['macro_roc_auc']:.6f} | {r['description']}"
        if r.get("dimension"): l += f" [{r['dimension']}]"
        if r.get("search_type"): l += f" [{r['search_type']}]"
        lines.append(l)
    prompt = (
        f"Results from {phase} search for bird species audio classification (BirdCLEF 2026).\n"
        f"234 species, mel spectrograms, metric=macro_roc_auc, trained on 300 samples 3 epochs.\n\n"
        f"Results (best to worst):\n" + "\n".join(lines) + "\n\n"
        "Write a concise analysis (~half page): biggest impact params, best values, "
        "surprises, recommendations for the final model. Plain text, no code."
    )
    print(f"\n  Generating {phase} insights...")
    resp = llm.generate_from_messages(
        messages=[{"role":"system","content":"You are an ML research analyst."},
                  {"role":"user","content":prompt}],
        temperature=temperature)
    out = dirs["logs"] / f"{phase}_final_result_insights.txt"
    out.write_text(resp, encoding="utf-8")
    print(f"  Saved to {out}")


def _maybe_generate_insights(config, llm, results, phase, dirs, temperature):
    if not config.get("logging", {}).get("generate_insights", True):
        print(f"\n  Skipping {phase} insights (generate_insights=false).")
        return
    _generate_insights(llm, results, phase, dirs, temperature)


def _generate_kaggle_notebook(params, output_path):
    nm, nf = params.get("n_mels", 64), params.get("n_frames", 128)
    nb = {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# BirdCLEF+ 2026 — Inference Notebook\n",
                "Auto-generated by structured search agent.\n",
                f"Config: {describe_params(params)}\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import os, numpy as np, pandas as pd, librosa, tensorflow as tf\n",
                "from pathlib import Path\n",
                f"\nSR = 32000\nCLIP_SECONDS = 5.0\nN_MELS = {nm}\nN_FRAMES = {nf}\n",
                "N_FFT = 1024\nHOP_LENGTH = 512\n",
                "\nCOMP_DIR = '/kaggle/input/birdclef-2026'\n",
                "MODEL_DIR = '/kaggle/input/birdclef-model'\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model.keras'))\n",
                "sample_sub = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))\n",
                "species_cols = [c for c in sample_sub.columns if c != 'row_id']\n",
                "print(f'Model loaded. Species: {len(species_cols)}')\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "test_dir = Path(COMP_DIR) / 'test_soundscapes'\n",
                "test_files = sorted(test_dir.glob('*.ogg')) if test_dir.exists() else []\n",
                "print(f'Test soundscapes: {len(test_files)}')\n",
                "\nall_rows = []\nseg_samples = int(SR * CLIP_SECONDS)\n",
                "\nfor fpath in test_files:\n",
                "    name = fpath.stem\n",
                "    y_full, _ = librosa.load(str(fpath), sr=SR, mono=True)\n",
                "    for start in range(0, len(y_full), seg_samples):\n",
                "        seg = y_full[start:start+seg_samples]\n",
                "        if len(seg) < seg_samples:\n",
                "            seg = np.pad(seg, (0, seg_samples - len(seg)))\n",
                "        end_sec = (start // seg_samples + 1) * CLIP_SECONDS\n",
                "        mel = librosa.feature.melspectrogram(\n",
                "            y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)\n",
                "        mel_db = librosa.power_to_db(mel, ref=np.max)\n",
                "        mel_r = tf.image.resize(mel_db[..., np.newaxis], (N_MELS, N_FRAMES)).numpy().astype(np.float32)\n",
                "        preds = model.predict(mel_r[np.newaxis, ...], verbose=0)[0]\n",
                "        row = {'row_id': f'{name}_{end_sec}'}\n",
                "        for col, p in zip(species_cols, preds):\n",
                "            row[col] = float(p)\n",
                "        all_rows.append(row)\n",
                "\nprint(f'Prediction rows: {len(all_rows)}')\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "sub = pd.DataFrame(all_rows) if all_rows else sample_sub.copy()\n",
                "sub = sample_sub[['row_id']].merge(sub, on='row_id', how='left')\n",
                "sub[species_cols] = sub[species_cols].fillna(0.0).clip(0.0, 1.0)\n",
                "sub.to_csv('/kaggle/working/submission.csv', index=False)\n",
                "print(f'Submission saved: {sub.shape}')\n"]},
        ]
    }
    output_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: FINAL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _final_execution_timeout_seconds(config: dict) -> int | None:
    """Timeout for Phase 3 subprocess. None = no limit (run until finish)."""
    ex = config.get("execution", {})
    if "final_timeout_seconds" not in ex:
        return None
    return ex["final_timeout_seconds"]


def _run_final_training(
    best_params,
    executor,
    evaluator,
    llm,
    temperature,
    project_root,
    dirs,
    search_cfg,
    best_slot_code=None,
    best_epoch_override=None,
    forced_final_epochs=None,
    notebook_kind: str = "cnn",
):
    print("\n" + "=" * 60)
    print("  PHASE 3: FINAL TRAINING")
    print("=" * 60)

    sub_dir = project_root / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(sub_dir / "model.keras")

    fp = dict(best_params)
    fp["max_samples"] = search_cfg["final"].get("max_samples")
    if forced_final_epochs is not None:
        fp["epochs"] = int(forced_final_epochs)
    elif best_epoch_override is not None:
        fp["epochs"] = int(best_epoch_override)
    else:
        fp["epochs"] = int(search_cfg["final"]["epochs"])
    fp["val_split"] = 0.0

    if best_slot_code:
        slot_code = best_slot_code + _final_config_override_block({
            "max_samples": fp["max_samples"],
            "sample_rate": 32000,
            "clip_seconds": 5.0,
            "n_mels": fp.get("n_mels", 64),
            "n_frames": fp.get("n_frames", 128),
            "epochs": fp["epochs"],
            "batch_size": fp.get("batch_size", 32),
            "learning_rate": fp.get("learning_rate", 1e-3),
            "optimizer": fp.get("optimizer", "adam"),
            "val_split": 0.0,
            "weight_decay": fp.get("weight_decay", 0.0),
            "classifier_hidden_units": fp.get("classifier_hidden_units", 0),
            "pooling_type": fp.get("pooling_type", "global_avg"),
            "use_best_checkpoint": True,
        })
    else:
        slot_code = generate_slot_code(fp)
    script = assemble_script(
        slot_code,
        is_final=True,
        model_save_path=model_path,
        focal_cache_dir=DEFAULT_FOCAL_MEL_CACHE_DIR,
    )
    script = _append_eval_wrapper(
        script, "final", dirs["eval"], mel_cache_dir=DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    )

    script_path = dirs["logs"] / "final_run.py"
    script_path.write_text(script, encoding="utf-8")
    (sub_dir / "best_model_code.py").write_text(slot_code, encoding="utf-8")

    print(f"  Config: {describe_params(best_params)}")
    best_epoch = best_params.get("epochs")
    if best_epoch is not None:
        print(f"  Best epochs found: {best_epoch}")
    print(f"  Epochs: {fp['epochs']}, Samples: {'ALL' if fp['max_samples'] is None else fp['max_samples']}")
    print("  Final split policy: val_split=0.0 (train on all data).")
    if best_epoch_override is not None:
        print(f"  Epoch source: medium/gate best epoch = {best_epoch_override}")
    print(f"  Final architecture source: {'winning slot code' if best_slot_code else 'parametric generator'}")

    t0 = time.time()
    result = executor.run_file(script_path)
    dt = time.time() - t0

    if result.success:
        print(f"  ✓ SUCCESS ({dt:.1f}s)")
        yt, yp = dirs["eval"] / "y_true_final.npy", dirs["eval"] / "y_pred_final.npy"
        if yt.exists() and yp.exists():
            ev = evaluator.evaluate_from_files(yt, yp)
            if ev.metrics.get("status") == "success":
                print(f"  macro_roc_auc = {ev.metrics['macro_roc_auc']:.6f}")
                print(f"  scored_species = {ev.metrics['num_scored_columns']}")
                print(f"  samples = {ev.metrics['num_samples']}")
        if Path(model_path).exists():
            print(f"  Model: {model_path} ({Path(model_path).stat().st_size/1024/1024:.1f} MB)")
        nb_path = sub_dir / "kaggle_inference.ipynb"
        if notebook_kind == "transfer" and best_slot_code:
            _generate_transfer_kaggle_notebook(best_slot_code, nb_path)
        else:
            _generate_kaggle_notebook(best_params, nb_path)
        print(f"  Kaggle notebook: {nb_path}")
        _save_json(best_params, sub_dir / "best_params.json")
        print(f"\n  submission/ contents:")
        for f in sorted(sub_dir.iterdir()):
            print(f"    {f.name} ({f.stat().st_size/1024:.1f} KB)")
    else:
        print(f"  ✗ FAILED ({dt:.1f}s) — attempting LLM fix...")
        slot_code, metrics, _, att = run_experiment_until_success(
            slot_code, "final_retry", dirs["logs"], dirs["eval"], executor, evaluator, llm, temperature)
        if metrics and metrics.get("status") == "success":
            print(f"  Fixed after {att} attempts. AUC={metrics['macro_roc_auc']:.6f}")
            (sub_dir / "best_model_code.py").write_text(slot_code, encoding="utf-8")
            if notebook_kind == "transfer" and slot_code:
                _generate_transfer_kaggle_notebook(slot_code, sub_dir / "kaggle_inference.ipynb")
            else:
                _generate_kaggle_notebook(best_params, sub_dir / "kaggle_inference.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 ONLY (standalone rerun)
# ═══════════════════════════════════════════════════════════════════════════


def _load_best_params_for_final(project_root: Path, best_params_path: Path | None) -> dict:
    """Resolve hyperparameters for final training from explicit file or saved search outputs."""
    if best_params_path is not None and best_params_path.exists():
        data = json.loads(best_params_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "params" in data and isinstance(data["params"], dict):
            return {**DEFAULT_PARAMS, **data["params"]}
        if isinstance(data, dict):
            return {**DEFAULT_PARAMS, **data}

    comp = project_root / "logs" / "search_comparison.json"
    if comp.exists():
        data = json.loads(comp.read_text(encoding="utf-8"))
        fp = data.get("final_params")
        if isinstance(fp, dict):
            return {**DEFAULT_PARAMS, **fp}

    bp = project_root / "submission" / "best_params.json"
    if bp.exists():
        data = json.loads(bp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {**DEFAULT_PARAMS, **data}

    return dict(DEFAULT_PARAMS)


def run_phase3_final_only(
    project_root: Path | None = None,
    config_path: Path | None = None,
    *,
    best_params_path: Path | None = None,
) -> None:
    """
    Run Phase 3 (final training + eval artifacts + notebook) only.

    Loads best hyperparameters from, in order:
    1) ``best_params_path`` if provided and exists
    2) ``logs/search_comparison.json`` → ``final_params``
    3) ``submission/best_params.json``
    4) ``DEFAULT_PARAMS``

    Uses ``execution.final_timeout_seconds`` from config (``null`` = no subprocess timeout).
    """
    root = project_root or Path(__file__).resolve().parents[1]
    cfg_path = config_path or (root / "configs" / "agent_config.json")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    logs = root / "logs"
    dirs = {
        "logs": logs,
        "eval": logs / "eval_artifacts",
        "linear_codes": logs / "linear_codes",
        "random_codes": logs / "random_codes",
        "baseline_codes": logs / "baseline_codes",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    sc = config.get("search", {})
    sc.setdefault("final", {"max_samples": None, "epochs": 20, "val_split": 0.1})

    best_params = _load_best_params_for_final(root, best_params_path)

    py_exe = config["execution"]["python_executable"]
    executor_final = CodeExecutor(
        python_executable=py_exe,
        timeout_seconds=_final_execution_timeout_seconds(config),
    )
    evaluator = Evaluator(row_id_column_name="row_id")
    llm = LLMClient(provider=config["llm"]["provider"], model=config["llm"]["model"])
    temp = config["llm"].get("temperature", 0.2)

    print("=" * 60)
    print("  PHASE 3 ONLY (final training)")
    print("=" * 60)
    print(f"  Params: {describe_params(best_params)}")
    ft = _final_execution_timeout_seconds(config)
    print(f"  Subprocess timeout: {'unlimited' if ft is None else f'{ft}s'}")
    print("=" * 60)

    _run_final_training(best_params, executor_final, evaluator, llm, temp, root, dirs, sc)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def agent_loop(config):
    global GENERATION_SYSTEM_PROMPT, TRANSFER_SYSTEM_PROMPT

    if config.get("cnn_staged"):
        try:
            from .cnn_staged import dispatch_cnn_staged
        except ImportError:
            from cnn_staged import dispatch_cnn_staged
        dispatch_cnn_staged(config)
        return

    root = Path(__file__).resolve().parents[1]
    cnn_cfg = config.get("cnn", {})
    logs = Path(cnn_cfg["logs_dir"]) if cnn_cfg.get("logs_dir") else root / "logs"
    preset = config.get("meta_aug_preset")
    if preset:
        print(f"  Meta aug baseline: {preset}")
    dirs = {k: logs / v for k, v in {
        "logs": "", "eval": "eval_artifacts", "linear_codes": "linear_codes",
        "random_codes": "random_codes", "baseline_codes": "baseline_codes"}.items()}
    dirs["logs"] = logs
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── EDA brief from meta Phase 0 (short 2-sentence summary only) ─────────────
    eda_brief = (config.get("eda_brief") or "").strip()
    if eda_brief:
        _eda_block = (
            "\n\n## DATA INSIGHTS (EDA — factual, data only)\n"
            + eda_brief
            + "\n## END OF EDA INSIGHTS\n"
        )
        GENERATION_SYSTEM_PROMPT = GENERATION_SYSTEM_PROMPT + _eda_block
        TRANSFER_SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT + _eda_block
        print(f"  EDA brief injected into CNN prompts ({len(eda_brief)} chars)")
    else:
        eda_cfg = config.get("eda", {})
        # Meta-agent subprocesses (staged 1a per baseline) must not re-run EDA each time.
        if config.get("arch_search_only") or config.get("meta_aug_preset"):
            eda_default = False
        else:
            eda_default = True
        if eda_cfg.get("enabled", eda_default):
            py_exe = config["execution"]["python_executable"]
            _eda_executor = CodeExecutor(python_executable=py_exe, timeout_seconds=120)
            _eda_llm = LLMClient(provider=config["llm"]["provider"], model=config["llm"]["model"])
            _eda_temp = config["llm"].get("temperature", 0.4)
            eda_insights = run_eda_phase(_eda_executor, _eda_llm, dirs["logs"], temperature=_eda_temp)
            if eda_insights.strip():
                _eda_block = (
                    "\n\n## DATA INSIGHTS (from autonomous EDA before training)\n"
                    + eda_insights.strip()
                    + "\n## END OF EDA INSIGHTS\n"
                )
                GENERATION_SYSTEM_PROMPT = GENERATION_SYSTEM_PROMPT + _eda_block
                TRANSFER_SYSTEM_PROMPT = TRANSFER_SYSTEM_PROMPT + _eda_block
        else:
            print("  EDA phase disabled (eda.enabled=false in config)")

    sc = config.get("search", {})
    if config.get("arch_search_only"):
        sc["linear_budget"] = 0
        sc.setdefault("phase2", {})["augmentation_tweak_experiments"] = 0
        sc.setdefault("cnn_exploration", {})["enabled"] = True
        sc.setdefault("transfer_exploration", {})["enabled"] = False
        sc.setdefault("medium_stage", {})["enabled"] = False
        sc.setdefault("reality_gate", {})["enabled"] = False
        sc["skip_final_training"] = True
    sc.setdefault("cheap", {"max_samples": 1000, "epochs": 3, "val_split": 0.2})
    sc.setdefault("medium", {"max_samples": 10000, "epochs": 15, "val_split": 0.2})
    sc.setdefault("final", {"max_samples": None, "epochs": 20, "val_split": 0.1})
    sc.setdefault("linear_budget", 75)
    sc.setdefault("random_budget", 50)
    p2 = sc.setdefault("phase2", {})
    p2.setdefault("random_experiments", 50)
    p2.setdefault("focused_experiments", 50)
    p2.setdefault("tweak_experiments", 50)
    p2.setdefault("augmentation_tweak_experiments", 5)
    p2.setdefault("ai_free_experiments", 50)
    p2.setdefault("final_tweak_experiments", 15)
    p2.setdefault("top_results_keep", 4)
    ms = sc.setdefault("medium_stage", {})
    ms.setdefault("enabled", True)
    ms.setdefault("total_runs", 10)
    ms.setdefault("promoted_runs", 4)
    ms.setdefault("resume_from_saved_phase2", False)
    rg = sc.setdefault("reality_gate", {})
    rg.setdefault("enabled", True)
    rg.setdefault("use_all_candidates", True)
    rg.setdefault("top_k", 5)
    rg.setdefault("max_samples", sc["medium"]["max_samples"])
    rg.setdefault("epochs", sc["medium"]["epochs"])
    rg.setdefault("val_split", sc["medium"].get("val_split", 0.2))
    rg.setdefault("split_mode", "group_holdout")
    sc.setdefault("random_seed", config.get("random_seed", 42))
    cnn_expl = sc.setdefault("cnn_exploration", {})
    cnn_expl.setdefault("enabled", True)
    tex = sc.setdefault("transfer_exploration", {})
    tex.setdefault("enabled", True)
    tex.setdefault("max_iterations", 10)
    tex.setdefault("interactive_pick_final", False)

    llm = LLMClient(provider=config["llm"]["provider"], model=config["llm"]["model"])

    # Dual-LLM: optional researcher (e.g. DeepSeek R1) that reasons about results
    # before the coder LLM proposes the next architecture.
    llm_researcher = None
    rc_cfg = config.get("llm_researcher", {})
    if rc_cfg.get("enabled", False):
        llm_researcher = LLMClient(provider=rc_cfg["provider"], model=rc_cfg["model"])
        print(f"  Researcher: {rc_cfg['model']} (dual-LLM mode)")
    else:
        print("  Researcher: disabled (single-LLM mode)")
    researcher_temp = rc_cfg.get("temperature", 0.6)

    py_exe = config["execution"]["python_executable"]
    search_timeout = config["execution"].get("timeout_seconds", 1800)
    executor_search = CodeExecutor(python_executable=py_exe, timeout_seconds=search_timeout)
    executor_final = CodeExecutor(
        python_executable=py_exe,
        timeout_seconds=_final_execution_timeout_seconds(config),
    )
    evaluator = Evaluator(row_id_column_name="row_id")
    temp = config["llm"].get("temperature", 0.2)

    print("=" * 60)
    print("  BirdCLEF Structured Search Agent")
    print("=" * 60)
    print(f"  LLM:     {config['llm']['model']}")
    print(f"  Linear:  {sc['linear_budget']} iters")
    print(
        "  Phase2: "
        f"A={p2['random_experiments']} random, "
        f"B={p2['focused_experiments']} focused, "
        f"C={p2['tweak_experiments']} tweaks, "
        f"C-aug={p2['augmentation_tweak_experiments']} aug-tweaks, "
        f"D={p2['ai_free_experiments']} ai-free, "
        f"C-final={p2['final_tweak_experiments']} tweaks"
    )
    print(f"  Cheap:   {sc['cheap']['max_samples']} samples, {sc['cheap']['epochs']} epochs")
    print(
        f"  Medium:  {sc['medium']['max_samples']} samples, {sc['medium']['epochs']} epochs "
        f"(runs={ms['total_runs']}, promoted={ms['promoted_runs']}, "
        f"resume_from_saved_phase2={ms['resume_from_saved_phase2']})"
    )
    print(
        f"  Gate:    enabled={rg['enabled']} top_k={rg['top_k']} "
        f"samples={rg['max_samples']} epochs={rg['epochs']} split={rg['split_mode']}"
    )
    print(f"  Final:   {'ALL' if sc['final']['max_samples'] is None else sc['final']['max_samples']} samples, {sc['final']['epochs']} epochs")
    print(f"  CNN exploration: enabled={cnn_expl.get('enabled', True)}")
    print(f"  Transfer exploration: enabled={tex.get('enabled', True)} iters={tex.get('max_iterations', 10)}")
    ft = _final_execution_timeout_seconds(config)
    print(f"  Final subprocess timeout: {'unlimited' if ft is None else f'{ft}s'}")
    print("=" * 60)

    t0 = time.time()

    if ms["resume_from_saved_phase2"]:
        print("\n" + "=" * 60)
        print("  RESUME MODE: STARTING FROM MEDIUM STAGE")
        print("=" * 60)
        lin_results, lin_best, rnd_results, best_rnd = _load_saved_search_state(dirs["logs"])
        print(f"  Loaded {len(lin_results)} linear results from {dirs['logs'] / 'linear.json'}")
        print(f"  Loaded {len(rnd_results)} phase-2 results from {dirs['logs'] / 'random_results.json'}")
    elif not cnn_expl.get("enabled", True):
        print("\n" + "=" * 60)
        print("  CNN EXPLORATION DISABLED — skipping baseline → random → medium pipeline")
        print("=" * 60)
        lin_results, lin_best = [], dict(DEFAULT_PARAMS)
        rnd_results, best_rnd = [], None
    elif config.get("arch_search_only"):
        print("\n" + "=" * 60)
        print("  ARCH SEARCH ONLY — random search with locked augmentation baseline")
        print("=" * 60)
        locked = _locked_cnn_aug(config)
        lin_best = {**dict(DEFAULT_PARAMS), **locked}
        baseline_slot = generate_slot_code({**lin_best, **sc["cheap"]})
        lin_results = []
        rnd_results, best_rnd = _run_random_search(
            baseline_slot, executor_search, evaluator, llm, temp, dirs, sc,
            llm_researcher=llm_researcher, researcher_temp=researcher_temp,
            config=config,
        )
    else:
        baseline_slot, _ = _run_baseline(executor_search, evaluator, llm, temp, dirs, sc)
        lin_results, lin_best = _run_linear_search(baseline_slot, executor_search, evaluator, llm, temp, dirs, sc)
        _maybe_generate_insights(config, llm, lin_results, "linear", dirs, temp)
        rnd_results, best_rnd = _run_random_search(
            baseline_slot, executor_search, evaluator, llm, temp, dirs, sc,
            llm_researcher=llm_researcher, researcher_temp=researcher_temp,
            config=config,
        )
        _maybe_generate_insights(config, llm, rnd_results, "random", dirs, temp)

    lin_ok = [r for r in lin_results if r.get("success") and r.get("macro_roc_auc")]
    best_lin_auc = max((r["macro_roc_auc"] for r in lin_ok), default=0.0)
    best_rnd_auc = best_rnd["macro_roc_auc"] if best_rnd else 0.0

    med_results, best_med = [], None
    gate_results, best_gate = [], None
    best_med_auc, best_gate_auc = 0.0, 0.0
    if cnn_expl.get("enabled", True) and ms.get("enabled", True):
        pre_medium_results = lin_results + rnd_results
        med_results, best_med = _run_medium_stage(pre_medium_results, executor_search, evaluator, llm, temp, dirs, sc)
        _maybe_generate_insights(config, llm, med_results, "medium", dirs, temp)
        best_med_auc = best_med["macro_roc_auc"] if best_med else 0.0

        gate_pool = _best_successful(med_results, len(med_results))
        gate_results, best_gate = _run_reality_check_gate(gate_pool, executor_search, evaluator, dirs, sc)
        best_gate_auc = best_gate["macro_roc_auc"] if best_gate else 0.0
    elif not cnn_expl.get("enabled", True):
        print("\n  Medium/gate skipped (CNN exploration disabled).")
    else:
        print("\n  Medium stage disabled by config (search.medium_stage.enabled=false).")

    print("\n" + "=" * 60)
    print("  COMPARISON: LINEAR vs RANDOM vs MEDIUM (+ GATE)")
    print("=" * 60)
    print(f"  Linear: AUC={best_lin_auc:.6f} | {describe_params(lin_best)}")
    print(f"  Random: AUC={_fmt_score(best_rnd_auc)} | {best_rnd['description'] if best_rnd else 'N/A'}")
    print(f"  Medium: AUC={_fmt_score(best_med_auc)} | {best_med['description'] if best_med else 'N/A'}")
    print(f"  Gate:   AUC={_fmt_score(best_gate_auc)} | {best_gate['description'] if best_gate else 'N/A'}")

    winner = "cnn_disabled"
    winner_auc: float | None = None
    best_params = dict(DEFAULT_PARAMS)
    best_slot_code = None
    winner_best_epoch = None
    phase1_num_scored = None

    if cnn_expl.get("enabled", True):
        if best_gate:
            winner, winner_auc = "gate", best_gate_auc
            best_params = best_gate["params"]
            best_slot_code = best_gate.get("slot_code")
            winner_best_epoch = best_gate.get("best_epoch")
            phase1_num_scored = best_gate.get("num_scored_columns")
            print("  Selection policy: gate winner (primary).")
        elif best_med:
            winner, winner_auc = "medium_fallback", best_med_auc
            best_params = best_med["params"]
            best_slot_code = best_med.get("slot_code")
            winner_best_epoch = best_med.get("best_epoch")
            phase1_num_scored = best_med.get("num_scored_columns")
            print("  Selection policy: medium fallback (gate had no successful run).")
        else:
            candidates = [
                (
                    "linear_fallback",
                    best_lin_auc,
                    lin_best,
                    generate_slot_code(
                        {
                            **lin_best,
                            "max_samples": sc["cheap"]["max_samples"],
                            "epochs": sc["cheap"]["epochs"],
                            "val_split": sc["cheap"].get("val_split", 0.2),
                        }
                    ),
                    None,
                )
            ]
            if best_rnd:
                candidates.append(
                    ("random_fallback", best_rnd_auc, best_rnd["params"], best_rnd.get("slot_code"), best_rnd.get("best_epoch"))
                )
            winner, winner_auc, best_params, best_slot_code, winner_best_epoch = max(candidates, key=lambda x: x[1])
            phase1_num_scored = None
            print("  Selection policy: emergency fallback (no gate/medium success).")
        print(f"\n  ★ CNN PHASE WINNER: {winner} (AUC={winner_auc:.6f})")
    else:
        print("\n  ★ CNN PHASE SKIPPED (no Phase-1 winner).")

    phase1_auc_for_compare = float(winner_auc) if isinstance(winner_auc, (int, float)) else -1.0
    print(f"  Phase-1 best AUC for global comparison: {phase1_auc_for_compare:.6f}")

    transfer_pack = None
    transfer_slot = None
    transfer_best_auc = -1.0
    if tex.get("enabled", True):
        print("\n" + "=" * 60)
        print("  TRANSFER EXPLORATION (pretrained / ImageNet backbones)")
        print("=" * 60)
        transfer_pack = run_transfer_exploration_phase(
            config,
            logs_dir=logs,
            eval_dir=dirs["eval"],
            llm=llm,
            executor=executor_search,
            evaluator=evaluator,
            interactive_pick_final=bool(tex.get("interactive_pick_final", False)),
        )
        transfer_slot = transfer_pack.get("effective_slot_code")
        transfer_best_auc = float(transfer_pack.get("best_macro_roc_auc") or -1.0)
        print(f"  Transfer best AUC: {transfer_best_auc:.6f}")

    global_winner = "cnn"
    notebook_kind = "cnn"
    forced_fe = None
    if transfer_pack and transfer_slot is not None and transfer_best_auc > phase1_auc_for_compare:
        global_winner = "transfer"
        notebook_kind = "transfer"
        best_slot_code = transfer_slot
        best_params = _best_params_from_transfer_slot(transfer_slot)
        winner_best_epoch = None
        tfo = config.get("transfer", {}).get("final_epochs_override")
        if tfo is not None:
            forced_fe = int(tfo)
        print(f"\n  ★ GLOBAL WINNER: transfer exploration (AUC={transfer_best_auc:.6f} > phase1={phase1_auc_for_compare:.6f})")
    elif phase1_auc_for_compare >= 0 or (
        cnn_expl.get("enabled", True)
        and (best_lin_auc > 0 or best_rnd_auc > 0 or best_med_auc > 0 or best_gate_auc > 0)
    ):
        print(f"\n  ★ GLOBAL WINNER: CNN pipeline ({winner}) AUC={phase1_auc_for_compare:.6f}")
    elif transfer_pack and transfer_slot is not None:
        global_winner = "transfer"
        notebook_kind = "transfer"
        best_slot_code = transfer_slot
        best_params = _best_params_from_transfer_slot(transfer_slot)
        winner_best_epoch = None
        tfo = config.get("transfer", {}).get("final_epochs_override")
        if tfo is not None:
            forced_fe = int(tfo)
        print("\n  ★ GLOBAL WINNER: transfer exploration (CNN pipeline had no comparable score)")
    else:
        global_winner = "none"

    _save_json(
        {
            "best_linear_auc": best_lin_auc,
            "best_linear_params": lin_best,
            "best_random_auc": best_rnd_auc,
            "best_random_params": best_rnd["params"] if best_rnd else None,
            "best_medium_auc": best_med_auc,
            "best_medium_params": best_med["params"] if best_med else None,
            "best_gate_auc": best_gate_auc,
            "best_gate_params": best_gate["params"] if best_gate else None,
            "cnn_exploration_enabled": cnn_expl.get("enabled", True),
            "phase1_winner": winner,
            "phase1_winner_auc": winner_auc,
            "phase1_selection_score": None,
            "transfer_exploration_enabled": tex.get("enabled", True),
            "transfer_adjusted_score": None,
            "transfer_best_auc": transfer_best_auc if transfer_pack else None,
            "global_winner": global_winner,
            "winner_best_epoch": winner_best_epoch,
            "final_params": best_params,
            "winner_slot_code_path": str(logs / "winner_slot_code.py"),
        },
        logs / "search_comparison.json",
    )
    if best_slot_code:
        (logs / "winner_slot_code.py").write_text(best_slot_code, encoding="utf-8")

    total = len(lin_results) + len(rnd_results) + len(med_results) + len(gate_results) + 1
    if transfer_pack:
        total += len(transfer_pack.get("metrics_history", []))
    print(f"\n  Total: {total} experiments in {(time.time()-t0)/60:.1f} min")

    ran_anything = (
        best_lin_auc > 0
        or best_rnd_auc > 0
        or best_med_auc > 0
        or best_gate_auc > 0
        or (transfer_pack and transfer_pack.get("effective_slot_code") is not None)
    )
    skip_final = bool(sc.get("skip_final_training", False))
    if skip_final:
        print("\n  Final training skipped (search.skip_final_training=true).")
        print("  Use best search checkpoint / winner_slot_code.py for submission artifacts.")
    elif ran_anything and best_slot_code:
        _run_final_training(
            best_params,
            executor_final,
            evaluator,
            llm,
            temp,
            root,
            dirs,
            sc,
            best_slot_code=best_slot_code,
            best_epoch_override=winner_best_epoch,
            forced_final_epochs=forced_fe,
            notebook_kind=notebook_kind,
        )
    else:
        print("\n  No successful experiments — skipping final training.")


def main():
    import argparse
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "configs" / "agent_config.json"))
    args   = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent_loop(config)

if __name__ == "__main__":
    main()
