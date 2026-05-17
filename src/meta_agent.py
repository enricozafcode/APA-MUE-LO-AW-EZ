"""
Meta Agent — BirdCLEF 2026
===========================
Legacy pipeline (``meta_agent.pipeline``: ``legacy``):
  Phase 0: EDA
  Phase 1–3: CNN / BirdNET / Perch single-pass agents
  Phase 4: Ensemble

Staged pipeline (``meta_agent.pipeline``: ``staged_1a``) — architecture search only:
  For each track (CNN, BirdNET, Perch) and each aug baseline (light / medium / high):
    run the track agent with locked augmentation → pick track winner by soundscape AP.

Stage 1b (optional): refine top-K perch models from 1a with adaptive iteration budget.
Stage 1c: LLM aug explore (wide search) + optional refine on winner; fixed 2000 head-train indices.
Stage 1d: full-data final retrain (best aug + best head) → ``logs/meta_agent/perch/final/``.
Use ``pipeline: staged_1c_only`` to run only 1c+1d after 1a/1b are done.
Set ``stage_1c.mode`` to ``llm`` | ``presets`` | ``both``; tune ``explore`` / ``refine`` blocks in config.

Run:
    python src/meta_agent.py --config configs/agent_config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from augmentation import (
    describe_baseline,
    get_aug_search_preset,
    get_audio_embedding_aug,
    get_cnn_baseline_aug,
    list_aug_search_preset_names,
    list_baseline_aug_names,
)
from soundscape_evaluator import (
    PRIMARY_META_METRIC,
    SoundscapeEvalSuite,
    SoundscapeScore,
    competition_macro_auc,
    format_soundscape_metrics_line,
    format_soundscape_score,
    macro_average_precision,
    median_per_class_auc,
    primary_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "notebooks" / "birdnet_cache"
PERCH_MEMORY = PROJECT_ROOT / "logs" / "perch_memory"
BIRDNET_LOGS = PROJECT_ROOT / "logs" / "birdnet_agent"
CNN_SUBMISSION = PROJECT_ROOT / "submission" / "model.keras"
META_LOGS    = PROJECT_ROOT / "logs" / "meta_agent"

SOUNDSCAPE_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
TRAIN_SOUNDSCAPES = DATA_DIR / "train_soundscapes"

BIRDNET_VAL_CACHE = CACHE_DIR / "val_emb1024.npz"
META_VAL_CACHE    = META_LOGS / "common_val.npz"
SOUNDSCAPE_LEADERBOARD = META_LOGS / "soundscape_leaderboard.json"
ARCH_SEARCH_1A_RESULTS = META_LOGS / "arch_search_1a_results.json"
ARCH_SEARCH_1B_RESULTS = META_LOGS / "arch_search_1b_results.json"
ARCH_SEARCH_1C_RESULTS = META_LOGS / "arch_search_1c_results.json"
HEAD_TRAIN_INDICES = META_LOGS / "perch" / "head_train_indices_2000.npy"
HEAD_TRAIN_CLIPS = META_LOGS / "perch" / "head_train_clips_2000.jsonl"
PERCH_SHARED_VAL_CACHE = META_LOGS / "perch_cache" / "val_emb.npz"
PERCH_FINAL_DIR = META_LOGS / "perch" / "final"

SR           = 32_000
CLIP_SEC     = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
PERCH_DIM    = 1536
PYTHON       = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _meta_primary_metric(config: dict) -> str:
    return str(
        config.get("meta_agent", {}).get("primary_metric", PRIMARY_META_METRIC)
    )


def _soundscape_suite(config: dict) -> SoundscapeEvalSuite:
    return SoundscapeEvalSuite(DATA_DIR, primary_metric=_meta_primary_metric(config))


def _print_soundscape_score(label: str, score: SoundscapeScore | None) -> None:
    if score is None:
        print(f"  [{label}] soundscape eval skipped (missing artifacts)")
        return
    print(
        f"  [{label}] {format_soundscape_score(score)} "
        f"| n_windows={score.n_windows} n_classes={score.n_scored_classes}"
    )


def _meta_cfg(config: dict) -> dict:
    return config.get("meta_agent", {})


def _baseline_names(config: dict) -> list[str]:
    names = _meta_cfg(config).get("aug_baselines")
    if names:
        return [str(n).lower() for n in names]
    return list_baseline_aug_names()


def _stage_1b_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1b") or {})


def _stage_1c_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1c") or {})


def _stage_1d_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1d") or {})


def _arch_iters_per_baseline(config: dict, track_iters: int) -> int:
    """Iterations per aug baseline when running staged 1a."""
    per = _meta_cfg(config).get("arch_search_iterations_per_aug")
    # Explicit positive value wins; 0 / null → split track_iters across aug baselines.
    if per is not None and int(per) > 0:
        return int(per)
    n_aug = max(1, len(_baseline_names(config)))
    return max(1, int(track_iters) // n_aug)


def _score_cnn_arch_search(logs_dir: Path, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    """Score best CNN search run from saved soundscape eval artifacts (no final keras needed)."""
    rr_path = logs_dir / "random_results.json"
    if not rr_path.exists():
        return None
    results = json.loads(rr_path.read_text(encoding="utf-8"))
    ok = [r for r in results if r.get("success")]
    if not ok:
        return None
    rank_key = (
        "macro_average_precision"
        if suite.primary_metric == PRIMARY_META_METRIC
        else "macro_roc_auc"
    )

    def _rank_val(r: dict) -> float:
        v = r.get("ranking_value") or r.get(rank_key) or r.get("macro_roc_auc")
        return float(v) if v is not None else -1.0

    best = max(ok, key=_rank_val)
    run_id = str(best.get("run_id", ""))
    eval_dir = logs_dir / "eval_artifacts"
    for suffix in ("_a5", "_a4", "_a3", "_a2", "_a1", ""):
        yp_path = eval_dir / f"y_pred_{run_id}{suffix}.npy"
        if yp_path.exists():
            y_pred = np.load(yp_path).astype(np.float32)
            return suite.score_arrays(y_pred)
    return None


def _run_subprocess(script: str, config_override: dict, base_config: dict) -> int:
    """Write a temp config with overrides and run a script as subprocess."""
    cfg = json.loads(json.dumps(base_config))
    cfg.update(config_override)
    # Child agents (CNN / BirdNET / Perch) must not re-run EDA — only meta Phase 0 does.
    if script != "eda_agent.py":
        cfg.setdefault("eda", {})
        if not config_override.get("eda", {}).get("enabled", False):
            cfg["eda"]["enabled"] = False
    tmp = Path(tempfile.gettempdir()) / f"meta_{Path(script).stem}_config.json"
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / "src" / script), "--config", str(tmp)],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0_eda(config: dict) -> None:
    print("\n" + "=" * 60)
    print("  PHASE 0 — Autonomous EDA")
    print("=" * 60)
    rc = _run_subprocess("eda_agent.py", {}, config)
    if rc != 0:
        print("  [Phase 0] EDA agent finished with errors (continuing).")
    else:
        insights_path = PROJECT_ROOT / "logs" / "eda" / "eda_insights.txt"
        if insights_path.exists():
            print(f"  [Phase 0] EDA insights saved → {insights_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — CNN
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_cnn(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 1] CNN skipped (cnn_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 1 — CNN  ({n_iterations} iterations)")
    print("=" * 60)

    # Build a search config that respects n_iterations as the total budget
    base_search = config.get("search", {})
    cnn_search_override = {
        **base_search,
        "linear_budget": 0,
        "random_budget": n_iterations,
        "cnn_exploration": {"enabled": True},
        "transfer_exploration": {
            "enabled": True,
            "max_iterations": max(1, n_iterations // 2),
            "interactive_pick_final": False,
        },
        "medium_stage": {"enabled": False},
        "reality_gate": {"enabled": False},
        "phase2": {
            **base_search.get("phase2", {}),
            "random_experiments":          n_iterations,
            "focused_experiments":         0,
            "tweak_experiments":           0,
            "augmentation_tweak_experiments": 0,
            "ai_free_experiments":         0,
            "final_tweak_experiments":     0,
        },
    }
    rc = _run_subprocess("cnn_agent.py", {"search": cnn_search_override}, config)
    if rc != 0:
        print("  [Phase 1] CNN agent finished with errors.")

    if not CNN_SUBMISSION.exists():
        print(f"  [Phase 1] No CNN model at {CNN_SUBMISSION} — skipping soundscape eval.")
        return None

    print("  [Phase 1] Soundscape eval (labeled train_soundscapes) …")
    score = suite.score_cnn(CNN_SUBMISSION)
    _print_soundscape_score("CNN", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BirdNET
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_birdnet(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 2] BirdNET skipped (birdnet_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 2 — BirdNET  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("birdnet_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 2] BirdNET agent finished with errors.")

    print("  [Phase 2] Soundscape eval on best BirdNET val predictions …")
    score = suite.score_birdnet_artifacts(BIRDNET_LOGS, BIRDNET_VAL_CACHE)
    _print_soundscape_score("BirdNET", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Perch
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_perch(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 3] Perch skipped (perch_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 3 — Perch  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("perch_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 3] Perch agent finished with errors.")

    head = PERCH_MEMORY / "best_head.keras"
    if not head.exists():
        head = PERCH_MEMORY / "final_head.keras"
    if not head.exists():
        print("  [Phase 3] No Perch head in logs/perch_memory — skipping soundscape eval.")
        return None

    print("  [Phase 3] Soundscape eval (ONNX Perch + best head) …")
    score = suite.score_perch(PERCH_MEMORY)
    _print_soundscape_score("Perch", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Build common val + ensemble search
# ─────────────────────────────────────────────────────────────────────────────

def _load_perch_onnx(config: dict):
    import onnxruntime as ort
    import kagglehub
    slug      = config.get("perch", {}).get("onnx_dataset", "rishikeshjani/perch-onnx-for-birdclef-2026")
    onnx_dir  = Path(kagglehub.dataset_download(slug))
    onnx_path = next(onnx_dir.rglob("*.onnx"))
    so        = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess      = ort.InferenceSession(str(onnx_path), sess_options=so,
                                     providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    dummy     = np.zeros((1, CLIP_SAMPLES), dtype=np.float32)
    outs      = sess.run(None, {inp_name: dummy})
    emb_idx   = next(i for i, o in enumerate(outs) if o.ndim == 2 and o.shape[-1] == PERCH_DIM)
    return sess, inp_name, emb_idx


def _build_perch_head(spec: dict, emb_dim: int, n_classes: int):
    import tensorflow as tf
    n_blocks   = int(spec.get("n_blocks",      2))
    hidden_dim = int(spec.get("hidden_dim",    512))
    proj_dim   = int(spec.get("proj_dim",      256))
    drop_block = float(spec.get("dropout_block", 0.3))
    drop_final = float(spec.get("dropout_final", 0.4))
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x   = tf.keras.layers.BatchNormalization()(inp)
    x   = tf.keras.layers.Dense(hidden_dim)(x)
    x   = tf.keras.layers.LayerNormalization()(x)
    for _ in range(n_blocks):
        h = tf.keras.layers.Dense(hidden_dim)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("gelu")(h)
        h = tf.keras.layers.Dropout(drop_block)(h)
        h = tf.keras.layers.Dense(hidden_dim)(h)
        x = tf.keras.layers.Add()([x, h])
        x = tf.keras.layers.LayerNormalization()(x)
    x   = tf.keras.layers.Dense(proj_dim, activation="gelu")(x)
    x   = tf.keras.layers.Dropout(drop_final)(x)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def build_common_val(config: dict) -> bool:
    """
    Embed the BirdNET soundscape val set with Perch ONNX as well,
    so both models can be evaluated on the exact same samples.
    Saves: META_VAL_CACHE with keys X_perch, X_birdnet, y_val.
    """
    if META_VAL_CACHE.exists():
        print(f"  [Common val] Already built: {META_VAL_CACHE}")
        return True

    if not BIRDNET_VAL_CACHE.exists():
        print("  [Common val] BirdNET val cache not found — skipping ensemble phase.")
        return False

    print("  [Common val] Embedding soundscapes with Perch ONNX...")
    import librosa

    bn        = np.load(str(BIRDNET_VAL_CACHE), allow_pickle=True)
    X_bn      = bn["X_val"].astype(np.float32)
    y_val     = bn["y_val"].astype(np.float32)
    row_ids   = bn["row_ids"].tolist()
    rid2idx   = {rid: i for i, rid in enumerate(row_ids)}

    sess, inp_name, emb_idx = _load_perch_onnx(config)

    def _hms(t: str) -> int:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    lab = pd.read_csv(SOUNDSCAPE_LABELS)
    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[
            {v.strip() for v in str(x).split(";") if x and str(x) != "nan"} for x in s
        ]))
        .reset_index()
    )

    X_perch_out, X_bn_out, y_out = [], [], []
    for row in grp.itertuples(index=False):
        stem    = Path(row.filename).stem
        end_sec = _hms(row.end)
        row_id  = f"{stem}_{end_sec}"
        if row_id not in rid2idx:
            continue
        fp = TRAIN_SOUNDSCAPES / row.filename
        if not fp.exists():
            continue
        start_sec = _hms(row.start)
        duration  = end_sec - start_sec
        try:
            wav, _ = librosa.load(str(fp), sr=SR, mono=True,
                                  offset=start_sec, duration=float(duration))
        except Exception:
            continue
        n   = int(duration * SR)
        wav = wav[:n] if len(wav) > n else np.pad(wav, (0, n - len(wav)))
        # Central 5-second clip
        if len(wav) >= CLIP_SAMPLES:
            s   = (len(wav) - CLIP_SAMPLES) // 2
            clip = wav[s: s + CLIP_SAMPLES]
        else:
            clip = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        clip = clip.astype(np.float32)

        outs = sess.run(None, {inp_name: clip[np.newaxis, :]})
        X_perch_out.append(outs[emb_idx][0])
        X_bn_out.append(X_bn[rid2idx[row_id]])
        y_out.append(y_val[rid2idx[row_id]])

    if not X_perch_out:
        print("  [Common val] No aligned samples found.")
        return False

    META_LOGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(META_VAL_CACHE),
                        X_perch   = np.stack(X_perch_out).astype(np.float32),
                        X_birdnet = np.stack(X_bn_out).astype(np.float32),
                        y_val     = np.stack(y_out).astype(np.float32))
    print(f"  [Common val] {len(X_perch_out)} aligned samples saved → {META_VAL_CACHE}")
    return True


def run_phase3_ensemble(
    config: dict,
    n_iterations: int,
    suite: SoundscapeEvalSuite,
    perch_score: SoundscapeScore | None,
    birdnet_score: SoundscapeScore | None,
) -> dict:
    if n_iterations <= 0:
        print("\n  [Phase 4] Ensemble skipped (ensemble_iterations=0)")
        return {}

    metric = suite.primary_metric
    print("\n" + "=" * 60)
    print(f"  PHASE 4 — Ensemble  ({n_iterations} blend weights)")
    p_val = perch_score.primary_value if perch_score else 0.0
    b_val = birdnet_score.primary_value if birdnet_score else 0.0
    print(f"  Perch {metric}={p_val:.5f}  BirdNET {metric}={b_val:.5f}")
    print("=" * 60)

    META_LOGS.mkdir(parents=True, exist_ok=True)

    print("  Building aligned soundscape predictions for blend search …")
    y_true = suite.y_true
    perch_preds = suite.aligned_perch_preds(PERCH_MEMORY)
    birdnet_preds = suite.aligned_birdnet_preds(BIRDNET_LOGS, BIRDNET_VAL_CACHE)
    if birdnet_preds is None:
        print("  [Phase 4] BirdNET val preds missing — skipping ensemble.")
        return {}

    perch_self = suite.score_arrays(perch_preds)
    bn_self = suite.score_arrays(birdnet_preds)
    print(f"  Perch alone:   {format_soundscape_score(perch_self)}")
    print(f"  BirdNET alone: {format_soundscape_score(bn_self)}")

    total = perch_self.primary_value + bn_self.primary_value
    w_start = (
        round(perch_self.primary_value / total, 2) if total > 0 else 0.5
    )

    n_blend = max(1, int(n_iterations))
    grid_n = max(n_blend, 9)
    candidates = sorted({round(float(w), 4) for w in np.linspace(0.01, 0.99, grid_n)})
    candidates = sorted(candidates, key=lambda w: abs(w - w_start))[:n_blend]

    results: list[dict] = []
    for i, w in enumerate(candidates, 1):
        blended = w * perch_preds + (1.0 - w) * birdnet_preds
        ap, n_scored = macro_average_precision(y_true, blended)
        auc, _ = competition_macro_auc(y_true, blended)
        med, _ = median_per_class_auc(y_true, blended)
        primary = primary_score(y_true, blended, metric)
        print(
            f"  [{i}/{len(candidates)}] perch_weight={w:.2f}  "
            f"{format_soundscape_metrics_line(macro_ap=ap, macro_auc=auc, median_auc=med, ranking_metric=metric)}"
        )
        results.append({
            "perch_weight": w,
            "primary_metric": metric,
            "primary_value": primary,
            "macro_average_precision": ap,
            "competition_macro_auc": auc,
            "median_per_class_auc": med,
            "n_scored": n_scored,
        })

    best = max(results, key=lambda r: r["primary_value"])
    best_weight = best["perch_weight"]
    best_primary = best["primary_value"]
    print(
        f"\n  [Phase 4] Best blend: perch_weight={best_weight:.2f}  "
        f"{format_soundscape_metrics_line(macro_ap=best['macro_average_precision'], macro_auc=best['competition_macro_auc'], median_auc=best['median_per_class_auc'], ranking_metric=metric)}"
    )

    ensemble_cfg = {
        "primary_metric": metric,
        "perch_weight": best_weight,
        "birdnet_weight": round(1.0 - best_weight, 4),
        "perch_soundscape": perch_self.to_dict(),
        "birdnet_soundscape": bn_self.to_dict(),
        "best_ensemble_primary": best_primary,
        "best_ensemble_macro_ap": best["macro_average_precision"],
        "best_ensemble_macro_auc": best["competition_macro_auc"],
        "best_ensemble_median_auc": best["median_per_class_auc"],
        "all_results": results,
    }
    out = META_LOGS / "ensemble_config.json"
    out.write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
    print(f"  Ensemble config saved → {out}")
    return ensemble_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Staged pipeline — Step 1a: architecture search × aug baselines
# ─────────────────────────────────────────────────────────────────────────────

def _run_cnn_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    max_samples: int,
    suite: SoundscapeEvalSuite,
) -> dict:
    logs_dir = META_LOGS / "cnn" / baseline
    logs_dir.mkdir(parents=True, exist_ok=True)
    cnn_aug = get_cnn_baseline_aug(baseline)
    base_search = config.get("search", {})
    override = {
        "meta_aug_preset": baseline,
        "arch_search_only": True,
        "lock_augmentation": True,
        "cnn_augmentation": cnn_aug,
        "cnn": {"logs_dir": str(logs_dir)},
        "search": {
            **base_search,
            "skip_final_training": True,
            "cheap": {
                **base_search.get("cheap", {}),
                "max_samples": max_samples,
            },
            "linear_budget": 0,
            "phase2": {
                **base_search.get("phase2", {}),
                "random_experiments": n_iters,
                "focused_experiments": 0,
                "tweak_experiments": 0,
                "augmentation_tweak_experiments": 0,
                "ai_free_experiments": 0,
                "final_tweak_experiments": 0,
            },
            "transfer_exploration": {"enabled": False},
            "medium_stage": {"enabled": False},
            "reality_gate": {"enabled": False},
        },
    }
    print(f"\n  [CNN / {baseline}] {describe_baseline(baseline)}")
    print(f"  logs → {logs_dir}  |  iters={n_iters}  |  max_samples={max_samples}")
    rc = _run_subprocess("cnn_agent.py", override, config)
    score = _score_cnn_arch_search(logs_dir, suite)
    _print_soundscape_score(f"CNN/{baseline}", score)
    return {
        "track": "cnn",
        "aug_baseline": baseline,
        "logs_dir": str(logs_dir),
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _run_birdnet_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    embed_frac: float,
    suite: SoundscapeEvalSuite,
) -> dict:
    logs_dir = META_LOGS / "birdnet" / baseline
    logs_dir.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"train_emb1024_{baseline}.npz"
    override = {
        "meta_aug_preset": baseline,
        "augmentation": get_audio_embedding_aug(baseline),
        "aug_preset_sweep": {"enabled": False},
        "train_sample_frac": embed_frac,
        "force_rebuild_cache": _meta_cfg(config).get("force_rebuild_embed_cache", False),
        "max_iterations": n_iters,
        "birdnet": {"logs_dir": str(logs_dir)},
        "train_cache_path": str(cache_path),
    }
    print(f"\n  [BirdNET / {baseline}] {describe_baseline(baseline)}")
    print(f"  logs → {logs_dir}  |  cache → {cache_path.name}  |  embed_frac={embed_frac}")
    if cache_path.exists() and not override.get("force_rebuild_cache"):
        print(f"  train embeddings on disk → {cache_path.name} (will skip rebuild)")
    rc = _run_subprocess("birdnet_agent.py", override, config)
    score = suite.score_birdnet_artifacts(logs_dir, BIRDNET_VAL_CACHE)
    _print_soundscape_score(f"BirdNET/{baseline}", score)
    return {
        "track": "birdnet",
        "aug_baseline": baseline,
        "logs_dir": str(logs_dir),
        "train_cache_path": str(cache_path),
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _run_perch_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    embed_frac: float,
    embed_max: int | None,
    suite: SoundscapeEvalSuite,
) -> dict:
    mem_dir = META_LOGS / "perch" / baseline
    cache_dir = META_LOGS / "perch_cache" / baseline
    code_dir = mem_dir / "codes"
    for d in (mem_dir, cache_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)
    perch_base = dict(config.get("perch", {}))
    train_cache = cache_dir / f"train_emb_{baseline}.npz"
    head_train_cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
    if head_train_cap is None:
        head_train_cap = perch_base.get("head_train_max_samples")
    override = {
        "meta_aug_preset": baseline,
        "augmentation": get_audio_embedding_aug(baseline),
        "train_sample_frac": embed_frac,
        "max_iterations": n_iters,
        "head_train_max_samples": head_train_cap,
        "perch": {
            **perch_base,
            "logs_dir": str(mem_dir.parent),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "max_train_samples": embed_max,
            "head_train_max_samples": head_train_cap,
            "force_rebuild_cache": _meta_cfg(config).get("force_rebuild_embed_cache", False),
            "skip_final_retrain": True,
        },
    }
    print(f"\n  [Perch / {baseline}] {describe_baseline(baseline)}")
    cap_msg = f"  |  head_train_cap={head_train_cap}" if head_train_cap else ""
    print(
        f"  memory → {mem_dir}  |  cache → {cache_dir}  |  "
        f"embed_frac={embed_frac} (stratified per species){cap_msg}"
    )
    if train_cache.exists() and not override["perch"]["force_rebuild_cache"]:
        print(f"  train embeddings on disk → {train_cache.name} (will skip rebuild)")
    rc = _run_subprocess("perch_agent.py", override, config)
    score = suite.score_perch(mem_dir)
    _print_soundscape_score(f"Perch/{baseline}", score)
    return {
        "track": "perch",
        "aug_baseline": baseline,
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _pick_track_winner(runs: list[dict], metric: str) -> dict | None:
    scored = [
        r for r in runs
        if r.get("score") and r["score"].get("primary_value") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r["score"]["primary_value"]))


def _ranking_value_from_run(entry: dict, metric: str) -> float:
    if metric == "macro_roc_auc":
        v = entry.get("macro_roc_auc")
    else:
        v = entry.get("macro_average_precision")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_average_precision")
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _load_perch_baseline_champion(aug_baseline: str, mem_dir: Path, metric: str) -> dict | None:
    """Best successful run from a stage-1a perch memory dir."""
    jsonl = mem_dir / "experiment_memory.jsonl"
    best_entry: dict | None = None
    best_val = -1.0
    best_typed_entry: dict | None = None
    best_typed_val = -1.0
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not entry.get("success"):
                    continue
                val = _ranking_value_from_run(entry, metric)
                spec = entry.get("spec") or {}
                if spec.get("arch_type"):
                    if val > best_typed_val:
                        best_typed_val = val
                        best_typed_entry = entry
                if val > best_val:
                    best_val = val
                    best_entry = entry
    # Prefer best run that has arch_type (required for stage-1b refine locking).
    if best_typed_entry is not None:
        best_entry = best_typed_entry
        best_val = best_typed_val
    elif best_entry is not None:
        spec0 = best_entry.get("spec") or {}
        if not spec0.get("arch_type"):
            best_entry = None
            best_val = -1.0

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec_info = info.get("spec") or {}
            if spec_info.get("arch_type"):
                val = float(info.get("ranking_value", info.get("macro_average_precision", -1)))
                if val > best_val:
                    best_val = val
                    best_entry = {
                        "success": True,
                        "spec": spec_info,
                        "macro_average_precision": info.get("macro_average_precision"),
                        "macro_roc_auc": info.get("macro_roc_auc"),
                        "metrics": info,
                    }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if best_entry is None:
        return None

    spec = dict(best_entry.get("spec") or {})
    if not spec.get("arch_type"):
        digest_path = mem_dir / "memory_digest.json"
        if digest_path.exists():
            try:
                digest = json.loads(digest_path.read_text(encoding="utf-8"))
                snap = digest.get("best_snapshot") or {}
                if snap.get("arch_type"):
                    spec["arch_type"] = snap["arch_type"]
                    spec.update(snap.get("spec_compact") or {})
            except (json.JSONDecodeError, OSError):
                pass
    spec.setdefault("arch_type", "residual_mlp")

    cache_dir = META_LOGS / "perch_cache" / aug_baseline
    return {
        "aug_baseline": aug_baseline,
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "arch_type": spec.get("arch_type"),
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": best_val,
        "macro_average_precision": best_entry.get("macro_average_precision"),
        "macro_roc_auc": best_entry.get("macro_roc_auc"),
        "median_per_class_auc": best_entry.get("median_per_class_auc"),
    }


def collect_perch_1a_top_candidates(config: dict, top_k: int = 2) -> list[dict]:
    """
    Pick top-K perch models from stage-1a (one champion per aug baseline, then global top-K).
    """
    metric = _meta_primary_metric(config)
    baselines = _baseline_names(config)

    if ARCH_SEARCH_1A_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            perch_runs = (summary.get("tracks") or {}).get("perch", {}).get("runs") or []
            if perch_runs:
                baselines = [r["aug_baseline"] for r in perch_runs if r.get("aug_baseline")]
        except (json.JSONDecodeError, OSError):
            pass

    candidates: list[dict] = []
    for baseline in baselines:
        mem_dir = META_LOGS / "perch" / baseline
        champ = _load_perch_baseline_champion(baseline, mem_dir, metric)
        if champ:
            candidates.append(champ)

    candidates.sort(key=lambda c: float(c["ranking_value"]), reverse=True)

    # Prefer distinct (aug_baseline, arch_type); then fill to top_k
    picked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        key = (c["aug_baseline"], str(c.get("arch_type", "?")))
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= top_k:
            break

    if len(picked) < top_k:
        for c in candidates:
            if c in picked:
                continue
            picked.append(c)
            if len(picked) >= top_k:
                break

    return picked[:top_k]


def _enrich_perch_winner_metrics(
    winner: dict,
    metric: str,
    suite: SoundscapeEvalSuite | None = None,
) -> dict:
    """Fill macro AP / AUC / median on winner dict from disk or one soundscape eval."""
    mem_dir = Path(winner["memory_dir"])
    ap = winner.get("macro_average_precision")
    auc = winner.get("macro_roc_auc")
    med = winner.get("median_per_class_auc")

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists() and (ap is None or auc is None):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            ap = ap if ap is not None else info.get("macro_average_precision")
            auc = auc if auc is not None else info.get("macro_roc_auc")
            med = med if med is not None else info.get("median_per_class_auc")
            if winner.get("ranking_value") is None and info.get("ranking_value") is not None:
                winner["ranking_value"] = float(info["ranking_value"])
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    score = winner.get("score") or {}
    if ap is None and score.get("macro_average_precision") is not None:
        ap = score["macro_average_precision"]
    if auc is None and score.get("macro_roc_auc") is not None:
        auc = score["macro_roc_auc"]
    if med is None and score.get("median_per_class_auc") is not None:
        med = score["median_per_class_auc"]

    if suite is not None and _perch_memory_has_head(mem_dir) and ap is None:
        try:
            sc = suite.score_perch(mem_dir)
            if sc is not None:
                ap = sc.macro_average_precision
                auc = sc.competition_macro_auc
                med = sc.median_per_class_auc
                winner["score"] = sc.to_dict()
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  [1c] Soundscape re-score skipped: {exc}")

    if ap is not None:
        winner["macro_average_precision"] = float(ap)
    if auc is not None:
        winner["macro_roc_auc"] = float(auc)
    if med is not None:
        winner["median_per_class_auc"] = float(med)
    if winner.get("ranking_value") is None and ap is not None:
        winner["ranking_value"] = float(ap) if metric != "macro_roc_auc" else float(auc or ap)
    winner.setdefault("ranking_metric", metric)
    return winner


def _print_stage_1c_locked_head(winner: dict, ranking_metric: str) -> None:
    """Show soundscape scores for the head locked before aug search."""
    mem_name = Path(winner.get("memory_dir", "?")).name
    arch = winner.get("arch_type", "?")
    aug = winner.get("aug_baseline", "?")
    ap = winner.get("macro_average_precision")
    auc = winner.get("macro_roc_auc")
    med = winner.get("median_per_class_auc")
    metrics_line = format_soundscape_metrics_line(
        macro_ap=float(ap) if ap is not None else None,
        macro_auc=float(auc) if auc is not None else None,
        median_auc=float(med) if med is not None else None,
        ranking_metric=ranking_metric,
        mark_ranking=True,
    )
    print("\n  STAGE 1C — LOCKED HEAD (current best before augmentation search)")
    print("  " + "─" * 72)
    print(f"  Memory dir     : {mem_name}")
    print(f"  arch_type      : {arch}")
    print(f"  aug baseline   : {aug}")
    print(f"  {metrics_line}")
    rv = winner.get("ranking_value")
    if rv is not None:
        print(
            f"  Seed {ranking_metric} for aug refine phase: {float(rv):.5f} "
            f"(must beat this to earn bonus refine tries)"
        )
    print("  " + "─" * 72)
    print("  Architecture is fixed in 1c — only augmentation configs change.\n")


def _print_stage_1b_champions(candidates: list[dict], ranking_metric: str) -> None:
    """Terminal summary of top-1a models selected for refine."""
    print(f"\n  STAGE 1B — TOP {len(candidates)} CHAMPIONS FROM STAGE 1A (soundscape val)")
    print("  " + "─" * 72)
    for i, c in enumerate(candidates, 1):
        arch = c.get("arch_type", "?")
        aug = c.get("aug_baseline", "?")
        ap = c.get("macro_average_precision")
        auc = c.get("macro_roc_auc")
        med = c.get("median_per_class_auc")
        metrics_line = format_soundscape_metrics_line(
            macro_ap=float(ap) if ap is not None else None,
            macro_auc=float(auc) if auc is not None else None,
            median_auc=float(med) if med is not None else None,
            ranking_metric=ranking_metric,
            mark_ranking=True,
        )
        print(f"  Winner #{i}: arch_type = {arch}")
        print(f"            aug baseline = {aug}")
        print(f"            {metrics_line}")
        if i < len(candidates):
            print("  " + "·" * 72)
    print("  " + "─" * 72)
    print(f"  Ranking metric for refine: {ranking_metric} (must beat seed macro_AP to earn bonus tries)\n")


def _run_perch_refine_1b(config: dict, candidate: dict, rank: int, refine_cfg: dict) -> dict:
    """Run one stage-1b refine campaign for a 1a champion."""
    aug = candidate["aug_baseline"]
    arch = str(candidate.get("arch_type", "residual_mlp"))
    slug = f"rank{rank}_{aug}_{arch}".replace("/", "_")[:80]
    mem_dir = META_LOGS / "perch" / "refine" / slug
    code_dir = mem_dir / "codes"
    cache_dir = Path(candidate["cache_dir"])

    for d in (mem_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)

    perch_base = dict(config.get("perch", {}))
    head_train_cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
    if head_train_cap is None:
        head_train_cap = perch_base.get("head_train_max_samples")

    override = {
        "meta_aug_preset": aug,
        "augmentation": get_audio_embedding_aug(aug),
        "perch_refine": {
            "enabled": True,
            "aug_baseline": aug,
            "locked_arch_type": arch,
            "seed_spec": candidate["spec"],
            "seed_score": float(candidate["ranking_value"]),
            "seed_macro_ap": candidate.get("macro_average_precision"),
            "seed_macro_auc": candidate.get("macro_roc_auc"),
            "seed_median_auc": candidate.get("median_per_class_auc"),
            "parent_memory_dir": candidate["memory_dir"],
            "initial_iterations": int(refine_cfg.get("initial_iterations", 5)),
            "bonus_iterations_on_improve": int(refine_cfg.get("bonus_iterations_on_improve", 5)),
            "max_iterations_per_model": int(refine_cfg.get("max_iterations_per_model", 25)),
        },
        "head_train_max_samples": head_train_cap,
        "perch": {
            **perch_base,
            "logs_dir": str(mem_dir.parent),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "force_rebuild_cache": False,
            "skip_final_retrain": True,
        },
    }

    print(f"\n  [Perch refine 1b / rank {rank}] aug={aug} arch={arch}")
    print(
        f"  seed {candidate['ranking_metric']}={float(candidate['ranking_value']):.5f}  "
        f"→ {mem_dir}"
    )
    rc = _run_subprocess("perch_agent.py", override, config)
    return {
        "rank": rank,
        "aug_baseline": aug,
        "arch_type": arch,
        "seed_ranking_value": float(candidate["ranking_value"]),
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "subprocess_rc": rc,
    }


def run_stage_1b_perch_refine(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """Refine top-K perch champions from stage 1a with adaptive iteration budgets."""
    refine_cfg = _stage_1b_cfg(config)
    if not refine_cfg.get("enabled", False):
        print("\n  [Stage 1b] Skipped (stage_1b.enabled=false)")
        return {}

    if _skip_if_completed(config, ARCH_SEARCH_1B_RESULTS, "Stage 1b"):
        try:
            return json.loads(ARCH_SEARCH_1B_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    top_k = max(1, int(refine_cfg.get("top_k_models", 2)))
    metric = _meta_primary_metric(config)

    print("\n" + "=" * 60)
    print(f"  STAGED PIPELINE — Step 1b: Refine top-{top_k} Perch models from 1a")
    print(
        f"  Budget per model: {refine_cfg.get('initial_iterations', 5)} initial, "
        f"+{refine_cfg.get('bonus_iterations_on_improve', 5)} on improve, "
        f"max {refine_cfg.get('max_iterations_per_model', 25)}"
    )
    print("=" * 60)

    candidates = collect_perch_1a_top_candidates(config, top_k=top_k)
    if not candidates:
        print("  [Stage 1b] No 1a candidates found — run stage 1a first.")
        return {"stage": "1b_perch_refine", "candidates": [], "refine_runs": []}

    _print_stage_1b_champions(candidates, metric)

    refine_runs = [
        _run_perch_refine_1b(config, cand, rank=i, refine_cfg=refine_cfg)
        for i, cand in enumerate(candidates, 1)
    ]

    scored: list[dict] = []
    for run in refine_runs:
        mem = Path(run["memory_dir"])
        sc = suite.score_perch(mem)
        _print_soundscape_score(f"Perch/refine/{run['arch_type']}", sc)
        entry = {**run, "score": sc.to_dict() if sc else None}
        if sc:
            entry["refined_ranking_value"] = sc.primary_value
        scored.append(entry)

    winner = _pick_track_winner(scored, metric)
    summary = {
        "stage": "1b_perch_refine",
        "primary_metric": metric,
        "top_k_models": top_k,
        "refine_config": {
            "initial_iterations": refine_cfg.get("initial_iterations", 5),
            "bonus_iterations_on_improve": refine_cfg.get("bonus_iterations_on_improve", 5),
            "max_iterations_per_model": refine_cfg.get("max_iterations_per_model", 25),
        },
        "candidates_from_1a": candidates,
        "refine_runs": scored,
        "winner": winner,
    }
    ARCH_SEARCH_1B_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Step 1b summary → {ARCH_SEARCH_1B_RESULTS}")
    if winner:
        print(
            f"\n  ★ Perch refine winner: aug={winner.get('aug_baseline')} "
            f"arch={winner.get('arch_type')}  "
            f"{metric}={float((winner.get('score') or {}).get('primary_value', 0)):.5f}"
        )
    return summary


def _perch_memory_has_head(mem_dir: Path) -> bool:
    """True if a refine / memory dir has artifacts needed for stage 1c."""
    if not (mem_dir / "best_head_code.py").exists():
        return False
    return (mem_dir / "best_head.keras").exists() or (
        mem_dir / "best_head.weights.h5"
    ).exists()


def _ranking_from_refine_memory(mem_dir: Path, metric: str) -> float:
    """Read ranking score from best_model_info or experiment_memory (no ONNX eval)."""
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if metric == "macro_roc_auc":
                v = info.get("macro_roc_auc")
            else:
                v = info.get("macro_average_precision") or info.get("ranking_value")
            if v is not None:
                return float(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    jsonl = mem_dir / "experiment_memory.jsonl"
    if jsonl.exists():
        best_val = -1.0
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not entry.get("success"):
                    continue
                best_val = max(best_val, _ranking_value_from_run(entry, metric))
        if best_val >= 0:
            return best_val
    return -1.0


def _refine_dir_to_candidate(mem_dir: Path, metric: str) -> dict | None:
    """Build a 1b-style winner dict from a refine memory directory."""
    if not _perch_memory_has_head(mem_dir):
        return None
    parts = mem_dir.name.split("_", 2)
    aug = parts[1] if len(parts) > 1 else "medium"
    arch = parts[2] if len(parts) > 2 else "unknown"
    val = _ranking_from_refine_memory(mem_dir, metric)
    spec: dict = {}
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec = dict(info.get("spec") or {})
        except (json.JSONDecodeError, OSError):
            pass
    spec.setdefault("arch_type", arch)
    macro_ap: float | None = None
    macro_auc: float | None = None
    median_auc: float | None = None
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            macro_ap = info.get("macro_average_precision")
            macro_auc = info.get("macro_roc_auc")
            median_auc = info.get("median_per_class_auc")
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "memory_dir": str(mem_dir),
        "cache_dir": str(META_LOGS / "perch_cache" / aug),
        "aug_baseline": aug,
        "arch_type": arch,
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": val,
        "macro_average_precision": macro_ap,
        "macro_roc_auc": macro_auc,
        "median_per_class_auc": median_auc,
    }


def _pick_perch_1b_winner(config: dict) -> dict | None:
    """Best refined perch model from stage 1b (by soundscape ranking metric)."""
    metric = _meta_primary_metric(config)

    if ARCH_SEARCH_1B_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1B_RESULTS.read_text(encoding="utf-8"))
            winner = summary.get("winner")
            if winner and winner.get("memory_dir"):
                mem = Path(winner["memory_dir"])
                if _perch_memory_has_head(mem):
                    return winner
                print(
                    f"  [1c] 1b results winner has no head ({mem.name}) — scanning refine dirs."
                )
        except (json.JSONDecodeError, OSError):
            pass

    refine_root = META_LOGS / "perch" / "refine"
    if not refine_root.exists():
        return _pick_perch_1a_fallback_champion(config, metric)

    best_run: dict | None = None
    best_val = -1.0
    for mem_dir in sorted(refine_root.iterdir()):
        if not mem_dir.is_dir():
            continue
        cand = _refine_dir_to_candidate(mem_dir, metric)
        if cand is None:
            if mem_dir.name.startswith("rank"):
                print(f"  [1c] Skipping incomplete refine run (no head): {mem_dir.name}")
            continue
        val = float(cand.get("ranking_value", -1.0))
        if val > best_val:
            best_val = val
            best_run = cand

    if best_run is not None:
        return best_run
    return _pick_perch_1a_fallback_champion(config, metric)


def _pick_perch_1a_fallback_champion(config: dict, metric: str) -> dict | None:
    """If 1b never produced a head, use best stage-1a perch baseline champion."""
    for baseline in _baseline_names(config):
        mem_dir = META_LOGS / "perch" / baseline
        champ = _load_perch_baseline_champion(baseline, mem_dir, metric)
        if champ and _perch_memory_has_head(mem_dir):
            champ["memory_dir"] = str(mem_dir)
            print(f"  [1c] Using stage-1a champion (no 1b head): {baseline} / {champ.get('arch_type')}")
            return champ
    return None


def _copy_perch_mapping_artifacts(src_mem: Path, dst_mem: Path) -> None:
    """Copy species / mapping files needed for soundscape scoring."""
    dst_mem.mkdir(parents=True, exist_ok=True)
    for name in (
        "bc_indices.npy",
        "proxy_map.json",
        "species_cols.json",
        "mapping_meta.json",
    ):
        src = src_mem / name
        if src.exists():
            shutil.copy2(src, dst_mem / name)


def _ensure_head_train_clips(
    config: dict,
    source_train_cache: Path,
    indices_path: Path,
    species_to_idx: dict,
) -> Path:
    """Clip list (label + filename) for the fixed head-train subset — fast 1c re-embed."""
    if HEAD_TRAIN_CLIPS.exists():
        n = sum(1 for _ in HEAD_TRAIN_CLIPS.open(encoding="utf-8") if _.strip())
        print(f"  [1c] Reusing head-train clip list ({n} clips) → {HEAD_TRAIN_CLIPS}")
        return HEAD_TRAIN_CLIPS

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data_io import load_core_tables, resolve_birdclef_paths
    from perch_agent import save_head_train_clip_subset

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    return save_head_train_clip_subset(
        source_train_cache,
        indices_path,
        HEAD_TRAIN_CLIPS,
        tables["train"],
        species_to_idx,
        paths.train_audio_dir,
    )


def _ensure_head_train_indices(
    config: dict,
    source_train_cache: Path,
    *,
    cap: int | None = None,
    seed: int = 42,
) -> Path:
    """Compute once and reuse the same 2000 head-train row indices across aug search."""
    if HEAD_TRAIN_INDICES.exists():
        idx = np.load(HEAD_TRAIN_INDICES)
        print(f"  [1c] Reusing head-train indices ({len(idx)} rows) → {HEAD_TRAIN_INDICES}")
        return HEAD_TRAIN_INDICES

    if cap is None:
        cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
        if cap is None:
            cap = config.get("perch", {}).get("head_train_max_samples", 2000)
    cap = int(cap)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from perch_agent import compute_and_save_head_train_indices

    if not source_train_cache.exists():
        raise FileNotFoundError(
            f"Cannot build head-train indices — cache missing: {source_train_cache}"
        )
    return compute_and_save_head_train_indices(
        source_train_cache, HEAD_TRAIN_INDICES, cap, random_state=seed
    )


def _perch_aug_search_cache_dir(preset: str, baselines: list[str]) -> Path:
    """Reuse stage-1a caches for light/medium/high; custom presets get aug_search/."""
    key = preset.strip().lower()
    if key in baselines:
        return META_LOGS / "perch_cache" / key
    return META_LOGS / "perch_cache" / "aug_search" / key


def _ensure_shared_1c_val_cache(config: dict) -> Path:
    """Build the shared soundscape val embedding cache once for all of stage 1c."""
    if PERCH_SHARED_VAL_CACHE.exists():
        import numpy as np

        d = np.load(str(PERCH_SHARED_VAL_CACHE))
        print(
            f"  [1c] Shared val cache ready: {PERCH_SHARED_VAL_CACHE.name} "
            f"(X={d['X'].shape})"
        )
        return PERCH_SHARED_VAL_CACHE

    print("\n  [1c] Building shared soundscape val cache (once for all aug trials)…")
    mem_dir = META_LOGS / "perch" / "aug_search" / "_val_build"
    code_dir = mem_dir / "codes"
    perch_base = dict(config.get("perch", {}))
    override = {
        "perch_build_val_only": True,
        "perch": {
            **perch_base,
            "logs_dir": str(META_LOGS / "perch"),
            "memory_dir": str(mem_dir),
            "cache_dir": str(META_LOGS / "perch_cache" / "aug_search"),
            "code_dir": str(code_dir),
            "val_cache_path": str(PERCH_SHARED_VAL_CACHE),
            "skip_final_retrain": True,
        },
    }
    _run_subprocess("perch_agent.py", override, config)
    return PERCH_SHARED_VAL_CACHE


def _run_perch_1c_trial(
    config: dict,
    *,
    preset: str,
    head_code_path: Path,
    train_cache: Path,
    val_cache: Path,
    indices_path: Path | None,
    mapping_src: Path,
    spec: dict,
    aug_dict: dict,
    force_rebuild_train: bool,
    clip_subset_path: Path | None,
) -> dict:
    """
    Single perch subprocess: embed train clips (if needed) + train head + eval on cached val.
    One ONNX load per iteration (not three).
    """
    mem_dir = META_LOGS / "perch" / "aug_search" / preset
    code_dir = mem_dir / "codes"
    cache_dir = train_cache.parent
    mem_dir.mkdir(parents=True, exist_ok=True)
    _copy_perch_mapping_artifacts(mapping_src, mem_dir)

    perch_base = dict(config.get("perch", {}))
    fixed_train: dict = {
        "enabled": True,
        "label": f"aug_{preset}",
        "head_code_path": str(head_code_path),
        "aug_preset": preset,
        "spec": spec,
    }
    if indices_path is not None:
        fixed_train["head_train_indices_path"] = str(indices_path)

    override: dict = {
        "meta_aug_preset": preset,
        "augmentation": aug_dict,
        "perch_fixed_train": fixed_train,
        "perch": {
            **perch_base,
            "logs_dir": str(META_LOGS / "perch"),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "val_cache_path": str(val_cache),
            "force_rebuild_cache": force_rebuild_train,
            "skip_final_retrain": True,
        },
    }
    if clip_subset_path is not None:
        override["perch_embed_clip_subset"] = str(clip_subset_path)

    print(
        f"\n  [1c trial] One pass: embed (if needed) + head train + cached-val metrics "
        f"→ {preset}"
    )
    rc = _run_subprocess("perch_agent.py", override, config)
    return {
        "aug_preset": preset,
        "memory_dir": str(mem_dir),
        "train_cache": str(train_cache),
        "subprocess_rc": rc,
    }


def _run_stage_1c_preset_grid(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    winner: dict,
    winner_mem: Path,
    head_code_path: Path,
    indices_path: Path | None,
    head_spec: dict,
    baselines: list[str],
    embed_frac: float,
    embed_max: int | None,
    presets: list[str],
    clip_subset_path: Path | None = None,
) -> list[dict]:
    """Fixed preset grid (legacy 1c mode)."""
    val_cache = PERCH_SHARED_VAL_CACHE
    runs: list[dict] = []
    for preset in presets:
        cache_dir = _perch_aug_search_cache_dir(preset, baselines)
        train_cache = cache_dir / f"train_emb_{preset}.npz"
        try:
            aug_dict = get_aug_search_preset(preset)
        except KeyError:
            aug_dict = get_audio_embedding_aug(preset)
        run = _run_perch_1c_trial(
            config,
            preset=preset,
            head_code_path=head_code_path,
            train_cache=train_cache,
            val_cache=val_cache,
            indices_path=indices_path if clip_subset_path is None else None,
            mapping_src=winner_mem,
            spec=head_spec,
            aug_dict=aug_dict,
            force_rebuild_train=not train_cache.exists(),
            clip_subset_path=clip_subset_path,
        )
        if not train_cache.exists():
            print(f"  [1c] Skip {preset} — train cache not built.")
            continue
        sc = suite.score_perch_mem_dir(Path(run["memory_dir"]))
        if sc is None:
            sc = suite.score_perch(Path(run["memory_dir"]))
        _print_soundscape_score(f"Perch/aug/{preset}", sc)
        run["score"] = sc.to_dict() if sc else None
        runs.append(run)
    return runs


def _run_one_llm_aug_trial(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    iteration: int,
    phase: str,
    researcher,
    memory,
    winner_mem: Path,
    head_code_path: Path,
    indices_path: Path | None,
    head_spec: dict,
    baselines: list[str],
    embed_frac: float,
    embed_max: int | None,
    metric: str,
    clip_subset_path: Path | None = None,
) -> dict | None:
    """One LLM aug proposal → cache build → fixed head train → soundscape score."""
    from aug_researcher import aug_dict_from_logged_spec, slug_from_spec
    from augmentation import validate_embedding_aug

    spec = researcher.next_experiment()
    spec["phase"] = phase
    spec["iteration"] = iteration
    try:
        aug_dict, _meta = validate_embedding_aug(spec)
        spec = {**_meta, **aug_dict}
    except ValueError as exc:
        print(f"  [1c LLM] Invalid aug spec: {exc}")
        memory.log(spec=spec, metrics={"status": "failed", "error": str(exc)}, code="")
        return None

    preset = slug_from_spec(spec, iteration, phase)
    val_cache = PERCH_SHARED_VAL_CACHE
    cache_dir = _perch_aug_search_cache_dir(preset, baselines)
    train_cache = cache_dir / f"train_emb_{preset}.npz"
    force_train = bool(_stage_1c_cfg(config).get("force_rebuild_embed_cache", False))

    print(f"\n  [1c LLM / {phase}] iter {iteration} → preset '{preset}'")

    aug_cfg_path = cache_dir / f"{preset}_aug_config.json"
    aug_cfg_path.write_text(json.dumps(aug_dict_from_logged_spec(spec), indent=2), encoding="utf-8")

    run = _run_perch_1c_trial(
        config,
        preset=preset,
        head_code_path=head_code_path,
        train_cache=train_cache,
        val_cache=val_cache,
        indices_path=indices_path if clip_subset_path is None else None,
        mapping_src=winner_mem,
        spec={**head_spec, "aug_preset": preset, "aug_phase": phase},
        aug_dict=aug_dict_from_logged_spec(spec),
        force_rebuild_train=force_train or not train_cache.exists(),
        clip_subset_path=clip_subset_path,
    )
    if not train_cache.exists():
        print(f"  [1c LLM] Train cache missing after trial: {train_cache}")
        memory.log(spec=spec, metrics={"status": "failed", "error": "cache_build"}, code="")
        return None

    mem_dir = Path(run["memory_dir"])
    metrics = None
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            metrics = {
                "status": "success",
                "macro_average_precision": info.get("macro_average_precision"),
                "macro_roc_auc": info.get("macro_roc_auc"),
                "median_per_class_auc": info.get("median_per_class_auc"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    sc = suite.score_perch_mem_dir(mem_dir)
    if sc is None:
        print("  [1c] No cached val preds — falling back to full soundscape ONNX eval")
        sc = suite.score_perch(mem_dir)
    _print_soundscape_score(f"Perch/aug/{preset}", sc)
    run["score"] = sc.to_dict() if sc else None
    run["aug_spec"] = spec
    run["aug_config_path"] = str(aug_cfg_path)

    memory.log(spec=spec, metrics=metrics, code=json.dumps(aug_dict_from_logged_spec(spec)))

    if sc:
        run["ranking_value"] = float(sc.primary_value)
    return run


def _run_stage_1c_llm_explore(
    config: dict,
    suite: SoundscapeEvalSuite,
    explore_cfg: dict,
    **common,
) -> tuple[list[dict], dict | None]:
    """Wide LLM exploration of augmentation configs."""
    from aug_researcher import AugResearcher
    from llm_client import LLMClient
    from memory import ExperimentMemory

    n_iters = int(explore_cfg.get("iterations", 10))
    mem_dir = META_LOGS / "perch" / "aug_search" / "memory_explore"
    mem_dir.mkdir(parents=True, exist_ok=True)

    researcher_cfg = config.get("researcher", {})
    llm_cfg = config.get("llm_researcher", {}) or {}
    provider = llm_cfg.get("provider") or config.get("llm", {}).get("provider", "ollama")
    model = researcher_cfg.get("model") or llm_cfg.get("model", "deepseek-r1:8b")
    temp = float(
        explore_cfg.get("temperature", researcher_cfg.get("temperature", 0.7))
    )

    llm_timeout = float(explore_cfg.get("researcher_timeout_seconds", 300))
    llm = LLMClient(provider=provider, model=model, timeout_seconds=llm_timeout)
    memory = ExperimentMemory(mem_dir, ranking_metric=common["metric"])
    researcher = AugResearcher(llm, memory, temperature=temp, refine_mode=False)
    print(f"  [1c explore] Researcher LLM timeout: {llm_timeout:.0f}s")

    print("\n" + "─" * 60)
    print(f"  STAGE 1c — EXPLORE: {n_iters} LLM augmentation experiments (wide search)")
    print("─" * 60)

    runs: list[dict] = []
    for it in range(1, n_iters + 1):
        run = _run_one_llm_aug_trial(
            config,
            suite,
            iteration=it,
            phase="explore",
            researcher=researcher,
            memory=memory,
            **common,
        )
        if run:
            runs.append(run)

    winner = _pick_track_winner(runs, common["metric"])
    return runs, winner


def _run_stage_1c_llm_refine(
    config: dict,
    suite: SoundscapeEvalSuite,
    refine_cfg: dict,
    explore_winner: dict,
    **common,
) -> tuple[list[dict], dict | None]:
    """Refine the explore-phase aug winner with adaptive iteration budget."""
    from aug_researcher import AugResearcher, aug_dict_from_logged_spec
    from llm_client import LLMClient
    from memory import ExperimentMemory

    seed_spec = explore_winner.get("aug_spec") or {}
    seed_score = float(
        (explore_winner.get("score") or {}).get("primary_value", -1.0)
    )
    seed_aug = aug_dict_from_logged_spec(seed_spec) if seed_spec else {}

    mem_dir = META_LOGS / "perch" / "aug_search" / "memory_refine"
    mem_dir.mkdir(parents=True, exist_ok=True)

    researcher_cfg = config.get("researcher", {})
    llm_cfg = config.get("llm_researcher", {}) or {}
    provider = llm_cfg.get("provider") or config.get("llm", {}).get("provider", "ollama")
    model = researcher_cfg.get("model") or llm_cfg.get("model", "deepseek-r1:8b")
    temp = float(refine_cfg.get("temperature", researcher_cfg.get("temperature", 0.5)))

    llm_timeout = float(refine_cfg.get("researcher_timeout_seconds", 300))
    llm = LLMClient(provider=provider, model=model, timeout_seconds=llm_timeout)
    memory = ExperimentMemory(mem_dir, ranking_metric=common["metric"])
    researcher = AugResearcher(
        llm,
        memory,
        temperature=temp,
        refine_mode=True,
        seed_aug=seed_aug,
        seed_score=seed_score,
    )
    print(f"  [1c refine] Researcher LLM timeout: {llm_timeout:.0f}s")

    initial = int(refine_cfg.get("initial_iterations", 5))
    bonus = int(refine_cfg.get("bonus_iterations_on_improve", 5))
    max_iters = int(refine_cfg.get("max_iterations", 15))

    print("\n" + "─" * 60)
    print(
        f"  STAGE 1c — REFINE: seed {common['metric']}={seed_score:.5f} "
        f"({explore_winner.get('aug_preset', '?')})"
    )
    print(
        f"  Budget: {initial} initial + {bonus} on improve, max {max_iters}"
    )
    print("─" * 60)

    runs: list[dict] = []
    best_val = seed_score
    it = 0
    remaining = initial

    while remaining > 0 and it < max_iters:
        it += 1
        remaining -= 1
        run = _run_one_llm_aug_trial(
            config,
            suite,
            iteration=it,
            phase="refine",
            researcher=researcher,
            memory=memory,
            **common,
        )
        if not run:
            continue
        runs.append(run)
        val = run.get("ranking_value")
        if val is not None and val > best_val:
            print(
                f"  [1c refine] Improved {common['metric']}: "
                f"{best_val:.5f} → {val:.5f} (+{bonus} bonus tries)"
            )
            best_val = val
            remaining += bonus

    winner = _pick_track_winner(runs, common["metric"])
    if winner is None:
        winner = explore_winner
    return runs, winner


def run_stage_1c_aug_search(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """
    Stage 1c: lock best 1b head; search augmentation (LLM explore + optional refine).
    """
    cfg = _stage_1c_cfg(config)
    if not cfg.get("enabled", False):
        print("\n  [Stage 1c] Skipped (stage_1c.enabled=false)")
        return {}

    if _skip_if_completed(config, ARCH_SEARCH_1C_RESULTS, "Stage 1c"):
        try:
            return json.loads(ARCH_SEARCH_1C_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    winner = _pick_perch_1b_winner(config)
    if not winner:
        print("\n  [Stage 1c] No 1b winner — run stage 1b first.")
        return {"stage": "1c_aug_search", "runs": [], "winner": None}

    winner_mem = Path(winner["memory_dir"])
    head_code_path = winner_mem / "best_head_code.py"
    if not head_code_path.exists():
        print(f"\n  [Stage 1c] Missing {head_code_path} — cannot lock head.")
        return {"stage": "1c_aug_search", "runs": [], "winner": None}

    metric = _meta_primary_metric(config)
    winner = _enrich_perch_winner_metrics(winner, metric, suite)
    _print_stage_1c_locked_head(winner, metric)

    baselines = _baseline_names(config)
    embed_frac = float(_meta_cfg(config).get("arch_search_embed_sample_frac", 0.5))
    embed_max = _meta_cfg(config).get("arch_search_embed_max_samples")
    if embed_max is not None:
        embed_max = int(embed_max)

    aug_preset = winner.get("aug_baseline", "medium")
    source_cache = Path(winner.get("cache_dir", META_LOGS / "perch_cache" / aug_preset))
    source_train = source_cache / f"train_emb_{aug_preset}.npz"
    if not source_train.exists():
        source_train = source_cache / "train_emb.npz"

    mode = str(cfg.get("mode", "llm")).lower()
    explore_cfg = dict(cfg.get("explore") or {})
    refine_cfg = dict(cfg.get("refine") or {})
    explore_cfg.setdefault("iterations", cfg.get("iterations", 10))

    _ensure_shared_1c_val_cache(config)

    print("=" * 60)
    print(f"  STAGED PIPELINE — Step 1c: Augmentation search (mode={mode})")
    print("=" * 60)

    indices_path = _ensure_head_train_indices(
        config, source_train, cap=cfg.get("head_train_samples")
    )

    species_cols = json.loads((winner_mem / "species_cols.json").read_text(encoding="utf-8"))
    species_to_idx = {s: i for i, s in enumerate(species_cols)}
    clip_subset_path: Path | None = None
    if cfg.get("embed_subset_only", True):
        clip_subset_path = _ensure_head_train_clips(
            config, source_train, indices_path, species_to_idx
        )
        print(
            f"  [1c] Fast mode: each aug trial embeds {cfg.get('head_train_samples', 2000)} "
            f"clips only (not the full {int(np.load(source_train)['X'].shape[0])} cache)"
        )

    head_spec: dict = {}
    info_path = winner_mem / "best_model_info.json"
    if info_path.exists():
        try:
            head_spec = json.loads(info_path.read_text(encoding="utf-8")).get("spec") or {}
        except (json.JSONDecodeError, OSError):
            pass
    head_spec.setdefault("arch_type", winner.get("arch_type"))

    common = {
        "winner_mem": winner_mem,
        "head_code_path": head_code_path,
        "indices_path": indices_path,
        "head_spec": head_spec,
        "baselines": baselines,
        "embed_frac": embed_frac,
        "embed_max": embed_max,
        "metric": metric,
        "clip_subset_path": clip_subset_path,
    }

    explore_runs: list[dict] = []
    refine_runs: list[dict] = []
    preset_runs: list[dict] = []
    explore_winner: dict | None = None
    refine_winner: dict | None = None
    final_winner: dict | None = None

    if mode in ("llm", "both"):
        explore_runs, explore_winner = _run_stage_1c_llm_explore(
            config, suite, explore_cfg, **common
        )
        if explore_winner and refine_cfg.get("enabled", True):
            refine_runs, refine_winner = _run_stage_1c_llm_refine(
                config, suite, refine_cfg, explore_winner, **common
            )
        final_winner = refine_winner or explore_winner

    if mode in ("presets", "both"):
        n_iters = int(explore_cfg.get("iterations", cfg.get("iterations", 10)))
        preset_names = cfg.get("aug_presets")
        if preset_names:
            presets = [str(p).lower() for p in preset_names][:n_iters]
        else:
            presets = list_aug_search_preset_names()[:n_iters]
        preset_runs = _run_stage_1c_preset_grid(
            config, suite, winner=winner, presets=presets, **common
        )
        preset_winner = _pick_track_winner(preset_runs, metric)
        if final_winner is None or (
            preset_winner
            and float((preset_winner.get("score") or {}).get("primary_value", -1))
            > float((final_winner.get("score") or {}).get("primary_value", -1))
        ):
            final_winner = preset_winner

    best_aug_config: dict | None = None
    if final_winner and final_winner.get("aug_spec"):
        from aug_researcher import aug_dict_from_logged_spec

        try:
            best_aug_config = aug_dict_from_logged_spec(final_winner["aug_spec"])
        except ValueError:
            best_aug_config = None
    if best_aug_config:
        best_aug_path = META_LOGS / "perch" / "aug_search" / "best_aug_config.json"
        best_aug_path.parent.mkdir(parents=True, exist_ok=True)
        best_aug_path.write_text(json.dumps(best_aug_config, indent=2), encoding="utf-8")

    summary = {
        "stage": "1c_aug_search",
        "mode": mode,
        "primary_metric": metric,
        "head_train_indices": str(indices_path),
        "locked_head_code": str(head_code_path),
        "refine_winner_1b": winner,
        "explore_runs": explore_runs,
        "explore_winner": explore_winner,
        "refine_runs": refine_runs,
        "refine_winner": refine_winner,
        "preset_runs": preset_runs,
        "winner": final_winner,
        "best_aug_config": str(META_LOGS / "perch" / "aug_search" / "best_aug_config.json")
        if best_aug_config
        else None,
    }
    ARCH_SEARCH_1C_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Step 1c summary → {ARCH_SEARCH_1C_RESULTS}")
    if final_winner:
        print(
            f"\n  ★ Aug-search winner: {final_winner.get('aug_preset')}  "
            f"{metric}={float((final_winner.get('score') or {}).get('primary_value', 0)):.5f}"
        )
    return summary


def run_stage_1d_final_train(config: dict) -> dict:
    """Stage 1d: full-data retrain with best 1b head + best 1c augmentation."""
    cfg = _stage_1d_cfg(config)
    if not cfg.get("enabled", False):
        print("\n  [Stage 1d] Skipped (stage_1d.enabled=false)")
        return {}

    if not ARCH_SEARCH_1C_RESULTS.exists():
        print("\n  [Stage 1d] No 1c results — run stage 1c first.")
        return {}

    try:
        s1c = json.loads(ARCH_SEARCH_1C_RESULTS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("\n  [Stage 1d] Could not read 1c results.")
        return {}

    aug_winner = s1c.get("winner")
    refine_winner = s1c.get("refine_winner_1b") or _pick_perch_1b_winner(config)
    if not aug_winner or not refine_winner:
        print("\n  [Stage 1d] Missing 1b/1c winners.")
        return {}

    preset = aug_winner.get("aug_preset", "medium")
    head_code_path = Path(s1c.get("locked_head_code", ""))
    if not head_code_path.exists():
        head_code_path = Path(refine_winner["memory_dir"]) / "best_head_code.py"
    if not head_code_path.exists():
        print(f"\n  [Stage 1d] Head code missing: {head_code_path}")
        return {}

    baselines = _baseline_names(config)
    full_frac = float(cfg.get("embed_sample_frac", 1.0))
    val_cache = PERCH_SHARED_VAL_CACHE if PERCH_SHARED_VAL_CACHE.exists() else (
        META_LOGS / "perch_cache" / "val_emb.npz"
    )

    best_aug_path = META_LOGS / "perch" / "aug_search" / "best_aug_config.json"
    aug_dict: dict | None = None
    if best_aug_path.exists():
        try:
            aug_dict = json.loads(best_aug_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if aug_dict is None:
        try:
            aug_dict = get_aug_search_preset(preset)
        except KeyError:
            aug_dict = get_audio_embedding_aug(
                preset if preset in baselines else "medium"
            )

    final_cache_dir = META_LOGS / "perch_cache" / "final"
    train_cache = final_cache_dir / f"train_emb_{preset}_full.npz"
    if cfg.get("rebuild_full_train_cache", True) or not train_cache.exists():
        print(
            f"\n  [Stage 1d] Building FULL train embeddings "
            f"(sample_frac={full_frac}, aug={preset}) — for Kaggle submission"
        )
        final_cache_dir.mkdir(parents=True, exist_ok=True)
        mem_dir = META_LOGS / "perch" / "final" / "_cache_build"
        code_dir = mem_dir / "codes"
        perch_base = dict(config.get("perch", {}))
        override = {
            "meta_aug_preset": f"{preset}_full",
            "augmentation": aug_dict,
            "train_sample_frac": full_frac,
            "perch_build_cache_only": True,
            "perch": {
                **perch_base,
                "logs_dir": str(META_LOGS / "perch"),
                "memory_dir": str(mem_dir),
                "cache_dir": str(final_cache_dir),
                "code_dir": str(code_dir),
                "val_cache_path": str(val_cache),
                "force_rebuild_cache": bool(cfg.get("rebuild_full_train_cache", True)),
                "skip_final_retrain": True,
            },
        }
        _run_subprocess("perch_agent.py", override, config)
        built = final_cache_dir / f"train_emb_{preset}_full.npz"
        if built.exists():
            train_cache = built
        elif not train_cache.exists():
            cache_dir = _perch_aug_search_cache_dir(preset, baselines)
            train_cache = cache_dir / f"train_emb_{preset}.npz"
            print(
                f"  [Stage 1d] Full cache build missing — falling back to search cache: "
                f"{train_cache.name}"
            )

    if not train_cache.exists():
        print(f"\n  [Stage 1d] Train cache missing: {train_cache}")
        return {}

    PERCH_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = PERCH_FINAL_DIR / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    _copy_perch_mapping_artifacts(Path(refine_winner["memory_dir"]), PERCH_FINAL_DIR)

    head_code = head_code_path.read_text(encoding="utf-8")
    print("\n" + "=" * 60)
    print("  STAGED PIPELINE — Step 1d: Final full-data retrain")
    print(f"  Head: {head_code_path.name}  |  Aug: {preset}")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from code_executor import CodeExecutor
    from perch_agent import _build_final_retrain_script

    py_exe = config.get("execution", {}).get("python_executable", "python3")
    timeout = config.get("execution", {}).get("timeout_seconds", 1800)
    if cfg.get("final_timeout_seconds"):
        timeout = int(cfg["final_timeout_seconds"])
    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)

    script = _build_final_retrain_script(
        head_code, PERCH_FINAL_DIR, train_cache, val_cache
    )
    script_path = code_dir / "final_retrain.py"
    script_path.write_text(script, encoding="utf-8")
    result = executor.run_file(script_path)

    ok = result.success and "FINAL_RETRAIN_DONE" in (result.stdout or "")
    if ok:
        weights = PERCH_FINAL_DIR / "final_head.weights.h5"
        keras_model = PERCH_FINAL_DIR / "final_head.keras"
        print(f"  Final model → {weights}")
        if cfg.get("copy_to_submission", True):
            sub = PROJECT_ROOT / "submission"
            sub.mkdir(parents=True, exist_ok=True)
            if weights.exists():
                shutil.copy2(weights, sub / "perch_final_head.weights.h5")
            if keras_model.exists():
                shutil.copy2(keras_model, sub / "perch_final_head.keras")
            shutil.copy2(head_code_path, sub / "perch_best_head_code.py")
            for name in (
                "bc_indices.npy",
                "proxy_map.json",
                "species_cols.json",
                "mapping_meta.json",
            ):
                src = PERCH_FINAL_DIR / name
                if src.exists():
                    shutil.copy2(src, sub / name)
            if best_aug_path.exists():
                shutil.copy2(best_aug_path, sub / "perch_best_aug_config.json")
            readme = sub / "PERCH_KAGGLE_README.txt"
            readme.write_text(
                "Perch overnight pipeline outputs (BirdCLEF 2026)\n"
                "============================================\n"
                f"Head code:     perch_best_head_code.py  (build_head + get_training_config)\n"
                f"Weights:       perch_final_head.weights.h5\n"
                f"Mapping:       bc_indices.npy, proxy_map.json, species_cols.json\n"
                f"Aug config:    perch_best_aug_config.json\n"
                f"Full train NPZ: ../logs/meta_agent/perch_cache/final/train_emb_{preset}_full.npz\n"
                f"Val NPZ:        ../logs/meta_agent/perch_cache/val_emb.npz\n"
                f"Run kaggle_inference.ipynb or your notebook with these paths.\n",
                encoding="utf-8",
            )
            print(f"  Kaggle-ready artifacts → {sub}")
    else:
        print(f"  [Stage 1d] Final retrain failed: {(result.stderr or '')[-500:]}")

    summary = {
        "stage": "1d_final_train",
        "aug_preset": preset,
        "train_cache": str(train_cache),
        "output_dir": str(PERCH_FINAL_DIR),
        "success": ok,
    }
    (META_LOGS / "arch_search_1d_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _skip_if_completed(config: dict, results_path: Path, stage_label: str) -> bool:
    if not _meta_cfg(config).get("skip_completed_stages", False):
        return False
    if results_path.exists():
        print(f"\n  [{stage_label}] Skipped — results already at {results_path.name}")
        return True
    return False


def run_stage_1a_arch_search(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """
    Step 1a: for each track, run architecture search on each aug baseline separately.
    Returns JSON-serialisable summary (also written to ARCH_SEARCH_1A_RESULTS).
    """
    if _skip_if_completed(config, ARCH_SEARCH_1A_RESULTS, "Stage 1a"):
        try:
            return json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    meta = _meta_cfg(config)
    baselines = _baseline_names(config)
    metric = _meta_primary_metric(config)

    cnn_iters = int(meta.get("cnn_iterations", 0))
    bird_iters = int(meta.get("birdnet_iterations", 0))
    perch_iters = int(meta.get("perch_iterations", 0))

    cnn_max = int(meta.get("arch_search_cnn_max_samples", 2000))
    embed_frac = float(meta.get("arch_search_embed_sample_frac", 0.5))
    embed_max = meta.get("arch_search_embed_max_samples")
    if embed_max is not None:
        embed_max = int(embed_max)

    print("\n" + "=" * 60)
    print("  STAGED PIPELINE — Step 1a: Architecture search × aug baselines")
    print(f"  Baselines: {', '.join(baselines)}")
    print(f"  Ranking metric: {metric}")
    print("=" * 60)

    summary: dict = {
        "stage": "1a_arch_search",
        "primary_metric": metric,
        "aug_baselines": baselines,
        "tracks": {},
    }

    if cnn_iters > 0:
        per = _arch_iters_per_baseline(config, cnn_iters)
        cnn_runs = [
            _run_cnn_baseline_1a(config, b, per, cnn_max, suite)
            for b in baselines
        ]
        winner = _pick_track_winner(cnn_runs, metric)
        summary["tracks"]["cnn"] = {"runs": cnn_runs, "winner": winner}
        if winner:
            sc = winner["score"]
            print(
                f"\n  ★ CNN track winner: aug={winner['aug_baseline']}  "
                f"{format_soundscape_metrics_line(macro_ap=sc.get('macro_average_precision'), macro_auc=sc.get('competition_macro_auc'), median_auc=sc.get('median_per_class_auc'), ranking_metric=metric)}"
            )

    if bird_iters > 0:
        per = _arch_iters_per_baseline(config, bird_iters)
        bn_runs = [
            _run_birdnet_baseline_1a(config, b, per, embed_frac, suite)
            for b in baselines
        ]
        winner = _pick_track_winner(bn_runs, metric)
        summary["tracks"]["birdnet"] = {"runs": bn_runs, "winner": winner}
        if winner:
            print(f"\n  ★ BirdNET track winner: aug={winner['aug_baseline']}")

    if perch_iters > 0:
        per = _arch_iters_per_baseline(config, perch_iters)
        p_runs = [
            _run_perch_baseline_1a(config, b, per, embed_frac, embed_max, suite)
            for b in baselines
        ]
        winner = _pick_track_winner(p_runs, metric)
        summary["tracks"]["perch"] = {"runs": p_runs, "winner": winner}
        if winner:
            print(f"\n  ★ Perch track winner: aug={winner['aug_baseline']}")

    META_LOGS.mkdir(parents=True, exist_ok=True)
    ARCH_SEARCH_1A_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Step 1a summary → {ARCH_SEARCH_1A_RESULTS}")
    return summary


def save_soundscape_leaderboard(
    scores: dict[str, SoundscapeScore | None],
    ensemble_cfg: dict,
    config: dict,
) -> None:
    """Persist ranked soundscape metrics for all tracks (v2 primary metric)."""
    META_LOGS.mkdir(parents=True, exist_ok=True)
    metric = _meta_primary_metric(config)
    rows = []
    for track, sc in scores.items():
        if sc is None:
            continue
        row = {"track": track, **sc.to_dict()}
        rows.append(row)
    if ensemble_cfg:
        rows.append({
            "track": "ensemble",
            "primary_metric": ensemble_cfg.get("primary_metric", metric),
            "primary_value": ensemble_cfg.get("best_ensemble_primary"),
            "macro_average_precision": ensemble_cfg.get("best_ensemble_macro_ap"),
            "competition_macro_auc": ensemble_cfg.get("best_ensemble_macro_auc"),
            "perch_weight": ensemble_cfg.get("perch_weight"),
            "birdnet_weight": ensemble_cfg.get("birdnet_weight"),
        })
    rows.sort(key=lambda r: float(r.get("primary_value") or -1.0), reverse=True)
    payload = {
        "primary_metric": metric,
        "evaluation": "labeled_train_soundscapes",
        "n_windows": scores.get("cnn") and scores["cnn"].n_windows,
        "ranking": rows,
    }
    SOUNDSCAPE_LEADERBOARD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Soundscape leaderboard ({metric}) → {SOUNDSCAPE_LEADERBOARD}")
    for i, row in enumerate(rows, 1):
        print(
            f"    #{i} {row['track']:10s}  "
            f"{format_soundscape_metrics_line(macro_ap=row.get('macro_average_precision'), macro_auc=row.get('competition_macro_auc'), median_auc=row.get('median_per_class_auc'), ranking_metric=metric)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "agent_config.json"))
    args   = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    meta_cfg        = _meta_cfg(config)
    pipeline        = str(meta_cfg.get("pipeline", "legacy")).lower()
    cnn_iters       = int(meta_cfg.get("cnn_iterations",        5))
    birdnet_iters   = int(meta_cfg.get("birdnet_iterations",   10))
    perch_iters     = int(meta_cfg.get("perch_iterations",     10))
    ensemble_iters  = int(meta_cfg.get("ensemble_iterations",   5))

    t0 = time.time()

    metric = _meta_primary_metric(config)
    suite = _soundscape_suite(config)
    print(f"\n  Meta-agent pipeline: {pipeline}")
    print(f"  Ranking metric: {metric} on labeled train_soundscapes")

    if pipeline == "staged_1c_only":
        if meta_cfg.get("run_eda", False):
            run_phase0_eda(config)
        run_stage_1c_aug_search(config, suite)
        if _stage_1d_cfg(config).get("enabled", False):
            run_stage_1d_final_train(config)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        return

    staged_pipelines = (
        "staged_1a",
        "staged_1a_1b",
        "staged_1a_1b_1c",
        "staged_full",
    )
    if pipeline in staged_pipelines:
        if meta_cfg.get("run_eda", False):
            run_phase0_eda(config)
        run_stage_1a_arch_search(config, suite)
        if pipeline in ("staged_1a_1b", "staged_1a_1b_1c", "staged_full") or _stage_1b_cfg(
            config
        ).get("enabled", False):
            run_stage_1b_perch_refine(config, suite)
        if pipeline in ("staged_1a_1b_1c", "staged_full") or _stage_1c_cfg(config).get(
            "enabled", False
        ):
            run_stage_1c_aug_search(config, suite)
        if pipeline == "staged_full" or _stage_1d_cfg(config).get("enabled", False):
            if _skip_if_completed(config, META_LOGS / "arch_search_1d_results.json", "Stage 1d"):
                pass
            else:
                run_stage_1d_final_train(config)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        print("\n  Overnight perch pipeline finished.")
        print(f"  Kaggle bundle (if 1d succeeded): {PROJECT_ROOT / 'submission'}")
        print(f"  Full model dir: {PERCH_FINAL_DIR}")
        return

    if meta_cfg.get("run_eda", True):
        run_phase0_eda(config)
    else:
        print("\n  [Phase 0] EDA skipped (meta_agent.run_eda=false)")

    cnn_score = run_phase1_cnn(config, cnn_iters, suite)
    birdnet_score = run_phase1_birdnet(config, birdnet_iters, suite)
    perch_score = run_phase2_perch(config, perch_iters, suite)
    ensemble = run_phase3_ensemble(
        config, ensemble_iters, suite, perch_score, birdnet_score
    )

    save_soundscape_leaderboard(
        {"cnn": cnn_score, "birdnet": birdnet_score, "perch": perch_score},
        ensemble,
        config,
    )

    def _line(sc: SoundscapeScore | None, name: str) -> None:
        if sc is None:
            print(f"  {name}:      (skipped)")
            return
        print(f"  {name}:      {format_soundscape_score(sc)}")

    print("\n" + "=" * 60)
    print("  META AGENT COMPLETE")
    print(
        f"  Model ranking uses macro_AP on labeled train_soundscapes; "
        f"also reports macro_AUC and median_AUC (v2 benchmark)"
    )
    _line(cnn_score, "CNN")
    _line(birdnet_score, "BirdNET")
    _line(perch_score, "Perch")
    if ensemble:
        print(
            f"  Ensemble: {format_soundscape_metrics_line(macro_ap=ensemble.get('best_ensemble_macro_ap'), macro_auc=ensemble.get('best_ensemble_macro_auc'), median_auc=ensemble.get('best_ensemble_median_auc'), ranking_metric=metric)}  "
            f"(perch={ensemble['perch_weight']:.2f} / birdnet={ensemble['birdnet_weight']:.2f})"
        )
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
