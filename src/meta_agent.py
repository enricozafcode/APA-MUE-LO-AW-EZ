"""
Meta Agent — BirdCLEF 2026
===========================
Five sequential phases:
  Phase 0: Autonomous EDA — explores the dataset, writes eda_insights.txt
  Phase 1: CNN agent explores scratch CNN architectures on raw audio spectrograms
  Phase 2: BirdNET agent explores MLP heads on BirdNET (1024-D) embeddings
  Phase 3: Perch agent explores MLP heads on Perch (1536-D) embeddings
  Phase 4: Ensemble — blend search on labeled train_soundscapes; ranking uses
           macro_average_precision (v2 benchmark metric, configurable)

Run:
    python src/meta_agent.py --config configs/agent_config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

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


def _run_subprocess(script: str, config_override: dict, base_config: dict) -> int:
    """Write a temp config with overrides and run a script as subprocess."""
    cfg = json.loads(json.dumps(base_config))
    cfg.update(config_override)
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

    meta_cfg        = config.get("meta_agent", {})
    cnn_iters       = int(meta_cfg.get("cnn_iterations",        5))
    birdnet_iters   = int(meta_cfg.get("birdnet_iterations",   10))
    perch_iters     = int(meta_cfg.get("perch_iterations",     10))
    ensemble_iters  = int(meta_cfg.get("ensemble_iterations",   5))

    t0 = time.time()

    metric = _meta_primary_metric(config)
    suite = _soundscape_suite(config)
    print(f"\n  Meta-agent ranking metric: {metric} on labeled train_soundscapes")

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
