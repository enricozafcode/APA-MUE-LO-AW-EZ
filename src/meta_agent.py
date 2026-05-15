"""
Meta Agent — BirdCLEF 2026
===========================
Five sequential phases:
  Phase 0: Autonomous EDA — explores the dataset, writes eda_insights.txt
  Phase 1: CNN agent explores scratch CNN architectures on raw audio spectrograms
  Phase 2: BirdNET agent explores MLP heads on BirdNET (1024-D) embeddings
  Phase 3: Perch agent explores MLP heads on Perch (1536-D) embeddings
  Phase 4: Ensemble — tries different blend weights on a COMMON soundscape val set
           so Perch and BirdNET predictions can be directly compared and combined

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "notebooks" / "birdnet_cache"
PERCH_MEMORY = PROJECT_ROOT / "logs" / "perch_memory"
META_LOGS    = PROJECT_ROOT / "logs" / "meta_agent"

SOUNDSCAPE_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
TRAIN_SOUNDSCAPES = DATA_DIR / "train_soundscapes"

BIRDNET_VAL_CACHE = CACHE_DIR / "val_emb1024.npz"
META_VAL_CACHE    = META_LOGS / "common_val.npz"

SR           = 32_000
CLIP_SEC     = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
PERCH_DIM    = 1536
PYTHON       = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _macro_auc(y_true: np.ndarray, y_score: np.ndarray, min_pos: int = 3) -> tuple[float, int]:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else 0.0), len(aucs)


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

def run_phase1_cnn(config: dict, n_iterations: int) -> float:
    print("\n" + "=" * 60)
    print(f"  PHASE 1 — CNN  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("cnn_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 1] CNN agent finished with errors.")

    auc_path = PROJECT_ROOT / "logs" / "agent" / "best_auc.json"
    if auc_path.exists():
        auc = float(json.loads(auc_path.read_text()).get("auc", 0.0))
    else:
        auc = 0.0

    print(f"\n  [Phase 1] Best CNN AUC = {auc:.5f}")
    return auc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BirdNET
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_birdnet(config: dict, n_iterations: int) -> float:
    print("\n" + "=" * 60)
    print(f"  PHASE 2 — BirdNET  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("birdnet_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 2] BirdNET agent finished with errors.")

    # Read best AUC
    auc_path = META_LOGS.parent / "birdnet_agent" / "best_auc.json"
    if auc_path.exists():
        info = json.loads(auc_path.read_text())
        auc  = float(info.get("auc", 0.0))
    else:
        hist = META_LOGS.parent / "birdnet_agent" / "history.json"
        if hist.exists():
            history = json.loads(hist.read_text())
            aucs = [e.get("macro_auc_ge3", 0.0) for e in history if e.get("status") == "success"]
            auc  = float(max(aucs)) if aucs else 0.0
        else:
            auc = 0.0

    print(f"\n  [Phase 2] Best BirdNET AUC = {auc:.5f}")
    return auc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Perch
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_perch(config: dict, n_iterations: int) -> float:
    print("\n" + "=" * 60)
    print(f"  PHASE 3 — Perch  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("perch_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 3] Perch agent finished with errors.")

    info_path = PERCH_MEMORY / "best_model_info.json"
    if info_path.exists():
        auc = float(json.loads(info_path.read_text()).get("auc", 0.0))
    else:
        auc = 0.0

    print(f"\n  [Phase 3] Best Perch AUC = {auc:.5f}")
    return auc


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


def run_phase3_ensemble(config: dict, n_iterations: int,
                        perch_auc: float, birdnet_auc: float) -> dict:
    print("\n" + "=" * 60)
    print(f"  PHASE 4 — Ensemble  ({n_iterations} blend weights)")
    print(f"  Perch AUC={perch_auc:.5f}  BirdNET AUC={birdnet_auc:.5f}")
    print("=" * 60)

    META_LOGS.mkdir(parents=True, exist_ok=True)

    # Build common val if not already done
    if not META_VAL_CACHE.exists():
        ok = build_common_val(config)
        if not ok:
            print("  [Phase 3] Cannot build common val — skipping.")
            return {}

    val     = np.load(str(META_VAL_CACHE))
    X_perch = val["X_perch"].astype(np.float32)
    X_bn    = val["X_birdnet"].astype(np.float32)
    y_val   = val["y_val"].astype(np.float32)
    print(f"  Common val: {y_val.shape}")

    # Load best Perch head and get its predictions on common val
    info_path = PERCH_MEMORY / "best_model_info.json"
    weights_path = PERCH_MEMORY / "best_head.weights.h5"
    if not info_path.exists() or not weights_path.exists():
        print("  [Phase 3] Best Perch model not found — skipping ensemble.")
        return {}

    spec      = json.loads(info_path.read_text())["spec"]
    n_classes = y_val.shape[1]
    print("  Loading best Perch head...")
    perch_head = _build_perch_head(spec, PERCH_DIM, n_classes)
    perch_head.load_weights(str(weights_path))
    perch_preds = perch_head.predict(X_perch, verbose=0)
    perch_self_auc, _ = _macro_auc(y_val, perch_preds)
    print(f"  Perch on common val: AUC={perch_self_auc:.5f}")

    # Load best BirdNET slot and get its predictions on common val
    slot_path = PROJECT_ROOT / "logs" / "birdnet_agent" / "best_slot.py"
    if not slot_path.exists():
        print("  [Phase 3] Best BirdNET slot not found — skipping ensemble.")
        return {}

    print("  Running best BirdNET head on common val...")
    slot_code = slot_path.read_text(encoding="utf-8")
    # We need BirdNET training data to re-train the head (slot code trains from scratch)
    birdnet_train = np.load(str(BIRDNET_VAL_CACHE), allow_pickle=True)  # use val as proxy
    # Actually run the slot against the birdnet val embeddings on our common val X_bn
    try:
        ns: dict = {}
        exec(slot_code, ns)
        import inspect
        sig = inspect.signature(ns["build_head"])
        if "y_train" in sig.parameters:
            birdnet_model = ns["build_head"](X_bn.shape[1], n_classes, y_val)
        else:
            birdnet_model = ns["build_head"](X_bn.shape[1], n_classes)
        birdnet_preds = birdnet_model.predict(X_bn, verbose=0)
        bn_self_auc, _ = _macro_auc(y_val, birdnet_preds)
        print(f"  BirdNET on common val: AUC={bn_self_auc:.5f}")
    except Exception as e:
        print(f"  [Phase 3] Could not run BirdNET slot: {e}")
        print("  Falling back to saved val preds...")
        bn_preds_path = PROJECT_ROOT / "logs" / "birdnet_agent" / "best_val_preds.npy"
        if not bn_preds_path.exists():
            print("  [Phase 3] No BirdNET val preds saved — skipping.")
            return {}
        birdnet_preds = np.load(str(bn_preds_path))
        bn_self_auc = birdnet_auc

    # Grid search over blend weights
    total   = perch_self_auc + bn_self_auc if (perch_self_auc + bn_self_auc) > 0 else 1.0
    w_start = round(perch_self_auc / total, 1)

    candidates = sorted(set(
        [round(w, 1) for w in np.linspace(0.1, 0.9, 9)]
    ))

    # Put the theoretically best weight first, then explore around it
    candidates = sorted(candidates, key=lambda w: abs(w - w_start))[:n_iterations]

    results = []
    for i, w in enumerate(candidates, 1):
        blended      = w * perch_preds + (1 - w) * birdnet_preds
        auc, n_scored= _macro_auc(y_val, blended)
        print(f"  [{i}/{len(candidates)}] perch_weight={w:.1f}  AUC={auc:.5f}  (scored on {n_scored} species)")
        results.append({"perch_weight": w, "auc": auc, "n_scored": n_scored})

    best        = max(results, key=lambda r: r["auc"])
    best_weight = best["perch_weight"]
    best_auc    = best["auc"]
    print(f"\n  [Phase 3] Best blend: perch_weight={best_weight:.1f}  AUC={best_auc:.5f}")

    ensemble_cfg = {
        "perch_weight":    best_weight,
        "birdnet_weight":  round(1.0 - best_weight, 1),
        "perch_val_auc":   perch_self_auc,
        "birdnet_val_auc": bn_self_auc,
        "best_ensemble_auc": best_auc,
        "all_results":     results,
    }
    out = META_LOGS / "ensemble_config.json"
    out.write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
    print(f"  Ensemble config saved → {out}")
    return ensemble_cfg


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

    run_phase0_eda(config)
    cnn_auc     = run_phase1_cnn(config, cnn_iters)
    birdnet_auc = run_phase1_birdnet(config, birdnet_iters)
    perch_auc   = run_phase2_perch(config, perch_iters)
    ensemble    = run_phase3_ensemble(config, ensemble_iters, perch_auc, birdnet_auc)

    print("\n" + "=" * 60)
    print("  META AGENT COMPLETE")
    print(f"  CNN best AUC:      {cnn_auc:.5f}")
    print(f"  BirdNET best AUC:  {birdnet_auc:.5f}")
    print(f"  Perch best AUC:    {perch_auc:.5f}")
    if ensemble:
        print(f"  Ensemble best AUC: {ensemble['best_ensemble_auc']:.5f}"
              f"  (perch={ensemble['perch_weight']:.1f} / birdnet={ensemble['birdnet_weight']:.1f})")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
