"""
Meta Agent — BirdCLEF 2026
===========================
Single autonomous loop exploring Perch, BirdNET, and ensemble approaches.
The LLM sees the full experiment history each iteration and decides:
  - "perch"    → train a fresh MLP on Perch (1536-D) embeddings
  - "birdnet"  → train a fresh MLP on BirdNET (1024-D) embeddings
  - "ensemble" → blend the current best Perch + best BirdNET predictions

All three approaches are evaluated on the SAME common validation set
(soundscapes from train_soundscapes_labels.csv, embedded with both models),
so results are directly comparable across approaches.

Setup (one-time, automatic):
  1. Perch agent builds Perch train embedding cache
  2. BirdNET agent builds BirdNET train embedding cache + soundscape val cache
  3. Meta agent re-embeds those same soundscapes with Perch ONNX → common val

Run:
    python src/meta_agent.py --config configs/agent_config.json
"""

from __future__ import annotations

import argparse
import json
import re
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

PERCH_TRAIN_CACHE   = PERCH_MEMORY / "train_emb.npz"
BIRDNET_TRAIN_CACHE = CACHE_DIR / "train_emb1024.npz"
BIRDNET_VAL_CACHE   = CACHE_DIR / "val_emb1024.npz"
META_VAL_CACHE      = META_LOGS / "common_val.npz"

SR           = 32_000
CLIP_SEC     = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
PERCH_DIM    = 1536
BIRDNET_DIM  = 1024

PYTHON = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Common validation set — embed soundscapes with BOTH models
# ─────────────────────────────────────────────────────────────────────────────

def _load_perch_onnx(config: dict):
    """Load Perch ONNX session; returns (session, inp_name, emb_idx)."""
    import onnxruntime as ort
    import kagglehub
    slug     = config.get("perch", {}).get("onnx_dataset", "rishikeshjani/perch-onnx-for-birdclef-2026")
    onnx_dir = Path(kagglehub.dataset_download(slug))
    onnx_path = next(onnx_dir.rglob("*.onnx"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, CLIP_SAMPLES), dtype=np.float32)
    outs  = sess.run(None, {inp_name: dummy})
    emb_idx = next(i for i, o in enumerate(outs) if o.ndim == 2 and o.shape[-1] == PERCH_DIM)
    return sess, inp_name, emb_idx


def build_common_val(config: dict) -> None:
    """
    Build a common soundscape val set embedded with BOTH Perch and BirdNET.
    BirdNET embeddings come from the birdnet_agent val cache.
    Perch embeddings are computed here from the same raw audio.
    """
    if META_VAL_CACHE.exists():
        print(f"  [Meta val] Cache exists: {META_VAL_CACHE}")
        return

    if not BIRDNET_VAL_CACHE.exists():
        print("  [Meta val] BirdNET val cache not found — run Phase 2 first.")
        return

    print("  [Meta val] Building common val cache (Perch + BirdNET on soundscapes)...")
    import librosa

    bn_data   = np.load(str(BIRDNET_VAL_CACHE), allow_pickle=True)
    X_bn      = bn_data["X_val"].astype(np.float32)
    y_val     = bn_data["y_val"].astype(np.float32)
    row_ids   = bn_data["row_ids"].tolist()

    print(f"    BirdNET val: {X_bn.shape} samples")

    sess, inp_name, emb_idx = _load_perch_onnx(config)

    def _hms(t: str) -> int:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    lab = pd.read_csv(SOUNDSCAPE_LABELS)
    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[{v.strip() for v in str(x).split(";") if x and str(x) != "nan"} for x in s]))
        .reset_index()
    )

    # Build row_id → audio index map from birdnet cache
    rid_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    X_perch_aligned = []
    bn_aligned      = []
    y_aligned       = []
    n_missing       = 0

    for row in grp.itertuples(index=False):
        fp       = TRAIN_SOUNDSCAPES / row.filename
        stem     = Path(row.filename).stem
        end_sec  = _hms(row.end)
        row_id   = f"{stem}_{end_sec}"

        if row_id not in rid_to_idx:
            continue
        if not fp.exists():
            n_missing += 1
            continue

        start_sec = _hms(row.start)
        duration  = end_sec - start_sec
        try:
            wav, _ = librosa.load(str(fp), sr=SR, mono=True,
                                  offset=start_sec, duration=float(duration))
        except Exception:
            continue

        n = int(duration * SR)
        wav = wav[:n] if len(wav) > n else np.pad(wav, (0, n - len(wav)))
        # Use central 5-second clip for Perch
        if len(wav) >= CLIP_SAMPLES:
            start = (len(wav) - CLIP_SAMPLES) // 2
            clip  = wav[start: start + CLIP_SAMPLES]
        else:
            clip = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        clip = clip.astype(np.float32)

        outs    = sess.run(None, {inp_name: clip[np.newaxis, :]})
        perch_e = outs[emb_idx][0]

        idx = rid_to_idx[row_id]
        X_perch_aligned.append(perch_e)
        bn_aligned.append(X_bn[idx])
        y_aligned.append(y_val[idx])

    if not X_perch_aligned:
        print("  [Meta val] No aligned samples found.")
        return

    META_LOGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(META_VAL_CACHE),
        X_perch  = np.stack(X_perch_aligned).astype(np.float32),
        X_birdnet= np.stack(bn_aligned).astype(np.float32),
        y_val    = np.stack(y_aligned).astype(np.float32),
    )
    print(f"  [Meta val] Saved {len(X_perch_aligned)} aligned samples → {META_VAL_CACHE}")
    if n_missing:
        print(f"  [Meta val] {n_missing} audio files not found (skipped)")


# ─────────────────────────────────────────────────────────────────────────────
# Training — fixed residual MLP, approach-agnostic
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(emb_dim: int, n_classes: int, spec: dict):
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


def train_head(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    spec: dict,
) -> tuple[float, object]:
    """Train a residual MLP head, return (val_auc, trained_model)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    emb_dim  = X_train.shape[1]
    n_classes= y_train.shape[1]
    lr       = float(spec.get("learning_rate", 1e-3))
    bs       = int(spec.get("batch_size",      256))
    epochs   = int(spec.get("epochs",          15))
    patience = int(spec.get("patience",        5))
    opt_name = str(spec.get("optimizer",       "adam"))
    val_split= float(spec.get("val_split",     0.1))

    # Train/val split on training data
    n_val   = max(1, int(len(X_train) * val_split))
    rng     = np.random.default_rng(42)
    perm    = rng.permutation(len(X_train))
    X_tr, y_tr = X_train[perm[n_val:]], y_train[perm[n_val:]]

    # Positive-class weighting for class imbalance
    pos = y_tr.sum(0).astype(np.float64)
    neg = len(y_tr) - pos
    pw  = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
    pw_t = tf.constant(pw)[tf.newaxis, :]

    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(
            pw_t * y_true * (-tf.math.log(y_pred))
            + (1 - y_true) * (-tf.math.log(1 - y_pred))
        )

    model = _build_mlp(emb_dim, n_classes, spec)
    opt   = (tf.keras.optimizers.SGD(lr, momentum=0.9)
             if opt_name == "sgd_momentum"
             else tf.keras.optimizers.Adam(lr))
    model.compile(optimizer=opt, loss=weighted_bce)

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                         monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(2, patience // 2),
                                              monitor="val_loss"),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=bs, callbacks=cbs, verbose=0)

    preds          = model.predict(X_val, verbose=0)
    auc, n_scored  = _macro_auc(y_val, preds)
    return auc, model


# ─────────────────────────────────────────────────────────────────────────────
# LLM Researcher
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous ML researcher for BirdCLEF 2026 bird sound classification.

You have access to three complementary approaches:
  - "perch":    Google Perch ONNX (1536-D embeddings) + MLP classification head
  - "birdnet":  BirdNET v2.4 (1024-D embeddings) + MLP classification head
  - "ensemble": blend best Perch predictions + best BirdNET predictions

All are evaluated on the SAME soundscape validation set (macro AUC, species with ≥3 positive samples).

Your job: each iteration, choose the best approach to try next and propose hyperparameters.
Reason about what has worked, what hasn't, and where the biggest gains are likely.

Hyperparameter search space:
  n_blocks:       [1, 2, 3]
  hidden_dim:     [256, 512, 1024]
  proj_dim:       [128, 256, 512]
  dropout_block:  [0.1, 0.2, 0.3, 0.4]
  dropout_final:  [0.2, 0.3, 0.4, 0.5, 0.6]
  learning_rate:  [0.01, 0.001, 0.0008, 0.0005, 0.0001]
  batch_size:     [128, 256, 512]
  optimizer:      ["adam", "sgd_momentum"]
  epochs:         [10, 15, 25]
  patience:       [3, 5, 7]
  val_split:      [0.1, 0.15, 0.2]
  perch_weight:   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  (ensemble only)

Respond with ONLY a single JSON object. Start with { and end with }.
Required keys for perch/birdnet: approach, n_blocks, hidden_dim, proj_dim, dropout_block,
  dropout_final, learning_rate, batch_size, optimizer, epochs, patience, val_split,
  reasoning, hypothesis.
Required keys for ensemble: approach, perch_weight, reasoning, hypothesis.

Example:
{"approach": "perch", "n_blocks": 2, "hidden_dim": 512, "proj_dim": 256, "dropout_block": 0.3, "dropout_final": 0.4, "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 15, "patience": 5, "val_split": 0.1, "reasoning": "Perch embeddings are high quality; start with a simple residual head.", "hypothesis": "Will establish a solid Perch baseline."}"""


def _extract_json(text: str) -> dict | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(cleaned[start:], start):
        if esc:   esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _fill_defaults(spec: dict) -> dict:
    defaults = dict(approach="perch", n_blocks=2, hidden_dim=512, proj_dim=256,
                    dropout_block=0.3, dropout_final=0.4, learning_rate=1e-3,
                    batch_size=256, optimizer="adam", epochs=15, patience=5,
                    val_split=0.1, perch_weight=0.5,
                    reasoning="defaults", hypothesis="baseline")
    return {**defaults, **spec}


def llm_propose(messages: list[dict], config: dict) -> dict:
    from llm_client import LLMClient
    llm_cfg = config.get("llm", {})
    llm = LLMClient(provider=llm_cfg.get("provider", "ollama"),
                    model=llm_cfg.get("model", "llama3.2:3b"))
    resp = llm.generate_from_messages(messages, temperature=llm_cfg.get("temperature", 0.5))
    spec = _extract_json(resp)
    if spec is None:
        print(f"  [LLM] Could not parse JSON, using defaults. Raw: {repr(resp[:200])}")
        spec = {}
    return _fill_defaults(spec), resp


def _build_history_prompt(history: list[dict], best_perch: float, best_birdnet: float) -> str:
    if not history:
        return ("No experiments yet. Start by trying one Perch run and one BirdNET run "
                "to establish baselines for both approaches.")
    lines = []
    for e in history:
        if e["approach"] == "ensemble":
            lines.append(f"  ensemble  perch_weight={e['spec'].get('perch_weight', '?')}  "
                         f"AUC={e['auc']:.5f}  {'✓' if e.get('is_best') else ''}")
        else:
            lines.append(f"  {e['approach']:8s}  AUC={e['auc']:.5f}  "
                         f"n_blocks={e['spec'].get('n_blocks','?')}  "
                         f"hidden={e['spec'].get('hidden_dim','?')}  "
                         f"lr={e['spec'].get('learning_rate','?')}  "
                         f"{'✓' if e.get('is_best') else ''}")
    return (f"Experiment history ({len(history)} runs):\n"
            + "\n".join(lines)
            + f"\n\nBest Perch AUC so far:   {best_perch:.5f}"
            + f"\nBest BirdNET AUC so far: {best_birdnet:.5f}"
            + "\n\nChoose the next experiment. Think about which approach or configuration "
              "is most likely to improve results.")


# ─────────────────────────────────────────────────────────────────────────────
# Cache bootstrap — run sub-agents if caches are missing
# ─────────────────────────────────────────────────────────────────────────────

import subprocess

def _bootstrap_caches(config: dict) -> None:
    meta_cfg = config.get("meta_agent", {})

    if not PERCH_TRAIN_CACHE.exists():
        print("\n  [Bootstrap] Perch train cache missing — running perch_agent...")
        tmp = Path(tempfile.gettempdir()) / "meta_perch_bootstrap.json"
        cfg = json.loads(json.dumps(config))
        cfg["max_iterations"] = 0
        cfg["perch"]["force_rebuild_cache"] = True
        tmp.write_text(json.dumps(cfg), encoding="utf-8")
        subprocess.run([PYTHON, str(PROJECT_ROOT / "src" / "perch_agent.py"),
                        "--config", str(tmp)], cwd=str(PROJECT_ROOT))

    if not BIRDNET_VAL_CACHE.exists():
        print("\n  [Bootstrap] BirdNET val cache missing — running birdnet_agent to build caches...")
        tmp = Path(tempfile.gettempdir()) / "meta_birdnet_bootstrap.json"
        cfg = json.loads(json.dumps(config))
        cfg["max_iterations"] = 0
        tmp.write_text(json.dumps(cfg), encoding="utf-8")
        subprocess.run([PYTHON, str(PROJECT_ROOT / "src" / "birdnet_agent.py"),
                        "--config", str(tmp)], cwd=str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def agent_loop(config: dict) -> None:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    meta_cfg    = config.get("meta_agent", {})
    n_iter      = int(meta_cfg.get("total_iterations", config.get("max_iterations", 25)))
    max_fail    = int(meta_cfg.get("max_failures", 3))

    META_LOGS.mkdir(parents=True, exist_ok=True)

    # ── Step 1: ensure caches exist ──
    _bootstrap_caches(config)
    build_common_val(config)

    # ── Step 2: load training data ──
    print("\n  Loading training caches...")
    if not PERCH_TRAIN_CACHE.exists():
        raise FileNotFoundError(f"Perch train cache not found: {PERCH_TRAIN_CACHE}")
    if not BIRDNET_TRAIN_CACHE.exists():
        raise FileNotFoundError(f"BirdNET train cache not found: {BIRDNET_TRAIN_CACHE}")
    if not META_VAL_CACHE.exists():
        raise FileNotFoundError(f"Common val cache not found: {META_VAL_CACHE}")

    p_data = np.load(str(PERCH_TRAIN_CACHE))
    X_perch_train = p_data["X"].astype(np.float32)
    y_perch_train = p_data["y"].astype(np.float32)

    b_data = np.load(str(BIRDNET_TRAIN_CACHE))
    X_bn_train = b_data["X"].astype(np.float32)
    y_bn_train = b_data["y"].astype(np.float32)

    val_data    = np.load(str(META_VAL_CACHE))
    X_perch_val = val_data["X_perch"].astype(np.float32)
    X_bn_val    = val_data["X_birdnet"].astype(np.float32)
    y_val       = val_data["y_val"].astype(np.float32)
    n_classes   = y_val.shape[1]

    print(f"  Perch train:   {X_perch_train.shape}  BirdNET train: {X_bn_train.shape}")
    print(f"  Common val:    {y_val.shape}  ({int(y_val.sum(0).clip(0,1).sum())} species with ≥1 pos)")

    # ── Step 3: agent loop ──
    history:      list[dict] = []
    messages:     list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    best_perch_auc  = 0.0
    best_birdnet_auc= 0.0
    best_overall_auc= 0.0
    best_perch_preds: np.ndarray | None = None
    best_birdnet_preds: np.ndarray | None = None
    consec_fail     = 0

    print("\n" + "=" * 60)
    print(f"  META AGENT — {n_iter} iterations exploring Perch, BirdNET, ensemble")
    print("=" * 60)

    for it in range(1, n_iter + 1):
        print(f"\n{'─'*55}")
        print(f"  ITERATION {it}/{n_iter}")
        print(f"{'─'*55}")

        # LLM proposes next experiment
        user_msg = _build_history_prompt(history, best_perch_auc, best_birdnet_auc)
        messages.append({"role": "user", "content": user_msg})

        print("  Querying LLM...")
        t0 = time.time()
        try:
            spec, raw_resp = llm_propose(messages, config)
        except Exception as e:
            print(f"  LLM failed: {e}")
            consec_fail += 1
            if consec_fail >= max_fail:
                break
            continue
        print(f"  LLM responded in {time.time()-t0:.1f}s")

        approach = spec.get("approach", "perch")
        print(f"  Approach: {approach}  | {spec.get('reasoning', '')[:80]}")

        # ── Execute experiment ──
        auc   = None
        preds = None
        t0    = time.time()
        try:
            if approach == "perch":
                auc, model = train_head(X_perch_train, y_perch_train,
                                        X_perch_val,   y_val, spec)
                preds = model.predict(X_perch_val, verbose=0)
                print(f"  Perch AUC = {auc:.5f}  ({time.time()-t0:.1f}s)")

            elif approach == "birdnet":
                auc, model = train_head(X_bn_train, y_bn_train,
                                        X_bn_val,   y_val, spec)
                preds = model.predict(X_bn_val, verbose=0)
                print(f"  BirdNET AUC = {auc:.5f}  ({time.time()-t0:.1f}s)")

            elif approach == "ensemble":
                if best_perch_preds is None or best_birdnet_preds is None:
                    print("  Ensemble skipped — need at least one Perch AND one BirdNET run first.")
                    messages.append({"role": "assistant", "content": raw_resp})
                    history.append({"approach": "ensemble", "auc": 0.0, "spec": spec,
                                    "skipped": True, "is_best": False})
                    continue
                w = float(spec.get("perch_weight", 0.5))
                blended = w * best_perch_preds + (1 - w) * best_birdnet_preds
                auc, n_scored = _macro_auc(y_val, blended)
                preds = blended
                print(f"  Ensemble AUC = {auc:.5f}  (perch_weight={w:.2f})  ({time.time()-t0:.1f}s)")

            else:
                print(f"  Unknown approach '{approach}' — skipping.")
                consec_fail += 1
                messages.append({"role": "assistant", "content": raw_resp})
                continue

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}\n  {traceback.format_exc()[-400:]}")
            consec_fail += 1
            messages.append({"role": "assistant", "content": raw_resp})
            history.append({"approach": approach, "auc": 0.0, "spec": spec,
                             "failed": True, "is_best": False})
            if consec_fail >= max_fail:
                print(f"\n  {max_fail} consecutive failures — stopping.")
                break
            continue

        consec_fail = 0
        is_best = auc > best_overall_auc

        # Track per-approach bests
        if approach == "perch" and auc > best_perch_auc:
            best_perch_auc   = auc
            best_perch_preds = preds
            # Save Perch head weights for later ensemble / Kaggle notebook
            weights_path = META_LOGS / "best_perch_head.weights.h5"
            model.save_weights(str(weights_path))
            with open(META_LOGS / "best_perch_spec.json", "w") as f:
                json.dump(spec, f, indent=2)

        elif approach == "birdnet" and auc > best_birdnet_auc:
            best_birdnet_auc   = auc
            best_birdnet_preds = preds
            weights_path = META_LOGS / "best_birdnet_head.weights.h5"
            model.save_weights(str(weights_path))
            with open(META_LOGS / "best_birdnet_spec.json", "w") as f:
                json.dump(spec, f, indent=2)

        if is_best:
            best_overall_auc = auc
            with open(META_LOGS / "best_overall.json", "w") as f:
                json.dump({"auc": auc, "approach": approach, "iteration": it, "spec": spec}, f, indent=2)
            print(f"  ★ NEW OVERALL BEST  AUC={auc:.5f}  approach={approach}")

        history.append({"approach": approach, "auc": auc, "spec": spec,
                        "failed": False, "is_best": is_best})
        messages.append({"role": "assistant", "content": raw_resp})

        # Save history after each iteration
        with open(META_LOGS / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"  [Best so far]  overall={best_overall_auc:.5f}  "
              f"perch={best_perch_auc:.5f}  birdnet={best_birdnet_auc:.5f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  META AGENT COMPLETE")
    ok = sum(1 for e in history if not e.get("failed") and not e.get("skipped"))
    print(f"  {ok}/{len(history)} successful experiments")
    print(f"  Best Perch AUC:   {best_perch_auc:.5f}")
    print(f"  Best BirdNET AUC: {best_birdnet_auc:.5f}")
    print(f"  Best overall AUC: {best_overall_auc:.5f}")
    print(f"  Results saved to: {META_LOGS}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "agent_config.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent_loop(config)


if __name__ == "__main__":
    main()
