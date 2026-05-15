"""
BirdCLEF 2026 — BirdNET Embedding Agent
========================================

Architecture:
  - Phase 1:  BirdNET acoustic encoder (v2.4) embeds all audio → 1024-D vectors, cached to .npz
  - Phase 2:  Agent loop — LLM writes only the classification head (build_head / get_head_config)
              Fixed harness handles embedding loading, training loop, macro_auc_ge3 evaluation
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from src.augmentation import AudioAugmenter
except ImportError:
    from augmentation import AudioAugmenter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "notebooks" / "birdnet_cache"
LOGS_DIR     = PROJECT_ROOT / "logs" / "birdnet_agent"

TRAIN_CSV           = DATA_DIR / "train.csv"
SAMPLE_SUB_CSV      = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR     = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES   = DATA_DIR / "train_soundscapes"
SOUNDSCAPE_LABELS   = DATA_DIR / "train_soundscapes_labels.csv"

NPZ_TRAIN         = CACHE_DIR / "train_emb1024.npz"
NPZ_TRAIN_AUG     = CACHE_DIR / "train_emb1024_aug.npz"
NPZ_TRAIN_FULL    = CACHE_DIR / "train_emb1024_full.npz"
NPZ_TRAIN_AUG_FULL= CACHE_DIR / "train_emb1024_aug_full.npz"
NPZ_VAL           = CACHE_DIR / "val_emb1024.npz"
NPZ_PSEUDO        = CACHE_DIR / "pseudo_emb1024.npz"

# ---------------------------------------------------------------------------
# BirdNET constants
# ---------------------------------------------------------------------------

SR_LOAD          = 32000
BIRDNET_SR       = 48000
CLIP_LOAD_SEC    = 5.0
BIRDNET_CHUNK_SEC = 3.0
EMB_DIM          = 1024
EMB_BATCH_SIZE   = 32

# ---------------------------------------------------------------------------
# BirdNET globals (lazy-initialised)
# ---------------------------------------------------------------------------

_bird_model   = None
_bird_session = None


def init_birdnet():
    global _bird_model
    if _bird_model is not None:
        return
    import birdnet
    print("Loading BirdNET acoustic encoder v2.4 ...")
    _bird_model = birdnet.load("acoustic", "2.4", "tf")
    print("BirdNET model loaded.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _prepare_chunk(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != BIRDNET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=BIRDNET_SR)
    max_keep = int(CLIP_LOAD_SEC * BIRDNET_SR)
    if len(y) > max_keep:
        y = y[:max_keep]
    need = int(BIRDNET_CHUNK_SEC * BIRDNET_SR)
    s0 = max(0, (len(y) - need) // 2)
    chunk = y[s0: s0 + need]
    if len(chunk) < need:
        chunk = np.pad(chunk, (0, need - len(chunk)))
    return chunk.astype(np.float32)


def _load_focal_safe(ap_str: str) -> np.ndarray | None:
    """Thread-safe audio loader; returns None on any error."""
    try:
        y, _ = librosa.load(ap_str, sr=SR_LOAD, mono=True)
        n = int(CLIP_LOAD_SEC * SR_LOAD)
        y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
        return y.astype(np.float32)
    except Exception:
        return None


def _load_soundscape_window(row_id: str) -> tuple[np.ndarray, int]:
    stem, end_s = row_id.rsplit("_", 1)
    end_sec = int(end_s)
    start_sec = max(0, end_sec - int(CLIP_LOAD_SEC))
    fp = TRAIN_SOUNDSCAPES / f"{stem}.ogg"
    y, _ = librosa.load(str(fp), sr=SR_LOAD, mono=True,
                        offset=start_sec, duration=CLIP_LOAD_SEC)
    n = int(CLIP_LOAD_SEC * SR_LOAD)
    y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32), SR_LOAD


def _load_random_noise(rng: np.random.Generator, noise_pool: list) -> np.ndarray | None:
    """Load a random 5-second segment from a soundscape file as background noise."""
    fp = noise_pool[int(rng.integers(0, len(noise_pool)))]
    try:
        dur = librosa.get_duration(path=str(fp))
        start_max = max(0.0, dur - CLIP_LOAD_SEC)
        offset = float(rng.uniform(0, start_max)) if start_max > 0 else 0.0
        n, _ = librosa.load(str(fp), sr=SR_LOAD, mono=True, offset=offset, duration=CLIP_LOAD_SEC)
        nt = int(CLIP_LOAD_SEC * SR_LOAD)
        n = n[:nt] if len(n) > nt else np.pad(n, (0, nt - len(n)))
        return n.astype(np.float32)
    except Exception:
        return None


def _mix_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal with noise at a given SNR (dB) using power-based scaling."""
    ps = np.mean(signal ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return np.clip(signal + scale * noise, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

def _embed_batch(audio_list: list[np.ndarray], sr: int) -> np.ndarray:
    if not audio_list:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    inputs = [
        (np.ascontiguousarray(_prepare_chunk(y, sr), dtype=np.float32), BIRDNET_SR)
        for y in audio_list
    ]
    res = _bird_session.run_arrays(inputs)
    emb = np.asarray(res.embeddings[:, 0, :], dtype=np.float32)
    return emb


# ---------------------------------------------------------------------------
# Phase 1: build / load embedding caches
# ---------------------------------------------------------------------------

def _build_species_map() -> tuple[list[str], dict[str, int]]:
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
    cols = [c for c in sample_sub.columns if c != "row_id"]
    return cols, {s: i for i, s in enumerate(cols)}


def build_train_cache(species_to_idx: dict[str, int],
                      max_samples: int | None = None,
                      sample_frac: float = 0.30,
                      aug_config: dict | None = None,
                      out_path: Path | None = None) -> None:
    aug_config  = aug_config or {}
    audio_aug   = AudioAugmenter(aug_config.get("audio", {}))
    use_snr     = bool(aug_config.get("use_snr_mixing", False))
    has_audio   = bool(audio_aug.active_strategies())
    use_any_aug = use_snr or has_audio
    n_views     = aug_config.get("n_mix_views", 3) if use_any_aug else 1
    mix_prob    = aug_config.get("mix_prob", 0.35) if use_snr else 0.0
    snr_min     = aug_config.get("snr_min_db", 0.0)
    snr_max     = aug_config.get("snr_max_db", 15.0)
    out_path    = out_path or NPZ_TRAIN

    label = f"(capped at {max_samples})" if max_samples else f"({int(sample_frac*100)}% sample)"
    if use_any_aug:
        parts = audio_aug.active_strategies() + (["SNR"] if use_snr else [])
        aug_label = f" + [{','.join(parts)}] ×{n_views}"
    else:
        aug_label = ""
    print(f"Building train embedding cache {label}{aug_label} ...")

    noise_pool: list[Path] = []
    if use_snr:
        noise_pool = sorted(TRAIN_SOUNDSCAPES.glob("*.ogg"))
        if not noise_pool:
            print("  Warning: no soundscapes for SNR mixing — SNR aug disabled.")
            use_snr = False

    rng = np.random.default_rng(42)

    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df[train_df["primary_label"].isin(species_to_idx)]
    train_df = train_df[train_df["filename"].apply(
        lambda f: (TRAIN_AUDIO_DIR / str(f)).is_file()
    )]
    chunks = [g.sample(frac=sample_frac, random_state=42) for _, g in train_df.groupby("primary_label")]
    train_df = pd.concat(chunks).reset_index(drop=True)
    if max_samples and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    n_species = len(species_to_idx)
    X, y, paths = [], [], []
    BATCH       = 256   # BirdNET embedding batch size
    IO_WORKERS  = 8     # parallel audio loaders
    LOAD_CHUNK  = 128   # rows loaded in parallel per chunk

    audio_buf, label_buf, path_buf = [], [], []

    def flush():
        if not audio_buf:
            return
        embs = _embed_batch(audio_buf, SR_LOAD)
        X.extend(embs)
        y.extend(label_buf)
        paths.extend(path_buf)
        audio_buf.clear(); label_buf.clear(); path_buf.clear()

    _n    = int(CLIP_LOAD_SEC * SR_LOAD)
    rows  = list(train_df.itertuples(index=False))
    total = len(rows)

    for chunk_start in range(0, total, LOAD_CHUNK):
        chunk_rows  = rows[chunk_start:chunk_start + LOAD_CHUNK]
        chunk_paths = [str(TRAIN_AUDIO_DIR / str(r.filename)) for r in chunk_rows]

        # Parallel I/O: load audio files concurrently
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            chunk_wavs = list(pool.map(_load_focal_safe, chunk_paths))

        for row, wav in zip(chunk_rows, chunk_wavs):
            if wav is None:
                continue
            ap_str = str(TRAIN_AUDIO_DIR / str(row.filename))

            yv = np.zeros(n_species, dtype=np.float32)
            yv[species_to_idx[row.primary_label]] = 1.0
            sec = getattr(row, "secondary_labels", None)
            if sec and sec not in ("[]", "", "nan"):
                try:
                    for lbl in ast.literal_eval(str(sec)):
                        lbl = str(lbl).strip()
                        if lbl in species_to_idx and yv[species_to_idx[lbl]] == 0.0:
                            yv[species_to_idx[lbl]] = 0.5
                except Exception:
                    pass

            # View 0: always clean; Views 1..N: augmented
            for v in range(n_views):
                if v == 0:
                    audio_view = wav
                else:
                    audio_view = wav.copy()
                    if has_audio:
                        audio_view = audio_aug.apply(audio_view, SR_LOAD)
                        audio_view = (audio_view[:_n] if len(audio_view) > _n
                                      else np.pad(audio_view, (0, _n - len(audio_view))))
                        audio_view = audio_view.astype(np.float32)
                    if use_snr and noise_pool and rng.random() < mix_prob:
                        noise = _load_random_noise(rng, noise_pool)
                        if noise is not None:
                            audio_view = _mix_snr(audio_view, noise, float(rng.uniform(snr_min, snr_max)))
                audio_buf.append(audio_view)
                label_buf.append(yv)
                path_buf.append(ap_str)

            if len(audio_buf) >= BATCH:
                flush()
                print(f"  {len(X)}/{total * n_views} embedded ...")

    flush()

    if not X:
        raise RuntimeError("No training samples embedded.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        X_train=np.stack(X).astype(np.float32),
                        y_train=np.stack(y).astype(np.float32),
                        paths=np.array(paths, dtype=object))
    print(f"Train cache saved: {out_path} ({len(X)} samples)")


def build_val_cache(species_to_idx: dict[str, int]) -> None:
    print("Building validation embedding cache (soundscapes) ...")
    if not SOUNDSCAPE_LABELS.exists():
        print("  No soundscape labels found — skipping val cache.")
        return

    n_species = len(species_to_idx)
    lab = pd.read_csv(SOUNDSCAPE_LABELS)

    def _tok(v):
        if pd.isna(v) or v == "":
            return set()
        return {t.strip() for t in str(v).split(";") if t.strip()}

    def _hms_to_sec(t: str) -> int:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[_tok(v) for v in s]))
        .reset_index()
    )

    X, y_list, row_ids = [], [], []
    BATCH = 64
    audio_buf, label_buf, rid_buf = [], [], []

    def flush():
        if not audio_buf:
            return
        embs = _embed_batch(audio_buf, SR_LOAD)
        X.extend(embs); y_list.extend(label_buf); row_ids.extend(rid_buf)
        audio_buf.clear(); label_buf.clear(); rid_buf.clear()

    for row in grp.itertuples(index=False):
        fp = TRAIN_SOUNDSCAPES / row.filename
        if not fp.exists():
            continue
        start_sec = _hms_to_sec(row.start)
        end_sec = _hms_to_sec(row.end)
        duration = end_sec - start_sec
        stem = Path(row.filename).stem
        row_id = f"{stem}_{end_sec}"
        try:
            y, _ = librosa.load(str(fp), sr=SR_LOAD, mono=True,
                                offset=start_sec, duration=float(duration))
            n = int(duration * SR_LOAD)
            y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
            wav = y.astype(np.float32)
        except Exception:
            continue
        yv = np.zeros(n_species, dtype=np.float32)
        for lbl in row.primary_label:
            if lbl in species_to_idx:
                yv[species_to_idx[lbl]] = 1.0
        audio_buf.append(wav); label_buf.append(yv); rid_buf.append(row_id)
        if len(audio_buf) >= BATCH:
            flush()
    flush()

    if not X:
        print("  No validation samples embedded.")
        return

    np.savez_compressed(NPZ_VAL,
                        X_val=np.stack(X).astype(np.float32),
                        y_val=np.stack(y_list).astype(np.float32),
                        row_ids=np.array(row_ids, dtype=object))
    print(f"Val cache saved: {NPZ_VAL} ({len(X)} samples)")


def build_pseudo_cache(max_samples: int = 500) -> None:
    """Embed unlabeled soundscape windows (not in train_soundscapes_labels) for pseudo labeling."""
    print(f"Building pseudo-label cache (max {max_samples} windows) ...")

    def _hms(t: str) -> int:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    labeled: set[tuple[str, int]] = set()
    if SOUNDSCAPE_LABELS.exists():
        lab = pd.read_csv(SOUNDSCAPE_LABELS)
        for row in lab.itertuples(index=False):
            labeled.add((Path(row.filename).stem, _hms(row.end)))

    soundscape_files = sorted(TRAIN_SOUNDSCAPES.glob("*.ogg"))
    if not soundscape_files:
        print("  No soundscape files found — skipping.")
        return

    X_list, rid_list = [], []
    BATCH = 64
    audio_buf, rid_buf = [], []

    def flush():
        if not audio_buf:
            return
        embs = _embed_batch(audio_buf, SR_LOAD)
        X_list.extend(embs)
        rid_list.extend(rid_buf)
        audio_buf.clear(); rid_buf.clear()

    count = 0
    window_sec = int(CLIP_LOAD_SEC)
    for fp in soundscape_files:
        if max_samples and count >= max_samples:
            break
        stem = fp.stem
        try:
            dur = librosa.get_duration(path=str(fp))
        except Exception:
            continue
        for end_sec in range(window_sec, int(dur) + 1, window_sec):
            if max_samples and count >= max_samples:
                break
            if (stem, end_sec) in labeled:
                continue
            start_sec = end_sec - window_sec
            try:
                y_a, _ = librosa.load(str(fp), sr=SR_LOAD, mono=True,
                                      offset=start_sec, duration=float(window_sec))
                n = int(window_sec * SR_LOAD)
                y_a = y_a[:n] if len(y_a) > n else np.pad(y_a, (0, n - len(y_a)))
                audio_buf.append(y_a.astype(np.float32))
                rid_buf.append(f"{stem}_{end_sec}")
                count += 1
            except Exception:
                continue
            if len(audio_buf) >= BATCH:
                flush()
    flush()

    if not X_list:
        print("  No unlabeled windows found.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PSEUDO,
                        X_pseudo=np.stack(X_list).astype(np.float32),
                        row_ids=np.array(rid_list, dtype=object))
    print(f"Pseudo cache saved: {NPZ_PSEUDO} ({len(X_list)} samples)")


def ensure_caches(config: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _bird_session
    species_cols, species_to_idx = _build_species_map()

    max_samples = config.get("max_train_samples") if config else None
    sample_frac = config.get("train_sample_frac", 0.30) if config else 0.30
    aug_config  = config.get("augmentation") if config else None

    _audio_aug = AudioAugmenter(aug_config.get("audio", {}) if aug_config else {})
    use_aug    = bool(aug_config and (
        aug_config.get("use_snr_mixing") or _audio_aug.active_strategies()
    ))
    npz_train  = NPZ_TRAIN_AUG if use_aug else NPZ_TRAIN

    if not npz_train.exists() or not NPZ_VAL.exists():
        init_birdnet()
        with _bird_model.encode_session(
            batch_size=EMB_BATCH_SIZE,
            prefetch_ratio=2,
            n_workers=4,
            n_producers=2,
        ) as session:
            _bird_session = session
            if not npz_train.exists():
                build_train_cache(species_to_idx, max_samples=max_samples,
                                  sample_frac=sample_frac, aug_config=aug_config,
                                  out_path=npz_train)
            if not NPZ_VAL.exists():
                build_val_cache(species_to_idx)
        _bird_session = None

    print("Loading embeddings from cache ...")
    dtr = np.load(npz_train, allow_pickle=True)
    X_train = dtr["X_train"].astype(np.float32)
    y_train = dtr["y_train"].astype(np.float32)

    dval = np.load(NPZ_VAL, allow_pickle=True)
    X_val  = dval["X_val"].astype(np.float32)
    y_val  = dval["y_val"].astype(np.float32)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Evaluation metric (verbatim from notebooks)
# ---------------------------------------------------------------------------

def macro_auc_ge3(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, int, int]:
    pos = y_true.sum(axis=0)
    keep = pos >= 3
    if not np.any(keep):
        return float("nan"), 0, int(np.sum(pos > 0))
    yt = y_true[:, keep]
    ys = y_score[:, keep]
    usable = [j for j in range(yt.shape[1]) if yt[:, j].min() == 0 and yt[:, j].max() == 1]
    if not usable:
        return float("nan"), int(np.sum(keep)), int(np.sum(pos > 0))
    usable = np.array(usable, dtype=int)
    auc = roc_auc_score(yt[:, usable], ys[:, usable], average="macro")
    return float(auc), int(np.sum(keep)), int(np.sum(pos > 0))


# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------

def _llm_call(messages: list[dict], config: dict) -> str:
    import urllib.request
    payload = json.dumps({
        "model": config["llm"]["model"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": config["llm"].get("temperature", 0.2)},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    llm_timeout = config.get("execution", {}).get("llm_timeout_seconds", 900)
    with urllib.request.urlopen(req, timeout=llm_timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _extract_code(response: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", response, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[0].strip()
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return ""


def _validate_slot(code: str) -> list[str]:
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    if not re.search(r"def\s+build_head\s*\(", code):
        issues.append("Missing: def build_head(emb_dim, num_classes)")
    if not re.search(r"def\s+get_head_config\s*\(", code):
        issues.append("Missing: def get_head_config()")
    if not re.search(r"EXPERIMENT_META\s*=\s*\{", code):
        issues.append("Missing: EXPERIMENT_META = {...}")
    if re.search(r"def\s+main\s*\(", code):
        issues.append("Do NOT define main()")
    return issues


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an ML research assistant. You write Python functions that define "
    "classification heads on top of BirdNET 1024-D audio embeddings for bird species classification.\n\n"
    "RULES:\n"
    "- Return ONLY a ```python``` code block, nothing else\n"
    "- Define exactly: get_head_config(), build_head(emb_dim, num_classes), and EXPERIMENT_META\n"
    "- Do NOT define main(), do NOT load data, do NOT embed audio\n"
    "- Input to the head is always a (N, 1024) float32 numpy array of BirdNET embeddings\n"
    "- build_head must return a compiled Keras model OR a sklearn-compatible object with fit/predict_proba\n"
    "- For Keras heads: last layer must be Dense(num_classes, activation='sigmoid')\n"
    "- num_classes is 234 (one per bird species), data is heavily class-imbalanced\n\n"
    "REQUIRED — weighted BCE loss:\n"
    "    The dataset is heavily imbalanced. You MUST use weighted BCE for Keras heads.\n"
    "    Compute pos_weight from y_train, then use it in a custom loss:\n"
    "        pos = y_train.sum(axis=0)\n"
    "        neg = len(y_train) - pos\n"
    "        pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)\n"
    "        pw = tf.constant(pos_weight[None, :], dtype=tf.float32)\n"
    "        def weighted_bce(y_true, y_pred):\n"
    "            y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)\n"
    "            return tf.reduce_mean(pw*y_true*(-tf.math.log(y_pred)) + (1-y_true)*(-tf.math.log(1-y_pred)))\n"
    "    Pass y_train as an argument: build_head(emb_dim, num_classes, y_train)\n\n"
    "REQUIRED — EarlyStopping:\n"
    "    Always use: tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')\n\n"
    "NOTE: The training embeddings may already include audio-level SNR noise mixing (harness-applied).\n"
    "      For additional augmentation at embedding level, use Mixup as described below.\n\n"
    "OPTIONS the LLM may explore freely:\n"
    "- Number of Dense layers and units\n"
    "- Dropout rate, learning rate, batch size, epochs\n"
    "- BatchNormalization\n"
    "- Embedding Mixup (strategy 2): lam ~ Beta(alpha, alpha); "
    "X_mix = lam*X + (1-lam)*X[perm]; y_mix = lam*y + (1-lam)*y[perm]; "
    "feed X_mix/y_mix to model.fit — alpha=0.2–0.4 works well\n"
    "- L2 regularisation, label smoothing\n"
    "- sklearn heads: LogisticRegression / MultiOutputClassifier\n\n"
    "get_head_config() must return a dict with at least: epochs, batch_size, learning_rate.\n\n"
    "EXPERIMENT_META must be a dict with keys:\n"
    "    head_type: 'keras_mlp' | 'logreg' | other\n"
    "    architecture: short description\n"
    "    change: what changed vs previous run, or 'baseline'\n"
    "    key_params: dict with main hyperparameters\n"
)

SEED_PROMPT = (
    "Write get_head_config() and build_head() for BirdCLEF 2026.\n\n"
    "Task: multi-label bird species classification from BirdNET 1024-D embeddings.\n"
    "Input shape: (N, 1024) float32.\n"
    "Output: 234 species probabilities (sigmoid).\n"
    "Metric: macro_auc_ge3 (AUC on species with ≥3 positive validation samples).\n\n"
    "Start with a weighted BCE MLP baseline as shown in the system prompt.\n"
    "Note: build_head receives y_train as a third argument to compute pos_weight.\n\n"
    "Starting code to modify:\n"
    "```python\n"
    "import numpy as np\n"
    "import tensorflow as tf\n\n"
    "EXPERIMENT_META = {\n"
    "    'head_type': 'keras_mlp',\n"
    "    'architecture': 'Dense(512) + Dropout(0.3) + weighted BCE',\n"
    "    'change': 'baseline',\n"
    "    'key_params': {'lr': 1e-3, 'batch_size': 256, 'epochs': 30},\n"
    "}\n\n"
    "def get_head_config():\n"
    "    return {'epochs': 30, 'batch_size': 256, 'learning_rate': 1e-3}\n\n"
    "def build_head(emb_dim, num_classes, y_train):\n"
    "    pos = y_train.sum(axis=0)\n"
    "    neg = len(y_train) - pos\n"
    "    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)\n"
    "    pw = tf.constant(pos_weight[None, :], dtype=tf.float32)\n"
    "    def weighted_bce(y_true, y_pred):\n"
    "        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)\n"
    "        return tf.reduce_mean(pw*y_true*(-tf.math.log(y_pred)) + (1-y_true)*(-tf.math.log(1-y_pred)))\n"
    "    inputs = tf.keras.Input(shape=(emb_dim,))\n"
    "    x = tf.keras.layers.Dense(512, activation='relu')(inputs)\n"
    "    x = tf.keras.layers.Dropout(0.3)(x)\n"
    "    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)\n"
    "    model = tf.keras.Model(inputs, out)\n"
    "    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_bce)\n"
    "    return model\n"
    "```"
)


# Each entry: (phase_name, directive injected into the feedback prompt for that iteration).
# Indexed as (it - 2) % len(...) so iteration 1 (SEED_PROMPT) is skipped.
_ITER_SCHEDULE: list[tuple[str, str]] = [
    ("depth",
     "Vary the NUMBER OF DENSE LAYERS: try 1, 2, 3, or 4 layers. Keep other params similar."),
    ("depth",
     "Try a DIFFERENT DEPTH from last time — add or remove at least one Dense layer."),
    ("width",
     "Vary HIDDEN UNITS PER LAYER: try 64, 128, 256, 512, or 1024. Keep the best depth."),
    ("width",
     "Try a DIFFERENT WIDTH from last time — pick a unit count you haven't tried yet."),
    ("regularisation",
     "Tune REGULARISATION: try Dropout rates 0.1–0.5 and/or add BatchNormalization after each Dense layer."),
    ("regularisation",
     "Try L2 WEIGHT DECAY (kernel_regularizer=tf.keras.regularizers.l2(1e-4)) or label smoothing on the BCE loss."),
    ("lr_batch",
     "Vary LEARNING RATE (1e-4 to 1e-2) and BATCH SIZE (64, 128, 256, 512). Keep the best architecture."),
    ("lr_batch",
     "Try a COSINE DECAY schedule: tf.keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps)."),
    ("augmentation",
     "Add EMBEDDING MIXUP inside build_head: lam ~ Beta(0.3, 0.3); "
     "perm = np.random.permutation(len(X)); "
     "X_mix = lam*X + (1-lam)*X[perm]; y_mix = lam*y + (1-lam)*y[perm]; train on X_mix/y_mix."),
    ("embedding_noise",
     "Add GAUSSIAN NOISE to embeddings during training: "
     "X_noisy = X + np.random.normal(0, 0.05, X.shape).astype(np.float32); train on X_noisy. "
     "Try noise_std in [0.01, 0.05, 0.1]. Can stack with Mixup from the previous iteration."),
    ("synthesis",
     "COMBINE the best ideas from all previous iterations into one final architecture. "
     "Include Mixup and/or embedding noise if they helped. Pick the best depth, width, regularization."),
]


# Augmentation presets for the pre-loop sweep.
# Each preset is evaluated on sweep_sample_frac of data with a fixed MLP head.
# The winner's audio+SNR params replace the main aug config; n_mix_views is kept from agent_config.
_AUG_PRESETS: list[dict] = [
    {
        "name": "no_aug",
        "use_snr_mixing": False, "n_mix_views": 1,
        "audio": {},
    },
    {
        "name": "snr_light",
        "use_snr_mixing": True, "n_mix_views": 2,
        "mix_prob": 0.20, "snr_min_db": 5.0, "snr_max_db": 20.0,
        "audio": {
            "noise_injection": {"enabled": True, "probability": 0.3, "noise_level": 0.003},
            "time_shift":      {"enabled": True, "probability": 0.4, "shift_max_fraction": 0.3},
        },
    },
    {
        "name": "snr_medium",
        "use_snr_mixing": True, "n_mix_views": 2,
        "mix_prob": 0.35, "snr_min_db": 0.0, "snr_max_db": 15.0,
        "audio": {
            "noise_injection": {"enabled": True, "probability": 0.4, "noise_level": 0.005},
            "time_shift":      {"enabled": True, "probability": 0.5, "shift_max_fraction": 0.5},
        },
    },
    {
        "name": "full_aug",
        "use_snr_mixing": True, "n_mix_views": 2,
        "mix_prob": 0.35, "snr_min_db": 0.0, "snr_max_db": 15.0,
        "audio": {
            "time_stretch":    {"enabled": True, "probability": 0.5, "rate_min": 0.9,  "rate_max": 1.1},
            "pitch_shift":     {"enabled": True, "probability": 0.3, "steps_min": -2,  "steps_max": 2},
            "noise_injection": {"enabled": True, "probability": 0.4, "noise_level": 0.005},
            "time_shift":      {"enabled": True, "probability": 0.5, "shift_max_fraction": 0.5},
            "gain_jitter":     {"enabled": True, "probability": 0.4, "min_db": -6.0, "max_db": 6.0},
        },
    },
    {
        "name": "heavy_snr",
        "use_snr_mixing": True, "n_mix_views": 2,
        "mix_prob": 0.55, "snr_min_db": 0.0, "snr_max_db": 8.0,
        "audio": {
            "noise_injection": {"enabled": True, "probability": 0.5, "noise_level": 0.01},
            "time_shift":      {"enabled": True, "probability": 0.5, "shift_max_fraction": 0.5},
            "gain_jitter":     {"enabled": True, "probability": 0.5, "min_db": -8.0, "max_db": 8.0},
        },
    },
]


def _select_views(X: np.ndarray, y: np.ndarray, n_total: int, n_active: int) -> tuple[np.ndarray, np.ndarray]:
    """Return only the first n_active augmented views per sample (for curriculum)."""
    if n_active >= n_total or n_total <= 1:
        return X, y
    n_samples = len(X) // n_total
    idx = (np.arange(n_samples)[:, None] * n_total + np.arange(n_active)[None, :]).ravel()
    return X[idx], y[idx]


def _curriculum_n_views(it: int, n_total: int) -> int:
    """Light → Heavy: clean only (iter 1) → +1 aug view (iters 2–4) → all views (iters 5+)."""
    if n_total <= 1:
        return n_total
    if it <= 1:
        return 1
    if it <= 4:
        return min(2, n_total)
    return n_total


def _build_feedback_prompt(*, auc: float | None, n_scored: int, stdout: str,
                            is_error: bool, stderr: str, current_code: str,
                            it: int = 2, n_iter: int = 10) -> str:
    if is_error:
        return (
            "The last run FAILED:\n"
            f"```\n{stderr[:3000]}\n```\n\n"
            "Fix the bug. Return the corrected get_head_config() and build_head().\n"
            f"Current code:\n```python\n{current_code}\n```"
        )
    phase_name, directive = _ITER_SCHEDULE[(it - 2) % len(_ITER_SCHEDULE)]
    metrics_str = f"macro_auc_ge3 = {auc:.6f} (scored on {n_scored} species)" if auc else "macro_auc_ge3 = N/A"
    return (
        f"Last experiment succeeded.\n\n"
        f"Results: {metrics_str}\n"
        f"Training output:\n{stdout[:2000]}\n\n"
        f"ITERATION FOCUS — {phase_name.upper()} (iter {it}/{n_iter}):\n"
        f"{directive}\n\n"
        f"Return the complete updated code.\n"
        f"Current code:\n```python\n{current_code}\n```"
    )


# ---------------------------------------------------------------------------
# Fixed harness: train head + evaluate
# ---------------------------------------------------------------------------

def _slot_predict(slot_code: str, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Re-run a slot silently and return val predictions (used for ensemble / pseudo eval)."""
    ns: dict = {}
    exec(slot_code, ns)
    build_head = ns["build_head"]
    cfg        = ns["get_head_config"]()

    import inspect, io, contextlib
    sig = inspect.signature(build_head)
    if len(sig.parameters) >= 3:
        head = build_head(X_train.shape[1], y_train.shape[1], y_train)
    else:
        head = build_head(X_train.shape[1], y_train.shape[1])

    with contextlib.redirect_stdout(io.StringIO()):
        if hasattr(head, "fit") and hasattr(head, "predict"):
            import tensorflow as tf
            cb = tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            )
            head.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=cfg.get("epochs", 30), batch_size=cfg.get("batch_size", 256),
                     callbacks=[cb], verbose=0, sample_weight=sample_weight)
            return head.predict(X_val, verbose=0).astype(np.float32)
        else:
            head.fit(X_train, y_train)
            return head.predict_proba(X_val).astype(np.float32)


def ensemble_top_slots(top_slots: list[tuple[float, str]],
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       top_n: int = 3) -> float | None:
    candidates = sorted(top_slots, key=lambda x: x[0], reverse=True)[:top_n]
    if len(candidates) < 2:
        print("  Need at least 2 successful slots for ensemble — skipping.")
        return None
    print(f"\n{'='*60}\n  ENSEMBLE (top {len(candidates)} slots)\n{'='*60}")
    preds = []
    for rank, (solo_auc, code) in enumerate(candidates, 1):
        print(f"  Re-running slot {rank} (solo AUC={solo_auc:.6f}) ...", end=" ", flush=True)
        try:
            y_pred = _slot_predict(code, X_train, y_train, X_val, y_val)
            preds.append(y_pred)
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
    if len(preds) < 2:
        print("  Not enough predictions for ensemble.")
        return None
    y_ens = np.mean(preds, axis=0)
    ens_auc, n_scored, _ = macro_auc_ge3(y_val, y_ens)
    print(f"  Ensemble macro_auc_ge3 = {ens_auc:.6f} | scored = {n_scored}")
    return ens_auc


def multiseed_ensemble(slot_code: str,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       n_seeds: int = 5) -> float | None:
    """Train the best slot N times with different random seeds and average predictions."""
    print(f"\n{'='*60}\n  MULTI-SEED ENSEMBLE (best slot × {n_seeds} seeds)\n{'='*60}")
    preds = []
    for seed in range(n_seeds):
        print(f"  Seed {seed} ...", end=" ", flush=True)
        try:
            np.random.seed(seed)
            import tensorflow as tf
            tf.random.set_seed(seed)
            y_pred = _slot_predict(slot_code, X_train, y_train, X_val, y_val)
            preds.append(y_pred)
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
    if len(preds) < 2:
        print("  Not enough predictions for multi-seed ensemble.")
        return None
    y_ens = np.mean(preds, axis=0)
    auc, n_scored, _ = macro_auc_ge3(y_val, y_ens)
    print(f"  Multi-seed macro_auc_ge3 = {auc:.6f} | scored = {n_scored}")
    return auc


def _slot_predict_unlabeled(slot_code: str, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_unlabeled: np.ndarray) -> np.ndarray:
    """Train slot on X_train, return predictions on X_unlabeled (not X_val)."""
    ns: dict = {}
    exec(slot_code, ns)
    build_head = ns["build_head"]
    cfg        = ns["get_head_config"]()

    import inspect, io, contextlib
    sig = inspect.signature(build_head)
    if len(sig.parameters) >= 3:
        head = build_head(X_train.shape[1], y_train.shape[1], y_train)
    else:
        head = build_head(X_train.shape[1], y_train.shape[1])

    with contextlib.redirect_stdout(io.StringIO()):
        if hasattr(head, "fit") and hasattr(head, "predict"):
            import tensorflow as tf
            cb = tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            )
            head.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=cfg.get("epochs", 30), batch_size=cfg.get("batch_size", 256),
                     callbacks=[cb], verbose=0)
            return head.predict(X_unlabeled, verbose=0).astype(np.float32)
        else:
            head.fit(X_train, y_train)
            return head.predict_proba(X_unlabeled).astype(np.float32)


def pseudo_label_round(best_slot: str, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       scaler, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Semi-supervised phase: embed unlabeled soundscape windows, predict with the best
    head, accept high-confidence outputs as soft labels (weight 0.8), return an
    expanded (X_train, y_train).
    """
    global _bird_session
    pl_cfg       = config.get("pseudo_labeling", {})
    threshold    = pl_cfg.get("threshold", 0.5)
    runnerup_max = pl_cfg.get("runnerup_max", 0.4)
    max_ps       = pl_cfg.get("max_pseudo_samples", 500)

    print(f"\n{'='*60}")
    print(f"  PSEUDO LABELING (threshold={threshold}, max_windows={max_ps})")
    print(f"{'='*60}")

    if not NPZ_PSEUDO.exists():
        init_birdnet()
        with _bird_model.encode_session(
            batch_size=EMB_BATCH_SIZE, prefetch_ratio=2, n_workers=4, n_producers=2
        ) as session:
            _bird_session = session
            build_pseudo_cache(max_samples=max_ps)
        _bird_session = None

    if not NPZ_PSEUDO.exists():
        print("  Pseudo cache empty — skipping.")
        return X_train, y_train, None

    data     = np.load(NPZ_PSEUDO, allow_pickle=True)
    X_pseudo = scaler.transform(data["X_pseudo"].astype(np.float32))

    print(f"  Predicting on {len(X_pseudo)} unlabeled windows ...")
    try:
        y_pseudo = _slot_predict_unlabeled(best_slot, X_train, y_train, X_val, y_val, X_pseudo)
    except Exception as exc:
        print(f"  Prediction failed: {exc} — skipping.")
        return X_train, y_train, None

    # Two-threshold acceptance: top-1 confident AND runner-up not too close (single-species windows)
    sorted_probs = np.sort(y_pseudo, axis=1)[:, ::-1]
    accepted = (sorted_probs[:, 0] >= threshold) & (sorted_probs[:, 1] < runnerup_max)
    n_acc = int(accepted.sum())
    print(f"  Accepted {n_acc}/{len(X_pseudo)} windows "
          f"(top1 ≥ {threshold} AND runner-up < {runnerup_max})")

    if n_acc == 0:
        print("  No confident predictions — pseudo labels not added.")
        return X_train, y_train, None

    # Binarise at threshold; down-weight to 0.8 to reflect label uncertainty
    y_pl  = (y_pseudo[accepted] >= threshold).astype(np.float32) * 0.8
    X_new = np.concatenate([X_train, X_pseudo[accepted]], axis=0)
    y_new = np.concatenate([y_train, y_pl], axis=0)
    # Sample weights: focal clips = 1.0, pseudo windows = 0.5
    sw = np.concatenate([
        np.ones(len(X_train), dtype=np.float32),
        np.full(n_acc, 0.5, dtype=np.float32),
    ])
    print(f"  Training set: {len(X_train)} → {len(X_new)} (+{n_acc} pseudo-labeled)")
    return X_new, y_new, sw


def _make_sweep_slot(hidden: int, n_layers: int, dropout: float, lr: float) -> str:
    """Generate executable slot code for a parameterised weighted-BCE MLP."""
    layer_lines = "\n".join(
        f"    x = tf.keras.layers.Dense({hidden}, activation='relu')(x)\n"
        f"    x = tf.keras.layers.Dropout({dropout})(x)"
        for _ in range(n_layers)
    )
    return (
        "import numpy as np\n"
        "import tensorflow as tf\n\n"
        f"EXPERIMENT_META = {{\n"
        f"    'head_type': 'keras_mlp',\n"
        f"    'architecture': 'Sweep {n_layers}x Dense({hidden}) + Dropout({dropout})',\n"
        f"    'change': 'hyperparam_sweep',\n"
        f"    'key_params': {{'hidden': {hidden}, 'n_layers': {n_layers}, 'dropout': {dropout}, 'lr': {lr}}},\n"
        f"}}\n\n"
        f"def get_head_config():\n"
        f"    return {{'epochs': 50, 'batch_size': 256, 'learning_rate': {lr}}}\n\n"
        f"def build_head(emb_dim, num_classes, y_train):\n"
        f"    pos = y_train.sum(axis=0)\n"
        f"    neg = len(y_train) - pos\n"
        f"    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)\n"
        f"    pw = tf.constant(pos_weight[None, :], dtype=tf.float32)\n"
        f"    def weighted_bce(y_true, y_pred):\n"
        f"        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)\n"
        f"        return tf.reduce_mean(pw*y_true*(-tf.math.log(y_pred)) + (1-y_true)*(-tf.math.log(1-y_pred)))\n"
        f"    inputs = tf.keras.Input(shape=(emb_dim,))\n"
        f"    x = inputs\n"
        f"{layer_lines}\n"
        f"    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)\n"
        f"    model = tf.keras.Model(inputs, out)\n"
        f"    model.compile(optimizer=tf.keras.optimizers.Adam({lr}), loss=weighted_bce)\n"
        f"    return model\n"
    )


def hyperparam_sweep(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     config: dict) -> tuple[float | None, dict | None]:
    """
    Random search over (hidden, n_layers, dropout, lr).
    Runs each candidate through the standard harness and returns the best (auc, params).
    """
    sweep_cfg = config.get("hyperparam_sweep", {})
    if not sweep_cfg.get("enabled", False):
        return None, None

    n_trials        = sweep_cfg.get("n_trials", 20)
    hidden_choices  = sweep_cfg.get("hidden",   [128, 256, 512])
    layer_choices   = sweep_cfg.get("n_layers", [1, 2, 3])
    dropout_choices = sweep_cfg.get("dropout",  [0.2, 0.3, 0.4])
    lr_choices      = sweep_cfg.get("lr",       [1e-4, 1e-3, 3e-3])

    rng = np.random.default_rng(config.get("random_seed", 42) + 999)

    print(f"\n{'='*60}\n  HYPERPARAMETER SWEEP ({n_trials} random trials)\n{'='*60}")

    best_auc: float = 0.0
    best_params: dict | None = None

    for trial in range(1, n_trials + 1):
        hidden   = int(rng.choice(hidden_choices))
        n_layers = int(rng.choice(layer_choices))
        dropout  = float(rng.choice(dropout_choices))
        lr       = float(rng.choice(lr_choices))

        slot = _make_sweep_slot(hidden, n_layers, dropout, lr)
        print(f"  [{trial:2d}/{n_trials}] {n_layers}×Dense({hidden:4d}) "
              f"drop={dropout} lr={lr:.0e}", end=" → ", flush=True)
        try:
            auc, _, _ = run_slot(slot, X_train, y_train, X_val, y_val)
            print(f"AUC={auc:.6f}")
            if auc is not None and auc > best_auc:
                best_auc    = auc
                best_params = {"hidden": hidden, "n_layers": n_layers,
                               "dropout": dropout, "lr": lr}
        except Exception as exc:
            print(f"FAILED ({exc})")

    print(f"\n  Best sweep AUC = {best_auc:.6f}")
    if best_params:
        print(f"  Best params    = {best_params}")
    return best_auc, best_params


def run_slot(slot_code: str, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             sample_weight: np.ndarray | None = None) -> tuple[float | None, int, str]:
    """Execute LLM slot code, train the head, return (auc, n_scored, stdout)."""
    ns: dict = {}
    exec(slot_code, ns)

    build_head   = ns["build_head"]
    get_config   = ns["get_head_config"]
    cfg          = get_config()
    epochs       = cfg.get("epochs", 30)
    batch_size   = cfg.get("batch_size", 256)
    lr           = cfg.get("learning_rate", 1e-3)

    import io, contextlib
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        import inspect
        sig = inspect.signature(build_head)
        if len(sig.parameters) >= 3:
            head = build_head(X_train.shape[1], y_train.shape[1], y_train)
        else:
            head = build_head(X_train.shape[1], y_train.shape[1])

        # Detect Keras model vs sklearn
        is_keras = hasattr(head, "fit") and hasattr(head, "predict")
        if is_keras:
            import tensorflow as tf
            # recompile with provided lr in case build_head hardcoded it
            head.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss=head.loss,
            )
            cb = tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            )
            head.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[cb],
                verbose=1,
                sample_weight=sample_weight,
            )
            y_pred = head.predict(X_val, verbose=0).astype(np.float32)
        else:
            head.fit(X_train, y_train)
            y_pred = head.predict_proba(X_val).astype(np.float32)

        auc, n_scored, _ = macro_auc_ge3(y_val, y_pred)
        print(f"macro_auc_ge3 = {auc:.6f} | scored_species = {n_scored}")

    return auc, n_scored, buf.getvalue()


# ---------------------------------------------------------------------------
# Kaggle submission artifact generator
# ---------------------------------------------------------------------------

def _generate_submission_artifacts(logs_dir: Path, best_slot: str,
                                    final_auc: float | None, config: dict) -> None:
    """
    Write kaggle_inference.ipynb + dataset-metadata.json to a timestamped
    submission_archive subfolder.  model.keras and scaler.npz must already
    exist in logs_dir before calling this.
    """
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    sub_dir  = PROJECT_ROOT / "submission_archive" / f"sub_birdnet_{date_str}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    (sub_dir / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "BirdNET Agent Submission Files",
            "id": "alexanderwanstrath/birdnet-agent-submission",
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2),
        encoding="utf-8",
    )

    def _code(lines: list[str]) -> dict:
        return {"cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": lines}

    def _md(lines: list[str]) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": lines}

    auc_note = f" (val macro_auc_ge3 = {final_auc:.6f})" if final_auc else ""
    cells = [
        _md([
            "# BirdCLEF 2026 — BirdNET Inference Notebook\n",
            f"Auto-generated by the autonomous BirdNET embedding agent{auc_note}.\n\n",
            "**Upload workflow:**\n",
            "1. Copy `model.keras` + `scaler.npz` from `logs/birdnet_agent/` into this folder\n",
            "2. Upload folder to Kaggle as dataset (`kaggle datasets create -p <this-folder>`)\n",
            "3. Set `MODEL_DIR` below to match your dataset slug\n",
            "4. Run all cells → `submission.csv` written to `/kaggle/working/`\n",
        ]),
        _code([
            "!pip install birdnet -q\n",
            "\n",
            "import os, numpy as np, pandas as pd, librosa, tensorflow as tf\n",
            "from pathlib import Path\n",
            "print('Libraries loaded.')",
        ]),
        _code([
            "COMP_DIR   = '/kaggle/input/birdclef-2026'\n",
            "MODEL_DIR  = '/kaggle/input/birdnet-agent-submission'\n",
            "\n",
            "SR_LOAD    = 32000\n",
            "BIRDNET_SR = 48000\n",
            "CLIP_SEC   = 5.0\n",
            "CHUNK_SEC  = 3.0\n",
            "BATCH      = 32",
        ]),
        _code([
            "import birdnet\n",
            "print('Loading BirdNET v2.4 ...')\n",
            "bnet = birdnet.load('acoustic', '2.4', 'tf')\n",
            "\n",
            "model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model.keras'))\n",
            "sc       = np.load(os.path.join(MODEL_DIR, 'scaler.npz'))\n",
            "sc_mean  = sc['mean'].astype(np.float32)\n",
            "sc_scale = sc['scale'].astype(np.float32)\n",
            "\n",
            "sample_sub   = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))\n",
            "species_cols = [c for c in sample_sub.columns if c != 'row_id']\n",
            "print(f'Ready. Species: {len(species_cols)}')",
        ]),
        _code([
            "def _prepare_chunk(y, sr):\n",
            "    y = np.asarray(y, dtype=np.float32).reshape(-1)\n",
            "    if sr != BIRDNET_SR:\n",
            "        y = librosa.resample(y, orig_sr=sr, target_sr=BIRDNET_SR)\n",
            "    keep = int(CLIP_SEC * BIRDNET_SR)\n",
            "    y = y[:keep] if len(y) > keep else y\n",
            "    need = int(CHUNK_SEC * BIRDNET_SR)\n",
            "    s0 = max(0, (len(y) - need) // 2)\n",
            "    chunk = y[s0:s0+need]\n",
            "    return np.pad(chunk, (0, max(0, need - len(chunk)))).astype(np.float32)\n",
            "\n",
            "def _embed_predict(audio_list, session):\n",
            "    inputs = [(np.ascontiguousarray(_prepare_chunk(y, SR_LOAD), np.float32), BIRDNET_SR)\n",
            "              for y in audio_list]\n",
            "    res  = session.run_arrays(inputs)\n",
            "    embs = np.asarray(res.embeddings[:, 0, :], np.float32)\n",
            "    embs = (embs - sc_mean) / sc_scale\n",
            "    return model.predict(embs, verbose=0).astype(np.float32)",
        ]),
        _code([
            "test_dir   = Path(COMP_DIR) / 'test_soundscapes'\n",
            "test_files = sorted(test_dir.glob('*.ogg'))\n",
            "print(f'Test soundscapes: {len(test_files)}')\n",
            "\n",
            "clip_n    = int(CLIP_SEC * SR_LOAD)\n",
            "rows      = []\n",
            "audio_buf = []\n",
            "row_ids   = []\n",
            "\n",
            "with bnet.encode_session(batch_size=BATCH, prefetch_ratio=2, n_workers=4, n_producers=2) as sess:\n",
            "    def flush():\n",
            "        if not audio_buf: return\n",
            "        preds = _embed_predict(audio_buf, sess)\n",
            "        for rid, pred in zip(row_ids, preds):\n",
            "            r = {'row_id': rid}\n",
            "            r.update(zip(species_cols, pred.tolist()))\n",
            "            rows.append(r)\n",
            "        audio_buf.clear(); row_ids.clear()\n",
            "\n",
            "    for fpath in test_files:\n",
            "        stem   = fpath.stem\n",
            "        y_full, _ = librosa.load(str(fpath), sr=SR_LOAD, mono=True)\n",
            "        n_win  = max(1, int(np.ceil(len(y_full) / clip_n)))\n",
            "        for w in range(n_win):\n",
            "            seg = y_full[w*clip_n:(w+1)*clip_n]\n",
            "            if len(seg) < clip_n:\n",
            "                seg = np.pad(seg, (0, clip_n - len(seg)))\n",
            "            audio_buf.append(seg.astype(np.float32))\n",
            "            row_ids.append(f'{stem}_{(w+1)*int(CLIP_SEC)}')\n",
            "            if len(audio_buf) >= BATCH:\n",
            "                flush()\n",
            "    flush()\n",
            "\n",
            "print(f'Predicted {len(rows)} windows')",
        ]),
        _code([
            "sub = pd.DataFrame(rows)\n",
            "sub = sample_sub[['row_id']].merge(sub, on='row_id', how='left')\n",
            "sub[species_cols] = sub[species_cols].fillna(1.0 / len(species_cols)).clip(0.0, 1.0)\n",
            "sub.to_csv('/kaggle/working/submission.csv', index=False)\n",
            "print(f'Submission saved: {sub.shape}')\n",
            "sub.head(3)",
        ]),
    ]

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    (sub_dir / "kaggle_inference.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
    (sub_dir / "best_slot.py").write_text(best_slot, encoding="utf-8")

    print(f"\n  Submission artifacts → {sub_dir}")
    print(f"  Steps to submit:")
    print(f"  1. Edit {sub_dir/'dataset-metadata.json'}: replace <your-kaggle-username>")
    print(f"  2. Copy logs/birdnet_agent/model.keras + scaler.npz into {sub_dir.name}/")
    print(f"  3. kaggle datasets create -p \"{sub_dir}\"")
    print(f"  4. On Kaggle: add that dataset + birdclef-2026 → run kaggle_inference.ipynb")


# ---------------------------------------------------------------------------
# Final retrain on full data
# ---------------------------------------------------------------------------

def final_retrain(best_slot: str, X_val: np.ndarray, y_val: np.ndarray,
                  config: dict) -> float | None:
    """
    Retrain the best head on 100 % of the training data and evaluate on val.
    Uses a fresh StandardScaler fitted on the full training set so normalisation
    is as accurate as possible.  The val embeddings are rescaled accordingly.
    """
    global _bird_session
    fr_cfg = config.get("final_retrain", {})
    if not fr_cfg.get("enabled", False):
        return None
    if not best_slot:
        print("  No best slot available — skipping final retrain.")
        return None

    print(f"\n{'='*60}\n  FINAL RETRAIN (full training data)\n{'='*60}")

    aug_config = config.get("augmentation")
    _a = AudioAugmenter(aug_config.get("audio", {}) if aug_config else {})
    use_aug  = bool(aug_config and (aug_config.get("use_snr_mixing") or _a.active_strategies()))
    npz_full = NPZ_TRAIN_AUG_FULL if use_aug else NPZ_TRAIN_FULL

    if not npz_full.exists():
        print("  Building full training cache (sample_frac=1.0) ...")
        _, species_to_idx = _build_species_map()
        init_birdnet()
        with _bird_model.encode_session(
            batch_size=EMB_BATCH_SIZE, prefetch_ratio=2, n_workers=4, n_producers=2
        ) as session:
            _bird_session = session
            build_train_cache(species_to_idx, max_samples=None, sample_frac=1.0,
                              aug_config=aug_config, out_path=npz_full)
        _bird_session = None

    if not npz_full.exists():
        print("  Full cache build failed — skipping.")
        return None

    dtr    = np.load(npz_full, allow_pickle=True)
    X_full = dtr["X_train"].astype(np.float32)
    y_full = dtr["y_train"].astype(np.float32)
    print(f"  Full train: {X_full.shape}")

    from sklearn.preprocessing import StandardScaler
    scaler_full   = StandardScaler().fit(X_full)
    X_full_scaled = scaler_full.transform(X_full)
    X_val_scaled  = scaler_full.transform(X_val)

    print("  Training best slot on full data ...")
    try:
        import inspect
        ns: dict = {}
        exec(best_slot, ns)
        build_head = ns["build_head"]
        cfg        = ns["get_head_config"]()
        sig        = inspect.signature(build_head)
        head = (build_head(X_full_scaled.shape[1], y_full.shape[1], y_full)
                if len(sig.parameters) >= 3
                else build_head(X_full_scaled.shape[1], y_full.shape[1]))

        import tensorflow as tf
        is_keras = hasattr(head, "fit") and hasattr(head, "predict")
        if is_keras:
            cb = tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            )
            head.fit(X_full_scaled, y_full,
                     validation_data=(X_val_scaled, y_val),
                     epochs=cfg.get("epochs", 30),
                     batch_size=cfg.get("batch_size", 256),
                     callbacks=[cb], verbose=1)
            y_pred = head.predict(X_val_scaled, verbose=0).astype(np.float32)

            model_path  = LOGS_DIR / "model.keras"
            scaler_path = LOGS_DIR / "scaler.npz"
            head.save(str(model_path))
            np.savez(str(scaler_path),
                     mean=scaler_full.mean_, scale=scaler_full.scale_)
            print(f"  Model saved:  {model_path}")
            print(f"  Scaler saved: {scaler_path}")
        else:
            head.fit(X_full_scaled, y_full)
            y_pred = head.predict_proba(X_val_scaled).astype(np.float32)

        auc, n_scored, _ = macro_auc_ge3(y_val, y_pred)
        print(f"  Full-retrain macro_auc_ge3 = {auc:.6f} | scored = {n_scored}")
        (LOGS_DIR / "best_slot_full.py").write_text(best_slot, encoding="utf-8")

        if is_keras:
            _generate_submission_artifacts(LOGS_DIR, best_slot, auc, config)

        return auc
    except Exception as exc:
        import traceback
        print(f"  Final retrain failed: {exc}")
        print(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Augmentation preset sweep (runs before the LLM loop)
# ---------------------------------------------------------------------------

def aug_preset_sweep(species_to_idx: dict, config: dict) -> dict:
    """
    Build small (sweep_sample_frac) caches for each preset in _AUG_PRESETS,
    evaluate each with a fixed weighted-BCE MLP, and return the aug_config of
    the winner.  The winning preset's audio+SNR params are returned; n_mix_views
    is taken from the main config so the actual LLM-loop run uses more views.
    """
    global _bird_session
    sweep_cfg     = config.get("aug_preset_sweep", {})
    sweep_frac    = sweep_cfg.get("sweep_sample_frac", 0.10)
    fixed_hidden  = sweep_cfg.get("fixed_head_hidden", 256)
    fixed_dropout = sweep_cfg.get("fixed_head_dropout", 0.3)
    main_n_views  = config.get("augmentation", {}).get("n_mix_views", 3)

    print(f"\n{'='*60}")
    print(f"  AUG PRESET SWEEP  ({len(_AUG_PRESETS)} presets | {sweep_frac*100:.0f}% data)")
    print(f"{'='*60}")

    # Ensure val cache exists first
    if not NPZ_VAL.exists():
        init_birdnet()
        with _bird_model.encode_session(
            batch_size=EMB_BATCH_SIZE, prefetch_ratio=2, n_workers=4, n_producers=2
        ) as session:
            _bird_session = session
            build_val_cache(species_to_idx)
        _bird_session = None

    dval      = np.load(NPZ_VAL, allow_pickle=True)
    X_val_raw = dval["X_val"].astype(np.float32)
    y_val     = dval["y_val"].astype(np.float32)

    # Build any missing preset caches in one BirdNET session
    missing = [p for p in _AUG_PRESETS
               if not (CACHE_DIR / f"aug_sweep_{p['name']}.npz").exists()]
    if missing:
        init_birdnet()
        with _bird_model.encode_session(
            batch_size=EMB_BATCH_SIZE, prefetch_ratio=2, n_workers=4, n_producers=2
        ) as session:
            _bird_session = session
            for preset in missing:
                aug = {k: v for k, v in preset.items() if k != "name"}
                print(f"\n  Building cache '{preset['name']}' ({sweep_frac*100:.0f}% data) ...")
                try:
                    build_train_cache(
                        species_to_idx,
                        max_samples=None,
                        sample_frac=sweep_frac,
                        aug_config=aug,
                        out_path=CACHE_DIR / f"aug_sweep_{preset['name']}.npz",
                    )
                except Exception as exc:
                    print(f"    Cache build failed: {exc}")
        _bird_session = None

    # Fixed slot used for all preset evaluations
    fixed_slot = _make_sweep_slot(hidden=fixed_hidden, n_layers=1,
                                  dropout=fixed_dropout, lr=1e-3)

    from sklearn.preprocessing import StandardScaler
    best_auc    = -1.0
    best_preset = _AUG_PRESETS[0]

    for preset in _AUG_PRESETS:
        cache_path = CACHE_DIR / f"aug_sweep_{preset['name']}.npz"
        if not cache_path.exists():
            print(f"  [{preset['name']}] cache missing — skip")
            continue

        data   = np.load(cache_path, allow_pickle=True)
        X_tr   = data["X_train"].astype(np.float32)
        y_tr   = data["y_train"].astype(np.float32)
        sc     = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_v_s  = sc.transform(X_val_raw)

        print(f"  [{preset['name']:12s}] {len(X_tr):5d} samples ...", end=" ", flush=True)
        try:
            auc, n_scored, _ = run_slot(fixed_slot, X_tr_s, y_tr, X_v_s, y_val)
            print(f"AUC={auc:.6f} (scored={n_scored})")
            if auc is not None and auc > best_auc:
                best_auc    = auc
                best_preset = preset
        except Exception as exc:
            print(f"FAILED ({exc})")

    print(f"\n  Winner: '{best_preset['name']}' (AUC={best_auc:.6f})")

    # Return winning aug config, keeping the main run's n_mix_views
    winning = {k: v for k, v in best_preset.items() if k != "name"}
    winning["n_mix_views"] = main_n_views
    return winning


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _trim_messages(messages: list[dict], max_turns: int = 6) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    rest   = [m for m in messages if m["role"] != "system"]
    return system + rest[-(max_turns * 2):]


def _save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def agent_loop(config: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: augmentation preset sweep (optional, before main cache) ──
    if config.get("aug_preset_sweep", {}).get("enabled", False):
        _species_cols, _species_to_idx = _build_species_map()
        winning_aug = aug_preset_sweep(_species_to_idx, config)
        config = {**config, "augmentation": winning_aug}
        # Remove stale aug cache so ensure_caches rebuilds with the winning config
        for _stale in [NPZ_TRAIN_AUG, NPZ_TRAIN_AUG_FULL]:
            if _stale.exists():
                _stale.unlink()
                print(f"  Removed stale cache: {_stale.name} (will rebuild with winning aug)")

    # ── Phase 1: embeddings ──
    X_train, y_train, X_val, y_val = ensure_caches(config)

    from sklearn.preprocessing import StandardScaler
    _scaler = StandardScaler().fit(X_train)
    X_train = _scaler.transform(X_train)
    X_val   = _scaler.transform(X_val)
    print("Embeddings standardized (zero-mean, unit-variance).")

    # Curriculum aug config — used to select views per iteration
    _aug_cfg      = config.get("augmentation", {})
    _use_aug      = _aug_cfg.get("use_snr_mixing", False) or bool(_aug_cfg.get("audio", {}))
    n_total_views = _aug_cfg.get("n_mix_views", 3) if _use_aug else 1

    n_iter   = config.get("max_iterations", 10)
    max_fail = config.get("max_failures_before_stop", 3)

    best_auc  = 0.0
    best_slot = ""
    top_slots: list[tuple[float, str]] = []
    history: list[dict] = []
    messages  = [{"role": "system", "content": SYSTEM_PROMPT}]

    consec_fail = 0
    current_slot = ""

    print("\n" + "=" * 60)
    print("  BirdNET Embedding Agent")
    print(f"  Model: {config['llm']['model']}  |  Iterations: {n_iter}")
    print("=" * 60)

    for it in range(1, n_iter + 1):
        print(f"\n{'─'*50}\n  ITERATION {it}/{n_iter}\n{'─'*50}")

        # ── LLM query ──
        user_msg = SEED_PROMPT if it == 1 else _build_feedback_prompt(
            auc=history[-1].get("macro_auc_ge3") if history else None,
            n_scored=history[-1].get("n_scored", 0) if history else 0,
            stdout=history[-1].get("stdout", "") if history else "",
            is_error=history[-1].get("status") == "failed" if history else False,
            stderr=history[-1].get("stderr", "") if history else "",
            current_code=current_slot,
            it=it,
            n_iter=n_iter,
        )
        messages = _trim_messages(messages)
        messages.append({"role": "user", "content": user_msg})

        print("Querying LLM ...")
        t0 = time.time()
        try:
            resp = _llm_call(messages, config)
        except TimeoutError:
            print(f"  LLM timed out after {time.time() - t0:.0f}s — skipping iteration.")
            consec_fail += 1
            if consec_fail >= max_fail:
                break
            continue
        except Exception as llm_exc:
            print(f"  LLM call failed: {llm_exc} — skipping iteration.")
            consec_fail += 1
            if consec_fail >= max_fail:
                break
            continue
        print(f"LLM responded in {time.time() - t0:.1f}s")

        slot = _extract_code(resp)
        if not slot:
            print("  No code block found — skipping.")
            consec_fail += 1
            if consec_fail >= max_fail:
                break
            continue

        issues = _validate_slot(slot)
        if issues:
            print(f"  Slot validation failed: {issues}")
            consec_fail += 1
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                             f"Your code has issues: {issues}\nFix them and return the corrected code."})
            if consec_fail >= max_fail:
                break
            continue

        # extract EXPERIMENT_META for display
        try:
            ns: dict = {}
            exec(slot, ns)
            meta = ns.get("EXPERIMENT_META", {})
        except Exception:
            meta = {}
        print(f"  [{meta.get('head_type', '?')}] {meta.get('architecture', '')} — {meta.get('change', '')}")

        # ── Run (curriculum: fewer aug views early, all views later) ──
        n_active = _curriculum_n_views(it, n_total_views)
        X_tr, y_tr = _select_views(X_train, y_train, n_total_views, n_active)
        if n_total_views > 1:
            print(f"  Curriculum aug: {n_active}/{n_total_views} views → {len(X_tr)} train samples")
        try:
            t0 = time.time()
            auc, n_scored, stdout = run_slot(slot, X_tr, y_tr, X_val, y_val)
            dt = time.time() - t0
            print(f"  SUCCESS ({dt:.1f}s)")
            print(f"  macro_auc_ge3 = {auc:.6f} | scored = {n_scored}")
            consec_fail = 0
            current_slot = slot

            entry = {
                "iteration": it, "status": "success",
                "macro_auc_ge3": auc, "n_scored": n_scored,
                "time": round(dt, 1), "stdout": stdout[:2000],
                "experiment_meta": meta,
            }
            if auc is not None:
                top_slots.append((auc, slot))
            if auc is not None and auc > best_auc:
                best_auc  = auc
                best_slot = slot
                print(f"  ★ NEW BEST ({best_auc:.6f})")
                (LOGS_DIR / "best_slot.py").write_text(slot, encoding="utf-8")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            dt = time.time() - t0
            print(f"  FAILED ({dt:.1f}s): {e}")
            consec_fail += 1
            entry = {
                "iteration": it, "status": "failed",
                "time": round(dt, 1), "stderr": tb,
                "experiment_meta": meta,
            }

        history.append(entry)
        messages.append({"role": "assistant", "content": resp})
        _save_history(history, LOGS_DIR / "history.json")

        if consec_fail >= max_fail:
            print(f"\nStopping: {max_fail} consecutive failures.")
            break

    # ── Summary ──
    print(f"\n{'='*60}")
    ok = sum(1 for e in history if e["status"] == "success")
    print(f"  DONE: {ok}/{len(history)} successful | best macro_auc_ge3 = {best_auc:.6f}")
    print(f"{'='*60}")

    if best_slot:
        (LOGS_DIR / "best_slot.py").write_text(best_slot, encoding="utf-8")
        print(f"  Best head saved to: {LOGS_DIR / 'best_slot.py'}")

    ens_auc = ensemble_top_slots(top_slots, X_train, y_train, X_val, y_val, top_n=3)
    if ens_auc is not None:
        print(f"  Arch-ensemble vs best single: {ens_auc:.6f} vs {best_auc:.6f} ({ens_auc - best_auc:+.6f})")

    if best_slot:
        ms_auc = multiseed_ensemble(best_slot, X_train, y_train, X_val, y_val, n_seeds=5)
        if ms_auc is not None:
            print(f"  Multi-seed vs best single:   {ms_auc:.6f} vs {best_auc:.6f} ({ms_auc - best_auc:+.6f})")

    # ── Phase 2b: hyperparameter sweep ──
    sweep_auc, sweep_params = hyperparam_sweep(X_train, y_train, X_val, y_val, config)
    if sweep_auc is not None:
        print(f"  Sweep vs agent best:         {sweep_auc:.6f} vs {best_auc:.6f} ({sweep_auc - best_auc:+.6f})")

    # ── Phase 3: pseudo labeling ──

    pl_cfg = config.get("pseudo_labeling", {})
    if pl_cfg.get("enabled", False) and best_slot:
        X_train_pl, y_train_pl, sw_pl = pseudo_label_round(
            best_slot, X_train, y_train, X_val, y_val, _scaler, config
        )
        if len(X_train_pl) > len(X_train):
            print("\n  Re-evaluating best slot with pseudo-labeled training data ...")
            try:
                y_pred_pl = _slot_predict(best_slot, X_train_pl, y_train_pl, X_val, y_val,
                                          sample_weight=sw_pl)
                pl_auc, _, _ = macro_auc_ge3(y_val, y_pred_pl)
                if pl_auc is not None:
                    print(f"  Pseudo-label AUC vs best: {pl_auc:.6f} vs {best_auc:.6f} ({pl_auc - best_auc:+.6f})")
            except Exception as exc:
                print(f"  Pseudo re-run failed: {exc}")

    # ── Phase 4: final retrain on full data ──
    # Use the better of the LLM agent's best slot and the sweep's best slot
    fr_slot  = best_slot
    fr_label = f"agent best ({best_auc:.6f})"
    ref_auc  = best_auc
    if sweep_auc is not None and sweep_params is not None and sweep_auc > best_auc:
        fr_slot  = _make_sweep_slot(**sweep_params)
        fr_label = f"sweep best ({sweep_auc:.6f})"
        ref_auc  = sweep_auc

    full_auc = final_retrain(fr_slot, X_val, y_val, config)
    if full_auc is not None:
        print(f"  Full-retrain ({fr_label}) vs subset: {full_auc:.6f} vs {ref_auc:.6f} ({full_auc - ref_auc:+.6f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config_path = PROJECT_ROOT / "configs" / "agent_config.json"
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    agent_loop(config)


if __name__ == "__main__":
    main()
