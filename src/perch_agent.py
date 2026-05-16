"""
perch_agent.py — Autonomous Perch-based BirdCLEF agent.

Researcher/Coder architecture over Google Perch 1536-d embeddings (ONNX).

Pipeline:
  1. Auto-install onnxruntime + kagglehub if missing
  2. Auto-download Perch ONNX model + labels CSV via kagglehub
  3. Build embedding cache ONCE: AudioAugmenter → ONNX Perch → (X, S, y) .npz
     X = 1536-d embeddings, S = mapped Perch logit scores (234 species)
  4. Build soundscape validation cache ONCE from train_soundscapes_labels.csv
  5. Agent loop (max_iterations):
     a. Researcher reads memory → produces JSON spec
     b. Coder writes build_head(emb_dim, n_classes) + get_training_config()
     c. Harness loads caches, trains head, blends with Perch logit scores
     d. Evaluate soundscape metrics (macro AP = ranking, macro AUC = diagnostic)
     e. Log to persistent memory

Run:
    python src/perch_agent.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient
from memory import ExperimentMemory
from soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict


# ─────────────────────────────────────────────────────────────────────────────
# Dependency management
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_deps() -> None:
    """Install onnxruntime and kagglehub if not already available."""
    missing = []
    for pkg, imp in [("onnxruntime", "onnxruntime"), ("kagglehub", "kagglehub")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  [Setup] Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("  [Setup] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Model + label downloads
# ─────────────────────────────────────────────────────────────────────────────

def _find_or_download_onnx(dataset_slug: str) -> Path:
    """Download Perch ONNX model via kagglehub; return path to .onnx file."""
    import kagglehub
    print(f"  [Setup] Locating ONNX model ({dataset_slug})...")
    onnx_dir = Path(kagglehub.dataset_download(dataset_slug))
    onnx_files = sorted(onnx_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx file found in {onnx_dir}.\n"
            f"Make sure kagglehub is authenticated: kagglehub.login()"
        )
    print(f"  [Setup] ONNX model: {onnx_files[0]}")
    return onnx_files[0]


def _find_or_download_perch_labels(model_slug: str) -> Path:
    """Download Perch SavedModel (for labels.csv) via kagglehub."""
    import kagglehub
    print(f"  [Setup] Locating Perch labels ({model_slug})...")
    model_dir = Path(kagglehub.model_download(model_slug))
    label_files = sorted(model_dir.rglob("labels.csv"))
    if not label_files:
        raise FileNotFoundError(
            f"No labels.csv found in {model_dir}.\n"
            f"Make sure kagglehub is authenticated: kagglehub.login()"
        )
    print(f"  [Setup] Labels CSV: {label_files[0]}")
    return label_files[0]


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Perch session
# ─────────────────────────────────────────────────────────────────────────────

def _load_onnx_session(onnx_path: Path):
    """Load ONNX session; auto-detect embedding (1536-d) and logit outputs."""
    import onnxruntime as ort
    import numpy as np
    print("  [Perch] Loading ONNX session...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp_name = sess.get_inputs()[0].name

    # Smoke test to identify output indices
    dummy = np.zeros((1, 160_000), dtype=np.float32)
    outs = sess.run(None, {inp_name: dummy})
    emb_idx, logit_idx = None, None
    for i, arr in enumerate(outs):
        if arr.ndim == 2 and arr.shape[-1] == 1536:
            emb_idx = i
        elif arr.ndim == 2 and arr.shape[-1] > 5_000:
            logit_idx = i
    if emb_idx is None:
        raise RuntimeError("Could not find 1536-d embedding output in ONNX model")
    if logit_idx is None:
        raise RuntimeError("Could not find large logits output (>5000 classes) in ONNX model")
    print(f"  [Perch] Loaded. Embedding index={emb_idx}  Logits index={logit_idx}")
    return sess, inp_name, emb_idx, logit_idx


def _perch_embed_batch(sess, inp_name: str, emb_idx: int, logit_idx: int, waveforms):
    """Run a batch of 5-second waveforms through Perch → (embeddings, logits)."""
    import numpy as np
    PERCH_SAMPLES = 160_000
    if isinstance(waveforms, list):
        batch = np.stack(waveforms, axis=0).astype(np.float32)
    else:
        batch = np.asarray(waveforms, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch[None, :]
    if batch.shape[1] != PERCH_SAMPLES:
        fixed = np.zeros((batch.shape[0], PERCH_SAMPLES), dtype=np.float32)
        for i in range(len(batch)):
            n = min(batch.shape[1], PERCH_SAMPLES)
            fixed[i, :n] = batch[i, :n]
        batch = fixed
    outs = sess.run(None, {inp_name: batch})
    return outs[emb_idx].astype(np.float32), outs[logit_idx].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Direct logit mapping: Perch vocab (~14k) → BirdCLEF 234 species
# ─────────────────────────────────────────────────────────────────────────────

def _build_logit_mapping(labels_csv: Path, taxonomy_df, species_cols: list):
    """
    Map Perch's internal species indices to BirdCLEF species positions.
    Unmapped species get a genus-level proxy if one exists.
    Returns: (MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
    """
    import numpy as np
    import pandas as pd

    perch_labels = pd.read_csv(labels_csv).reset_index().rename(columns={"index": "perch_idx"})
    sci_col = next(
        (c for c in perch_labels.columns if any(k in c.lower() for k in ["sci", "name"])),
        perch_labels.columns[1],
    )
    perch_sci_to_idx = {
        str(row[sci_col]).strip(): int(row["perch_idx"])
        for _, row in perch_labels.iterrows()
    }
    NO_LABEL = len(perch_labels)

    tax_sci   = taxonomy_df.set_index("primary_label")["scientific_name"].to_dict()
    tax_class = taxonomy_df.set_index("primary_label")["class_name"].to_dict()

    BC_INDICES = np.array([
        perch_sci_to_idx.get(str(tax_sci.get(sp, "")).strip(), NO_LABEL)
        for sp in species_cols
    ], dtype=np.int32)

    MAPPED_MASK   = BC_INDICES != NO_LABEL
    MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
    UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

    # Genus-level proxy for classes not directly in Perch vocab
    PROXY_TAXA = {"Aves", "Amphibia", "Insecta", "Reptilia"}
    proxy_map: dict[int, list[int]] = {}
    for sp_idx in UNMAPPED_POS:
        sp  = species_cols[int(sp_idx)]
        cls = tax_class.get(sp, "")
        if cls not in PROXY_TAXA:
            continue
        sci = str(tax_sci.get(sp, "")).strip()
        genus = sci.split()[0] if sci else ""
        if not genus:
            continue
        pat = re.compile(rf"^{re.escape(genus)}\s")
        hits = [pidx for psci, pidx in perch_sci_to_idx.items() if pat.match(psci)]
        if hits:
            proxy_map[int(sp_idx)] = hits

    n = len(species_cols)
    print(
        f"  [Mapping] Direct: {MAPPED_MASK.sum()}/{n} | "
        f"Genus proxy: {len(proxy_map)}/{(~MAPPED_MASK).sum()} unmapped"
    )
    return MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL


def _apply_logit_mapping(logits, n_species: int, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL):
    """Convert Perch logits (B, vocab) → BirdCLEF species scores (B, 234)."""
    import numpy as np
    B = logits.shape[0]
    out = np.broadcast_to(
        logits.mean(axis=1, keepdims=True), (B, n_species)
    ).astype(np.float32).copy()
    out[:, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
    for sp_idx, bc_idxs in proxy_map.items():
        out[:, sp_idx] = logits[:, bc_idxs].mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_pad(audio_path: Path, sr: int = 32_000, clip_sec: float = 5.0):
    import numpy as np
    import librosa
    target = int(sr * clip_sec)
    wav, _ = librosa.load(str(audio_path), sr=sr, mono=True, duration=clip_sec)
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)))
    else:
        wav = wav[:target]
    return wav.astype(np.float32)


def _enforce_length(wav, target: int):
    import numpy as np
    if len(wav) < target:
        return np.pad(wav, (0, target - len(wav)))
    return wav[:target]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache — built once, reused across all agent iterations
# ─────────────────────────────────────────────────────────────────────────────

def _embedding_cache_meta_path(npz_path: Path) -> Path:
    return npz_path.parent / f"{npz_path.stem}.meta.json"


def _write_embedding_cache_manifest(npz_path: Path, meta: dict) -> None:
    _embedding_cache_meta_path(npz_path).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _read_embedding_cache_manifest(npz_path: Path) -> dict | None:
    p = _embedding_cache_meta_path(npz_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _select_train_df_stratified(
    train_df,
    species_to_idx: dict,
    audio_dir: Path,
    sample_frac: float = 1.0,
    max_samples: int | None = None,
    random_state: int = 42,
):
    """Per-species stratified subsample (same policy as BirdNET build_train_cache)."""
    import pandas as pd

    lcol = "primary_label" if "primary_label" in train_df.columns else "species_code"
    fcol = "filename" if "filename" in train_df.columns else "filepath"
    df = train_df[train_df[lcol].isin(species_to_idx)].copy()
    df = df[df[fcol].apply(lambda f: (audio_dir / str(f)).is_file())]
    if sample_frac < 1.0:
        chunks = [
            g.sample(frac=sample_frac, random_state=random_state)
            for _, g in df.groupby(lcol)
        ]
        df = pd.concat(chunks).reset_index(drop=True)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    return df, lcol, fcol


def _subsample_cached_train(
    X,
    S,
    y,
    max_samples: int | None,
    random_state: int = 42,
):
    """Stratified subsample of cached rows (for fast head training); full cache unchanged on disk."""
    import numpy as np

    n = int(X.shape[0])
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return X, S, y

    rng = np.random.default_rng(random_state)
    labels = np.argmax(y, axis=1)
    idx_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        n_take = max(1, int(round(max_samples * len(cls_idx) / n)))
        n_take = min(n_take, len(cls_idx))
        idx_parts.append(rng.choice(cls_idx, size=n_take, replace=False))
    idx = np.concatenate(idx_parts)
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    elif len(idx) < max_samples:
        pool = np.setdiff1d(np.arange(n), idx, assume_unique=True)
        need = min(max_samples - len(idx), len(pool))
        if need:
            idx = np.concatenate([idx, rng.choice(pool, size=need, replace=False)])
    idx = np.sort(idx)
    return X[idx], S[idx], y[idx]


def _build_train_cache(
    sess, inp_name, emb_idx, logit_idx,
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
    train_df, species_to_idx: dict, audio_dir: Path, n_species: int,
    cache_path: Path,
    sample_frac: float = 1.0,
    max_samples: int | None = None,
    batch_size: int = 16,
    sr: int = 32_000, clip_sec: float = 5.0,
    aug_config: dict | None = None,
    soundscapes_dir: Path | None = None,
    aug_preset: str | None = None,
) -> None:
    """Embed training clips with AudioAugmenter (+ optional SNR mix) → save (X, S, y).npz."""
    import numpy as np
    from augmentation import AudioAugmenter, load_random_soundscape_noise, mix_snr

    aug_config = aug_config or {}
    audio_cfg = aug_config.get("audio", {})
    use_snr = bool(aug_config.get("use_snr_mixing", False))
    mix_prob = float(aug_config.get("mix_prob", 0.35))
    snr_min = float(aug_config.get("snr_min_db", 0.0))
    snr_max = float(aug_config.get("snr_max_db", 15.0))
    augmenter = AudioAugmenter(audio_cfg)
    rng = np.random.default_rng(42)
    noise_pool: list[Path] = []
    if use_snr and soundscapes_dir and soundscapes_dir.exists():
        noise_pool = sorted(soundscapes_dir.glob("*.ogg"))
        if not noise_pool:
            print("  [Cache] Warning: no soundscapes for SNR mixing — SNR disabled.")
            use_snr = False

    train_df, lcol, fcol = _select_train_df_stratified(
        train_df, species_to_idx, audio_dir,
        sample_frac=sample_frac, max_samples=max_samples, random_state=42,
    )
    rows = [(lb, fn) for lb, fn in train_df[[lcol, fcol]].dropna().values.tolist()]

    frac_pct = int(sample_frac * 100) if sample_frac < 1.0 else 100
    cap_note = f", cap={max_samples}" if max_samples else ""
    print(
        f"  [Cache] Building train cache: {len(rows)} clips "
        f"({frac_pct}% stratified per species{cap_note}) → {cache_path.name}"
    )
    target = int(sr * clip_sec)
    X_parts, S_parts, y_parts = [], [], []
    batch_wavs, batch_labels = [], []

    def _flush():
        if not batch_wavs:
            return
        embs, logits = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, batch_wavs)
        scores = _apply_logit_mapping(logits, n_species, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
        X_parts.append(embs); S_parts.append(scores)
        y_parts.extend(batch_labels)

    for i, (label, fname) in enumerate(rows):
        p = audio_dir / fname
        if not p.exists():
            continue
        try:
            wav = _load_and_pad(p, sr, clip_sec)
            wav = augmenter.apply(wav, sr)
            if use_snr and noise_pool and rng.random() < mix_prob:
                noise = load_random_soundscape_noise(
                    rng, noise_pool, sr=sr, clip_sec=clip_sec
                )
                if noise is not None:
                    snr_db = float(rng.uniform(snr_min, snr_max))
                    wav = mix_snr(wav, noise, snr_db)
            wav = _enforce_length(wav, target)
        except Exception:
            continue

        vec = np.zeros(n_species, dtype=np.float32)
        vec[species_to_idx[label]] = 1.0
        batch_wavs.append(wav)
        batch_labels.append(vec)

        if len(batch_wavs) >= batch_size:
            _flush()
            batch_wavs.clear()
            batch_labels.clear()

        if (i + 1) % 1000 == 0 or (i + 1) == len(rows):
            print(f"    {i+1}/{len(rows)} files processed...", flush=True)

    _flush()
    if not X_parts:
        raise RuntimeError("No samples could be embedded for training cache.")

    X = np.concatenate(X_parts).astype(np.float32)
    S = np.concatenate(S_parts).astype(np.float32)
    y = np.stack(y_parts).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, S=S, y=y)
    _write_embedding_cache_manifest(cache_path, {
        "kind": "perch_train_embeddings",
        "aug_preset": aug_preset,
        "sample_frac": sample_frac,
        "max_samples": max_samples,
        "n_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "n_species": int(y.shape[1]),
    })
    print(f"  [Cache] Saved: X={X.shape}  S={S.shape}  y={y.shape}")


def _build_val_cache(
    sess, inp_name, emb_idx, logit_idx,
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
    soundscapes_dir: Path, labels_csv: Path,
    species_to_idx: dict, n_species: int,
    cache_path: Path, batch_size: int = 16,
    sr: int = 32_000, clip_sec: float = 5.0,
) -> bool:
    """Embed soundscape validation windows → save (X, S, y).npz. Returns True on success."""
    import numpy as np
    import pandas as pd
    import librosa

    if not labels_csv.exists():
        return False

    lab = pd.read_csv(labels_csv)

    def _tok(v):
        if pd.isna(v) or str(v).strip() == "":
            return set()
        return {t.strip() for t in str(v).split(";") if t.strip()}

    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set.union(*[_tok(v) for v in s]))
        .reset_index()
    )
    grp["start_sec"] = pd.to_timedelta(grp["start"]).dt.total_seconds().astype(float)
    grp["end_sec"]   = pd.to_timedelta(grp["end"]).dt.total_seconds().astype(int)

    target = int(sr * clip_sec)
    wavs, labels = [], []

    for _, row in grp.iterrows():
        fp = soundscapes_dir / row["filename"]
        if not fp.exists():
            continue
        try:
            offset = float(row["start_sec"])
            wav, _ = librosa.load(str(fp), sr=sr, mono=True, offset=offset, duration=clip_sec)
            wav = _enforce_length(wav.astype(np.float32), target)
        except Exception:
            continue
        vec = np.zeros(n_species, dtype=np.float32)
        for code in row["primary_label"]:
            j = species_to_idx.get(str(code))
            if j is not None:
                vec[j] = 1.0
        wavs.append(wav)
        labels.append(vec)

    if not wavs:
        return False

    print(f"  [Cache] Building val cache: {len(wavs)} soundscape windows → {cache_path.name}")
    X_parts, S_parts = [], []
    for start in range(0, len(wavs), batch_size):
        chunk = wavs[start:start + batch_size]
        embs, logits = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, chunk)
        scores = _apply_logit_mapping(logits, n_species, MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL)
        X_parts.append(embs); S_parts.append(scores)

    X = np.concatenate(X_parts).astype(np.float32)
    S = np.concatenate(S_parts).astype(np.float32)
    y = np.stack(labels).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, S=S, y=y)
    print(f"  [Cache] Saved val: X={X.shape}  S={S.shape}  y={y.shape}")
    return True


def _build_focal_val_fallback(train_cache: Path, val_cache: Path) -> None:
    """Fallback: use last 10% of training cache as validation set."""
    import numpy as np
    d = np.load(str(train_cache))
    X, S, y = d["X"], d["S"], d["y"]
    n_val = max(50, int(len(X) * 0.1))
    np.savez_compressed(str(val_cache), X=X[-n_val:], S=S[-n_val:], y=y[-n_val:])
    print(f"  [Cache] Fallback val: {n_val} samples from end of train cache")


# ─────────────────────────────────────────────────────────────────────────────
# Perch Researcher — outer loop, reads history, decides head config
# ─────────────────────────────────────────────────────────────────────────────

def _extract_first_json_object(text: str) -> dict | None:
    """Find the first complete JSON object in text by counting braces."""
    start = text.find('{')
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PERCH_SEARCH_SPACE = {
    "arch_types": [
        "residual_mlp",       # dense residual blocks with skip connections (baseline)
        "attention_mlp",      # self-attention / multi-head attention on projected features
        "gated_mlp",          # GLU-style gating: value * sigmoid(gate), two separate Dense layers
        "highway_network",    # highway gates: out = H*transform + (1-H)*input carry
        "bottleneck_mlp",     # wide → narrow → wide projection bottleneck
        "multi_scale_mlp",    # parallel branches at different widths, merged by concat or add
        "transformer_block",  # one or two transformer encoder blocks (MHA + FFN + LayerNorm)
        "mixture_of_experts", # K parallel expert MLPs with soft gating router
        "dense_connections",  # DenseNet-style: each layer receives concat of all prior outputs
        "linear_probe",       # single Dense layer — minimal baseline
    ],
    "hidden_dim":    [256, 512, 1024, 2048],
    "proj_dim":      [128, 256, 512],
    "n_layers":      [1, 2, 3, 4],
    "dropout":       [0.1, 0.2, 0.3, 0.4, 0.5],
    "activation":    ["gelu", "relu", "swish"],
    "normalization": ["layer_norm", "batch_norm", "none"],
    "learning_rate": [1e-2, 5e-3, 1e-3, 8e-4, 5e-4, 1e-4],
    "batch_size":    [64, 128, 256, 512],
    "optimizer":     ["adam", "adamw", "sgd_momentum"],
    "epochs":        [15, 25, 40, 60],
    "patience":      [3, 5, 7, 10],
    "perch_weight":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

PERCH_RESEARCHER_SYSTEM_PROMPT = """You are an expert ML researcher optimizing a BirdCLEF classification head on top of frozen Google Perch 1536-d embeddings.
The Perch ONNX backbone is completely FIXED — you control ONLY the classification head architecture and training config.
Final predictions blend head output with Perch's own logit scores:
  y_pred = perch_weight * perch_scores + (1 - perch_weight) * head_output

YOUR PRIMARY OBJECTIVE: Explore STRUCTURALLY DIVERSE head architectures. Simply varying n_blocks or dropout of the same residual MLP topology is NOT sufficient exploration. Each run should test a meaningfully different inductive bias or architectural design pattern.

Architecture types you can propose (pick from arch_types in the search space):
- residual_mlp:       Dense residual blocks with skip connections (BN→Dense→LN→blocks→proj→sigmoid). This is the baseline — avoid repeating it unless refining a genuinely strong result.
- attention_mlp:      Multi-head self-attention on projected features. Project input to hidden_dim, apply MHA, then FFN. Good for capturing feature interactions.
- gated_mlp:          GLU-style gating: two parallel Dense(hidden_dim) — one linear (value), one sigmoid (gate) — multiplied element-wise, then residual add. Selects which embedding dims are informative.
- highway_network:    Transform gate H (sigmoid) and carry gate (1-H): out = H*Dense(x) + (1-H)*x. Learnable depth blending.
- bottleneck_mlp:     Project wide→narrow→wide to force information compression: Dense(2048)→Dense(128)→Dense(2048). Forces the head to learn a compact representation.
- multi_scale_mlp:    Parallel branches at different widths (e.g. 256, 512, 1024) processing the same input, outputs merged by concatenation then projected.
- transformer_block:  Reshape to (batch,1,emb_dim), apply MultiHeadAttention + FFN + LayerNorm (1-2 blocks). Classic transformer encoder.
- mixture_of_experts: K parallel expert Dense layers with a soft gating router (Dense(K, softmax)), weighted sum of expert outputs.
- dense_connections:  DenseNet-style: each layer receives concat of all prior layer outputs. Helps gradient flow and feature reuse.
- linear_probe:       Single Dense(n_classes, sigmoid). Strong sanity-check baseline for raw embedding quality.

Reasoning guidelines:
- Look at the history above and identify which arch_types have already been tried.
- Strongly prefer an arch_type NOT yet explored (strategy: "explore").
- Only revisit a past arch_type if it was the clear best and you are refining it (strategy: "exploit").
- Consider what inductive biases suit 1536-d Perch embeddings, 234 species, severe class imbalance, and soundscape audio.
- Tune learning_rate, batch_size, optimizer, epochs, patience, and perch_weight to match the architecture's typical training dynamics.

You MUST respond with ONLY a single JSON object — no prose, no explanation, no markdown, no code fences.
Start your response with { and end with }.

The arch_description field must be a precise, implementable description of the head: layer types, sizes, activations, normalization, and how layers connect. The coder will implement it verbatim in TF/Keras.

Example response:
{"arch_type": "gated_mlp", "arch_description": "LayerNorm on input. Dense(1024) projection. Then 2 GLU blocks: each block has two parallel Dense(1024) — one with linear activation (value) and one with sigmoid activation (gate) — multiplied element-wise, then added to a residual Dense(1024) projection of the block input, followed by LayerNorm. Final Dense(512, gelu), Dropout(0.3), Dense(n_classes, sigmoid).", "hidden_dim": 1024, "n_layers": 2, "dropout": 0.3, "activation": "gelu", "normalization": "layer_norm", "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 30, "patience": 5, "perch_weight": 0.2, "reasoning": "GLU gating not yet tried; may help the head learn to selectively weight Perch embedding dimensions.", "hypothesis": "Gated selection of Perch embedding features should outperform uniform residual blending for multi-label species classification.", "strategy": "explore"}

Required keys: arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."""


class PerchResearcher:
    def __init__(self, llm: LLMClient, memory: ExperimentMemory, temperature: float = 0.6) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature

    def next_experiment(self) -> dict:
        history  = self.memory.researcher_context()
        best     = self.memory.best_runs(1)
        total    = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"

        user_prompt = (
            f"{history}\n\n"
            f"Available search space:\n{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
            f"Total experiments so far: {total}\n"
            f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
            "IMPORTANT: Look at the 'arch_type' field in each past experiment above. "
            "Identify which arch_types have already been tried and choose one that has NOT been explored yet. "
            "Architectural diversity is the top priority.\n\n"
            "Pick the next experiment. Respond with ONLY a JSON object — no prose, no markdown.\n"
            "Start with { and end with }. Include all required keys: "
            "arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, "
            "learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."
        )

        print(f"\n  [Researcher] Analyzing {total} experiments, best {best_str}...")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": PERCH_RESEARCHER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        spec = self._parse_spec(response)
        print(
            f"  [Researcher] Strategy: {spec.get('strategy', '?')} | "
            f"arch={spec.get('arch_type')} hidden={spec.get('hidden_dim')} "
            f"lr={spec.get('learning_rate')} perch_w={spec.get('perch_weight')}"
        )
        print(f"  [Researcher] Reasoning: {spec.get('reasoning', '')[:120]}")
        return spec

    def _parse_spec(self, response: str) -> dict:
        # Strip deepseek-r1 thinking tokens before searching for JSON
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Try ```json ... ``` block
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if m:
            try:
                return _perch_fill_defaults(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

        # Walk character by character to find the first complete JSON object
        spec = _extract_first_json_object(cleaned)
        if spec is not None:
            return _perch_fill_defaults(spec)

        print("  [Researcher] Warning: could not parse JSON, using safe defaults.")
        print(f"  [Researcher] Raw response (first 400 chars): {repr(cleaned[:400])}")
        return _perch_safe_defaults()


def _perch_safe_defaults() -> dict:
    return {
        "arch_type":        "residual_mlp",
        "arch_description": (
            "BatchNorm on input. Dense(1024) projection with LayerNorm. "
            "Then 2 residual blocks: each block applies Dense(1024), LayerNorm, GELU, "
            "Dropout(0.3), Dense(1024), then adds the block input (skip connection), "
            "followed by LayerNorm. Final Dense(512, gelu), Dropout(0.4), Dense(n_classes, sigmoid)."
        ),
        "hidden_dim":       1024,
        "n_layers":         2,
        "dropout":          0.3,
        "activation":       "gelu",
        "normalization":    "layer_norm",
        "learning_rate":    8e-4,
        "batch_size":       256,
        "optimizer":        "adam",
        "epochs":           25,
        "patience":         5,
        "perch_weight":     0.2,
        "reasoning":        "Fallback defaults — researcher output could not be parsed.",
        "hypothesis":       "Baseline residual head config.",
        "strategy":         "explore",
    }


def _perch_fill_defaults(spec: dict) -> dict:
    defaults = _perch_safe_defaults()
    for k, v in defaults.items():
        spec.setdefault(k, v)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Coder — inner loop, writes TF/Keras head given spec
# ─────────────────────────────────────────────────────────────────────────────

PERCH_CODER_SYSTEM_PROMPT = """You are a Python ML engineer writing TF/Keras classification head code.
You receive an architecture specification and must return ONLY a Python code block containing exactly two functions:

1. build_head(emb_dim, num_classes) -> tf.keras.Model
   - Input shape: (emb_dim,) — Perch 1536-d embeddings, already available as tf
   - Output shape: (num_classes,) with sigmoid activation
   - Implement EXACTLY the architecture described in arch_description
   - Use tf.keras layers: Dense, LayerNormalization, BatchNormalization, Dropout, Add,
     Multiply, Activation, MultiHeadAttention, Reshape, Concatenate, etc.
   - Use the Keras functional API: define inp = tf.keras.layers.Input(shape=(emb_dim,)),
     build the graph, then return tf.keras.Model(inp, out)
   - For attention: reshape to (batch, 1, emb_dim) with tf.keras.layers.Reshape((1, emb_dim))
     before MultiHeadAttention, then flatten back with tf.keras.layers.Reshape((emb_dim,)) or Flatten
   - For gating: use two separate Dense layers (value and gate), not tensor slicing
   - The last layer must be Dense(num_classes, activation="sigmoid")

2. get_training_config() -> dict
   - Returns all training hyperparameters as a plain Python dict
   - Must include: learning_rate, batch_size, optimizer, epochs, patience, perch_weight
   - Do NOT import anything inside get_training_config()

Rules:
- tf is already imported — do NOT write import tensorflow or import tf
- No top-level executable statements
- No main() or other functions, no class definitions
- Both functions must be present in the same code block

Project-specific implementation tips:
- This is MULTI-LABEL classification (234 bird species): output MUST use sigmoid, never softmax — each species is predicted independently
- Class imbalance (~200:1 neg/pos) is handled externally by weighted BCE — do NOT add any extra loss terms or temperature scaling inside build_head
- Perch embeddings are already strong 1536-d features — prefer LayerNormalization over BatchNormalization on internal layers (BN behaves differently between train/eval mode on imbalanced class distributions)
- For ATTENTION architectures: MultiHeadAttention needs 3D input (batch, seq, dim) — use tf.keras.layers.Reshape((1, emb_dim)) before attention, then tf.keras.layers.Flatten() or tf.keras.layers.Reshape((hidden_dim,)) after
- For GATING: always use two separate Dense layers (one for value, one for gate) — never slice tensors with Python indexing like x[:, :512] as it breaks the Keras functional API
- For MIXTURE OF EXPERTS: use soft routing with Dense(n_experts, activation='softmax') — hard argmax is not differentiable and will break training
- For DENSE CONNECTIONS: use tf.keras.layers.Concatenate()([a, b]) — not tf.concat
- Avoid Lambda layers — they cause issues with model saving/loading. Use built-in Keras layers instead
- build_head MUST return a tf.keras.Model(inp, out), not a layer or tensor

*** CRITICAL — Add() SHAPE RULE (most common failure) ***
Add() requires BOTH inputs to have the EXACT SAME shape.
For any residual or skip connection, you MUST project the skip tensor to match the block output:
    skip = tf.keras.layers.Dense(block_output_dim)(skip_input)   # ensure shapes match
    x = tf.keras.layers.Add()([x, skip])
NEVER do: Add()([tensor_of_1024, tensor_of_512]) — this WILL crash.
In gated blocks: the gated output and the residual path must both be projected to hidden_dim BEFORE Add().
In the example below, note how BOTH the gated output AND the residual input go through Dense(1024) so their shapes match.

Example for a gated MLP:
```python
def build_head(emb_dim, num_classes):
    hidden_dim = 1024  # single constant so shapes always match
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x = tf.keras.layers.LayerNormalization()(inp)
    x = tf.keras.layers.Dense(hidden_dim)(x)          # project to hidden_dim
    x = tf.keras.layers.LayerNormalization()(x)
    for _ in range(2):
        v = tf.keras.layers.Dense(hidden_dim, activation="linear")(x)   # value: hidden_dim
        g = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")(x)  # gate:  hidden_dim
        gated = tf.keras.layers.Multiply()([v, g])                       # hidden_dim
        gated = tf.keras.layers.Dense(hidden_dim)(gated)                 # MUST match x (hidden_dim)
        x = tf.keras.layers.Add()([x, gated])         # OK: both hidden_dim
        x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 0.001,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 30,
        "patience": 5,
        "perch_weight": 0.2,
    }
```"""


def _spec_to_coder_prompt(spec: dict) -> str:
    arch_type = spec.get("arch_type", "residual_mlp")
    arch_desc = spec.get("arch_description", "Standard residual MLP with skip connections.")
    training_keys = ("learning_rate", "batch_size", "optimizer", "epochs", "patience", "perch_weight")
    training_cfg = {k: spec[k] for k in training_keys if k in spec}
    return (
        f"Architecture type: {arch_type}\n"
        f"Architecture description: {arch_desc}\n\n"
        f"Training config to use in get_training_config():\n{json.dumps(training_cfg, indent=2)}\n\n"
        f"Write both functions:\n"
        f"1. build_head(emb_dim, num_classes) — implement the architecture described above using TF/Keras functional API\n"
        f"2. get_training_config() — return the training config dict above\n\n"
        f"Return ONLY both functions in a single ```python``` code block. Nothing else."
    )


def _extract_code(response: str) -> str | None:
    match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        first = candidate.splitlines()[0].strip().lower() if candidate else ""
        if first in ("python", "py", ""):
            candidate = "\n".join(candidate.splitlines()[1:]).strip() if first else candidate
        return candidate or None
    return None


def _validate_perch_code(code: str) -> list[str]:
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "build_head" not in names:
        issues.append("Missing: build_head(emb_dim, num_classes)")
    if "get_training_config" not in names:
        issues.append("Missing: get_training_config()")
    return issues


# Known-good fallback: used when the coder exhausts all retries.
# Implements the safe-default residual MLP so no iteration is ever wasted.
# Simplest possible head: single linear layer (pure linear probe on Perch embeddings).
# Used as (a) mandatory iteration-0 baseline and (b) fallback if the coder fails all retries.
_SAFE_DEFAULT_SLOT_CODE = '''
def build_head(emb_dim, num_classes):
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(inp)
    return tf.keras.Model(inp, out)

def get_training_config():
    return {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "optimizer": "adam",
        "epochs": 20,
        "patience": 5,
        "perch_weight": 0.2,
    }
'''.strip()

_BASELINE_SPEC = {
    "arch_type":        "linear_probe",
    "arch_description": "Single Dense(num_classes, sigmoid) — pure linear probe on raw 1536-d Perch embeddings. No hidden layers.",
    "hidden_dim":       0,
    "n_layers":         0,
    "dropout":          0.0,
    "activation":       "sigmoid",
    "normalization":    "none",
    "learning_rate":    1e-3,
    "batch_size":       256,
    "optimizer":        "adam",
    "epochs":           20,
    "patience":         5,
    "perch_weight":     0.2,
    "reasoning":        "Mandatory baseline — establishes the linear separability ceiling of raw Perch embeddings.",
    "hypothesis":       "Linear probe sets the floor; any non-linear head should beat this.",
    "strategy":         "baseline",
}


def generate_perch_code(
    coder_llm: LLMClient, spec: dict, temperature: float, max_retries: int = 5
) -> str | None:
    prompt = _spec_to_coder_prompt(spec)
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        print(f"  [Coder] Attempt {attempt}/{max_retries}...")
        response = coder_llm.generate_from_messages(
            messages=[
                {"role": "system", "content": PERCH_CODER_SYSTEM_PROMPT},
                {"role": "user",   "content": current_prompt},
            ],
            temperature=temperature,
        )

        if response.startswith("Error communicating"):
            print(f"  [Coder] LLM error: {response[:150]}")
            break

        code = _extract_code(response)
        if not code:
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        issues = _validate_perch_code(code) if code else ["No code found in response."]
        if not issues:
            print("  [Coder] Code valid.")
            return code

        print(f"  [Coder] Issues: {issues}")
        missing_build_head = any("build_head" in i for i in issues)
        build_head_hint = (
            "\n\nCRITICAL: build_head(emb_dim, num_classes) is MISSING from your output. "
            "You MUST define it. It must accept (emb_dim: int, num_classes: int) and return "
            "a tf.keras.Model. Minimal valid example:\n"
            "def build_head(emb_dim, num_classes):\n"
            "    inp = tf.keras.layers.Input(shape=(emb_dim,))\n"
            "    x = tf.keras.layers.Dense(512, activation='gelu')(inp)\n"
            "    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)\n"
            "    return tf.keras.Model(inp, out)\n"
            "Now implement the full architecture from the spec AND include get_training_config()."
        ) if missing_build_head else ""
        current_prompt = (
            "Your code had issues:\n" + "\n".join(f"- {i}" for i in issues) +
            build_head_hint +
            f"\n\nOriginal spec:\n{_spec_to_coder_prompt(spec)}\n\n"
            "Fix all issues and return BOTH functions in a single ```python``` code block."
        )

    return None


def _repair_perch_code(
    coder_llm: LLMClient, spec: dict, previous_code: str, error: str, temperature: float
) -> str | None:
    """Feed a runtime error back to the coder and ask for a fix."""
    error_tail = error[-1500:] if len(error) > 1500 else error

    # Detect the specific shape-mismatch pattern and inject a targeted hint
    shape_hint = ""
    if "incompatible shapes" in error.lower() or "elemwise_op_output_shape" in error:
        shape_hint = (
            "\n*** SHAPE MISMATCH DETECTED — THIS IS THE SPECIFIC BUG IN YOUR CODE ***\n"
            "Add() crashed because two tensors have different sizes (e.g. (1024,) vs (512,)).\n"
            "THE FIX: Before every Add(), project BOTH tensors to the same dimension with Dense():\n"
            "    skip = tf.keras.layers.Dense(hidden_dim)(skip_input)  # match block output dim\n"
            "    block_out = tf.keras.layers.Dense(hidden_dim)(block_out)\n"
            "    x = tf.keras.layers.Add()([skip, block_out])  # now both hidden_dim — OK\n"
            "Use ONE consistent hidden_dim variable throughout the entire build_head function.\n"
            "The final Dense(512, gelu) projection comes AFTER all residual blocks — never inside them.\n"
        )

    prompt = (
        f"Your previously generated code failed at runtime with this error:\n"
        f"```\n{error_tail}\n```\n"
        f"{shape_hint}\n"
        f"Architecture spec:\n{_spec_to_coder_prompt(spec)}\n\n"
        f"Your previous code:\n```python\n{previous_code}\n```\n\n"
        "Fix the runtime error. General causes to check:\n"
        "- Add() shape mismatch: both inputs must have the EXACT same shape — project with Dense() if needed\n"
        "- MultiHeadAttention requires 3D input: reshape (batch, emb_dim) → (batch, 1, emb_dim) first\n"
        "- Concatenate needs tensors with matching non-concatenation dimensions\n"
        "- Tensor slicing (x[:, :512]) breaks the functional API — use separate Dense layers\n"
        "- Lambda layers may fail on save — replace with explicit Keras layers\n"
        "- build_head must return tf.keras.Model(inp, out), not a layer or tensor\n\n"
        "Return ONLY the corrected code in a ```python``` code block. "
        "Both build_head(emb_dim, num_classes) and get_training_config() must be present."
    )
    response = coder_llm.generate_from_messages(
        messages=[
            {"role": "system", "content": PERCH_CODER_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
    )
    if response.startswith("Error communicating"):
        print(f"  [Coder] LLM error during repair: {response[:150]}")
        return None
    code = _extract_code(response)
    if not code:
        lines = response.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    issues = _validate_perch_code(code) if code else ["No code found in repair response."]
    if issues:
        print(f"  [Coder] Repair still has issues: {issues}")
        return None
    print("  [Coder] Repair code validated.")
    return code


# ─────────────────────────────────────────────────────────────────────────────
# Script harness — wraps Coder's slot code into a runnable training script
# ─────────────────────────────────────────────────────────────────────────────

def _harness_subsample_block(head_train_max_samples: int | None) -> str:
    if not head_train_max_samples or head_train_max_samples <= 0:
        return ""
    cap = int(head_train_max_samples)
    return f"""
_HEAD_TRAIN_CAP = {cap}

def _subsample_train_stratified(X, S, y, cap, seed=42):
    import numpy as _np
    n = int(X.shape[0])
    if cap is None or cap <= 0 or n <= cap:
        return X, S, y
    rng = _np.random.default_rng(seed)
    labels = _np.argmax(y, axis=1)
    parts = []
    for cls in _np.unique(labels):
        cls_idx = _np.flatnonzero(labels == cls)
        n_take = max(1, int(round(cap * len(cls_idx) / n)))
        n_take = min(n_take, len(cls_idx))
        parts.append(rng.choice(cls_idx, size=n_take, replace=False))
    idx = _np.concatenate(parts)
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    elif len(idx) < cap:
        pool = _np.setdiff1d(_np.arange(n), idx, assume_unique=True)
        need = min(cap - len(idx), len(pool))
        if need:
            idx = _np.concatenate([idx, rng.choice(pool, size=need, replace=False)])
    idx = _np.sort(idx)
    return X[idx], S[idx], y[idx]

_CACHE_N_FULL = len(X_train)
X_train, S_train, y_train = _subsample_train_stratified(
    X_train, S_train, y_train, _HEAD_TRAIN_CAP, seed=42
)
print(f"  Head training subset: {{len(X_train)}} / {{_CACHE_N_FULL}} cached embeddings")
"""


def _build_harness_prefix(
    train_cache: Path,
    val_cache: Path,
    head_train_max_samples: int | None = None,
) -> str:
    sub = _harness_subsample_block(head_train_max_samples)
    return f'''
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

_PROJECT_ROOT = None
for _cand in Path(__file__).resolve().parents:
    if (_cand / "src").exists() and (_cand / "configs").exists():
        _PROJECT_ROOT = _cand
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

def _load_cache(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    return d["X"].astype(np.float32), d["S"].astype(np.float32), d["y"].astype(np.float32)

_train_cache = Path(r"{train_cache}")
_val_cache   = Path(r"{val_cache}")
X_train, S_train, y_train = _load_cache(_train_cache)
X_val,   S_val,   y_val   = _load_cache(_val_cache)
{sub}
EMB_DIM   = X_train.shape[1]
N_CLASSES = y_train.shape[1]

print(f"  Loaded train cache: {{_train_cache.name}}")
print(f"  Training head on:   X={{X_train.shape}}  y={{y_train.shape}}")
print(f"  Soundscape val:     X={{X_val.shape}}    y={{y_val.shape}}")
'''.strip()


HARNESS_SUFFIX = r"""
# build_head(emb_dim, num_classes) and get_training_config() are defined in the slot code above.


def main():
    tf.keras.utils.set_random_seed(42)
    cfg = get_training_config()

    lr           = float(cfg.get("learning_rate", 8e-4))
    batch_size   = int(cfg.get("batch_size", 256))
    epochs       = int(cfg.get("epochs", 50))
    patience     = int(cfg.get("patience", 7))
    perch_weight = float(cfg.get("perch_weight", 0.2))
    val_split    = float(cfg.get("val_split", 0.1))
    opt_name     = str(cfg.get("optimizer", "adam"))

    # Train/val split on cached embeddings
    n_val   = max(1, int(len(X_train) * val_split))
    perm    = np.random.default_rng(42).permutation(len(X_train))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
    X_vl, y_vl = X_train[val_idx], y_train[val_idx]

    # Positive class weighting — handles severe species imbalance (200:1 neg/pos ratio)
    pos = y_tr.sum(axis=0).astype(np.float64)
    neg = len(y_tr) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
    pw = tf.constant(pos_weight)[tf.newaxis, :]

    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        return tf.reduce_mean(
            pw * y_true * (-tf.math.log(y_pred))
            + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
        )

    # Build head (Coder-generated architecture)
    head = build_head(EMB_DIM, N_CLASSES)

    # Compile
    if opt_name == "sgd_momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif opt_name == "adamw":
        try:
            opt = tf.keras.optimizers.AdamW(lr)
        except AttributeError:
            opt = tf.keras.optimizers.Adam(lr)
    else:
        opt = tf.keras.optimizers.Adam(lr)
    head.compile(optimizer=opt, loss=weighted_bce)

    # Train with early stopping and LR reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True, monitor="val_loss", verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(2, patience // 2), min_lr=1e-6, verbose=0
        ),
    ]
    head.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict on soundscape validation
    head_probs  = head.predict(X_val, batch_size=batch_size, verbose=0)
    perch_probs = 1.0 / (1.0 + np.exp(-S_val))   # sigmoid of Perch logit scores

    # Blend: perch_weight controls how much to trust Perch's own classifier
    y_pred = perch_weight * perch_probs + (1.0 - perch_weight) * head_probs

    # Save trained head so the main loop can promote it if it's the new best
    head.save(str(Path(tempfile.gettempdir()) / "_trained_head.keras"))
    # Also save weights-only file (Keras-version-agnostic, used by Kaggle notebook)
    head.save_weights(str(Path(tempfile.gettempdir()) / "_trained_head.weights.h5"))

    # Save artifacts for the evaluator
    _tmp = Path(tempfile.gettempdir())
    np.save(str(_tmp / "_y_true.npy"), y_val)
    np.save(str(_tmp / "_y_pred.npy"), y_pred)
    print("EVAL_ARTIFACTS_SAVED")


if __name__ == "__main__":
    main()
""".strip()


def _build_script(
    slot_code: str,
    train_cache: Path,
    val_cache: Path,
    head_train_max_samples: int | None = None,
) -> str:
    prefix = _build_harness_prefix(train_cache, val_cache, head_train_max_samples)
    return prefix + "\n\n" + slot_code + "\n\n" + HARNESS_SUFFIX


# ─────────────────────────────────────────────────────────────────────────────
# Final retrain on full data (train + val combined)
# ─────────────────────────────────────────────────────────────────────────────

def _build_final_retrain_script(best_code: str, mem_dir: Path, cache_dir: Path) -> str:
    """Build final retrain script using the best iteration's coder-generated build_head + get_training_config."""
    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_CACHE_DIR = Path(r"{cache_dir}")
_MEM_DIR   = Path(r"{mem_dir}")

def _load(p):
    d = np.load(str(p), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)

X_tr, y_tr = _load(_CACHE_DIR / "train_emb.npz")
X_vl, y_vl = _load(_CACHE_DIR / "val_emb.npz")

X_full = np.concatenate([X_tr, X_vl], axis=0)
y_full = np.concatenate([y_tr, y_vl], axis=0)

EMB_DIM   = X_full.shape[1]
N_CLASSES = y_full.shape[1]
print(f"  Final retrain: X={{X_full.shape}}  y={{y_full.shape}}")

# --- Best iteration's build_head + get_training_config (coder-generated) ---
{best_code}
# ---------------------------------------------------------------------------

cfg        = get_training_config()
lr         = float(cfg.get("learning_rate", 8e-4))
batch_size = int(cfg.get("batch_size",    256))
epochs     = int(cfg.get("epochs",         50))
opt_name   = str(cfg.get("optimizer",   "adam"))

head = build_head(EMB_DIM, N_CLASSES)

pos = y_full.sum(axis=0).astype(np.float64)
neg = len(y_full) - pos
pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
pw = tf.constant(pos_weight)[tf.newaxis, :]

def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(
        pw * y_true * (-tf.math.log(y_pred))
        + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
    )

if opt_name == "sgd_momentum":
    opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
elif opt_name == "adamw":
    try:
        opt = tf.keras.optimizers.AdamW(lr)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(lr)
else:
    opt = tf.keras.optimizers.Adam(lr)
head.compile(optimizer=opt, loss=weighted_bce)

tf.keras.utils.set_random_seed(42)
head.fit(X_full, y_full, epochs=epochs, batch_size=batch_size, verbose=1)

head.save(str(_MEM_DIR / "final_head.keras"))
head.save_weights(str(_MEM_DIR / "final_head.weights.h5"))
print("FINAL_RETRAIN_DONE")
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def _ranking_metric_from_config(config: dict) -> str:
    return str(config.get("meta_agent", {}).get("primary_metric", PRIMARY_META_METRIC))


def _ranking_value_from_metrics(metrics: dict | None) -> float | None:
    if not metrics or metrics.get("status") != "success":
        return None
    key = metrics.get("ranking_metric", PRIMARY_META_METRIC)
    if key == "macro_roc_auc":
        return metrics.get("macro_roc_auc")
    return metrics.get("macro_average_precision")


def _format_iteration_metrics(metrics: dict | None) -> str:
    return format_metrics_dict(metrics, ranking_metric=PRIMARY_META_METRIC)


def run(config: dict) -> None:
    perch_paths = config.get("perch", {})
    logs_dir = Path(perch_paths["logs_dir"]) if perch_paths.get("logs_dir") else ROOT / "logs"
    code_dir = Path(perch_paths["code_dir"]) if perch_paths.get("code_dir") else logs_dir / "perch_agent_codes"
    cache_dir = Path(perch_paths["cache_dir"]) if perch_paths.get("cache_dir") else logs_dir / "perch_cache"
    mem_dir = Path(perch_paths["memory_dir"]) if perch_paths.get("memory_dir") else logs_dir / "perch_memory"
    for d in [logs_dir, code_dir, cache_dir, mem_dir]:
        d.mkdir(parents=True, exist_ok=True)

    preset = config.get("meta_aug_preset")
    if preset:
        print(f"  Meta aug baseline: {preset}")

    perch_cfg      = config.get("perch", {})
    onnx_slug      = perch_cfg.get("onnx_dataset",       "rishikeshjani/perch-onnx-for-birdclef-2026")
    labels_slug    = perch_cfg.get("perch_labels_model",  "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu")
    max_samples    = perch_cfg.get("max_train_samples",   None)
    sample_frac    = float(config.get("train_sample_frac", 1.0))
    embed_bs       = perch_cfg.get("embed_batch_size",    16)
    force_rebuild  = perch_cfg.get("force_rebuild_cache", False)
    head_train_max = config.get("head_train_max_samples")
    if head_train_max is None:
        head_train_max = perch_cfg.get("head_train_max_samples")
    if head_train_max is not None:
        head_train_max = int(head_train_max)
    max_iterations = config.get("max_iterations", 10)

    researcher_model = config.get("researcher", {}).get("model",       "deepseek-r1:8b")
    coder_model      = config.get("llm",        {}).get("model",       "deepseek-r1:8b")
    provider         = config.get("llm",        {}).get("provider",    "ollama")
    researcher_temp  = config.get("researcher", {}).get("temperature", 0.6)
    coder_temp       = config.get("llm",        {}).get("temperature", 0.2)
    py_exe           = config.get("execution",  {}).get("python_executable", "python3")
    timeout          = config.get("execution",  {}).get("timeout_seconds",   1800)

    print("=" * 60)
    print("  BirdCLEF Perch Agent — Researcher / Coder Architecture")
    print("=" * 60)

    # ── Step 1: Install deps ──────────────────────────────────────────────
    _ensure_deps()

    # ── Step 2: Download ONNX model + Perch labels ────────────────────────
    onnx_path   = _find_or_download_onnx(onnx_slug)
    labels_path = _find_or_download_perch_labels(labels_slug)

    # ── Step 3: Load ONNX session ─────────────────────────────────────────
    sess, inp_name, emb_idx, logit_idx = _load_onnx_session(onnx_path)

    # ── Step 4: Load data ─────────────────────────────────────────────────
    import pandas as pd
    os.environ.setdefault("BIRDCLEF_DATA_DIR", str(ROOT / "data"))
    from data_io import (
        load_core_tables, resolve_birdclef_paths,
        species_columns_from_sample_submission, validate_required_files,
    )
    paths   = resolve_birdclef_paths()
    missing = validate_required_files(paths)
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")

    tables      = load_core_tables(paths)
    train_df    = tables["train"]
    sample_sub  = tables["sample_submission"]
    _tax = tables.get("taxonomy")
    taxonomy_df = _tax if _tax is not None else pd.read_csv(paths.taxonomy_csv)

    species_cols   = species_columns_from_sample_submission(sample_sub)
    species_to_idx = {s: i for i, s in enumerate(species_cols)}
    n_species      = len(species_cols)

    # ── Step 5: Build logit mapping ───────────────────────────────────────
    MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL = _build_logit_mapping(
        labels_path, taxonomy_df, species_cols
    )

    # ── Save species mapping for Kaggle notebook ──────────────────────────
    import numpy as _np_map
    bc_indices_full = _np_map.full(n_species, int(NO_LABEL), dtype=_np_map.int32)
    bc_indices_full[MAPPED_POS] = MAPPED_BC_IDX
    _np_map.save(str(mem_dir / "bc_indices.npy"), bc_indices_full)
    with open(mem_dir / "proxy_map.json", "w") as _f:
        json.dump({str(k): v for k, v in proxy_map.items()}, _f)
    with open(mem_dir / "species_cols.json", "w") as _f:
        json.dump(species_cols, _f)
    with open(mem_dir / "mapping_meta.json", "w") as _f:
        json.dump({"NO_LABEL": int(NO_LABEL), "n_species": n_species}, _f)

    # ── Step 6: Build embedding caches (once) ─────────────────────────────
    train_cache = cache_dir / (
        f"train_emb_{preset}.npz" if preset else "train_emb.npz"
    )
    # Meta staged 1a: one val cache shared across aug baselines (val has no train aug).
    val_cache = (
        cache_dir.parent / "val_emb.npz" if preset else cache_dir / "val_emb.npz"
    )
    aug_config = config.get("augmentation")
    train_meta_path = _embedding_cache_meta_path(train_cache)
    if force_rebuild:
        for p in (train_cache, train_meta_path):
            if p.exists():
                p.unlink()
                print(f"  [Cache] Removed {p.name} (force_rebuild_cache)")

    if not train_cache.exists():
        print("\n  [Setup] Building training embedding cache (runs once, ~30-60 min)...")
        _build_train_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            train_df, species_to_idx, paths.train_audio_dir, n_species,
            train_cache,
            sample_frac=sample_frac,
            max_samples=max_samples,
            batch_size=embed_bs,
            aug_config=aug_config,
            soundscapes_dir=paths.train_soundscapes_dir,
            aug_preset=preset,
        )
    else:
        import numpy as np
        d = np.load(str(train_cache))
        cached = _read_embedding_cache_manifest(train_cache)
        print(f"  [Cache] Reusing existing train embeddings → {train_cache}")
        if cached:
            print(
                f"  [Cache]   preset={cached.get('aug_preset')}  "
                f"sample_frac={cached.get('sample_frac')}  "
                f"n={cached.get('n_samples', d['X'].shape[0])}"
            )
        print(f"  [Cache] Train cache loaded: X={d['X'].shape}  y={d['y'].shape}")

    if force_rebuild and val_cache.exists():
        val_cache.unlink()
        print(f"  [Cache] Removed {val_cache.name} (force_rebuild_cache)")

    if not val_cache.exists():
        print("\n  [Setup] Building validation embedding cache...")
        ok = _build_val_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            paths.train_soundscapes_dir, paths.train_soundscapes_labels_csv,
            species_to_idx, n_species, val_cache, embed_bs,
        )
        if not ok:
            print("  [Cache] Soundscape labels not found — using focal-clip fallback for val.")
            _build_focal_val_fallback(train_cache, val_cache)
    else:
        import numpy as np
        d = np.load(str(val_cache))
        print(f"  [Cache] Reusing existing val embeddings → {val_cache}")
        print(f"  [Cache] Val cache loaded: X={d['X'].shape}  y={d['y'].shape}")

    # ── Step 7: Set up agent components ──────────────────────────────────
    researcher_llm = LLMClient(provider=provider, model=researcher_model)
    coder_llm      = LLMClient(provider=provider, model=coder_model)
    ranking_metric = _ranking_metric_from_config(config)
    memory         = ExperimentMemory(mem_dir, ranking_metric=ranking_metric)
    researcher     = PerchResearcher(researcher_llm, memory, temperature=researcher_temp)
    executor       = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator      = Evaluator(row_id_column_name="row_id")

    print(f"\n  Researcher model : {researcher_model}")
    print(f"  Coder model      : {coder_model}")
    print(f"  Max iterations   : {max_iterations}")
    if head_train_max and head_train_max > 0:
        print(
            f"  Head train cap   : {head_train_max} embeddings per iteration "
            f"(full cache on disk unchanged)"
        )
    print(
        f"  Ranking metric   : {ranking_metric} "
        f"(each run logs macro_AP, macro_AUC, median_AUC)"
    )
    prior = memory.total()
    if prior:
        best = memory.best_runs(1)
        best_str = memory._format_run_score(best[0]) if best else "none"
        print(f"  Memory           : {prior} prior runs | best {best_str}")
    else:
        print("  Memory           : fresh start")
    print("=" * 60)

    # ── Step 8: Agent loop ────────────────────────────────────────────────
    y_true_path        = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path        = Path(tempfile.gettempdir()) / "_y_pred.npy"
    trained_head_path  = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path     = mem_dir / "best_head.keras"
    _prior_best = memory.best_runs(1)
    best_score_ever = memory._ranking_value(_prior_best[0]) if _prior_best else -1.0

    # ── Iteration 0: mandatory linear-probe baseline (fresh starts only) ─────
    if memory.total() == 0:
        print(f"\n{'─'*60}")
        print("  ITERATION 0 — Linear probe baseline (single Dense layer)")
        print(f"{'─'*60}")
        _bl_script = _build_script(_SAFE_DEFAULT_SLOT_CODE, train_cache, val_cache)
        _bl_path   = code_dir / "iter_000_baseline.py"
        _bl_path.write_text(_bl_script, encoding="utf-8")
        _bl_result = executor.run_file(_bl_path)
        _bl_metrics = None
        if _bl_result.success and "EVAL_ARTIFACTS_SAVED" in (_bl_result.stdout or ""):
            if y_true_path.exists() and y_pred_path.exists():
                _bl_summary  = evaluator.evaluate_from_files(y_true_path, y_pred_path)
                _bl_metrics  = _bl_summary.metrics
        print(f"  [Baseline] {_format_iteration_metrics(_bl_metrics)}")
        if not _bl_result.success and _bl_result.stderr:
            print(f"  [Baseline error] {_bl_result.stderr[-400:]}")
        memory.log(spec=_BASELINE_SPEC, metrics=_bl_metrics, code=_SAFE_DEFAULT_SLOT_CODE)
        _bl_rank = _ranking_value_from_metrics(_bl_metrics)
        if _bl_rank is not None and _bl_rank > best_score_ever:
            best_score_ever = _bl_rank
            import shutil
            if trained_head_path.exists():
                shutil.copy2(str(trained_head_path), str(best_head_path))
                _ws = Path(tempfile.gettempdir()) / "_trained_head.weights.h5"
                if _ws.exists():
                    shutil.copy2(str(_ws), str(mem_dir / "best_head.weights.h5"))
            if y_pred_path.exists():
                shutil.copy2(str(y_pred_path), str(mem_dir / "best_val_preds.npy"))
            if y_true_path.exists():
                shutil.copy2(str(y_true_path), str(mem_dir / "y_val.npy"))
            (mem_dir / "best_head_code.py").write_text(_SAFE_DEFAULT_SLOT_CODE, encoding="utf-8")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'─'*60}")

        spec      = researcher.next_experiment()
        slot_code = generate_perch_code(coder_llm, spec, coder_temp)

        if slot_code is None:
            print("  [Coder] All AST retries exhausted — falling back to safe default residual MLP.")
            slot_code = _SAFE_DEFAULT_SLOT_CODE

        # Execution retry loop — up to 5 attempts; on failure the coder sees the error and repairs
        _MAX_EXEC_ATTEMPTS = 5
        metrics = None
        final_slot_code = slot_code
        for exec_attempt in range(1, _MAX_EXEC_ATTEMPTS + 1):
            script      = _build_script(
                final_slot_code, train_cache, val_cache, head_train_max_samples=head_train_max
            )
            script_path = code_dir / f"iter_{iteration:03d}_a{exec_attempt}.py"
            script_path.write_text(script, encoding="utf-8")

            print(f"  [Executor] Attempt {exec_attempt}/{_MAX_EXEC_ATTEMPTS} — {script_path.name} ...")
            result = executor.run_file(script_path)

            if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
                if y_true_path.exists() and y_pred_path.exists():
                    summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
                    metrics = summary.metrics
                break

            error_msg = (result.stderr or "")[-1500:]
            print(f"  [Executor] Attempt {exec_attempt} failed.")
            if error_msg:
                # TF logs to stderr too — show tail where the real Python error is
                print(f"  [Error]  {error_msg[-600:]}")

            if exec_attempt == _MAX_EXEC_ATTEMPTS:
                print(f"  [Coder] All {_MAX_EXEC_ATTEMPTS} execution attempts exhausted — skipping iteration.")
                break

            repaired = _repair_perch_code(coder_llm, spec, final_slot_code, error_msg, coder_temp)
            if repaired is None:
                print("  [Coder] Repair failed — skipping remaining attempts.")
                break
            final_slot_code = repaired

        slot_code = final_slot_code  # use the final (possibly repaired) code for logging
        rank_val = _ranking_value_from_metrics(metrics)
        print(f"  [Result] {_format_iteration_metrics(metrics)}")

        memory.log(spec=spec, metrics=metrics, code=slot_code)

        # Promote to best model if this run beat the previous best (by ranking metric)
        if rank_val is not None and rank_val > best_score_ever:
            best_score_ever = rank_val
            auc = metrics.get("macro_roc_auc") if metrics else None
            ap = metrics.get("macro_average_precision") if metrics else None
            med = metrics.get("median_per_class_auc") if metrics else None
            if trained_head_path.exists():
                import shutil
                shutil.copy2(str(trained_head_path), str(best_head_path))
                # Also copy weights-only file for Kaggle (no Keras version dependency)
                _weights_src = Path(tempfile.gettempdir()) / "_trained_head.weights.h5"
                if _weights_src.exists():
                    shutil.copy2(str(_weights_src), str(mem_dir / "best_head.weights.h5"))
                with open(mem_dir / "best_model_info.json", "w") as _f:
                    json.dump({
                        "ranking_metric": ranking_metric,
                        "ranking_value": rank_val,
                        "macro_average_precision": ap,
                        "macro_roc_auc": auc,
                        "median_per_class_auc": med,
                        "auc": auc,
                        "iteration": iteration,
                        "spec": spec,
                    }, _f, indent=2)
                # Save val preds for meta-agent ensemble phase
                _y_pred_tmp = Path(tempfile.gettempdir()) / "_y_pred.npy"
                _y_true_tmp = Path(tempfile.gettempdir()) / "_y_true.npy"
                if _y_pred_tmp.exists():
                    shutil.copy2(str(_y_pred_tmp), str(mem_dir / "best_val_preds.npy"))
                if _y_true_tmp.exists():
                    shutil.copy2(str(_y_true_tmp), str(mem_dir / "y_val.npy"))
                # Save best slot code (build_head + get_training_config) for final retrain
                (mem_dir / "best_head_code.py").write_text(slot_code, encoding="utf-8")
                print(
                    f"  [Best] NEW BEST {_format_iteration_metrics(metrics)} "
                    f"— head saved to {best_head_path.name}"
                )

        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] {memory._format_run_score(best[0])}")

    print(f"\n{'='*60}")
    print("  DONE")
    best = memory.best_runs(3)
    for i, r in enumerate(best, 1):
        print(f"  #{i} {memory._format_run_score(r)} | {r['reasoning'][:80]}")
    print(f"{'='*60}")

    # ── Final retrain on full data (train + val) with best spec ──────────────
    if config.get("perch", {}).get("skip_final_retrain", False):
        print("\n  Final retrain skipped (perch.skip_final_retrain=true).")
        print("  Use logs/perch_memory/best_head.weights.h5 from the search loop.")
        return

    best_runs = memory.best_runs(1)
    if best_runs:
        best_auc  = best_runs[0].get("macro_roc_auc", 0.0)
        best_code_path = mem_dir / "best_head_code.py"
        if not best_code_path.exists():
            print("  No best_head_code.py found — skipping final retrain.")
            return
        best_code = best_code_path.read_text(encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"  FINAL RETRAIN — full data (val AUC={best_auc:.5f})")
        print(f"{'='*60}")
        final_script      = _build_final_retrain_script(best_code, mem_dir, cache_dir)
        final_script_path = code_dir / "final_retrain.py"
        final_script_path.write_text(final_script, encoding="utf-8")
        result = executor.run_file(final_script_path)
        if result.success and "FINAL_RETRAIN_DONE" in (result.stdout or ""):
            print(f"  Final head saved → {mem_dir / 'final_head.weights.h5'}")
        else:
            print(f"  Final retrain failed: {(result.stderr or '')[-400:]}")
    else:
        print("  No successful runs — skipping final retrain.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "agent_config.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config)


if __name__ == "__main__":
    main()
