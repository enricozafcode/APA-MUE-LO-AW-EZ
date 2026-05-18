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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from code_executor import CodeExecutor
from evaluator import Evaluator
from llm_client import LLMClient, llm_response_failed as _llm_response_failed
from memory import ExperimentMemory
from perch_memory import PerchExperimentMemory
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


def configure_tensorflow_cpu_only() -> None:
    """
    Use CPU-only TensorFlow (avoids Metal/CUDA hangs when ONNX Runtime is loaded too).

    Matches the Perch/BirdNET notebooks: ONNX for embeddings, lightweight head on CPU.
    Call before ``import tensorflow`` in pseudo-labeling paths.
    """
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import tensorflow as tf

        for dev in tf.config.list_physical_devices():
            if dev.device_type in ("GPU", "TPU"):
                try:
                    tf.config.set_visible_devices([], dev.device_type)
                except (RuntimeError, ValueError):
                    pass
        # macOS: Keras model.fit() can hang during graph compile; eager is safer.
        try:
            tf.config.run_functions_eagerly(True)
        except (RuntimeError, ValueError, AttributeError):
            pass
        try:
            tf.config.optimizer.set_jit(False)
        except (RuntimeError, ValueError, AttributeError):
            pass
    except ImportError:
        pass


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
    """Run a batch of audio waveforms through Perch → (embeddings, logits)."""
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


def _embedding_cache_clips_path(npz_path: Path) -> Path:
    return npz_path.parent / f"{npz_path.stem}_clips.jsonl"


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


def _subsample_train_indices(
    y,
    max_samples: int | None,
    random_state: int = 42,
) -> np.ndarray:
    """Stratified row indices for head training (same logic as harness subsample)."""
    import numpy as np

    n = int(y.shape[0])
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=np.int64)

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
    return np.sort(idx.astype(np.int64))


def compute_and_save_head_train_indices(
    train_cache: Path,
    out_path: Path,
    max_samples: int,
    random_state: int = 42,
) -> Path:
    """Persist fixed head-train row indices so all aug-search runs use the same clips."""
    import numpy as np

    d = np.load(str(train_cache), allow_pickle=True)
    y = d["y"].astype(np.float32)
    idx = _subsample_train_indices(y, max_samples, random_state=random_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), idx)
    print(
        f"  [Indices] Saved {len(idx)} head-train indices → {out_path} "
        f"(from {y.shape[0]} cached rows, seed={random_state})"
    )
    return out_path


def _load_clip_rows_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"label": str(row["label"]), "filename": str(row["filename"])})
    return rows


def _save_clip_rows_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def backfill_cache_clips_manifest(
    train_cache: Path,
    train_df,
    species_to_idx: dict,
    audio_dir: Path,
    *,
    sr: int = 32_000,
    clip_sec: float = 5.0,
) -> Path:
    """
    Reconstruct per-row clip list for an existing cache (no re-embedding).
    Matches cache-build skips (missing files / load errors).
    """
    clips_path = _embedding_cache_clips_path(train_cache)
    if clips_path.exists():
        return clips_path

    import numpy as np

    meta = _read_embedding_cache_manifest(train_cache) or {}
    sample_frac = float(meta.get("sample_frac", 1.0))
    max_samples = meta.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    train_df, lcol, fcol = _select_train_df_stratified(
        train_df,
        species_to_idx,
        audio_dir,
        sample_frac=sample_frac,
        max_samples=max_samples,
        random_state=42,
    )
    embedded: list[dict[str, str]] = []
    for label, fname in train_df[[lcol, fcol]].dropna().values.tolist():
        p = audio_dir / str(fname)
        if not p.exists():
            continue
        try:
            _load_and_pad(p, sr, clip_sec)
            embedded.append({"label": str(label), "filename": str(fname)})
        except Exception:
            continue

    n_cache = int(np.load(str(train_cache), allow_pickle=True)["X"].shape[0])
    if len(embedded) != n_cache:
        raise RuntimeError(
            f"Clip manifest replay ({len(embedded)} rows) != cache rows ({n_cache}). "
            f"Rebuild source cache with a current perch_agent to record clip paths."
        )
    _save_clip_rows_jsonl(clips_path, embedded)
    print(f"  [Clips] Backfilled {len(embedded)} row→clip mappings → {clips_path.name}")
    return clips_path


def save_head_train_clip_subset(
    source_train_cache: Path,
    indices_path: Path,
    out_clips_path: Path,
    train_df,
    species_to_idx: dict,
    audio_dir: Path,
) -> Path:
    """Write the 2000 (or fewer) clips used for head training (for fast 1c re-embed)."""
    import numpy as np

    if out_clips_path.exists():
        rows = _load_clip_rows_jsonl(out_clips_path)
        print(f"  [Clips] Reusing head-train clip list ({len(rows)} clips) → {out_clips_path}")
        return out_clips_path

    clips_path = backfill_cache_clips_manifest(
        source_train_cache, train_df, species_to_idx, audio_dir
    )
    all_rows = _load_clip_rows_jsonl(clips_path)
    idx = np.load(str(indices_path))
    subset = [all_rows[int(i)] for i in idx]
    _save_clip_rows_jsonl(out_clips_path, subset)
    print(
        f"  [Clips] Head-train subset: {len(subset)} clips → {out_clips_path.name} "
        f"(for fast stage-1c embedding)"
    )
    return out_clips_path


def _subsample_cached_train(
    X,
    S,
    y,
    max_samples: int | None,
    random_state: int = 42,
):
    """Stratified subsample of cached rows (for fast head training); full cache unchanged on disk."""
    idx = _subsample_train_indices(y, max_samples, random_state)
    if len(idx) == int(y.shape[0]):
        return X, S, y
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
    clip_rows: list[dict[str, str]] | None = None,
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

    embedded_clips: list[dict[str, str]] = []
    if clip_rows is not None:
        rows = [(r["label"], r["filename"]) for r in clip_rows]
        print(
            f"  [Cache] Building train cache: {len(rows)} clips "
            f"(fixed subset, new augmentation) → {cache_path.name}"
        )
    else:
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
        embedded_clips.append({"label": str(label), "filename": str(fname)})

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
        "clip_subset": clip_rows is not None,
    })
    if embedded_clips:
        _save_clip_rows_jsonl(_embedding_cache_clips_path(cache_path), embedded_clips)
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
    wavs, labels, row_ids = [], [], []

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
        end_sec = int(row["end_sec"])
        row_ids.append(
            str(row["filename"]).replace(".ogg", "") + f"_{end_sec}"
        )

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
    np.savez_compressed(
        str(cache_path),
        X=X,
        S=S,
        y=y,
        row_ids=np.array(row_ids, dtype=object),
    )
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

def _extract_first_json_array(text: str) -> list | None:
    """Find the first complete JSON array in text by counting brackets."""
    start = text.find("[")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    return None
    return None


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
    "search_space_note": (
        "All lists below are suggested values / examples — not hard limits. "
        "You may pick values outside these lists when arch_description requires it."
    ),
    "arch_types_note": (
        "Examples only — explore many families; reuse or tweak labels freely; invent names when helpful. "
        "Stage 1a maps the space, not final optimization."
    ),
    "arch_types": [
        "residual_mlp",       # dense residual blocks with skip connections (baseline)
        "attention_mlp",      # self-attention / multi-head attention on projected features
        "gated_mlp",          # GLU-style gating: value * sigmoid(gate), two separate Dense layers
        "highway_network",    # highway gates: out = H*transform + (1-H)*input carry
        "bottleneck_mlp",     # wide → narrow → wide projection bottleneck
        "multi_scale_mlp",    # parallel branches at different widths, merged by concat or add
        "multi_tower_ensemble",  # 3–5 parallel specialist towers, fuse logits (Average or Concat→Dense)
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

# Shared domain context — included in researcher + coder system prompts every LLM call.
PERCH_TASK_CONTEXT = """
TASK — BirdCLEF 2026 (jungle / rainforest soundscapes):
- You classify bird species from audio using frozen Google Perch v2 embeddings (1536-d vectors).
- Training audio comes from tropical soundscape recordings; labels are multi-label over 234 species.
- Class imbalance is severe: many species are rare; most clip×species pairs are negative.
- The Perch ONNX encoder is FIXED. You design only the TF/Keras classification HEAD and training hyperparameters.

COMPETITION CONTEXT (read before proposing architectures — background only, not a coding constraint):
- Recordings are jungle / rainforest soundscapes; the task is multi-label species detection in noisy, imbalanced data.
- On Kaggle, long test soundscapes are evaluated as a sequence of ~60-second windows (not one score per full file).
- Temporal intuition: if the same species is active across several consecutive 60s segments, it is more likely a true
  presence than a single isolated spike—design heads that output well-calibrated per-segment probabilities (not
  over-confident on weak evidence), so later temporal aggregation can filter false alarms.
- You do NOT implement windowing or sequence models in build_head; the harness trains on cached Perch embeddings
  (one 1536-d vector per training row). Use this context for hypotheses and architecture choices only.

FINAL BLEND (harness, not part of build_head graph):
  y_pred = perch_weight * perch_scores + (1 - perch_weight) * head_output
  where perch_scores are Perch's own mapped species logits on the same window.
""".strip()

PERCH_RESEARCHER_SYSTEM_PROMPT = (
    """You are an expert ML researcher optimizing a BirdCLEF classification head on top of frozen Google Perch 1536-d embeddings.

"""
    + PERCH_TASK_CONTEXT
    + """

ITERATIVE CAMPAIGN (important mindset):
- You are one step in a long search (many iterations). There is NO pressure to nail the best model in one shot.
- Treat each run as an experiment: learn from scores in the history, form a direction, then propose one concrete try.
- Failed or mediocre runs are useful signal — adjust and try again. Exploration is expected.

HOW TO WORK EACH ITERATION (keep your thinking focused; output stays compact JSON):
1. Read the experiment history — what helped, what hurt, what is still uncertain?
2. State your direction in "reasoning" (what you learned + what to try next).
3. State a testable claim in "hypothesis" (one line).
4. Propose the run in arch_type, arch_description, and hyperparameters.

BREVITY (critical — long answers slow the pipeline and are not needed):
- reasoning: at most 2 short sentences (~50 words). Lead with what you learned from past results.
- hypothesis: 1 sentence (~25 words).
- arch_description: at most 4 sentences (~150 words). Enough for the coder to implement; no essays.
- Respond with ONLY the JSON object — no markdown, no preamble, no chain-of-thought outside JSON.

YOUR PRIMARY OBJECTIVE (architecture-discovery phase): Propose the next head to train.
You are **mapping the search space**, not optimizing a single winner yet. Spread tries across many
different architecture ideas — attention, gating, MoE, ensembles, residuals, etc.
You may also revisit or tweak an existing family (same arch_type, different layout or hypers) when that
helps you learn how that family behaves. Labels do not need to be new or invented.
A later refine phase will seriously tune the most promising designs; here breadth and learning matter more than +0.001 on the leaderboard.

EXPLORING THE SPACE (both are valid in 1a):
- Try **different structural families** you have not exercised much yet (highest priority over the campaign).
- **Tweak or extend** a family you already ran — different depth, fusion, dropout, perch_weight — when it fills a gap in your understanding.
- Past runs in the registry are guidance, NOT a checklist. Under-sampled regions of the space are worth a visit.

CUSTOM arch_type NAMES (you have full freedom):
- The arch_types list in the search space is EXAMPLES ONLY, not a closed menu.
- You MAY invent new arch_type strings (snake_case, descriptive, e.g. dual_path_gated_residual,
  calibrated_ensemble_v2) whenever the design does not fit a canned label.
- arch_description must fully specify the Keras graph; the coder implements from that text, not from the label alone.

Example architecture families (starting points — combine or extend freely):
- residual_mlp:       Dense residual blocks with skip connections (BN→Dense→LN→blocks→proj→sigmoid). Solid baseline; combine with exploring other families too.
- attention_mlp:      Multi-head self-attention on projected features. Project input to hidden_dim, apply MHA, then FFN. Good for capturing feature interactions.
- gated_mlp:          GLU-style gating: two parallel Dense(hidden_dim) — one linear (value), one sigmoid (gate) — multiplied element-wise, then residual add. Selects which embedding dims are informative.
- highway_network:    Transform gate H (sigmoid) and carry gate (1-H): out = H*Dense(x) + (1-H)*x. Learnable depth blending.
- bottleneck_mlp:     Project wide→narrow→wide to force information compression: Dense(2048)→Dense(128)→Dense(2048). Forces the head to learn a compact representation.
- multi_scale_mlp:    Parallel branches at different widths (e.g. 256, 512, 1024) processing the same input, outputs merged by concatenation then projected.
- multi_tower_ensemble: 3–5 parallel specialist towers on the same stem (each tower a different topology), per-tower
  logits fused by Average or Concatenate→Dense(sigmoid). Best when you want explicit ensemble diversity in one trainable model.
- transformer_block:  Reshape to (batch,1,emb_dim), apply MultiHeadAttention + FFN + LayerNorm (1-2 blocks). Classic transformer encoder.
- mixture_of_experts: K parallel expert Dense(hidden_dim) layers; router Dense(K, softmax) on x; weighted sum of experts (all tensors hidden_dim — never concat experts then multiply by router).
- dense_connections:  DenseNet-style: concat prior features, then Dense(hidden_dim) to compress — never Add() concat-sized tensors with hidden_dim tensors.
- linear_probe:       Single Dense(n_classes, sigmoid). Strong sanity-check baseline for raw embedding quality.
- ensemble head:      Different head topologies (residual, gated, attention, MoE, etc.) on the same Perch embedding (which ones to use should be decided by the researcher).

Learning from history:
- Note which families look strong, weak, or under-tested — use that to choose the **next region of the space** to probe.
- Do not repeat identical failed configs.
- strategy "explore" (typical in 1a): sample a different part of the space — new family, invented label, OR a meaningful variant/tweak of one you already tried.
- strategy "exploit" (also fine in 1a when useful): adjust an existing configuration to learn its behavior — not for squeezing the global best, but for fair comparison within a family.
- Avoid spending most runs only micro-tuning the current #1 unless the space is already well covered.
- Soundscape context: noisy jungle audio, rare species, calibrated per-window probabilities (see TASK above).

arch_description must be implementable (stem Dense(hidden_dim), blocks, fusion, final sigmoid) but stay brief.

OUTPUT: ONLY a single JSON object — start with { and end with }. No other text.

SHAPE-SAFE arch_description rules (embeddings are 1536-d; hidden_dim is your working width, typically 512–1024):
- Always state: "Project input Dense(hidden_dim) from emb_dim first" before any residual, MoE, or concat blocks.
- For residual/highway/gated blocks: every Add() input must already be hidden_dim (project skip connections if needed).
- For mixture_of_experts: "K experts each Dense(hidden_dim); router Dense(K, softmax); weighted sum stays hidden_dim" — do NOT describe concat of expert outputs.
- For dense_connections: "Concatenate then Dense(hidden_dim) to compress" after each growth step — do NOT Add() a wide concat tensor to a narrow tensor.
- For multi_tower_ensemble: list each tower (e.g. 5 towers: residual, gated, attention, bottleneck, linear_probe_on_stem);
  each tower outputs num_classes logits before fusion; state Average vs Concatenate→Dense fusion.

Example (note short reasoning / hypothesis / arch_description):
{"arch_type": "mixture_of_experts", "arch_description": "Dense(1024) stem. Four expert Dense(1024) branches, Concatenate, Dense(1024) compress, residual Add, LayerNorm. Dropout 0.3, Dense(n_classes, sigmoid).", "hidden_dim": 1024, "n_layers": 2, "dropout": 0.3, "activation": "gelu", "normalization": "layer_norm", "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 30, "patience": 5, "perch_weight": 0.2, "reasoning": "Registry is heavy on residual_mlp; MoE not meaningfully tested. Try routed experts for species-specific features.", "hypothesis": "MoE routing helps rare species in noisy soundscapes.", "strategy": "explore"}

Required keys: arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."""
)

PERCH_BATCH_PLANNER_ADDENDUM = """
BATCH PLANNING — return exactly {batch_size} experiments in ONE JSON object (one LLM call per round):
{{
  "planner_note": "optional one short sentence",
  "experiments": [
    {{ "slot": "tweak", ...all experiment keys... }},
    {{ "slot": "explore", ... }},
    {{ "slot": "free", ... }}
  ]
}}

Slot roles (when batch_size=3):
- tweak:  adjust or extend a promising family (layout or hypers) — learn/compare within a strong idea
- explore: sample a different part of the architecture space (different family or clear structural change)
- free:   your choice — fill a gap, sanity check, or creative wildcard

Each experiment MUST include arch_description (2–4 sentences, implementable Keras layout) — the coder
implements from that text; hyperparameters alone are NOT enough.
Required keys per experiment: slot, arch_type, arch_description, hidden_dim, n_layers, dropout, activation,
normalization, learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy.
Keep reasoning/hypothesis/arch_description SHORT.
Search-space lists are hints only (e.g. n_layers: 5 is allowed).
Output either {{"planner_note":"...", "experiments":[...]}} OR a JSON array of 3 experiment objects — not separate objects.
"""

PERCH_STAGE_1A_EXPLORE_ADDENDUM = """
STAGE 1a — EXPLORE THE ARCHITECTURE SPACE (you are in this phase now):
- Goal: learn what kinds of heads work — cover many different architectures over the campaign, not maximize one score yet.
- Favor variety: rotate through different families (attention, gated, MoE, multi-tower, highway, residual, etc.).
- Tweaking an existing arch_type (layout or hypers) is welcome when it helps you compare or understand that family.
- Nothing must be "new" — example labels and invented names are both fine; what matters is useful coverage of ideas.
- Do not obsess over beating the current best each iteration; small improvements to the leader are low priority vs unexplored space.
- Serious optimization of the top candidates happens in a LATER refine phase.
"""

PERCH_REFINE_RESEARCHER_ADDENDUM = """
REFINE MODE (stage 1b) — optimize the current best LOCKED head (same arch_type for every experiment).
- Do NOT switch architecture family. Decide quickly — short JSON, short reasoning.
- All experiments are independent tries to beat the champion score (hypers and/or layout within the family).
- There are NO explore/tweak/free roles — every slot is just another optimization attempt.
- strategy MUST be "exploit" for every experiment.
- Keep reasoning/hypothesis/arch_description SHORT (same limits as stage 1a).
"""

PERCH_REFINE_BATCH_PLANNER_ADDENDUM = """
STAGE 1b BATCH — return exactly {batch_size} refine experiments in ONE JSON (one LLM call per planner round):
{{
  "planner_note": "optional one short sentence",
  "experiments": [
    {{ "arch_type": "<LOCKED>", ...all experiment keys... }},
    {{ "arch_type": "<LOCKED>", ... }},
    ...
  ]
}}

arch_type MUST equal the locked champion family ({locked}) for every experiment.
Each experiment is another attempt to improve the same best model — vary hypers and/or layout as you see fit.
Optional "slot" field is only a label (e.g. r1, r2) — not a role.

Each experiment MUST include arch_description (2–4 sentences). Required keys: arch_type,
arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size,
optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy (all exploit).
Output {{"planner_note":"...", "experiments":[...]}} OR a JSON array of {batch_size} objects.
"""


PERCH_BATCH_SLOTS = ("tweak", "explore", "free")
REFINE_CHAMPION_SPEC_FILE = "refine_champion_spec.json"
LEGACY_CHAMPION_SPEC_FILE = "stage_1a_champion_spec.json"


def _refine_batch_slot_labels(batch_size: int) -> tuple[str, ...]:
    """Neutral per-run labels for 1b (not explore/tweak/free roles)."""
    return tuple(f"r{i + 1}" for i in range(max(1, batch_size)))


class PerchResearcher:
    def __init__(
        self,
        llm: LLMClient,
        memory: ExperimentMemory,
        temperature: float = 0.6,
        *,
        refine_mode: bool = False,
        locked_arch_type: str | None = None,
        seed_spec: dict | None = None,
        seed_score: float | None = None,
        batch_size: int = 1,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.locked_arch_type = locked_arch_type
        self.seed_spec = seed_spec or {}
        self.seed_score = seed_score
        self.batch_size = max(1, int(batch_size))

    def next_experiment(self) -> dict:
        return self.next_experiments()[0]

    def next_experiments(self) -> list[dict]:
        history  = self.memory.researcher_context()
        best     = self.memory.best_runs(1)
        total    = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"

        batch_size = self.batch_size
        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "residual_mlp")
            user_prompt = (
                f"{history}\n\n"
                f"Available search space (hints only):\n"
                f"{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
                f"Refine runs so far: {total} | best in this campaign: {best_str}\n\n"
            )
            if batch_size > 1:
                user_prompt += (
                    f"Stage 1b batch — propose exactly {batch_size} independent optimization tries "
                    f"(all arch_type={locked}).\n"
                    "Each experiment should try to beat the champion config/score — no explore roles.\n\n"
                    f'Respond with ONE JSON: {{"planner_note": "...", "experiments": [ ... ]}} '
                    f"with {batch_size} items. No prose outside JSON.\n"
                )
                refine_batch = PERCH_REFINE_BATCH_PLANNER_ADDENDUM.format(
                    batch_size=batch_size, locked=locked
                )
                system_prompt = (
                    PERCH_RESEARCHER_SYSTEM_PROMPT
                    + "\n\n"
                    + PERCH_REFINE_RESEARCHER_ADDENDUM
                    + "\n\n"
                    + refine_batch
                )
            else:
                user_prompt += (
                    "Propose ONE refine experiment (arch_type locked). Keep JSON short. strategy=exploit.\n"
                )
                system_prompt = PERCH_RESEARCHER_SYSTEM_PROMPT + "\n\n" + PERCH_REFINE_RESEARCHER_ADDENDUM
        elif batch_size > 1:
            user_prompt = (
                f"{history}\n\n"
                f"Available search space:\n{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
                f"Total experiments so far: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                f"Stage 1a batch plan — propose exactly {batch_size} experiments for this round "
                f"(slots: {', '.join(PERCH_BATCH_SLOTS[:batch_size])}).\n"
                "Explore the space across the three slots; not optimizing the leaderboard in one shot.\n\n"
                f'Respond with ONE JSON object: {{"planner_note": "...", "experiments": [ ... ]}} '
                f"with {batch_size} items. No prose outside JSON.\n"
            )
            batch_addendum = PERCH_BATCH_PLANNER_ADDENDUM.format(batch_size=batch_size)
            system_prompt = (
                PERCH_RESEARCHER_SYSTEM_PROMPT
                + "\n\n"
                + PERCH_STAGE_1A_EXPLORE_ADDENDUM
                + "\n\n"
                + batch_addendum
            )
        else:
            user_prompt = (
                f"{history}\n\n"
                f"Available search space:\n{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
                f"Total experiments so far: {total}\n"
                f"Best so far ({self.memory.ranking_metric}): {best_str}\n\n"
                "Stage 1a — explore the architecture space: try different families across iterations; tweaks to existing configs are OK too.\n"
                "Workflow: (1) what is under-tested or unclear? (2) pick next experiment to learn (3) compact JSON. Not optimizing the leaderboard yet.\n\n"
                "Respond with ONLY a short JSON object — no prose outside JSON. Start with { and end with }. Keys: "
                "arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, "
                "learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."
            )
            system_prompt = PERCH_RESEARCHER_SYSTEM_PROMPT + "\n\n" + PERCH_STAGE_1A_EXPLORE_ADDENDUM

        batch_label = f", batch={batch_size}" if batch_size > 1 else ""
        print(
            f"\n  [Researcher] Planning next "
            f"{'experiment' if batch_size == 1 else f'{batch_size} experiments'} "
            f"({total} in memory, best {best_str}{batch_label}, "
            f"timeout {self.llm.timeout_seconds:.0f}s)..."
        )
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        if _llm_response_failed(response):
            print(
                f"  [Researcher] LLM unavailable or timed out — "
                f"using safe defaults for this round."
            )
            if response:
                print(f"  [Researcher] Detail: {str(response)[:200]}")
            specs = self._refine_or_explore_batch_fallback(batch_size, timed_out=True)
        elif batch_size > 1:
            specs = self._parse_batch_specs(response, batch_size)
        else:
            specs = [self._parse_spec(response)]
        if self.refine_mode and self.locked_arch_type:
            slot_labels = _refine_batch_slot_labels(len(specs))
            for i, spec in enumerate(specs):
                spec["arch_type"] = self.locked_arch_type
                spec["strategy"] = "exploit"
                if i < len(slot_labels):
                    spec["slot"] = slot_labels[i]
        planner_note = ""
        if batch_size > 1 and specs and specs[0].get("_planner_note"):
            planner_note = specs[0].pop("_planner_note", "")
        if planner_note:
            print(f"  [Researcher] Plan: {planner_note[:160]}")
        for i, spec in enumerate(specs, 1):
            slot = spec.get("slot", f"s{i}")
            note = ""
            if spec.pop("_arch_description_synthesized", False):
                note = " | desc=synthesized (planner omitted arch_description)"
            desc_prev = str(spec.get("arch_description", ""))
            if len(desc_prev) > 70:
                desc_prev = desc_prev[:70] + "…"
            print(
                f"  [Researcher] [{slot}] {spec.get('strategy', '?')} | "
                f"arch={spec.get('arch_type')} | desc={desc_prev}{note}"
            )
        return specs

    @staticmethod
    def _clean_llm_response(response: str) -> str:
        _open, _close = "<" + "think" + ">", "</" + "think" + ">"
        cleaned = re.sub(
            re.escape(_open) + r"[\s\S]*?" + re.escape(_close),
            "",
            response,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<think>[\s\S]*?</think>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _parse_spec(self, response: str) -> dict:
        cleaned = self._clean_llm_response(response)

        # Try ```json ... ``` block
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if m:
            try:
                return _perch_fill_defaults(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass

        root = self._parse_batch_root(cleaned)
        if isinstance(root, dict):
            exps = self._experiments_from_root(root)
            if exps:
                return exps[0]
            if root.get("arch_type"):
                normalized = _normalize_experiment_item(root)
                if normalized is not None:
                    return normalized

        print("  [Researcher] Warning: could not parse JSON, using safe defaults.")
        print(f"  [Researcher] Raw response (first 400 chars): {repr(cleaned[:400])}")
        return _perch_safe_defaults()

    def _parse_batch_root(self, cleaned: str) -> dict | None:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return {"experiments": parsed}
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        arr = _extract_first_json_array(cleaned)
        if arr is not None:
            return {"experiments": arr}
        root = _extract_first_json_object(cleaned)
        if root is not None:
            if "experiments" not in root and isinstance(root.get("arch_type"), str):
                return {"experiments": [root]}
            return root
        return None

    def _experiments_from_root(self, root: dict) -> list[dict]:
        raw_list = root.get("experiments")
        if not isinstance(raw_list, list):
            return []
        specs: list[dict] = []
        for item in raw_list:
            normalized = _normalize_experiment_item(item)
            if normalized is not None and _is_valid_experiment_spec(normalized):
                specs.append(normalized)
        return specs

    def _parse_batch_specs(self, response: str, batch_size: int) -> list[dict]:
        cleaned = self._clean_llm_response(response)
        root = self._parse_batch_root(cleaned)
        specs: list[dict] = []
        planner_note = ""

        if isinstance(root, dict):
            planner_note = str(root.get("planner_note") or root.get("batch_note") or "")[:200]
            specs = self._experiments_from_root(root)

        valid = [s for s in specs if _is_valid_experiment_spec(s)]
        if not valid:
            print("  [Researcher] Warning: could not parse batch JSON — using slot fallbacks.")
            print(f"  [Researcher] Raw response (first 500 chars): {repr(cleaned[:500])}")
            return self._refine_or_explore_batch_fallback(batch_size, timed_out=False)

        specs = valid
        slot_pool = (
            _refine_batch_slot_labels(batch_size)
            if self.refine_mode
            else PERCH_BATCH_SLOTS
        )
        for i, spec in enumerate(specs):
            if i < len(slot_pool):
                spec["slot"] = slot_pool[i]
        if planner_note and specs:
            specs[0]["_planner_note"] = planner_note
        if len(specs) < batch_size:
            print(
                f"  [Researcher] Warning: parsed {len(specs)}/{batch_size} experiments — "
                f"filling remainder with slot fallbacks."
            )
            specs.extend(
                self._refine_or_explore_batch_fallback(
                    batch_size - len(specs), timed_out=False
                )
            )
        return specs[:batch_size]

    def _refine_or_explore_batch_fallback(
        self, batch_size: int, *, timed_out: bool
    ) -> list[dict]:
        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "residual_mlp")
            return _perch_refine_batch_fallback(
                batch_size,
                timed_out=timed_out,
                locked_arch_type=str(locked),
                seed_spec=self.seed_spec,
            )
        return _perch_batch_fallback(batch_size, timed_out=timed_out)


def _resolve_experiments_per_round(config: dict, *, refine_mode: bool = False) -> int:
    """Experiments planned per researcher LLM call (1a explore + 1b refine)."""
    rc = config.get("researcher") or {}
    perch = config.get("perch") or {}
    if refine_mode:
        refine = config.get("perch_refine") or {}
        raw = refine.get("experiments_per_researcher_call")
        if raw is None:
            raw = rc.get("batch_size", 3)
        return max(1, int(raw))
    raw = rc.get("batch_size")
    if raw is None:
        raw = perch.get("experiments_per_researcher_call", 1)
    return max(1, int(raw))


def _ranking_value_from_memory_entry(entry: dict, metric: str) -> float:
    if metric == "macro_roc_auc":
        v = entry.get("macro_roc_auc")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_roc_auc")
    else:
        v = entry.get("macro_average_precision")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_average_precision")
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _best_run_from_memory_dir(
    mem_dir: Path,
    *,
    ranking_metric: str,
    locked_arch_type: str | None = None,
) -> dict | None:
    """Best successful run in a perch memory directory (jsonl + best_model_info)."""
    mem_dir = Path(mem_dir)
    if not mem_dir.is_dir():
        return None

    best_entry: dict | None = None
    best_val = -1.0

    jsonl = mem_dir / "experiment_memory.jsonl"
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
                spec = entry.get("spec") or {}
                at = spec.get("arch_type")
                if locked_arch_type and at and at != locked_arch_type:
                    continue
                val = _ranking_value_from_memory_entry(entry, ranking_metric)
                if val > best_val:
                    best_val = val
                    best_entry = dict(entry)
                    best_entry["_source_memory_dir"] = str(mem_dir)
                    best_entry["_ranking_value"] = val

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec_info = info.get("spec") or {}
            at = spec_info.get("arch_type")
            if locked_arch_type and at and at != locked_arch_type:
                pass
            else:
                val = float(
                    info.get("ranking_value", info.get("macro_average_precision", -1))
                )
                if val > best_val:
                    best_val = val
                    best_entry = {
                        "success": True,
                        "spec": spec_info,
                        "macro_average_precision": info.get("macro_average_precision"),
                        "macro_roc_auc": info.get("macro_roc_auc"),
                        "median_per_class_auc": info.get("median_per_class_auc"),
                        "metrics": info,
                        "_source_memory_dir": str(mem_dir),
                        "_ranking_value": val,
                    }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return best_entry


def _resolve_refine_champion(
    mem_dir: Path,
    refine_cfg: dict,
    *,
    ranking_metric: str,
) -> dict:
    """
    Scan refine + parent experiment memories and pick the best successful run
  (same locked arch_type). Beats relying only on the 1a handoff score.
    """
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )
    candidates: list[dict] = []

    handoff_spec = dict(refine_cfg.get("seed_spec") or {})
    handoff_score = float(refine_cfg.get("seed_score", -1.0))
    if handoff_spec:
        candidates.append(
            {
                "spec": handoff_spec,
                "ranking_value": handoff_score,
                "macro_average_precision": refine_cfg.get("seed_macro_ap"),
                "macro_roc_auc": refine_cfg.get("seed_macro_auc"),
                "median_per_class_auc": refine_cfg.get("seed_median_auc"),
                "source": "stage_1a_handoff",
                "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
            }
        )

    seen_dirs: set[str] = set()
    for raw in (refine_cfg.get("parent_memory_dir"), str(mem_dir)):
        if not raw:
            continue
        d = Path(raw)
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        best = _best_run_from_memory_dir(
            d, ranking_metric=ranking_metric, locked_arch_type=str(locked)
        )
        if best is None:
            continue
        spec = dict(best.get("spec") or {})
        val = float(best.get("_ranking_value", _ranking_value_from_memory_entry(best, ranking_metric)))
        candidates.append(
            {
                "spec": spec,
                "ranking_value": val,
                "macro_average_precision": best.get("macro_average_precision"),
                "macro_roc_auc": best.get("macro_roc_auc"),
                "median_per_class_auc": best.get("median_per_class_auc"),
                "source": f"memory:{d.name}",
                "memory_dir": best.get("_source_memory_dir", str(d)),
            }
        )

    if not candidates:
        return {
            "spec": handoff_spec or {"arch_type": locked},
            "ranking_value": handoff_score,
            "source": "default",
            "memory_dir": refine_cfg.get("parent_memory_dir") or str(mem_dir),
        }

    winner = max(candidates, key=lambda c: float(c.get("ranking_value", -1.0)))
    return winner


def _load_refine_champion_spec(mem_dir: Path) -> dict | None:
    """Canonical champion written at refine start (full spec + metadata)."""
    mem_dir = Path(mem_dir)
    for name in (REFINE_CHAMPION_SPEC_FILE, LEGACY_CHAMPION_SPEC_FILE):
        path = mem_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        spec = dict(payload.get("spec") or {})
        if payload.get("locked_arch_type"):
            spec["arch_type"] = payload["locked_arch_type"]
        return spec or None
    return None


def _persist_refine_champion_artifacts(
    mem_dir: Path, refine_cfg: dict, *, ranking_metric: str
) -> tuple[dict, float, dict]:
    """
    Resolve best model from experiment memories, write refine_champion_spec.json,
    and copy winning head artifacts into the refine memory dir.
    """
    import shutil

    mem_dir.mkdir(parents=True, exist_ok=True)
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    winner = _resolve_refine_champion(mem_dir, refine_cfg, ranking_metric=ranking_metric)
    seed_spec = dict(winner.get("spec") or {})
    seed_score = float(winner.get("ranking_value", refine_cfg.get("seed_score", -1.0)))
    source = str(winner.get("source", "?"))
    winner_mem = Path(winner.get("memory_dir") or mem_dir)

    if winner_mem.is_dir() and winner_mem != mem_dir:
        info_path = winner_mem / "best_model_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                for k, v in (info.get("spec") or {}).items():
                    if v is not None and k not in seed_spec:
                        seed_spec[k] = v
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists():
                shutil.copy2(src, mem_dir / name)
    elif winner_mem.is_dir():
        for name in ("best_head_code.py", "best_model_info.json", "best_head.keras"):
            src = winner_mem / name
            if src.exists() and not (mem_dir / name).exists():
                shutil.copy2(src, mem_dir / name)

    seed_spec["arch_type"] = locked
    if not seed_spec.get("arch_description"):
        seed_spec["arch_description"] = _synthesize_arch_description(seed_spec)

    payload = {
        "locked_arch_type": locked,
        "aug_baseline": refine_cfg.get("aug_baseline"),
        "champion_score": seed_score,
        "ranking_metric": ranking_metric,
        "source": source,
        "winner_memory_dir": str(winner_mem),
        "parent_memory_dir": refine_cfg.get("parent_memory_dir"),
        "macro_average_precision": winner.get("macro_average_precision"),
        "macro_roc_auc": winner.get("macro_roc_auc"),
        "median_per_class_auc": winner.get("median_per_class_auc"),
        "spec": seed_spec,
    }
    (mem_dir / REFINE_CHAMPION_SPEC_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"  [Refine] Champion from {source}: {ranking_metric}={seed_score:.5f} "
        f"({locked})"
    )
    return seed_spec, seed_score, winner


def _perch_refine_batch_fallback(
    batch_size: int,
    *,
    timed_out: bool,
    locked_arch_type: str,
    seed_spec: dict,
) -> list[dict]:
    """Refine fallbacks — independent optimization tries near the current champion."""
    reason = (
        "Researcher LLM timed out; refine fallback."
        if timed_out
        else "Refine batch JSON parse failed; fallback."
    )
    base = _perch_fill_defaults(dict(seed_spec))
    base["arch_type"] = locked_arch_type
    base["strategy"] = "exploit"
    if not base.get("arch_description"):
        base["arch_description"] = _synthesize_arch_description(base)

    def _vary(label: str, **overrides) -> dict:
        s = dict(base)
        s.update(overrides)
        s["slot"] = label
        s["strategy"] = "exploit"
        s["arch_type"] = locked_arch_type
        s.setdefault("hypothesis", f"Optimize champion ({label}).")
        s["reasoning"] = reason + f" {label}."
        return s

    lr = float(base.get("learning_rate", 1e-3))
    drop = float(base.get("dropout", 0.3))
    pw = float(base.get("perch_weight", 0.2))
    nl = int(base.get("n_layers", 2))
    templates: list[dict] = [
        _vary("r1", learning_rate=max(lr * 0.5, 1e-5)),
        _vary("r2", dropout=min(0.5, drop + 0.1), n_layers=min(4, nl + 1)),
        _vary("r3", perch_weight=max(0.0, pw - 0.1)),
    ]
    labels = _refine_batch_slot_labels(batch_size)
    out: list[dict] = []
    for i in range(batch_size):
        s = dict(templates[i % len(templates)])
        s["slot"] = labels[i]
        out.append(s)
    return out


def _perch_batch_fallback(batch_size: int, *, timed_out: bool) -> list[dict]:
    """Diverse safe specs when batch JSON fails — one per slot."""
    reason = (
        "Researcher LLM timed out; slot fallback."
        if timed_out
        else "Batch JSON parse failed; slot fallback."
    )
    templates: list[tuple[str, dict]] = [
        (
            "tweak",
            {
                **_perch_safe_defaults(),
                "strategy": "exploit",
                "reasoning": reason + " Residual MLP tweak slot.",
            },
        ),
        (
            "explore",
            {
                **_perch_safe_defaults(),
                "arch_type": "gated_mlp",
                "arch_description": (
                    "Dense(1024) stem. Two GLU blocks with residual add and LayerNorm. "
                    "Dropout 0.3, Dense(n_classes, sigmoid)."
                ),
                "strategy": "explore",
                "reasoning": reason + " Gated explore slot.",
            },
        ),
        (
            "free",
            {
                **_perch_safe_defaults(),
                "arch_type": "attention_mlp",
                "arch_description": (
                    "Dense(1024) stem. One MHA block on reshaped (1, H), residual add. "
                    "Dropout 0.3, Dense(n_classes, sigmoid)."
                ),
                "strategy": "explore",
                "reasoning": reason + " Attention free slot.",
            },
        ),
    ]
    out: list[dict] = []
    for i in range(batch_size):
        slot, spec = templates[i % len(templates)]
        s = dict(spec)
        s["slot"] = slot
        s.setdefault("hypothesis", f"Fallback {slot} head.")
        out.append(s)
    return out


def _resolve_researcher_timeout_seconds(config: dict) -> float:
    """Researcher LLM timeout; stage 1c uses meta_agent.stage_1c.* — perch reads researcher/llm_researcher."""
    meta = config.get("meta_agent") or {}
    for key in ("researcher_timeout_seconds",):
        if meta.get(key) is not None:
            return float(meta[key])
    rc = config.get("researcher") or {}
    llm_rc = config.get("llm_researcher") or {}
    for block in (rc, llm_rc):
        if block.get("timeout_seconds") is not None:
            return float(block["timeout_seconds"])
    return 300.0


_PERCH_DEFAULT_ARCH_DESCRIPTION = (
    "BatchNorm on input. Dense(1024) projection with LayerNorm. "
    "Then 2 residual blocks: each block applies Dense(1024), LayerNorm, GELU, "
    "Dropout(0.3), Dense(1024), then adds the block input (skip connection), "
    "followed by LayerNorm. Final Dense(512, gelu), Dropout(0.4), Dense(n_classes, sigmoid)."
)

# Hyperparameter defaults only — never copy parse-failure reasoning into LLM-parsed specs.
_PERCH_SPEC_FIELD_DEFAULTS: dict = {
    "hidden_dim":       1024,
    "proj_dim":         512,
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
    "strategy":         "explore",
    "reasoning":        "",
    "hypothesis":       "",
}


def _perch_safe_defaults() -> dict:
    return {
        "arch_type":        "residual_mlp",
        "arch_description": _PERCH_DEFAULT_ARCH_DESCRIPTION,
        **_PERCH_SPEC_FIELD_DEFAULTS,
        "reasoning":        "Fallback defaults — researcher output could not be parsed.",
        "hypothesis":       "Baseline residual head config.",
        "strategy":         "explore",
    }


def _perch_fill_defaults(spec: dict) -> dict:
    """Fill missing hyperparameter keys; preserve arch_type/description/reasoning from the LLM."""
    out = dict(spec)
    for k, v in _PERCH_SPEC_FIELD_DEFAULTS.items():
        if k not in out or out[k] is None:
            out[k] = v
    if not out.get("arch_type"):
        out["arch_type"] = "residual_mlp"
    return out


def _synthesize_arch_description(spec: dict) -> str:
    """Minimal layout hint when the planner omits arch_description (better than wrong residual template)."""
    at = str(spec.get("arch_type") or "mlp")
    h = int(spec.get("hidden_dim") or 1024)
    nl = int(spec.get("n_layers") or 2)
    drop = spec.get("dropout", 0.3)
    act = spec.get("activation", "gelu")
    norm = spec.get("normalization", "layer_norm")
    if at == "linear_probe":
        return "Single Dense(num_classes, sigmoid) on raw embeddings."
    if at == "transformer_block":
        return (
            f"Dense({h}) stem from emb_dim. Reshape (1, {h}), one MultiHeadAttention + FFN + {norm}, "
            f"residual, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    if at == "mixture_of_experts":
        return (
            f"Dense({h}) stem. {nl} expert Dense({h}) branches, Concatenate, Dense({h}) compress, "
            f"residual Add, {norm}, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    if at == "dense_connections":
        return (
            f"Dense({h}) stem. {nl} dense blocks: Concatenate growth then Dense({h}) compress each step, "
            f"{act}, Dropout({drop}), Dense(num_classes, sigmoid)."
        )
    return (
        f"{at}: Dense({h}) stem from emb_dim. {nl} blocks with {act}, {norm}, Dropout({drop}), "
        f"Dense(num_classes, sigmoid). Follow standard shape-safe residual/gated patterns for this family."
    )


def _normalize_experiment_item(raw: dict) -> dict | None:
    if not isinstance(raw, dict) or not str(raw.get("arch_type") or "").strip():
        return None
    out = _perch_fill_defaults(dict(raw))
    desc = str(raw.get("arch_description") or "").strip()
    if desc:
        out["arch_description"] = desc
    else:
        out["arch_description"] = _synthesize_arch_description(out)
        out["_arch_description_synthesized"] = True
    return out


def _is_valid_experiment_spec(spec: dict) -> bool:
    return bool(str(spec.get("arch_type") or "").strip()) and bool(
        str(spec.get("arch_description") or "").strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coder — inner loop, writes TF/Keras head given spec
# ─────────────────────────────────────────────────────────────────────────────

PERCH_CODER_SYSTEM_PROMPT = (
    """You are a Python ML engineer building TF/Keras classification heads on frozen 1536-d Perch embeddings.

"""
    + PERCH_TASK_CONTEXT
    + """

The researcher proposes an experimental STRATEGY (architecture family, key hyperparameters, hypothesis).
arch_type may be any descriptive snake_case label (including names not in the example list); arch_description
is authoritative. YOU implement the exact Keras head in build_head — not a fixed template. Follow arch_description closely.
Advanced designs are encouraged when specified, especially multi_tower_ensemble (3–5 diverse parallel towers
on the same embedding, fuse per-tower sigmoid logits with Average or Concatenate→Dense(num_classes, sigmoid)).
Goal: accurate, well-calibrated per-window species probabilities for long jungle soundscape recordings.

You must output TWO functions in a single ```python``` code block:
  1. build_head(emb_dim, num_classes) → tf.keras.Model
  2. get_training_config() → dict with learning_rate, batch_size, optimizer, epochs, patience, perch_weight

HARD RULES (the harness depends on these):
- tf is already imported — do NOT add ANY imports
- No top-level code, no main(), no class definitions — only the two functions
- build_head MUST return tf.keras.Model(inp, out)
- The FINAL layer MUST be Dense(num_classes, activation="sigmoid")  (multi-label)
- Do NOT use tf.keras.layers.Lambda (breaks model save/load)
- Do NOT slice tensors with [:, a:b] (breaks the functional API) — use separate Dense layers instead

*** MANDATORY STEM (fixes 1536 vs 512 crashes) ***
emb_dim is 1536. Pick hidden_dim (e.g. 512) from the spec. IMMEDIATELY after Input:
  inp = tf.keras.layers.Input(shape=(emb_dim,))
  x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)   # REQUIRED — never Add() against raw inp
From this point on, the main trunk tensor x must stay shape (batch, hidden_dim) unless you explicitly project back.

*** CRITICAL — Add() / Multiply() SHAPE RULE ***
tf.keras.layers.Add()([a, b]) and Multiply()([a, b]) CRASH if shapes differ (e.g. (1536,) vs (512,), or (2048,) vs (512,)).
- Use ONE hidden_dim everywhere inside blocks; both Add() inputs must be (batch, hidden_dim).
- Project the skip path: skip = tf.keras.layers.Dense(hidden_dim)(skip) before Add().
- Add() takes exactly TWO same-shaped tensors — never Add()([x, a, b]) with three different branches unless all three are Dense(hidden_dim) first.
- After Concatenate, width grows — you CANNOT Add() the concat to x. Use Dense(hidden_dim) on the concat output first.

OTHER SHAPE GOTCHAS:
- MultiHeadAttention needs 3D: Reshape((1, hidden_dim)) → MHA → Reshape((hidden_dim,)).
- Concatenate(axis=-1) stacks widths (512+512=1024). Follow with Dense(hidden_dim) before any Add() with x.
- Do NOT Concatenate expert outputs then Multiply with a router — use the MoE pattern below.

REFERENCE PATTERNS (copy the shape logic; adapt layer counts):

# Stem (always)
inp = tf.keras.layers.Input(shape=(emb_dim,))
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)

# Residual block — x and h both hidden_dim
h = tf.keras.layers.Dense(hidden_dim)(x)
h = tf.keras.layers.LayerNormalization()(h)
h = tf.keras.layers.Activation("gelu")(h)
h = tf.keras.layers.Dropout(dropout)(h)
h = tf.keras.layers.Dense(hidden_dim)(h)
x = tf.keras.layers.Add()([x, h])

# Mixture of experts (K=4) — concat experts, compress (shape-safe; no router multiply bugs)
num_experts = 4
expert_outs = [tf.keras.layers.Dense(hidden_dim, activation="gelu")(x) for _ in range(num_experts)]
moe = tf.keras.layers.Concatenate()(expert_outs)              # (batch, K*hidden_dim)
moe = tf.keras.layers.Dense(hidden_dim, activation="gelu")(moe)  # learn mixture weights
x = tf.keras.layers.Add()([x, moe])

# DenseNet-style dense connection — concat then compress
dense_in = tf.keras.layers.Concatenate()([x, h])
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(dense_in)

# Multi-scale — concat branches then project
b1 = tf.keras.layers.Dense(256, activation="gelu")(x)
b2 = tf.keras.layers.Dense(512, activation="gelu")(x)
merged = tf.keras.layers.Concatenate()([b1, b2])
x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(merged)

# Gated (GLU) block
v = tf.keras.layers.Dense(hidden_dim, activation="linear")(x)
g = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")(x)
gated = tf.keras.layers.Multiply()([v, g])
x = tf.keras.layers.Add()([x, gated])

# Attention block
x_3d = tf.keras.layers.Reshape((1, hidden_dim))(x)
attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=hidden_dim // 4)(x_3d, x_3d)
attn = tf.keras.layers.Reshape((hidden_dim,))(attn)
x = tf.keras.layers.Add()([x, attn])

# Classifier head (after trunk is stable)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

# Multi-tower ensemble (e.g. 5 towers) — shape-safe; each tower: stem branch → Dense(num_classes, sigmoid)
inp = tf.keras.layers.Input(shape=(emb_dim,))
stem = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)
def _tower_residual(x):
    h = tf.keras.layers.Dense(hidden_dim, activation="gelu")(x)
    h = tf.keras.layers.Dense(hidden_dim)(h)
    return tf.keras.layers.Add()([x, h])
def _tower_gated(x):
    v = tf.keras.layers.Dense(hidden_dim, activation="linear")(x)
    g = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")(x)
    return tf.keras.layers.Multiply()([v, g])
t1 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(_tower_residual(stem))
t2 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(_tower_gated(stem))
t3 = tf.keras.layers.Dense(num_classes, activation="sigmoid")(stem)  # linear-ish tower
# ... add t4, t5 with attention / bottleneck as needed ...
out = tf.keras.layers.Average()([t1, t2, t3])  # all (batch, num_classes) — OK

For mixture_of_experts: K experts → Concatenate → Dense(hidden_dim) → Add with x (do NOT Multiply softmax router against concat experts).
For dense_connections: Concatenate([x, h]) → Dense(hidden_dim) replaces x (do NOT Add concat to x).
For multi_tower_ensemble: each tower must output (batch, num_classes) before Average/Concat fusion; never Add() towers at hidden_dim then one shared classifier unless you design it that way explicitly.

Keep build_head shape-safe. Prefer a working simpler model over a broken exotic one."""
)


def _spec_to_coder_prompt(spec: dict) -> str:
    arch_type        = spec.get("arch_type", "residual_mlp")
    arch_description = spec.get("arch_description", "(no description provided)")
    hypothesis       = spec.get("hypothesis", "")
    reasoning        = spec.get("reasoning", "")
    strategy         = spec.get("strategy", "explore")

    arch_keys = ("hidden_dim", "proj_dim", "n_layers", "dropout", "activation", "normalization")
    arch_cfg  = {k: spec[k] for k in arch_keys if k in spec}

    training_keys = ("learning_rate", "batch_size", "optimizer", "epochs", "patience", "perch_weight")
    training_cfg  = {k: spec[k] for k in training_keys if k in spec}

    return (
        f"Researcher's experimental proposal:\n"
        f"  arch_type:  {arch_type}\n"
        f"  strategy:   {strategy}\n"
        f"  hypothesis: {hypothesis}\n"
        f"  reasoning:  {reasoning}\n\n"
        f"Architecture description (use as design guidance — implement faithfully but make your own concrete choices):\n"
        f"  {arch_description}\n\n"
        f"Suggested architecture hyperparameters (these are HINTS — adapt as needed for shape safety):\n"
        f"{json.dumps(arch_cfg, indent=2)}\n\n"
        f"Training config to return from get_training_config() (use these values verbatim unless they would break training):\n"
        f"{json.dumps(training_cfg, indent=2)}\n\n"
        f"{_arch_type_hint(arch_type)}"
        f"Implement build_head(emb_dim, num_classes) freely in idiomatic Keras functional API, "
        f"obeying all HARD RULES, the MANDATORY STEM, and Add() SHAPE RULE. "
        f"Return BOTH functions in a single ```python``` code block."
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


def _dry_run_build_head(
    code: str, emb_dim: int = 1536, num_classes: int = 234
) -> list[str]:
    """Instantiate build_head with dummy shapes — catches Add/Concat bugs in seconds."""
    import tensorflow as tf

    namespace: dict = {"tf": tf}
    try:
        exec(code, namespace)  # noqa: S102 — trusted coder slot in isolated process
    except Exception as e:
        return [f"exec failed: {type(e).__name__}: {e}"]
    build_head = namespace.get("build_head")
    if build_head is None:
        return ["build_head not defined after exec"]
    try:
        model = build_head(emb_dim, num_classes)
        if not isinstance(model, tf.keras.Model):
            return ["build_head must return tf.keras.Model(inp, out)"]
        model(tf.zeros((2, emb_dim), dtype=tf.float32), training=False)
    except Exception as e:
        return [f"build_head shape error: {type(e).__name__}: {e}"]
    return []


_ARCH_TYPE_SHAPE_HINTS: dict[str, str] = {
    "mixture_of_experts": (
        "MoE: K parallel Dense(hidden_dim)(x), Concatenate expert outputs, "
        "Dense(hidden_dim) to compress, Add with x. Do NOT softmax-multiply router against concat."
    ),
    "dense_connections": (
        "DenseNet: h = Dense(hidden_dim)(x); dense_in = Concatenate([x, h]); "
        "x = Dense(hidden_dim)(dense_in). Never Add() concat tensor to x directly."
    ),
    "multi_scale_mlp": (
        "Parallel Dense branches → Concatenate → Dense(hidden_dim) before Add with trunk."
    ),
    "bottleneck_mlp": (
        "Dense(wide)→Dense(narrow)→Dense(hidden_dim) before any Add() with x; x must stay hidden_dim."
    ),
    "residual_mlp": (
        "Stem Dense(hidden_dim)(inp) first; each block ends with Dense(hidden_dim) before Add([x, h])."
    ),
    "attention_mlp": "Project to hidden_dim, Reshape (1, H) for MHA, Reshape (H,) back, Add with x.",
    "transformer_block": "Stem to hidden_dim; MHA on (batch, 1, hidden_dim).",
    "multi_tower_ensemble": (
        "Shared stem Dense(hidden_dim)(inp). Build 3–5 parallel towers (different topologies). "
        "Each tower ends with Dense(num_classes, sigmoid). Fuse with Average([t1,t2,...]) or "
        "Concatenate(towers)→Dense(num_classes, sigmoid). All tower outputs must be (batch, num_classes)."
    ),
}


def _arch_type_hint(arch_type: str) -> str:
    hint = _ARCH_TYPE_SHAPE_HINTS.get(arch_type)
    if not hint:
        return ""
    return f"\nShape hint for {arch_type}: {hint}\n"


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

        if _llm_response_failed(response):
            print(f"  [Coder] LLM error: {str(response)[:150]}")
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
            shape_issues = _dry_run_build_head(code)
            if shape_issues:
                issues = shape_issues
        if not issues:
            print("  [Coder] Code valid (shape check passed).")
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
        shape_fix = ""
        if any("shape" in i.lower() or "incompatible" in i.lower() for i in issues):
            shape_fix = (
                "\n\nSHAPE FIX CHECKLIST:\n"
                "1. First layer after Input: Dense(hidden_dim)(inp) — emb_dim is 1536.\n"
                "2. Every Add()/Multiply(): both inputs are (batch, hidden_dim).\n"
                "3. After Concatenate: Dense(hidden_dim) before Add with x.\n"
            )
        current_prompt = (
            "Your code had issues:\n" + "\n".join(f"- {i}" for i in issues) +
            shape_fix +
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
        arch_type = spec.get("arch_type", "")
        arch_extra = _arch_type_hint(arch_type)
        shape_hint = (
            "\n*** SHAPE MISMATCH DETECTED — THIS IS THE SPECIFIC BUG IN YOUR CODE ***\n"
            "Add() or Multiply() crashed because two tensors have different feature sizes.\n"
            "Common fixes:\n"
            "  - Missing stem: x = Dense(hidden_dim)(inp) right after Input — inp is 1536-d.\n"
            "  - Residual without projection: Add([x, h]) where x is 1536 and h is 512.\n"
            "  - MoE bug: Concatenate experts (2048) then Multiply with router (512) — use Concat→Dense(hidden_dim) instead.\n"
            "  - DenseNet bug: Add([x, concat]) where concat is wider than x — use Dense(hidden_dim) on concat.\n"
            "THE FIX: Before every Add(), both tensors must be Dense(hidden_dim):\n"
            "    x = tf.keras.layers.Dense(hidden_dim)(inp)   # stem first\n"
            "    h = ... ; h = tf.keras.layers.Dense(hidden_dim)(h)\n"
            "    x = tf.keras.layers.Add()([x, h])\n"
            f"{arch_extra}"
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
    if not issues:
        shape_issues = _dry_run_build_head(code)
        if shape_issues:
            issues = shape_issues
    if issues:
        print(f"  [Coder] Repair still has issues: {issues}")
        return None
    print("  [Coder] Repair code validated (shape check passed).")
    return code


# ─────────────────────────────────────────────────────────────────────────────
# Script harness — wraps Coder's slot code into a runnable training script
# ─────────────────────────────────────────────────────────────────────────────

def _harness_subsample_block(
    head_train_max_samples: int | None,
    head_train_indices_path: Path | None = None,
) -> str:
    if head_train_indices_path is not None:
        return f"""
_HEAD_TRAIN_IDX = np.load(r"{head_train_indices_path}")
_CACHE_N_FULL = len(X_train)
X_train, S_train, y_train = X_train[_HEAD_TRAIN_IDX], S_train[_HEAD_TRAIN_IDX], y_train[_HEAD_TRAIN_IDX]
print(f"  Head training subset: {{len(X_train)}} / {{_CACHE_N_FULL}} cached embeddings (fixed indices)")
"""
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
    head_train_indices_path: Path | None = None,
) -> str:
    sub = _harness_subsample_block(head_train_max_samples, head_train_indices_path)
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
    head_train_indices_path: Path | None = None,
) -> str:
    prefix = _build_harness_prefix(
        train_cache, val_cache, head_train_max_samples, head_train_indices_path
    )
    return prefix + "\n\n" + slot_code + "\n\n" + HARNESS_SUFFIX


# ─────────────────────────────────────────────────────────────────────────────
# Final retrain on full data (train + val combined)
# ─────────────────────────────────────────────────────────────────────────────

def _build_final_retrain_script(
    best_code: str,
    mem_dir: Path,
    train_cache: Path,
    val_cache: Path,
) -> str:
    """Build final retrain script using the best iteration's coder-generated build_head + get_training_config."""
    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_MEM_DIR   = Path(r"{mem_dir}")

def _load(p):
    d = np.load(str(p), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)

X_tr, y_tr = _load(Path(r"{train_cache}"))
X_vl, y_vl = _load(Path(r"{val_cache}"))

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


def _build_pseudo_refine_script(
    best_code: str,
    mem_dir: Path,
    train_cache: Path,
    pseudo_cache: Path,
    *,
    val_cache: Path | None = None,
    init_weights: Path | None = None,
    fine_tune_lr: float = 2e-4,
    epochs: int = 15,
    val_split: float = 0.1,
    sample_weight_supervised: float = 1.0,
    sample_weight_labeled_val: float = 1.0,
    sample_weight_pseudo: float = 0.5,
) -> str:
    """Fine-tune the 1d head on focal train + optional labeled val + pseudo windows."""
    init_block = ""
    if init_weights is not None and Path(init_weights).exists():
        init_block = f"""
_wpath = Path(r"{init_weights}")
if _wpath.suffix == ".keras" and _wpath.exists():
    head = tf.keras.models.load_model(str(_wpath), compile=False)
    print(f"  Warm-start from {{_wpath.name}}")
else:
    head = build_head(EMB_DIM, N_CLASSES)
    if _wpath.exists():
        head.load_weights(str(_wpath))
        print(f"  Warm-start weights from {{_wpath.name}}")
"""
    else:
        init_block = "\nhead = build_head(EMB_DIM, N_CLASSES)\n"

    val_block = ""
    if val_cache is not None and Path(val_cache).exists():
        val_block = f"""
X_lv, y_lv = _load_focal(Path(r"{val_cache}"))
X_parts.append(X_lv)
y_parts.append(y_lv)
sw_parts.append(np.full(len(X_lv), {sample_weight_labeled_val}, dtype=np.float32))
n_labeled_val = len(X_lv)
"""

    return f"""
import numpy as np
import tensorflow as tf
from pathlib import Path

_MEM_DIR = Path(r"{mem_dir}")

def _load_focal(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    if "X" in d.files:
        X = d["X"]
    elif "X_train" in d.files:
        X = d["X_train"]
    else:
        raise KeyError(f"No X or X_train in {{npz_path}}")
    if "y" in d.files:
        y = d["y"]
    elif "y_train" in d.files:
        y = d["y_train"]
    else:
        raise KeyError(f"No y or y_train in {{npz_path}}")
    return X.astype(np.float32), y.astype(np.float32)

def _load_pseudo(npz_path):
    d = np.load(str(npz_path), allow_pickle=True)
    if "X_pseudo" not in d.files or "y_pseudo" not in d.files:
        raise KeyError(
            f"Expected X_pseudo and y_pseudo in {{npz_path}}, got {{list(d.files)}}"
        )
    return d["X_pseudo"].astype(np.float32), d["y_pseudo"].astype(np.float32)

X_parts, y_parts, sw_parts = [], [], []
n_focal = n_labeled_val = 0

X_tr, y_tr = _load_focal(Path(r"{train_cache}"))
X_parts.append(X_tr)
y_parts.append(y_tr)
sw_parts.append(np.full(len(X_tr), {sample_weight_supervised}, dtype=np.float32))
n_focal = len(X_tr)
{val_block}
Xp, yp = _load_pseudo(Path(r"{pseudo_cache}"))
if len(Xp) > 0:
    X_parts.append(Xp)
    y_parts.append(yp)
    sw_parts.append(np.full(len(Xp), {sample_weight_pseudo}, dtype=np.float32))
    n_pseudo = len(Xp)
else:
    n_pseudo = 0
    print("  [1e] No pseudo windows in cache — fine-tuning on supervised (+ val) only")

X_all = np.concatenate(X_parts, axis=0)
y_all = np.concatenate(y_parts, axis=0)
sw_all = np.concatenate(sw_parts, axis=0)

EMB_DIM   = X_all.shape[1]
N_CLASSES = y_all.shape[1]
print(
    f"  Pseudo refine: focal={{n_focal}}  labeled_val={{n_labeled_val}}  "
    f"pseudo={{n_pseudo}}  total={{X_all.shape[0]}}"
)

# --- Locked head architecture (from stage 1b / 1d) ---
{best_code}
# ---------------------------------------------------------------------------
{init_block}

pos = y_all.sum(axis=0).astype(np.float64)
neg = len(y_all) - pos
pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 25.0).astype(np.float32)
pw = tf.constant(pos_weight)[tf.newaxis, :]

def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(
        pw * y_true * (-tf.math.log(y_pred))
        + (1.0 - y_true) * (-tf.math.log(1.0 - y_pred))
    )

lr = {fine_tune_lr}
batch_size = int(get_training_config().get("batch_size", 256))
epochs = {epochs}
val_split = {val_split}
opt_name = str(get_training_config().get("optimizer", "adam"))

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
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_loss", verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]
head.fit(
    X_all, y_all,
    sample_weight=sw_all,
    validation_split=val_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1,
)

head.save(str(_MEM_DIR / "final_head_pseudo.keras"))
head.save_weights(str(_MEM_DIR / "final_head_pseudo.weights.h5"))
print("PSEUDO_REFINE_DONE")
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_quiet(config: dict) -> bool:
    return bool(
        config.get("perch_quiet")
        or (config.get("perch") or {}).get("quiet_trial")
    )


def _plog(config: dict, msg: str, *, force: bool = False) -> None:
    if force or not _cfg_quiet(config):
        print(msg, flush=True)


def _copy_perch_mapping_artifacts(src_mem: Path, dst_mem: Path) -> None:
    import shutil

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


def run_1c_aug_sweep(
    config: dict,
    *,
    sess,
    inp_name,
    emb_idx,
    logit_idx,
    MAPPED_POS,
    MAPPED_BC_IDX,
    proxy_map,
    NO_LABEL,
    train_df,
    paths,
    species_to_idx: dict,
    n_species: int,
    embed_bs: int,
    sample_frac: float,
    max_samples: int | None,
) -> None:
    """
    Stage-1c batch: load ONNX once, run many aug presets (embed + fixed head train).
    """
    sweep = config.get("perch_1c_sweep") or {}
    trials: list[dict] = list(sweep.get("trials") or [])
    if not trials:
        return

    perch_paths = config.get("perch", {})
    val_cache = Path(
        perch_paths.get("val_cache_path") or (Path(perch_paths.get("cache_dir", "")) / "val_emb.npz")
    )
    quiet = _cfg_quiet(config)
    n = len(trials)
    _plog(config, f"\n  [1c sweep] {n} augmentation trial(s) — single ONNX session", force=True)

    if not val_cache.exists():
        _plog(config, "  [1c sweep] Building shared val cache…", force=True)
        ok = _build_val_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            paths.train_soundscapes_dir, paths.train_soundscapes_labels_csv,
            species_to_idx, n_species, val_cache, embed_bs,
        )
        if not ok:
            _build_focal_val_fallback(
                Path(trials[0].get("train_cache", val_cache.parent / "train_emb.npz")),
                val_cache,
            )

    clip_subset_path = config.get("perch_embed_clip_subset")
    clip_rows = None
    if clip_subset_path:
        clip_rows = _load_clip_rows_jsonl(Path(clip_subset_path))

    mapping_src = Path(sweep.get("mapping_src", ""))
    for i, trial in enumerate(trials, 1):
        preset = str(trial["preset"])
        train_cache = Path(trial["train_cache"])
        mem_dir = Path(trial["memory_dir"])
        code_dir = Path(trial["code_dir"])
        mem_dir.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)
        if mapping_src.is_dir():
            _copy_perch_mapping_artifacts(mapping_src, mem_dir)

        aug_config = trial.get("augmentation") or config.get("augmentation")
        force_rebuild = bool(trial.get("force_rebuild_train", False))
        train_meta_path = _embedding_cache_meta_path(train_cache)
        clips_sidecar = _embedding_cache_clips_path(train_cache)
        if force_rebuild:
            for p in (train_cache, train_meta_path, clips_sidecar):
                if p.exists():
                    p.unlink()

        if not train_cache.exists():
            _plog(
                config,
                f"  [1c sweep {i}/{n}] Embedding train cache ({preset})…",
                force=True,
            )
            _build_train_cache(
                sess, inp_name, emb_idx, logit_idx,
                MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
                train_df, species_to_idx, paths.train_audio_dir, n_species,
                train_cache,
                sample_frac=1.0 if clip_rows else sample_frac,
                max_samples=None if clip_rows else max_samples,
                batch_size=embed_bs,
                aug_config=aug_config,
                soundscapes_dir=paths.train_soundscapes_dir,
                aug_preset=preset,
                clip_rows=clip_rows,
            )
        elif not quiet:
            _plog(config, f"  [1c sweep {i}/{n}] Reuse train cache → {train_cache.name}")

        fixed_cfg = dict(trial.get("perch_fixed_train") or {})
        fixed_cfg["enabled"] = True
        if not train_cache.exists():
            _plog(config, f"  [1c sweep {i}/{n}] {preset} — cache build failed", force=True)
            continue

        metrics = run_fixed_head_train(
            config,
            fixed_cfg,
            train_cache=train_cache,
            val_cache=val_cache,
            mem_dir=mem_dir,
            code_dir=code_dir,
        )
        if quiet and metrics:
            from soundscape_evaluator import PRIMARY_META_METRIC

            ap = metrics.get("macro_average_precision")
            _plog(
                config,
                f"  [1c sweep {i}/{n}] {preset} — head train done "
                f"({PRIMARY_META_METRIC}={float(ap):.5f})" if ap is not None else
                f"  [1c sweep {i}/{n}] {preset} — head train done",
                force=True,
            )


def run_fixed_head_train(
    config: dict,
    fixed_cfg: dict,
    *,
    train_cache: Path,
    val_cache: Path,
    mem_dir: Path,
    code_dir: Path,
) -> dict | None:
    """Train a locked head on cached embeddings (stage 1c aug search). No LLM loop."""
    head_path = Path(fixed_cfg["head_code_path"])
    if not head_path.exists():
        raise FileNotFoundError(f"Fixed head code not found: {head_path}")

    head_code = head_path.read_text(encoding="utf-8")
    indices_path = fixed_cfg.get("head_train_indices_path")
    if indices_path:
        indices_path = Path(indices_path)
        if not indices_path.exists():
            raise FileNotFoundError(f"Head train indices not found: {indices_path}")
    else:
        indices_path = None

    py_exe = config.get("execution", {}).get("python_executable", "python3")
    timeout = config.get("execution", {}).get("timeout_seconds", 1800)
    ranking_metric = _ranking_metric_from_config(config)
    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator = Evaluator(row_id_column_name="row_id")

    label = fixed_cfg.get("label", "fixed_head_train")
    quiet = _cfg_quiet(config)
    if not quiet:
        print(f"\n  [Fixed train] {label}")
        print(f"  train cache → {train_cache.name}")
        if indices_path:
            print(f"  head indices → {indices_path.name}")

    script = _build_script(
        head_code,
        train_cache,
        val_cache,
        head_train_indices_path=indices_path,
    )
    code_dir.mkdir(parents=True, exist_ok=True)
    script_path = code_dir / f"{label.replace('/', '_')[:60]}.py"
    script_path.write_text(script, encoding="utf-8")

    y_true_path = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path = Path(tempfile.gettempdir()) / "_y_pred.npy"
    trained_head_path = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path = mem_dir / "best_head.keras"

    result = executor.run_file(script_path)
    metrics = None
    if result.success and "EVAL_ARTIFACTS_SAVED" in (result.stdout or ""):
        if y_true_path.exists() and y_pred_path.exists():
            summary = evaluator.evaluate_from_files(y_true_path, y_pred_path)
            metrics = summary.metrics

    rank_val = _ranking_value_from_metrics(metrics)
    if not quiet:
        print(f"  [Fixed train] {_format_iteration_metrics(metrics)}")

    spec = dict(fixed_cfg.get("spec") or {})
    spec.setdefault("mode", "fixed_head_train")
    spec["aug_preset"] = fixed_cfg.get("aug_preset")

    if rank_val is not None:
        prior_best = -1.0
        info_path = mem_dir / "best_model_info.json"
        if info_path.exists():
            try:
                prev = json.loads(info_path.read_text(encoding="utf-8"))
                prior_best = float(
                    prev.get("ranking_value")
                    or prev.get("macro_average_precision", -1.0)
                )
            except (json.JSONDecodeError, TypeError, ValueError, OSError):
                prior_best = -1.0
        promotion_quiet = quiet or bool(fixed_cfg.get("fixed_1c_trial"))
        _promote_best_head(
            rank_val=rank_val,
            metrics=metrics,
            best_score_ever=prior_best,
            iteration=0,
            spec=spec,
            slot_code=head_code,
            trained_head_path=trained_head_path,
            best_head_path=best_head_path,
            mem_dir=mem_dir,
            ranking_metric=ranking_metric,
            quiet=promotion_quiet,
        )
    elif not result.success and not quiet:
        print(f"  [Fixed train] failed: {(result.stderr or '')[-600:]}")

    return metrics


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


def _promote_best_head(
    *,
    rank_val: float | None,
    metrics: dict | None,
    best_score_ever: float,
    iteration: int,
    spec: dict,
    slot_code: str,
    trained_head_path: Path,
    best_head_path: Path,
    mem_dir: Path,
    ranking_metric: str,
    quiet: bool = False,
) -> float:
    """Save head artifacts when a new best ranking score is achieved."""
    if rank_val is None or rank_val <= best_score_ever:
        return best_score_ever
    import shutil

    auc = metrics.get("macro_roc_auc") if metrics else None
    ap = metrics.get("macro_average_precision") if metrics else None
    med = metrics.get("median_per_class_auc") if metrics else None
    if trained_head_path.exists():
        shutil.copy2(str(trained_head_path), str(best_head_path))
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
        _y_pred_tmp = Path(tempfile.gettempdir()) / "_y_pred.npy"
        _y_true_tmp = Path(tempfile.gettempdir()) / "_y_true.npy"
        if _y_pred_tmp.exists():
            shutil.copy2(str(_y_pred_tmp), str(mem_dir / "best_val_preds.npy"))
        if _y_true_tmp.exists():
            shutil.copy2(str(_y_true_tmp), str(mem_dir / "y_val.npy"))
        (mem_dir / "best_head_code.py").write_text(slot_code, encoding="utf-8")
        if not quiet:
            print(
                f"  [Best] NEW BEST {_format_iteration_metrics(metrics)} "
                f"— head saved to {best_head_path.name}"
            )
    return rank_val


def _execute_perch_iteration(
    iteration: int,
    *,
    researcher: PerchResearcher,
    coder_llm: LLMClient,
    coder_temp: float,
    executor: CodeExecutor,
    evaluator: Evaluator,
    memory: PerchExperimentMemory,
    train_cache: Path,
    val_cache: Path,
    head_train_max: int | None,
    code_dir: Path,
    spec: dict | None = None,
    slot_suffix: str = "",
) -> tuple[dict | None, str, dict | None]:
    """Coder→train for one spec (researcher may have planned several per round)."""
    if spec is None:
        spec = researcher.next_experiment()
    slot_code = generate_perch_code(coder_llm, spec, coder_temp)
    if slot_code is None:
        print("  [Coder] All AST retries exhausted — falling back to safe default residual MLP.")
        slot_code = _SAFE_DEFAULT_SLOT_CODE

    _MAX_EXEC_ATTEMPTS = 5
    metrics = None
    final_slot_code = slot_code
    y_true_path = Path(tempfile.gettempdir()) / "_y_true.npy"
    y_pred_path = Path(tempfile.gettempdir()) / "_y_pred.npy"

    for exec_attempt in range(1, _MAX_EXEC_ATTEMPTS + 1):
        script = _build_script(
            final_slot_code, train_cache, val_cache, head_train_max_samples=head_train_max
        )
        slot_tag = f"_{_slug_slot_suffix(slot_suffix)}" if slot_suffix else ""
        script_path = code_dir / f"iter_{iteration:03d}{slot_tag}_a{exec_attempt}.py"
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
            print(f"  [Error]  {error_msg[-600:]}")

        if exec_attempt == _MAX_EXEC_ATTEMPTS:
            print(f"  [Coder] All {_MAX_EXEC_ATTEMPTS} execution attempts exhausted — skipping slot.")
            break

        repaired = _repair_perch_code(coder_llm, spec, final_slot_code, error_msg, coder_temp)
        if repaired is None:
            print("  [Coder] Repair failed — skipping remaining attempts.")
            break
        final_slot_code = repaired

    return metrics, final_slot_code, spec


def _slug_slot_suffix(slot: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(slot).strip().lower()).strip("_")
    return (s[:24] or "slot")


def _promote_if_best(
    *,
    rank_val: float | None,
    metrics: dict | None,
    spec: dict,
    iteration: int,
    slot_label: str,
    best_score_ever: float,
    trained_head_path: Path,
    best_head_path: Path,
    mem_dir: Path,
    code_dir: Path,
    slot_code: str,
    ranking_metric: str,
) -> float:
    if rank_val is None or rank_val <= best_score_ever:
        return best_score_ever
    import shutil

    best_score_ever = rank_val
    auc = metrics.get("macro_roc_auc") if metrics else None
    ap = metrics.get("macro_average_precision") if metrics else None
    med = metrics.get("median_per_class_auc") if metrics else None
    if trained_head_path.exists():
        shutil.copy2(str(trained_head_path), str(best_head_path))
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
                "slot": slot_label,
                "spec": spec,
            }, _f, indent=2)
        _y_pred_tmp = Path(tempfile.gettempdir()) / "_y_pred.npy"
        _y_true_tmp = Path(tempfile.gettempdir()) / "_y_true.npy"
        if _y_pred_tmp.exists():
            shutil.copy2(str(_y_pred_tmp), str(mem_dir / "best_val_preds.npy"))
        if _y_true_tmp.exists():
            shutil.copy2(str(_y_true_tmp), str(mem_dir / "y_val.npy"))
        (mem_dir / "best_head_code.py").write_text(slot_code, encoding="utf-8")
        print(
            f"  [Best] NEW BEST {_format_iteration_metrics(metrics)} "
            f"(slot={slot_label}) — head saved to {best_head_path.name}"
        )
    return best_score_ever


def _run_refine_loop(
    config: dict,
    refine_cfg: dict,
    *,
    researcher: PerchResearcher,
    coder_llm: LLMClient,
    coder_temp: float,
    executor: CodeExecutor,
    evaluator: Evaluator,
    memory: PerchExperimentMemory,
    train_cache: Path,
    val_cache: Path,
    head_train_max: int | None,
    code_dir: Path,
    mem_dir: Path,
    ranking_metric: str,
    experiments_per_round: int = 1,
) -> None:
    """Adaptive exploit loop: initial training tries, +bonus on improve, hard cap."""
    initial = max(1, int(refine_cfg.get("initial_iterations", 5)))
    bonus = max(1, int(refine_cfg.get("bonus_iterations_on_improve", 5)))
    max_total = max(initial, int(refine_cfg.get("max_iterations_per_model", 25)))
    seed_score = float(refine_cfg.get("seed_score", -1.0))
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    refine_cfg = dict(refine_cfg)
    refine_cfg["ranking_metric"] = ranking_metric
    champion_spec, seed_score, winner = _persist_refine_champion_artifacts(
        mem_dir, refine_cfg, ranking_metric=ranking_metric
    )
    refine_cfg["seed_spec"] = champion_spec
    refine_cfg["seed_score"] = seed_score
    refine_cfg["seed_macro_ap"] = winner.get("macro_average_precision")
    refine_cfg["seed_macro_auc"] = winner.get("macro_roc_auc")
    refine_cfg["seed_median_auc"] = winner.get("median_per_class_auc")
    researcher.seed_spec = champion_spec
    researcher.seed_score = seed_score
    researcher.locked_arch_type = str(locked)

    parent_dir = refine_cfg.get("parent_memory_dir")
    if parent_dir and memory.total() == 0:
        memory.seed_from_stage_1a(
            Path(parent_dir),
            arch_type=str(locked),
            aug_baseline=str(refine_cfg.get("aug_baseline", "?")),
            seed_score=seed_score,
            seed_spec=champion_spec,
        )
    memory.sync_refine_champion(
        arch_type=str(locked),
        aug_baseline=str(refine_cfg.get("aug_baseline", "?")),
        seed_score=seed_score,
        seed_spec=champion_spec,
        parent_memory_dir=str(parent_dir) if parent_dir else None,
        champion_source=str(winner.get("source", "?")),
    )

    seed_spec = champion_spec
    seed_ap = refine_cfg.get("seed_macro_ap")
    if seed_ap is None:
        seed_ap = seed_spec.get("macro_average_precision") or refine_cfg.get(
            "macro_average_precision"
        )
    seed_auc = refine_cfg.get("seed_macro_auc")
    if seed_auc is None:
        seed_auc = refine_cfg.get("macro_roc_auc")
    seed_med = refine_cfg.get("seed_median_auc")
    if seed_med is None:
        seed_med = refine_cfg.get("median_per_class_auc")

    print(f"\n  REFINE CAMPAIGN — locked arch_type: {locked}")
    print(f"  Aug baseline: {refine_cfg.get('aug_baseline', '?')}")
    seed_metrics = {
        "status": "success",
        "macro_average_precision": seed_ap,
        "macro_roc_auc": seed_auc,
        "median_per_class_auc": seed_med,
        "ranking_metric": ranking_metric,
    }
    print(f"  Champion to beat: {format_metrics_dict(seed_metrics, ranking_metric=ranking_metric)}")
    epr_note = (
        f", {experiments_per_round} experiments per planner call"
        if experiments_per_round > 1
        else ""
    )
    print(
        f"  Budget: {initial} initial training tries{epr_note}, +{bonus} on each improvement, "
        f"max {max_total} total training runs"
    )
    print(f"  Champion spec → {mem_dir / REFINE_CHAMPION_SPEC_FILE}")

    best_score_ever = max(seed_score, -1.0)
    _prior = memory.best_runs(1)
    if _prior:
        best_score_ever = max(best_score_ever, memory._ranking_value(_prior[0]))

    tries_left = initial
    total_done = 0
    run_index = 0
    planner_round = 0
    trained_head_path = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path = mem_dir / "best_head.keras"

    while total_done < max_total and tries_left > 0:
        planner_round += 1
        print(f"\n{'─'*60}")
        if experiments_per_round > 1:
            print(
                f"  REFINE PLANNER ROUND {planner_round}  "
                f"(training runs {total_done}/{max_total}, queue {tries_left})"
            )
        else:
            print(
                f"  REFINE {planner_round}  (done {total_done}/{max_total}, "
                f"queue {tries_left} remaining)"
            )
        print(f"{'─'*60}")

        specs = researcher.next_experiments()
        for slot_i, spec in enumerate(specs, 1):
            if tries_left <= 0 or total_done >= max_total:
                break

            run_index += 1
            total_done += 1
            tries_left -= 1
            slot_label = str(spec.get("slot") or f"s{slot_i}")
            if experiments_per_round > 1:
                print(f"\n  ▸ Try {slot_i}/{len(specs)}: {slot_label}")

            metrics, slot_code, spec = _execute_perch_iteration(
                run_index,
                spec=spec,
                slot_suffix=slot_label,
                researcher=researcher,
                coder_llm=coder_llm,
                coder_temp=coder_temp,
                executor=executor,
                evaluator=evaluator,
                memory=memory,
                train_cache=train_cache,
                val_cache=val_cache,
                head_train_max=head_train_max,
                code_dir=code_dir,
            )

            rank_val = _ranking_value_from_metrics(metrics)
            print(f"  [Result] [{slot_label}] {_format_iteration_metrics(metrics)}")
            memory.log(spec=spec, metrics=metrics, code=slot_code)

            if rank_val is not None and rank_val > best_score_ever:
                prev = best_score_ever
                best_score_ever = _promote_best_head(
                    rank_val=rank_val,
                    metrics=metrics,
                    best_score_ever=best_score_ever,
                    iteration=run_index,
                    spec=spec,
                    slot_code=slot_code,
                    trained_head_path=trained_head_path,
                    best_head_path=best_head_path,
                    mem_dir=mem_dir,
                    ranking_metric=ranking_metric,
                )
                bonus_add = min(bonus, max_total - total_done)
                tries_left += bonus_add
                print(
                    f"  [Refine] Improved {ranking_metric} {prev:.5f} → {best_score_ever:.5f} "
                    f"| +{bonus_add} bonus tries (queue={tries_left})"
                )
            elif rank_val is not None:
                print(
                    f"  [Refine] No improvement (best {ranking_metric}={best_score_ever:.5f})"
                )

        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] {memory._format_run_score(best[0])}")

    print(
        f"\n  [Refine] Finished: {total_done} training runs, "
        f"best {ranking_metric}={best_score_ever:.5f}"
    )


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
        _plog(config, f"  Meta aug baseline: {preset}")

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

    researcher_model = config.get("researcher", {}).get("model",       "gemma3:4b")
    coder_model      = config.get("llm",        {}).get("model",       "deepseek-r1:8b")
    provider         = config.get("llm",        {}).get("provider",    "ollama")
    researcher_temp  = config.get("researcher", {}).get("temperature", 0.6)
    researcher_timeout = _resolve_researcher_timeout_seconds(config)
    researcher_stream = bool(
        config.get("researcher", {}).get("stream_debug")
        or config.get("llm_researcher", {}).get("stream_debug")
    )
    refine_cfg_early = config.get("perch_refine") or {}
    refine_enabled_early = bool(refine_cfg_early.get("enabled"))
    experiments_per_round = _resolve_experiments_per_round(
        config, refine_mode=refine_enabled_early
    )
    coder_temp       = config.get("llm",        {}).get("temperature", 0.2)
    py_exe           = config.get("execution",  {}).get("python_executable", "python3")
    timeout          = config.get("execution",  {}).get("timeout_seconds",   1800)

    if not _cfg_quiet(config):
        print("=" * 60)
        print("  BirdCLEF Perch Agent — Researcher / Coder Architecture")
        print("=" * 60)

    eda_brief = (config.get("eda_brief") or "").strip()
    if eda_brief:
        global PERCH_RESEARCHER_SYSTEM_PROMPT, PERCH_CODER_SYSTEM_PROMPT
        _eda_block = (
            "\n\n## DATA INSIGHTS (EDA — factual, data only)\n"
            + eda_brief
            + "\n## END OF EDA INSIGHTS\n"
        )
        PERCH_RESEARCHER_SYSTEM_PROMPT = PERCH_RESEARCHER_SYSTEM_PROMPT + _eda_block
        PERCH_CODER_SYSTEM_PROMPT = PERCH_CODER_SYSTEM_PROMPT + _eda_block
        if not _cfg_quiet(config):
            print(f"  EDA brief injected into Perch prompts ({len(eda_brief)} chars)")

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

    if config.get("perch_1c_sweep"):
        run_1c_aug_sweep(
            config,
            sess=sess,
            inp_name=inp_name,
            emb_idx=emb_idx,
            logit_idx=logit_idx,
            MAPPED_POS=MAPPED_POS,
            MAPPED_BC_IDX=MAPPED_BC_IDX,
            proxy_map=proxy_map,
            NO_LABEL=NO_LABEL,
            train_df=train_df,
            paths=paths,
            species_to_idx=species_to_idx,
            n_species=n_species,
            embed_bs=embed_bs,
            sample_frac=sample_frac,
            max_samples=max_samples,
        )
        return

    # ── Step 6: Build embedding caches (once) ─────────────────────────────
    train_cache = cache_dir / (
        f"train_emb_{preset}.npz" if preset else "train_emb.npz"
    )
    # Shared soundscape val cache (meta-agent: logs/meta_agent/perch_cache/val_emb.npz).
    val_cache = Path(perch_paths.get("val_cache_path") or (cache_dir.parent / "val_emb.npz"))
    aug_config = config.get("augmentation")
    train_meta_path = _embedding_cache_meta_path(train_cache)
    clip_subset_path = config.get("perch_embed_clip_subset")
    clip_rows = None
    if clip_subset_path:
        clip_rows = _load_clip_rows_jsonl(Path(clip_subset_path))
        print(f"  [Cache] Clip subset mode: {len(clip_rows)} clips (fast aug search)")

    clips_sidecar = _embedding_cache_clips_path(train_cache)
    if force_rebuild and not config.get("perch_build_val_only"):
        for p in (train_cache, train_meta_path, clips_sidecar):
            if p.exists():
                p.unlink()
                print(f"  [Cache] Removed {p.name} (force_rebuild_cache)")

    if config.get("perch_build_val_only"):
        if not val_cache.exists():
            print("\n  [Setup] Building shared validation embedding cache...")
            ok = _build_val_cache(
                sess, inp_name, emb_idx, logit_idx,
                MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
                paths.train_soundscapes_dir, paths.train_soundscapes_labels_csv,
                species_to_idx, n_species, val_cache, embed_bs,
            )
            if not ok:
                raise RuntimeError("Could not build soundscape val cache for stage 1c.")
        else:
            import numpy as np
            d = np.load(str(val_cache))
            print(f"  [Cache] Reusing existing val embeddings → {val_cache}")
            print(f"  [Cache] Val cache loaded: X={d['X'].shape}  y={d['y'].shape}")
        print("\n  [Cache-only] Val embedding cache ready — exiting.")
        return

    if not train_cache.exists():
        est = "a few minutes" if clip_rows else "~30-60 min"
        print(f"\n  [Setup] Building training embedding cache (runs once, {est})...")
        _build_train_cache(
            sess, inp_name, emb_idx, logit_idx,
            MAPPED_POS, MAPPED_BC_IDX, proxy_map, NO_LABEL,
            train_df, species_to_idx, paths.train_audio_dir, n_species,
            train_cache,
            sample_frac=1.0 if clip_rows else sample_frac,
            max_samples=None if clip_rows else max_samples,
            batch_size=embed_bs,
            aug_config=aug_config,
            soundscapes_dir=paths.train_soundscapes_dir,
            aug_preset=preset,
            clip_rows=clip_rows,
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

    if config.get("perch_build_cache_only"):
        print("\n  [Cache-only] Train embedding cache ready — exiting (no head training).")
        return

    if config.get("perch_build_val_only"):
        print("\n  [Cache-only] Val embedding cache ready — exiting.")
        return

    fixed_cfg = config.get("perch_fixed_train") or {}
    if fixed_cfg.get("enabled"):
        run_fixed_head_train(
            config,
            fixed_cfg,
            train_cache=train_cache,
            val_cache=val_cache,
            mem_dir=mem_dir,
            code_dir=code_dir,
        )
        return

    # ── Step 7: Set up agent components ──────────────────────────────────
    researcher_llm = LLMClient(
        provider=provider,
        model=researcher_model,
        timeout_seconds=researcher_timeout,
        stream_debug=researcher_stream,
    )
    coder_llm      = LLMClient(provider=provider, model=coder_model)
    ranking_metric = _ranking_metric_from_config(config)
    memory = PerchExperimentMemory(mem_dir, ranking_metric=ranking_metric)
    use_llm_memory = bool(config.get("perch", {}).get("use_llm_memory_summaries", False))
    memory.configure_summaries(
        use_llm=use_llm_memory,
        llm=researcher_llm if use_llm_memory else None,
    )
    if use_llm_memory and memory.total() > memory._digest.get("summarized_run_count", 0):
        print("  [Memory] Catching up LLM digest for prior runs...")
        memory._catch_up_summaries()
    else:
        memory._update_digest_best()

    refine_cfg = config.get("perch_refine") or {}
    refine_enabled = bool(refine_cfg.get("enabled"))
    if refine_enabled:
        locked_arch = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
            "arch_type", "residual_mlp"
        )
        seed_spec = _load_refine_champion_spec(mem_dir) or dict(
            refine_cfg.get("seed_spec") or {}
        )
        researcher = PerchResearcher(
            researcher_llm,
            memory,
            temperature=researcher_temp,
            refine_mode=True,
            locked_arch_type=str(locked_arch),
            seed_spec=seed_spec,
            seed_score=refine_cfg.get("seed_score"),
            batch_size=experiments_per_round,
        )
    else:
        researcher = PerchResearcher(
            researcher_llm,
            memory,
            temperature=researcher_temp,
            batch_size=experiments_per_round,
        )

    executor       = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator      = Evaluator(row_id_column_name="row_id")

    print(f"\n  Researcher model : {researcher_model}")
    print(f"  Researcher timeout: {researcher_timeout:.0f}s (then safe-default spec for that iter)")
    if researcher_stream:
        print("  Researcher stream : ON (live think/answer tokens in terminal)")
    print(f"  Coder model      : {coder_model}")
    if refine_enabled:
        print("  Mode             : REFINE (stage 1b — locked arch, batch planner, adaptive budget)")
        print(
            f"  Refine budget    : {refine_cfg.get('initial_iterations', 5)} initial training tries, "
            f"+{refine_cfg.get('bonus_iterations_on_improve', 5)} on improve, "
            f"max {refine_cfg.get('max_iterations_per_model', 25)}"
        )
        if experiments_per_round > 1:
            print(
                f"  Experiments/round: {experiments_per_round} "
                f"(labels: {', '.join(_refine_batch_slot_labels(experiments_per_round))})"
            )
    else:
        print(f"  Planner rounds   : {max_iterations}")
        if experiments_per_round > 1:
            approx = max_iterations * experiments_per_round
            print(
                f"  Experiments/round: {experiments_per_round} "
                f"(~{approx} coder+train runs, slots: {', '.join(PERCH_BATCH_SLOTS[:experiments_per_round])})"
            )
        else:
            print(f"  Experiments/round: 1")
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
        mem_mode = "lean run log → researcher" + (
            " (+ optional LLM digest)" if use_llm_memory else ""
        )
        print(
            f"  Memory           : {prior} prior runs | best {best_str} | {mem_mode}"
        )
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

    if refine_enabled:
        _run_refine_loop(
            config,
            refine_cfg,
            researcher=researcher,
            coder_llm=coder_llm,
            coder_temp=coder_temp,
            executor=executor,
            evaluator=evaluator,
            memory=memory,
            train_cache=train_cache,
            val_cache=val_cache,
            head_train_max=head_train_max,
            code_dir=code_dir,
            mem_dir=mem_dir,
            ranking_metric=ranking_metric,
            experiments_per_round=experiments_per_round,
        )
        print(f"\n{'='*60}")
        print("  DONE (refine campaign)")
        best = memory.best_runs(3)
        for i, r in enumerate(best, 1):
            print(f"  #{i} {memory._format_run_score(r)} | {r['reasoning'][:80]}")
        print(f"{'='*60}")
        if config.get("perch", {}).get("skip_final_retrain", False):
            print("\n  Final retrain skipped (perch.skip_final_retrain=true).")
            return
        best_runs = memory.best_runs(1)
        if best_runs and (mem_dir / "best_head_code.py").exists():
            best_code = (mem_dir / "best_head_code.py").read_text(encoding="utf-8")
            print(f"\n{'='*60}")
            print("  FINAL RETRAIN — full data (refine winner)")
            print(f"{'='*60}")
            final_script = _build_final_retrain_script(
                best_code, mem_dir, train_cache, val_cache
            )
            final_script_path = code_dir / "final_retrain.py"
            final_script_path.write_text(final_script, encoding="utf-8")
            result = executor.run_file(final_script_path)
            if result.success and "FINAL_RETRAIN_DONE" in (result.stdout or ""):
                print(f"  Final head saved → {mem_dir / 'final_head.weights.h5'}")
        return

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
        round_label = (
            f"  PLANNER ROUND {iteration}/{max_iterations}"
            if experiments_per_round > 1
            else f"  ITERATION {iteration}/{max_iterations}"
        )
        print(round_label)
        print(f"{'─'*60}")

        specs = researcher.next_experiments()
        for slot_i, spec in enumerate(specs, 1):
            slot_label = str(spec.get("slot") or f"s{slot_i}")
            print(f"\n  ▸ Slot {slot_i}/{len(specs)}: {slot_label}")
            metrics, slot_code, spec = _execute_perch_iteration(
                iteration,
                spec=spec,
                slot_suffix=slot_label,
                researcher=researcher,
                coder_llm=coder_llm,
                coder_temp=coder_temp,
                executor=executor,
                evaluator=evaluator,
                memory=memory,
                train_cache=train_cache,
                val_cache=val_cache,
                head_train_max=head_train_max,
                code_dir=code_dir,
            )
            rank_val = _ranking_value_from_metrics(metrics)
            print(f"  [Result] [{slot_label}] {_format_iteration_metrics(metrics)}")
            memory.log(spec=spec, metrics=metrics, code=slot_code)
            best_score_ever = _promote_if_best(
                rank_val=rank_val,
                metrics=metrics,
                spec=spec,
                iteration=iteration,
                slot_label=slot_label,
                best_score_ever=best_score_ever,
                trained_head_path=trained_head_path,
                best_head_path=best_head_path,
                mem_dir=mem_dir,
                code_dir=code_dir,
                slot_code=slot_code,
                ranking_metric=ranking_metric,
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
        final_script      = _build_final_retrain_script(
            best_code, mem_dir, train_cache, val_cache
        )
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
