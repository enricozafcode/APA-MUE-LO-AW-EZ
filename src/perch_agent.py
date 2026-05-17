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

YOUR PRIMARY OBJECTIVE: Explore STRUCTURALLY DIVERSE head architectures — including advanced designs.
Simply varying n_blocks or dropout of the same residual MLP is NOT sufficient. Each run should test a
meaningfully different inductive bias. Strongly consider sophisticated patterns when not yet tried:
- multi_tower_ensemble: 3–5 parallel sub-heads (e.g. residual, gated, attention, bottleneck, linear) on the same
  embedding, each ending in Dense(num_classes); fuse via Average or Concatenate→Dense(num_classes, sigmoid).
  Diversity across towers reduces window-level false positives before temporal smoothing on long recordings.
- mixture_of_experts, multi_scale_mlp, transformer_block, dense_connections (see below).

Architecture types you can propose (pick from arch_types in the search space):
- residual_mlp:       Dense residual blocks with skip connections (BN→Dense→LN→blocks→proj→sigmoid). This is the baseline — avoid repeating it unless refining a genuinely strong result.
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

Reasoning guidelines:
- Look at the history above and identify which arch_types have already been tried.
- Strongly prefer an arch_type NOT yet explored (strategy: "explore").
- Only revisit a past arch_type if it was the clear best and you are refining it (strategy: "exploit").
- Consider jungle soundscape noise, rare species, and per-window calibration (see TASK context above).
- Prefer advanced structures (multi_tower_ensemble, MoE, attention) once basic families have been tried.
- Tune learning_rate, batch_size, optimizer, epochs, patience, and perch_weight to match the architecture's typical training dynamics.
- In arch_description, be explicit: stem projection, each tower/block, fusion rule, dropout, final sigmoid head.

You MUST respond with ONLY a single JSON object — no prose, no explanation, no markdown, no code fences.
Start your response with a single JSON object (opening brace) and end with a closing brace.

The arch_description field must be a precise, implementable description of the head: layer types, sizes, activations, normalization, and how layers connect. The coder will implement it verbatim in TF/Keras.

SHAPE-SAFE arch_description rules (embeddings are 1536-d; hidden_dim is your working width, typically 512–1024):
- Always state: "Project input Dense(hidden_dim) from emb_dim first" before any residual, MoE, or concat blocks.
- For residual/highway/gated blocks: every Add() input must already be hidden_dim (project skip connections if needed).
- For mixture_of_experts: "K experts each Dense(hidden_dim); router Dense(K, softmax); weighted sum stays hidden_dim" — do NOT describe concat of expert outputs.
- For dense_connections: "Concatenate then Dense(hidden_dim) to compress" after each growth step — do NOT Add() a wide concat tensor to a narrow tensor.
- For multi_tower_ensemble: list each tower (e.g. 5 towers: residual, gated, attention, bottleneck, linear_probe_on_stem);
  each tower outputs num_classes logits before fusion; state Average vs Concatenate→Dense fusion.

Example response:
{"arch_type": "gated_mlp", "arch_description": "LayerNorm on input. Dense(1024) projection. Then 2 GLU blocks: each block has two parallel Dense(1024) — one with linear activation (value) and one with sigmoid activation (gate) — multiplied element-wise, then added to a residual Dense(1024) projection of the block input, followed by LayerNorm. Final Dense(512, gelu), Dropout(0.3), Dense(n_classes, sigmoid).", "hidden_dim": 1024, "n_layers": 2, "dropout": 0.3, "activation": "gelu", "normalization": "layer_norm", "learning_rate": 0.001, "batch_size": 256, "optimizer": "adam", "epochs": 30, "patience": 5, "perch_weight": 0.2, "reasoning": "GLU gating not yet tried; may help the head learn to selectively weight Perch embedding dimensions.", "hypothesis": "Gated selection of Perch embedding features should outperform uniform residual blending for multi-label species classification.", "strategy": "explore"}

Required keys: arch_type, arch_description, hidden_dim, n_layers, dropout, activation, normalization, learning_rate, batch_size, optimizer, epochs, patience, perch_weight, reasoning, hypothesis, strategy."""
)

PERCH_REFINE_RESEARCHER_ADDENDUM = """
REFINE MODE — you are optimizing ONE fixed architecture family to beat a stage-1a champion score.
- arch_type is LOCKED to the value given in the user message — do NOT propose a different arch_type.
- strategy MUST be "exploit" (hyperparameters, layer depth, dropout, lr, perch_weight, fusion details).
- Keep the same structural family; improve training config and head details in arch_description.
- Hypothesis should explain why this specific tweak may beat the seed score on soundscape macro AP.
"""


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
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.temperature = temperature
        self.refine_mode = refine_mode
        self.locked_arch_type = locked_arch_type
        self.seed_spec = seed_spec or {}
        self.seed_score = seed_score

    def next_experiment(self) -> dict:
        history  = self.memory.researcher_context()
        best     = self.memory.best_runs(1)
        total    = self.memory.total()
        best_str = self.memory._format_run_score(best[0]) if best else "none"

        if self.refine_mode:
            locked = self.locked_arch_type or self.seed_spec.get("arch_type", "residual_mlp")
            seed_line = ""
            if self.seed_score is not None:
                seed_line = (
                    f"\nSTAGE-1A CHAMPION TO BEAT ({self.memory.ranking_metric}): "
                    f"{float(self.seed_score):.5f}\n"
                )
            if self.seed_spec:
                seed_line += f"Seed spec (refine from this):\n{json.dumps(self.seed_spec, indent=2)}\n"

            user_prompt = (
                f"{history}\n\n"
                f"REFINE CAMPAIGN — locked arch_type: {locked}\n"
                f"{seed_line}\n"
                f"Total refine experiments so far: {total}\n"
                f"Best in this refine run ({self.memory.ranking_metric}): {best_str}\n\n"
                "Your goal: beat the champion score by tuning hyperparameters and head details "
                "within the SAME arch_type. Do NOT switch architecture family.\n\n"
                f"Allowed hyperparameter search space:\n"
                f"{json.dumps(PERCH_SEARCH_SPACE, indent=2)}\n\n"
                "Respond with ONLY a JSON object. arch_type MUST equal the locked value. strategy MUST be exploit.\n"
                "Include all required keys: arch_type, arch_description, hidden_dim, n_layers, dropout, "
                "activation, normalization, learning_rate, batch_size, optimizer, epochs, patience, "
                "perch_weight, reasoning, hypothesis, strategy."
            )
            system_prompt = PERCH_RESEARCHER_SYSTEM_PROMPT + "\n\n" + PERCH_REFINE_RESEARCHER_ADDENDUM
        else:
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
            system_prompt = PERCH_RESEARCHER_SYSTEM_PROMPT

        print(f"\n  [Researcher] Analyzing {total} experiments, best {best_str}...")
        response = self.llm.generate_from_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        spec = self._parse_spec(response)
        if self.refine_mode and self.locked_arch_type:
            spec["arch_type"] = self.locked_arch_type
            spec["strategy"] = "exploit"
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

PERCH_CODER_SYSTEM_PROMPT = (
    """You are a Python ML engineer building TF/Keras classification heads on frozen 1536-d Perch embeddings.

"""
    + PERCH_TASK_CONTEXT
    + """

The researcher proposes an experimental STRATEGY (architecture family, key hyperparameters, hypothesis).
YOU implement the exact Keras head in build_head — not a fixed template. Follow arch_description closely.
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


# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  [Fixed train] {_format_iteration_metrics(metrics)}")

    spec = dict(fixed_cfg.get("spec") or {})
    spec.setdefault("mode", "fixed_head_train")
    spec["aug_preset"] = fixed_cfg.get("aug_preset")

    if rank_val is not None:
        _promote_best_head(
            rank_val=rank_val,
            metrics=metrics,
            best_score_ever=-1.0,
            iteration=0,
            spec=spec,
            slot_code=head_code,
            trained_head_path=trained_head_path,
            best_head_path=best_head_path,
            mem_dir=mem_dir,
            ranking_metric=ranking_metric,
        )
    elif not result.success:
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
) -> tuple[dict | None, str, dict | None]:
    """One researcher→coder→train iteration. Returns (metrics, slot_code, spec)."""
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
            print(f"  [Error]  {error_msg[-600:]}")

        if exec_attempt == _MAX_EXEC_ATTEMPTS:
            print(f"  [Coder] All {_MAX_EXEC_ATTEMPTS} execution attempts exhausted — skipping iteration.")
            break

        repaired = _repair_perch_code(coder_llm, spec, final_slot_code, error_msg, coder_temp)
        if repaired is None:
            print("  [Coder] Repair failed — skipping remaining attempts.")
            break
        final_slot_code = repaired

    return metrics, final_slot_code, spec


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
) -> None:
    """Adaptive exploit loop: initial tries, +bonus on each improvement, hard cap."""
    initial = max(1, int(refine_cfg.get("initial_iterations", 5)))
    bonus = max(1, int(refine_cfg.get("bonus_iterations_on_improve", 5)))
    max_total = max(initial, int(refine_cfg.get("max_iterations_per_model", 25)))
    seed_score = float(refine_cfg.get("seed_score", -1.0))
    locked = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
        "arch_type", "residual_mlp"
    )

    parent_dir = refine_cfg.get("parent_memory_dir")
    if parent_dir and memory.total() == 0:
        memory.seed_from_stage_1a(
            Path(parent_dir),
            arch_type=str(locked),
            aug_baseline=str(refine_cfg.get("aug_baseline", "?")),
            seed_score=seed_score,
            seed_spec=dict(refine_cfg.get("seed_spec") or {}),
        )

    seed_spec = refine_cfg.get("seed_spec") or {}
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
    print(f"  1a champion to beat: {format_metrics_dict(seed_metrics, ranking_metric=ranking_metric)}")
    print(
        f"  Budget: {initial} initial tries, +{bonus} on each improvement, "
        f"max {max_total} total iterations"
    )

    best_score_ever = max(seed_score, -1.0)
    _prior = memory.best_runs(1)
    if _prior:
        best_score_ever = max(best_score_ever, memory._ranking_value(_prior[0]))

    tries_left = initial
    total_done = 0
    iteration = 0
    trained_head_path = Path(tempfile.gettempdir()) / "_trained_head.keras"
    best_head_path = mem_dir / "best_head.keras"

    while total_done < max_total and tries_left > 0:
        iteration += 1
        total_done += 1
        tries_left -= 1

        print(f"\n{'─'*60}")
        print(
            f"  REFINE {iteration}  (done {total_done}/{max_total}, "
            f"queue {tries_left} remaining)"
        )
        print(f"{'─'*60}")

        metrics, slot_code, spec = _execute_perch_iteration(
            iteration,
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
        print(f"  [Result] {_format_iteration_metrics(metrics)}")
        memory.log(spec=spec, metrics=metrics, code=slot_code)

        if rank_val is not None and rank_val > best_score_ever:
            prev = best_score_ever
            best_score_ever = _promote_best_head(
                rank_val=rank_val,
                metrics=metrics,
                best_score_ever=best_score_ever,
                iteration=iteration,
                spec=spec,
                slot_code=slot_code,
                trained_head_path=trained_head_path,
                best_head_path=best_head_path,
                mem_dir=mem_dir,
                ranking_metric=ranking_metric,
            )
            gain = bonus_add = min(bonus, max_total - total_done)
            tries_left += bonus_add
            print(
                f"  [Refine] Improved {ranking_metric} {prev:.5f} → {best_score_ever:.5f} "
                f"| +{bonus_add} bonus tries (queue={tries_left})"
            )
        elif rank_val is not None:
            print(f"  [Refine] No improvement (best {ranking_metric}={best_score_ever:.5f})")

        best = memory.best_runs(1)
        if best:
            print(f"  [Best so far] {memory._format_run_score(best[0])}")

    print(f"\n  [Refine] Finished: {total_done} iterations, best {ranking_metric}={best_score_ever:.5f}")


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

    researcher_model = config.get("researcher", {}).get("model",       "gemma3:4b")
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
    researcher_llm = LLMClient(provider=provider, model=researcher_model)
    coder_llm      = LLMClient(provider=provider, model=coder_model)
    ranking_metric = _ranking_metric_from_config(config)
    memory = PerchExperimentMemory(mem_dir, ranking_metric=ranking_metric)
    memory.attach_summarizer(researcher_llm)
    if memory.total() > memory._digest.get("summarized_run_count", 0):
        print("  [Memory] Catching up compact summaries for prior runs...")
        memory._catch_up_summaries()

    refine_cfg = config.get("perch_refine") or {}
    refine_enabled = bool(refine_cfg.get("enabled"))
    if refine_enabled:
        locked_arch = refine_cfg.get("locked_arch_type") or (refine_cfg.get("seed_spec") or {}).get(
            "arch_type", "residual_mlp"
        )
        researcher = PerchResearcher(
            researcher_llm,
            memory,
            temperature=researcher_temp,
            refine_mode=True,
            locked_arch_type=str(locked_arch),
            seed_spec=refine_cfg.get("seed_spec"),
            seed_score=refine_cfg.get("seed_score"),
        )
    else:
        researcher = PerchResearcher(researcher_llm, memory, temperature=researcher_temp)

    executor       = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)
    evaluator      = Evaluator(row_id_column_name="row_id")

    print(f"\n  Researcher model : {researcher_model}")
    print(f"  Coder model      : {coder_model}")
    if refine_enabled:
        print("  Mode             : REFINE (stage 1b — locked arch, adaptive budget)")
        print(
            f"  Refine budget    : {refine_cfg.get('initial_iterations', 5)} initial, "
            f"+{refine_cfg.get('bonus_iterations_on_improve', 5)} on improve, "
            f"max {refine_cfg.get('max_iterations_per_model', 25)}"
        )
    else:
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
        snap = memory._digest.get("best_snapshot") or {}
        conf = snap.get("confidence_tier", "?")
        print(
            f"  Memory           : {prior} prior runs | best {best_str} | "
            f"compact digest ({int(memory._digest.get('summarized_run_count', 0))} summarized, "
            f"confidence {conf})"
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
