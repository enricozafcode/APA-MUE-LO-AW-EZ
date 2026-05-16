"""Local metric benchmarks for archived submissions vs Kaggle LB scores."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator import Evaluator, score as competition_macro_auc
from soundscape_evaluator import (
    PRIMARY_META_METRIC,
    competition_macro_auc as soundscape_competition_macro_auc,
    macro_average_precision,
    median_per_class_auc,
)

SR = 32_000
CLIP_SECONDS = 5.0
N_MELS = 64
N_FRAMES = 128
N_FFT = 1024
HOP_LENGTH = 512
SEG_SAMPLES = int(SR * CLIP_SECONDS)
TARGET_SECONDS = 60.0
TARGET_SAMPLES = int(SR * TARGET_SECONDS)
N_WINDOWS = int(TARGET_SECONDS / CLIP_SECONDS)


def build_soundscape_ground_truth(
    labels_csv: Path,
    sample_sub_csv: Path,
    soundscapes_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    sample_sub = pd.read_csv(sample_sub_csv)
    species_cols = [c for c in sample_sub.columns if c != "row_id"]
    sp2i = {s: i for i, s in enumerate(species_cols)}

    raw = pd.read_csv(labels_csv)

    def _tok(val: object) -> set[str]:
        if pd.isna(val) or val == "":
            return set()
        return {t.strip() for t in str(val).split(";") if t.strip()}

    grp = (
        raw.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[_tok(v) for v in s]))
        .reset_index()
    )
    grp["end_sec"] = pd.to_timedelta(grp["end"]).dt.total_seconds().astype(int)
    grp["row_id"] = grp["filename"].str.replace(".ogg", "", regex=False) + "_" + grp["end_sec"].astype(str)

    rows: list[tuple[str, np.ndarray]] = []
    for r in grp.itertuples(index=False):
        vec = np.zeros(len(species_cols), dtype=np.float32)
        for code in r.primary_label:
            j = sp2i.get(code)
            if j is not None:
                vec[j] = 1.0
        rows.append((r.row_id, vec))

    y_true_df = pd.DataFrame(
        [v for _, v in rows],
        index=[rid for rid, _ in rows],
        columns=species_cols,
    ).sort_index()
    if not y_true_df.index.is_unique:
        y_true_df = y_true_df.groupby(level=0).max()

    stems = {rid.rsplit("_", 1)[0] for rid in y_true_df.index}
    ogg_paths = sorted(
        p for stem in stems if (p := soundscapes_dir / f"{stem}.ogg").exists()
    )
    return y_true_df, ogg_paths


def macro_auc_ge3(y_true: np.ndarray, y_score: np.ndarray, min_pos: int = 3) -> tuple[float, int]:
    """BirdNET / meta_agent ensemble style."""
    pos = y_true.sum(axis=0)
    keep = pos >= min_pos
    if not np.any(keep):
        return float("nan"), 0
    yt = y_true[:, keep]
    ys = y_score[:, keep]
    usable = [
        j for j in range(yt.shape[1]) if yt[:, j].min() == 0 and yt[:, j].max() == 1
    ]
    if not usable:
        return float("nan"), int(np.sum(keep))
    usable = np.array(usable, dtype=int)
    return float(roc_auc_score(yt[:, usable], ys[:, usable], average="macro")), int(np.sum(keep))


def macro_auc_min_pos(y_true: np.ndarray, y_score: np.ndarray, min_pos: int) -> tuple[float, int]:
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except ValueError:
            pass
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def evaluator_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    ev = Evaluator()
    summary = ev.evaluate_arrays(y_true, y_score)
    return summary.metrics


def macro_map(y_true: np.ndarray, y_score: np.ndarray, min_pos: int = 1) -> tuple[float, int]:
    """Macro AP with optional min-positive filter (legacy). Prefer ``macro_average_precision``."""
    if min_pos <= 1:
        return macro_average_precision(y_true, y_score)
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() < min_pos:
            continue
        yt = y_true[:, c]
        if yt.max() == 0 or yt.min() == 1:
            continue
        try:
            aps.append(average_precision_score(yt, y_score[:, c]))
        except ValueError:
            pass
    return (float(np.mean(aps)) if aps else float("nan")), len(aps)


def align_predictions(
    y_true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Align soundscape labels to predictions by row_id.

    Do not read prediction scores from ``merged[species_cols]`` after a join with
    overlapping column names — pandas keeps the left (label) frame for those names.
    """
    species_cols = list(y_true_df.columns)
    pred = pred_df.set_index("row_id") if "row_id" in pred_df.columns else pred_df
    common = y_true_df.index.intersection(pred.index)
    if len(common) == 0:
        raise ValueError("No overlapping row_id between ground truth and predictions.")

    yt = y_true_df.loc[common, species_cols].to_numpy(dtype=np.float32)
    yp = pred.loc[common, species_cols].to_numpy(dtype=np.float32)
    merged = y_true_df.loc[common].join(
        pred.loc[common, species_cols].rename(columns=lambda c: f"{c}_pred"),
    )
    if np.allclose(yt, yp):
        raise ValueError(
            "Predictions are identical to ground-truth labels after alignment. "
            "This usually indicates a bug in align_predictions or cached .npz files "
            "built with the old join logic — delete notebooks/cache/submission_preds/ "
            "and re-run inference."
        )
    return yt, yp, merged


def _mel_batch_from_segment(seg: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=seg,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (
        tf.image.resize(mel_db[..., np.newaxis], (N_MELS, N_FRAMES))
        .numpy()
        .astype(np.float32)
    )


def _cnn_forward(model: tf.keras.Model, mels: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Run CNN without Keras predict() progress bars (can hang in Jupyter without ipywidgets)."""
    if mels.ndim == 3:
        mels = mels[..., np.newaxis]
    n = mels.shape[0]
    if n == 0:
        return np.zeros((0, model.output_shape[-1]), dtype=np.float32)
    outs: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch = mels[start : start + batch_size]
        out = model(batch, training=False)
        outs.append(np.asarray(out, dtype=np.float32))
    return np.concatenate(outs, axis=0)


_CNN_WARMED_UP = False


def _warmup_cnn(model: tf.keras.Model) -> None:
    global _CNN_WARMED_UP
    if _CNN_WARMED_UP:
        return
    print("  Warming up TensorFlow graph (first predict can take 1–3 min in notebooks) …", flush=True)
    dummy = np.zeros((1, N_MELS, N_FRAMES, 1), dtype=np.float32)
    _ = model(dummy, training=False)
    _CNN_WARMED_UP = True
    print("  Warmup done.", flush=True)


def predict_cnn_on_labeled_rows(
    model: tf.keras.Model,
    y_true_df: pd.DataFrame,
    soundscapes_dir: Path,
    *,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Agent/CNN eval convention: 5s windows at 5,10,...,60s from padded 60s file start."""
    _warmup_cnn(model)
    species_cols = list(y_true_df.columns)
    by_stem: dict[str, list[str]] = {}
    for rid in y_true_df.index:
        stem, end_s = rid.rsplit("_", 1)
        by_stem.setdefault(stem, []).append(rid)

    rows: list[dict] = []
    stems = sorted(by_stem.keys())
    n_stems = len(stems)
    print(
        f"  CNN inference: {n_stems} labeled soundscape file(s), {len(y_true_df)} row_id(s)",
        flush=True,
    )
    for si, stem in enumerate(stems):
        if si == 0 or (si + 1) % 10 == 0 or si + 1 == n_stems:
            print(f"  … file {si + 1}/{n_stems}: {stem}", flush=True)
        fp = soundscapes_dir / f"{stem}.ogg"
        if not fp.exists():
            continue
        y_full, _ = librosa.load(
            str(fp), sr=SR, mono=True, duration=TARGET_SECONDS, offset=0.0
        )
        if len(y_full) < TARGET_SAMPLES:
            y_full = np.pad(y_full, (0, TARGET_SAMPLES - len(y_full)))

        rid_to_wi = {}
        for rid in by_stem[stem]:
            end_sec = int(rid.rsplit("_", 1)[1])
            if end_sec % CLIP_SECONDS != 0 or end_sec < CLIP_SECONDS or end_sec > TARGET_SECONDS:
                continue
            wi = end_sec // int(CLIP_SECONDS) - 1
            if 0 <= wi < N_WINDOWS:
                rid_to_wi[rid] = wi

        if not rid_to_wi:
            continue

        batch_mels = []
        batch_rids = []
        for rid, wi in sorted(rid_to_wi.items(), key=lambda x: x[1]):
            start = wi * SEG_SAMPLES
            seg = y_full[start : start + SEG_SAMPLES]
            batch_mels.append(_mel_batch_from_segment(seg))
            batch_rids.append(rid)

        mels = np.stack(batch_mels, axis=0)
        if si == 0:
            print(f"    predict {len(batch_rids)} window(s) …", flush=True)
        preds = _cnn_forward(model, mels, batch_size=batch_size)
        if si == 0:
            print("    first file done.", flush=True)
        for rid, p in zip(batch_rids, preds):
            row = {"row_id": rid}
            for col, val in zip(species_cols, p):
                row[col] = float(val)
            rows.append(row)

        if (si + 1) % 50 == 0:
            print(f"  CNN inference: {si + 1}/{len(stems)} soundscape files", flush=True)

    return pd.DataFrame(rows)


PERCH_SAMPLES = 160_000  # 5 s @ 32 kHz — Google Perch ONNX input length


def resolve_perch_onnx_path(onnx_path: Path | None = None) -> Path:
    """Locate Perch ONNX: explicit path, env var, kagglehub cache, or download."""
    if onnx_path is not None and onnx_path.exists():
        return onnx_path
    env = os.environ.get("PERCH_ONNX_PATH")
    if env and Path(env).exists():
        return Path(env)
    cache_glob = sorted(
        Path.home().glob(
            ".cache/kagglehub/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/**/perch_v2.onnx"
        )
    )
    if cache_glob:
        return cache_glob[-1]
    from perch_agent import _ensure_deps, _find_or_download_onnx

    _ensure_deps()
    cfg_path = ROOT / "configs" / "agent_config.json"
    slug = "rishikeshjani/perch-onnx-for-birdclef-2026"
    if cfg_path.exists():
        slug = json.loads(cfg_path.read_text(encoding="utf-8")).get("perch", {}).get(
            "onnx_dataset", slug
        )
    return _find_or_download_onnx(slug)


def load_perch_archive(archive_dir: Path) -> dict[str, Any]:
    """Load Perch head + species mapping artifacts from a submission archive."""
    archive_dir = Path(archive_dir)
    head_path = archive_dir / "best_head.keras"
    if not head_path.exists():
        head_path = archive_dir / "final_head.keras"
    if not head_path.exists():
        raise FileNotFoundError(f"No Perch head in {archive_dir}")

    species_cols = json.loads((archive_dir / "species_cols.json").read_text(encoding="utf-8"))
    meta = json.loads((archive_dir / "mapping_meta.json").read_text(encoding="utf-8"))
    no_label = int(meta["NO_LABEL"])
    bc_idx = np.load(archive_dir / "bc_indices.npy").astype(np.int32)
    proxy_raw = json.loads((archive_dir / "proxy_map.json").read_text(encoding="utf-8"))
    proxy_map = {int(k): [int(x) for x in v] for k, v in proxy_raw.items()}

    mapped_mask = bc_idx != no_label
    mapped_pos = np.where(mapped_mask)[0].astype(np.int32)
    mapped_bc_idx = bc_idx[mapped_mask].astype(np.int32)

    perch_weight = 0.2
    info_path = archive_dir / "best_model_info.json"
    if info_path.exists():
        perch_weight = float(
            json.loads(info_path.read_text(encoding="utf-8")).get("spec", {}).get(
                "perch_weight", perch_weight
            )
        )

    return {
        "head_path": head_path,
        "species_cols": species_cols,
        "mapped_pos": mapped_pos,
        "mapped_bc_idx": mapped_bc_idx,
        "proxy_map": proxy_map,
        "no_label": no_label,
        "perch_weight": perch_weight,
        "n_species": len(species_cols),
    }


def predict_perch_on_labeled_rows(
    archive_dir: Path,
    y_true_df: pd.DataFrame,
    soundscapes_dir: Path,
    *,
    onnx_path: Path | None = None,
    embed_batch_size: int = 16,
    head_batch_size: int = 256,
) -> pd.DataFrame:
    """
    Live Perch soundscape eval aligned to ``y_true_df`` row_ids.

    For each labeled window ``{stem}_{end_sec}``, loads the 5 s clip at
    ``offset = end_sec - 5`` (same grid as ``train_soundscapes_labels.csv``),
    runs ONNX Perch → blends mapped Perch logits with the archived Keras head.
    """
    from perch_agent import (
        _apply_logit_mapping,
        _ensure_deps,
        _enforce_length,
        _load_onnx_session,
        _perch_embed_batch,
    )

    _ensure_deps()
    art = load_perch_archive(archive_dir)
    species_cols = list(y_true_df.columns)
    arch_cols = art["species_cols"]
    if arch_cols != species_cols:
        raise ValueError(
            "species column order in archive does not match sample_submission / y_true_df."
        )

    onnx = resolve_perch_onnx_path(onnx_path)
    print(f"  Perch ONNX: {onnx}", flush=True)
    sess, inp_name, emb_idx, logit_idx = _load_onnx_session(onnx)

    print(f"  Loading Perch head: {art['head_path'].name}", flush=True)
    head = tf.keras.models.load_model(str(art["head_path"]), compile=False)
    weights_path = art["head_path"].with_suffix(".weights.h5")
    if weights_path.exists():
        head.load_weights(str(weights_path))
    perch_w = float(art["perch_weight"])
    n_species = int(art["n_species"])
    mp, mbc, proxy, no_label = (
        art["mapped_pos"],
        art["mapped_bc_idx"],
        art["proxy_map"],
        art["no_label"],
    )

    rids = list(y_true_df.index)
    print(
        f"  Perch inference: {len(rids)} labeled window(s) (embed batch={embed_batch_size})",
        flush=True,
    )

    rows: list[dict] = []
    wavs: list[np.ndarray] = []
    batch_rids: list[str] = []

    def _flush() -> None:
        if not wavs:
            return
        embs, logits = _perch_embed_batch(sess, inp_name, emb_idx, logit_idx, wavs)
        perch_scores = _apply_logit_mapping(
            logits, n_species, mp, mbc, proxy, no_label
        )
        perch_probs = (1.0 / (1.0 + np.exp(-perch_scores))).astype(np.float32)

        head_probs_parts: list[np.ndarray] = []
        for start in range(0, len(embs), head_batch_size):
            chunk = embs[start : start + head_batch_size]
            out = head(chunk, training=False)
            head_probs_parts.append(np.asarray(out, dtype=np.float32))
        head_probs = np.concatenate(head_probs_parts, axis=0)

        blended = perch_w * perch_probs + (1.0 - perch_w) * head_probs
        for rid, scores in zip(batch_rids, blended):
            row: dict[str, Any] = {"row_id": rid}
            for col, val in zip(species_cols, scores):
                row[col] = float(val)
            rows.append(row)
        wavs.clear()
        batch_rids.clear()

    for i, rid in enumerate(rids):
        stem, end_s = rid.rsplit("_", 1)
        end_sec = int(end_s)
        offset = float(end_sec - int(CLIP_SECONDS))
        if offset < 0:
            continue
        fp = soundscapes_dir / f"{stem}.ogg"
        if not fp.exists():
            continue
        try:
            wav, _ = librosa.load(
                str(fp), sr=SR, mono=True, offset=offset, duration=CLIP_SECONDS
            )
            wav = _enforce_length(wav.astype(np.float32), PERCH_SAMPLES)
        except Exception:
            continue
        wavs.append(wav)
        batch_rids.append(rid)
        if len(wavs) >= embed_batch_size:
            _flush()
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(rids):
            print(f"  … window {i + 1}/{len(rids)}", flush=True)

    _flush()
    tf.keras.backend.clear_session()
    return pd.DataFrame(rows)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    if mask is not None:
        y_true = y_true[mask]
        y_score = y_score[mask]
    if len(y_true) == 0:
        return {"n_samples": 0}

    out: dict[str, float] = {"n_samples": float(len(y_true))}
    ev = evaluator_macro_auc(y_true, y_score)
    out["evaluator_macro_auc"] = float(ev.get("macro_roc_auc", float("nan")))
    out["evaluator_n_scored_species"] = float(ev.get("num_scored_columns", 0))

    for mp in (1, 2, 3, 5, 10):
        auc, nsp = macro_auc_min_pos(y_true, y_score, mp)
        out[f"macro_auc_minpos_{mp}"] = auc
        out[f"n_species_minpos_{mp}"] = float(nsp)

    auc3, n3 = macro_auc_ge3(y_true, y_score, 3)
    out["meta_agent_ge3"] = auc3
    out["meta_agent_ge3_n_species"] = float(n3)

    m_ap, n_ap = macro_average_precision(y_true, y_score)
    out["macro_map_minpos_1"] = m_ap
    out[PRIMARY_META_METRIC] = m_ap
    out["competition_macro_auc_v2"] = float(
        soundscape_competition_macro_auc(y_true, y_score)[0]
    )
    med, _ = median_per_class_auc(y_true, y_score)
    out["median_per_class_auc"] = med

    return out


def subsample_masks(y_true: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n, n_classes = y_true.shape
    masks: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}

    # Rows with at least one positive label (multi-label soundscape segments)
    row_pos = (y_true.sum(axis=1) > 0)
    masks["rows_with_any_label"] = row_pos

    if row_pos.any():
        idx = np.where(row_pos)[0]
        half = max(1, len(idx) // 2)
        masks["half_labeled_rows"] = np.zeros(n, dtype=bool)
        masks["half_labeled_rows"][rng.choice(idx, size=half, replace=False)] = True

    # Top species by positive count on soundscape val
    sp_counts = y_true.sum(axis=0)
    for k in (10, 25, 50):
        top = np.argsort(-sp_counts)[:k]
        col_mask = np.zeros(n_classes, dtype=bool)
        col_mask[top] = True
        active = (y_true[:, col_mask].sum(axis=1) > 0)
        masks[f"rows_top{k}_species"] = active

    # Species with >=3 positives (closer to ge3 / stable AUC)
    ge3_cols = sp_counts >= 3
    if ge3_cols.any():
        masks["rows_ge3_species_present"] = (y_true[:, ge3_cols].sum(axis=1) > 0)

    return masks


def load_archived_focal_auc(archive_dir: Path) -> float | None:
    p = archive_dir / "final_eval_metrics.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    v = data.get("macro_roc_auc")
    return float(v) if v is not None else None
