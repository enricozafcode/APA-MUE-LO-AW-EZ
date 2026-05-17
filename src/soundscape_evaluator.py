"""
Soundscape evaluation metrics (v2 benchmark — competition-faithful).

Primary metric for meta-agent model ranking: ``macro_average_precision``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Meta-agent uses this to rank CNN / BirdNET / Perch / ensemble on labeled train_soundscapes.
PRIMARY_META_METRIC = "macro_average_precision"


def competition_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """
    BirdCLEF macro ROC-AUC: mean per-class AUC over species with both positives
    and negatives in ``y_true`` (skip all-zero / all-one columns).
    """
    aucs: list[float] = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.max() == 0 or yt.min() == 1:
            continue
        aucs.append(float(roc_auc_score(yt, y_pred[:, c])))
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def median_per_class_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Median per-class ROC-AUC (v2 benchmark); same scoreable-class filter as macro metrics."""
    aucs: list[float] = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.max() == 0 or yt.min() == 1:
            continue
        aucs.append(float(roc_auc_score(yt, y_pred[:, c])))
    if not aucs:
        return float("nan"), 0
    return float(np.median(aucs)), len(aucs)


def macro_average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """
    Macro average precision (v2 benchmark ranking metric).

    Same class filter as ``competition_macro_auc``: skip columns with no positives
    or no negatives in ``y_true``.
    """
    aps: list[float] = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.max() == 0 or yt.min() == 1:
            continue
        aps.append(float(average_precision_score(yt, y_pred[:, c])))
    if not aps:
        return float("nan"), 0
    return float(np.mean(aps)), len(aps)


def compute_metric_panel(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Core v2 soundscape metrics shown in agents and meta-agent logs."""
    out: dict[str, float] = {}
    auc, k = competition_macro_auc(y_true, y_pred)
    out["competition_macro_auc"] = auc
    out["n_scored_classes"] = float(k)
    ap, _ = macro_average_precision(y_true, y_pred)
    out[PRIMARY_META_METRIC] = ap
    med, _ = median_per_class_auc(y_true, y_pred)
    out["median_per_class_auc"] = med
    return out


def format_soundscape_metrics_line(
    *,
    macro_ap: float | None = None,
    macro_auc: float | None = None,
    median_auc: float | None = None,
    ranking_metric: str = PRIMARY_META_METRIC,
    mark_ranking: bool = True,
) -> str:
    """Standard one-line metric display: macro_AP (ranking) | macro_AUC | median_AUC."""
    parts: list[str] = []

    def _f(v: float | None) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return f"{v:.5f}"

    ap_s = _f(macro_ap)
    auc_s = _f(macro_auc)
    med_s = _f(median_auc)

    if ap_s is not None:
        tag = " (ranking)" if mark_ranking and ranking_metric == PRIMARY_META_METRIC else ""
        parts.append(f"macro_AP={ap_s}{tag}")
    if auc_s is not None:
        tag = " (ranking)" if mark_ranking and ranking_metric == "macro_roc_auc" else ""
        parts.append(f"macro_AUC={auc_s}{tag}")
    if med_s is not None:
        parts.append(f"median_AUC={med_s}")
    return " | ".join(parts) if parts else "no score"


def format_metrics_dict(
    metrics: dict[str, Any] | None,
    *,
    ranking_metric: str = PRIMARY_META_METRIC,
) -> str:
    if not metrics or metrics.get("status") != "success":
        return "FAILED"
    ap = metrics.get("macro_average_precision")
    auc = metrics.get("competition_macro_auc_v2", metrics.get("macro_roc_auc"))
    med = metrics.get("median_per_class_auc")
    return format_soundscape_metrics_line(
        macro_ap=float(ap) if ap is not None else None,
        macro_auc=float(auc) if auc is not None else None,
        median_auc=float(med) if med is not None else None,
        ranking_metric=ranking_metric,
    )


def format_soundscape_score(score: "SoundscapeScore") -> str:
    return format_soundscape_metrics_line(
        macro_ap=score.macro_average_precision,
        macro_auc=score.competition_macro_auc,
        median_auc=score.median_per_class_auc,
        ranking_metric=score.primary_metric,
    )


def primary_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str | None = None) -> float:
    """Return the configured primary scalar score (higher is better)."""
    name = metric or PRIMARY_META_METRIC
    panel = compute_metric_panel(y_true, y_pred)
    return float(panel.get(name, float("nan")))


def enrich_soundscape_metrics(
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Add v2 soundscape metrics (macro AP + faithful macro AUC) to an evaluator metrics dict."""
    if metrics.get("status") != "success":
        return metrics
    ap, _ = macro_average_precision(y_true, y_pred)
    auc_v2, n_v2 = competition_macro_auc(y_true, y_pred)
    med, _ = median_per_class_auc(y_true, y_pred)
    metrics["macro_average_precision"] = ap
    metrics["competition_macro_auc_v2"] = auc_v2
    metrics["median_per_class_auc"] = med
    metrics["n_scored_classes_v2"] = int(n_v2)
    metrics["ranking_metric"] = PRIMARY_META_METRIC
    metrics["ranking_value"] = ap
    return metrics


@dataclass
class SoundscapeScore:
    """Scores on labeled train_soundscapes (aligned windows)."""

    primary_metric: str
    primary_value: float
    competition_macro_auc: float
    macro_average_precision: float
    median_per_class_auc: float
    n_scored_classes: int
    n_windows: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "competition_macro_auc": self.competition_macro_auc,
            "macro_average_precision": self.macro_average_precision,
            "median_per_class_auc": self.median_per_class_auc,
            "n_scored_classes": self.n_scored_classes,
            "n_windows": self.n_windows,
            **self.metrics,
        }


class SoundscapeEvalSuite:
    """
    Evaluate archived / trained models on labeled ``train_soundscapes``
    using the same ground truth as ``submission_metric_benchmark_v2``.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        primary_metric: str = PRIMARY_META_METRIC,
    ) -> None:
        from submission_benchmark import build_soundscape_ground_truth

        self.data_dir = Path(data_dir)
        self.soundscapes_dir = self.data_dir / "train_soundscapes"
        self.primary_metric = primary_metric
        self.y_true_df, _ = build_soundscape_ground_truth(
            self.data_dir / "train_soundscapes_labels.csv",
            self.data_dir / "sample_submission.csv",
            self.soundscapes_dir,
        )
        self.species_cols = list(self.y_true_df.columns)
        self.y_true = self.y_true_df.to_numpy(dtype=np.float32)

    def score_arrays(self, y_pred: np.ndarray) -> SoundscapeScore:
        if y_pred.shape != self.y_true.shape:
            raise ValueError(
                f"Prediction shape {y_pred.shape} != ground truth {self.y_true.shape}"
            )
        panel = compute_metric_panel(self.y_true, y_pred)
        primary = float(panel.get(self.primary_metric, float("nan")))
        return SoundscapeScore(
            primary_metric=self.primary_metric,
            primary_value=primary,
            competition_macro_auc=float(panel["competition_macro_auc"]),
            macro_average_precision=float(panel[PRIMARY_META_METRIC]),
            median_per_class_auc=float(panel["median_per_class_auc"]),
            n_scored_classes=int(panel["n_scored_classes"]),
            n_windows=int(len(self.y_true)),
            metrics=panel,
        )

    def score_from_pred_df(self, pred_df: pd.DataFrame) -> SoundscapeScore:
        from submission_benchmark import align_predictions

        yt, yp, _ = align_predictions(self.y_true_df, pred_df)
        return self.score_arrays(yp)

    def score_cnn(self, model_path: Path, *, batch_size: int = 32) -> SoundscapeScore:
        import tensorflow as tf
        from submission_benchmark import predict_cnn_on_labeled_rows

        model = tf.keras.models.load_model(str(model_path), compile=False)
        try:
            pred_df = predict_cnn_on_labeled_rows(
                model, self.y_true_df, self.soundscapes_dir, batch_size=batch_size
            )
            return self.score_from_pred_df(pred_df)
        finally:
            tf.keras.backend.clear_session()

    def score_perch(self, archive_dir: Path, **kwargs: Any) -> SoundscapeScore:
        from submission_benchmark import predict_perch_on_labeled_rows

        pred_df = predict_perch_on_labeled_rows(
            Path(archive_dir), self.y_true_df, self.soundscapes_dir, **kwargs
        )
        return self.score_from_pred_df(pred_df)

    def score_perch_mem_dir(self, archive_dir: Path) -> SoundscapeScore | None:
        """
        Score from ``best_val_preds.npy`` saved during cached-val head training
        (avoids re-running ONNX on every soundscape window).
        """
        archive_dir = Path(archive_dir)
        preds_path = archive_dir / "best_val_preds.npy"
        if not preds_path.exists():
            return None
        preds = np.load(preds_path).astype(np.float32)
        if preds.shape != self.y_true.shape:
            return None
        return self.score_arrays(preds)

    def score_birdnet_val_preds(
        self,
        preds: np.ndarray,
        row_ids: list[str] | np.ndarray,
    ) -> SoundscapeScore:
        """Align BirdNET val predictions (row_id order) to benchmark ground truth."""
        from submission_benchmark import align_predictions

        row_ids = [str(r) for r in row_ids]
        pred_df = pd.DataFrame(preds, columns=self.species_cols)
        pred_df.insert(0, "row_id", row_ids)
        yt, yp, _ = align_predictions(self.y_true_df, pred_df)
        return self.score_arrays(yp)

    def score_birdnet_artifacts(
        self,
        logs_dir: Path,
        val_npz: Path,
    ) -> SoundscapeScore | None:
        aligned = self.aligned_birdnet_preds(logs_dir, val_npz)
        if aligned is None:
            return None
        return self.score_arrays(aligned)

    def aligned_birdnet_preds(
        self,
        logs_dir: Path,
        val_npz: Path,
    ) -> np.ndarray | None:
        """BirdNET scores aligned to ``y_true_df`` row order (for ensembling)."""
        preds_path = Path(logs_dir) / "best_val_preds.npy"
        if not preds_path.exists() or not Path(val_npz).exists():
            return None
        d = np.load(str(val_npz), allow_pickle=True)
        preds = np.load(str(preds_path)).astype(np.float32)
        row_ids = d["row_ids"]
        from submission_benchmark import align_predictions

        pred_df = pd.DataFrame(preds, columns=self.species_cols)
        pred_df.insert(0, "row_id", [str(r) for r in row_ids])
        _, yp, _ = align_predictions(self.y_true_df, pred_df)
        return yp

    def aligned_perch_preds(self, archive_dir: Path, **kwargs: Any) -> np.ndarray:
        from submission_benchmark import align_predictions, predict_perch_on_labeled_rows

        pred_df = predict_perch_on_labeled_rows(
            Path(archive_dir), self.y_true_df, self.soundscapes_dir, **kwargs
        )
        _, yp, _ = align_predictions(self.y_true_df, pred_df)
        return yp
