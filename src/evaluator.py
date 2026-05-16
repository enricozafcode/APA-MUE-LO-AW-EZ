from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


@dataclass
class EvaluationSummary:
    metrics: dict[str, Any]
    analysis_prompt: str


class ParticipantVisibleError(Exception):
    """Competition-style visible error for invalid submissions."""


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Version of macro-averaged ROC-AUC score that ignores all classes
    that have no true positive labels.
    """
    solution = solution.copy()
    submission = submission.copy()

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {
            column: submission[column].dtype
            for column in submission.columns
            if not pandas.api.types.is_numeric_dtype(submission[column])
        }
        raise ParticipantVisibleError(f"Invalid submission data types found: {bad_dtypes}")

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    if len(scored_columns) == 0:
        raise ParticipantVisibleError("No classes with positive labels were found.")

    try:
        return sklearn.metrics.roc_auc_score(
            solution[scored_columns].values,
            submission[scored_columns].values,
            average="macro",
        )
    except Exception as exc:
        raise ParticipantVisibleError(f"ROC-AUC computation failed: {exc}") from exc


class Evaluator:
    """Computes competition-aligned metrics from prediction artifacts."""

    def __init__(self, row_id_column_name: str = "row_id") -> None:
        self.row_id_column_name = row_id_column_name

    def evaluate_from_files(self, y_true_path: Path, y_pred_path: Path) -> EvaluationSummary:
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        return self.evaluate_arrays(y_true=y_true, y_pred=y_pred)

    def evaluate_arrays(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationSummary:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim != 2 or y_pred.ndim != 2:
            metrics = {
                "status": "error",
                "reason": "Expected 2D arrays for y_true and y_pred.",
                "y_true_shape": list(y_true.shape),
                "y_pred_shape": list(y_pred.shape),
            }
            return EvaluationSummary(
                metrics=metrics,
                analysis_prompt=(
                    "The experiment ran, but evaluation failed because predictions/labels are not 2D. "
                    "Ensure y_train and model.predict(X_train) produce shape (n_samples, n_classes)."
                ),
            )

        if y_true.shape != y_pred.shape:
            metrics = {
                "status": "error",
                "reason": "Shape mismatch between y_true and y_pred.",
                "y_true_shape": list(y_true.shape),
                "y_pred_shape": list(y_pred.shape),
            }
            return EvaluationSummary(
                metrics=metrics,
                analysis_prompt=(
                    "The experiment ran, but evaluation failed because predictions and labels have "
                    "different shapes. Fix the output layer/label dimensions."
                ),
            )

        row_count, class_count = y_true.shape
        columns = [f"class_{idx}" for idx in range(class_count)]

        solution = pd.DataFrame(y_true, columns=columns)
        submission = pd.DataFrame(y_pred, columns=columns)
        solution.insert(0, self.row_id_column_name, np.arange(row_count))
        submission.insert(0, self.row_id_column_name, np.arange(row_count))

        try:
            macro_roc_auc = score(
                solution=solution,
                submission=submission,
                row_id_column_name=self.row_id_column_name,
            )
            error_metric = 1.0 - float(macro_roc_auc)
            scored_columns = int((solution.drop(columns=[self.row_id_column_name]).sum(axis=0) > 0).sum())
            metrics = {
                "status": "success",
                "macro_roc_auc": float(macro_roc_auc),
                "error_metric": error_metric,
                "num_scored_columns": scored_columns,
                "num_classes": class_count,
                "num_samples": row_count,
            }
            from soundscape_evaluator import enrich_soundscape_metrics

            metrics = enrich_soundscape_metrics(metrics, y_true, y_pred)
            analysis_prompt = (
                "The experiment succeeded and was externally evaluated on soundscape validation. "
                f"Use macro_average_precision as the primary score for model selection "
                f"(better local proxy for Kaggle LB than macro_roc_auc alone). "
                f"Metrics: {metrics}. Analyze this result, propose a small architectural improvement, "
                "and provide a complete updated script."
            )
            return EvaluationSummary(metrics=metrics, analysis_prompt=analysis_prompt)
        except ParticipantVisibleError as exc:
            metrics = {
                "status": "error",
                "reason": str(exc),
                "num_classes": class_count,
                "num_samples": row_count,
            }
            return EvaluationSummary(
                metrics=metrics,
                analysis_prompt=(
                    "The experiment ran, but external evaluation failed with this metric error: "
                    f"{exc}. Fix the script so outputs are valid probabilities aligned with labels."
                ),
            )
