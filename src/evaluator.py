"""Reads and summarises experiment metrics produced by generated training code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """
    Reads metrics.json written by the generated training script.

    Returns a failure dict if the file is missing or cannot be parsed.
    """
    if not metrics_path.exists():
        return {
            "success": False,
            "error_type": "missing_metrics",
            "error_message": "metrics.json was not written by the training script",
        }
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_type": "invalid_metrics",
            "error_message": f"Could not parse metrics.json: {e}",
        }


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    """Returns a one-block human-readable summary of a metrics dict."""
    if not metrics.get("success", False):
        error_type = metrics.get("error_type", "unknown")
        error_msg = metrics.get("error_message", "")
        return f"FAILED [{error_type}]: {error_msg}"

    lines = [
        f"  Model:    {metrics.get('model_type', '?')}",
        f"  Epochs:   {metrics.get('epochs_completed', '?')}",
        f"  Train loss: {metrics.get('train_loss', '?')}",
        f"  Val loss:   {metrics.get('val_loss', '?')}",
        f"  Val AUC:    {metrics.get('val_auc', '?')}",
        f"  Runtime:  {metrics.get('runtime_seconds', 0):.1f}s",
    ]
    return "\n".join(lines)
