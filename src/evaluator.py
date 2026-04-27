"""Reads and summarises experiment metrics; manages the cross-run registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def load_registry() -> List[Dict[str, Any]]:
    """Loads the cross-run experiment registry, returning an empty list if missing."""
    from paths import registry_path
    path = registry_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_to_registry(entry: Dict[str, Any]) -> None:
    """Appends one experiment record to the shared registry."""
    from paths import registry_path
    path = registry_path()
    registry = load_registry()
    registry.append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _entry_status(entry: Dict[str, Any]) -> str:
    """Backward-compatible status: new entries have 'status', old have metrics.success."""
    if "status" in entry:
        return entry["status"]
    return "success" if entry.get("metrics", {}).get("success") else "failed"


def _entry_metric(entry: Dict[str, Any]) -> float:
    """Backward-compatible main metric: new entries have 'main_metric', old have metrics.val_auc."""
    if "main_metric" in entry:
        return float(entry["main_metric"])
    return float(entry.get("metrics", {}).get("val_auc", 0.0))


def get_best_experiment(registry: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Returns the registry entry with the highest main_metric among successful runs."""
    successful = [e for e in registry if _entry_status(e) == "success"]
    if not successful:
        return None
    return max(successful, key=_entry_metric)


def count_consecutive_failures(registry: List[Dict[str, Any]]) -> int:
    """Counts how many consecutive failed entries appear at the end of the registry."""
    count = 0
    for entry in reversed(registry):
        if _entry_status(entry) == "failed":
            count += 1
        else:
            break
    return count


def compute_strategy(registry: List[Dict[str, Any]]) -> str:
    """
    Returns a strategy label for the next experiment based on recent history.

    exploration            – no history, try anything
    last_failed_fix        – last run failed, minimal fix
    multiple_failures_simplify – 2+ consecutive failures, simplify
    improved_continue      – metric improved, continue direction
    succeeded_weak_change  – success but weak metric, tweak 1-2 params
    no_improvement_switch  – flat metrics across last 3 runs, try new arch
    """
    if not registry:
        return "exploration"

    if count_consecutive_failures(registry) >= 2:
        return "multiple_failures_simplify"

    last = registry[-1]
    if _entry_status(last) == "failed":
        return "last_failed_fix"

    best = get_best_experiment(registry)
    last_metric = _entry_metric(last)
    best_metric = _entry_metric(best) if best else 0.0

    if len(registry) > 1 and last_metric >= best_metric:
        return "improved_continue"

    recent = registry[-3:]
    recent_metrics = [_entry_metric(e) for e in recent if _entry_status(e) == "success"]
    if len(recent_metrics) >= 2 and max(recent_metrics) - min(recent_metrics) < 0.02:
        return "no_improvement_switch"

    return "succeeded_weak_change"


# ---------------------------------------------------------------------------

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
