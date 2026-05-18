"""
Reporting helper — collect staged-pipeline experiment metrics into one chronological table.

Used by ``experiment_metrics_timeline.ipynb`` (not part of the training / meta-agent pipeline).

Sources (under ``logs/meta_agent/``):
  - ``**/experiment_memory.jsonl`` — per-trial runs (1a, 1b refine, 1c LLM aug, …)
  - ``arch_search_1a_results.json`` — 1a track champions
  - ``arch_search_1c_results.json`` — 1c preset / LLM aug trials
  - ``arch_search_1d_results.json`` / ``arch_search_1e_results.json`` — finalize milestones
  - ``cnn_arch_search_1c_results.json`` — CNN 1c (if present)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_LOGS = PROJECT_ROOT / "logs" / "meta_agent"

METRIC_COLUMNS = (
    "macro_average_precision",
    "median_per_class_auc",
    "competition_macro_auc",
    "macro_roc_auc",
    "soundscape_macro_ap",
)


def _parse_ts(value: str | None, fallback: float) -> datetime:
    if not value:
        return datetime.fromtimestamp(fallback, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.fromtimestamp(fallback, tz=timezone.utc)


def _score_dict_to_metrics(score: dict | None) -> dict[str, float | None]:
    if not score:
        return {k: None for k in METRIC_COLUMNS}
    return {
        "macro_average_precision": _f(
            score.get("macro_average_precision", score.get("primary_value"))
        ),
        "median_per_class_auc": _f(score.get("median_per_class_auc")),
        "competition_macro_auc": _f(
            score.get("competition_macro_auc", score.get("macro_roc_auc"))
        ),
        "macro_roc_auc": _f(score.get("macro_roc_auc")),
        "soundscape_macro_ap": _f(score.get("soundscape_macro_ap")),
    }


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _metrics_from_entry(entry: dict) -> dict[str, float | None]:
    out = {
        "macro_average_precision": _f(entry.get("macro_average_precision")),
        "median_per_class_auc": _f(entry.get("median_per_class_auc")),
        "competition_macro_auc": _f(
            entry.get("competition_macro_auc")
            or entry.get("macro_roc_auc")
        ),
        "macro_roc_auc": _f(entry.get("macro_roc_auc")),
        "soundscape_macro_ap": _f(entry.get("soundscape_macro_ap")),
    }
    nested = entry.get("metrics") or {}
    for key in METRIC_COLUMNS:
        if out[key] is None and nested.get(key) is not None:
            out[key] = _f(nested.get(key))
    if out["macro_average_precision"] is None:
        out["macro_average_precision"] = _f(nested.get("ranking_value"))
    return out


def _infer_context(jsonl_path: Path, meta_logs: Path) -> dict[str, str]:
    rel = jsonl_path.parent.relative_to(meta_logs)
    parts = list(rel.parts)
    track = "unknown"
    stage = "trial"
    label = jsonl_path.parent.name

    if parts and parts[0] in ("cnn", "perch", "birdnet"):
        track = parts[0]
        if len(parts) == 1 or parts[1] in ("high", "medium", "low"):
            stage = "1a"
            label = parts[1] if len(parts) > 1 else "default"
        elif parts[1] == "refine":
            stage = "1b"
            label = parts[2] if len(parts) > 2 else "refine"
        elif parts[1] == "aug_search":
            stage = "1c"
            label = parts[2] if len(parts) > 2 else "aug_search"
        else:
            label = "/".join(parts[1:])
    elif "aug_search" in parts:
        track = "perch" if "perch" in str(jsonl_path) else "cnn"
        stage = "1c"
        label = "llm_memory" if "_llm_memory" in str(jsonl_path) else "aug_search"

    return {"track": track, "stage": stage, "label": label}


def _row(
    *,
    ts: datetime,
    source: str,
    track: str,
    stage: str,
    label: str,
    success: bool,
    metrics: dict[str, float | None],
    arch_type: str | None = None,
    run_index: int | None = None,
) -> dict:
    row = {
        "timestamp": ts,
        "source": source,
        "track": track,
        "stage": stage,
        "label": label,
        "success": success,
        "arch_type": arch_type or "",
        "run_index": run_index,
    }
    row.update(metrics)
    return row


def collect_jsonl_runs(
    meta_logs: Path,
    *,
    exclude_backup: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(meta_logs.rglob("experiment_memory.jsonl")):
        if exclude_backup and "BACKUP" in path.parts:
            continue
        ctx = _infer_context(path, meta_logs)
        file_mtime = path.stat().st_mtime
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spec = entry.get("spec") or {}
                ts = _parse_ts(entry.get("timestamp"), file_mtime + i * 0.001)
                arch = spec.get("arch_type") or spec.get("preset_name") or ""
                label = ctx["label"]
                if spec.get("preset_name"):
                    label = str(spec["preset_name"])[:48]
                elif spec.get("aug_preset"):
                    label = str(spec["aug_preset"])[:48]
                elif arch:
                    label = f"{ctx['label']}:{arch}" if ctx["label"] != arch else arch
                slot = spec.get("slot")
                if slot:
                    label = f"{label} ({slot})"
                rows.append(
                    _row(
                        ts=ts,
                        source=str(path.relative_to(meta_logs)),
                        track=ctx["track"],
                        stage=ctx["stage"],
                        label=label[:80],
                        success=bool(entry.get("success")),
                        metrics=_metrics_from_entry(entry),
                        arch_type=arch or None,
                        run_index=i + 1,
                    )
                )
    return rows


def _collect_1a(meta_logs: Path) -> list[dict]:
    path = meta_logs / "arch_search_1a_results.json"
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for track, block in (data.get("tracks") or {}).items():
        winner = block.get("winner") or {}
        score = winner.get("score")
        arch = winner.get("arch_type", "")
        rows.append(
            _row(
                ts=_parse_ts(None, mtime),
                source=path.name,
                track=str(track),
                stage="1a",
                label=f"winner:{arch or track}",
                success=True,
                metrics=_score_dict_to_metrics(score),
                arch_type=arch or None,
            )
        )
    return rows


def _collect_1c_presets(meta_logs: Path, filename: str, track: str) -> list[dict]:
    path = meta_logs / filename
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key in ("preset_runs", "llm_runs", "runs"):
        for i, run in enumerate(data.get(key) or []):
            preset = (
                run.get("aug_preset")
                or run.get("preset")
                or run.get("preset_name")
                or f"run_{i}"
            )
            score = run.get("score")
            ts = _parse_ts(run.get("timestamp"), mtime + i * 0.01)
            rows.append(
                _row(
                    ts=ts,
                    source=f"{path.name}:{key}",
                    track=track,
                    stage="1c",
                    label=str(preset)[:80],
                    success=score is not None,
                    metrics=_score_dict_to_metrics(score),
                )
            )
    champ = data.get("champion") or data.get("winner") or data.get("best")
    if champ and champ.get("score"):
        rows.append(
            _row(
                ts=_parse_ts(None, mtime + 1),
                source=f"{path.name}:champion",
                track=track,
                stage="1c",
                label="champion",
                success=True,
                metrics=_score_dict_to_metrics(champ.get("score")),
            )
        )
    return rows


def _collect_finalize(meta_logs: Path) -> list[dict]:
    rows = []
    for fname, stage, _labels in (
        ("arch_search_1d_results.json", "1d", ("soundscape",)),
        ("arch_search_1e_results.json", "1e", ("1d_pre_pseudo", "1e_pseudo")),
    ):
        path = meta_logs / fname
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8"))
        if stage == "1d":
            sc = data.get("soundscape_score")
            if sc:
                rows.append(
                    _row(
                        ts=_parse_ts(None, mtime),
                        source=path.name,
                        track="perch",
                        stage="1d",
                        label="full_train_head",
                        success=bool(data.get("success")),
                        metrics=_score_dict_to_metrics(sc),
                    )
                )
        else:
            for i, key in enumerate(("score_1d", "score_1e")):
                sc = data.get(key)
                if sc:
                    rows.append(
                        _row(
                            ts=_parse_ts(None, mtime + i * 0.01),
                            source=path.name,
                            track="perch",
                            stage="1e",
                            label=key.replace("score_", ""),
                            success=bool(data.get("success")),
                            metrics=_score_dict_to_metrics(sc),
                        )
                    )
    return rows


def collect_experiment_timeline(
    meta_logs: Path | None = None,
    *,
    exclude_backup: bool = True,
    successes_only: bool = False,
) -> pd.DataFrame:
    """Build a single DataFrame of all known experiments, sorted by ``timestamp``."""
    meta_logs = Path(meta_logs or DEFAULT_META_LOGS)
    rows: list[dict] = []
    rows.extend(collect_jsonl_runs(meta_logs, exclude_backup=exclude_backup))
    rows.extend(_collect_1a(meta_logs))
    rows.extend(_collect_1c_presets(meta_logs, "arch_search_1c_results.json", "perch"))
    rows.extend(_collect_1c_presets(meta_logs, "cnn_arch_search_1c_results.json", "cnn"))
    rows.extend(_collect_finalize(meta_logs))

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "track",
                "stage",
                "label",
                "success",
                "arch_type",
                "run_index",
                *METRIC_COLUMNS,
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["experiment_id"] = range(1, len(df) + 1)
    if successes_only:
        df = df[df["success"]].reset_index(drop=True)
        df["experiment_id"] = range(1, len(df) + 1)
    return df


def export_timeline_csv(
    out_path: Path | None = None,
    **kwargs: Any,
) -> Path:
    df = collect_experiment_timeline(**kwargs)
    out_path = Path(out_path or (DEFAULT_META_LOGS / "experiment_timeline.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
