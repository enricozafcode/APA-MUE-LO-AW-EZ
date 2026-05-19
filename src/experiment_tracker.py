"""
Aggregate experiment metrics into one timeline JSON and refresh progress plots.

Presentation only — does not change training, scoring, or memory semantics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMELINE = PROJECT_ROOT / "logs" / "meta_agent" / "experiment_timeline.json"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "logs" / "meta_agent" / "experiment_plots"

_SETTINGS: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_settings() -> dict[str, Any]:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS
    defaults = {
        "enabled": True,
        "timeline_json": str(DEFAULT_TIMELINE),
        "plots_dir": str(DEFAULT_PLOTS_DIR),
        "live_window": 5,
        "write_pipeline_plot": True,
        "terminal_notice": False,
    }
    cfg_path = PROJECT_ROOT / "configs" / "agent_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            user = (cfg.get("meta_agent") or {}).get("experiment_tracker") or {}
            if isinstance(user, dict):
                defaults.update(user)
        except Exception:
            pass
    _SETTINGS = defaults
    return _SETTINGS


def _stage_key(track: str, stage: str) -> str:
    return f"{track.lower()}_{stage.lower()}"


def _ranking_value(entry: dict, ranking_metric: str) -> float | None:
    if entry.get("soundscape_macro_ap") is not None:
        return float(entry["soundscape_macro_ap"])
    if ranking_metric == "macro_roc_auc" and entry.get("macro_roc_auc") is not None:
        return float(entry["macro_roc_auc"])
    if entry.get("macro_average_precision") is not None:
        return float(entry["macro_average_precision"])
    m = entry.get("metrics") or {}
    if m.get("soundscape_macro_ap") is not None:
        return float(m["soundscape_macro_ap"])
    if ranking_metric == "macro_roc_auc" and m.get("macro_roc_auc") is not None:
        return float(m["macro_roc_auc"])
    if m.get("macro_average_precision") is not None:
        return float(m["macro_average_precision"])
    return None


def _compact_experiment(
    entry: dict,
    *,
    seq: int,
    stage_key: str,
    stage_ctx: dict,
    memory_dir: Path,
    ranking_metric: str,
) -> dict[str, Any]:
    spec = entry.get("spec") or {}
    rv = _ranking_value(entry, ranking_metric)
    return {
        "seq": seq,
        "stage_key": stage_key,
        "track": stage_ctx.get("track"),
        "stage": stage_ctx.get("stage"),
        "stage_label": stage_ctx.get("label"),
        "memory_dir": str(memory_dir),
        "timestamp": entry.get("timestamp"),
        "slot": spec.get("slot"),
        "arch_type": spec.get("arch_type"),
        "strategy": spec.get("strategy"),
        "aug_preset": spec.get("aug_preset") or spec.get("aug_baseline"),
        "success": bool(entry.get("success")),
        "macro_average_precision": entry.get("macro_average_precision"),
        "median_per_class_auc": entry.get("median_per_class_auc"),
        "macro_roc_auc": entry.get("macro_roc_auc"),
        "train_loss": entry.get("train_loss"),
        "val_loss": entry.get("val_loss"),
        "soundscape_macro_ap": entry.get("soundscape_macro_ap"),
        "ranking_metric": ranking_metric,
        "ranking_value": rv,
        "hypothesis": (spec.get("hypothesis") or "")[:160],
    }


def _best_in_stage(experiments: list[dict], ranking_metric: str) -> dict | None:
    ok = [e for e in experiments if e.get("success")]
    if not ok:
        return None
    return max(
        ok,
        key=lambda e: (
            float(e["ranking_value"])
            if e.get("ranking_value") is not None
            else -1.0
        ),
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_experiment(
    entry: dict,
    *,
    stage_ctx: dict | None,
    memory_dir: Path,
    ranking_metric: str,
) -> None:
    """Append one experiment to the timeline and refresh plots. Never raises."""
    try:
        settings = _load_settings()
        if not settings.get("enabled", True):
            return
        if not stage_ctx:
            return

        track = str(stage_ctx.get("track", "unknown"))
        stage = str(stage_ctx.get("stage", "unknown"))
        label = str(stage_ctx.get("label") or f"{track.upper()} {stage}")
        sk = _stage_key(track, stage)

        timeline_path = Path(settings.get("timeline_json", DEFAULT_TIMELINE))
        plots_dir = Path(settings.get("plots_dir", DEFAULT_PLOTS_DIR))
        plots_dir.mkdir(parents=True, exist_ok=True)

        timeline: dict[str, Any]
        if timeline_path.is_file():
            try:
                timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            except Exception:
                timeline = {}
        else:
            timeline = {}

        timeline.setdefault("schema_version", 1)
        timeline.setdefault("experiments", [])
        timeline.setdefault("stages", {})
        timeline["ranking_metric"] = ranking_metric
        timeline["updated_at"] = _utc_now()
        if not timeline.get("run_started_at"):
            timeline["run_started_at"] = entry.get("timestamp") or _utc_now()

        seq = len(timeline["experiments"]) + 1
        row = _compact_experiment(
            entry,
            seq=seq,
            stage_key=sk,
            stage_ctx={**stage_ctx, "label": label},
            memory_dir=memory_dir,
            ranking_metric=ranking_metric,
        )
        timeline["experiments"].append(row)

        stage_bucket = timeline["stages"].setdefault(
            sk,
            {
                "track": track,
                "stage": stage,
                "label": label,
                "memory_dir": str(memory_dir),
                "experiments": [],
            },
        )
        stage_bucket["label"] = label
        stage_bucket["memory_dir"] = str(memory_dir)
        stage_bucket["experiments"].append(row)
        stage_bucket["experiment_count"] = len(stage_bucket["experiments"])
        stage_bucket["best"] = _best_in_stage(stage_bucket["experiments"], ranking_metric)

        _atomic_write_json(timeline_path, timeline)

        live_n = int(settings.get("live_window", 5))
        history_path = plots_dir / f"{sk}_history.png"
        live_path = plots_dir / f"{sk}_live.png"
        pipeline_path = plots_dir / "pipeline_progress.png"

        _write_stage_plots(
            stage_bucket["experiments"],
            stage_label=label,
            history_path=history_path,
            live_path=live_path,
            live_window=live_n,
            ranking_metric=ranking_metric,
        )
        stage_bucket.setdefault("plots", {})
        stage_bucket["plots"]["history"] = str(history_path)
        stage_bucket["plots"]["live"] = str(live_path)

        if settings.get("write_pipeline_plot", True):
            _write_pipeline_plot(timeline["experiments"], pipeline_path)
            timeline["pipeline_plot"] = str(pipeline_path)

        timeline["plots_dir"] = str(plots_dir)
        timeline["timeline_json"] = str(timeline_path)
        _atomic_write_json(timeline_path, timeline)

        if settings.get("terminal_notice"):
            print(
                f"  [metrics] {label} → {live_path.name}  "
                f"(timeline #{seq})",
                flush=True,
            )
    except Exception:
        return


def _write_stage_plots(
    experiments: list[dict],
    *,
    stage_label: str,
    history_path: Path,
    live_path: Path,
    live_window: int,
    ranking_metric: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ok = [e for e in experiments if e.get("success")]
    if not ok:
        return

    # ── history: scores (top) + training losses (bottom) ─────────────────────
    xs = list(range(1, len(experiments) + 1))
    ap_series: list[float | None] = []
    med_series: list[float | None] = []
    train_loss_series: list[float | None] = []
    val_loss_series: list[float | None] = []
    for e in experiments:
        ap_series.append(
            float(e["macro_average_precision"])
            if e.get("macro_average_precision") is not None
            else None
        )
        med_series.append(
            float(e["median_per_class_auc"])
            if e.get("median_per_class_auc") is not None
            else None
        )
        train_loss_series.append(
            float(e["train_loss"]) if e.get("train_loss") is not None else None
        )
        val_loss_series.append(
            float(e["val_loss"]) if e.get("val_loss") is not None else None
        )

    fig, (ax1, ax_loss) = plt.subplots(
        2, 1, figsize=(9, 6.2), dpi=120, sharex=True, height_ratios=[2, 1]
    )
    fig.suptitle(f"{stage_label} — metrics over experiments", fontsize=11)

    ax1.set_ylabel("macro AP", color="#2980b9")
    ax1.plot(
        xs,
        ap_series,
        color="#2980b9",
        marker="o",
        linewidth=1.6,
        label="macro AP",
    )
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("median AUC", color="#c0392b")
    ax2.plot(
        xs,
        med_series,
        color="#c0392b",
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label="median AUC",
    )
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    for i, e in enumerate(experiments):
        if not e.get("success"):
            ax1.scatter(
                [i + 1],
                [ap_series[i] if ap_series[i] is not None else 0],
                c="#95a5a6",
                s=28,
                zorder=3,
            )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    ax_loss.set_xlabel("Experiment # (stage order)")
    ax_loss.set_ylabel("loss")
    if any(v is not None for v in train_loss_series):
        ax_loss.plot(
            xs,
            train_loss_series,
            color="#8e44ad",
            marker="o",
            linewidth=1.4,
            label="train_loss",
        )
    if any(v is not None for v in val_loss_series):
        ax_loss.plot(
            xs,
            val_loss_series,
            color="#d35400",
            marker="s",
            linewidth=1.4,
            linestyle="--",
            label="val_loss",
        )
    ax_loss.grid(True, alpha=0.25)
    handles, labels = ax_loss.get_legend_handles_labels()
    if labels:
        ax_loss.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(history_path, bbox_inches="tight")
    plt.close(fig)

    # ── live: newest N successes + all-time best in stage ────────────────────
    recent = [e for e in experiments if e.get("success")][-live_window:]
    best = _best_in_stage(experiments, ranking_metric)
    selected: list[dict] = []
    seen_seq: set[int] = set()
    for e in recent + ([best] if best else []):
        if e and e.get("seq") not in seen_seq:
            selected.append(e)
            seen_seq.add(int(e["seq"]))
    if not selected:
        return

    labels = []
    for e in selected:
        slot = e.get("slot") or e.get("arch_type") or "run"
        tag = "★" if best and e.get("seq") == best.get("seq") else ""
        labels.append(f"{tag}{slot}"[:18])

    ap_vals = [
        float(e["macro_average_precision"]) if e.get("macro_average_precision") is not None else 0.0
        for e in selected
    ]
    med_vals = [
        float(e["median_per_class_auc"]) if e.get("median_per_class_auc") is not None else 0.0
        for e in selected
    ]
    tl_vals = [
        float(e["train_loss"]) if e.get("train_loss") is not None else 0.0 for e in selected
    ]
    vl_vals = [
        float(e["val_loss"]) if e.get("val_loss") is not None else 0.0 for e in selected
    ]

    x = range(len(selected))
    width = 0.18
    fig, (ax_score, ax_loss) = plt.subplots(
        2, 1, figsize=(max(7, len(selected) * 1.2), 5.8), dpi=120, height_ratios=[2, 1]
    )
    fig.suptitle(f"{stage_label} — latest {live_window} + best", fontsize=11)
    ax_score.bar([i - width for i in x], ap_vals, width=width, color="#2980b9", label="macro AP")
    ax_score.bar([i for i in x], med_vals, width=width, color="#c0392b", label="median AUC")
    ax_score.set_xticks(list(x))
    ax_score.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_score.set_ylabel("score")
    ax_score.legend(loc="upper right", fontsize=8)
    ax_score.grid(True, axis="y", alpha=0.25)

    ax_loss.bar([i - width / 2 for i in x], tl_vals, width=width, color="#8e44ad", label="train_loss")
    ax_loss.bar([i + width / 2 for i in x], vl_vals, width=width, color="#d35400", label="val_loss")
    ax_loss.set_xticks(list(x))
    ax_loss.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(live_path, bbox_inches="tight")
    plt.close(fig)


def _write_pipeline_plot(all_experiments: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ok = [e for e in all_experiments if e.get("success")]
    if len(ok) < 2:
        return

    xs = [e["seq"] for e in all_experiments]
    ap = [
        float(e["macro_average_precision"]) if e.get("macro_average_precision") is not None else None
        for e in all_experiments
    ]
    med = [
        float(e["median_per_class_auc"]) if e.get("median_per_class_auc") is not None else None
        for e in all_experiments
    ]
    train_loss = [
        float(e["train_loss"]) if e.get("train_loss") is not None else None
        for e in all_experiments
    ]
    val_loss = [
        float(e["val_loss"]) if e.get("val_loss") is not None else None
        for e in all_experiments
    ]
    stage_keys = [e.get("stage_key", "") for e in all_experiments]
    uniq = sorted({s for s in stage_keys if s})
    cmap = plt.get_cmap("tab10")
    color_map = {s: cmap(i % 10) for i, s in enumerate(uniq)}

    fig, (ax1, ax_loss) = plt.subplots(
        2, 1, figsize=(10, 6.5), dpi=120, sharex=True, height_ratios=[2, 1]
    )
    fig.suptitle("Pipeline progress — macro AP, median AUC, train/val loss", fontsize=11)
    ax1.set_ylabel("macro AP", color="#2980b9")
    ax1.plot(xs, ap, color="#2980b9", linewidth=1.4, alpha=0.85)
    ax2 = ax1.twinx()
    ax2.set_ylabel("median AUC", color="#c0392b")
    ax2.plot(xs, med, color="#c0392b", linewidth=1.2, linestyle="--", alpha=0.85)

    for i, sk in enumerate(stage_keys):
        if not all_experiments[i].get("success"):
            continue
        ax1.scatter(
            [xs[i]],
            [ap[i] if ap[i] is not None else 0],
            c=[color_map.get(sk, "#7f8c8d")],
            s=36,
            zorder=3,
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[s], markersize=8, label=s)
        for s in uniq
    ]
    if handles:
        ax1.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)

    ax_loss.set_xlabel("Global experiment #")
    ax_loss.set_ylabel("loss")
    if any(v is not None for v in train_loss):
        ax_loss.plot(xs, train_loss, color="#8e44ad", linewidth=1.2, alpha=0.85, label="train_loss")
    if any(v is not None for v in val_loss):
        ax_loss.plot(
            xs, val_loss, color="#d35400", linewidth=1.2, linestyle="--", alpha=0.85, label="val_loss"
        )
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def replot_from_timeline(timeline_path: Path | str | None = None) -> Path | None:
    """Rebuild all plots from an existing timeline JSON (offline use)."""
    settings = _load_settings()
    path = Path(timeline_path or settings.get("timeline_json", DEFAULT_TIMELINE))
    if not path.is_file():
        return None
    try:
        timeline = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    plots_dir = Path(settings.get("plots_dir", DEFAULT_PLOTS_DIR))
    plots_dir.mkdir(parents=True, exist_ok=True)
    live_n = int(settings.get("live_window", 5))
    metric = str(timeline.get("ranking_metric", "macro_average_precision"))
    for sk, bucket in (timeline.get("stages") or {}).items():
        exps = bucket.get("experiments") or []
        if not exps:
            continue
        label = bucket.get("label", sk)
        _write_stage_plots(
            exps,
            stage_label=label,
            history_path=plots_dir / f"{sk}_history.png",
            live_path=plots_dir / f"{sk}_live.png",
            live_window=live_n,
            ranking_metric=metric,
        )
    if settings.get("write_pipeline_plot", True):
        _write_pipeline_plot(timeline.get("experiments") or [], plots_dir / "pipeline_progress.png")
    return path
