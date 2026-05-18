"""
Meta Agent — BirdCLEF 2026
===========================
Legacy pipeline (``meta_agent.pipeline``: ``legacy``):
  Phase 0: EDA
  Phase 1–3: CNN / BirdNET / Perch single-pass agents
  Phase 4: Ensemble

Staged pipeline — **sequential tracks** (``meta_agent.track_order``):

  **Tournament** (``tournament.enabled`` or ``pipeline: staged_tournament``):
    Phase A: each track through 1c on sample → ``tournament_results.json`` picks global winner.
    Phase B: ``pipeline: staged_finalize`` (or ``tournament.auto_finalize: true``) runs 1d+1e
    for the winner only (full data + pseudo).

  **Non-tournament** (``staged_full`` with ``tournament.enabled: false``):
    Each track runs through ``pipeline`` max stage (1a…1e) before the next track starts.

Stage 1b (optional): refine top-K models from 1a with adaptive iteration budget.
Stage 1c: try fixed embedding aug presets (default: medium + high) with locked 1b head; optional ``mode: llm``.
Stage 1d: full-data final retrain (best aug + best head) → ``logs/meta_agent/perch/final/``.
Stage 1e: pseudo-label unlabeled soundscapes (hardcoded thresholds) + fine-tune the 1d head.
Use ``pipeline: staged_1c_only`` (Perch) or ``staged_cnn_1c_only`` (CNN) to re-run stage 1c after 1a/1b.
After a crashed tournament mid–Perch 1c: ``staged_perch_1c_only`` then ``staged_tournament_resume``.
Set ``stage_1c.mode`` to ``presets`` (recommended) | ``llm`` | ``both``; ``aug_presets``: [``medium``, ``high``].

Run:
    python src/meta_agent.py --config configs/agent_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from augmentation import (
    BASELINE_AUG_NAMES,
    describe_baseline,
    get_aug_search_preset,
    describe_embedding_aug_compact,
    get_audio_embedding_aug,
    get_cnn_baseline_aug,
    list_aug_search_preset_names,
    list_baseline_aug_names,
    normalize_baseline_aug_name,
)
from soundscape_evaluator import (
    PRIMARY_META_METRIC,
    SoundscapeEvalSuite,
    SoundscapeScore,
    competition_macro_auc,
    format_soundscape_metrics_line,
    format_soundscape_score,
    macro_average_precision,
    median_per_class_auc,
    primary_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / "notebooks" / "birdnet_cache"
PERCH_MEMORY = PROJECT_ROOT / "logs" / "perch_memory"
BIRDNET_LOGS = PROJECT_ROOT / "logs" / "birdnet_agent"
CNN_SUBMISSION = PROJECT_ROOT / "submission" / "model.keras"
META_LOGS    = PROJECT_ROOT / "logs" / "meta_agent"

SOUNDSCAPE_LABELS = DATA_DIR / "train_soundscapes_labels.csv"
TRAIN_SOUNDSCAPES = DATA_DIR / "train_soundscapes"

BIRDNET_VAL_CACHE = CACHE_DIR / "val_emb1024.npz"
META_VAL_CACHE    = META_LOGS / "common_val.npz"
SOUNDSCAPE_LEADERBOARD = META_LOGS / "soundscape_leaderboard.json"
ARCH_SEARCH_1A_RESULTS = META_LOGS / "arch_search_1a_results.json"
ARCH_SEARCH_1B_RESULTS = META_LOGS / "arch_search_1b_results.json"
ARCH_SEARCH_1C_RESULTS = META_LOGS / "arch_search_1c_results.json"
HEAD_TRAIN_INDICES = META_LOGS / "perch" / "head_train_indices_2000.npy"
HEAD_TRAIN_CLIPS = META_LOGS / "perch" / "head_train_clips_2000.jsonl"
PERCH_SHARED_VAL_CACHE = META_LOGS / "perch_cache" / "val_emb.npz"
PERCH_FINAL_DIR = META_LOGS / "perch" / "final"
CNN_FINAL_DIR = META_LOGS / "cnn" / "final"
CNN_ARCH_1B_RESULTS = META_LOGS / "cnn_arch_search_1b_results.json"
CNN_ARCH_1C_RESULTS = META_LOGS / "cnn_arch_search_1c_results.json"
CNN_ARCH_1D_RESULTS = META_LOGS / "cnn_arch_search_1d_results.json"
CNN_ARCH_1E_RESULTS = META_LOGS / "cnn_arch_search_1e_results.json"
CNN_PSEUDO_LABELS_NPZ = META_LOGS / "cnn_cache" / "pseudo_labels.npz"
BIRDNET_SHARED_VAL_CACHE = META_LOGS / "birdnet_cache" / "val_emb.npz"
BIRDNET_FINAL_DIR = META_LOGS / "birdnet" / "final"
BIRDNET_ARCH_1B_RESULTS = META_LOGS / "birdnet_arch_search_1b_results.json"
BIRDNET_ARCH_1C_RESULTS = META_LOGS / "birdnet_arch_search_1c_results.json"
BIRDNET_ARCH_1D_RESULTS = META_LOGS / "birdnet_arch_search_1d_results.json"
BIRDNET_ARCH_1E_RESULTS = META_LOGS / "birdnet_arch_search_1e_results.json"
BIRDNET_PSEUDO_LABELS_NPZ = META_LOGS / "birdnet_cache" / "pseudo_labels.npz"
TOURNAMENT_RESULTS = META_LOGS / "tournament_results.json"
ARCH_SEARCH_1D_RESULTS = META_LOGS / "arch_search_1d_results.json"
ARCH_SEARCH_1E_RESULTS = META_LOGS / "arch_search_1e_results.json"
PSEUDO_LABELS_NPZ = META_LOGS / "perch_cache" / "pseudo_labels.npz"

SR           = 32_000
CLIP_SEC     = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
PERCH_DIM    = 1536
PYTHON       = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _meta_primary_metric(config: dict) -> str:
    return str(
        config.get("meta_agent", {}).get("primary_metric", PRIMARY_META_METRIC)
    )


def _soundscape_suite(config: dict) -> SoundscapeEvalSuite:
    return SoundscapeEvalSuite(DATA_DIR, primary_metric=_meta_primary_metric(config))


def _print_soundscape_score(label: str, score: SoundscapeScore | None) -> None:
    if score is None:
        print(f"  [{label}] soundscape eval skipped (missing artifacts)")
        return
    print(
        f"  [{label}] {format_soundscape_score(score)} "
        f"| n_windows={score.n_windows} n_classes={score.n_scored_classes}"
    )


def _safe_copy2(src: Path, dst: Path) -> None:
    """Copy file; no-op when source and destination are the same path."""
    src_r, dst_r = Path(src).resolve(), Path(dst).resolve()
    if src_r == dst_r:
        return
    shutil.copy2(src_r, dst_r)


def _print_1c_phase_header(title: str, *, lines: list[str] | None = None) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for line in lines or []:
        print(f"  {line}")


def _short_aug_display_name(preset: str, spec: dict | None = None) -> str:
    if spec and spec.get("preset_name"):
        return str(spec["preset_name"])[:32]
    p = str(preset)
    for prefix in ("aug_r", "aug_"):
        if p.startswith(prefix):
            parts = p.split("_", 3)
            if len(parts) >= 4:
                return parts[3][:32]
    return p[:32]


def _print_1c_iteration_result(
    idx: int,
    total: int,
    label: str,
    score: SoundscapeScore | None,
    *,
    subset_ap: float | None = None,
    failed: bool = False,
) -> None:
    if failed:
        print(f"  {idx:2d}/{total}  {label:<30}  FAILED")
        return
    ap = float(score.primary_value) if score else float("nan")
    auc = float(score.competition_macro_auc) if score else float("nan")
    line = f"  {idx:2d}/{total}  {label:<30}  soundscape_AP={ap:.5f}  macro_AUC={auc:.5f}"
    if subset_ap is not None:
        line += f"  subset_AP={subset_ap:.5f}"
    print(line)


def _meta_cfg(config: dict) -> dict:
    return config.get("meta_agent", {})


def _baseline_names(config: dict) -> list[str]:
    names = _meta_cfg(config).get("aug_baselines")
    if names:
        return [str(n).lower() for n in names]
    return list_baseline_aug_names()


def _stage_1b_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1b") or {})


def _stage_1c_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1c") or {})


def _stage_1d_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1d") or {})


def _stage_1e_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("stage_1e") or {})


def _tournament_cfg(config: dict) -> dict:
    return dict(_meta_cfg(config).get("tournament") or {})


def _use_tournament_mode(config: dict, pipeline: str) -> bool:
    """Phase A: all tracks through best aug on sample (1c), then pick one global winner."""
    pipeline = str(pipeline).lower()
    if pipeline == "staged_finalize":
        return False
    if pipeline in ("staged_tournament",):
        return True
    return bool(_tournament_cfg(config).get("enabled", False))


def _should_auto_finalize(config: dict, pipeline: str) -> bool:
    if str(pipeline).lower() == "staged_finalize":
        return True
    return bool(_tournament_cfg(config).get("auto_finalize", False))


def _primary_value_from_score(score: dict | None, metric: str) -> float | None:
    if not score:
        return None
    if score.get("primary_value") is not None:
        return float(score["primary_value"])
    if metric == PRIMARY_META_METRIC:
        v = score.get("macro_average_precision")
    else:
        v = score.get("competition_macro_auc") or score.get("macro_roc_auc")
    return float(v) if v is not None else None


def _arch_iters_per_baseline(config: dict, track_iters: int) -> int:
    """Iterations per aug baseline when running staged 1a."""
    per = _meta_cfg(config).get("arch_search_iterations_per_aug")
    # Explicit positive value wins; 0 / null → split track_iters across aug baselines.
    if per is not None and int(per) > 0:
        return int(per)
    n_aug = max(1, len(_baseline_names(config)))
    return max(1, int(track_iters) // n_aug)


def _stage_1d_embed_cap(config: dict, cfg: dict) -> int | None:
    """
    Clip limit for stage-1d embed only.

    ``stage_1d.max_train_samples: null`` → all focal rows in train.csv (no cap).
    Search stages keep ``arch_search_embed_max_samples`` / ``perch.max_train_samples``.
    """
    if "max_train_samples" in cfg:
        cap = cfg.get("max_train_samples")
        return int(cap) if cap is not None else None
    full_frac = float(cfg.get("embed_sample_frac", 1.0))
    if full_frac >= 1.0:
        return None
    return _embed_max_samples(config)


def _embed_max_samples(config: dict) -> int | None:
    """Cap audio→embedding cache builds (Perch/BirdNET) during meta-agent search."""
    v = _meta_cfg(config).get("arch_search_embed_max_samples")
    if v is None:
        v = config.get("perch", {}).get("max_train_samples")
    return int(v) if v is not None else None


_PIPELINE_MAX_STAGE = {
    "staged_1a": "1a",
    "staged_1a_1b": "1b",
    "staged_1a_1b_1c": "1c",
    "staged_full": "1e",
}

_STAGE_RANK = {"1a": 1, "1b": 2, "1c": 3, "1d": 4, "1e": 5}


def _track_order(config: dict) -> list[str]:
    """Order in which full track pipelines run (one track finished before the next starts)."""
    meta = _meta_cfg(config)
    raw = meta.get("track_order", ["cnn", "perch"])
    valid = {"cnn", "birdnet", "perch"}
    excluded = {str(t).lower() for t in (meta.get("excluded_tracks") or [])}
    order = [
        str(t).lower()
        for t in raw
        if str(t).lower() in valid and str(t).lower() not in excluded
    ]
    return order or ["cnn", "perch"]


def _track_iterations(config: dict, track: str) -> int:
    meta = _meta_cfg(config)
    keys = {
        "cnn": "cnn_iterations",
        "birdnet": "birdnet_iterations",
        "perch": "perch_iterations",
    }
    return int(meta.get(keys[track], 0))


def _pipeline_max_stage(pipeline: str) -> str:
    return _PIPELINE_MAX_STAGE.get(str(pipeline).lower(), "1e")


def _pipeline_includes_stage(pipeline: str, stage: str) -> bool:
    max_s = _pipeline_max_stage(pipeline)
    return _STAGE_RANK[stage] <= _STAGE_RANK[max_s]


def _load_1a_summary(config: dict) -> dict:
    metric = _meta_primary_metric(config)
    baselines = _baseline_names(config)
    if ARCH_SEARCH_1A_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            summary.setdefault("tracks", {})
            return summary
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "stage": "1a_arch_search",
        "primary_metric": metric,
        "aug_baselines": baselines,
        "track_order": _track_order(config),
        "tracks": {},
    }


def _save_1a_summary(summary: dict) -> None:
    META_LOGS.mkdir(parents=True, exist_ok=True)
    ARCH_SEARCH_1A_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _1a_track_done(summary: dict, track: str) -> bool:
    entry = (summary.get("tracks") or {}).get(track)
    return bool(entry and entry.get("winner"))


def _track_active(config: dict, track: str, pipeline: str) -> bool:
    """True if this track should run (1a and/or follow-on stages enabled)."""
    if _track_iterations(config, track) > 0:
        return True
    if track == "cnn":
        return any(
            _pipeline_includes_stage(pipeline, stage)
            and _cnn_stage_cfg(config, f"stage_{stage}").get("enabled", False)
            for stage in ("1b", "1c", "1d", "1e")
        )
    if track == "perch":
        return any(
            _pipeline_includes_stage(pipeline, stage)
            and {
                "1b": _stage_1b_cfg,
                "1c": _stage_1c_cfg,
                "1d": _stage_1d_cfg,
                "1e": _stage_1e_cfg,
            }[stage](config).get("enabled", False)
            for stage in ("1b", "1c", "1d", "1e")
        )
    if track == "birdnet":
        if _track_iterations(config, track) > 0:
            return True
        return any(
            _pipeline_includes_stage(pipeline, stage)
            and _birdnet_stage_cfg(config, f"birdnet_stage_{stage}").get("enabled", False)
            for stage in ("1b", "1c", "1d", "1e")
        )
    return False


def run_stage_1a_for_track(
    track: str,
    config: dict,
    suite: SoundscapeEvalSuite,
) -> dict | None:
    """Stage 1a architecture search for a single track (CNN / birdnet / perch)."""
    track = track.lower()
    iters = _track_iterations(config, track)
    if iters <= 0:
        print(f"\n  [1a / {track}] Skipped ({track}_iterations=0)")
        return None

    baselines = _baseline_names(config)
    metric = _meta_primary_metric(config)
    cnn_max = int(_meta_cfg(config).get("arch_search_cnn_max_samples", 2000))
    embed_frac = float(_meta_cfg(config).get("arch_search_embed_sample_frac", 0.5))
    embed_max = _embed_max_samples(config)

    per = _arch_iters_per_baseline(config, iters)

    print("\n" + "=" * 60)
    print(f"  STAGE 1a — {track.upper()} architecture search")
    print(f"  Baselines: {', '.join(baselines)}  |  planner rounds/baseline: {per}")
    print(f"  Ranking metric: {metric}")
    print("=" * 60)

    if track == "cnn":
        runs = [_run_cnn_baseline_1a(config, b, per, cnn_max, suite) for b in baselines]
    elif track == "birdnet":
        runs = [_run_birdnet_baseline_1a(config, b, per, embed_frac, suite) for b in baselines]
    elif track == "perch":
        runs = [
            _run_perch_baseline_1a(config, b, per, embed_frac, embed_max, suite)
            for b in baselines
        ]
    else:
        return None

    winner = _pick_track_winner(runs, metric)
    entry = {"runs": runs, "winner": winner}
    if winner:
        sc = winner.get("score") or {}
        if track == "cnn":
            print(
                f"\n  ★ CNN 1a winner: aug={winner.get('aug_baseline')}  "
                f"{format_soundscape_metrics_line(macro_ap=sc.get('macro_average_precision'), macro_auc=sc.get('competition_macro_auc'), median_auc=sc.get('median_per_class_auc'), ranking_metric=metric)}"
            )
        elif track == "birdnet":
            print(
                f"\n  ★ BirdNET 1a winner: aug={winner.get('aug_baseline')}  "
                f"{format_soundscape_metrics_line(macro_ap=sc.get('macro_average_precision'), macro_auc=sc.get('competition_macro_auc'), median_auc=sc.get('median_per_class_auc'), ranking_metric=metric)}"
            )
        else:
            print(f"\n  ★ {track.upper()} 1a winner: aug={winner.get('aug_baseline')}")
    return entry


def _stage_included(max_stage: str, stage: str) -> bool:
    return _STAGE_RANK[stage] <= _STAGE_RANK[max_stage]


def _run_cnn_track_post_1a(
    config: dict, suite: SoundscapeEvalSuite, max_stage: str
) -> None:
    if _stage_included(max_stage, "1b") and _cnn_stage_cfg(config, "stage_1b").get(
        "enabled", False
    ):
        run_stage_1b_cnn_refine(config, suite)
    if _stage_included(max_stage, "1c") and _cnn_stage_cfg(config, "stage_1c").get(
        "enabled", False
    ):
        run_stage_1c_cnn_aug_search(config, suite)
    if _stage_included(max_stage, "1d") and _cnn_stage_cfg(config, "stage_1d").get(
        "enabled", False
    ):
        if not _skip_if_completed(config, CNN_ARCH_1D_RESULTS, "CNN 1d"):
            run_stage_1d_cnn_final_train(config)
    if _stage_included(max_stage, "1e") and _cnn_stage_cfg(config, "stage_1e").get(
        "enabled", False
    ):
        if not _skip_if_completed(config, CNN_ARCH_1E_RESULTS, "CNN 1e"):
            run_stage_1e_cnn_pseudo_refine(config, suite)


def _run_perch_track_post_1a(
    config: dict, suite: SoundscapeEvalSuite, max_stage: str
) -> None:
    if _stage_included(max_stage, "1b") and _stage_1b_cfg(config).get("enabled", False):
        run_stage_1b_perch_refine(config, suite)
    if _stage_included(max_stage, "1c") and _stage_1c_cfg(config).get("enabled", False):
        run_stage_1c_aug_search(config, suite)
    if _stage_included(max_stage, "1d") and _stage_1d_cfg(config).get("enabled", False):
        if not _skip_if_completed(config, ARCH_SEARCH_1D_RESULTS, "Perch 1d"):
            run_stage_1d_final_train(config)
    if _stage_included(max_stage, "1e") and _stage_1e_cfg(config).get("enabled", False):
        if not _skip_if_completed(config, ARCH_SEARCH_1E_RESULTS, "Perch 1e"):
            run_stage_1e_pseudo_refine(config, suite)


def _birdnet_stage_cfg(config: dict, key: str) -> dict:
    return dict(_meta_cfg(config).get(key) or {})


def _run_birdnet_track_post_1a(
    config: dict, suite: SoundscapeEvalSuite, max_stage: str
) -> None:
    if _stage_included(max_stage, "1b") and _birdnet_stage_cfg(config, "birdnet_stage_1b").get(
        "enabled", False
    ):
        from meta_birdnet_stages import run_stage_1b_birdnet_refine

        run_stage_1b_birdnet_refine(config, suite)
    if _stage_included(max_stage, "1c") and _birdnet_stage_cfg(config, "birdnet_stage_1c").get(
        "enabled", False
    ):
        from meta_birdnet_stages import run_stage_1c_birdnet_aug_search

        run_stage_1c_birdnet_aug_search(config, suite)
    if _stage_included(max_stage, "1d") and _birdnet_stage_cfg(config, "birdnet_stage_1d").get(
        "enabled", False
    ):
        if not _skip_if_completed(config, BIRDNET_ARCH_1D_RESULTS, "BirdNET 1d"):
            from meta_birdnet_stages import run_stage_1d_birdnet_final_train

            run_stage_1d_birdnet_final_train(config)
    if _stage_included(max_stage, "1e") and _birdnet_stage_cfg(config, "birdnet_stage_1e").get(
        "enabled", False
    ):
        if not _skip_if_completed(config, BIRDNET_ARCH_1E_RESULTS, "BirdNET 1e"):
            from meta_birdnet_stages import run_stage_1e_birdnet_pseudo_refine

            run_stage_1e_birdnet_pseudo_refine(config, suite)


def _run_track_post_1a(
    track: str,
    config: dict,
    suite: SoundscapeEvalSuite,
    max_stage: str,
) -> None:
    if track == "cnn":
        _run_cnn_track_post_1a(config, suite, max_stage)
    elif track == "birdnet":
        _run_birdnet_track_post_1a(config, suite, max_stage)
    elif track == "perch":
        _run_perch_track_post_1a(config, suite, max_stage)


def _collect_tournament_candidate(track: str, config: dict, metric: str) -> dict | None:
    """Best sample metric for a track after aug search (1c; BirdNET falls back to 1a)."""
    track = track.lower()
    checkpoint = "1c"
    winner: dict | None = None
    results_path: Path | None = None
    locked_arch_path: str | None = None

    if track == "cnn":
        results_path = CNN_ARCH_1C_RESULTS
        if not results_path.exists():
            return None
        try:
            summary = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        winner = summary.get("winner")
        locked = summary.get("locked_slot_code")
        if locked:
            locked_arch_path = str(locked)
        elif winner:
            slot = Path(winner.get("memory_dir", "")) / "best_model_slot.py"
            if slot.exists():
                locked_arch_path = str(slot)
    elif track == "perch":
        results_path = ARCH_SEARCH_1C_RESULTS
        if not results_path.exists():
            return None
        try:
            summary = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        winner = summary.get("winner")
        locked = summary.get("locked_head_code")
        if locked:
            locked_arch_path = str(locked)
        elif winner:
            mem = Path(winner.get("memory_dir", ""))
            for name in ("best_head_code.py", "best_model_slot.py"):
                p = mem / name
                if p.exists():
                    locked_arch_path = str(p)
                    break
    elif track == "birdnet":
        checkpoint = "1c"
        results_path = BIRDNET_ARCH_1C_RESULTS
        if not results_path.exists():
            checkpoint = "1a"
            if not ARCH_SEARCH_1A_RESULTS.exists():
                return None
            try:
                summary = json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            winner = (summary.get("tracks") or {}).get("birdnet", {}).get("winner")
        else:
            try:
                summary = json.loads(results_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            winner = summary.get("winner")
            locked = summary.get("locked_head_code")
            if locked:
                locked_arch_path = str(locked)
            elif winner:
                p = Path(winner.get("memory_dir", "")) / "best_head_code.py"
                if p.exists():
                    locked_arch_path = str(p)
    else:
        return None

    if not winner:
        return None

    score = dict(winner.get("score") or {})
    primary = _primary_value_from_score(score, metric)
    if primary is None:
        mem_dir = winner.get("memory_dir") or winner.get("logs_dir")
        if mem_dir and track == "cnn":
            sc = _score_cnn_arch_search(Path(mem_dir), _soundscape_suite(config))
            if sc:
                primary = sc.primary_value
                score = sc.to_dict()
        elif mem_dir and track == "perch":
            sc = _soundscape_suite(config).score_perch(Path(mem_dir))
            if sc:
                primary = sc.primary_value
                score = sc.to_dict()
        elif mem_dir and track == "birdnet":
            sc = _soundscape_suite(config).score_birdnet_mem_dir(
                Path(mem_dir), val_cache=BIRDNET_SHARED_VAL_CACHE
            )
            if sc:
                primary = sc.primary_value
                score = sc.to_dict()
    if primary is None:
        return None

    mem_dir = winner.get("memory_dir") or winner.get("logs_dir")
    arch_type = winner.get("arch_type")
    trial_id = winner.get("trial_id")
    if results_path and results_path.exists():
        try:
            summary = json.loads(results_path.read_text(encoding="utf-8"))
            rw = summary.get("refine_winner_1b") or {}
            arch_type = arch_type or rw.get("arch_type") or (rw.get("spec") or {}).get(
                "arch_type"
            )
        except (json.JSONDecodeError, OSError):
            pass
    if mem_dir:
        info_path = Path(mem_dir) / "best_model_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                spec = info.get("spec") or {}
                arch_type = arch_type or spec.get("arch_type") or info.get("arch_type")
                if score.get("macro_average_precision") is None:
                    score["macro_average_precision"] = info.get("macro_average_precision")
                if score.get("competition_macro_auc") is None:
                    score["competition_macro_auc"] = info.get("macro_roc_auc")
                if score.get("median_per_class_auc") is None:
                    score["median_per_class_auc"] = info.get("median_per_class_auc")
            except (json.JSONDecodeError, OSError):
                pass

    macro_ap = score.get("macro_average_precision")
    macro_auc = score.get("competition_macro_auc") or score.get("macro_roc_auc")
    median_auc = score.get("median_per_class_auc")

    return {
        "track": track,
        "checkpoint": checkpoint,
        "primary_metric": metric,
        "primary_value": primary,
        "macro_average_precision": macro_ap,
        "macro_roc_auc": macro_auc,
        "competition_macro_auc": macro_auc,
        "median_per_class_auc": median_auc,
        "aug_preset": winner.get("aug_preset") or winner.get("aug_baseline"),
        "aug_baseline": winner.get("aug_baseline"),
        "trial_id": trial_id,
        "memory_dir": mem_dir,
        "locked_arch_path": locked_arch_path,
        "results_path": str(results_path) if results_path else None,
        "arch_type": arch_type or "unknown",
    }


def _pick_global_tournament_winner(
    candidates: list[dict], metric: str
) -> dict | None:
    ok = [c for c in candidates if c.get("primary_value") is not None]
    if not ok:
        return None
    return max(ok, key=lambda c: float(c["primary_value"]))


def _tournament_metrics_line(row: dict, metric: str) -> str:
    """Same display as stage 1b/1c model selection: macro_AP | macro_AUC | median_AUC."""
    ap = row.get("macro_average_precision")
    auc = row.get("competition_macro_auc") or row.get("macro_roc_auc")
    med = row.get("median_per_class_auc")
    return format_soundscape_metrics_line(
        macro_ap=float(ap) if ap is not None else None,
        macro_auc=float(auc) if auc is not None else None,
        median_auc=float(med) if med is not None else None,
        ranking_metric=metric,
        mark_ranking=True,
    )


def _print_tournament_final_comparison(
    config: dict,
    candidates: list[dict],
    metric: str,
    global_winner: dict | None = None,
) -> None:
    """Print each track's best model (CNN / BirdNET / Perch) with full soundscape metrics."""
    order = _track_order(config)
    by_track = {str(c.get("track", "")).lower(): c for c in candidates}
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("primary_value") or -1.0),
        reverse=True,
    )
    rank_by_track = {
        str(r.get("track", "")).lower(): i + 1 for i, r in enumerate(ranked)
    }

    print("\n" + "=" * 72)
    print("  TOURNAMENT — FINAL COMPARISON (best model per architecture, sample eval)")
    print(f"  Ranking metric: {metric} on labeled train_soundscapes")
    print("  " + "─" * 72)

    for track in order:
        row = by_track.get(track)
        if row is None:
            print(f"\n  {track.upper():8s}  —  (not entered / no valid score)")
            continue

        rank = rank_by_track.get(track, "?")
        is_winner = (
            global_winner is not None
            and str(global_winner.get("track", "")).lower() == track
        )
        winner_tag = "  ← GLOBAL WINNER" if is_winner else ""
        mem_name = Path(row.get("memory_dir") or "?").name
        arch = row.get("arch_type", "?")
        aug = row.get("aug_preset") or row.get("aug_baseline", "?")
        checkpoint = row.get("checkpoint", "?")
        trial = row.get("trial_id")

        print(f"\n  Rank #{rank}  {track.upper()}{winner_tag}")
        print(f"    arch_type      : {arch}")
        print(f"    aug (best)     : {aug}")
        print(f"    stage reached  : {checkpoint} (sample train / embed)")
        if trial:
            print(f"    trial_id       : {trial}")
        print(f"    artifacts      : {mem_name}")
        print(f"    {_tournament_metrics_line(row, metric)}")
        rv = row.get("primary_value")
        if rv is not None:
            print(f"    primary_value  : {float(rv):.5f}  ({metric})")

    print("\n  " + "─" * 72)
    if global_winner:
        gw = global_winner
        print(
            f"  Selected for finalize (1d+1e): {str(gw.get('track', '?')).upper()}  "
            f"|  {_tournament_metrics_line(gw, metric)}"
        )
    else:
        print("  No global winner — cannot run finalize.")
    print("=" * 72)


def _save_tournament_results(
    config: dict,
    candidates: list[dict],
    global_winner: dict | None,
) -> dict:
    payload = {
        "stage": "tournament",
        "primary_metric": _meta_primary_metric(config),
        "track_order": _track_order(config),
        "candidates": candidates,
        "global_winner": global_winner,
        "finalize_ready": global_winner is not None,
    }
    META_LOGS.mkdir(parents=True, exist_ok=True)
    TOURNAMENT_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_staged_tournament_phase(
    config: dict,
    suite: SoundscapeEvalSuite,
    pipeline: str,
) -> dict:
    """Phase A: all tracks through 1c on sample; pick global winner (no 1d/1e)."""
    order = _track_order(config)
    metric = _meta_primary_metric(config)
    meta = _meta_cfg(config)

    print("\n" + "=" * 60)
    print("  TOURNAMENT PHASE A — stop after best aug on sample (stage 1c)")
    print(f"  Track order: {' → '.join(order)}")
    print("=" * 60)

    ensure_meta_eda_before_tracks(config)

    summary = _load_1a_summary(config)
    candidates: list[dict] = []

    for track in order:
        print(f"\n{'#' * 60}\n  TRACK: {track.upper()}\n{'#' * 60}")
        if not _track_active(config, track, pipeline):
            print("  Skipped — not active for this pipeline.")
            continue

        if _track_iterations(config, track) > 0:
            if meta.get("skip_completed_stages") and _1a_track_done(summary, track):
                print(f"  [1a / {track}] Skipped — already in {ARCH_SEARCH_1A_RESULTS.name}")
            else:
                entry = run_stage_1a_for_track(track, config, suite)
                if entry is not None:
                    summary.setdefault("tracks", {})[track] = entry
                    _save_1a_summary(summary)
        else:
            print(f"  [1a / {track}] Skipped — using existing 1a artifacts")

        _run_track_post_1a(track, config, suite, max_stage="1c")

        cand = _collect_tournament_candidate(track, config, metric)
        if cand:
            candidates.append(cand)
            print(
                f"\n  [Tournament] {track} qualified: "
                f"{metric}={float(cand['primary_value']):.5f}"
            )
        else:
            print(f"\n  [Tournament] {track} — no score (complete 1c or 1a first)")

    global_winner = _pick_global_tournament_winner(candidates, metric)
    _print_tournament_final_comparison(config, candidates, metric, global_winner)
    payload = _save_tournament_results(config, candidates, global_winner)

    if global_winner:
        print(f"\n  Tournament results saved → {TOURNAMENT_RESULTS}")
        print("  Next: pipeline staged_finalize (or tournament.auto_finalize: true)")
    else:
        print("\n  [Tournament] No global winner — no track has a valid sample score.")

    return payload


def run_staged_finalize_winner(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    winner: dict | None = None,
) -> dict:
    """Phase B: full train + pseudo for the tournament global winner only."""
    if winner is None:
        if not TOURNAMENT_RESULTS.exists():
            print("\n  [Finalize] Missing tournament_results.json — run tournament first.")
            return {"success": False}
        try:
            payload = json.loads(TOURNAMENT_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print("\n  [Finalize] Could not read tournament results.")
            return {"success": False}
        winner = payload.get("global_winner")
        if not winner:
            print("\n  [Finalize] No global_winner in tournament results.")
            return {"success": False}

    track = str(winner.get("track", "")).lower()
    metric = _meta_primary_metric(config)

    print("\n" + "=" * 60)
    print("  TOURNAMENT PHASE B — finalize global winner only")
    print(f"  Track: {track.upper()}  |  aug={winner.get('aug_preset')}")
    print(f"  {_tournament_metrics_line(winner, metric)}")
    print("=" * 60)

    ok = True
    if track == "cnn":
        if _cnn_stage_cfg(config, "stage_1d").get("enabled", True):
            if not _skip_if_completed(config, CNN_ARCH_1D_RESULTS, "CNN 1d finalize"):
                r = run_stage_1d_cnn_final_train(config)
                ok = ok and bool(r.get("success"))
        if _cnn_stage_cfg(config, "stage_1e").get("enabled", True):
            if not _skip_if_completed(config, CNN_ARCH_1E_RESULTS, "CNN 1e finalize"):
                r = run_stage_1e_cnn_pseudo_refine(config, suite)
                ok = ok and bool(r.get("success"))
    elif track == "perch":
        if _stage_1d_cfg(config).get("enabled", True):
            if not _skip_if_completed(config, ARCH_SEARCH_1D_RESULTS, "Perch 1d finalize"):
                run_stage_1d_final_train(config, suite)
        if _stage_1e_cfg(config).get("enabled", True):
            if not _skip_if_completed(config, ARCH_SEARCH_1E_RESULTS, "Perch 1e finalize"):
                run_stage_1e_pseudo_refine(config, suite)
    elif track == "birdnet":
        if _birdnet_stage_cfg(config, "birdnet_stage_1d").get("enabled", True):
            if not _skip_if_completed(config, BIRDNET_ARCH_1D_RESULTS, "BirdNET 1d"):
                from meta_birdnet_stages import run_stage_1d_birdnet_final_train

                r = run_stage_1d_birdnet_final_train(config)
                ok = ok and bool(r.get("success"))
        if _birdnet_stage_cfg(config, "birdnet_stage_1e").get("enabled", True):
            if not _skip_if_completed(config, BIRDNET_ARCH_1E_RESULTS, "BirdNET 1e"):
                from meta_birdnet_stages import run_stage_1e_birdnet_pseudo_refine

                r = run_stage_1e_birdnet_pseudo_refine(config, suite)
                ok = ok and bool(r.get("success"))
    else:
        print(f"\n  [Finalize] Unknown track: {track}")
        ok = False

    summary = {"stage": "tournament_finalize", "global_winner": winner, "success": ok}
    out = META_LOGS / "tournament_finalize_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Finalize summary → {out}")
    return summary


def run_staged_pipeline_sequential(
    config: dict,
    suite: SoundscapeEvalSuite,
    pipeline: str,
) -> None:
    """
    Sequential tracks (``track_order``). With ``tournament.enabled``, Phase A stops
    at 1c and picks a global winner; Phase B (finalize) runs 1d+1e for the winner only.
    """
    pipeline = str(pipeline).lower()
    if _use_tournament_mode(config, pipeline):
        run_staged_tournament_phase(config, suite, pipeline)
        if _should_auto_finalize(config, pipeline):
            run_staged_finalize_winner(config, suite)
        return

    order = _track_order(config)
    max_stage = _pipeline_max_stage(pipeline)
    meta = _meta_cfg(config)

    print("\n" + "=" * 60)
    print("  STAGED PIPELINE — sequential tracks (full per track)")
    print(f"  Order: {' → '.join(order)}  |  Through stage: {max_stage}")
    print("=" * 60)

    ensure_meta_eda_before_tracks(config)

    summary = _load_1a_summary(config)
    if max_stage == "1a" and _skip_if_completed(
        config, ARCH_SEARCH_1A_RESULTS, "Stage 1a (all tracks)"
    ):
        pass
    else:
        for track in order:
            print(f"\n{'#' * 60}\n  TRACK: {track.upper()}\n{'#' * 60}")
            if not _track_active(config, track, pipeline):
                print(
                    f"  Skipped (set {track}_iterations > 0 and/or enable "
                    f"{track} stage_1b–1e / cnn_stage_* in config)."
                )
                continue

            if _track_iterations(config, track) > 0:
                if meta.get("skip_completed_stages") and _1a_track_done(summary, track):
                    print(
                        f"  [1a / {track}] Skipped — already in "
                        f"{ARCH_SEARCH_1A_RESULTS.name}"
                    )
                else:
                    entry = run_stage_1a_for_track(track, config, suite)
                    if entry is not None:
                        summary.setdefault("tracks", {})[track] = entry
                        _save_1a_summary(summary)
                        print(f"  [1a / {track}] Saved → {ARCH_SEARCH_1A_RESULTS.name}")
            else:
                print(f"  [1a / {track}] Skipped — using existing 1a artifacts")

            _run_track_post_1a(track, config, suite, max_stage)

    print(f"\n  Step 1a summary (all tracks) → {ARCH_SEARCH_1A_RESULTS}")


def _score_cnn_arch_search(logs_dir: Path, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    """Score the CNN staged champion on labeled soundscapes (no final keras needed)."""
    logs_dir = Path(logs_dir)
    promoted = logs_dir / "best_val_preds.npy"
    if promoted.exists():
        return suite.score_arrays(np.load(str(promoted)).astype(np.float32))

    eval_dir = logs_dir / "eval_artifacts"
    info_path = logs_dir / "best_model_info.json"
    run_ids: list[str] = []
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            rid = str(info.get("run_id") or "").strip()
            if rid:
                run_ids.append(rid)
        except (json.JSONDecodeError, OSError):
            pass

    staged_path = logs_dir / "staged_results.json"
    results: list[dict] = []
    if staged_path.exists():
        try:
            results = json.loads(staged_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            results = []
    rr_path = logs_dir / "random_results.json"
    if rr_path.exists():
        try:
            results = results + json.loads(rr_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    ok = [r for r in results if r.get("success")]
    if ok:
        rank_key = (
            "macro_average_precision"
            if suite.primary_metric == PRIMARY_META_METRIC
            else "macro_roc_auc"
        )

        def _rank_val(r: dict) -> float:
            v = r.get("ranking_value") or r.get(rank_key) or r.get("macro_roc_auc")
            return float(v) if v is not None else -1.0

        best = max(ok, key=_rank_val)
        base_rid = str(best.get("run_id", "")).strip()
        if base_rid and base_rid not in run_ids:
            run_ids.append(base_rid)

    # Prefer _a1 (latest successful attempt) before _a2, … when resolving y_pred artifacts.
    attempt_suffixes = ("_a1", "_a2", "_a3", "_a4", "_a5", "")
    for run_id in run_ids:
        for suffix in attempt_suffixes:
            yp_path = eval_dir / f"y_pred_{run_id}{suffix}.npy"
            if yp_path.exists():
                return suite.score_arrays(np.load(str(yp_path)).astype(np.float32))
        yp_path = eval_dir / f"y_pred_{run_id}.npy"
        if yp_path.exists():
            return suite.score_arrays(np.load(str(yp_path)).astype(np.float32))
    return None


# Truncated ``eda_brief.txt`` body for CNN/Perch system prompts (set by Phase 0).
_META_EDA_BRIEF: str = ""
DEFAULT_EDA_BRIEF_MAX_CHARS = 600


def _eda_logs_dir() -> Path:
    return PROJECT_ROOT / "logs" / "eda"


def _eda_brief_max_chars(config: dict) -> int:
    meta_eda = _meta_cfg(config).get("eda") or {}
    eda_cfg = config.get("eda") or {}
    return int(
        meta_eda.get("brief_max_chars")
        or eda_cfg.get("brief_max_chars")
        or DEFAULT_EDA_BRIEF_MAX_CHARS
    )


def _strip_eda_report_header(text: str) -> str:
    """Drop markdown report headers; keep only injectable body."""
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines).strip()


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def prepare_eda_brief_for_prompts(raw: str, config: dict) -> str:
    """Strip report headers and enforce ``brief_max_chars`` before prompt injection."""
    return _truncate_for_prompt(
        _strip_eda_report_header(raw),
        _eda_brief_max_chars(config),
    )


def _run_subprocess(
    script: str,
    config_override: dict,
    base_config: dict,
    *,
    quiet: bool = False,
) -> int:
    """Write a temp config with overrides and run a script as subprocess."""
    cfg = json.loads(json.dumps(base_config))
    cfg.update(config_override)
    # Child agents (CNN / Perch) must not re-run EDA — only meta Phase 0 does.
    if script != "eda_agent.py":
        cfg.setdefault("eda", {})
        if not config_override.get("eda", {}).get("enabled", False):
            cfg["eda"]["enabled"] = False
        if script in ("cnn_agent.py", "perch_agent.py") and _META_EDA_BRIEF:
            cfg["eda_brief"] = _META_EDA_BRIEF
    tmp = Path(tempfile.gettempdir()) / f"meta_{Path(script).stem}_config.json"
    tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    cmd = [PYTHON, str(PROJECT_ROOT / "src" / script), "--config", str(tmp)]
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if quiet:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"  [subprocess {script}] failed (exit {result.returncode})")
            if err:
                print(err[-1200:])
        return result.returncode
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=child_env)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — EDA
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0_eda(config: dict) -> str:
    """
    Run EDA in-process (summary + 2-sentence brief). Returns truncated brief for prompts.
    """
    global _META_EDA_BRIEF

    print("\n" + "=" * 60)
    print("  PHASE 0 — Autonomous EDA")
    print("=" * 60)

    try:
        from eda_agent import build_eda_clients, load_eda_brief, run_eda_phase
    except ImportError:
        from .eda_agent import build_eda_clients, load_eda_brief, run_eda_phase

    logs_dir = _eda_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_eda = _meta_cfg(config).get("eda") or {}
    force_rebuild = bool(meta_eda.get("force_rebuild", False))
    executor, llm, max_wall = build_eda_clients(config)
    temperature = float(
        meta_eda.get("temperature")
        or config.get("llm", {}).get("temperature", 0.4)
    )

    try:
        run_eda_phase(
            executor,
            llm,
            logs_dir,
            temperature=temperature,
            use_llm_codegen=bool(meta_eda.get("use_llm_codegen", True)),
            max_codegen_attempts=int(meta_eda.get("max_codegen_attempts", 5)),
            max_codegen_wall_seconds=int(
                meta_eda.get("max_codegen_wall_seconds", max_wall)
            ),
            force_rebuild=force_rebuild,
            write_brief=True,
        )
    except Exception as exc:
        print(f"  [Phase 0] EDA failed ({exc}) — continuing without brief.")

    raw_brief = load_eda_brief(logs_dir)
    brief = prepare_eda_brief_for_prompts(raw_brief, config) if raw_brief.strip() else ""
    _META_EDA_BRIEF = brief

    if brief:
        cap = _eda_brief_max_chars(config)
        print(
            f"  [Phase 0] EDA brief ready ({len(brief)} chars, cap {cap}) "
            f"→ {_eda_logs_dir() / 'eda_brief.txt'}"
        )
    else:
        print("  [Phase 0] No EDA brief — CNN/Perch will run without data insights.")

    print("=" * 60)
    return brief


def load_meta_eda_brief(config: dict) -> str:
    """Load ``eda_brief.txt`` from disk without re-running EDA."""
    try:
        from eda_agent import load_eda_brief
    except ImportError:
        from .eda_agent import load_eda_brief

    raw = load_eda_brief(_eda_logs_dir())
    if not raw.strip():
        return ""
    return prepare_eda_brief_for_prompts(raw, config)


def ensure_meta_eda_before_tracks(config: dict, *, allow_run: bool = True) -> str:
    """
    Phase 0 before the first track: run EDA when ``run_eda`` is true, else optionally
    load a cached brief (``meta_agent.eda.use_cached_brief``).

    Set ``allow_run=False`` on finalize-only pipelines to skip a full EDA re-run while
    still injecting a cached brief into subprocess prompts.
    """
    global _META_EDA_BRIEF

    if _META_EDA_BRIEF:
        return _META_EDA_BRIEF

    meta = _meta_cfg(config)
    meta_eda = meta.get("eda") or {}

    if allow_run and meta.get("run_eda", False):
        return run_phase0_eda(config)

    if meta_eda.get("use_cached_brief", False):
        brief = load_meta_eda_brief(config)
        _META_EDA_BRIEF = brief
        if brief:
            print(
                f"  [EDA] Using cached brief ({len(brief)} chars, "
                f"cap {_eda_brief_max_chars(config)})"
            )
        return brief

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — CNN
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_cnn(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 1] CNN skipped (cnn_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 1 — CNN  ({n_iterations} iterations)")
    print("=" * 60)

    # Build a search config that respects n_iterations as the total budget
    base_search = config.get("search", {})
    cnn_search_override = {
        **base_search,
        "linear_budget": 0,
        "random_budget": n_iterations,
        "cnn_exploration": {"enabled": True},
        "transfer_exploration": {
            "enabled": True,
            "max_iterations": max(1, n_iterations // 2),
            "interactive_pick_final": False,
        },
        "medium_stage": {"enabled": False},
        "reality_gate": {"enabled": False},
        "phase2": {
            **base_search.get("phase2", {}),
            "random_experiments":          n_iterations,
            "focused_experiments":         0,
            "tweak_experiments":           0,
            "augmentation_tweak_experiments": 0,
            "ai_free_experiments":         0,
            "final_tweak_experiments":     0,
        },
    }
    rc = _run_subprocess("cnn_agent.py", {"search": cnn_search_override}, config)
    if rc != 0:
        print("  [Phase 1] CNN agent finished with errors.")

    if not CNN_SUBMISSION.exists():
        print(f"  [Phase 1] No CNN model at {CNN_SUBMISSION} — skipping soundscape eval.")
        return None

    print("  [Phase 1] Soundscape eval (labeled train_soundscapes) …")
    score = suite.score_cnn(CNN_SUBMISSION)
    _print_soundscape_score("CNN", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BirdNET
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_birdnet(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 2] BirdNET skipped (birdnet_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 2 — BirdNET  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("birdnet_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 2] BirdNET agent finished with errors.")

    print("  [Phase 2] Soundscape eval on best BirdNET val predictions …")
    score = suite.score_birdnet_artifacts(BIRDNET_LOGS, BIRDNET_VAL_CACHE)
    _print_soundscape_score("BirdNET", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Perch
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_perch(config: dict, n_iterations: int, suite: SoundscapeEvalSuite) -> SoundscapeScore | None:
    if n_iterations <= 0:
        print("\n  [Phase 3] Perch skipped (perch_iterations=0)")
        return None

    print("\n" + "=" * 60)
    print(f"  PHASE 3 — Perch  ({n_iterations} iterations)")
    print("=" * 60)

    rc = _run_subprocess("perch_agent.py", {"max_iterations": n_iterations}, config)
    if rc != 0:
        print("  [Phase 3] Perch agent finished with errors.")

    head = PERCH_MEMORY / "best_head.keras"
    if not head.exists():
        head = PERCH_MEMORY / "final_head.keras"
    if not head.exists():
        print("  [Phase 3] No Perch head in logs/perch_memory — skipping soundscape eval.")
        return None

    print("  [Phase 3] Soundscape eval (ONNX Perch + best head) …")
    score = suite.score_perch(PERCH_MEMORY)
    _print_soundscape_score("Perch", score)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Build common val + ensemble search
# ─────────────────────────────────────────────────────────────────────────────

def _load_perch_onnx(config: dict):
    import onnxruntime as ort
    import kagglehub
    slug      = config.get("perch", {}).get("onnx_dataset", "rishikeshjani/perch-onnx-for-birdclef-2026")
    onnx_dir  = Path(kagglehub.dataset_download(slug))
    onnx_path = next(onnx_dir.rglob("*.onnx"))
    so        = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess      = ort.InferenceSession(str(onnx_path), sess_options=so,
                                     providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    dummy     = np.zeros((1, CLIP_SAMPLES), dtype=np.float32)
    outs      = sess.run(None, {inp_name: dummy})
    emb_idx   = next(i for i, o in enumerate(outs) if o.ndim == 2 and o.shape[-1] == PERCH_DIM)
    return sess, inp_name, emb_idx


def _build_perch_head(spec: dict, emb_dim: int, n_classes: int):
    import tensorflow as tf
    n_blocks   = int(spec.get("n_blocks",      2))
    hidden_dim = int(spec.get("hidden_dim",    512))
    proj_dim   = int(spec.get("proj_dim",      256))
    drop_block = float(spec.get("dropout_block", 0.3))
    drop_final = float(spec.get("dropout_final", 0.4))
    inp = tf.keras.layers.Input(shape=(emb_dim,))
    x   = tf.keras.layers.BatchNormalization()(inp)
    x   = tf.keras.layers.Dense(hidden_dim)(x)
    x   = tf.keras.layers.LayerNormalization()(x)
    for _ in range(n_blocks):
        h = tf.keras.layers.Dense(hidden_dim)(x)
        h = tf.keras.layers.LayerNormalization()(h)
        h = tf.keras.layers.Activation("gelu")(h)
        h = tf.keras.layers.Dropout(drop_block)(h)
        h = tf.keras.layers.Dense(hidden_dim)(h)
        x = tf.keras.layers.Add()([x, h])
        x = tf.keras.layers.LayerNormalization()(x)
    x   = tf.keras.layers.Dense(proj_dim, activation="gelu")(x)
    x   = tf.keras.layers.Dropout(drop_final)(x)
    out = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def build_common_val(config: dict) -> bool:
    """
    Embed the BirdNET soundscape val set with Perch ONNX as well,
    so both models can be evaluated on the exact same samples.
    Saves: META_VAL_CACHE with keys X_perch, X_birdnet, y_val.
    """
    if META_VAL_CACHE.exists():
        print(f"  [Common val] Already built: {META_VAL_CACHE}")
        return True

    if not BIRDNET_VAL_CACHE.exists():
        print("  [Common val] BirdNET val cache not found — skipping ensemble phase.")
        return False

    print("  [Common val] Embedding soundscapes with Perch ONNX...")
    import librosa

    bn        = np.load(str(BIRDNET_VAL_CACHE), allow_pickle=True)
    X_bn      = bn["X_val"].astype(np.float32)
    y_val     = bn["y_val"].astype(np.float32)
    row_ids   = bn["row_ids"].tolist()
    rid2idx   = {rid: i for i, rid in enumerate(row_ids)}

    sess, inp_name, emb_idx = _load_perch_onnx(config)

    def _hms(t: str) -> int:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    lab = pd.read_csv(SOUNDSCAPE_LABELS)
    grp = (
        lab.groupby(["filename", "start", "end"], sort=False)["primary_label"]
        .agg(lambda s: set().union(*[
            {v.strip() for v in str(x).split(";") if x and str(x) != "nan"} for x in s
        ]))
        .reset_index()
    )

    X_perch_out, X_bn_out, y_out = [], [], []
    for row in grp.itertuples(index=False):
        stem    = Path(row.filename).stem
        end_sec = _hms(row.end)
        row_id  = f"{stem}_{end_sec}"
        if row_id not in rid2idx:
            continue
        fp = TRAIN_SOUNDSCAPES / row.filename
        if not fp.exists():
            continue
        start_sec = _hms(row.start)
        duration  = end_sec - start_sec
        try:
            wav, _ = librosa.load(str(fp), sr=SR, mono=True,
                                  offset=start_sec, duration=float(duration))
        except Exception:
            continue
        n   = int(duration * SR)
        wav = wav[:n] if len(wav) > n else np.pad(wav, (0, n - len(wav)))
        # Central 5-second clip
        if len(wav) >= CLIP_SAMPLES:
            s   = (len(wav) - CLIP_SAMPLES) // 2
            clip = wav[s: s + CLIP_SAMPLES]
        else:
            clip = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        clip = clip.astype(np.float32)

        outs = sess.run(None, {inp_name: clip[np.newaxis, :]})
        X_perch_out.append(outs[emb_idx][0])
        X_bn_out.append(X_bn[rid2idx[row_id]])
        y_out.append(y_val[rid2idx[row_id]])

    if not X_perch_out:
        print("  [Common val] No aligned samples found.")
        return False

    META_LOGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(META_VAL_CACHE),
                        X_perch   = np.stack(X_perch_out).astype(np.float32),
                        X_birdnet = np.stack(X_bn_out).astype(np.float32),
                        y_val     = np.stack(y_out).astype(np.float32))
    print(f"  [Common val] {len(X_perch_out)} aligned samples saved → {META_VAL_CACHE}")
    return True


def run_phase3_ensemble(
    config: dict,
    n_iterations: int,
    suite: SoundscapeEvalSuite,
    perch_score: SoundscapeScore | None,
    birdnet_score: SoundscapeScore | None,
) -> dict:
    if n_iterations <= 0:
        print("\n  [Phase 4] Ensemble skipped (ensemble_iterations=0)")
        return {}

    metric = suite.primary_metric
    print("\n" + "=" * 60)
    print(f"  PHASE 4 — Ensemble  ({n_iterations} blend weights)")
    p_val = perch_score.primary_value if perch_score else 0.0
    b_val = birdnet_score.primary_value if birdnet_score else 0.0
    print(f"  Perch {metric}={p_val:.5f}  BirdNET {metric}={b_val:.5f}")
    print("=" * 60)

    META_LOGS.mkdir(parents=True, exist_ok=True)

    print("  Building aligned soundscape predictions for blend search …")
    y_true = suite.y_true
    perch_preds = suite.aligned_perch_preds(PERCH_MEMORY)
    birdnet_preds = suite.aligned_birdnet_preds(BIRDNET_LOGS, BIRDNET_VAL_CACHE)
    if birdnet_preds is None:
        print("  [Phase 4] BirdNET val preds missing — skipping ensemble.")
        return {}

    perch_self = suite.score_arrays(perch_preds)
    bn_self = suite.score_arrays(birdnet_preds)
    print(f"  Perch alone:   {format_soundscape_score(perch_self)}")
    print(f"  BirdNET alone: {format_soundscape_score(bn_self)}")

    total = perch_self.primary_value + bn_self.primary_value
    w_start = (
        round(perch_self.primary_value / total, 2) if total > 0 else 0.5
    )

    n_blend = max(1, int(n_iterations))
    grid_n = max(n_blend, 9)
    candidates = sorted({round(float(w), 4) for w in np.linspace(0.01, 0.99, grid_n)})
    candidates = sorted(candidates, key=lambda w: abs(w - w_start))[:n_blend]

    results: list[dict] = []
    for i, w in enumerate(candidates, 1):
        blended = w * perch_preds + (1.0 - w) * birdnet_preds
        ap, n_scored = macro_average_precision(y_true, blended)
        auc, _ = competition_macro_auc(y_true, blended)
        med, _ = median_per_class_auc(y_true, blended)
        primary = primary_score(y_true, blended, metric)
        print(
            f"  [{i}/{len(candidates)}] perch_weight={w:.2f}  "
            f"{format_soundscape_metrics_line(macro_ap=ap, macro_auc=auc, median_auc=med, ranking_metric=metric)}"
        )
        results.append({
            "perch_weight": w,
            "primary_metric": metric,
            "primary_value": primary,
            "macro_average_precision": ap,
            "competition_macro_auc": auc,
            "median_per_class_auc": med,
            "n_scored": n_scored,
        })

    best = max(results, key=lambda r: r["primary_value"])
    best_weight = best["perch_weight"]
    best_primary = best["primary_value"]
    print(
        f"\n  [Phase 4] Best blend: perch_weight={best_weight:.2f}  "
        f"{format_soundscape_metrics_line(macro_ap=best['macro_average_precision'], macro_auc=best['competition_macro_auc'], median_auc=best['median_per_class_auc'], ranking_metric=metric)}"
    )

    ensemble_cfg = {
        "primary_metric": metric,
        "perch_weight": best_weight,
        "birdnet_weight": round(1.0 - best_weight, 4),
        "perch_soundscape": perch_self.to_dict(),
        "birdnet_soundscape": bn_self.to_dict(),
        "best_ensemble_primary": best_primary,
        "best_ensemble_macro_ap": best["macro_average_precision"],
        "best_ensemble_macro_auc": best["competition_macro_auc"],
        "best_ensemble_median_auc": best["median_per_class_auc"],
        "all_results": results,
    }
    out = META_LOGS / "ensemble_config.json"
    out.write_text(json.dumps(ensemble_cfg, indent=2), encoding="utf-8")
    print(f"  Ensemble config saved → {out}")
    return ensemble_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Staged pipeline — Step 1a: architecture search × aug baselines
# ─────────────────────────────────────────────────────────────────────────────

def _cnn_stage_cfg(config: dict, key: str) -> dict:
    """CNN staged settings: ``meta_agent.cnn_stage_1b`` etc., else shared ``stage_1b``."""
    meta = _meta_cfg(config)
    cnn_key = f"cnn_{key}"
    if cnn_key in meta:
        return dict(meta[cnn_key])
    return dict(meta.get(key, {}))


def _run_cnn_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    max_samples: int,
    suite: SoundscapeEvalSuite,
) -> dict:
    logs_dir = META_LOGS / "cnn" / baseline
    code_dir = logs_dir / "codes"
    logs_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    cnn_aug = get_cnn_baseline_aug(baseline)
    base_search = config.get("search", {})
    batch = config.get("researcher", {}).get("batch_size", 3)
    override = {
        "meta_aug_preset": baseline,
        "cnn_staged": True,
        "cnn_explore": True,
        "max_iterations": n_iters,
        "lock_augmentation": True,
        "cnn_augmentation": cnn_aug,
        "researcher": {**config.get("researcher", {}), "batch_size": batch},
        "cnn": {
            "logs_dir": str(logs_dir),
            "memory_dir": str(logs_dir),
            "code_dir": str(code_dir),
        },
        "search": {
            **base_search,
            "skip_final_training": True,
            "cheap": {
                **base_search.get("cheap", {}),
                "max_samples": max_samples,
            },
        },
    }
    print(f"\n  [CNN staged 1a / {baseline}] {describe_baseline(baseline)}")
    print(
        f"  logs → {logs_dir}  |  planner_rounds={n_iters}  |  "
        f"3 coder runs/round  |  max_samples={max_samples}"
    )
    rc = _run_subprocess("cnn_agent.py", override, config)
    score = _score_cnn_arch_search(logs_dir, suite)
    _print_soundscape_score(f"CNN/{baseline}", score)
    arch_type = "unknown"
    info_path = logs_dir / "best_model_info.json"
    if info_path.exists():
        try:
            arch_type = json.loads(info_path.read_text(encoding="utf-8")).get("spec", {}).get(
                "arch_type", arch_type
            )
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "track": "cnn",
        "aug_baseline": baseline,
        "logs_dir": str(logs_dir),
        "memory_dir": str(logs_dir),
        "arch_type": arch_type,
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _parse_cnn_memory_dir_slug(mem_dir: Path) -> tuple[str, str]:
    """Infer (aug_baseline, arch_type) from memory dir name (1a or refine layout)."""
    name = mem_dir.name
    if name.startswith("rank") and "_" in name:
        parts = name.split("_", 2)
        if len(parts) >= 3 and parts[0].startswith("rank"):
            return parts[1], parts[2]
        if len(parts) >= 2:
            return parts[1], "unknown"
    if name in BASELINE_AUG_NAMES:
        return name, "unknown"
    return name, "unknown"


def _load_cnn_champion(mem_dir: Path, metric: str) -> dict | None:
    info_path = mem_dir / "best_model_info.json"
    if not (mem_dir / "best_model_slot.py").exists():
        return None
    val = -1.0
    macro_ap = None
    macro_auc = None
    spec: dict = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec = dict(info.get("spec") or {})
            macro_ap = info.get("macro_average_precision")
            macro_auc = info.get("macro_roc_auc")
            # Only trust the stored `ranking_value` when it was produced
            # under the *currently configured* primary metric — otherwise
            # (e.g. an older run was ranked by macro_roc_auc) prefer the
            # explicit metric field that matches `metric` so cross-stage
            # selection stays consistent.
            saved_metric = info.get("ranking_metric")
            preferred = macro_ap if metric == PRIMARY_META_METRIC else macro_auc
            if preferred is not None and (saved_metric and saved_metric != metric):
                val = float(preferred)
            elif info.get("ranking_value") is not None:
                val = float(info["ranking_value"])
            elif preferred is not None:
                val = float(preferred)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    aug, arch_from_name = _parse_cnn_memory_dir_slug(mem_dir)
    aug = normalize_baseline_aug_name(
        spec.get("aug_preset") or spec.get("aug_baseline") or aug,
        default=aug if aug in BASELINE_AUG_NAMES else "medium",
    )
    return {
        "memory_dir": str(mem_dir),
        "logs_dir": str(mem_dir),
        "aug_baseline": aug,
        "arch_type": spec.get("arch_type", arch_from_name),
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": val,
        "macro_average_precision": macro_ap if macro_ap is not None else spec.get("macro_average_precision"),
        "macro_roc_auc": macro_auc,
    }


def collect_cnn_1a_top_candidates(config: dict, top_k: int = 2) -> list[dict]:
    metric = _meta_primary_metric(config)
    candidates: list[dict] = []
    if ARCH_SEARCH_1A_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            runs = (summary.get("tracks") or {}).get("cnn", {}).get("runs") or []
            for run in runs:
                mem = Path(run.get("memory_dir") or run.get("logs_dir", ""))
                cand = _load_cnn_champion(mem, metric)
                if cand:
                    sc = run.get("score") or {}
                    if sc.get("macro_average_precision") is not None:
                        cand["macro_average_precision"] = sc.get("macro_average_precision")
                    # Always rank by the meta soundscape rescore (primary_value)
                    # when present — it is the metric every other stage compares
                    # against. The earlier max(local, rescored) variant could
                    # promote a model whose locally-reported AP was higher than
                    # its actual soundscape AP, contradicting the configured
                    # primary metric (macro_average_precision) at stage 1b.
                    rescored = sc.get("primary_value")
                    if rescored is not None:
                        cand["ranking_value"] = float(rescored)
                    candidates.append(cand)
        except (json.JSONDecodeError, OSError):
            pass
    if not candidates:
        for baseline in _baseline_names(config):
            mem = META_LOGS / "cnn" / baseline
            cand = _load_cnn_champion(mem, metric)
            if cand:
                candidates.append(cand)
    candidates.sort(key=lambda c: float(c.get("ranking_value") or -1), reverse=True)
    return candidates[:top_k]


def _run_cnn_refine_1b(config: dict, candidate: dict, rank: int, refine_cfg: dict) -> dict:
    aug = candidate["aug_baseline"]
    arch = str(candidate.get("arch_type", "shallow_cnn"))
    slug = f"rank{rank}_{aug}_{arch}".replace("/", "_")[:80]
    mem_dir = META_LOGS / "cnn" / "refine" / slug
    code_dir = mem_dir / "codes"
    for d in (mem_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)
    batch = refine_cfg.get("experiments_per_researcher_call")
    if batch is None:
        batch = config.get("researcher", {}).get("batch_size", 3)
    override = {
        "meta_aug_preset": aug,
        "cnn_staged": True,
        "lock_augmentation": True,
        "cnn_augmentation": get_cnn_baseline_aug(aug),
        "cnn_refine": {
            "enabled": True,
            "aug_baseline": aug,
            "locked_arch_type": arch,
            "seed_spec": candidate.get("spec") or {},
            "seed_score": float(candidate.get("ranking_value", -1)),
            "parent_memory_dir": candidate["memory_dir"],
            "experiments_per_researcher_call": int(batch),
            "initial_iterations": int(refine_cfg.get("initial_iterations", 6)),
            "bonus_iterations_on_improve": int(refine_cfg.get("bonus_iterations_on_improve", 6)),
            "max_iterations_per_model": int(refine_cfg.get("max_iterations_per_model", 30)),
        },
        "cnn": {
            "logs_dir": str(mem_dir.parent),
            "memory_dir": str(mem_dir),
            "code_dir": str(code_dir),
        },
    }
    print(f"\n  [CNN refine 1b / rank {rank}] aug={aug} arch={arch}")
    rc = _run_subprocess("cnn_agent.py", override, config)
    return {
        "rank": rank,
        "aug_baseline": aug,
        "arch_type": arch,
        "memory_dir": str(mem_dir),
        "subprocess_rc": rc,
    }


def run_stage_1b_cnn_refine(config: dict, suite: SoundscapeEvalSuite) -> dict:
    refine_cfg = _cnn_stage_cfg(config, "stage_1b")
    if not refine_cfg.get("enabled", False):
        print("\n  [CNN Stage 1b] Skipped (cnn_stage_1b.enabled=false)")
        return {}
    if _skip_if_completed(config, CNN_ARCH_1B_RESULTS, "CNN Stage 1b"):
        try:
            return json.loads(CNN_ARCH_1B_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    top_k = max(1, int(refine_cfg.get("top_k_models", 2)))
    metric = _meta_primary_metric(config)
    print("\n" + "=" * 60)
    print(f"  CNN STAGED — Step 1b: Refine top-{top_k} from 1a")
    print("=" * 60)
    candidates = collect_cnn_1a_top_candidates(config, top_k=top_k)
    if not candidates:
        print("  [CNN 1b] No 1a candidates — run stage 1a with cnn_iterations > 0.")
        return {"stage": "1b_cnn_refine", "candidates": [], "refine_runs": []}
    refine_runs = [
        _run_cnn_refine_1b(config, c, rank=i, refine_cfg=refine_cfg)
        for i, c in enumerate(candidates, 1)
    ]
    scored = []
    for run in refine_runs:
        mem = Path(run["memory_dir"])
        sc = _score_cnn_arch_search(mem, suite)
        entry = {**run, "score": sc.to_dict() if sc else None}
        if sc:
            entry["refined_ranking_value"] = sc.primary_value
        scored.append(entry)
    winner = _pick_track_winner(scored, metric)
    summary = {
        "stage": "1b_cnn_refine",
        "primary_metric": metric,
        "refine_runs": scored,
        "winner": winner,
    }
    CNN_ARCH_1B_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _pick_cnn_1b_winner(config: dict) -> dict | None:
    metric = _meta_primary_metric(config)
    if CNN_ARCH_1B_RESULTS.exists():
        try:
            w = json.loads(CNN_ARCH_1B_RESULTS.read_text(encoding="utf-8")).get("winner")
            if w and Path(w.get("memory_dir", "")).exists():
                mem = Path(w["memory_dir"])
                fresh = _load_cnn_champion(mem, metric)
                # `w` was selected in stage 1b by the rescored soundscape
                # primary_value (see `_pick_track_winner`). Prefer that
                # explicit selection score over the child-local ranking_value
                # so the displayed and downstream ranking stays consistent
                # with how the winner was actually chosen.
                sc = (w.get("score") or {}) if isinstance(w, dict) else {}
                rescored = sc.get("primary_value")
                if rescored is None:
                    rescored = w.get("refined_ranking_value") if isinstance(w, dict) else None
                if fresh:
                    if rescored is not None:
                        fresh["ranking_value"] = float(rescored)
                        fresh["soundscape_primary_value"] = float(rescored)
                    if sc.get("macro_average_precision") is not None:
                        fresh["macro_average_precision"] = sc.get("macro_average_precision")
                    return fresh
                w["aug_baseline"] = normalize_baseline_aug_name(
                    w.get("aug_baseline", ""), default="medium"
                )
                if rescored is not None:
                    w["ranking_value"] = float(rescored)
                return w
        except (json.JSONDecodeError, OSError):
            pass
    refine_root = META_LOGS / "cnn" / "refine"
    best_run, best_val = None, -1.0
    if refine_root.exists():
        for mem_dir in refine_root.iterdir():
            if not mem_dir.is_dir():
                continue
            cand = _load_cnn_champion(mem_dir, metric)
            if cand and float(cand.get("ranking_value", -1)) > best_val:
                best_val = float(cand["ranking_value"])
                best_run = cand
    if best_run:
        return best_run
    cands = collect_cnn_1a_top_candidates(config, top_k=1)
    return cands[0] if cands else None


def _append_cnn_1c_champion_reference(
    runs: list[dict],
    *,
    winner: dict,
    winner_mem: Path,
    suite: SoundscapeEvalSuite,
    cfg: dict,
) -> None:
    """Score 1b champion on soundscapes (no retrain); keeps e.g. high@1b unless 1c beats it."""
    if not cfg.get("include_champion_reference", True):
        return
    if not cfg.get("include_winner_aug_baseline", True):
        return
    w_aug = normalize_baseline_aug_name(winner.get("aug_baseline", ""), default="high")
    sc = _score_cnn_arch_search(winner_mem, suite)
    if sc is None:
        print(f"  [CNN/1c/champion] Skipped — no soundscape artifacts in {winner_mem}")
        return
    try:
        aug_dict = get_cnn_baseline_aug(w_aug)
    except KeyError:
        aug_dict = get_cnn_baseline_aug("high")
    tid = "champion_1b"
    aug_path = META_LOGS / "cnn" / "aug_search" / tid / f"{tid}_aug.json"
    aug_path.parent.mkdir(parents=True, exist_ok=True)
    aug_path.write_text(json.dumps(aug_dict, indent=2), encoding="utf-8")
    runs.append(
        {
            "trial_id": tid,
            "aug_preset": w_aug,
            "memory_dir": str(winner_mem),
            "aug_config_path": str(aug_path),
            "subprocess_rc": 0,
            "failed": False,
            "champion_reference": True,
            "score": sc.to_dict(),
        }
    )
    _print_soundscape_score(f"CNN/1c/champion_1b ({w_aug} from 1b)", sc)


def _run_cnn_1c_llm_search(
    config: dict,
    suite: SoundscapeEvalSuite,
    cfg: dict,
    *,
    winner_mem: Path,
    locked_slot: Path,
    max_samples: int,
    metric: str,
) -> list[dict]:
    """LLM planner for CNN 1c: audio/SNR + spectrogram knobs on locked architecture."""
    from cnn_aug_researcher import CnnAugResearcher, cnn_aug_trial_id, spec_to_cnn_training_aug
    from cnn_soundscape_cache import DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR
    from llm_client import LLMClient
    from memory import ExperimentMemory

    runs: list[dict] = []
    aug_mem = META_LOGS / "cnn" / "aug_search" / "_llm_memory"
    aug_mem.mkdir(parents=True, exist_ok=True)
    cache_dir = DEFAULT_SOUNDSCAPE_MEL_CACHE_DIR

    provider, model = _resolve_stage_1c_researcher_llm(config, cfg)
    llm_timeout = float(cfg.get("researcher_timeout_seconds", 600))
    llm = LLMClient(
        provider=provider,
        model=model,
        timeout_seconds=llm_timeout,
        stream_debug=bool(cfg.get("stream_debug", False)),
    )
    memory = ExperimentMemory(aug_mem, ranking_metric=metric)
    researcher = CnnAugResearcher(
        llm,
        memory,
        temperature=float(cfg.get("temperature", config.get("researcher", {}).get("temperature", 0.35))),
        batch_size=int(cfg.get("experiments_per_round", 3)),
        format_json=bool(cfg.get("researcher_format_json", True)),
        num_predict=int(cfg["researcher_num_predict"])
        if cfg.get("researcher_num_predict") is not None
        else 4096,
    )
    n_rounds = int(cfg.get("planner_rounds", 3))
    print(f"\n  [CNN 1c LLM] {n_rounds} round(s) × {researcher.batch_size} configs / round")

    for round_i in range(1, n_rounds + 1):
        print(f"\n  ── CNN 1c LLM round {round_i}/{n_rounds} ──")
        specs = researcher.next_experiments(round_i=round_i)
        for slot_i, spec in enumerate(specs, 1):
            tid = cnn_aug_trial_id(spec, round_i, slot_i)
            try:
                aug_dict, cache_preset = spec_to_cnn_training_aug(spec, cache_dir=cache_dir)
            except (ValueError, TypeError, KeyError) as exc:
                # Log the bad spec so the researcher's next prompt actually
                # contains a "RECENT FAILURES" entry pointing at it.
                print(f"  [CNN 1c LLM] Invalid spec {tid}: {exc}")
                memory.log(
                    spec=spec,
                    metrics={"status": "failed", "reason": f"spec_invalid: {exc}"},
                )
                continue
            aug_path = META_LOGS / "cnn" / "aug_search" / tid / f"{tid}_aug.json"
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            aug_path.write_text(
                json.dumps({**aug_dict, "llm_spec": spec}, indent=2),
                encoding="utf-8",
            )
            run = _run_cnn_1c_trial_meta(
                config,
                winner_mem=winner_mem,
                locked_slot=locked_slot,
                aug_preset=cache_preset,
                aug_dict=aug_dict,
                trial_id=tid,
                max_samples=max_samples,
            )
            run["aug_config_path"] = str(aug_path)
            run["aug_spec"] = spec
            runs.append(run)
            label = str(spec.get("preset_name", tid))
            if run.get("failed"):
                print(f"  [CNN/1c/{label}] trial failed (subprocess_rc={run.get('subprocess_rc')})")
                # Surface subprocess failures to the researcher too — without
                # this, every failed run was invisible to the LLM and it kept
                # proposing nearby configs.
                memory.log(
                    spec=spec,
                    metrics={
                        "status": "failed",
                        "reason": f"subprocess_rc={run.get('subprocess_rc')}",
                    },
                )
            else:
                sc = _score_cnn_arch_search(Path(run["memory_dir"]), suite)
                if sc:
                    run["score"] = sc.to_dict()
                _print_soundscape_score(f"CNN/1c/{label}", sc)
                if sc is None:
                    # subprocess succeeded but no predictions were saved —
                    # treat as a failure for memory purposes so we don't
                    # promote a configuration with no score.
                    memory.log(
                        spec=spec,
                        metrics={
                            "status": "failed",
                            "reason": "no_soundscape_score_artifacts",
                        },
                    )
                else:
                    memory.log(
                        spec=spec,
                        metrics={
                            "status": "success",
                            "macro_average_precision": sc.macro_average_precision,
                            "macro_roc_auc": sc.competition_macro_auc,
                            "median_per_class_auc": sc.median_per_class_auc,
                            "soundscape_macro_ap": sc.primary_value,
                        },
                    )
    return runs


def _run_cnn_1c_trial_meta(
    config: dict,
    *,
    winner_mem: Path,
    locked_slot: Path,
    aug_preset: str,
    aug_dict: dict,
    trial_id: str,
    max_samples: int,
) -> dict:
    cache_preset = str(aug_dict.get("aug_preset") or aug_preset)
    if cache_preset in BASELINE_AUG_NAMES:
        cache_preset = normalize_baseline_aug_name(cache_preset)
    trial_dir = META_LOGS / "cnn" / "aug_search" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    override = {
        "cnn_staged": True,
        "cnn_1c_trial": {
            "locked_slot_path": str(locked_slot),
            "aug_dict": aug_dict,
            "aug_preset": cache_preset,
            "trial_id": trial_id,
            "max_samples": max_samples,
        },
        "cnn": {
            "logs_dir": str(trial_dir),
            "memory_dir": str(trial_dir),
            "code_dir": str(trial_dir / "codes"),
        },
    }
    rc = _run_subprocess("cnn_agent.py", override, config)
    success = rc == 0
    sc = _score_cnn_arch_search(trial_dir, _soundscape_suite(config)) if success else None
    return {
        "trial_id": trial_id,
        "aug_preset": cache_preset,
        "memory_dir": str(trial_dir),
        "aug_config_path": str(trial_dir / f"{trial_id}_aug.json"),
        "subprocess_rc": rc,
        "failed": not success,
        "score": sc.to_dict() if sc else None,
    }


def run_stage_1c_cnn_aug_search(config: dict, suite: SoundscapeEvalSuite) -> dict:
    cfg = _cnn_stage_cfg(config, "stage_1c")
    if not cfg.get("enabled", False):
        print("\n  [CNN Stage 1c] Skipped (cnn_stage_1c.enabled=false)")
        return {}
    if _skip_if_completed(config, CNN_ARCH_1C_RESULTS, "CNN Stage 1c"):
        try:
            return json.loads(CNN_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    winner = _pick_cnn_1b_winner(config)
    if not winner:
        print("\n  [CNN 1c] No 1b/1a winner.")
        return {"stage": "1c_cnn_aug_search", "runs": [], "winner": None}
    winner_mem = Path(winner["memory_dir"])
    locked_slot = winner_mem / "best_model_slot.py"
    if not locked_slot.exists():
        print(f"\n  [CNN 1c] Missing {locked_slot}")
        return {"stage": "1c_cnn_aug_search", "runs": [], "winner": None}
    metric = _meta_primary_metric(config)
    max_samples = int(cfg.get("head_train_samples", 2000))
    baselines = _baseline_names(config)
    mode = str(cfg.get("mode", "presets")).lower()
    preset_list: list[str] = []
    if mode in ("presets", "both") and cfg.get("include_preset_baselines", True):
        seen_presets: set[str] = set()
        for raw in cfg.get("aug_presets") or ["medium", "high", "light"]:
            p = normalize_baseline_aug_name(str(raw))
            if p and p not in seen_presets:
                seen_presets.add(p)
                preset_list.append(p)
        if cfg.get("include_winner_aug_baseline", True):
            w_aug = normalize_baseline_aug_name(winner.get("aug_baseline", ""), default="")
            if w_aug and w_aug not in preset_list:
                preset_list.insert(0, w_aug)
    print("\n" + "=" * 60)
    print("  CNN STAGED — Step 1c: Augmentation search (locked architecture)")
    print(f"  mode={mode}  presets={preset_list if mode in ('presets', 'both') else '—'}")
    print("=" * 60)
    runs: list[dict] = []
    _append_cnn_1c_champion_reference(
        runs, winner=winner, winner_mem=winner_mem, suite=suite, cfg=cfg
    )
    if mode in ("presets", "both") and preset_list:
        for preset in preset_list:
            try:
                aug_dict = get_cnn_baseline_aug(preset)
            except KeyError:
                aug_dict = get_cnn_baseline_aug("medium")
            trial_id = f"preset_{preset}"
            aug_path = META_LOGS / "cnn" / "aug_search" / trial_id / f"{trial_id}_aug.json"
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            aug_path.write_text(json.dumps(aug_dict, indent=2), encoding="utf-8")
            run = _run_cnn_1c_trial_meta(
                config,
                winner_mem=winner_mem,
                locked_slot=locked_slot,
                aug_preset=preset,
                aug_dict=aug_dict,
                trial_id=trial_id,
                max_samples=max_samples,
            )
            run["aug_config_path"] = str(aug_path)
            runs.append(run)
            if run.get("failed"):
                print(f"  [CNN/1c/{preset}] trial failed (subprocess_rc={run.get('subprocess_rc')})")
            else:
                sc = _score_cnn_arch_search(Path(run["memory_dir"]), suite)
                if sc:
                    run["score"] = sc.to_dict()
                _print_soundscape_score(f"CNN/1c/{preset}", sc)

    if mode in ("llm", "both"):
        runs.extend(
            _run_cnn_1c_llm_search(
                config,
                suite,
                cfg,
                winner_mem=winner_mem,
                locked_slot=locked_slot,
                max_samples=max_samples,
                metric=metric,
            )
        )

    aug_winner = _pick_track_winner(runs, metric)
    summary = {
        "stage": "1c_cnn_aug_search",
        "primary_metric": metric,
        "locked_slot_code": str(locked_slot),
        "refine_winner_1b": winner,
        "runs": runs,
        "winner": aug_winner,
    }
    CNN_ARCH_1C_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if aug_winner:
        print(f"\n  ★ CNN aug winner: {aug_winner.get('aug_preset')}")
    return summary


def run_stage_1d_cnn_final_train(config: dict) -> dict:
    cfg = _cnn_stage_cfg(config, "stage_1d")
    if not cfg.get("enabled", False):
        print("\n  [CNN Stage 1d] Skipped (cnn_stage_1d.enabled=false)")
        return {}
    if not CNN_ARCH_1C_RESULTS.exists():
        print("\n  [CNN 1d] Run CNN stage 1c first.")
        return {}
    s1c = json.loads(CNN_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
    aug_winner = s1c.get("winner")
    refine_winner = s1c.get("refine_winner_1b") or _pick_cnn_1b_winner(config)
    if not aug_winner or not refine_winner:
        return {"stage": "1d_cnn_final_train", "success": False}
    locked_slot = Path(s1c.get("locked_slot_code", ""))
    if not locked_slot.exists():
        locked_slot = Path(refine_winner["memory_dir"]) / "best_model_slot.py"
    aug_preset = aug_winner.get("aug_preset", "medium")
    aug_path = aug_winner.get("aug_config_path")
    if aug_path and Path(aug_path).exists():
        aug_dict = json.loads(Path(aug_path).read_text(encoding="utf-8"))
    else:
        aug_dict = get_cnn_baseline_aug(str(aug_preset))
    CNN_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CNN_FINAL_DIR / "model.keras"
    sc_final = (config.get("search") or {}).get("final") or {}
    final_overrides = {
        "locked_slot_path": str(locked_slot),
        "aug_dict": aug_dict,
        "model_save_path": str(model_path),
        "epochs": cfg.get("epochs"),
        "max_samples": cfg.get("max_samples", sc_final.get("max_samples")),
        "val_split": cfg.get("val_split", sc_final.get("val_split", 0.0)),
        "rebuild_focal_cache": cfg.get("rebuild_focal_cache", False),
        "final_timeout_seconds": cfg.get("final_timeout_seconds"),
    }
    # Drop any explicitly-None override so the downstream defaults take effect.
    final_overrides = {k: v for k, v in final_overrides.items() if v is not None or k in {"max_samples"}}
    override = {
        "cnn_staged": True,
        "cnn_final_train": final_overrides,
        "cnn": {
            "logs_dir": str(CNN_FINAL_DIR),
            "memory_dir": str(CNN_FINAL_DIR),
            "code_dir": str(CNN_FINAL_DIR / "codes"),
        },
    }
    print("\n" + "=" * 60)
    print(f"  CNN STAGED — Step 1d: Final train → {model_path}")
    print("=" * 60)
    rc = _run_subprocess("cnn_agent.py", override, config)
    ok = rc == 0 and model_path.exists()
    if ok:
        shutil.copy2(locked_slot, CNN_FINAL_DIR / "best_model_slot.py")
    if ok and cfg.get("copy_to_submission", True):
        sub = PROJECT_ROOT / "submission"
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, sub / "model.keras")
        shutil.copy2(locked_slot, sub / "cnn_best_slot.py")
    summary = {"stage": "1d_cnn_final_train", "success": ok, "model_path": str(model_path), "aug_preset": aug_preset}
    CNN_ARCH_1D_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_stage_1e_cnn_pseudo_refine(config: dict, suite: SoundscapeEvalSuite) -> dict:
    cfg = _cnn_stage_cfg(config, "stage_1e")
    if not cfg.get("enabled", False):
        print("\n  [CNN Stage 1e] Skipped (cnn_stage_1e.enabled=false)")
        return {}
    if _skip_if_completed(config, CNN_ARCH_1E_RESULTS, "CNN Stage 1e"):
        try:
            return json.loads(CNN_ARCH_1E_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    teacher = CNN_FINAL_DIR / "model.keras"
    if not teacher.exists():
        print("\n  [CNN 1e] Run CNN stage 1d first.")
        return {"stage": "1e_cnn_pseudo_refine", "success": False}
    locked_slot = CNN_FINAL_DIR / "best_model_slot.py"
    if not locked_slot.exists() and CNN_ARCH_1C_RESULTS.exists():
        s1c = json.loads(CNN_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
        locked_slot = Path(s1c.get("locked_slot_code", locked_slot))
    pseudo_model = CNN_FINAL_DIR / "model_pseudo.keras"
    aug_dict: dict = {}
    if CNN_ARCH_1C_RESULTS.exists():
        try:
            s1c = json.loads(CNN_ARCH_1C_RESULTS.read_text(encoding="utf-8"))
            aug_winner = s1c.get("winner") or {}
            aug_path = aug_winner.get("aug_config_path")
            if aug_path and Path(aug_path).exists():
                aug_dict = json.loads(Path(aug_path).read_text(encoding="utf-8"))
            else:
                aug_dict = get_cnn_baseline_aug(str(aug_winner.get("aug_preset", "high")))
        except (json.JSONDecodeError, OSError, KeyError):
            aug_dict = get_cnn_baseline_aug("high")
    else:
        aug_dict = get_cnn_baseline_aug("high")
    override = {
        "cnn_staged": True,
        "cnn_pseudo_refine": {
            "teacher_model": str(teacher),
            "locked_slot_path": str(locked_slot),
            "pseudo_npz": str(CNN_PSEUDO_LABELS_NPZ),
            "model_save_path": str(pseudo_model),
            "aug_dict": aug_dict,
            "rebuild_pseudo_cache": cfg.get("rebuild_pseudo_cache", False),
            "top1_threshold": cfg.get("top1_threshold", 0.55),
            "runnerup_max": cfg.get("runnerup_max", 0.35),
            "pseudo_label_weight": cfg.get("pseudo_label_weight", 0.8),
            "sample_weight_supervised": cfg.get("sample_weight_supervised", 1.0),
            "sample_weight_pseudo": cfg.get("sample_weight_pseudo", 0.5),
            "fine_tune_epochs": cfg.get("fine_tune_epochs", 15),
            "fine_tune_lr": cfg.get("fine_tune_lr", 2e-4),
            "refine_timeout_seconds": cfg.get("refine_timeout_seconds", 7200),
            "max_soundscape_files": cfg.get("max_soundscape_files"),
            "max_pseudo_windows": cfg.get("max_pseudo_windows"),
            "heartbeat_every": cfg.get("heartbeat_every", 10),
            # null/missing → use all supervised focal mels (real run);
            # an int caps the supervised side of the fine-tune for fast tests.
            "max_supervised_samples": cfg.get("max_supervised_samples"),
            "skip_fine_tune_without_pseudo": cfg.get("skip_fine_tune_without_pseudo", True),
        },
        "cnn": {
            "logs_dir": str(CNN_FINAL_DIR),
            "memory_dir": str(CNN_FINAL_DIR),
            "code_dir": str(CNN_FINAL_DIR / "codes"),
        },
    }
    print("\n" + "=" * 60)
    print("  CNN STAGED — Step 1e: Pseudo-label refine")
    print("=" * 60)
    rc = _run_subprocess("cnn_agent.py", override, config)
    ok = rc == 0 and pseudo_model.exists()
    sub = PROJECT_ROOT / "submission"
    if ok and cfg.get("copy_to_submission", True):
        sub.mkdir(parents=True, exist_ok=True)
        # Kaggle notebook loads submission/model.keras; keep explicit pseudo copy too.
        shutil.copy2(pseudo_model, sub / "model.keras")
        shutil.copy2(pseudo_model, sub / "model_pseudo.keras")
        if locked_slot.exists():
            shutil.copy2(locked_slot, sub / "cnn_best_slot.py")
        try:
            from perch_agent import configure_tensorflow_cpu_only

            configure_tensorflow_cpu_only()
            import tensorflow as tf
            _m = tf.keras.models.load_model(str(pseudo_model), compile=False)
            _w = sub / "model_pseudo.weights.h5"
            _m.save_weights(str(_w))
            shutil.copy2(_w, sub / "model.weights.h5")
        except Exception as exc:
            print(f"  [CNN 1e] Optional weights.h5 export skipped: {exc}")
        print(f"  [CNN 1e] Submission bundle updated → {sub}")
    summary = {
        "stage": "1e_cnn_pseudo_refine",
        "success": ok,
        "model_path": str(pseudo_model),
        "teacher_path": str(teacher),
        "submission_model": str(sub / "model.keras") if ok else None,
        "submission_slot": str(sub / "cnn_best_slot.py") if ok and locked_slot.exists() else None,
        "locked_slot_path": str(locked_slot),
    }
    CNN_ARCH_1E_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_birdnet_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    embed_frac: float,
    suite: SoundscapeEvalSuite,
) -> dict:
    mem_dir = META_LOGS / "birdnet" / baseline
    cache_dir = META_LOGS / "birdnet_cache" / baseline
    code_dir = mem_dir / "codes"
    for d in (mem_dir, cache_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)
    head_train_cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
    embed_max = _embed_max_samples(config)
    override = {
        "meta_aug_preset": baseline,
        "augmentation": get_audio_embedding_aug(baseline),
        "birdnet_staged": True,
        "train_sample_frac": embed_frac,
        "max_iterations": n_iters,
        "head_train_max_samples": head_train_cap,
        "force_rebuild_cache": _meta_cfg(config).get("force_rebuild_embed_cache", False),
        "birdnet": {
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "val_cache_path": str(BIRDNET_SHARED_VAL_CACHE),
        },
    }
    if embed_max is not None:
        override["max_train_samples"] = embed_max
    cap_note = f"  |  embed_cap={embed_max}" if embed_max is not None else ""
    print(f"\n  [BirdNET staged 1a / {baseline}] {describe_baseline(baseline)}")
    print(
        f"  memory → {mem_dir}  |  planner rounds={n_iters}  |  embed_frac={embed_frac}{cap_note}"
    )
    rc = _run_subprocess("birdnet_agent.py", override, config)
    score = suite.score_birdnet_mem_dir(mem_dir, val_cache=BIRDNET_SHARED_VAL_CACHE)
    if score is None:
        score = suite.score_birdnet_artifacts(mem_dir, BIRDNET_VAL_CACHE)
    _print_soundscape_score(f"BirdNET/{baseline}", score)
    arch_type = "unknown"
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            arch_type = json.loads(info_path.read_text(encoding="utf-8")).get("spec", {}).get(
                "arch_type", arch_type
            )
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "track": "birdnet",
        "aug_baseline": baseline,
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "arch_type": arch_type,
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _run_perch_baseline_1a(
    config: dict,
    baseline: str,
    n_iters: int,
    embed_frac: float,
    embed_max: int | None,
    suite: SoundscapeEvalSuite,
) -> dict:
    mem_dir = META_LOGS / "perch" / baseline
    cache_dir = META_LOGS / "perch_cache" / baseline
    code_dir = mem_dir / "codes"
    for d in (mem_dir, cache_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)
    perch_base = dict(config.get("perch", {}))
    train_cache = cache_dir / f"train_emb_{baseline}.npz"
    head_train_cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
    if head_train_cap is None:
        head_train_cap = perch_base.get("head_train_max_samples")
    override = {
        "meta_aug_preset": baseline,
        "augmentation": get_audio_embedding_aug(baseline),
        "train_sample_frac": embed_frac,
        "max_iterations": n_iters,
        "head_train_max_samples": head_train_cap,
        "perch": {
            **perch_base,
            "logs_dir": str(mem_dir.parent),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "max_train_samples": embed_max,
            "head_train_max_samples": head_train_cap,
            "force_rebuild_cache": _meta_cfg(config).get("force_rebuild_embed_cache", False),
            "skip_final_retrain": True,
        },
    }
    print(f"\n  [Perch / {baseline}] {describe_baseline(baseline)}")
    cap_msg = f"  |  head_train_cap={head_train_cap}" if head_train_cap else ""
    print(
        f"  memory → {mem_dir}  |  cache → {cache_dir}  |  "
        f"embed_frac={embed_frac} (stratified per species){cap_msg}"
    )
    if train_cache.exists() and not override["perch"]["force_rebuild_cache"]:
        print(f"  train embeddings on disk → {train_cache.name} (will skip rebuild)")
    rc = _run_subprocess("perch_agent.py", override, config)
    score = suite.score_perch(mem_dir)
    _print_soundscape_score(f"Perch/{baseline}", score)
    return {
        "track": "perch",
        "aug_baseline": baseline,
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "subprocess_rc": rc,
        "score": score.to_dict() if score else None,
    }


def _pick_track_winner(runs: list[dict], metric: str) -> dict | None:
    """Best completed run by soundscape score (primary_value), not fixed-train subset AP."""
    scored = [
        r for r in runs
        if not r.get("failed")
        and r.get("score")
        and r["score"].get("primary_value") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r["score"]["primary_value"]))


def _load_aug_dict_for_1c_winner(winner: dict, baselines: list[str] | None = None) -> dict:
    """Load the exact augmentation JSON used for a 1c trial (custom presets included)."""
    from aug_researcher import aug_dict_from_logged_spec

    path = winner.get("aug_config_path")
    if path:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    preset = str(winner.get("aug_preset") or "")
    if preset:
        for candidate in (
            META_LOGS / "perch_cache" / "aug_search" / preset / f"{preset}_aug_config.json",
            META_LOGS / "perch_cache" / preset / f"{preset}_aug_config.json",
        ):
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))

    if winner.get("aug_spec"):
        try:
            return aug_dict_from_logged_spec(winner["aug_spec"])
        except ValueError:
            pass

    if preset in (baselines or []):
        return get_audio_embedding_aug(preset)
    try:
        return get_aug_search_preset(preset)
    except KeyError:
        return get_audio_embedding_aug("medium")


def _enrich_1c_winner(winner: dict, baselines: list[str]) -> dict:
    preset = str(winner.get("aug_preset", ""))
    cache_dir = _perch_aug_search_cache_dir(preset, baselines)
    cfg_path = cache_dir / f"{preset}_aug_config.json"
    winner["cache_dir"] = str(cache_dir)
    winner["aug_config_path"] = str(cfg_path)
    winner["train_cache"] = str(cache_dir / f"train_emb_{preset}.npz")
    if cfg_path.exists() and not winner.get("aug_spec"):
        try:
            from aug_researcher import aug_dict_from_logged_spec

            winner["aug_spec"] = aug_dict_from_logged_spec(
                json.loads(cfg_path.read_text(encoding="utf-8"))
            )
        except (ValueError, json.JSONDecodeError, OSError):
            pass
    return winner


def _finalize_1c_winner(
    runs: list[dict],
    *,
    metric: str,
    baselines: list[str],
    announce: bool = False,
) -> dict | None:
    """Pick best completed 1c run by soundscape AP; enrich paths for stage 1d."""
    winner = _pick_track_winner(runs, metric)
    if winner is None:
        return None
    winner = _enrich_1c_winner(winner, baselines)
    if announce:
        sc = winner.get("score") or {}
        aug_for_embed = _load_aug_dict_for_1c_winner(winner, baselines)
        label = _short_aug_display_name(
            str(winner.get("aug_preset", "")), winner.get("aug_spec")
        )
        print(
            f"\n  ★ 1c winner → {label} ({winner.get('aug_preset')})  "
            f"soundscape_AP={float(sc.get('primary_value', 0)):.5f}  "
            f"mix_prob={aug_for_embed.get('mix_prob')}  "
            f"use_snr={aug_for_embed.get('use_snr_mixing')}"
        )
    return winner


def _ranking_value_from_run(entry: dict, metric: str) -> float:
    if metric == "macro_roc_auc":
        v = entry.get("macro_roc_auc")
    else:
        v = entry.get("macro_average_precision")
        if v is None:
            v = (entry.get("metrics") or {}).get("macro_average_precision")
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _load_perch_baseline_champion(aug_baseline: str, mem_dir: Path, metric: str) -> dict | None:
    """Best successful run from a stage-1a perch memory dir."""
    jsonl = mem_dir / "experiment_memory.jsonl"
    best_entry: dict | None = None
    best_val = -1.0
    best_typed_entry: dict | None = None
    best_typed_val = -1.0
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
                val = _ranking_value_from_run(entry, metric)
                spec = entry.get("spec") or {}
                if spec.get("arch_type"):
                    if val > best_typed_val:
                        best_typed_val = val
                        best_typed_entry = entry
                if val > best_val:
                    best_val = val
                    best_entry = entry
    # Prefer best run that has arch_type (required for stage-1b refine locking).
    if best_typed_entry is not None:
        best_entry = best_typed_entry
        best_val = best_typed_val
    elif best_entry is not None:
        spec0 = best_entry.get("spec") or {}
        if not spec0.get("arch_type"):
            best_entry = None
            best_val = -1.0

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec_info = info.get("spec") or {}
            if spec_info.get("arch_type"):
                val = float(info.get("ranking_value", info.get("macro_average_precision", -1)))
                if val > best_val:
                    best_val = val
                    best_entry = {
                        "success": True,
                        "spec": spec_info,
                        "macro_average_precision": info.get("macro_average_precision"),
                        "macro_roc_auc": info.get("macro_roc_auc"),
                        "metrics": info,
                    }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if best_entry is None:
        return None

    spec = dict(best_entry.get("spec") or {})
    if not spec.get("arch_type"):
        digest_path = mem_dir / "memory_digest.json"
        if digest_path.exists():
            try:
                digest = json.loads(digest_path.read_text(encoding="utf-8"))
                snap = digest.get("best_snapshot") or {}
                if snap.get("arch_type"):
                    spec["arch_type"] = snap["arch_type"]
                    spec.update(snap.get("spec_compact") or {})
            except (json.JSONDecodeError, OSError):
                pass
    spec.setdefault("arch_type", "residual_mlp")

    cache_dir = META_LOGS / "perch_cache" / aug_baseline
    return {
        "aug_baseline": aug_baseline,
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "arch_type": spec.get("arch_type"),
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": best_val,
        "macro_average_precision": best_entry.get("macro_average_precision"),
        "macro_roc_auc": best_entry.get("macro_roc_auc"),
        "median_per_class_auc": best_entry.get("median_per_class_auc"),
    }


def collect_perch_1a_top_candidates(config: dict, top_k: int = 2) -> list[dict]:
    """
    Pick top-K perch models from stage-1a (one champion per aug baseline, then global top-K).
    """
    metric = _meta_primary_metric(config)
    baselines = _baseline_names(config)

    if ARCH_SEARCH_1A_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
            perch_runs = (summary.get("tracks") or {}).get("perch", {}).get("runs") or []
            if perch_runs:
                baselines = [r["aug_baseline"] for r in perch_runs if r.get("aug_baseline")]
        except (json.JSONDecodeError, OSError):
            pass

    candidates: list[dict] = []
    for baseline in baselines:
        mem_dir = META_LOGS / "perch" / baseline
        champ = _load_perch_baseline_champion(baseline, mem_dir, metric)
        if champ:
            candidates.append(champ)

    candidates.sort(key=lambda c: float(c["ranking_value"]), reverse=True)

    # Prefer distinct (aug_baseline, arch_type); then fill to top_k
    picked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for c in candidates:
        key = (c["aug_baseline"], str(c.get("arch_type", "?")))
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= top_k:
            break

    if len(picked) < top_k:
        for c in candidates:
            if c in picked:
                continue
            picked.append(c)
            if len(picked) >= top_k:
                break

    return picked[:top_k]


def _enrich_perch_winner_metrics(
    winner: dict,
    metric: str,
    suite: SoundscapeEvalSuite | None = None,
) -> dict:
    """Fill macro AP / AUC / median on winner dict from disk or one soundscape eval."""
    mem_dir = Path(winner["memory_dir"])
    ap = winner.get("macro_average_precision")
    auc = winner.get("macro_roc_auc")
    med = winner.get("median_per_class_auc")

    info_path = mem_dir / "best_model_info.json"
    if info_path.exists() and (ap is None or auc is None):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            ap = ap if ap is not None else info.get("macro_average_precision")
            auc = auc if auc is not None else info.get("macro_roc_auc")
            med = med if med is not None else info.get("median_per_class_auc")
            if winner.get("ranking_value") is None and info.get("ranking_value") is not None:
                winner["ranking_value"] = float(info["ranking_value"])
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    score = winner.get("score") or {}
    if ap is None and score.get("macro_average_precision") is not None:
        ap = score["macro_average_precision"]
    if auc is None and score.get("macro_roc_auc") is not None:
        auc = score["macro_roc_auc"]
    if med is None and score.get("median_per_class_auc") is not None:
        med = score["median_per_class_auc"]

    if suite is not None and _perch_memory_has_head(mem_dir) and ap is None:
        try:
            sc = suite.score_perch(mem_dir)
            if sc is not None:
                ap = sc.macro_average_precision
                auc = sc.competition_macro_auc
                med = sc.median_per_class_auc
                winner["score"] = sc.to_dict()
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  [1c] Soundscape re-score skipped: {exc}")

    if ap is not None:
        winner["macro_average_precision"] = float(ap)
    if auc is not None:
        winner["macro_roc_auc"] = float(auc)
    if med is not None:
        winner["median_per_class_auc"] = float(med)
    if winner.get("ranking_value") is None and ap is not None:
        winner["ranking_value"] = float(ap) if metric != "macro_roc_auc" else float(auc or ap)
    winner.setdefault("ranking_metric", metric)
    return winner


def _print_stage_1c_locked_head(winner: dict, ranking_metric: str) -> None:
    """Show soundscape scores for the head locked before aug search."""
    mem_name = Path(winner.get("memory_dir", "?")).name
    arch = winner.get("arch_type", "?")
    aug = winner.get("aug_baseline", "?")
    ap = winner.get("macro_average_precision")
    auc = winner.get("macro_roc_auc")
    med = winner.get("median_per_class_auc")
    metrics_line = format_soundscape_metrics_line(
        macro_ap=float(ap) if ap is not None else None,
        macro_auc=float(auc) if auc is not None else None,
        median_auc=float(med) if med is not None else None,
        ranking_metric=ranking_metric,
        mark_ranking=True,
    )
    print("\n  STAGE 1C — LOCKED HEAD (current best before augmentation search)")
    print("  " + "─" * 72)
    print(f"  Memory dir     : {mem_name}")
    print(f"  arch_type      : {arch}")
    print(f"  aug baseline   : {aug}")
    print(f"  {metrics_line}")
    rv = winner.get("ranking_value")
    if rv is not None:
        print(
            f"  Locked head {ranking_metric}: {float(rv):.5f} "
            f"(aug search tries to beat this on soundscape val)"
        )
    print("  " + "─" * 72)
    print("  Architecture is fixed in 1c — only augmentation configs change.\n")


def _print_stage_1b_champions(candidates: list[dict], ranking_metric: str) -> None:
    """Terminal summary of top-1a models selected for refine."""
    print(f"\n  STAGE 1B — TOP {len(candidates)} CHAMPIONS FROM STAGE 1A (soundscape val)")
    print("  " + "─" * 72)
    for i, c in enumerate(candidates, 1):
        arch = c.get("arch_type", "?")
        aug = c.get("aug_baseline", "?")
        ap = c.get("macro_average_precision")
        auc = c.get("macro_roc_auc")
        med = c.get("median_per_class_auc")
        metrics_line = format_soundscape_metrics_line(
            macro_ap=float(ap) if ap is not None else None,
            macro_auc=float(auc) if auc is not None else None,
            median_auc=float(med) if med is not None else None,
            ranking_metric=ranking_metric,
            mark_ranking=True,
        )
        print(f"  Winner #{i}: arch_type = {arch}")
        print(f"            aug baseline = {aug}")
        print(f"            {metrics_line}")
        if i < len(candidates):
            print("  " + "·" * 72)
    print("  " + "─" * 72)
    print(f"  Ranking metric for refine: {ranking_metric} (must beat seed macro_AP to earn bonus tries)\n")


def _run_perch_refine_1b(config: dict, candidate: dict, rank: int, refine_cfg: dict) -> dict:
    """Run one stage-1b refine campaign for a 1a champion."""
    aug = candidate["aug_baseline"]
    arch = str(candidate.get("arch_type", "residual_mlp"))
    slug = f"rank{rank}_{aug}_{arch}".replace("/", "_")[:80]
    mem_dir = META_LOGS / "perch" / "refine" / slug
    code_dir = mem_dir / "codes"
    cache_dir = Path(candidate["cache_dir"])

    for d in (mem_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)

    perch_base = dict(config.get("perch", {}))
    head_train_cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
    if head_train_cap is None:
        head_train_cap = perch_base.get("head_train_max_samples")

    batch = refine_cfg.get("experiments_per_researcher_call")
    if batch is None:
        batch = config.get("researcher", {}).get("batch_size", 3)

    override = {
        "meta_aug_preset": aug,
        "augmentation": get_audio_embedding_aug(aug),
        "perch_refine": {
            "enabled": True,
            "aug_baseline": aug,
            "locked_arch_type": arch,
            "seed_spec": candidate["spec"],
            "seed_score": float(candidate["ranking_value"]),
            "seed_macro_ap": candidate.get("macro_average_precision"),
            "seed_macro_auc": candidate.get("macro_roc_auc"),
            "seed_median_auc": candidate.get("median_per_class_auc"),
            "parent_memory_dir": candidate["memory_dir"],
            "experiments_per_researcher_call": int(batch),
            "initial_iterations": int(refine_cfg.get("initial_iterations", 5)),
            "bonus_iterations_on_improve": int(refine_cfg.get("bonus_iterations_on_improve", 5)),
            "max_iterations_per_model": int(refine_cfg.get("max_iterations_per_model", 25)),
        },
        "head_train_max_samples": head_train_cap,
        "perch": {
            **perch_base,
            "logs_dir": str(mem_dir.parent),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "force_rebuild_cache": False,
            "skip_final_retrain": True,
        },
    }

    print(f"\n  [Perch refine 1b / rank {rank}] aug={aug} arch={arch}")
    print(
        f"  seed {candidate['ranking_metric']}={float(candidate['ranking_value']):.5f}  "
        f"→ {mem_dir}"
    )
    rc = _run_subprocess("perch_agent.py", override, config)
    return {
        "rank": rank,
        "aug_baseline": aug,
        "arch_type": arch,
        "seed_ranking_value": float(candidate["ranking_value"]),
        "memory_dir": str(mem_dir),
        "cache_dir": str(cache_dir),
        "subprocess_rc": rc,
    }


def run_stage_1b_perch_refine(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """Refine top-K perch champions from stage 1a with adaptive iteration budgets."""
    refine_cfg = _stage_1b_cfg(config)
    if not refine_cfg.get("enabled", False):
        print("\n  [Stage 1b] Skipped (stage_1b.enabled=false)")
        return {}

    if _skip_if_completed(config, ARCH_SEARCH_1B_RESULTS, "Stage 1b"):
        try:
            return json.loads(ARCH_SEARCH_1B_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    top_k = max(1, int(refine_cfg.get("top_k_models", 2)))
    metric = _meta_primary_metric(config)

    print("\n" + "=" * 60)
    print(f"  STAGED PIPELINE — Step 1b: Refine top-{top_k} Perch models from 1a")
    print(
        f"  Budget per model: {refine_cfg.get('initial_iterations', 5)} initial, "
        f"+{refine_cfg.get('bonus_iterations_on_improve', 5)} on improve, "
        f"max {refine_cfg.get('max_iterations_per_model', 25)}"
    )
    print("=" * 60)

    candidates = collect_perch_1a_top_candidates(config, top_k=top_k)
    if not candidates:
        print("  [Stage 1b] No 1a candidates found — run stage 1a first.")
        return {"stage": "1b_perch_refine", "candidates": [], "refine_runs": []}

    _print_stage_1b_champions(candidates, metric)

    refine_runs = [
        _run_perch_refine_1b(config, cand, rank=i, refine_cfg=refine_cfg)
        for i, cand in enumerate(candidates, 1)
    ]

    scored: list[dict] = []
    for run in refine_runs:
        mem = Path(run["memory_dir"])
        sc = suite.score_perch(mem)
        _print_soundscape_score(f"Perch/refine/{run['arch_type']}", sc)
        entry = {**run, "score": sc.to_dict() if sc else None}
        if sc:
            entry["refined_ranking_value"] = sc.primary_value
        scored.append(entry)

    winner = _pick_track_winner(scored, metric)
    summary = {
        "stage": "1b_perch_refine",
        "primary_metric": metric,
        "top_k_models": top_k,
        "refine_config": {
            "initial_iterations": refine_cfg.get("initial_iterations", 5),
            "bonus_iterations_on_improve": refine_cfg.get("bonus_iterations_on_improve", 5),
            "max_iterations_per_model": refine_cfg.get("max_iterations_per_model", 25),
        },
        "candidates_from_1a": candidates,
        "refine_runs": scored,
        "winner": winner,
    }
    ARCH_SEARCH_1B_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Step 1b summary → {ARCH_SEARCH_1B_RESULTS}")
    if winner:
        print(
            f"\n  ★ Perch refine winner: aug={winner.get('aug_baseline')} "
            f"arch={winner.get('arch_type')}  "
            f"{metric}={float((winner.get('score') or {}).get('primary_value', 0)):.5f}"
        )
    return summary


def _perch_memory_has_head(mem_dir: Path) -> bool:
    """True if a refine / memory dir has artifacts needed for stage 1c."""
    if not (mem_dir / "best_head_code.py").exists():
        return False
    return (mem_dir / "best_head.keras").exists() or (
        mem_dir / "best_head.weights.h5"
    ).exists()


def _ranking_from_refine_memory(mem_dir: Path, metric: str) -> float:
    """Read ranking score from best_model_info or experiment_memory (no ONNX eval)."""
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if metric == "macro_roc_auc":
                v = info.get("macro_roc_auc")
            else:
                v = info.get("macro_average_precision") or info.get("ranking_value")
            if v is not None:
                return float(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    jsonl = mem_dir / "experiment_memory.jsonl"
    if jsonl.exists():
        best_val = -1.0
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
                best_val = max(best_val, _ranking_value_from_run(entry, metric))
        if best_val >= 0:
            return best_val
    return -1.0


def _refine_dir_to_candidate(mem_dir: Path, metric: str) -> dict | None:
    """Build a 1b-style winner dict from a refine memory directory."""
    if not _perch_memory_has_head(mem_dir):
        return None
    parts = mem_dir.name.split("_", 2)
    aug = parts[1] if len(parts) > 1 else "medium"
    arch = parts[2] if len(parts) > 2 else "unknown"
    val = _ranking_from_refine_memory(mem_dir, metric)
    spec: dict = {}
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            spec = dict(info.get("spec") or {})
        except (json.JSONDecodeError, OSError):
            pass
    spec.setdefault("arch_type", arch)
    macro_ap: float | None = None
    macro_auc: float | None = None
    median_auc: float | None = None
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            macro_ap = info.get("macro_average_precision")
            macro_auc = info.get("macro_roc_auc")
            median_auc = info.get("median_per_class_auc")
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "memory_dir": str(mem_dir),
        "cache_dir": str(META_LOGS / "perch_cache" / aug),
        "aug_baseline": aug,
        "arch_type": arch,
        "spec": spec,
        "ranking_metric": metric,
        "ranking_value": val,
        "macro_average_precision": macro_ap,
        "macro_roc_auc": macro_auc,
        "median_per_class_auc": median_auc,
    }


def _pick_perch_1b_winner(config: dict) -> dict | None:
    """Best refined perch model from stage 1b (by soundscape ranking metric)."""
    metric = _meta_primary_metric(config)

    if ARCH_SEARCH_1B_RESULTS.exists():
        try:
            summary = json.loads(ARCH_SEARCH_1B_RESULTS.read_text(encoding="utf-8"))
            winner = summary.get("winner")
            if winner and winner.get("memory_dir"):
                mem = Path(winner["memory_dir"])
                if _perch_memory_has_head(mem):
                    return winner
                print(
                    f"  [1c] 1b results winner has no head ({mem.name}) — scanning refine dirs."
                )
        except (json.JSONDecodeError, OSError):
            pass

    refine_root = META_LOGS / "perch" / "refine"
    if not refine_root.exists():
        return _pick_perch_1a_fallback_champion(config, metric)

    best_run: dict | None = None
    best_val = -1.0
    for mem_dir in sorted(refine_root.iterdir()):
        if not mem_dir.is_dir():
            continue
        cand = _refine_dir_to_candidate(mem_dir, metric)
        if cand is None:
            if mem_dir.name.startswith("rank"):
                print(f"  [1c] Skipping incomplete refine run (no head): {mem_dir.name}")
            continue
        val = float(cand.get("ranking_value", -1.0))
        if val > best_val:
            best_val = val
            best_run = cand

    if best_run is not None:
        return best_run
    return _pick_perch_1a_fallback_champion(config, metric)


def _pick_perch_1a_fallback_champion(config: dict, metric: str) -> dict | None:
    """If 1b never produced a head, use best stage-1a perch baseline champion."""
    for baseline in _baseline_names(config):
        mem_dir = META_LOGS / "perch" / baseline
        champ = _load_perch_baseline_champion(baseline, mem_dir, metric)
        if champ and _perch_memory_has_head(mem_dir):
            champ["memory_dir"] = str(mem_dir)
            print(f"  [1c] Using stage-1a champion (no 1b head): {baseline} / {champ.get('arch_type')}")
            return champ
    return None


def _copy_perch_mapping_artifacts(src_mem: Path, dst_mem: Path) -> None:
    """Copy species / mapping files needed for soundscape scoring."""
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


def _ensure_head_train_clips(
    config: dict,
    source_train_cache: Path,
    indices_path: Path,
    species_to_idx: dict,
) -> Path:
    """Clip list (label + filename) for the fixed head-train subset — fast 1c re-embed."""
    if HEAD_TRAIN_CLIPS.exists():
        n = sum(1 for _ in HEAD_TRAIN_CLIPS.open(encoding="utf-8") if _.strip())
        print(f"  [1c] Reusing head-train clip list ({n} clips) → {HEAD_TRAIN_CLIPS}")
        return HEAD_TRAIN_CLIPS

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data_io import load_core_tables, resolve_birdclef_paths
    from perch_agent import save_head_train_clip_subset

    paths = resolve_birdclef_paths()
    tables = load_core_tables(paths)
    return save_head_train_clip_subset(
        source_train_cache,
        indices_path,
        HEAD_TRAIN_CLIPS,
        tables["train"],
        species_to_idx,
        paths.train_audio_dir,
    )


def _ensure_head_train_indices(
    config: dict,
    source_train_cache: Path,
    *,
    cap: int | None = None,
    seed: int = 42,
) -> Path:
    """Compute once and reuse the same 2000 head-train row indices across aug search."""
    if HEAD_TRAIN_INDICES.exists():
        idx = np.load(HEAD_TRAIN_INDICES)
        print(f"  [1c] Reusing head-train indices ({len(idx)} rows) → {HEAD_TRAIN_INDICES}")
        return HEAD_TRAIN_INDICES

    if cap is None:
        cap = _meta_cfg(config).get("arch_search_head_train_max_samples")
        if cap is None:
            cap = config.get("perch", {}).get("head_train_max_samples", 2000)
    cap = int(cap)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from perch_agent import compute_and_save_head_train_indices

    if not source_train_cache.exists():
        raise FileNotFoundError(
            f"Cannot build head-train indices — cache missing: {source_train_cache}"
        )
    return compute_and_save_head_train_indices(
        source_train_cache, HEAD_TRAIN_INDICES, cap, random_state=seed
    )


def _perch_aug_search_cache_dir(preset: str, baselines: list[str]) -> Path:
    """Reuse stage-1a caches for light/medium/high; custom presets get aug_search/."""
    key = preset.strip().lower()
    if key in baselines:
        return META_LOGS / "perch_cache" / key
    return META_LOGS / "perch_cache" / "aug_search" / key


def _ensure_shared_1c_val_cache(config: dict) -> Path:
    """Build the shared soundscape val embedding cache once for all of stage 1c."""
    if PERCH_SHARED_VAL_CACHE.exists():
        import numpy as np

        d = np.load(str(PERCH_SHARED_VAL_CACHE))
        print(
            f"  [1c] Shared val cache ready: {PERCH_SHARED_VAL_CACHE.name} "
            f"(X={d['X'].shape})"
        )
        return PERCH_SHARED_VAL_CACHE

    print("\n  [1c] Building shared soundscape val cache (once for all aug trials)…")
    mem_dir = META_LOGS / "perch" / "aug_search" / "_val_build"
    code_dir = mem_dir / "codes"
    perch_base = dict(config.get("perch", {}))
    override = {
        "perch_build_val_only": True,
        "perch": {
            **perch_base,
            "logs_dir": str(META_LOGS / "perch"),
            "memory_dir": str(mem_dir),
            "cache_dir": str(META_LOGS / "perch_cache" / "aug_search"),
            "code_dir": str(code_dir),
            "val_cache_path": str(PERCH_SHARED_VAL_CACHE),
            "skip_final_retrain": True,
        },
    }
    _run_subprocess("perch_agent.py", override, config)
    return PERCH_SHARED_VAL_CACHE


_PERCH_HEAD_ARTIFACTS = ("best_head.keras", "final_head.keras", "best_head.weights.h5")


def _perch_mem_dir_has_head(mem_dir: Path) -> bool:
    mem_dir = Path(mem_dir)
    return any((mem_dir / name).exists() for name in _PERCH_HEAD_ARTIFACTS)


def _trial_index_from_aug_preset(preset: str) -> int | None:
    m = re.match(r"aug_r(\d+)", preset)
    return int(m.group(1)) if m else None


def _merge_1c_runs_by_preset(runs: list[dict]) -> list[dict]:
    by_preset: dict[str, dict] = {}
    for run in runs:
        preset = str(run.get("aug_preset", ""))
        if preset:
            by_preset[preset] = run
    return sorted(
        by_preset.values(),
        key=lambda r: _trial_index_from_aug_preset(str(r.get("aug_preset", ""))) or 0,
    )


def _collect_resumed_perch_1c_llm_runs(
    suite: SoundscapeEvalSuite,
    *,
    baselines: list[str],
    total_runs: int,
    leader: _Stage1cLeader | None,
) -> tuple[list[dict], int]:
    """Reload completed aug_search trials that already have a trained head."""
    aug_root = META_LOGS / "perch" / "aug_search"
    runs: list[dict] = []
    if not aug_root.is_dir():
        return runs, 1

    candidates: list[tuple[int, Path]] = []
    for mem_dir in aug_root.iterdir():
        if not mem_dir.is_dir() or mem_dir.name == "memory":
            continue
        if not _perch_mem_dir_has_head(mem_dir):
            continue
        trial_idx = _trial_index_from_aug_preset(mem_dir.name)
        if trial_idx is None:
            continue
        candidates.append((trial_idx, mem_dir))
    candidates.sort(key=lambda x: x[0])

    for trial_idx, mem_dir in candidates:
        preset = mem_dir.name
        cache_dir = _perch_aug_search_cache_dir(preset, baselines)
        train_cache = cache_dir / f"train_emb_{preset}.npz"
        aug_cfg_path = cache_dir / f"{preset}_aug_config.json"
        aug_spec: dict = {}
        if aug_cfg_path.exists():
            try:
                aug_spec = {"aug_config": json.loads(aug_cfg_path.read_text(encoding="utf-8"))}
            except (json.JSONDecodeError, OSError):
                pass
        run = {
            "aug_preset": preset,
            "memory_dir": str(mem_dir),
            "train_cache": str(train_cache),
            "cache_dir": str(cache_dir),
            "aug_config_path": str(aug_cfg_path),
            "aug_spec": aug_spec,
            "resumed": True,
        }
        run = _score_1c_run(
            suite,
            run,
            preset=preset,
            aug_spec=aug_spec,
            run_index=trial_idx,
            total=total_runs,
            leader=leader,
        )
        runs.append(run)

    next_idx = max((idx for idx, _ in candidates), default=0) + 1
    return runs, next_idx


def _retry_incomplete_perch_1c_head_trains(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    baselines: list[str],
    head_code_path: Path,
    head_spec: dict,
    indices_path: Path | None,
    mapping_src: Path,
    clip_subset_path: Path | None,
    total_runs: int,
    leader: _Stage1cLeader | None,
) -> list[dict]:
    """Re-run head training when embed cache exists but best_head was never written."""
    aug_root = META_LOGS / "perch" / "aug_search"
    runs: list[dict] = []
    if not aug_root.is_dir():
        return runs

    pending: list[tuple[int, Path, Path]] = []
    for mem_dir in aug_root.iterdir():
        if not mem_dir.is_dir() or mem_dir.name == "memory":
            continue
        if _perch_mem_dir_has_head(mem_dir):
            continue
        trial_idx = _trial_index_from_aug_preset(mem_dir.name)
        if trial_idx is None:
            continue
        preset = mem_dir.name
        cache_dir = _perch_aug_search_cache_dir(preset, baselines)
        train_cache = cache_dir / f"train_emb_{preset}.npz"
        if train_cache.exists():
            pending.append((trial_idx, mem_dir, train_cache))
    pending.sort(key=lambda x: x[0])
    if not pending:
        return runs

    print(
        f"\n  [1c] Retrying head train for {len(pending)} trial(s) with embed cache but no head…",
        flush=True,
    )
    for trial_idx, mem_dir, train_cache in pending:
        preset = mem_dir.name
        cache_dir = train_cache.parent
        aug_cfg_path = cache_dir / f"{preset}_aug_config.json"
        aug_dict: dict = {}
        if aug_cfg_path.exists():
            try:
                aug_dict = json.loads(aug_cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        trial = _build_1c_trial_payload(
            preset=preset,
            aug_dict=aug_dict,
            head_code_path=head_code_path,
            head_spec={**head_spec, "aug_preset": preset, "aug_phase": "aug"},
            indices_path=indices_path if clip_subset_path is None else None,
            train_cache=train_cache,
            force_rebuild_train=False,
        )
        _run_perch_1c_sweep_subprocess(
            config,
            [trial],
            mapping_src=mapping_src,
            clip_subset_path=clip_subset_path,
        )
        if not _perch_mem_dir_has_head(mem_dir):
            label = _short_aug_display_name(preset, aug_dict)
            _print_1c_iteration_result(trial_idx, total_runs, label, None, failed=True)
            continue
        aug_spec = {"aug_config": aug_dict} if aug_dict else {}
        run = {
            "aug_preset": preset,
            "memory_dir": str(mem_dir),
            "train_cache": str(train_cache),
            "cache_dir": str(cache_dir),
            "aug_config_path": str(aug_cfg_path),
            "aug_spec": aug_spec,
            "retried_head_train": True,
        }
        run = _score_1c_run(
            suite,
            run,
            preset=preset,
            aug_spec=aug_spec,
            run_index=trial_idx,
            total=total_runs,
            leader=leader,
        )
        runs.append(run)
    return runs


class _Stage1cLeader:
    """Track best soundscape AP across all completed 1c trials (presets + LLM)."""

    def __init__(self) -> None:
        self.ap: float = -1.0
        self.label: str = ""
        self.preset: str = ""

    def update(self, ap: float, label: str, preset: str) -> bool:
        if ap > self.ap:
            prev_ap, prev_label = self.ap, self.label
            self.ap = ap
            self.label = label
            self.preset = preset
            if prev_ap < 0:
                print(
                    f"  ▸ 1c leader: {label} ({preset})  soundscape_AP={ap:.5f}",
                    flush=True,
                )
            else:
                print(
                    f"  ▸ 1c leader: {label} ({preset})  soundscape_AP={ap:.5f}  "
                    f"(was {prev_label} {prev_ap:.5f})",
                    flush=True,
                )
            return True
        return False


def _need_rebuild_1c_train_cache(
    train_cache: Path,
    *,
    clip_subset_path: Path | None,
    indices_path: Path | None,
) -> bool:
    """Rebuild if missing or not the fixed 2000-clip 1c subset (avoid reusing 1a partial caches)."""
    if not train_cache.exists():
        return True
    if clip_subset_path is None:
        return False
    import numpy as np

    try:
        n = int(np.load(str(train_cache))["X"].shape[0])
    except (OSError, KeyError, ValueError):
        return True
    want = 2000
    if indices_path is not None and Path(indices_path).exists():
        want = int(np.load(str(indices_path)).size)
    return n != want


def _build_1c_trial_payload(
    *,
    preset: str,
    aug_dict: dict,
    head_code_path: Path,
    head_spec: dict,
    indices_path: Path | None,
    train_cache: Path,
    force_rebuild_train: bool,
) -> dict:
    mem_dir = META_LOGS / "perch" / "aug_search" / preset
    cache_dir = train_cache.parent
    fixed_train: dict = {
        "enabled": True,
        "fixed_1c_trial": True,
        "label": f"aug_{preset}",
        "head_code_path": str(head_code_path),
        "aug_preset": preset,
        "spec": head_spec,
    }
    if indices_path is not None:
        fixed_train["head_train_indices_path"] = str(indices_path)
    return {
        "preset": preset,
        "augmentation": aug_dict,
        "train_cache": str(train_cache),
        "memory_dir": str(mem_dir),
        "code_dir": str(mem_dir / "codes"),
        "cache_dir": str(cache_dir),
        "force_rebuild_train": force_rebuild_train,
        "perch_fixed_train": fixed_train,
    }


def _run_perch_1c_sweep_subprocess(
    config: dict,
    trials: list[dict],
    *,
    mapping_src: Path,
    clip_subset_path: Path | None = None,
) -> int:
    """One perch process: ONNX load once, then all trials in the sweep."""
    perch_base = dict(config.get("perch", {}))
    override: dict = {
        "perch_quiet": True,
        "perch_1c_sweep": {
            "trials": trials,
            "mapping_src": str(mapping_src),
        },
        "perch": {
            **perch_base,
            "quiet_trial": True,
            "logs_dir": str(META_LOGS / "perch"),
            "cache_dir": str(META_LOGS / "perch_cache"),
            "val_cache_path": str(PERCH_SHARED_VAL_CACHE),
            "skip_final_retrain": True,
        },
    }
    if clip_subset_path is not None:
        override["perch_embed_clip_subset"] = str(clip_subset_path)
    return _run_subprocess("perch_agent.py", override, config, quiet=True)


def _score_1c_run(
    suite: SoundscapeEvalSuite,
    run: dict,
    *,
    preset: str,
    aug_spec: dict,
    run_index: int,
    total: int,
    leader: _Stage1cLeader | None = None,
) -> dict:
    mem_dir = Path(run["memory_dir"])
    if not _perch_mem_dir_has_head(mem_dir):
        label = _short_aug_display_name(preset, aug_spec)
        _print_1c_iteration_result(run_index, total, label, None, failed=True)
        run["score"] = None
        run["aug_preset"] = preset
        run["aug_spec"] = aug_spec
        run["error"] = "head_train"
        return run
    sc = suite.score_perch_mem_dir(mem_dir, val_cache=PERCH_SHARED_VAL_CACHE)
    if sc is None:
        try:
            sc = suite.score_perch(mem_dir)
        except FileNotFoundError:
            sc = None
    subset_ap = None
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            subset_ap = float(
                json.loads(info_path.read_text(encoding="utf-8")).get(
                    "macro_average_precision", 0
                )
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    label = _short_aug_display_name(preset, aug_spec)
    _print_1c_iteration_result(run_index, total, label, sc, subset_ap=subset_ap)
    if leader is not None and sc is not None:
        leader.update(float(sc.primary_value), label, preset)
    run["score"] = sc.to_dict() if sc else None
    run["aug_preset"] = preset
    run["aug_spec"] = aug_spec
    if sc:
        run["ranking_value"] = float(sc.primary_value)
    return run


def _run_perch_1c_trial(
    config: dict,
    *,
    preset: str,
    head_code_path: Path,
    train_cache: Path,
    val_cache: Path,
    indices_path: Path | None,
    mapping_src: Path,
    spec: dict,
    aug_dict: dict,
    force_rebuild_train: bool,
    clip_subset_path: Path | None,
) -> dict:
    """
    Single perch subprocess: embed train clips (if needed) + train head + eval on cached val.
    One ONNX load per iteration (not three).
    """
    mem_dir = META_LOGS / "perch" / "aug_search" / preset
    code_dir = mem_dir / "codes"
    cache_dir = train_cache.parent
    mem_dir.mkdir(parents=True, exist_ok=True)
    _copy_perch_mapping_artifacts(mapping_src, mem_dir)

    perch_base = dict(config.get("perch", {}))
    fixed_train: dict = {
        "enabled": True,
        "label": f"aug_{preset}",
        "head_code_path": str(head_code_path),
        "aug_preset": preset,
        "spec": spec,
    }
    if indices_path is not None:
        fixed_train["head_train_indices_path"] = str(indices_path)

    override: dict = {
        "meta_aug_preset": preset,
        "augmentation": aug_dict,
        "perch_quiet": True,
        "perch_fixed_train": fixed_train,
        "perch": {
            **perch_base,
            "quiet_trial": True,
            "logs_dir": str(META_LOGS / "perch"),
            "memory_dir": str(mem_dir),
            "cache_dir": str(cache_dir),
            "code_dir": str(code_dir),
            "val_cache_path": str(val_cache),
            "force_rebuild_cache": force_rebuild_train,
            "skip_final_retrain": True,
        },
    }
    if clip_subset_path is not None:
        override["perch_embed_clip_subset"] = str(clip_subset_path)

    rc = _run_subprocess("perch_agent.py", override, config, quiet=True)
    cache_dir = _perch_aug_search_cache_dir(preset, _baseline_names(config))
    aug_cfg_path = cache_dir / f"{preset}_aug_config.json"
    return {
        "aug_preset": preset,
        "memory_dir": str(mem_dir),
        "train_cache": str(train_cache),
        "cache_dir": str(cache_dir),
        "aug_config_path": str(aug_cfg_path),
        "subprocess_rc": rc,
    }


def _resolve_stage_1c_presets(cfg: dict, winner: dict) -> list[str]:
    """Which embedding aug baselines to try in preset mode (default medium + high)."""
    raw = cfg.get("aug_presets") or cfg.get("baselines") or ["medium", "high"]
    presets = [str(p).strip().lower() for p in raw if str(p).strip()]
    if not presets:
        presets = ["medium", "high"]
    winner_aug = str(winner.get("aug_baseline", "")).strip().lower()
    if cfg.get("include_winner_aug_baseline", True) and winner_aug and winner_aug not in presets:
        presets = [winner_aug] + presets
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for p in presets:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _preset_aug_logged_spec(preset: str, aug_dict: dict) -> dict:
    """Minimal spec dict for 1c results / best_aug_config.json (no LLM)."""
    return {
        "preset_name": preset,
        "strategy": "preset",
        "reasoning": f"Fixed BirdCLEF embedding baseline: {preset}",
        "hypothesis": describe_embedding_aug_compact(preset),
        **aug_dict,
    }


def _run_stage_1c_preset_grid(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    winner: dict,
    winner_mem: Path,
    head_code_path: Path,
    indices_path: Path | None,
    head_spec: dict,
    baselines: list[str],
    embed_frac: float,
    embed_max: int | None,
    presets: list[str],
    clip_subset_path: Path | None = None,
    run_index_start: int = 1,
    run_index_total: int | None = None,
    leader: _Stage1cLeader | None = None,
) -> list[dict]:
    """Try fixed light/medium/high-style embedding aug presets — no LLM researcher."""
    runs: list[dict] = []
    total = run_index_total if run_index_total is not None else len(presets)
    trials: list[dict] = []
    trial_meta: list[tuple[str, dict]] = []
    if leader is None:
        leader = _Stage1cLeader()

    for preset in presets:
        cache_dir = _perch_aug_search_cache_dir(preset, baselines)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache = cache_dir / f"train_emb_{preset}.npz"
        try:
            aug_dict = get_aug_search_preset(preset)
        except KeyError:
            aug_dict = get_audio_embedding_aug(preset)
        aug_spec = _preset_aug_logged_spec(preset, aug_dict)
        (cache_dir / f"{preset}_aug_config.json").write_text(
            json.dumps(aug_dict, indent=2), encoding="utf-8"
        )
        force_rebuild = _need_rebuild_1c_train_cache(
            train_cache,
            clip_subset_path=clip_subset_path,
            indices_path=indices_path if clip_subset_path is None else indices_path,
        )
        trials.append(
            _build_1c_trial_payload(
                preset=preset,
                aug_dict=aug_dict,
                head_code_path=head_code_path,
                head_spec=head_spec,
                indices_path=indices_path if clip_subset_path is None else None,
                train_cache=train_cache,
                force_rebuild_train=force_rebuild,
            )
        )
        trial_meta.append((preset, aug_spec))

    if trials:
        print(
            f"  [1c] Running {len(trials)} preset trial(s) in one Perch session "
            f"(ONNX loaded once)…",
            flush=True,
        )
        _run_perch_1c_sweep_subprocess(
            config,
            trials,
            mapping_src=winner_mem,
            clip_subset_path=clip_subset_path,
        )

    for j, (preset, aug_spec) in enumerate(trial_meta):
        idx = run_index_start + j
        train_cache = Path(trials[j]["train_cache"])
        if not train_cache.exists():
            _print_1c_iteration_result(idx, total, preset, None, failed=True)
            continue
        run = {
            "aug_preset": preset,
            "memory_dir": trials[j]["memory_dir"],
            "train_cache": str(train_cache),
            "cache_dir": trials[j]["cache_dir"],
            "aug_config_path": str(
                Path(trials[j]["cache_dir"]) / f"{preset}_aug_config.json"
            ),
        }
        runs.append(
            _score_1c_run(
                suite,
                run,
                preset=preset,
                aug_spec=aug_spec,
                run_index=idx,
                total=total,
                leader=leader,
            )
        )
    return runs


def _run_one_llm_aug_trial(
    config: dict,
    suite: SoundscapeEvalSuite,
    *,
    iteration: int,
    phase: str,
    researcher,
    memory,
    winner_mem: Path,
    head_code_path: Path,
    indices_path: Path | None,
    head_spec: dict,
    baselines: list[str],
    embed_frac: float,
    embed_max: int | None,
    metric: str,
    clip_subset_path: Path | None = None,
    spec: dict | None = None,
    slot_label: str = "",
) -> dict | None:
    """One aug config → cache build → fixed head train → soundscape score."""
    from aug_researcher import aug_dict_from_logged_spec, slug_from_spec

    if spec is None:
        spec = researcher.next_experiment()
    spec = dict(spec)
    spec["phase"] = phase
    spec["iteration"] = iteration
    slot = slot_label or str(spec.get("slot") or "")

    preset = slug_from_spec(spec, iteration, phase, slot=slot)
    val_cache = PERCH_SHARED_VAL_CACHE
    cache_dir = _perch_aug_search_cache_dir(preset, baselines)
    train_cache = cache_dir / f"train_emb_{preset}.npz"
    force_train = bool(_stage_1c_cfg(config).get("force_rebuild_embed_cache", False))
    display = _short_aug_display_name(preset, spec)

    cache_dir.mkdir(parents=True, exist_ok=True)
    aug_cfg_path = cache_dir / f"{preset}_aug_config.json"
    aug_cfg_path.write_text(json.dumps(aug_dict_from_logged_spec(spec), indent=2), encoding="utf-8")

    run = _run_perch_1c_trial(
        config,
        preset=preset,
        head_code_path=head_code_path,
        train_cache=train_cache,
        val_cache=val_cache,
        indices_path=indices_path if clip_subset_path is None else None,
        mapping_src=winner_mem,
        spec={**head_spec, "aug_preset": preset, "aug_phase": phase},
        aug_dict=aug_dict_from_logged_spec(spec),
        force_rebuild_train=force_train or not train_cache.exists(),
        clip_subset_path=clip_subset_path,
    )
    if not train_cache.exists():
        memory.log(spec=spec, metrics={"status": "failed", "error": "cache_build"}, code="")
        return {"aug_preset": preset, "aug_spec": spec, "failed": True, "display_name": display}

    mem_dir = Path(run["memory_dir"])
    metrics = None
    info_path = mem_dir / "best_model_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            metrics = {
                "status": "success",
                "macro_average_precision": info.get("macro_average_precision"),
                "macro_roc_auc": info.get("macro_roc_auc"),
                "median_per_class_auc": info.get("median_per_class_auc"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    sc = suite.score_perch_mem_dir(mem_dir, val_cache=PERCH_SHARED_VAL_CACHE)
    if sc is None:
        sc = suite.score_perch(mem_dir)
    run["score"] = sc.to_dict() if sc else None
    run["aug_spec"] = spec
    run["aug_config_path"] = str(aug_cfg_path)
    run["display_name"] = display
    subset_ap = (metrics or {}).get("macro_average_precision")

    if metrics is None:
        metrics = {}
    if sc is not None:
        metrics["soundscape_macro_ap"] = float(sc.primary_value)
    memory.log(spec=spec, metrics=metrics, code=json.dumps(aug_dict_from_logged_spec(spec)))

    if sc:
        run["ranking_value"] = float(sc.primary_value)
    run["_subset_ap"] = subset_ap
    run["_soundscape_score"] = sc
    return run


def _stage_1c_search_cfg(cfg: dict) -> dict:
    """LLM aug search settings (supports legacy explore/refine keys)."""
    search = dict(cfg.get("search") or cfg.get("explore") or {})
    for key in (
        "iterations",
        "planner_rounds",
        "experiments_per_round",
        "batch_size",
        "temperature",
        "researcher_timeout_seconds",
        "researcher_model",
        "researcher_provider",
        "researcher_format_json",
        "researcher_num_predict",
        "researcher_compact_prompt",
        "stream_debug",
        "quiet_logs",
    ):
        if cfg.get(key) is not None and key not in search:
            search[key] = cfg[key]
    search.setdefault("iterations", 10)
    search.setdefault("planner_rounds", search.get("iterations", 10))
    epr = search.get("experiments_per_round", search.get("batch_size", 3))
    search["experiments_per_round"] = max(1, int(epr))
    search.setdefault("temperature", 0.6)
    search.setdefault("researcher_timeout_seconds", 600)
    search.setdefault("researcher_format_json", True)
    search.setdefault("researcher_compact_prompt", True)
    search.setdefault("researcher_num_predict", 4096)
    if search.get("stream_debug") is None:
        search["stream_debug"] = False
    return search


def _resolve_stage_1c_researcher_llm(config: dict, search_cfg: dict) -> tuple[str, str]:
    """Provider + model for stage-1c aug researcher (overrides global researcher when set)."""
    llm_cfg = config.get("llm_researcher", {}) or {}
    researcher_cfg = config.get("researcher", {}) or {}
    provider = (
        search_cfg.get("researcher_provider")
        or llm_cfg.get("provider")
        or config.get("llm", {}).get("provider", "ollama")
    )
    model = (
        search_cfg.get("researcher_model")
        or researcher_cfg.get("model")
        or llm_cfg.get("model", "deepseek-r1:8b")
    )
    return str(provider), str(model)


def _run_stage_1c_llm_search(
    config: dict,
    suite: SoundscapeEvalSuite,
    search_cfg: dict,
    *,
    leader: _Stage1cLeader | None = None,
    **common,
) -> tuple[list[dict], dict | None]:
    """Single-phase LLM aug search: propose → embed → train head → score."""
    from aug_researcher import AugResearcher
    from llm_client import LLMClient
    from memory import ExperimentMemory

    planner_rounds = int(search_cfg.get("planner_rounds", search_cfg.get("iterations", 3)))
    experiments_per_round = int(search_cfg.get("experiments_per_round", 3))
    mem_dir = META_LOGS / "perch" / "aug_search" / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)

    researcher_cfg = config.get("researcher", {})
    stream_debug = bool(
        search_cfg.get("stream_debug")
        if search_cfg.get("stream_debug") is not None
        else researcher_cfg.get("stream_debug", False)
    )
    quiet_logs = bool(search_cfg.get("quiet_logs", True))
    provider, model = _resolve_stage_1c_researcher_llm(config, search_cfg)
    if provider.lower() == "ollama":
        from llm_client import resolve_ollama_model

        resolved, err = resolve_ollama_model(model)
        if resolved is None:
            print(f"\n  [1c] FATAL: {err}\n")
            raise RuntimeError(err)
        if resolved != model:
            print(f"  [1c] Resolved researcher_model {model!r} → {resolved!r}")
            model = resolved
    temp = float(search_cfg.get("temperature", researcher_cfg.get("temperature", 0.6)))

    llm_timeout = float(search_cfg.get("researcher_timeout_seconds", 300))
    llm = LLMClient(
        provider=provider,
        model=model,
        timeout_seconds=llm_timeout,
        stream_debug=stream_debug,
    )
    memory = ExperimentMemory(mem_dir, ranking_metric=common["metric"])
    researcher = AugResearcher(
        llm,
        memory,
        temperature=temp,
        compact_prompt=bool(search_cfg.get("researcher_compact_prompt", True)),
        format_json=bool(search_cfg.get("researcher_format_json", True)),
        num_predict=int(search_cfg["researcher_num_predict"])
        if search_cfg.get("researcher_num_predict") is not None
        else 4096,
        batch_size=experiments_per_round,
        quiet=quiet_logs,
    )

    runs: list[dict] = []
    run_idx = int(common.get("run_index_start", 1)) - 1
    total_runs = int(common.get("run_index_total", planner_rounds * experiments_per_round))
    trial_kw = {
        k: v
        for k, v in common.items()
        if k not in ("run_index_start", "run_index_total")
    }
    resume_partial = bool(
        search_cfg.get(
            "resume_partial_trials",
            _stage_1c_cfg(config).get("resume_partial_trials", True),
        )
    )

    winner_mem = trial_kw["winner_mem"]
    head_code_path = trial_kw["head_code_path"]
    indices_path = trial_kw["indices_path"]
    head_spec = trial_kw["head_spec"]
    baselines = trial_kw["baselines"]
    clip_subset_path = trial_kw.get("clip_subset_path")
    if leader is None:
        leader = _Stage1cLeader()
    from aug_researcher import aug_dict_from_logged_spec, slug_from_spec

    if resume_partial:
        resumed, next_idx = _collect_resumed_perch_1c_llm_runs(
            suite,
            baselines=baselines,
            total_runs=total_runs,
            leader=leader,
        )
        retried = _retry_incomplete_perch_1c_head_trains(
            config,
            suite,
            baselines=baselines,
            head_code_path=head_code_path,
            head_spec=head_spec,
            indices_path=indices_path,
            mapping_src=winner_mem,
            clip_subset_path=clip_subset_path,
            total_runs=total_runs,
            leader=leader,
        )
        runs = _merge_1c_runs_by_preset(resumed + retried)
        if runs:
            print(
                f"\n  [1c] Resumed {len(runs)} trial(s) from disk "
                f"(next new trial index: {next_idx})",
                flush=True,
            )
            run_idx = max(next_idx - 1, run_idx)

    for round_i in range(1, planner_rounds + 1):
        if run_idx >= total_runs:
            break
        print(f"\n  Round {round_i}/{planner_rounds} — LLM planner ({experiments_per_round} configs)…")
        specs = researcher.next_experiments()
        trials: list[dict] = []
        round_meta: list[tuple[int, dict, str]] = []

        for slot_i, spec in enumerate(specs, 1):
            if run_idx >= total_runs:
                break
            run_idx += 1
            slot = str(spec.get("slot") or f"a{slot_i}")
            spec = dict(spec)
            spec["phase"] = "aug"
            spec["iteration"] = run_idx
            preset = slug_from_spec(spec, run_idx, "aug", slot=slot)
            trial_idx = run_idx
            cache_dir = _perch_aug_search_cache_dir(preset, baselines)
            cache_dir.mkdir(parents=True, exist_ok=True)
            train_cache = cache_dir / f"train_emb_{preset}.npz"
            aug_dict = aug_dict_from_logged_spec(spec)
            (cache_dir / f"{preset}_aug_config.json").write_text(
                json.dumps(aug_dict, indent=2), encoding="utf-8"
            )
            force_rebuild = _need_rebuild_1c_train_cache(
                train_cache,
                clip_subset_path=clip_subset_path,
                indices_path=indices_path if clip_subset_path is None else indices_path,
            )
            if bool(_stage_1c_cfg(config).get("force_rebuild_embed_cache", False)):
                force_rebuild = True
            trials.append(
                _build_1c_trial_payload(
                    preset=preset,
                    aug_dict=aug_dict,
                    head_code_path=head_code_path,
                    head_spec={**head_spec, "aug_preset": preset, "aug_phase": "aug"},
                    indices_path=indices_path if clip_subset_path is None else None,
                    train_cache=train_cache,
                    force_rebuild_train=force_rebuild,
                )
            )
            round_meta.append((trial_idx, spec, preset))

        if trials:
            print(
                f"  [1c] Running {len(trials)} LLM trial(s) in one Perch session (ONNX loaded once)…",
                flush=True,
            )
            _run_perch_1c_sweep_subprocess(
                config,
                trials,
                mapping_src=winner_mem,
                clip_subset_path=clip_subset_path,
            )

        for (trial_idx, spec, preset), trial in zip(round_meta, trials):
            label = _short_aug_display_name(preset, spec)
            train_cache = Path(trial["train_cache"])
            if not train_cache.exists():
                memory.log(
                    spec=spec,
                    metrics={"status": "failed", "error": "cache_build"},
                    code="",
                )
                _print_1c_iteration_result(trial_idx, total_runs, label, None, failed=True)
                continue
            mem_path = Path(trial["memory_dir"])
            if not _perch_mem_dir_has_head(mem_path):
                memory.log(
                    spec=spec,
                    metrics={"status": "failed", "error": "head_train"},
                    code="",
                )
                _print_1c_iteration_result(trial_idx, total_runs, label, None, failed=True)
                continue
            run = {
                "aug_preset": preset,
                "memory_dir": trial["memory_dir"],
                "train_cache": str(train_cache),
                "cache_dir": trial["cache_dir"],
                "aug_config_path": str(Path(trial["cache_dir"]) / f"{preset}_aug_config.json"),
                "aug_spec": spec,
            }
            run = _score_1c_run(
                suite,
                run,
                preset=preset,
                aug_spec=spec,
                run_index=trial_idx,
                total=total_runs,
                leader=leader,
            )
            metrics = None
            info_path = Path(run["memory_dir"]) / "best_model_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    metrics = {
                        "status": "success",
                        "macro_average_precision": info.get("macro_average_precision"),
                        "macro_roc_auc": info.get("macro_roc_auc"),
                        "median_per_class_auc": info.get("median_per_class_auc"),
                    }
                except (json.JSONDecodeError, OSError):
                    pass
            if metrics is None:
                metrics = {}
            if run.get("score"):
                metrics["soundscape_macro_ap"] = float(run["score"]["primary_value"])
            memory.log(
                spec=spec,
                metrics=metrics,
                code=json.dumps(aug_dict_from_logged_spec(spec)),
            )
            runs.append(run)

    return _merge_1c_runs_by_preset(runs), None


def run_staged_tournament_resume_phase(
    config: dict,
    suite: SoundscapeEvalSuite,
) -> dict:
    """
    Finish tournament Phase A after CNN 1c completed and Perch 1c was interrupted.
    Reuses CNN candidate from ``cnn_arch_search_1c_results.json``; runs/resumes Perch 1c.
    """
    metric = _meta_primary_metric(config)
    order = _track_order(config)

    print("\n" + "=" * 60)
    print("  TOURNAMENT RESUME — CNN from checkpoint + Perch 1c (with partial resume)")
    print(f"  Track order: {' → '.join(order)}")
    print("=" * 60)

    candidates: list[dict] = []
    for track in order:
        if not _track_active(config, track, "staged_tournament"):
            continue
        if track == "cnn":
            cand = _collect_tournament_candidate("cnn", config, metric)
            if cand:
                candidates.append(cand)
                print(
                    f"\n  [Tournament resume] CNN from checkpoint: "
                    f"{metric}={float(cand['primary_value']):.5f}"
                )
            else:
                print("\n  [Tournament resume] CNN — no 1c results; run CNN 1c first.")
        elif track == "perch":
            if not _skip_if_completed(config, ARCH_SEARCH_1C_RESULTS, "Perch 1c (resume)"):
                run_stage_1c_aug_search(config, suite)
            cand = _collect_tournament_candidate("perch", config, metric)
            if cand:
                candidates.append(cand)
                print(
                    f"\n  [Tournament resume] Perch qualified: "
                    f"{metric}={float(cand['primary_value']):.5f}"
                )
            else:
                print("\n  [Tournament resume] Perch — no valid 1c winner after resume.")
        elif track == "birdnet":
            cand = _collect_tournament_candidate("birdnet", config, metric)
            if cand:
                candidates.append(cand)

    global_winner = _pick_global_tournament_winner(candidates, metric)
    _print_tournament_final_comparison(config, candidates, metric, global_winner)
    payload = _save_tournament_results(config, candidates, global_winner)
    if global_winner:
        print(f"\n  Tournament results saved → {TOURNAMENT_RESULTS}")
        print("  Next: pipeline staged_finalize (or tournament.auto_finalize: true)")
    return payload


def run_stage_1c_aug_search(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """
    Stage 1c: lock best 1b head; single-phase LLM augmentation search (propose → test).
    """
    cfg = _stage_1c_cfg(config)
    if not cfg.get("enabled", False):
        print("\n  [Stage 1c] Skipped (stage_1c.enabled=false)")
        return {}

    if _skip_if_completed(config, ARCH_SEARCH_1C_RESULTS, "Stage 1c"):
        try:
            return json.loads(ARCH_SEARCH_1C_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    winner = _pick_perch_1b_winner(config)
    if not winner:
        print("\n  [Stage 1c] No 1b winner — run stage 1b first.")
        return {"stage": "1c_aug_search", "runs": [], "winner": None}

    winner_mem = Path(winner["memory_dir"])
    head_code_path = winner_mem / "best_head_code.py"
    if not head_code_path.exists():
        print(f"\n  [Stage 1c] Missing {head_code_path} — cannot lock head.")
        return {"stage": "1c_aug_search", "runs": [], "winner": None}

    metric = _meta_primary_metric(config)
    winner = _enrich_perch_winner_metrics(winner, metric, suite)
    _print_stage_1c_locked_head(winner, metric)

    baselines = _baseline_names(config)
    embed_frac = float(_meta_cfg(config).get("arch_search_embed_sample_frac", 0.5))
    embed_max = _embed_max_samples(config)

    aug_preset = winner.get("aug_baseline", "medium")
    source_cache = Path(winner.get("cache_dir", META_LOGS / "perch_cache" / aug_preset))
    source_train = source_cache / f"train_emb_{aug_preset}.npz"
    if not source_train.exists():
        source_train = source_cache / "train_emb.npz"

    mode = str(cfg.get("mode", "presets")).lower()
    if mode in ("baseline", "baselines", "fixed"):
        mode = "presets"
    search_cfg = _stage_1c_search_cfg(cfg)
    preset_list = _resolve_stage_1c_presets(cfg, winner)

    _ensure_shared_1c_val_cache(config)

    include_presets = bool(cfg.get("include_preset_baselines", mode in ("presets", "both")))
    if mode in ("llm", "llm_batch") and include_presets:
        mode_label = "presets + LLM"
    else:
        mode_label = mode

    indices_path = _ensure_head_train_indices(
        config, source_train, cap=cfg.get("head_train_samples")
    )

    species_cols = json.loads((winner_mem / "species_cols.json").read_text(encoding="utf-8"))
    species_to_idx = {s: i for i, s in enumerate(species_cols)}
    clip_subset_path: Path | None = None
    if cfg.get("embed_subset_only", True):
        clip_subset_path = _ensure_head_train_clips(
            config, source_train, indices_path, species_to_idx
        )
    fast_line = (
        f"Fast embed: {cfg.get('head_train_samples', 2000)} clips/trial "
        f"(locked head from 1b)"
    )
    _print_1c_phase_header(
        f"STAGE 1c — AUGMENTATION SEARCH ({mode_label})",
        lines=[
            f"Metric: {metric} on labeled train_soundscapes",
            fast_line if cfg.get("embed_subset_only", True) else "Full train embed per trial",
        ],
    )

    head_spec: dict = {}
    info_path = winner_mem / "best_model_info.json"
    if info_path.exists():
        try:
            head_spec = json.loads(info_path.read_text(encoding="utf-8")).get("spec") or {}
        except (json.JSONDecodeError, OSError):
            pass
    head_spec.setdefault("arch_type", winner.get("arch_type"))

    common = {
        "winner_mem": winner_mem,
        "head_code_path": head_code_path,
        "indices_path": indices_path,
        "head_spec": head_spec,
        "baselines": baselines,
        "embed_frac": embed_frac,
        "embed_max": embed_max,
        "metric": metric,
        "clip_subset_path": clip_subset_path,
    }

    llm_runs: list[dict] = []
    preset_runs: list[dict] = []
    llm_winner: dict | None = None
    final_winner: dict | None = None

    n_presets = len(preset_list) if include_presets else 0
    n_llm = 0
    if mode in ("llm", "llm_batch", "both"):
        n_llm = int(search_cfg.get("planner_rounds", 3)) * int(
            search_cfg.get("experiments_per_round", 3)
        )
    total_trials = n_presets + n_llm
    run_idx = 1
    leader = _Stage1cLeader()

    if include_presets and n_presets:
        print(f"\n  Preset baselines ({n_presets})")
        preset_runs = _run_stage_1c_preset_grid(
            config,
            suite,
            winner=winner,
            winner_mem=common["winner_mem"],
            head_code_path=common["head_code_path"],
            indices_path=common["indices_path"],
            head_spec=common["head_spec"],
            baselines=common["baselines"],
            embed_frac=common["embed_frac"],
            embed_max=common["embed_max"],
            presets=preset_list,
            clip_subset_path=common["clip_subset_path"],
            run_index_start=run_idx,
            run_index_total=total_trials,
            leader=leader,
        )
        run_idx += n_presets

    if mode in ("llm", "llm_batch", "both"):
        print(f"\n  LLM custom configs (~{n_llm} trials)")
        common_llm = {
            **common,
            "run_index_start": run_idx,
            "run_index_total": total_trials,
        }
        llm_runs, _ = _run_stage_1c_llm_search(
            config, suite, search_cfg, leader=leader, **common_llm
        )

    all_runs = preset_runs + llm_runs
    final_winner = _finalize_1c_winner(
        all_runs, metric=metric, baselines=baselines, announce=True
    )
    llm_winner = _finalize_1c_winner(llm_runs, metric=metric, baselines=baselines)
    preset_winner = _finalize_1c_winner(preset_runs, metric=metric, baselines=baselines)

    best_aug_config: dict | None = None
    if final_winner:
        best_aug_config = _load_aug_dict_for_1c_winner(final_winner, baselines)
    if best_aug_config:
        best_aug_path = META_LOGS / "perch" / "aug_search" / "best_aug_config.json"
        best_aug_path.parent.mkdir(parents=True, exist_ok=True)
        best_aug_path.write_text(json.dumps(best_aug_config, indent=2), encoding="utf-8")

    summary = {
        "stage": "1c_aug_search",
        "mode": mode,
        "primary_metric": metric,
        "head_train_indices": str(indices_path),
        "locked_head_code": str(head_code_path),
        "refine_winner_1b": winner,
        "llm_runs": llm_runs,
        "llm_winner": llm_winner,
        "explore_runs": llm_runs,
        "explore_winner": llm_winner,
        "refine_runs": [],
        "refine_winner": None,
        "preset_runs": preset_runs,
        "preset_winner": preset_winner,
        "winner": final_winner,
        "best_aug_config": str(META_LOGS / "perch" / "aug_search" / "best_aug_config.json")
        if best_aug_config
        else None,
    }
    ARCH_SEARCH_1C_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {ARCH_SEARCH_1C_RESULTS}")
    return summary


def run_stage_1d_final_train(
    config: dict,
    suite: SoundscapeEvalSuite | None = None,
) -> dict:
    """Stage 1d: full-data retrain with best 1b head + best 1c augmentation."""
    if suite is None:
        suite = _soundscape_suite(config)
    cfg = _stage_1d_cfg(config)
    if not cfg.get("enabled", False):
        print("\n  [Stage 1d] Skipped (stage_1d.enabled=false)")
        return {}

    if not ARCH_SEARCH_1C_RESULTS.exists():
        print("\n  [Stage 1d] No 1c results — run stage 1c first.")
        return {}

    try:
        s1c = json.loads(ARCH_SEARCH_1C_RESULTS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("\n  [Stage 1d] Could not read 1c results.")
        return {}

    aug_winner = s1c.get("winner")
    refine_winner = s1c.get("refine_winner_1b") or _pick_perch_1b_winner(config)
    if not aug_winner or not refine_winner:
        print("\n  [Stage 1d] Missing 1b/1c winners.")
        return {}

    preset = aug_winner.get("aug_preset", "medium")
    head_code_path = Path(s1c.get("locked_head_code", ""))
    if not head_code_path.exists():
        head_code_path = Path(refine_winner["memory_dir"]) / "best_head_code.py"
    if not head_code_path.exists():
        print(f"\n  [Stage 1d] Head code missing: {head_code_path}")
        return {}

    baselines = _baseline_names(config)
    full_frac = float(cfg.get("embed_sample_frac", 1.0))
    val_cache = PERCH_SHARED_VAL_CACHE if PERCH_SHARED_VAL_CACHE.exists() else (
        META_LOGS / "perch_cache" / "val_emb.npz"
    )

    best_aug_path = META_LOGS / "perch" / "aug_search" / "best_aug_config.json"
    aug_dict = _load_aug_dict_for_1c_winner(aug_winner, baselines)
    if not best_aug_path.exists() and aug_dict:
        best_aug_path.parent.mkdir(parents=True, exist_ok=True)
        best_aug_path.write_text(json.dumps(aug_dict, indent=2), encoding="utf-8")
        print(f"  [Stage 1d] Wrote {best_aug_path.name} from 1c winner config")

    final_cache_dir = META_LOGS / "perch_cache" / "final"
    train_cache = final_cache_dir / f"train_emb_{preset}_full.npz"
    if cfg.get("rebuild_full_train_cache", True) or not train_cache.exists():
        print(
            f"\n  [Stage 1d] Building FULL train embeddings "
            f"(sample_frac={full_frac}, aug={preset}) — for Kaggle submission"
        )
        final_cache_dir.mkdir(parents=True, exist_ok=True)
        mem_dir = META_LOGS / "perch" / "final" / "_cache_build"
        code_dir = mem_dir / "codes"
        perch_base = dict(config.get("perch", {}))
        embed_cap = _stage_1d_embed_cap(config, cfg)
        cap_msg = (
            f"all train.csv focal clips (no cap)"
            if embed_cap is None
            else f"cap={embed_cap} clips"
        )
        print(f"  [Stage 1d] Embed budget: {cap_msg}")
        perch_1d = {
            **perch_base,
            "logs_dir": str(META_LOGS / "perch"),
            "memory_dir": str(mem_dir),
            "cache_dir": str(final_cache_dir),
            "code_dir": str(code_dir),
            "val_cache_path": str(val_cache),
            "force_rebuild_cache": bool(cfg.get("rebuild_full_train_cache", True)),
            "skip_final_retrain": True,
            "max_train_samples": embed_cap,
            "head_train_max_samples": None,
        }
        override = {
            "meta_aug_preset": f"{preset}_full",
            "augmentation": aug_dict,
            "train_sample_frac": full_frac,
            "head_train_max_samples": None,
            "perch_build_cache_only": True,
            "perch": perch_1d,
        }
        _run_subprocess("perch_agent.py", override, config)
        meta_path = final_cache_dir / f"train_emb_{preset}_full.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                print(
                    f"  [Stage 1d] Full embed done: n_samples={meta.get('n_samples')} "
                    f"dim={meta.get('embedding_dim')}"
                )
            except (json.JSONDecodeError, OSError):
                pass
        built = final_cache_dir / f"train_emb_{preset}_full.npz"
        if built.exists():
            train_cache = built
        elif not train_cache.exists():
            cache_dir = _perch_aug_search_cache_dir(preset, baselines)
            train_cache = cache_dir / f"train_emb_{preset}.npz"
            print(
                f"  [Stage 1d] Full cache build missing — falling back to search cache: "
                f"{train_cache.name}"
            )

    if not train_cache.exists():
        print(f"\n  [Stage 1d] Train cache missing: {train_cache}")
        return {}

    PERCH_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = PERCH_FINAL_DIR / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    _copy_perch_mapping_artifacts(Path(refine_winner["memory_dir"]), PERCH_FINAL_DIR)

    head_code = head_code_path.read_text(encoding="utf-8")
    print("\n" + "=" * 60)
    print("  STAGED PIPELINE — Step 1d: Final full-data retrain")
    print(f"  Head: {head_code_path.name}  |  Aug: {preset}")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from code_executor import CodeExecutor
    from perch_agent import _build_final_retrain_script

    py_exe = config.get("execution", {}).get("python_executable", "python3")
    timeout = config.get("execution", {}).get("timeout_seconds", 1800)
    if cfg.get("final_timeout_seconds"):
        timeout = int(cfg["final_timeout_seconds"])
    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)

    script = _build_final_retrain_script(
        head_code, PERCH_FINAL_DIR, train_cache, val_cache
    )
    script_path = code_dir / "final_retrain.py"
    script_path.write_text(script, encoding="utf-8")
    result = executor.run_file(script_path)

    ok = result.success and "FINAL_RETRAIN_DONE" in (result.stdout or "")
    metric = _meta_primary_metric(config)
    soundscape_score: dict | None = None
    if ok:
        info_src = Path(refine_winner["memory_dir"]) / "best_model_info.json"
        if info_src.exists():
            shutil.copy2(info_src, PERCH_FINAL_DIR / "best_model_info.json")
        if cfg.get("score_after_retrain", True):
            try:
                sc = suite.score_perch(PERCH_FINAL_DIR)
                if sc:
                    soundscape_score = sc.to_dict()
                    _print_soundscape_score("Perch/1d (full train head, live soundscapes)", sc)
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"  [Stage 1d] Soundscape score skipped: {exc}")
        weights = PERCH_FINAL_DIR / "final_head.weights.h5"
        keras_model = PERCH_FINAL_DIR / "final_head.keras"
        print(f"  Final model → {weights}")
        if cfg.get("copy_to_submission", True):
            sub = PROJECT_ROOT / "submission"
            sub.mkdir(parents=True, exist_ok=True)
            if weights.exists():
                shutil.copy2(weights, sub / "perch_final_head.weights.h5")
            if keras_model.exists():
                shutil.copy2(keras_model, sub / "perch_final_head.keras")
            shutil.copy2(head_code_path, sub / "perch_best_head_code.py")
            for name in (
                "bc_indices.npy",
                "proxy_map.json",
                "species_cols.json",
                "mapping_meta.json",
            ):
                src = PERCH_FINAL_DIR / name
                if src.exists():
                    shutil.copy2(src, sub / name)
            if best_aug_path.exists():
                shutil.copy2(best_aug_path, sub / "perch_best_aug_config.json")
            readme = sub / "PERCH_KAGGLE_README.txt"
            readme.write_text(
                "Perch overnight pipeline outputs (BirdCLEF 2026)\n"
                "============================================\n"
                f"Head code:     perch_best_head_code.py  (build_head + get_training_config)\n"
                f"Weights:       perch_final_head.weights.h5\n"
                f"Mapping:       bc_indices.npy, proxy_map.json, species_cols.json\n"
                f"Aug config:    perch_best_aug_config.json\n"
                f"Full train NPZ: ../logs/meta_agent/perch_cache/final/train_emb_{preset}_full.npz\n"
                f"Val NPZ:        ../logs/meta_agent/perch_cache/val_emb.npz\n"
                f"Run kaggle_inference.ipynb or your notebook with these paths.\n",
                encoding="utf-8",
            )
            print(f"  Kaggle-ready artifacts → {sub}")
    else:
        print(f"  [Stage 1d] Final retrain failed: {(result.stderr or '')[-500:]}")

    summary = {
        "stage": "1d_final_train",
        "aug_preset": preset,
        "train_cache": str(train_cache),
        "output_dir": str(PERCH_FINAL_DIR),
        "success": ok,
        "primary_metric": metric,
        "soundscape_score": soundscape_score,
    }
    ARCH_SEARCH_1D_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_staged_perch_1d_1e(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """Perch finalize only: full-data 1d (embed + head + soundscape score) then 1e."""
    r1d = run_stage_1d_final_train(config, suite)
    r1e: dict = {}
    if _stage_1e_cfg(config).get("enabled", True):
        if not _skip_if_completed(config, ARCH_SEARCH_1E_RESULTS, "Perch 1e"):
            r1e = run_stage_1e_pseudo_refine(config, suite)
    return {"stage_1d": r1d, "stage_1e": r1e}


def run_stage_1e_pseudo_refine(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """
    Stage 1e: embed unlabeled soundscapes, soft pseudo-labels, fine-tune the 1d head.

    Supervised data = focal full train cache from 1d plus labeled soundscape val cache
    (same as 1d final retrain). Pseudo windows are embedded without augmentation and
    filtered with fixed top1/runner-up thresholds.
    """
    cfg = _stage_1e_cfg(config)
    if not cfg.get("enabled", False):
        print("\n  [Stage 1e] Skipped (stage_1e.enabled=false)")
        return {}

    if _skip_if_completed(config, ARCH_SEARCH_1E_RESULTS, "Stage 1e"):
        try:
            return json.loads(ARCH_SEARCH_1E_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    teacher_head = PERCH_FINAL_DIR / "final_head.keras"
    teacher_weights = PERCH_FINAL_DIR / "final_head.weights.h5"
    if not teacher_head.exists() and not teacher_weights.exists():
        print("\n  [Stage 1e] No stage-1d final head — run stage 1d first.")
        return {"stage": "1e_pseudo_refine", "success": False}

    train_cache: Path | None = None
    preset = "medium"
    if ARCH_SEARCH_1D_RESULTS.exists():
        try:
            s1d = json.loads(ARCH_SEARCH_1D_RESULTS.read_text(encoding="utf-8"))
            if s1d.get("train_cache"):
                train_cache = Path(s1d["train_cache"])
            preset = str(s1d.get("aug_preset", preset))
        except (json.JSONDecodeError, OSError):
            pass
    if train_cache is None or not train_cache.exists():
        train_cache = META_LOGS / "perch_cache" / "final" / f"train_emb_{preset}_full.npz"
    if not train_cache.exists():
        print(f"\n  [Stage 1e] Train cache missing: {train_cache}")
        return {"stage": "1e_pseudo_refine", "success": False}

    head_code_path: Path | None = None
    if ARCH_SEARCH_1C_RESULTS.exists():
        try:
            s1c = json.loads(ARCH_SEARCH_1C_RESULTS.read_text(encoding="utf-8"))
            locked = s1c.get("locked_head_code")
            if locked:
                head_code_path = Path(locked)
        except (json.JSONDecodeError, OSError):
            pass
    for candidate in (PERCH_FINAL_DIR / "best_head_code.py",):
        if candidate.exists():
            head_code_path = candidate
            break
    if head_code_path is None:
        sub_head = PROJECT_ROOT / "submission" / "perch_best_head_code.py"
        if sub_head.exists():
            head_code_path = sub_head
    if head_code_path is None or not head_code_path.exists():
        print("\n  [Stage 1e] Missing best_head_code.py — cannot refine architecture.")
        return {"stage": "1e_pseudo_refine", "success": False}

    top1 = float(cfg.get("top1_threshold", 0.55))
    runnerup = float(cfg.get("runnerup_max", 0.35))
    pl_weight = float(cfg.get("pseudo_label_weight", 0.8))
    sw_sup = float(cfg.get("sample_weight_supervised", 1.0))
    sw_val = float(cfg.get("sample_weight_labeled_val", sw_sup))
    sw_ps = float(cfg.get("sample_weight_pseudo", 0.5))
    rebuild = bool(cfg.get("rebuild_pseudo_cache", False))
    include_labeled_val = bool(cfg.get("include_labeled_val_in_refine", True))
    val_cache = PERCH_SHARED_VAL_CACHE if include_labeled_val and PERCH_SHARED_VAL_CACHE.exists() else None

    print("\n" + "=" * 60)
    print("  STAGED PIPELINE — Step 1e: Pseudo-label refine")
    print(f"  Thresholds: top1≥{top1}  runner-up<{runnerup}  soft_label×{pl_weight}")
    print(f"  Supervised cache: {train_cache.name}  (focal train, from 1d)")
    if val_cache is not None:
        print(f"  Labeled val cache: {val_cache.name}  (competition soundscape windows)")
    else:
        print("  Labeled val cache: skipped")
    print("=" * 60)

    pseudo_stats: dict = {}
    if rebuild or not PSEUDO_LABELS_NPZ.exists():
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from perch_pseudo import build_pseudo_label_cache

        teacher_path = teacher_head if teacher_head.exists() else teacher_weights
        pseudo_stats = build_pseudo_label_cache(
            config=config,
            teacher_head_path=teacher_path,
            out_path=PSEUDO_LABELS_NPZ,
            soundscapes_dir=TRAIN_SOUNDSCAPES,
            labels_csv=SOUNDSCAPE_LABELS,
            top1_threshold=top1,
            runnerup_max=runnerup,
            pseudo_label_weight=pl_weight,
            file_batch_size=int(cfg.get("file_batch_size", 16)),
            embed_batch_size=int(config.get("perch", {}).get("embed_batch_size", 16)),
            max_files=cfg.get("max_soundscape_files"),
        )
        if pseudo_stats.get("empty_pseudo"):
            print("\n  [1e] Continuing with supervised-only fine-tune (no pseudo windows).")
    else:
        d = np.load(str(PSEUDO_LABELS_NPZ), allow_pickle=True)
        pseudo_stats = {
            "n_accepted": int(d["n_accepted"]) if "n_accepted" in d.files else int(d["X_pseudo"].shape[0]),
            "out_path": str(PSEUDO_LABELS_NPZ),
            "reused_cache": True,
        }
        print(
            f"\n  [1e] Reusing pseudo cache → {PSEUDO_LABELS_NPZ.name} "
            f"({pseudo_stats['n_accepted']} windows)"
        )

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from code_executor import CodeExecutor
    from perch_agent import _build_pseudo_refine_script

    PERCH_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = PERCH_FINAL_DIR / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    head_code = head_code_path.read_text(encoding="utf-8")

    py_exe = config.get("execution", {}).get("python_executable", "python3")
    timeout = int(config.get("execution", {}).get("timeout_seconds", 1800))
    if cfg.get("refine_timeout_seconds"):
        timeout = int(cfg["refine_timeout_seconds"])
    executor = CodeExecutor(python_executable=py_exe, timeout_seconds=timeout)

    init_weights = teacher_head if teacher_head.exists() else teacher_weights
    script = _build_pseudo_refine_script(
        head_code,
        PERCH_FINAL_DIR,
        train_cache,
        PSEUDO_LABELS_NPZ,
        val_cache=val_cache,
        init_weights=init_weights,
        fine_tune_lr=float(cfg.get("fine_tune_lr", 2e-4)),
        epochs=int(cfg.get("fine_tune_epochs", 15)),
        val_split=float(cfg.get("val_split", 0.1)),
        sample_weight_supervised=sw_sup,
        sample_weight_labeled_val=sw_val,
        sample_weight_pseudo=sw_ps,
    )
    script_path = code_dir / "pseudo_refine.py"
    script_path.write_text(script, encoding="utf-8")
    score_1d: dict | None = None
    if teacher_head.exists():
        for stale in ("final_head_pseudo.keras", "final_head_pseudo.weights.h5"):
            (PERCH_FINAL_DIR / stale).unlink(missing_ok=True)
        try:
            sc_before = suite.score_perch(PERCH_FINAL_DIR)
            if sc_before:
                score_1d = sc_before.to_dict()
                _print_soundscape_score("Perch/1d (pre-pseudo head)", sc_before)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  [1e] Pre-pseudo score skipped: {exc}")

    print("\n  [1e] Fine-tuning head on supervised + pseudo embeddings…")
    result = executor.run_file(script_path)
    ok = result.success and "PSEUDO_REFINE_DONE" in (result.stdout or "")

    score_1e: dict | None = None
    if ok:
        pseudo_keras = PERCH_FINAL_DIR / "final_head_pseudo.keras"
        pseudo_weights = PERCH_FINAL_DIR / "final_head_pseudo.weights.h5"
        print(f"  [1e] Saved → {pseudo_keras.name}")
        # Prefer pseudo head for downstream scoring / submission
        if pseudo_keras.exists():
            shutil.copy2(pseudo_keras, PERCH_FINAL_DIR / "best_head.keras")
        if pseudo_weights.exists():
            shutil.copy2(pseudo_weights, PERCH_FINAL_DIR / "best_head.weights.h5")
        try:
            sc_after = suite.score_perch(PERCH_FINAL_DIR)
            if sc_after:
                score_1e = sc_after.to_dict()
                _print_soundscape_score("Perch/1e (pseudo-refined)", sc_after)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  [1e] Post-pseudo score skipped: {exc}")
        if cfg.get("copy_to_submission", True):
            sub = PROJECT_ROOT / "submission"
            sub.mkdir(parents=True, exist_ok=True)
            if pseudo_keras.exists():
                _safe_copy2(pseudo_keras, sub / "perch_final_head_pseudo.keras")
                _safe_copy2(pseudo_keras, sub / "perch_final_head.keras")
            if pseudo_weights.exists():
                _safe_copy2(pseudo_weights, sub / "perch_final_head_pseudo.weights.h5")
                _safe_copy2(pseudo_weights, sub / "perch_final_head.weights.h5")
            _safe_copy2(head_code_path, sub / "perch_best_head_code.py")
            print(f"  [1e] Submission bundle updated → {sub}")
    else:
        print(f"  [Stage 1e] Pseudo refine failed: {(result.stderr or '')[-500:]}")

    summary = {
        "stage": "1e_pseudo_refine",
        "success": ok,
        "train_cache": str(train_cache),
        "pseudo_cache": str(PSEUDO_LABELS_NPZ),
        "pseudo_stats": pseudo_stats,
        "thresholds": {"top1": top1, "runnerup_max": runnerup, "pseudo_label_weight": pl_weight},
        "output_dir": str(PERCH_FINAL_DIR),
        "score_1d": score_1d,
        "score_1e": score_1e,
    }
    ARCH_SEARCH_1E_RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {ARCH_SEARCH_1E_RESULTS}")
    return summary


def _skip_if_completed(config: dict, results_path: Path, stage_label: str) -> bool:
    if not _meta_cfg(config).get("skip_completed_stages", False):
        return False
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            if payload.get("success") is False:
                return False
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        print(f"\n  [{stage_label}] Skipped — results already at {results_path.name}")
        return True
    return False


def run_stage_1a_arch_search(config: dict, suite: SoundscapeEvalSuite) -> dict:
    """
    Step 1a only — all tracks in ``track_order`` (architecture search per track).
    For full per-track pipelines use ``run_staged_pipeline_sequential``.
    """
    if _skip_if_completed(config, ARCH_SEARCH_1A_RESULTS, "Stage 1a"):
        try:
            return json.loads(ARCH_SEARCH_1A_RESULTS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    run_staged_pipeline_sequential(config, suite, "staged_1a")
    return _load_1a_summary(config)


def save_soundscape_leaderboard(
    scores: dict[str, SoundscapeScore | None],
    ensemble_cfg: dict,
    config: dict,
) -> None:
    """Persist ranked soundscape metrics for all tracks (v2 primary metric)."""
    META_LOGS.mkdir(parents=True, exist_ok=True)
    metric = _meta_primary_metric(config)
    rows = []
    for track, sc in scores.items():
        if sc is None:
            continue
        row = {"track": track, **sc.to_dict()}
        rows.append(row)
    if ensemble_cfg:
        rows.append({
            "track": "ensemble",
            "primary_metric": ensemble_cfg.get("primary_metric", metric),
            "primary_value": ensemble_cfg.get("best_ensemble_primary"),
            "macro_average_precision": ensemble_cfg.get("best_ensemble_macro_ap"),
            "competition_macro_auc": ensemble_cfg.get("best_ensemble_macro_auc"),
            "perch_weight": ensemble_cfg.get("perch_weight"),
            "birdnet_weight": ensemble_cfg.get("birdnet_weight"),
        })
    rows.sort(key=lambda r: float(r.get("primary_value") or -1.0), reverse=True)
    payload = {
        "primary_metric": metric,
        "evaluation": "labeled_train_soundscapes",
        "n_windows": scores.get("cnn") and scores["cnn"].n_windows,
        "ranking": rows,
    }
    SOUNDSCAPE_LEADERBOARD.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Soundscape leaderboard ({metric}) → {SOUNDSCAPE_LEADERBOARD}")
    for i, row in enumerate(rows, 1):
        print(
            f"    #{i} {row['track']:10s}  "
            f"{format_soundscape_metrics_line(macro_ap=row.get('macro_average_precision'), macro_auc=row.get('competition_macro_auc'), median_auc=row.get('median_per_class_auc'), ranking_metric=metric)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "agent_config.json"))
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Override meta_agent.pipeline (e.g. staged_cnn_1c_only, staged_finalize, staged_tournament)",
    )
    args   = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    meta_cfg        = _meta_cfg(config)
    pipeline        = str(args.pipeline or meta_cfg.get("pipeline", "legacy")).lower()
    cnn_iters       = int(meta_cfg.get("cnn_iterations",        5))
    birdnet_iters   = int(meta_cfg.get("birdnet_iterations",   10))
    perch_iters     = int(meta_cfg.get("perch_iterations",     10))
    ensemble_iters  = int(meta_cfg.get("ensemble_iterations",   5))

    t0 = time.time()

    metric = _meta_primary_metric(config)
    suite = _soundscape_suite(config)
    print(f"\n  Meta-agent pipeline: {pipeline}")
    print(f"  Ranking metric: {metric} on labeled train_soundscapes")

    if pipeline == "staged_cnn_1c_only":
        ensure_meta_eda_before_tracks(config)
        run_stage_1c_cnn_aug_search(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        return

    if pipeline == "staged_perch_1c_only":
        ensure_meta_eda_before_tracks(config)
        run_stage_1c_aug_search(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        return

    if pipeline == "staged_perch_1d_1e":
        ensure_meta_eda_before_tracks(config, allow_run=False)
        run_staged_perch_1d_1e(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        print(f"  Perch final: {PERCH_FINAL_DIR}")
        print(f"  Submission: {PROJECT_ROOT / 'submission'}")
        return

    if pipeline == "staged_tournament_resume":
        ensure_meta_eda_before_tracks(config)
        run_staged_tournament_resume_phase(config, suite)
        if _should_auto_finalize(config, pipeline):
            run_staged_finalize_winner(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        print(f"\n  Tournament results: {TOURNAMENT_RESULTS}")
        print(f"  Kaggle bundle: {PROJECT_ROOT / 'submission'}")
        return

    if pipeline == "staged_1c_only":
        ensure_meta_eda_before_tracks(config)
        run_stage_1c_aug_search(config, suite)
        if _stage_1d_cfg(config).get("enabled", False):
            run_stage_1d_final_train(config)
        if _stage_1e_cfg(config).get("enabled", False):
            if _skip_if_completed(config, ARCH_SEARCH_1E_RESULTS, "Stage 1e"):
                pass
            else:
                run_stage_1e_pseudo_refine(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        return

    if pipeline == "staged_finalize":
        ensure_meta_eda_before_tracks(config, allow_run=False)
        run_staged_finalize_winner(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        print("\n  Tournament finalize finished.")
        print(f"  Kaggle bundle: {PROJECT_ROOT / 'submission'}")
        return

    if pipeline == "staged_1e_only":
        print("\n  Pipeline: staged_1e_only — pseudo-refine only (requires 1d final head + train cache)")
        if not (PERCH_FINAL_DIR / "final_head.keras").exists() and not (
            PERCH_FINAL_DIR / "final_head.weights.h5"
        ).exists():
            print("  [1e] Missing stage-1d head in logs/meta_agent/perch/final/ — run stage 1d first.")
        else:
            run_stage_1e_pseudo_refine(config, suite)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        print(f"  Perch final: {PERCH_FINAL_DIR}")
        print(f"  Submission: {PROJECT_ROOT / 'submission'}")
        return

    staged_pipelines = (
        "staged_1a",
        "staged_1a_1b",
        "staged_1a_1b_1c",
        "staged_full",
        "staged_tournament",
    )
    if pipeline in staged_pipelines:
        run_staged_pipeline_sequential(config, suite, pipeline)
        print(f"\n  Total time: {(time.time() - t0) / 60:.1f} min")
        print("=" * 60)
        mode = "tournament" if _use_tournament_mode(config, pipeline) else "sequential"
        print(f"\n  Staged pipeline finished ({mode}).")
        print(f"  Tournament results: {TOURNAMENT_RESULTS}")
        print(f"  Kaggle bundle: {PROJECT_ROOT / 'submission'}")
        print(f"  Perch final: {PERCH_FINAL_DIR}  |  CNN final: {CNN_FINAL_DIR}")
        return

    if meta_cfg.get("run_eda", True):
        ensure_meta_eda_before_tracks(config)
    else:
        print("\n  [Phase 0] EDA skipped (meta_agent.run_eda=false)")

    cnn_score = run_phase1_cnn(config, cnn_iters, suite)
    birdnet_score = run_phase1_birdnet(config, birdnet_iters, suite)
    perch_score = run_phase2_perch(config, perch_iters, suite)
    ensemble = run_phase3_ensemble(
        config, ensemble_iters, suite, perch_score, birdnet_score
    )

    save_soundscape_leaderboard(
        {"cnn": cnn_score, "birdnet": birdnet_score, "perch": perch_score},
        ensemble,
        config,
    )

    def _line(sc: SoundscapeScore | None, name: str) -> None:
        if sc is None:
            print(f"  {name}:      (skipped)")
            return
        print(f"  {name}:      {format_soundscape_score(sc)}")

    print("\n" + "=" * 60)
    print("  META AGENT COMPLETE")
    print(
        f"  Model ranking uses macro_AP on labeled train_soundscapes; "
        f"also reports macro_AUC and median_AUC (v2 benchmark)"
    )
    _line(cnn_score, "CNN")
    _line(birdnet_score, "BirdNET")
    _line(perch_score, "Perch")
    if ensemble:
        print(
            f"  Ensemble: {format_soundscape_metrics_line(macro_ap=ensemble.get('best_ensemble_macro_ap'), macro_auc=ensemble.get('best_ensemble_macro_auc'), median_auc=ensemble.get('best_ensemble_median_auc'), ranking_metric=metric)}  "
            f"(perch={ensemble['perch_weight']:.2f} / birdnet={ensemble['birdnet_weight']:.2f})"
        )
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
