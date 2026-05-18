"""
Uniform terminal output for meta-agent, CNN, and Perch runs (presentation only).

Does not change training, evaluation, or memory logic — formatting and safe reads only.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

# ── layout ────────────────────────────────────────────────────────────────────

W = 62
_BAR = "═" * W
_THIN = "─" * W


def _out(msg: str = "") -> None:
    print(msg, flush=True)


def banner(title: str, *, subtitle: str | None = None) -> None:
    _out()
    _out(f"  ╔{_BAR}╗")
    for line in _wrap(title, W - 4):
        _out(f"  ║  {line:<{W - 4}}║")
    if subtitle:
        _out(f"  ║  {' ' * (W - 4)}║")
        for line in _wrap(subtitle, W - 4):
            _out(f"  ║  {line:<{W - 4}}║")
    _out(f"  ╚{_BAR}╝")
    _out()


def _wrap(text: str, width: int) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return [""]
    return textwrap.wrap(text, width=width) or [""]


def section_start(title: str, *, detail: str | None = None) -> None:
    _out()
    _out(f"  ┌{'─' * (W - 2)}┐")
    _out(f"  │ {title:<{W - 4}} │")
    if detail:
        for line in _wrap(detail, W - 6):
            _out(f"  │   {line:<{W - 6}} │")
    _out(f"  └{'─' * (W - 2)}┘")


def section_summary(title: str, lines: list[str]) -> None:
    """End-of-section recap for graders."""
    _out()
    _out(f"  ▼ SECTION SUMMARY — {title}")
    _out(f"  {_THIN}")
    if not lines:
        _out("    · (no details recorded)")
    for line in lines:
        for part in _wrap(line, W - 6):
            _out(f"    · {part}")
    _out(f"  {_THIN}")
    _out()


def heartbeat(msg: str) -> None:
    _out(f"  ♥ {msg}")


def moving_on(context: str = "") -> None:
    suffix = f" ({context})" if context else ""
    _out(f"  → Moving on...{suffix}")


# ── roadmap ───────────────────────────────────────────────────────────────────

def print_pipeline_roadmap(config: dict, pipeline: str) -> None:
    meta = config.get("meta_agent") or {}
    order = meta.get("track_order") or ["cnn", "perch"]
    excluded = {str(t).lower() for t in (meta.get("excluded_tracks") or [])}
    tracks = [t for t in order if str(t).lower() not in excluded]
    metric = meta.get("primary_metric", "macro_average_precision")
    tournament = bool((meta.get("tournament") or {}).get("enabled")) or "tournament" in pipeline

    steps: list[str] = []
    if meta.get("run_eda", True) and pipeline not in ("staged_finalize", "bundle_kaggle_dataset"):
        steps.append("Phase 0 — EDA (dataset brief for researchers)")

    if tournament or "tournament" in pipeline:
        steps.append(
            f"Phase A — Tournament on sample data: {' → '.join(t.upper() for t in tracks)} "
            f"(1a arch search → 1b refine → 1c aug); metric: {metric}"
        )
        steps.append("Phase B — Finalize global winner (1d full train → 1e pseudo-label)")
    elif pipeline.startswith("staged"):
        max_s = _pipeline_max_stage_label(pipeline)
        steps.append(
            f"Staged pipeline ({max_s}): each track runs fully before the next — "
            f"{', '.join(t.upper() for t in tracks)}"
        )
    else:
        steps.append("Legacy pipeline: CNN → BirdNET → Perch → Ensemble")

    steps.append("Artifacts: logs/meta_agent/, submission/, tournament_results.json")

    banner("BirdCLEF Meta-Agent — Run Roadmap", subtitle=f"pipeline: {pipeline}")
    for i, step in enumerate(steps, 1):
        _out(f"  {i}. {step}")
    _out()


def _pipeline_max_stage_label(pipeline: str) -> str:
    p = pipeline.lower()
    if "1a_1b_1c" in p or p == "staged_full":
        return "through 1e"
    if "1a_1b" in p:
        return "through 1b"
    if "1a" in p:
        return "through 1a"
    if "1c" in p:
        return "1c (+ optional 1d/1e)"
    if "finalize" in p:
        return "finalize only"
    return p


# ── researcher / results ──────────────────────────────────────────────────────

_SPEC_KEYS = (
    "arch_type",
    "arch_description",
    "strategy",
    "hypothesis",
    "reasoning",
    "aug_preset",
    "aug_baseline",
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "hidden_dim",
    "proj_dim",
    "n_layers",
    "optimizer",
    "slot",
    "planner_note",
)


def print_researcher_proposals(
    specs: list[dict],
    *,
    track: str = "",
    round_label: str = "",
) -> None:
    if not specs:
        _out("  ◆ Researcher returned no experiments — Moving on...")
        return
    tag = f" [{track.upper()}]" if track else ""
    hdr = f"◆ RESEARCHER PLAN{tag}"
    if round_label:
        hdr += f" — {round_label}"
    _out()
    _out(f"  {hdr}")
    _out(f"  {_THIN}")
    for i, spec in enumerate(specs, 1):
        slot = spec.get("slot") or f"exp-{i}"
        arch = spec.get("arch_type", "—")
        strategy = spec.get("strategy", "—")
        _out(f"  ┌─ Experiment {i}/{len(specs)}  slot={slot}  arch={arch}  strategy={strategy}")
        for key in _SPEC_KEYS:
            val = spec.get(key)
            if val is None or val == "" or val == "—":
                continue
            text = str(val).strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "…"
            _out(f"  │  {key}: {text}")
        _out("  └" + "─" * (W - 4))
    _out()


def print_run_result(
    *,
    slot: str = "",
    metrics: dict | None = None,
    success: bool | None = None,
    stderr_tail: str = "",
    ranking_metric: str = "macro_average_precision",
) -> None:
    _out()
    label = f"  ◇ RUN RESULT"
    if slot:
        label += f"  [{slot}]"
    _out(label)
    if metrics and (metrics.get("status") == "success" or success is not False):
        line = _format_metrics_line(metrics, ranking_metric)
        _out(f"    ✓ {line}")
        ss = metrics.get("soundscape_macro_ap")
        if ss is not None:
            _out(f"    soundscape macro_AP: {float(ss):.5f}")
    elif success is False or (metrics and metrics.get("status") != "success"):
        _out("    ✗ run failed or incomplete")
        if stderr_tail:
            tail = stderr_tail.strip()[-280:]
            for ln in tail.splitlines()[-4:]:
                _out(f"      {ln}")
    else:
        _out("    — no metrics captured")
    _out()


def _format_metrics_line(metrics: dict, ranking_metric: str) -> str:
    try:
        from soundscape_evaluator import format_metrics_dict

        return format_metrics_dict(metrics, ranking_metric=ranking_metric)
    except Exception:
        ap = metrics.get("macro_average_precision")
        auc = metrics.get("macro_roc_auc")
        parts = []
        if ap is not None:
            parts.append(f"macro_AP={float(ap):.5f}")
        if auc is not None:
            parts.append(f"macro_AUC={float(auc):.5f}")
        return "  ".join(parts) if parts else str(metrics.get("status", "unknown"))


# ── architecture snapshot ─────────────────────────────────────────────────────

def print_final_architecture(
    *,
    track: str,
    spec: dict | None = None,
    memory_dir: str | Path | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    _out()
    banner(f"Final architecture — {track.upper()}")
    if spec:
        _out(f"  arch_type     : {spec.get('arch_type', '—')}")
        desc = (spec.get("arch_description") or spec.get("reasoning") or "")[:320]
        if desc:
            for line in _wrap(desc, W - 4):
                _out(f"  description   : {line}")
        for key in ("aug_preset", "aug_baseline", "learning_rate", "batch_size", "epochs", "dropout"):
            if spec.get(key) is not None:
                _out(f"  {key:<14}: {spec[key]}")
    if memory_dir:
        _out(f"  memory_dir    : {memory_dir}")
    for line in extra_lines or []:
        _out(f"  {line}")
    _out()


def architecture_from_memory_dir(
    mem_dir: Path,
    *,
    track: str,
) -> dict | None:
    """Best-effort read of champion spec; returns None if missing."""
    mem_dir = Path(mem_dir)
    candidates = [
        mem_dir / "best_model_info.json",
        mem_dir / "refine_champion_spec.json",
        mem_dir / "stage_1a_champion_spec.json",
    ]
    for p in candidates:
        data = safe_load_json(p)
        if not data:
            continue
        if "spec" in data and isinstance(data["spec"], dict):
            return data["spec"]
        if "arch_type" in data:
            return data
    if track.lower() == "perch" and (mem_dir / "best_head_code.py").exists():
        return {"arch_type": "(see best_head_code.py)", "note": str(mem_dir / "best_head_code.py")}
    if (mem_dir / "best_model_slot.py").exists():
        return {"arch_type": "(see best_model_slot.py)", "note": str(mem_dir / "best_model_slot.py")}
    return None


# ── safe I/O ──────────────────────────────────────────────────────────────────

def safe_load_json(path: Path | str) -> dict | None:
    try:
        p = Path(path)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_text(path: Path | str, *, max_chars: int = 500) -> str | None:
    try:
        p = Path(path)
        if not p.is_file():
            return None
        return p.read_text(encoding="utf-8")[:max_chars]
    except Exception:
        return None


# ── one-time subprocess / setup lines ─────────────────────────────────────────

class _OnceKeys:
    _seen: set[str] = set()

    @classmethod
    def mark(cls, key: str) -> bool:
        """Return True the first time this key is seen."""
        if key in cls._seen:
            return False
        cls._seen.add(key)
        return True

    @classmethod
    def reset(cls) -> None:
        cls._seen.clear()


def setup_once(key: str, message: str) -> None:
    if _OnceKeys.mark(key):
        _out(message)


# Patterns suppressed after first occurrence in streamed child output
_SUPPRESS_REPEAT_SUBSTRINGS = (
    "[Perch] Loading ONNX session",
    "[Perch] Loaded. Embedding index",
    "[Setup] Locating ONNX model",
    "[Setup] ONNX model:",
    "Locating ONNX model",
    "Loading ONNX session",
)


def filter_subprocess_line(line: str) -> bool:
    """Return False to hide duplicate setup noise from streamed logs."""
    for pat in _SUPPRESS_REPEAT_SUBSTRINGS:
        if pat in line:
            key = f"subproc:{pat}"
            if not _OnceKeys.mark(key):
                return False
    return True


def summarize_stage_json(
    title: str,
    result_path: Path | str,
    *,
    winner_keys: tuple[str, ...] = ("winner",),
) -> None:
    """Read a stage results JSON and print section summary; never raises."""
    path = Path(result_path)
    data = safe_load_json(path)
    if not data:
        section_summary(title, [f"Results file not found or unreadable — Moving on..."])
        return
    lines: list[str] = []
    if data.get("stage"):
        lines.append(f"Stage: {data['stage']}")
    if data.get("primary_metric"):
        lines.append(f"Metric: {data['primary_metric']}")
    winner = None
    for k in winner_keys:
        if data.get(k):
            winner = data[k]
            break
        tr = data.get("tracks") or {}
        if isinstance(tr, dict):
            for tname, tentry in tr.items():
                if isinstance(tentry, dict) and tentry.get("winner"):
                    winner = tentry["winner"]
                    lines.append(f"Track: {tname}")
                    break
    if winner and isinstance(winner, dict):
        lines.append(f"Winner aug: {winner.get('aug_baseline') or winner.get('aug_preset', '—')}")
        lines.append(f"Winner arch: {winner.get('arch_type', '—')}")
        sc = winner.get("score") or {}
        pv = sc.get("primary_value") or winner.get("ranking_value") or winner.get("primary_value")
        if pv is not None:
            lines.append(f"Score: {float(pv):.5f}")
        md = winner.get("memory_dir")
        if md:
            lines.append(f"memory: {md}")
    gw = data.get("global_winner")
    if gw and isinstance(gw, dict):
        lines.append(f"Global winner track: {gw.get('track', '—')}")
        lines.append(f"aug: {gw.get('aug_preset', '—')}")
        if gw.get("primary_value") is not None:
            lines.append(f"score: {float(gw['primary_value']):.5f}")
        if gw.get("arch_type"):
            lines.append(f"arch: {gw.get('arch_type')}")
    if not lines:
        lines.append("Completed (see JSON for details).")
    section_summary(title, lines)


def print_pipeline_epilogue(
    *,
    tournament_results: Path | str,
    finalize_results: Path | str | None = None,
) -> None:
    """Closing snapshot after a full staged run; never raises."""
    tpath = Path(tournament_results)
    data = safe_load_json(tpath)
    winner = None
    if data:
        winner = data.get("global_winner")
        if not winner:
            for tr in (data.get("tracks") or {}).values():
                if isinstance(tr, dict) and tr.get("winner"):
                    winner = tr["winner"]
                    break
    if finalize_results:
        fin = safe_load_json(finalize_results)
        if fin and fin.get("global_winner"):
            winner = fin["global_winner"]
    if not winner:
        moving_on("pipeline champion snapshot")
        return
    track = str(winner.get("track", "winner"))
    spec = architecture_from_memory_dir(Path(winner.get("memory_dir", "")), track=track)
    print_final_architecture(
        track=track,
        spec=spec,
        memory_dir=winner.get("memory_dir"),
        extra_lines=["Pipeline run complete — see logs/meta_agent/ and submission/"],
    )
