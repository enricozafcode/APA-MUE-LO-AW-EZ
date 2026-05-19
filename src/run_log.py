"""
Uniform terminal output for meta-agent, CNN, and Perch runs (presentation only).

Does not change training, evaluation, or memory logic — formatting and safe reads only.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

_compact_terminal = True


def set_compact_terminal(enabled: bool) -> None:
    global _compact_terminal
    _compact_terminal = bool(enabled)


def compact_terminal() -> bool:
    return _compact_terminal


def configure_terminal_from_config(config: dict | None) -> None:
    meta = (config or {}).get("meta_agent") or {}
    logging_cfg = (config or {}).get("logging") or {}
    raw = meta.get("compact_terminal")
    if raw is None:
        raw = logging_cfg.get("compact_terminal", True)
    set_compact_terminal(bool(raw))

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
        steps.append("Legacy pipeline: CNN → Perch")

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
    "preset_name",
    "audio_anchor",
    "mix_prob",
    "aug_prob",
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

_RATIONALE_KEYS = frozenset({"planner_note", "reasoning", "hypothesis"})

_PLACEHOLDER_REASONING = frozenset({
    "safe default cnn.",
    "researcher fallback.",
    "fallback defaults — researcher output could not be parsed.",
    "batch fallback —",
    "parser fallback — varied cnn baseline (seed-based).",
})

_GENERIC_CNN_ARCH_SNIPPET = "Three Conv2D blocks (32→64→128"

# Hypers shown for refine / champion diffs (not full arch_description boilerplate).
CNN_SPEC_DELTA_KEYS = (
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "weight_decay",
    "depth",
    "filters_base",
    "filter_pattern",
    "pooling_type",
    "classifier_hidden_units",
    "batch_norm",
    "residuals",
    "optimizer",
    "n_mels",
    "n_frames",
    "aug_preset",
    "aug_prob",
    "aug_noise_std",
    "aug_time_mask",
    "aug_freq_mask",
)
_PLACEHOLDER_HYPOTHESIS = frozenset({
    "baseline mel-cnn should train.",
    "safe default augmentation.",
})


def is_placeholder_rationale(value: Any, *, field: str = "reasoning") -> bool:
    """True when value is empty or a known code placeholder (not real LLM text)."""
    text = str(value or "").strip()
    if not text or text == "—":
        return True
    low = text.lower()
    placeholders = (
        _PLACEHOLDER_HYPOTHESIS if field == "hypothesis" else _PLACEHOLDER_REASONING
    )
    if low in placeholders:
        return True
    return any(low.startswith(p) for p in placeholders if p.endswith("—"))


def _resolve_planner_note(specs: list[dict], planner_note: str = "") -> str:
    note = (planner_note or "").strip()
    if note:
        return note
    for spec in specs:
        note = str(spec.get("_planner_note") or spec.get("planner_note") or "").strip()
        if note:
            return note
    return ""


def is_generic_cnn_arch_description(desc: Any) -> bool:
    text = str(desc or "").strip()
    if not text:
        return True
    return text.startswith(_GENERIC_CNN_ARCH_SNIPPET) or "GlobalAveragePooling, Dropout 0.3" in text


def _format_hyper_value(val: Any) -> str:
    if isinstance(val, float):
        if (val != 0 and abs(val) < 0.01) or abs(val) >= 1000:
            return f"{val:.4g}"
        return f"{val:g}"
    return str(val)


def _values_differ(a: Any, b: Any) -> bool:
    if a == b:
        return False
    try:
        return abs(float(a) - float(b)) > 1e-12
    except (TypeError, ValueError):
        return str(a) != str(b)


def format_cnn_spec_delta_lines(spec: dict, seed_spec: dict | None) -> list[str]:
    """Human-readable lines for what this experiment changes vs champion seed."""
    if not seed_spec:
        return []
    lines: list[str] = []
    deltas: list[str] = []
    for key in CNN_SPEC_DELTA_KEYS:
        if key not in spec and key not in seed_spec:
            continue
        new_v = spec.get(key)
        old_v = seed_spec.get(key)
        if new_v is None:
            continue
        if key not in seed_spec or _values_differ(new_v, old_v):
            if key in seed_spec:
                deltas.append(f"{key}={_format_hyper_value(new_v)} (champion {_format_hyper_value(old_v)})")
            else:
                deltas.append(f"{key}={_format_hyper_value(new_v)}")
    if deltas:
        lines.append("proposed: " + "; ".join(deltas))
    else:
        lines.append("proposed: no hyper change vs champion (coder may still edit code)")
    desc = str(spec.get("arch_description") or "").strip()
    if (
        desc
        and not is_generic_cnn_arch_description(desc)
        and "structure unchanged vs champion" not in desc.lower()
    ):
        short = desc if len(desc) <= 220 else desc[:217] + "…"
        lines.append("layout: " + short)
    return lines


def apply_planner_rationale_fallback(
    specs: list[dict],
    planner_note: str = "",
) -> None:
    """If per-slot reasoning/hypothesis missing, copy batch planner_note into reasoning."""
    note = _resolve_planner_note(specs, planner_note)
    if not note:
        return
    for spec in specs:
        if is_placeholder_rationale(spec.get("reasoning"), field="reasoning"):
            spec["reasoning"] = note
        if is_placeholder_rationale(spec.get("hypothesis"), field="hypothesis"):
            spec.setdefault("hypothesis", "")


def _print_wrapped_field(prefix: str, label: str, text: str, *, width: int = W - 6) -> None:
    body = str(text).strip().replace("\n", " ")
    if not body:
        return
    _out(f"  {prefix}{label}:")
    for line in _wrap(body, width):
        _out(f"  {prefix}  {line}")


def print_researcher_proposals(
    specs: list[dict],
    *,
    track: str = "",
    round_label: str = "",
    planner_note: str = "",
    seed_spec: dict | None = None,
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

    apply_planner_rationale_fallback(specs, planner_note)

    note = _resolve_planner_note(specs, planner_note)
    if note:
        _out("  ◆ LLM planner note (batch rationale)")
        _print_wrapped_field("│", "planner_note", note)
        _out()

    for i, spec in enumerate(specs, 1):
        slot = spec.get("slot") or f"exp-{i}"
        arch = spec.get("arch_type") or spec.get("preset_name") or "—"
        strategy = spec.get("strategy", "—")
        _out(f"  ┌─ Experiment {i}/{len(specs)}  slot={slot}  kind={arch}  strategy={strategy}")

        for key in ("reasoning", "hypothesis"):
            val = spec.get(key)
            if is_placeholder_rationale(val, field=key):
                _out(f"  │  {key}: —")
            else:
                _print_wrapped_field("│", key, str(val))

        delta_lines = format_cnn_spec_delta_lines(spec, seed_spec) if seed_spec else []
        skip_keys = (
            set(CNN_SPEC_DELTA_KEYS) | {"arch_type", "strategy", "slot", "arch_description"}
            if seed_spec
            else set()
        )
        if delta_lines:
            for line in delta_lines:
                for part in _wrap(line, W - 6):
                    _out(f"  │  {part}")
        elif seed_spec:
            _out("  │  proposed: —")

        for key in _SPEC_KEYS:
            if key in _RATIONALE_KEYS or key in skip_keys:
                continue
            if key == "arch_description" and (
                is_generic_cnn_arch_description(spec.get(key)) or seed_spec
            ):
                continue
            val = spec.get(key)
            if val is None or val == "" or val == "—":
                continue
            text = str(val).strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "…"
            _out(f"  │  {key}: {text}")
        _out("  └" + "─" * (W - 4))
    _out()


def print_run_loading(label: str, *, detail: str = "training + soundscape eval") -> None:
    _out(f"  → {label}  ({detail}…)")


def improvement_tag(
    value: float | None,
    best_so_far: float | None,
    *,
    higher_is_better: bool = True,
) -> str:
    if value is None:
        return ""
    if best_so_far is None:
        return "✓"
    eps = 1e-9
    if higher_is_better:
        if value > best_so_far + eps:
            return "↑ improving"
        if value < best_so_far - eps:
            return "↓ not improving"
        return "→ tied"
    if value < best_so_far - eps:
        return "↑ improving"
    if value > best_so_far + eps:
        return "↓ not improving"
    return "→ tied"


def print_run_outcome(
    *,
    slot: str = "",
    metrics: dict | None = None,
    ranking_value: float | None = None,
    best_so_far: float | None = None,
    ranking_metric: str = "macro_average_precision",
    failed: bool = False,
    stderr_tail: str = "",
) -> None:
    label = slot or "run"
    if failed or (metrics and metrics.get("status") != "success"):
        _out(f"  ◇ {label}: FAILED")
        if stderr_tail:
            for ln in stderr_tail.strip()[-280:].splitlines()[-3:]:
                _out(f"      {ln}")
        return
    rv = ranking_value
    if rv is None and metrics:
        rv = metrics.get(ranking_metric) or metrics.get("soundscape_macro_ap")
        if rv is None:
            rv = metrics.get("macro_average_precision")
    tag = improvement_tag(rv, best_so_far)
    line = _format_metrics_line(metrics, ranking_metric) if metrics else (
        f"{ranking_metric}={float(rv):.5f}" if rv is not None else "no metrics"
    )
    suffix = f"  {tag}" if tag else ""
    _out(f"  ◇ {label}:{suffix}  {line}")


def print_run_result(
    *,
    slot: str = "",
    metrics: dict | None = None,
    success: bool | None = None,
    stderr_tail: str = "",
    ranking_metric: str = "macro_average_precision",
    ranking_value: float | None = None,
    best_so_far: float | None = None,
) -> None:
    if compact_terminal():
        _out()
        print_run_outcome(
            slot=slot,
            metrics=metrics,
            ranking_value=ranking_value,
            best_so_far=best_so_far,
            ranking_metric=ranking_metric,
            failed=success is False or bool(metrics and metrics.get("status") != "success"),
            stderr_tail=stderr_tail,
        )
        return
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
        seed = spec.get("_seed_spec") if isinstance(spec.get("_seed_spec"), dict) else None
        delta_lines = format_cnn_spec_delta_lines(spec, seed) if seed else []
        if delta_lines:
            for dl in delta_lines:
                for line in _wrap(dl, W - 4):
                    _out(f"  changes       : {line}")
        desc = (spec.get("arch_description") or spec.get("reasoning") or "")
        if desc and not is_generic_cnn_arch_description(desc):
            for line in _wrap(str(desc)[:320], W - 4):
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


_COMPACT_DROP_SUBSTRINGS = (
    "PHASE3_DEBUG:",
    "CONFIG_ACTIVE:",
    "AUG_CFG:",
    "split_stats",
    "AUG_APPLIED:",
    "FOCAL_CLIP_MANIFEST:",
    "Manifest clips=",
    "FOCAL_TRAIN_CACHE:",
    "selected samples",
    "Focal cache clips=",
    "DATA: X_train=",
    "TRAIN_START:",
    "checkpoint_enabled",
    "has_validation",
    "TRAIN_BATCH:",
    "BEST_EPOCH_BY_VAL_LOSS:",
    "TRAIN_LOSS:",
    "EVAL_ARTIFACTS",
    "SOUNDSCAPE_EVAL_START",
    "SOUNDSCAPE_EVAL_CACHE:",
    "SOUNDSCAPE_EVAL_PROGRESS:",
    "SOUNDSCAPE_EVAL_READY:",
    "TRAIN_DONE",
    "SUBMISSION:",
    "Model: \"functional\"",
    "Total params:",
    "Trainable params:",
    "Non-trainable params:",
    "Layer (type)",
    "Candidate files=",
    "MODEL_READY",
    "[CNN Run]",
    "[CNN] Experiment subprocess timeout",
    "UserWarning:",
    "warnings.warn(",
    "site-packages/",
    ".venv/lib",
)

_COMPACT_DROP_RES = (
    re.compile(r"Loaded\s+\d+/\d+\s"),
    re.compile(r"\d+/\d+.*ms/step"),
    re.compile(r"Epoch\s+\d+/\d+"),
)

_KERAS_TABLE_CHARS = frozenset("┏┡╇┃├└┳━─│╶═")


def _is_keras_summary_line(line: str) -> bool:
    if any(c in line for c in _KERAS_TABLE_CHARS):
        return True
    s = line.strip()
    return bool(s.startswith("│ ") and ("Conv2D" in s or "Dense" in s or "InputLayer" in s))


def _compact_should_drop_line(line: str) -> bool:
    if _is_keras_summary_line(line):
        return True
    for pat in _COMPACT_DROP_SUBSTRINGS:
        if pat in line:
            return True
    for rx in _COMPACT_DROP_RES:
        if rx.search(line):
            return True
    return False


def filter_subprocess_line(line: str) -> bool:
    """Return False to hide duplicate setup noise from streamed logs."""
    if compact_terminal() and _compact_should_drop_line(line):
        return False
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
