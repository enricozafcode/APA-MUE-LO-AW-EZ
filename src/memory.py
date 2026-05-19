"""
Persistent experiment memory — append-only JSONL log.

Every run is one line. The Researcher reads this to reason about what to try next.
The Coder never sees this — it only gets a clean spec from the Researcher.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from soundscape_evaluator import PRIMARY_META_METRIC, format_metrics_dict


def resolve_researcher_history_max_runs(config: dict | None) -> int | None:
    """None = include all runs from experiment_memory.jsonl in planner context."""
    if not config:
        return None
    ma = config.get("meta_agent") or {}
    v = ma.get("researcher_history_max_runs")
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("all", "none", ""):
        return None
    try:
        n = int(v)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


class ExperimentMemory:
    FILE = "experiment_memory.jsonl"

    def __init__(
        self,
        logs_dir: Path,
        ranking_metric: str = PRIMARY_META_METRIC,
        *,
        researcher_history_max_runs: int | None = None,
    ) -> None:
        self.path = Path(logs_dir) / self.FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_metric = ranking_metric
        self.researcher_history_max_runs = researcher_history_max_runs
        self._runs: list[dict] = []
        self._stage_ctx: dict | None = None
        self._load()
        self._resumed_run_count = len(self._runs)

    def set_stage(
        self,
        *,
        track: str,
        stage: str,
        label: str | None = None,
    ) -> None:
        """Tag subsequent ``log()`` calls for the experiment timeline JSON/plots."""
        self._stage_ctx = {
            "track": track,
            "stage": stage,
            "label": label or f"{track.upper()} Stage {stage}",
        }

    def _ranking_value(self, entry: dict) -> float:
        """Scalar used to sort runs (higher is better)."""
        ss = entry.get("soundscape_macro_ap")
        if ss is None:
            ss = (entry.get("metrics") or {}).get("soundscape_macro_ap")
        if ss is not None:
            return float(ss)
        if self.ranking_metric == "macro_roc_auc":
            v = entry.get("macro_roc_auc")
        else:
            v = entry.get("macro_average_precision")
            if v is None:
                v = entry.get("metrics", {}).get("macro_average_precision")
        return float(v) if v is not None else -1.0

    # ------------------------------------------------------------------ write

    def log(self, *, spec: dict, metrics: dict | None, code: str = "") -> None:
        success = bool(metrics and metrics.get("status") == "success")
        auc = float(metrics["macro_roc_auc"]) if success and metrics else None
        ap = (
            float(metrics["macro_average_precision"])
            if success and metrics.get("macro_average_precision") is not None
            else None
        )
        med = (
            float(metrics["median_per_class_auc"])
            if success and metrics.get("median_per_class_auc") is not None
            else None
        )
        train_loss = (
            float(metrics["train_loss"])
            if success and metrics.get("train_loss") is not None
            else None
        )
        val_loss = (
            float(metrics["val_loss"])
            if success and metrics.get("val_loss") is not None
            else None
        )
        soundscape_ap = (
            float(metrics["soundscape_macro_ap"])
            if success and metrics and metrics.get("soundscape_macro_ap") is not None
            else None
        )
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "spec": spec,
            "success": success,
            "macro_roc_auc": auc,
            "macro_average_precision": ap,
            "median_per_class_auc": med,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "soundscape_macro_ap": soundscape_ap,
            "ranking_metric": self.ranking_metric,
            "metrics": metrics or {},
            "reasoning": spec.get("reasoning", ""),
            "hypothesis": spec.get("hypothesis", ""),
            "code_snippet": code[:400] if code else "",
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._runs.append(entry)
        if self._stage_ctx:
            try:
                from experiment_tracker import record_experiment

                record_experiment(
                    entry,
                    stage_ctx=self._stage_ctx,
                    memory_dir=self.path.parent,
                    ranking_metric=self.ranking_metric,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ read

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def all_runs(self) -> list[dict]:
        return list(self._runs)

    def successful_runs(self) -> list[dict]:
        return [r for r in self._runs if r["success"]]

    def failed_runs(self) -> list[dict]:
        return [r for r in self._runs if not r["success"]]

    def best_runs(self, k: int = 5) -> list[dict]:
        ok = sorted(self.successful_runs(), key=self._ranking_value, reverse=True)
        return ok[:k]

    def total(self) -> int:
        return len(self._runs)

    def _format_run_score(self, r: dict) -> str:
        if r.get("soundscape_macro_ap") is not None:
            ss = float(r["soundscape_macro_ap"])
            subset = r.get("macro_average_precision")
            subset_s = f" subset_train_AP={float(subset):.5f}" if subset is not None else ""
            return f"soundscape_AP={ss:.5f} (ranking){subset_s}"
        if r.get("metrics"):
            return format_metrics_dict(r["metrics"], ranking_metric=self.ranking_metric)
        return format_metrics_dict(
            {
                "status": "success",
                "macro_average_precision": r.get("macro_average_precision"),
                "macro_roc_auc": r.get("macro_roc_auc"),
                "median_per_class_auc": r.get("median_per_class_auc"),
                "train_loss": r.get("train_loss"),
                "val_loss": r.get("val_loss"),
            },
            ranking_metric=self.ranking_metric,
        )

    # ------------------------------------------------------------------ researcher context

    _DESC_MAX_CHARS = 220
    _RATIONALE_MAX_CHARS = 900

    def announce_resumed_history(
        self,
        *,
        track: str = "",
        stage: str = "",
    ) -> None:
        """Log once when a stage reuses an existing experiment_memory.jsonl."""
        n = self._resumed_run_count
        if n <= 0:
            return
        tag = ""
        if track or stage:
            tag = f" [{track.upper()} {stage}]" if track and stage else f" [{track or stage}]"
        print(
            f"  [Memory]{tag} Loaded {n} prior run(s) from {self.path.name} "
            f"(planner will see full history up to cap)",
            flush=True,
        )

    @staticmethod
    def _hyperparams_short(spec: dict) -> str:
        parts: list[str] = []
        if spec.get("hidden_dim") is not None:
            parts.append(f"h{spec['hidden_dim']}")
        if spec.get("n_layers") is not None:
            parts.append(f"L{spec['n_layers']}")
        if spec.get("dropout") is not None:
            parts.append(f"drop{spec['dropout']}")
        if spec.get("learning_rate") is not None:
            parts.append(f"lr{spec['learning_rate']}")
        if spec.get("batch_size") is not None:
            parts.append(f"bs{spec['batch_size']}")
        if spec.get("optimizer"):
            parts.append(str(spec["optimizer"])[:5])
        if spec.get("preset_name"):
            parts.append(f"preset={str(spec['preset_name'])[:28]}")
        if spec.get("mix_prob") is not None:
            parts.append(f"mix_p={spec['mix_prob']}")
        if spec.get("aug_prob") is not None:
            parts.append(f"aug_p={spec['aug_prob']}")
        if spec.get("audio_anchor"):
            parts.append(f"anchor={spec['audio_anchor']}")
        return "|".join(parts) if parts else "default"

    def _run_kind_label(self, spec: dict) -> str:
        if spec.get("arch_type"):
            return str(spec["arch_type"])
        if spec.get("preset_name"):
            return f"aug:{spec.get('preset_name')}"
        return "?"

    def _scores_line(self, entry: dict) -> str:
        if not entry.get("success"):
            err = (entry.get("metrics") or {}).get("reason", "")
            return f"FAILED{f' ({err[:80]})' if err else ''}"
        return self._format_run_score(entry)

    def _run_researcher_line(self, entry: dict, run_index: int) -> list[str]:
        spec = entry.get("spec") or {}
        kind = self._run_kind_label(spec)
        desc = (spec.get("arch_description") or spec.get("preset_name") or "").strip()
        desc = str(desc).replace("\n", " ")
        if len(desc) > self._DESC_MAX_CHARS:
            desc = desc[: self._DESC_MAX_CHARS - 1] + "…"
        slot = spec.get("slot")
        slot_s = f" slot={slot}" if slot else ""
        hp = self._hyperparams_short(spec)
        strategy = spec.get("strategy")
        strat_s = f" {strategy}" if strategy else ""
        head = f"#{run_index} | {kind}{slot_s}{strat_s} | {self._scores_line(entry)}"
        lines = [head, f"  hyperparams: {hp}"]
        if desc:
            lines.append(f"  label: {desc}")
        reason = (entry.get("reasoning") or spec.get("reasoning") or "").strip()
        hyp = (entry.get("hypothesis") or spec.get("hypothesis") or "").strip()
        if reason:
            r = reason if len(reason) <= self._RATIONALE_MAX_CHARS else reason[: self._RATIONALE_MAX_CHARS - 1] + "…"
            lines.append(f"  reasoning: {r}")
        if hyp:
            h = hyp if len(hyp) <= self._RATIONALE_MAX_CHARS else hyp[: self._RATIONALE_MAX_CHARS - 1] + "…"
            lines.append(f"  hypothesis: {h}")
        return lines

    def researcher_context(self, *, max_runs: int | None = None) -> str:
        """
        Full history formatted for the Researcher model.
        The Researcher reads this once per iteration to decide what to try next.
        The Coder never sees this.
        """
        total = len(self._runs)
        if total == 0:
            return "No experiments have been run yet. This is the very first iteration."

        cap = max_runs if max_runs is not None else self.researcher_history_max_runs
        ok = self.successful_runs()
        fails = self.failed_runs()
        best = self.best_runs(3)

        lines = [
            "EXPERIMENT LOG (loaded from experiment_memory.jsonl on disk — do not repeat failed configs)",
            f"Runs: {total} ({len(ok)} ok, {len(fails)} failed) | "
            f"optimize: {self.ranking_metric} "
            f"(also: macro_AP, macro_AUC, median_AUC, train_loss, val_loss)",
            "",
        ]

        if best:
            b = best[0]
            lines.append(f"BEST SO FAR: {self._run_kind_label(b.get('spec') or {})} | {self._format_run_score(b)}")
            lines.append("")

        show = self._runs[-cap:] if cap and total > cap else self._runs
        start_idx = total - len(show) + 1
        cap_note = f" (capped to last {cap})" if cap and total > cap else ""
        lines.append(f"RUNS (chronological, {len(show)} shown{cap_note}):")
        for offset, entry in enumerate(show):
            lines.extend(self._run_researcher_line(entry, start_idx + offset))
        lines.append("")

        recent_fails = fails[-5:]
        if recent_fails:
            lines.append("RECENT FAILURES (avoid repeating):")
            for r in recent_fails:
                spec = r.get("spec") or {}
                lines.append(
                    f"  - {self._run_kind_label(spec)} | "
                    f"{(r.get('metrics') or {}).get('reason', 'failed')}"
                )
            lines.append("")

        return "\n".join(lines)
