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


class ExperimentMemory:
    FILE = "experiment_memory.jsonl"

    def __init__(
        self,
        logs_dir: Path,
        ranking_metric: str = PRIMARY_META_METRIC,
    ) -> None:
        self.path = Path(logs_dir) / self.FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_metric = ranking_metric
        self._runs: list[dict] = []
        self._load()

    def _ranking_value(self, entry: dict) -> float:
        """Scalar used to sort runs (higher is better)."""
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
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "spec": spec,
            "success": success,
            "macro_roc_auc": auc,
            "macro_average_precision": ap,
            "median_per_class_auc": med,
            "ranking_metric": self.ranking_metric,
            "metrics": metrics or {},
            "reasoning": spec.get("reasoning", ""),
            "hypothesis": spec.get("hypothesis", ""),
            "code_snippet": code[:400] if code else "",
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._runs.append(entry)

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
        if r.get("metrics"):
            return format_metrics_dict(r["metrics"], ranking_metric=self.ranking_metric)
        return format_metrics_dict(
            {
                "status": "success",
                "macro_average_precision": r.get("macro_average_precision"),
                "macro_roc_auc": r.get("macro_roc_auc"),
                "median_per_class_auc": r.get("median_per_class_auc"),
            },
            ranking_metric=self.ranking_metric,
        )

    # ------------------------------------------------------------------ researcher context

    def researcher_context(self) -> str:
        """
        Full history formatted for the Researcher model.
        The Researcher reads this once per iteration to decide what to try next.
        The Coder never sees this.
        """
        total = len(self._runs)
        if total == 0:
            return "No experiments have been run yet. This is the very first iteration."

        ok = self.successful_runs()
        fails = self.failed_runs()
        best = self.best_runs(5)

        lines = [
            f"EXPERIMENT HISTORY: {total} runs | {len(ok)} succeeded | {len(fails)} failed",
            f"RANKING METRIC (optimize this): {self.ranking_metric} "
            f"(also log macro_AUC and median_AUC each run; AP tracks Kaggle LB best)",
            "",
        ]

        if best:
            lines.append("TOP RESULTS (sorted by ranking metric):")
            for i, r in enumerate(best, 1):
                spec = {k: v for k, v in r["spec"].items()
                        if k not in ("reasoning", "hypothesis")}
                lines.append(f"  #{i} {self._format_run_score(r)} | spec={spec}")
                if r.get("reasoning"):
                    lines.append(f"       reasoning: {r['reasoning']}")
            lines.append("")

        recent_fails = fails[-5:]
        if recent_fails:
            lines.append("RECENT FAILURES (do not repeat these):")
            for r in recent_fails:
                spec = {k: v for k, v in r["spec"].items()
                        if k not in ("reasoning", "hypothesis")}
                lines.append(f"  - {spec}")
            lines.append("")

        recent_ok = [r for r in ok[-8:] if self._ranking_value(r) >= 0]
        if len(recent_ok) >= 2:
            vals = [f"{self._ranking_value(r):.4f}" for r in recent_ok]
            trend = (
                "↑ improving"
                if self._ranking_value(recent_ok[-1]) > self._ranking_value(recent_ok[0])
                else "↓ declining"
            )
            lines.append(
                f"RECENT {self.ranking_metric} TREND: {' → '.join(vals)} ({trend})"
            )
            lines.append("")

        tried_params = []
        for r in self._runs[-15:]:
            s = {k: v for k, v in r["spec"].items()
                 if k not in ("reasoning", "hypothesis")}
            tried_params.append(str(s))
        lines.append(f"LAST {len(tried_params)} CONFIGS TRIED (avoid repeating):")
        for p in tried_params:
            lines.append(f"  {p}")

        return "\n".join(lines)
