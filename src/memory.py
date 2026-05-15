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


class ExperimentMemory:
    FILE = "experiment_memory.jsonl"

    def __init__(self, logs_dir: Path) -> None:
        self.path = Path(logs_dir) / self.FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ write

    def log(self, *, spec: dict, metrics: dict | None, code: str = "") -> None:
        success = bool(metrics and metrics.get("status") == "success")
        auc = float(metrics["macro_roc_auc"]) if success and metrics else None
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "spec": spec,
            "success": success,
            "macro_roc_auc": auc,
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
        ok = sorted(self.successful_runs(), key=lambda r: r["macro_roc_auc"] or 0, reverse=True)
        return ok[:k]

    def total(self) -> int:
        return len(self._runs)

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
            "",
        ]

        # Best results
        if best:
            lines.append("TOP RESULTS:")
            for i, r in enumerate(best, 1):
                auc = r["macro_roc_auc"]
                spec = {k: v for k, v in r["spec"].items()
                        if k not in ("reasoning", "hypothesis")}
                lines.append(f"  #{i} AUC={auc:.5f} | spec={spec}")
                if r.get("reasoning"):
                    lines.append(f"       reasoning: {r['reasoning']}")
            lines.append("")

        # Recent failures
        recent_fails = fails[-5:]
        if recent_fails:
            lines.append("RECENT FAILURES (do not repeat these):")
            for r in recent_fails:
                spec = {k: v for k, v in r["spec"].items()
                        if k not in ("reasoning", "hypothesis")}
                lines.append(f"  - {spec}")
            lines.append("")

        # AUC trend
        recent_ok = [r for r in ok[-8:] if r["macro_roc_auc"] is not None]
        if len(recent_ok) >= 2:
            aucs = [f"{r['macro_roc_auc']:.4f}" for r in recent_ok]
            trend = "↑ improving" if recent_ok[-1]["macro_roc_auc"] > recent_ok[0]["macro_roc_auc"] else "↓ declining"
            lines.append(f"RECENT AUC TREND: {' → '.join(aucs)} ({trend})")
            lines.append("")

        # What was already tried (param combos to avoid)
        tried_params = []
        for r in self._runs[-15:]:
            s = {k: v for k, v in r["spec"].items()
                 if k not in ("reasoning", "hypothesis")}
            tried_params.append(str(s))
        lines.append(f"LAST {len(tried_params)} CONFIGS TRIED (avoid repeating):")
        for p in tried_params:
            lines.append(f"  {p}")

        return "\n".join(lines)
