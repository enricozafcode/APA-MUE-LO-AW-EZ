from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvaluationSummary:
    metrics: Dict[str, Any]
    analysis_prompt: str


class Evaluator:
    """Parses training script output into structured metrics for the agent loop."""

    def build_summary(self, stdout: str, stderr: str) -> EvaluationSummary:
        metrics = self._parse_metrics(stdout)

        if not metrics:
            metrics = {
                "status": "no_metrics_found",
                "error": stderr.splitlines()[-1][:200] if stderr else "no output",
            }

        analysis_prompt = (
            f"The last experiment produced these results:\n{json.dumps(metrics, indent=2)}\n\n"
            "Based on these results, propose the next experiment. "
            "If the run failed, diagnose the error and suggest a fix. "
            "If it succeeded, propose an improvement."
        )
        return EvaluationSummary(metrics=metrics, analysis_prompt=analysis_prompt)

    def _parse_metrics(self, stdout: str) -> Dict[str, Any]:
        """Look for a line like: METRICS: {"roc_auc": 0.72, "val_loss": 0.43}"""
        for line in stdout.splitlines():
            if line.startswith("METRICS:"):
                try:
                    return json.loads(line[len("METRICS:"):].strip())
                except json.JSONDecodeError:
                    pass
        return {}
